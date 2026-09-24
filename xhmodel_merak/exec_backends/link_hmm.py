from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from xhmodel_merak.exec_backends.path_utils import TCIM_SUFFIXES

_ONNX_SUFFIX = ".onnx"
_DEFAULT_SOURCE_META = "golden_meta_info.json"
_DEFAULT_OUTPUT_META = "hmm_meta_info.json"
_ROLE_ALIASES = {
    "encode": "encode",
    "encoder": "encode",
    "prefill": "prefill",
    "decode": "decode",
}
_VISUAL_ROLE = re.compile(r"^visual_m\d+$", re.IGNORECASE)
_ROLE_IN_TEXT = re.compile(
    r"(?:^|[/_-])(visual_m\d+|encoder|encode|prefill|decode)(?:$|[/_.-])",
    re.IGNORECASE,
)


def _normalize_role(token: str) -> str | None:
    lowered = token.lower()
    if lowered in _ROLE_ALIASES:
        return _ROLE_ALIASES[lowered]
    if _VISUAL_ROLE.fullmatch(lowered):
        return lowered
    return None


def _role_from_text(text: str) -> str | None:
    matches = [_normalize_role(match.group(1)) for match in _ROLE_IN_TEXT.finditer(text)]
    roles = [role for role in matches if role is not None]
    return roles[-1] if roles else None


def _role_from_path(path: Path, root: Path) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    for part in reversed(relative.parts):
        stem = part
        lowered = stem.lower()
        for suffix in (*TCIM_SUFFIXES, _ONNX_SUFFIX):
            if lowered.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        role = _normalize_role(stem) or _role_from_text(stem)
        if role is not None:
            return role
    return _role_from_text(str(relative))


def _collect_hmm_by_role(hmm_dir: Path) -> dict[str, Path]:
    files = [path for path in hmm_dir.rglob("*") if path.is_file() and path.suffix.lower() in TCIM_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"no .hmm/.hmms files found under {hmm_dir}")
    by_role: dict[str, Path] = {}
    for path in files:
        role = _role_from_path(path, hmm_dir)
        if role is None:
            continue
        if role in by_role and by_role[role].resolve() != path.resolve():
            raise ValueError(f"multiple HMM files for role {role}: {by_role[role]} and {path}")
        by_role[role] = path
    if not by_role:
        raise FileNotFoundError(f"no encode/prefill/decode/visual HMM found under {hmm_dir}")
    return by_role


def _load_hmm_model_info(path: Path) -> dict[str, Any]:
    try:
        import tcim_lite as tcim
    except ImportError as exc:
        raise ImportError("tcim_lite is required to generate hmm_meta_info.json") from exc
    value = json.loads(tcim.runtime.load_model_info(str(path)))
    if not isinstance(value, dict):
        raise ValueError(f"TCIM model info must be an object: {path}")
    return value


def _cache_kind(name: str) -> str | None:
    lowered = name.lower()
    if "kcache" in lowered:
        return "key"
    if "vcache" in lowered:
        return "value"
    return None


def _normalize_tcim_dtype(dtype: Any) -> str:
    normalized = str(dtype).strip().upper()
    aliases = {
        "BOOL": "bool",
        "FLOAT16": "float16",
        "BFLOAT16": "bfloat16",
        "FLOAT32": "float32",
        "FLOAT64": "float64",
        "INT8": "int8",
        "UINT8": "uint8",
        "INT16": "int16",
        "UINT16": "uint16",
        "INT32": "int32",
        "UINT32": "uint32",
        "INT64": "int64",
        "UINT64": "uint64",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported TCIM cache dtype: {dtype!r}") from exc


def _one_cache_spec(
    role: str,
    kind: str,
    specs: set[tuple[tuple[int, ...], str]],
) -> tuple[tuple[int, ...], str] | None:
    if not specs:
        return None
    if len(specs) != 1:
        raise ValueError(f"HMM role {role} has multiple {kind} cache ABIs: {sorted(specs)}")
    return next(iter(specs))


def _hmm_cache_contract(
    path: Path,
    role: str,
    cache_axis: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int, str] | None:
    info = _load_hmm_model_info(path)
    specs: dict[str, set[tuple[tuple[int, ...], str]]] = {"key": set(), "value": set()}
    for item in info.get("input", []):
        if not isinstance(item, dict):
            continue
        kind = _cache_kind(str(item.get("name", "")))
        if kind is None:
            continue
        shape = tuple(int(dim) for dim in item.get("shape", []))
        if len(shape) <= cache_axis:
            raise ValueError(f"HMM cache input {item.get('name')!r} has invalid shape {shape}: {path}")
        specs[kind].add((shape, _normalize_tcim_dtype(item.get("dtype"))))
    key_spec = _one_cache_spec(role, "key", specs["key"])
    value_spec = _one_cache_spec(role, "value", specs["value"])
    if key_spec is None and value_spec is None:
        return None
    if key_spec is None or value_spec is None:
        raise ValueError(f"HMM role {role} must expose both key and value cache inputs: {path}")
    key_shape, key_dtype = key_spec
    value_shape, value_dtype = value_spec
    key_context = int(key_shape[cache_axis])
    value_context = int(value_shape[cache_axis])
    if key_context != value_context:
        raise ValueError(
            f"HMM role {role} has different key/value context lengths: {key_context} vs {value_context}"
        )
    if key_dtype != value_dtype:
        raise ValueError(f"HMM role {role} has different key/value cache dtypes: {key_dtype} vs {value_dtype}")
    return key_shape, value_shape, key_context, key_dtype


def _shared_cache_contract(
    hmm_by_role: dict[str, Path], cache_axis: int
) -> tuple[tuple[int, ...], tuple[int, ...], int, str] | None:
    contracts = {
        role: contract
        for role in ("prefill", "decode")
        if role in hmm_by_role
        if (contract := _hmm_cache_contract(hmm_by_role[role], role, cache_axis)) is not None
    }
    if not contracts:
        return None
    unique = set(contracts.values())
    if len(unique) != 1:
        rendered = {
            role: {"key_shape": list(item[0]), "value_shape": list(item[1]), "context": item[2], "dtype": item[3]}
            for role, item in contracts.items()
        }
        raise ValueError(f"prefill/decode HMM cache ABI mismatch: {rendered}")
    return next(iter(unique))


def _metadata_cache_axis(meta: dict[str, Any], override: int | None) -> int:
    if override is not None:
        return int(override)
    kv_cache = meta.get("kv_cache")
    if isinstance(kv_cache, dict) and "cache_axis" in kv_cache:
        return int(kv_cache["cache_axis"])
    return 2


def _metadata_cache_shape(existing: Any, key_shape: tuple[int, ...], value_shape: tuple[int, ...]):
    key = list(key_shape)
    value = list(value_shape)
    if key != value:
        return [key, value]
    if isinstance(existing, list) and existing and isinstance(existing[0], (list, tuple)):
        return [key, value]
    return key


def _patch_runtime_contract(
    value: Any,
    contract: tuple[tuple[int, ...], tuple[int, ...], int, str] | None,
) -> Any:
    if contract is None:
        return value
    key_shape, value_shape, context_tokens, cache_dtype = contract

    def visit(item: Any, parent_key: str | None = None) -> None:
        if isinstance(item, dict):
            for key, child in list(item.items()):
                if key == "kv_cache_shape":
                    item[key] = _metadata_cache_shape(child, key_shape, value_shape)
                elif key == "shape" and parent_key == "kv_cache":
                    item[key] = _metadata_cache_shape(child, key_shape, value_shape)
                elif key == "cache_dtype" and parent_key == "kv_cache":
                    item[key] = cache_dtype
                elif key in {"context_max_length", "max_context_tokens"}:
                    item[key] = context_tokens
                else:
                    visit(child, key)
        elif isinstance(item, list):
            for child in item:
                visit(child, parent_key)

    visit(value)
    return value


def _relative_path(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rewrite_model_paths(value: Any, meta_root: Path, hmm_by_role: dict[str, Path]) -> None:
    if isinstance(value, dict):
        for key, child in list(value.items()):
            if isinstance(child, str):
                role = _role_from_text(key) or _role_from_text(child)
                lowered = child.lower()
                if role in hmm_by_role and lowered.endswith((_ONNX_SUFFIX, *TCIM_SUFFIXES)):
                    hmm_path = hmm_by_role[role]
                    value[key] = _relative_path(hmm_path, meta_root)
                    checksum_key = f"{key}_md5"
                    if checksum_key in value:
                        value[checksum_key] = _file_md5(hmm_path)
            else:
                _rewrite_model_paths(child, meta_root, hmm_by_role)
    elif isinstance(value, list):
        for child in value:
            _rewrite_model_paths(child, meta_root, hmm_by_role)


def generate_hmm_meta(
    hmm_dir: str | Path,
    hmonnx_dir: str | Path,
    *,
    source_meta: str | Path | None = None,
    output_meta: str | Path | None = None,
    cache_axis: int | None = None,
    force: bool = False,
) -> Path:
    """Create an HMM-specific runtime metadata file without modifying the source metadata."""
    hmm_root = Path(hmm_dir).expanduser().resolve()
    hmonnx_root = Path(hmonnx_dir).expanduser().resolve()
    if not hmm_root.is_dir():
        raise NotADirectoryError(hmm_root)
    if not hmonnx_root.is_dir():
        raise NotADirectoryError(hmonnx_root)
    source = Path(source_meta).expanduser() if source_meta else hmonnx_root / _DEFAULT_SOURCE_META
    if not source.is_absolute():
        source = hmonnx_root / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = Path(output_meta).expanduser() if output_meta else source.parent / _DEFAULT_OUTPUT_META
    if not destination.is_absolute():
        destination = source.parent / destination
    if destination.exists() and not force:
        raise FileExistsError(f"refusing to replace {destination}; pass --force")
    meta = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"metadata must contain an object: {source}")
    hmm_by_role = _collect_hmm_by_role(hmm_root)
    _rewrite_model_paths(meta, destination.parent, hmm_by_role)

    axis = _metadata_cache_axis(meta, cache_axis)
    contract = _shared_cache_contract(hmm_by_role, axis)
    _patch_runtime_contract(meta, contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hmm_meta_info.json for compiled TCIM models.")
    parser.add_argument(
        "--hmm-path", "--hmm_path", dest="hmm_path", required=True, help="Directory containing .hmm/.hmms files."
    )
    parser.add_argument(
        "--hmonnx-path",
        "--hmonnx_path",
        dest="hmonnx_path",
        required=True,
        help="HMONNX export directory containing golden_meta_info.json.",
    )
    parser.add_argument("--source-meta", help="Source metadata; defaults to golden_meta_info.json.")
    parser.add_argument("--output-meta", help="Output metadata; defaults to hmm_meta_info.json.")
    parser.add_argument("--cache-axis", type=int, help="Override the KV-cache context axis (default: metadata or 2).")
    parser.add_argument("--force", action="store_true", help="Replace an existing output metadata file.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output = generate_hmm_meta(
        args.hmm_path,
        args.hmonnx_path,
        source_meta=args.source_meta,
        output_meta=args.output_meta,
        cache_axis=args.cache_axis,
        force=args.force,
    )
    print(output)


if __name__ == "__main__":
    main()
