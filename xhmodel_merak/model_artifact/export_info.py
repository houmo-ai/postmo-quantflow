"""Integrator-facing inventory, separate from model-specific runtime metadata."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError

from .files import confined_file
from . import MANIFEST_NAME


EXPORT_INFO_NAME = "export_meta_info.json"
RUNTIME_INFO_NAME = "runtime_meta_info.json"
EXPORT_TYPES = frozenset({"prefill", "decode", "hmonnx", "onnx", "tokenizer", "embedding", "json"})


def validate_export_info(root: Path, info: Any, *, inventoried: set[str] | None = None) -> None:
    """Validate the public schema and every path before publication or loading."""
    root = root.resolve()

    def invalid(message: str) -> None:
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, f"Invalid {EXPORT_INFO_NAME}: {message}")

    def file(value: str, name: str) -> None:
        confined_file(root, value, name)
        if inventoried is not None and value not in inventoried:
            invalid(f"{name} references a file outside the Manifest inventory: {value}")

    def entries(value: Any, *, allow_lora: bool) -> None:
        if not isinstance(value, Mapping) or not value:
            invalid("expected a non-empty component mapping")
        for name, entry in value.items():
            if not isinstance(name, str) or re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", name) is None:
                invalid(f"invalid component/checkpoint name {name!r}")
            if name == "lora" and allow_lora:
                if not isinstance(entry, Mapping) or not entry:
                    invalid("lora must contain checkpoint mappings")
                for checkpoint, children in entry.items():
                    if not isinstance(checkpoint, str) or not checkpoint or any(c in checkpoint for c in "/\\"):
                        invalid("invalid LoRA checkpoint name")
                    entries(children, allow_lora=False)
                continue
            if not isinstance(entry, Mapping) or set(entry) != {"path", "type"}:
                invalid(f"{name} must contain exactly path and type")
            kind, path = entry["type"], entry["path"]
            if not isinstance(kind, str) or kind not in EXPORT_TYPES:
                invalid(f"unsupported type for {name}: {kind!r}")
            if (
                not isinstance(path, str)
                or not path
                or "\\" in path
                or PureWindowsPath(path).drive
                or path.startswith("/")
                or any(part in {"", ".", ".."} for part in path.split("/"))
            ):
                invalid(f"{name}.path must be a portable package-relative path")
            if kind == "tokenizer":
                directory = root / path
                if not directory.is_dir() or not directory.resolve().is_relative_to(root):
                    invalid(f"missing tokenizer directory: {path}")
                files = list(directory.rglob("*"))
                if not any(item.is_file() for item in files):
                    invalid(f"empty tokenizer directory: {path}")
                for item in files:
                    if item.is_file() or item.is_symlink():
                        relative = item.relative_to(root).as_posix()
                        # Runtime caches may be added to a tokenizer directory.
                        # Only the delivered tokenizer files belong to its entry.
                        confined_file(root, relative, name)
                if inventoried is not None and not any(value.startswith(path + "/") for value in inventoried):
                    invalid(f"tokenizer directory is absent from Manifest: {path}")
            else:
                file(path, name)
                if kind in {"prefill", "decode", "hmonnx", "onnx"} and Path(path).suffix not in {".onnx", ".hmonnx"}:
                    invalid(f"{name} must identify a graph file")
                if kind == "json":
                    if Path(path).suffix.lower() != ".json":
                        invalid(f"{name} must identify a JSON file")
                    try:
                        json.loads((root / path).read_text(encoding="utf-8"))
                    except (ValueError, OSError) as error:
                        invalid(f"{name} is not readable JSON: {error}")

    entries(info, allow_lora=True)


def publish_export_info(
    root: Path,
    artifact_metadata: dict[str, Any],
    *,
    embedding_metadata: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Reserve the public filename while retaining the validated runtime contract."""
    info = artifact_metadata.pop("export_entries")
    validate_export_info(root, info)
    _describe_embedding_bins(info, {entry["pt"]: entry["bin"] for entry in embedding_metadata})
    validate_export_info(root, info)
    destination = root / EXPORT_INFO_NAME
    if artifact_metadata["metadata_file"] == EXPORT_INFO_NAME:
        runtime = root / RUNTIME_INFO_NAME
        if runtime.exists():
            raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, f"Runtime metadata collision: {runtime}")
        destination.rename(runtime)
        artifact_metadata["metadata_file"] = RUNTIME_INFO_NAME
    elif destination.exists():
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, f"Export inventory collision: {destination}")
    _describe_json_files(root, info, artifact_metadata.pop("export_json_roots", {}))
    validate_export_info(root, info)
    destination.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifact_metadata["export_meta_info"] = EXPORT_INFO_NAME


def _describe_embedding_bins(info: dict[str, Any], bins: Mapping[str, str]) -> None:
    """Use the embedding writer's result, retaining each adapter's resource scope."""
    for name, entry in list(info.items()):
        if name == "lora":
            for checkpoint in entry.values():
                _describe_embedding_bins(checkpoint, bins)
        elif entry["type"] == "embedding" and entry["path"] in bins:
            bin_name = f"{name}_bin"
            bin_entry = {"path": bins[entry["path"]], "type": "embedding"}
            if bin_name in info and info[bin_name] != bin_entry:
                raise ModelError(
                    ErrorCode.ARTIFACT_VALIDATION_FAILED, "Embedding export entry name collision", name=bin_name
                )
            info[bin_name] = bin_entry


def _describe_json_files(root: Path, info: dict[str, Any], lora_roots: Mapping[str, str]) -> None:
    """Index delivered JSON after runtime metadata has received its final name.

    The build Manifest is the enclosing inventory and is written later. It and
    this index itself are excluded; nested model metadata with the same filename
    is an ordinary JSON resource and must be included.
    """
    scopes = []
    for checkpoint, value in lora_roots.items():
        relative = Path(value)
        if (
            not value
            or relative.is_absolute()
            or PureWindowsPath(value).drive
            or "\\" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Invalid LoRA JSON resource root", path=value)
        if checkpoint not in info.get("lora", {}):
            raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "LoRA JSON root has no checkpoint entry")
        if any(
            relative == other or relative.is_relative_to(other) or other.is_relative_to(relative) for other, _ in scopes
        ):
            raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Overlapping LoRA JSON resource roots")
        scopes.append((relative, info["lora"][checkpoint]))
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.suffix.lower() != ".json" or relative.as_posix() in {EXPORT_INFO_NAME, MANIFEST_NAME}:
            continue
        if not path.is_file() and not path.is_symlink():
            continue
        entries, local = info, relative
        for scope, child in scopes:
            if relative.is_relative_to(scope):
                entries, local = child, relative.relative_to(scope)
                break
        entry = {"path": relative.as_posix(), "type": "json"}
        if entry in entries.values():
            continue
        name = re.sub(r"[^a-z0-9_.-]", "_", "__".join(local.with_suffix("").parts).lower())
        if not name or not name[0].isalnum():
            name = "json_" + name
        if name in entries:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED, "JSON export entry name collision", name=name, path=str(relative)
            )
        entries[name] = entry
