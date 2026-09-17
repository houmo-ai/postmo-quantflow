"""Quantize, export, and optionally dump Golden data for WeMM-Embedding."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any


XH2MODELZOO_URI_PREFIX = "xh2modelzoo://"
REPO_URI_PREFIX = "repo://"
GPTQMODEL_URI_PREFIX = "gptqmodel://"
FILE_URI_PREFIX = "file://"
SUPPORTED_URI_SCHEMES = ("xh2modelzoo", "repo", "gptqmodel", "file")

QUANT_METHODS = ("autoround", "gptq")
MODEL_SPECS = {}
for size in ("2b", "4b", "9b"):
    root = f"work_dirs/wemm_embedding/{size}"
    config_root = f"configs_merak/workflows/xh2a/llm_models/wemm_embedding/{size}"
    MODEL_SPECS[size] = {
        "model_dir": f"/data01/datasets/WeMM-Embedding-{size.upper()}",
        "methods": {
            "autoround": {
                "config_path": f"{config_root}/wemm_embedding_{size}_xh2a_w4a8.yaml",
                "quant_output_dir": f"{root}/wemm_embedding_quant",
                "export_output_dir": f"{root}/wemm_embedding_export",
            },
            "gptq": {
                "config_path": f"{config_root}/wemm_embedding_{size}_xh2a_w4a8_gptq.yaml",
                "quant_output_dir": f"{root}/wemm_embedding_gptq_quant",
                "export_output_dir": f"{root}/wemm_embedding_gptq_export",
            },
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=sorted(MODEL_SPECS), default="2b")
    parser.add_argument("--quant-method", choices=QUANT_METHODS, default="autoround")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--quant-output-dir", default=None)
    parser.add_argument("--export-output-dir", default=None)
    parser.add_argument(
        "--quant-model-dir",
        default=None,
        help="Reuse an existing quantized checkpoint instead of running quantization",
    )
    parser.add_argument(
        "--skip-quant",
        action="store_true",
        help="Reuse --quant-model-dir or the size-specific default quant directory",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument(
        "--golden-mode",
        choices=("text", "image", "mixed"),
        default="text",
    )
    parser.add_argument("--golden-text", default="Represent this input.")
    parser.add_argument("--golden-image", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> dict[str, str]:
    model_defaults = MODEL_SPECS[args.size]
    defaults = model_defaults["methods"][args.quant_method]
    model_root = os.getenv("WEMM_MODEL_ROOT")
    default_model_dir = model_defaults["model_dir"]
    if model_root:
        default_model_dir = str(Path(model_root) / f"WeMM-Embedding-{args.size.upper()}")
    return {
        "model_dir": args.model_dir or default_model_dir,
        "config_path": args.config_path or defaults["config_path"],
        "quant_output_dir": args.quant_output_dir or defaults["quant_output_dir"],
        "export_output_dir": args.export_output_dir or defaults["export_output_dir"],
    }


def _repo_root() -> Path:
    env_root = os.environ.get("XH2MODELZOO_ROOT")
    if env_root and Path(env_root).expanduser().is_dir():
        return Path(env_root).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs_merak").is_dir() and (parent / "xhmodel_merak").is_dir():
            return parent
    return Path(__file__).resolve().parents[3]


def resolve_xh2modelzoo_uri(value: str, description: str = "resource") -> Path:
    """Resolve recipe URIs and filesystem paths.

    Supported schemes are ``xh2modelzoo://``, ``repo://``, ``gptqmodel://``,
    and ``file://``. Absolute and relative filesystem paths are also accepted.
    """

    return _resolve_resource(value, description)


def _first_existing_file(candidates: list[Path], value: str, description: str) -> Path:
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved.is_file():
            try:
                with resolved.open("rb") as stream:
                    if not stream.read(1):
                        raise ValueError(f"{description} is empty: {resolved}")
            except PermissionError as exc:
                raise PermissionError(f"{description} is not readable: {resolved}") from exc
            return resolved.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{description} not found: {value!r}; searched {searched}")


def _resolve_resource(value: str, description: str) -> Path:
    raw = os.path.expandvars(str(value)).strip()
    if not raw:
        raise FileNotFoundError(f"{description} path is empty")
    if raw.startswith(GPTQMODEL_URI_PREFIX):
        import importlib.util

        relative = raw.removeprefix(GPTQMODEL_URI_PREFIX).lstrip("/")
        spec = importlib.util.find_spec("gptqmodel")
        locations = list(spec.submodule_search_locations or []) if spec else []
        candidates = [Path(package_root) / relative for package_root in locations]
        try:
            return _first_existing_file(candidates, value, description)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{description} gptqmodel resource not found: {raw}. "
                "Install gptqmodel with calibration data, or reuse a quantized "
                "checkpoint with --skip-quant --quant-model-dir."
            ) from exc
    if raw.startswith(XH2MODELZOO_URI_PREFIX):
        relative = raw.removeprefix(XH2MODELZOO_URI_PREFIX).lstrip("/")
        return _first_existing_file(
            [_repo_root() / relative, Path.cwd() / relative],
            value,
            description,
        )
    if raw.startswith(REPO_URI_PREFIX):
        relative = raw.removeprefix(REPO_URI_PREFIX).lstrip("/")
        return _first_existing_file(
            [_repo_root() / relative, Path.cwd() / relative],
            value,
            description,
        )
    if raw.startswith(FILE_URI_PREFIX):
        from urllib.parse import unquote, urlparse

        parsed = urlparse(raw)
        path_text = unquote(parsed.path or "")
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            path_text = f"/{parsed.netloc}{path_text}"
        if not path_text:
            raise FileNotFoundError(f"{description} file URI path is empty: {raw}")
        path = Path(path_text).expanduser()
        if path.is_absolute():
            return _first_existing_file([path], value, description)
        return _first_existing_file(
            [_repo_root() / path_text, Path.cwd() / path_text],
            value,
            description,
        )
    if "://" in raw:
        scheme = raw.split("://", 1)[0]
        raise ValueError(
            f"{description} uses unsupported URI scheme {scheme!r}: {raw}. "
            f"Supported schemes: {', '.join(SUPPORTED_URI_SCHEMES)}, "
            "absolute paths, and relative paths."
        )
    path = Path(raw).expanduser()
    if path.is_absolute():
        return _first_existing_file([path], value, description)
    relative = raw.lstrip("/")
    return _first_existing_file([_repo_root() / relative, Path.cwd() / relative], value, description)


def require_placeholder_hf_model(export_cfg: Mapping[str, Any] | None) -> None:
    """Reject a non-null YAML ``hf_model`` before workflow overwrite."""

    model_cfg = export_cfg.get("model") if isinstance(export_cfg, Mapping) else None
    if not isinstance(model_cfg, Mapping):
        raise ValueError("WeMM export config is missing a model mapping")
    if "hf_model" not in model_cfg:
        raise ValueError("WeMM YAML export.model.hf_model is missing; set it to null before overwrite")
    hf_model = model_cfg["hf_model"]
    if hf_model is not None:
        raise ValueError(
            "WeMM YAML export.model.hf_model must stay null before overwrite; "
            f"got {hf_model!r}. Pass the real checkpoint with --model-dir."
        )


def require_export_hf_model_dir(path: str | Path | None) -> str:
    """Validate the directory that will overwrite YAML ``hf_model: null``."""

    if path in (None, "", "null"):
        raise ValueError(
            "WeMM export.model.hf_model overwrite target is empty; pass --model-dir or a quantized model directory"
        )
    hf_path = Path(str(path)).expanduser()
    if hf_path.is_file():
        raise ValueError(f"WeMM export.model.hf_model overwrite target is a file, expected a directory: {hf_path}")
    if not hf_path.is_dir():
        raise FileNotFoundError(f"WeMM export.model.hf_model overwrite target is not a directory: {hf_path}")
    return str(hf_path)


def preflight_recipe(
    *,
    config_path: str | Path,
    model_dir: str | Path,
    quant_model_dir: str | Path | None = None,
    skip_quant: bool = False,
) -> dict[str, Any]:
    """Fail closed before quant/export when recipe inputs are not usable."""

    from xhquant.api import Config

    config_file = Path(config_path).expanduser()
    if not config_file.is_file():
        raise FileNotFoundError(f"WeMM workflow config not found: {config_file}")

    cfg = Config.fromfile(str(config_file))
    require_placeholder_hf_model(cfg.export)

    model_path = Path(model_dir).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"WeMM model directory does not exist: {model_path}. "
            "YAML export.model.hf_model is a null placeholder overwritten at "
            "export; pass --model-dir or set WEMM_MODEL_ROOT."
        )
    if model_path.is_file():
        raise ValueError(f"WeMM model path is a file, expected a directory containing config.json: {model_path}")
    if not model_path.is_dir():
        raise ValueError(f"WeMM model path is not a directory: {model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"WeMM model directory is missing config.json: {model_path}")

    calibration_jsonl: str | None = None
    if skip_quant or quant_model_dir is not None:
        if quant_model_dir is None:
            raise FileNotFoundError("Quantized model directory is required when --skip-quant is set")
        quant_dir = Path(quant_model_dir).expanduser()
        if not quant_dir.is_dir():
            raise FileNotFoundError(
                f"Quantized model directory not found: {quant_dir}. "
                "Provide --quant-model-dir or a completed quant output directory."
            )
    else:
        jsonl = cfg.quant.calibration.jsonl
        calibration_jsonl = str(resolve_xh2modelzoo_uri(str(jsonl), "quantization calibration jsonl"))

    return {
        "config_path": str(config_file.resolve()),
        "model_dir": str(model_path.resolve()),
        "hf_model_placeholder": True,
        "calibration_jsonl": calibration_jsonl,
    }


def _remove_output_dir(path_text: str) -> None:
    path = Path(path_text).resolve()
    if path in {Path("/"), Path.cwd().resolve(), Path.home().resolve()}:
        raise ValueError(f"Refusing to remove protected output directory: {path}")
    if path.exists():
        shutil.rmtree(path)


def _golden_item(args: argparse.Namespace) -> dict[str, Any]:
    if args.golden_mode == "text":
        return {"text": args.golden_text}
    if not args.golden_image:
        raise ValueError(f"--golden-image is required for --golden-mode {args.golden_mode}")
    image_path = Path(args.golden_image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Golden image not found: {image_path}")
    item: dict[str, Any] = {"image": str(image_path)}
    if args.golden_mode == "mixed":
        item["text"] = args.golden_text
    return item


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = resolve_paths(args)
    from xhmodel_merak.workflows import AutoWorkflow
    from xhmodel_merak.xh_llm.models.wemm_embedding.workflow import XHWeMMEmbeddingWorkflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    reuse_quant = args.skip_quant or args.quant_model_dir is not None
    preflight = preflight_recipe(
        config_path=paths["config_path"],
        model_dir=paths["model_dir"],
        quant_model_dir=args.quant_model_dir or (paths["quant_output_dir"] if reuse_quant else None),
        skip_quant=reuse_quant,
    )
    print(f"preflight: {preflight}")
    workflow = AutoWorkflow.from_config(
        model_dir=paths["model_dir"],
        config_path=paths["config_path"],
    )
    if reuse_quant:
        quant_model_dir = args.quant_model_dir or paths["quant_output_dir"]
        if not Path(quant_model_dir).is_dir():
            raise FileNotFoundError(f"Quantized model directory not found: {quant_model_dir}")
        XHWeMMEmbeddingWorkflow.ensure_quant_provenance_aliases(quant_model_dir)
        quant_result = QuantResult(
            raw_model_dir=paths["model_dir"],
            quanted_model_dir=quant_model_dir,
        )
    else:
        if args.overwrite:
            _remove_output_dir(paths["quant_output_dir"])
        quant_result = workflow.quant(paths["quant_output_dir"], args.device, {})

    if args.overwrite:
        _remove_output_dir(paths["export_output_dir"])
    export_model_dir = paths["model_dir"] if quant_result.skipped else quant_result.quanted_model_dir
    require_export_hf_model_dir(export_model_dir)
    export_result = workflow.export(
        quant_result,
        paths["export_output_dir"],
        args.device,
        {},
    )
    print(f"quant_result: {quant_result}")
    print(f"export_result: {export_result}")
    if args.dump_golden:
        golden_dir = workflow.dump_golden(
            export_result,
            args.device,
            _golden_item(args),
        )
        print(f"golden_dir: {golden_dir}")


__all__ = [
    "FILE_URI_PREFIX",
    "GPTQMODEL_URI_PREFIX",
    "REPO_URI_PREFIX",
    "SUPPORTED_URI_SCHEMES",
    "XH2MODELZOO_URI_PREFIX",
    "parse_args",
    "preflight_recipe",
    "require_export_hf_model_dir",
    "require_placeholder_hf_model",
    "resolve_paths",
    "resolve_xh2modelzoo_uri",
]


if __name__ == "__main__":
    main()
