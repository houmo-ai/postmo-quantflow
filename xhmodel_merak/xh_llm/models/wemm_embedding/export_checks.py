"""Fail-closed checks for WeMM export config placeholders."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def is_placeholder_hf_model(value: Any) -> bool:
    """Return True when YAML ``hf_model`` is still the null placeholder."""

    return value is None


def require_placeholder_hf_model(export_cfg: Mapping[str, Any] | None) -> None:
    """Reject a non-null YAML ``hf_model`` before workflow overwrite."""

    model_cfg = _model_config(export_cfg)
    if "hf_model" not in model_cfg:
        raise ValueError("WeMM YAML export.model.hf_model is missing; set it to null before overwrite")
    hf_model = model_cfg["hf_model"]
    if not is_placeholder_hf_model(hf_model):
        raise ValueError(
            "WeMM YAML export.model.hf_model must stay null before overwrite; "
            f"got {hf_model!r}. Pass the real checkpoint with --model-dir."
        )


def require_overwritten_hf_model(export_cfg: Mapping[str, Any] | None) -> str:
    """Reject leftover YAML ``hf_model: null`` after workflow overwrite."""

    model_cfg = _model_config(export_cfg)
    if "hf_model" not in model_cfg:
        raise ValueError("WeMM export.model.hf_model is missing after workflow overwrite")
    return require_hf_model_directory(model_cfg["hf_model"])


def require_hf_model_directory(path: Any) -> str:
    """Reject leftover placeholders, files, and missing paths."""

    if path is None or path == "" or path == "null":
        raise ValueError(
            "WeMM export.model.hf_model is still null after workflow overwrite; "
            "the YAML placeholder must be replaced with the quantized model directory."
        )
    hf_path = Path(str(path)).expanduser()
    if hf_path.is_file():
        raise ValueError(f"WeMM export.model.hf_model is a file, expected a directory: {hf_path}")
    if not hf_path.is_dir():
        raise FileNotFoundError(f"WeMM export.model.hf_model is not a directory: {hf_path}")
    return str(hf_path)


def _model_config(export_cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    model_cfg = export_cfg.get("model") if isinstance(export_cfg, Mapping) else None
    if not isinstance(model_cfg, Mapping):
        raise ValueError("WeMM export config is missing a model mapping")
    return model_cfg


__all__ = [
    "is_placeholder_hf_model",
    "require_hf_model_directory",
    "require_overwritten_hf_model",
    "require_placeholder_hf_model",
]
