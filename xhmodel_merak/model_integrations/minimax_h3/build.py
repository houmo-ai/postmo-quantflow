"""Package MiniMax-H3 multi-component HMONNX exports for the unified build API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import confined_file
from xhmodel_merak.model_build.contracts import BuildPlan

from .artifact import COMPONENTS, load_metadata, validate_runtime_contract


def normalize_export(stage_path: Path, export_result: Any, plan: BuildPlan) -> str | None:
    del stage_path, export_result, plan
    return None


def package_runtime(stage_path: Path, plan: BuildPlan) -> None:
    # MiniMax-H3 ships graphs + export_meta_info.json; host pipeline assets stay in model_dir.
    del stage_path, plan


def sanitize_artifact(stage_path: Path, plan: BuildPlan) -> None:
    del stage_path, plan


def validate_artifact(stage_path: Path, plan: BuildPlan) -> dict[str, Any]:
    root = stage_path.resolve()
    try:
        meta_path = confined_file(root, "export_meta_info.json", "metadata_file")
        meta = load_metadata(meta_path)
        validate_runtime_contract(
            meta,
            components=plan.components,
            resolve_file=lambda value, field: confined_file(root, value, field),
        )
    except ModelError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, str(error), model=plan.model_id) from error
    return {
        "metadata_file": meta_path.name,
        "export_entries": describe_export(meta),
        "runtime_kind": "minimax_h3_hmonnx",
        "components": list(plan.components),
        "requires_source": True,
        "loadable": True,
        "quantized": True,
    }


def describe_export(meta: dict[str, Any]) -> dict[str, Any]:
    components = meta["components"]
    return {
        name: {"path": components[name]["hmonnx_file"], "type": "hmonnx"} for name in COMPONENTS if name in components
    }


__all__ = ["normalize_export", "package_runtime", "sanitize_artifact", "validate_artifact", "describe_export"]
