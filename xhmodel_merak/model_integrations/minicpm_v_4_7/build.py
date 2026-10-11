"""Package MiniCPM-V-4.7 root-metadata exports without flattening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import confined_file
from xhmodel_merak.model_build.contracts import BuildPlan

from .artifact import describe_export, load_metadata, validate_runtime_contract


def normalize_export(stage_path: Path, export_result: Any, plan: BuildPlan) -> str | None:
    """Keep the workflow root layout; bypass autoregressive flatten/rewrite."""

    del export_result, plan
    meta_path = stage_path / "export_meta_info.json"
    if not meta_path.is_file():
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "MiniCPM export must write export_meta_info.json into the build stage",
            path=str(meta_path),
        )
    return None


def package_runtime(stage_path: Path, plan: BuildPlan) -> None:
    # Workflow already packages vision graphs, llm graphs, and hf_config paths.
    del stage_path, plan


def sanitize_artifact(stage_path: Path, plan: BuildPlan) -> None:
    # Root metadata already uses stage-relative graph and config paths.
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
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            str(error),
            model=plan.model_id,
        ) from error
    return {
        "metadata_file": "export_meta_info.json",
        "export_entries": describe_export(meta),
        "runtime_kind": "minicpm_v_4_7_root_meta",
        "merak_model_type": meta["merak_model_type"],
        "components": list(plan.components),
        "requires_source": False,
        "loadable": True,
        "quantized": True,
    }


__all__ = ["normalize_export", "package_runtime", "sanitize_artifact", "validate_artifact"]
