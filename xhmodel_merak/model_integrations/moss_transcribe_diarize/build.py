from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import confined_file
from xhmodel_merak.model_build.contracts import BuildPlan
from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.packaging import copy_official_runtime_files
from .artifact import load_metadata, validate_runtime_contract


def normalize_export(stage_path: Path, export_result: Any, plan: BuildPlan) -> str | None:
    del stage_path, export_result, plan
    return None


def package_runtime(stage_path: Path, plan: BuildPlan) -> None:
    copy_official_runtime_files(plan.model_dir, stage_path)


def sanitize_artifact(stage_path: Path, plan: BuildPlan) -> None:
    del stage_path, plan


def validate_artifact(stage_path: Path, plan: BuildPlan) -> dict[str, Any]:
    root = stage_path.resolve()
    try:
        meta_path = confined_file(root, "export_meta_info.json", "metadata_file")
        meta = load_metadata(meta_path)
        validate_runtime_contract(
            meta,
            resolve_file=lambda value, field: confined_file(root, value, field),
        )
    except ModelError:
        raise
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, str(error), model=plan.model_id) from error
    return {
        "metadata_file": "export_meta_info.json",
        "runtime_kind": "moss_transcribe_diarize_hmonnx",
        "components": list(plan.components),
        "requires_source": False,
        "loadable": True,
        "quantized": True,
    }


__all__ = ["normalize_export", "package_runtime", "sanitize_artifact", "validate_artifact"]
