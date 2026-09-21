from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import confined_file
from xhmodel_merak.model_build.contracts import BuildPlan
from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.packaging import copy_official_runtime_files, OFFICIAL_RUNTIME_FILES
from .artifact import load_metadata, validate_runtime_contract


def normalize_export(stage_path: Path, export_result: Any, plan: BuildPlan) -> str | None:
    del stage_path, export_result, plan
    return None


def package_runtime(stage_path: Path, plan: BuildPlan) -> None:
    copy_official_runtime_files(plan.model_dir, stage_path)
    hf_config = stage_path / "hf_config"
    hf_config.mkdir(exist_ok=True)
    for name in OFFICIAL_RUNTIME_FILES:
        source = stage_path / name
        if source.is_file():
            source.replace(hf_config / name)
    for name in ("export_meta_info.json", "golden_meta_info.json"):
        path = stage_path / name
        if path.is_file():
            meta = load_metadata(path)
            meta["hf_config"] = "hf_config"
            path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
        "export_entries": describe_export(meta),
        "runtime_kind": "moss_transcribe_diarize_hmonnx",
        "components": list(plan.components),
        "requires_source": False,
        "loadable": True,
        "quantized": True,
    }


def describe_export(meta: dict[str, Any]) -> dict[str, Any]:
    components = meta["components"]
    result = {"encode": {"path": components["encode"]["hmonnx_file"], "type": "hmonnx"}}
    result["tokenizer"] = {"path": meta["hf_config"], "type": "tokenizer"}
    for phase in ("prefill", "decode"):
        result[phase] = {"path": components["prefill_decode"][phase]["hmonnx_file"], "type": phase}
    return result


__all__ = ["normalize_export", "package_runtime", "sanitize_artifact", "validate_artifact"]
