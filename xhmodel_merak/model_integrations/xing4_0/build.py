# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Package Xing config aliases and validate its local native tokenizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_build.contracts import BuildPlan
from xhmodel_merak.model_integrations.autoregressive import build as text_build
from xhmodel_merak.model_integrations.autoregressive.build import normalize_export, package_runtime, sanitize_artifact
from xhmodel_merak.model_integrations.autoregressive.export_info import describe_export
from xhmodel_merak.model_integrations.autoregressive.vision import validate_dynamic_vision

from .checkpoint import inspect_quant_regions


def _validate_native_tokenizer(path: Path, plan: BuildPlan) -> None:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "runtime validation"}], tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer(text, return_token_type_ids=False).get("input_ids")
        if not isinstance(input_ids, list) or not input_ids:
            raise ValueError("tokenizer produced no input_ids")
    except Exception as exc:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Xing Artifact has no usable local tokenizer snapshot: {exc}",
            model=plan.model_id,
            path=str(path),
        ) from exc


def validate_artifact(stage_path: Path, plan: BuildPlan) -> dict[str, Any]:
    meta_path = stage_path / "golden_meta_info.json"
    merak_path = stage_path / "merak_config.json"
    missing = [path.name for path in (meta_path, merak_path) if not path.is_file()]
    if missing:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Exported autoregressive Artifact is missing required runtime files: {missing}",
            model=plan.model_id,
            missing=missing,
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"golden_meta_info.json must contain an object: {meta_path}",
            model=plan.model_id,
        )

    dynamic_vision = None
    if plan.vision_gears:
        dynamic_vision = validate_dynamic_vision(
            stage_path, meta.get("visual_config"), model_id=plan.model_id,
            vision_profile=plan.vision_profile, vision_gears=plan.vision_gears,
        )
    text_build._validate_runtime_lengths(meta, plan)
    text_build._validate_speculative_assets(stage_path, meta, plan)
    text_build._validate_local_runtime_assets(stage_path, meta, plan)
    _validate_native_tokenizer(stage_path / str(meta["hf_config"]), plan)
    regions = inspect_quant_regions(stage_path / str(meta["hf_config"]), model_id=plan.model_id)
    children = [
        {"path": child.parent.relative_to(stage_path).as_posix(), "metadata": validate_artifact(child.parent, plan)}
        for child in text_build._lora_metadata_paths(stage_path, meta)
    ]
    return {
        "children": children,
        "metadata_file": meta_path.name,
        "export_entries": describe_export(stage_path, meta),
        "export_json_roots": {
            adapter["name"]: Path(adapter["meta_file"]).parent.as_posix()
            for adapter in meta.get("lora_adapters", ())
        },
        "golden_meta_info": meta_path.name,
        "merak_config": merak_path.name,
        "components": list(plan.components),
        "golden_components": list(plan.golden_components),
        "golden_artifacts": text_build._collect_golden_artifacts(stage_path, plan),
        "vision_profile": plan.vision_profile,
        "dynamic_vision": dynamic_vision,
        "requires_source": False,
        "loadable": True,
        "checkpoint_precision_regions": regions,
    }


__all__ = ["normalize_export", "package_runtime", "sanitize_artifact", "validate_artifact"]
