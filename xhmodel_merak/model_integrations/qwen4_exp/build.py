"""Reuse autoregressive publication with Qwen4's additional state contract."""

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_build.contracts import BuildPlan
from xhmodel_merak.model_catalog.registry import get_recipe
from xhmodel_merak.model_integrations.autoregressive.build import (
    _validate_speculative_assets,
    package_runtime,
)
from xhmodel_merak.model_integrations.autoregressive.build import (
    sanitize_artifact as sanitize_autoregressive_artifact,
)
from xhmodel_merak.model_integrations.autoregressive.build import (
    validate_artifact as validate_autoregressive_artifact,
)

from .artifact import (
    mtp_v3_graph_entries,
    validate_component_vision,
    validate_external_ple_contract,
    validate_serving_contract,
)


def checkpoint_terminal_mismatches(
    tensor_shapes: Mapping[str, tuple[int, ...]], plan: BuildPlan,
) -> list[dict[str, Any]]:
    """Require Qwen4's native HC terminal weights and shapes before export."""
    config = json.loads((Path(plan.model_dir) / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    mismatches: list[dict[str, Any]] = []
    for suffix, dimensions in get_recipe(plan.model_id).checkpoint_terminal_tensors:
        expected = []
        invalid_fields = set()
        for expression in dimensions:
            size = 1
            for field in expression.split("*"):
                value = text.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    invalid_fields.add(field)
                else:
                    size *= value
            expected.append(size)
        if invalid_fields:
            mismatches.append({"tensor": suffix, "invalid_config_fields": sorted(invalid_fields)})
            continue
        pattern = re.compile(r"^(?:model(?:\.language_model)?|language_model)\." + re.escape(suffix) + "$")
        matches = [name for name in tensor_shapes if pattern.search(name)]
        if not matches:
            mismatches.append({"tensor": suffix, "expected": expected, "actual": None})
        for name in matches:
            if tensor_shapes[name] != tuple(expected):
                mismatches.append({"tensor": name, "expected": expected, "actual": list(tensor_shapes[name])})
    return mismatches


def project_checkpoint_config(config: Mapping[str, Any], *, source: bool) -> dict[str, Any]:
    """Compare the native Qwen4 identities across HF/GPTQModel serialization."""
    del source
    result = deepcopy(dict(config))
    text = result.get("text_config")
    if isinstance(text, dict) and isinstance(text.get("layer_types"), list):
        # Qwen4ExpTextConfig.__post_init__ renames each legacy full-attention
        # entry; layer order, count and all attention geometry remain identity.
        text["layer_types"] = [
            "qwen_sparse_attention" if layer == "full_attention" else layer
            for layer in text["layer_types"]
        ]
    vision = result.get("vision_config")
    if isinstance(vision, dict) and vision.get("model_type") == "qwen4_exp":
        # Qwen4ExpConfig explicitly repairs this legacy vision model_type.
        vision["model_type"] = "qwen4_exp_vision"
    if result.get("use_cache") is None or isinstance(result.get("use_cache"), bool):
        # GPTQModel's calibration looper toggles this composite runtime flag.
        # Native Qwen4 caches are forward arguments, not checkpoint parameters;
        # keep text_config.use_cache and every other field under strict checks.
        result.pop("use_cache", None)
    return result


def normalize_export(stage_path: Path, export_result: Any, plan: BuildPlan) -> None:
    if (
        Path(export_result.work_dir).resolve() != stage_path.resolve()
        or not (stage_path / "golden_meta_info.json").is_file()
    ):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Qwen4 serving export must publish its metadata at the build staging root",
            model=plan.model_id,
        )
    from xhmodel_merak.xh_llm.models.qwen4_exp.shared_weights import publish_qwen4_exp_shared_weights

    publish_qwen4_exp_shared_weights(stage_path)


def sanitize_artifact(stage_path: Path, plan: BuildPlan) -> None:
    metadata_path = stage_path / "golden_meta_info.json"
    metadata = json.loads(metadata_path.read_text())
    resources = validate_external_ple_contract(stage_path, metadata)
    if resources:
        # The shared publisher retains files referenced by runtime metadata.
        # Resolve Qwen4's Host PLE manifest here, without extending that API.
        metadata["qwen4_exp_runtime_assets"] = sorted({entry["path"] for entry in resources.values()})
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sanitize_autoregressive_artifact(stage_path, plan)


def validate_artifact(stage_path: Path, plan: BuildPlan) -> dict[str, Any]:
    def speculative_validator(root, metadata, resolved_plan):
        spec = metadata.get("spec_decode")
        if isinstance(spec, dict) and spec.get("runtime_contract_version") in (3, 4, 5):
            default_count = 4 if spec["runtime_contract_version"] >= 4 else 1
            requested = resolved_plan.config_overrides.get("export.model.num_draft_tokens", default_count)
            if ("mtp" not in resolved_plan.components
                    or requested != spec.get("num_draft_tokens")):
                raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Unexpected Qwen4 MTP candidate contract")
            validate_serving_contract(root, metadata)
        else:
            _validate_speculative_assets(root, metadata, resolved_plan)

    metadata = json.loads((stage_path / "golden_meta_info.json").read_text(encoding="utf-8"))
    vision = validate_component_vision(stage_path, metadata, image_token_gears=plan.vision_gears)
    # The shared validator's vision profile is the legacy whole-gear ABI.
    # Qwen4 validates its component ABI first, retaining all shared text checks.
    text_plan = replace(
        plan, vision_gears=(), vision_profile=None,
        components=tuple(c for c in plan.components if c != "visual"),
    ) if vision else plan
    result = validate_autoregressive_artifact(stage_path, text_plan, speculative_validator=speculative_validator)
    if vision:
        result["components"] = list(plan.components)
        result["vision_profile"] = plan.vision_profile
        result["dynamic_vision"] = {k: v for k, v in vision.items() if k != "export_entries"}
        result["export_entries"].update(vision["export_entries"])
    validate_serving_contract(stage_path, metadata)
    result["export_entries"].update(validate_external_ple_contract(stage_path, metadata))
    result["export_entries"].update(mtp_v3_graph_entries(metadata))
    return result


__all__ = [
    "checkpoint_terminal_mismatches", "normalize_export", "package_runtime", "project_checkpoint_config",
    "sanitize_artifact", "validate_artifact",
]
