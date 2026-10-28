"""Dynamic vision artifact contract shared by build and inference."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import confined_file
from xhmodel_merak.model_catalog.contracts import ModelRecipe


def validate_dynamic_vision(
    stage_path: Path,
    visual: Any,
    *,
    model_id: str,
    vision_profile: str | None,
    vision_gears: Sequence[int],
) -> dict[str, Any]:
    if not isinstance(visual, Mapping):
        raise _invalid_dynamic_vision(model_id, "visual_config must be an object")
    expected_gears = list(vision_gears)
    if visual.get("image_token_gears") != expected_gears:
        raise _invalid_dynamic_vision(
            model_id, "image_token_gears must match the registered dynamic vision token gears"
        )
    gear_entries = visual.get("gears")
    if not isinstance(gear_entries, list) or len(gear_entries) != len(expected_gears):
        raise _invalid_dynamic_vision(model_id, "visual_config.gears must map every registered token gear")
    if visual.get("visual_input_mode") != "patches":
        raise _invalid_dynamic_vision(model_id, "visual_input_mode must be 'patches'")
    if visual.get("image_token_capacity") != expected_gears[-1]:
        raise _invalid_dynamic_vision(model_id, "image_token_capacity must equal the largest registered gear")
    if visual.get("routing_policy") != "smallest_fit":
        raise _invalid_dynamic_vision(model_id, "routing_policy must be 'smallest_fit'")

    root = stage_path.resolve()
    manifest_value = visual.get("gear_manifest")
    if manifest_value != "visual_gears.json":
        raise _invalid_dynamic_vision(model_id, "gear_manifest must be root-level visual_gears.json")
    manifest_path = confined_file(root, manifest_value, "visual_config.gear_manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _invalid_dynamic_vision(model_id, f"cannot parse gear_manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise _invalid_dynamic_vision(model_id, "gear_manifest must contain an object")
    manifest_gears = manifest.get("gears")
    if not isinstance(manifest_gears, list) or len(manifest_gears) != len(expected_gears):
        raise _invalid_dynamic_vision(model_id, "gear_manifest.gears must map every registered token gear")
    schema_version = manifest.get("schema_version")
    if schema_version not in (1, 2) or isinstance(schema_version, bool):
        raise _invalid_dynamic_vision(model_id, "unsupported gear_manifest schema_version")
    required_manifest_values = {
        "visual_input_mode": "patches",
        "token_unit": "post_spatial_merge",
        "routing_policy": "smallest_fit",
        "overflow_policy": "reject",
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in required_manifest_values.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise _invalid_dynamic_vision(model_id, "gear_manifest routing contract is invalid", mismatches=mismatches)

    if schema_version == 2:
        extended_fields = (
            "position_format",
            "rotary_position_format",
            "attention_mask_format",
            "attention_mask_shape",
            "attention_mask_operator",
            "spatial_merge_size",
            "rotary_cache_length",
        )
        mismatches = {
            key: {"expected": visual.get(key), "actual": manifest.get(key)}
            for key in extended_fields
            if visual.get(key) is None or manifest.get(key) != visual.get(key)
        }
        if mismatches:
            raise _invalid_dynamic_vision(model_id, "gear_manifest visual geometry is invalid", mismatches=mismatches)

    spatial_merge_size = visual.get("spatial_merge_size")
    if not isinstance(spatial_merge_size, int) or isinstance(spatial_merge_size, bool) or spatial_merge_size <= 0:
        raise _invalid_dynamic_vision(model_id, "spatial_merge_size must be a positive integer")
    graphs: list[dict[str, Any]] = []
    for expected, entry, manifest_entry in zip(expected_gears, gear_entries, manifest_gears, strict=True):
        if not isinstance(entry, Mapping) or not isinstance(manifest_entry, Mapping):
            raise _invalid_dynamic_vision(model_id, "every dynamic vision gear must be an object")
        patch_capacity = expected * spatial_merge_size**2
        if (
            entry.get("image_token_capacity") != expected
            or entry.get("patch_token_capacity") != patch_capacity
            or manifest_entry.get("image_token_capacity") != expected
            or manifest_entry.get("patch_token_capacity") != patch_capacity
        ):
            raise _invalid_dynamic_vision(
                model_id,
                "dynamic vision gear capacity does not match the Registry",
                expected_image_tokens=expected,
                expected_patch_tokens=patch_capacity,
            )
        value = entry.get("hmonnx")
        graph_path = confined_file(root, value, "visual_config.gears.hmonnx")
        relative = Path(value)
        if len(relative.parts) != 2 or relative.parts[0] != f"visual_m{expected}":
            raise _invalid_dynamic_vision(
                model_id,
                f"visual gear {expected} must use visual_m{expected}/ at the Artifact root",
                path=value,
            )
        if value != manifest_entry.get("hmonnx"):
            raise _invalid_dynamic_vision(
                model_id,
                "dynamic vision metadata does not resolve each gear to the same local graph",
                gear=expected,
            )
        graphs.append(
            {
                "image_token_capacity": expected,
                "patch_token_capacity": patch_capacity,
                "hmonnx": graph_path.relative_to(root).as_posix(),
            }
        )
    if visual.get("hmonnx") != graphs[-1]["hmonnx"]:
        raise _invalid_dynamic_vision(model_id, "visual_config.hmonnx must identify the largest gear graph")
    if (root / "visual_meta_info.json").exists() and (root / "golden_meta_info.json").exists():
        raise _invalid_dynamic_vision(model_id, "a full VLM package must not contain visual_meta_info.json")
    return {
        "profile": vision_profile,
        "token_unit": manifest["token_unit"],
        "routing_policy": manifest["routing_policy"],
        "overflow_policy": manifest["overflow_policy"],
        "gear_manifest": manifest_path.relative_to(root).as_posix(),
        "gears": graphs,
    }


def _invalid_dynamic_vision(model_id: str, message: str, **details: Any) -> ModelError:
    return ModelError(
        ErrorCode.ARTIFACT_VALIDATION_FAILED,
        f"Registered dynamic vision Artifact is invalid: {message}",
        model=model_id,
        **details,
    )


def validate_visual_workflow(recipe: ModelRecipe, workflow: Mapping[str, Any]) -> None:
    export = workflow.get("export")
    model = export.get("model") if isinstance(export, Mapping) else None
    visual = model.get("visual_config") if isinstance(model, Mapping) else None
    expected = {
        "visual_input_mode": "patches",
        "image_token_gears": list(recipe.vision_gears),
        "image_token_capacity": max(recipe.vision_gears),
        "spatial_merge_size": 2,
    }
    mismatches = {}
    for field, value in expected.items():
        actual = visual.get(field) if isinstance(visual, Mapping) else None
        if actual != value or type(actual) is not type(value):
            mismatches[field] = {"expected": value, "actual": actual}
    if isinstance(visual, Mapping):
        for field in ("max_size_w", "max_size_h"):
            if field in visual:
                mismatches[field] = {"expected": "absent in patches mode", "actual": visual[field]}
    if mismatches:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Qwen Workflow visual contract does not match the registered Recipe",
            model=recipe.model_id,
            mismatches=mismatches,
        )
