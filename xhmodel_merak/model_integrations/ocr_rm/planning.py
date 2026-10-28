# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Validate the registered Qwen vision gears before loading RM weights."""

from xhmodel_merak.errors import ErrorCode, ModelError


def _validate_visual_workflow(recipe, workflow):
    visual = workflow.get("export", {}).get("model", {}).get("visual_config", {})
    expected = {
        "visual_input_mode": "patches",
        "spatial_merge_size": 2,
        "image_token_gears": list(recipe.vision_gears),
        "image_token_capacity": max(recipe.vision_gears),
    }
    mismatches = {
        key: {"expected": value, "actual": visual.get(key)}
        for key, value in expected.items()
        if visual.get(key) != value or type(visual.get(key)) is not type(value)
    }
    for key in ("max_size_w", "max_size_h"):
        if key in visual:
            mismatches[key] = {"expected": "absent in patches mode", "actual": visual[key]}
    if mismatches:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "OCR RM visual workflow does not match its registered gears",
            model=recipe.model_id,
            mismatches=mismatches,
        )


def resolve_config(*, recipe, workflow, **kwargs):
    _validate_visual_workflow(recipe, workflow)
    return {}
