"""PaddleOCR-VL Recipe planning and visual Workflow validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_build.contracts import SourceRef
from xhmodel_merak.model_catalog.contracts import ModelRecipe, VariantRecipe


def resolve_config(
    *,
    recipe: ModelRecipe,
    variant: VariantRecipe,
    model_dir: Path,
    sources: Mapping[str, SourceRef],
    workflow: Mapping[str, Any],
    quant_checkpoint: str | None = None,
    precision=None,
) -> dict[str, Any]:
    """Reject drift from the visual ABI exported by PaddleOCR-VL."""
    del variant, model_dir, sources
    _validate_visual_workflow(recipe, workflow)
    return {}


def _validate_visual_workflow(recipe: ModelRecipe, workflow: Mapping[str, Any]) -> None:
    export = workflow.get("export")
    model = export.get("model") if isinstance(export, Mapping) else None
    visual = model.get("visual_config") if isinstance(model, Mapping) else None
    expected = {
        "visual_input_mode": "patches",
        "image_token_gears": list(recipe.vision_gears),
        "image_token_capacity": max(recipe.vision_gears),
        "patch_size": 14,
        "spatial_merge_size": 2,
        "max_size_w": 896,
        "max_size_h": 896,
    }
    mismatches = {}
    for field, value in expected.items():
        actual = visual.get(field) if isinstance(visual, Mapping) else None
        if actual != value or type(actual) is not type(value):
            mismatches[field] = {"expected": value, "actual": actual}
    if mismatches:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "PaddleOCR-VL Workflow visual contract does not match the registered Recipe",
            model=recipe.model_id,
            mismatches=mismatches,
        )


__all__ = ["resolve_config"]
