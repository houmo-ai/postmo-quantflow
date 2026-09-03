from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xhmodel_merak.errors import unsupported_combination

from .contracts import ModelRecipe, ResolvedPrecision, VariantRecipe


def select_context_workflow(
    recipe: ModelRecipe,
    variant: VariantRecipe,
    context_length: int | None,
) -> str:
    if not variant.context_workflows:
        return variant.workflow_config
    if context_length is None:
        raise RuntimeError(f"Recipe {recipe.model_id!r} has context-specific workflows without a context length")
    for max_context_length, workflow_config in variant.context_workflows:
        if context_length <= max_context_length:
            return workflow_config
    raise unsupported_combination(
        recipe.model_id,
        f"variant {variant.name!r} has no Workflow for context_length={context_length}",
        variant=variant.name,
        context_length=context_length,
        supported_context_lengths=[limit for limit, _ in variant.context_workflows],
    )


def source_constraint_mismatches(recipe: ModelRecipe, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    mismatches = []
    for dotted_path, expected in recipe.source_constraints:
        actual = _nested_value(config, dotted_path)
        if actual != expected:
            mismatches.append({"field": dotted_path, "expected": expected, "actual": actual})
    return mismatches


def build_artifact_name(
    recipe: ModelRecipe,
    variant: VariantRecipe,
    model_name: str,
    precision: ResolvedPrecision,
    prefill: int | None,
    context: int | None,
) -> str:
    if recipe.fixed_artifact_name is not None:
        return recipe.fixed_artifact_name
    if prefill is None or context is None or recipe.max_pe_length is None:
        raise RuntimeError(f"Recipe {recipe.model_id!r} has no artifact naming contract")
    variant_token = f"_{variant.name}" if variant.name != "full" else ""
    precision_token = "wmix_amix" if precision.components else f"{precision.weight}{precision.activation}"
    vision_token = f"_{recipe.vision_profile}" if recipe.vision_profile else ""
    return (
        f"hmquant_xh2_{model_name}{variant_token}_{precision_token}_{_length_token(prefill)}_"
        f"{_length_token(context)}_mpe{_length_token(recipe.max_pe_length)}{vision_token}"
    )


def _nested_value(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
            continue
        return None
    return current


def matches_manifest_length(value: Any, expected: Any) -> bool:
    """Match integer Recipe dimensions without accepting booleans or floats."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and isinstance(expected, int)
        and not isinstance(expected, bool)
        and value == expected
    )


def _length_token(value: int) -> str:
    if value % 1024 == 0:
        return f"{value // 1024}k"
    return str(value)


__all__ = [
    "build_artifact_name",
    "matches_manifest_length",
    "select_context_workflow",
    "source_constraint_mismatches",
]
