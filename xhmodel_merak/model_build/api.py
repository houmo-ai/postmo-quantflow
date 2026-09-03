from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from xhmodel_merak.model_catalog import get_recipe, list_recipes

from .contracts import (
    UNSET,
    BuildResult,
    LoRARequest,
    PrecisionRequest,
    QuantMethod,
    ResourceRef,
    SourceRef,
    SpeculativeRequest,
    UnsetType,
    VisionRequest,
)
from .description import describe_build_parameters
from .resolver import resolve_build_plan


def list_models() -> list[dict[str, Any]]:
    """Return the stable, lightweight model Registry surface."""

    return [recipe.to_summary() for recipe in list_recipes()]


def describe_model(model_id: str, *, full: bool = False) -> dict[str, Any]:
    """Describe editable build parameters, or all model metadata with ``full=True``.

    The default result maps ``build_model`` argument names to their defaults,
    choices and constraints. It is documentation, not a dictionary of kwargs.
    Fixed deployment settings and inapplicable parameters are omitted.
    """

    if not isinstance(full, bool):
        raise TypeError("full must be a bool")
    recipe = get_recipe(model_id)
    return recipe.to_description() if full else describe_build_parameters(recipe)


def build_model(
    *,
    model_id: str,
    model_dir: str,
    output_dir: str,
    device: str | None = None,
    quant_output_dir: str | None = None,
    sources: Mapping[str, SourceRef | Mapping[str, Any] | str] | None = None,
    resources: Mapping[str, ResourceRef | Mapping[str, Any] | str] | None = None,
    profile: str | None = None,
    precision: PrecisionRequest | Mapping[str, Any] | None = None,
    quant_method: QuantMethod | None | UnsetType = UNSET,
    quant_checkpoint: str | None = None,
    context_length: int | None = None,
    prefill_chunk_length: int | None = None,
    components: Sequence[str] | None = None,
    vision: VisionRequest | Mapping[str, Any] | None = None,
    attention: str | None = None,
    speculative: SpeculativeRequest | Mapping[str, Any] | None = None,
    lora: LoRARequest | Mapping[str, Any] | None = None,
) -> BuildResult:
    """Resolve a public Recipe and execute it through the existing Workflow layer."""

    plan = resolve_build_plan(
        model_id=model_id,
        device=device,
        model_dir=model_dir,
        output_dir=output_dir,
        quant_output_dir=quant_output_dir,
        sources=sources,
        resources=resources,
        profile=profile,
        precision=precision,
        quant_method=quant_method,
        quant_checkpoint=quant_checkpoint,
        context_length=context_length,
        prefill_chunk_length=prefill_chunk_length,
        components=components,
        vision=vision,
        attention=attention,
        speculative=speculative,
        lora=lora,
    )
    from .executor import execute_build

    return execute_build(plan)


__all__ = ["build_model", "describe_model", "list_models"]
