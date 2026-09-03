"""Lightweight catalog shared by model build and inference."""

from .contracts import AdapterSpec, ArtifactFormatSpec, ModelRecipe, ResolvedPrecision, VariantRecipe
from .registry import ModelCatalog, get_adapter_spec, get_artifact_format_spec, get_recipe, list_recipes
from .resources import resolve_workflow_config


__all__ = [
    "AdapterSpec",
    "ArtifactFormatSpec",
    "ModelCatalog",
    "ModelRecipe",
    "ResolvedPrecision",
    "VariantRecipe",
    "get_adapter_spec",
    "get_artifact_format_spec",
    "get_recipe",
    "list_recipes",
    "resolve_workflow_config",
]
