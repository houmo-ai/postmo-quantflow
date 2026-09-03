"""Load a pinned Workflow and invoke its registered, lightweight planning hook."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_catalog.contracts import ModelRecipe, VariantRecipe

from .contracts import SourceRef


def resolve_recipe_config(
    recipe: ModelRecipe,
    variant: VariantRecipe,
    config_path: Path,
    *,
    model_dir: Path,
    sources: Mapping[str, SourceRef],
) -> tuple[str, dict[str, Any]]:
    """Hash the exact bytes validated by the family before building a plan.

    The optional hook belongs to the installed Recipe, never to a user path or
    artifact manifest. It may validate local inputs and return dotted Workflow
    replacements; it must not import a runtime, quantize, or export a model.
    """
    try:
        config_bytes = config_path.read_bytes()
        workflow = yaml.safe_load(config_bytes)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Cannot read registered Workflow configuration",
            model=recipe.model_id,
            path=str(config_path),
        ) from exc
    if not isinstance(workflow, Mapping):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Registered Workflow configuration must contain a mapping",
            model=recipe.model_id,
            path=str(config_path),
        )
    replacements: dict[str, Any] = {}
    if recipe.config_resolver is not None:
        module_name, _, attribute = recipe.config_resolver.partition(":")
        try:
            resolver = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as exc:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Registered Recipe config resolver is unavailable",
                model=recipe.model_id,
                target=recipe.config_resolver,
            ) from exc
        if not callable(resolver):
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Registered Recipe config resolver must be callable",
                model=recipe.model_id,
                target=recipe.config_resolver,
            )
        replacements = resolver(
            recipe=recipe, variant=variant, model_dir=model_dir, sources=dict(sources), workflow=workflow
        )
        if not isinstance(replacements, dict) or any(
            not isinstance(key, str) or not key or any(not part for part in key.split(".")) for key in replacements
        ):
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Recipe config resolver must return dotted-path replacements",
                model=recipe.model_id,
                target=recipe.config_resolver,
            )
    return hashlib.sha256(config_bytes).hexdigest(), replacements
