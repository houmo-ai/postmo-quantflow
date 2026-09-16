from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xhmodel_merak._requirements import validate_worker_capabilities, validate_worker_requirements
from xhmodel_merak.errors import ErrorCode, ModelError, invalid_parameter
from xhmodel_merak.model_artifact.files import artifact_file
from xhmodel_merak.model_artifact.validation import (
    read_manifest,
    validate_format_artifact,
    validate_manifest_files,
    validate_recipe_contract,
    validate_schema,
)
from xhmodel_merak.model_catalog.contracts import JSONValue, ModelRecipe
from xhmodel_merak.model_catalog.loading import load_adapter
from xhmodel_merak.model_catalog.registry import get_recipe

from .contracts import InferenceResult
from .golden import _golden_artifacts, _remove_stale_golden_outputs, _resolve_golden
from .inputs import validate_inputs


def inference_model(
    *,
    artifact_dir: str,
    inputs: Mapping[str, JSONValue],
    expected_model_id: str | None = None,
    device: str | None = None,
    golden: Sequence[str] | None = None,
) -> InferenceResult:
    """Validate and run one published Registry artifact."""

    if not isinstance(inputs, Mapping):
        raise invalid_parameter("inference", "inputs", inputs, "must be a mapping")
    if golden is not None and (not isinstance(golden, Sequence) or isinstance(golden, (str, bytes))):
        raise invalid_parameter("inference", "golden", golden, "must be a sequence of component names")
    root, manifest = read_manifest(artifact_dir)
    validate_schema(manifest)
    manifest_model = manifest.get("model_id")
    if expected_model_id is not None and manifest_model != expected_model_id:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact model_id does not match the expected model",
            expected_model_id=expected_model_id,
            model_id=manifest_model,
        )
    if not isinstance(manifest_model, str) or not manifest_model:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact Manifest must declare a Registry model ID",
            model=manifest_model,
        )
    recipe = get_recipe(manifest_model)
    recipe_data = manifest.get("recipe", {})
    if not isinstance(recipe_data, Mapping):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact Manifest recipe must contain a mapping",
            model=recipe.model_id,
        )
    variant_name = recipe_data.get("variant")
    variant = recipe.get_variant(variant_name)
    if variant is None:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact Manifest declares an unregistered Recipe variant",
            model=recipe.model_id,
            variant=variant_name,
            allowed_variants=[variant.name for variant in recipe.variants],
        )
    runtime_inputs = validate_inputs(recipe, inputs)
    device = _resolve_device(device)
    validate_recipe_contract(root, manifest, recipe, variant)
    artifact = manifest.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("loadable") is not True:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact Manifest does not declare the package loadable",
            model=recipe.model_id,
            artifact_dir=str(root),
        )

    _remove_stale_golden_outputs(root, artifact, recipe.golden_components)
    validate_manifest_files(root, artifact)
    meta_file = artifact_file(
        root,
        manifest,
        artifact.get("metadata_file"),
        "artifact.metadata_file",
    )
    validate_format_artifact(recipe, root, manifest, meta_file)
    components = recipe.components + variant.extra_components
    golden_artifacts = _golden_artifacts(root, manifest, artifact)
    selected_golden = _resolve_golden(recipe, components, golden, golden_artifacts)
    adapter = load_adapter("inference", recipe.inference_adapter)
    _validate_adapter_artifact(adapter, recipe, root, manifest, meta_file)
    validate_worker_requirements(recipe.worker_requirements, model=recipe.model_id, operation="inference")
    validate_worker_capabilities(recipe.worker_capabilities, model=recipe.model_id, operation="inference")
    try:
        result = adapter.run(
            recipe=recipe,
            root=root,
            manifest=manifest,
            meta_file=meta_file,
            inputs=runtime_inputs,
            device=device,
            components=components,
            selected_golden=selected_golden,
            golden_artifacts=golden_artifacts,
        )
        validate_manifest_files(
            root,
            artifact,
            runtime_components=selected_golden,
        )
        return result
    except ModelError:
        raise
    except Exception as error:
        raise ModelError(
            ErrorCode.INFERENCE_FAILED,
            "Model runtime failed",
            model=recipe.model_id,
            task=recipe.task,
            adapter=recipe.inference_adapter,
            reason=str(error),
        ) from error


def _validate_adapter_artifact(
    adapter: Any,
    recipe: ModelRecipe,
    root: Path,
    manifest: Mapping[str, Any],
    meta_file: Path,
) -> None:
    validator = getattr(adapter, "validate_artifact", None)
    if not callable(validator):
        return
    try:
        validator(recipe=recipe, root=root, manifest=manifest, meta_file=meta_file)
    except ModelError:
        raise
    except Exception as error:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Inference Artifact does not satisfy the registered adapter contract",
            model=recipe.model_id,
            adapter=recipe.inference_adapter,
            reason=str(error),
        ) from error


def _resolve_device(value: Any) -> str:
    if value is not None and not isinstance(value, str):
        raise invalid_parameter("inference", "device", value, "must be a string")
    device = str(os.getenv("XHMODEL_INFERENCE_DEVICE", "cuda:0") if value is None else value).strip()
    if not device:
        raise invalid_parameter("inference", "device", value, "must be a non-empty string")
    return device


__all__ = ["inference_model"]
