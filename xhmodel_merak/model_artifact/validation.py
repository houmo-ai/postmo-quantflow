"""Artifact contract checks, independent of build and inference execution."""

from __future__ import annotations

import importlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from xhmodel_merak.errors import ErrorCode, ModelError, invalid_parameter
from xhmodel_merak.model_catalog.contracts import ModelRecipe, ResolvedPrecision, VariantRecipe
from xhmodel_merak.model_catalog.identity import build_artifact_name as _build_artifact_name
from xhmodel_merak.model_catalog.identity import matches_manifest_length as _matches_manifest_length
from xhmodel_merak.model_catalog.precision import deployment_precision_from_workflow, resolve_deployment_precision
from xhmodel_merak.model_catalog.registry import get_artifact_format_spec

from . import MANIFEST_NAME as _MANIFEST_NAME
from . import SCHEMA_VERSION
from .files import artifact_file, confined_file, matches_component_directory
from .files import file_sha256 as _sha256
from .files import path_contains_symlink as _path_contains_symlink


def read_manifest(artifact_dir: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(artifact_dir, (str, os.PathLike)) or not str(artifact_dir).strip():
        raise invalid_parameter("inference", "artifact_dir", artifact_dir, "must be an artifact directory")
    unresolved_root = Path(artifact_dir).expanduser().absolute()
    if _path_contains_symlink(unresolved_root):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact root must not resolve through a symlink",
            artifact_dir=str(unresolved_root),
        )
    root = unresolved_root.resolve()
    manifest_path = root / _MANIFEST_NAME
    if not root.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Published Artifact or Manifest was not found",
            artifact_dir=str(root),
            manifest=str(manifest_path),
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Failed to read Artifact Manifest",
            manifest=str(manifest_path),
            reason=str(error),
        ) from error
    if not isinstance(manifest, dict):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact Manifest must contain a mapping",
            manifest=str(manifest_path),
        )
    return root, manifest


def validate_schema(manifest: Mapping[str, Any]) -> None:
    value = manifest.get("schema_version")
    if type(value) is not int or value != SCHEMA_VERSION:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Unsupported artifact schema_version; rebuild with the current build_model API. "
            "Legacy loading is not supported.",
            schema_version=value,
            supported=[SCHEMA_VERSION],
        )


def validate_recipe_contract(
    root: Path,
    manifest: Mapping[str, Any],
    recipe: ModelRecipe,
    variant: VariantRecipe,
) -> None:
    recipe_data = manifest.get("recipe")
    format_data = manifest.get("format")
    artifact = manifest.get("artifact")
    parameters = manifest.get("parameters")
    if (
        not isinstance(recipe_data, Mapping)
        or not isinstance(format_data, Mapping)
        or not isinstance(artifact, Mapping)
        or not isinstance(parameters, Mapping)
    ):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Manifest is missing Recipe, Format, Parameters, or Artifact contract",
            model=recipe.model_id,
        )
    format_spec = get_artifact_format_spec(recipe.artifact_format)
    expected = {
        "format.name": format_spec.name,
        "format.version": format_spec.version,
        "task": recipe.task,
        "modalities": list(recipe.modalities),
    }
    actual = {
        "format.name": format_data.get("name"),
        "format.version": format_data.get("version"),
        "task": manifest.get("task"),
        "modalities": manifest.get("modalities"),
    }
    mismatches = {
        field: {"expected": value, "actual": actual[field]}
        for field, value in expected.items()
        if actual[field] != value or (field == "format.version" and type(actual[field]) is not int)
    }
    # Build provenance is not the runtime ABI. Never reopen today's source YAML
    # or import a module named by an untrusted artifact.
    for field in ("version", "source_config"):
        if not isinstance(recipe_data.get(field), str) or not recipe_data[field]:
            mismatches[f"recipe.{field}"] = {
                "expected": "non-empty provenance string",
                "actual": recipe_data.get(field),
            }
    source_digest = recipe_data.get("source_config_sha256")
    if not isinstance(source_digest, str) or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None:
        mismatches["recipe.source_config_sha256"] = {"expected": "build-time SHA-256", "actual": source_digest}
    effective_workflow_value = recipe_data.get("effective_workflow")
    effective_workflow_sha256 = recipe_data.get("effective_workflow_sha256")
    try:
        effective_workflow = artifact_file(
            root,
            manifest,
            effective_workflow_value,
            "recipe.effective_workflow",
        )
    except ModelError:
        effective_workflow = None
    if effective_workflow_value != "effective_workflow.yaml":
        mismatches["recipe.effective_workflow"] = {
            "expected": "effective_workflow.yaml",
            "actual": effective_workflow_value,
        }
    effective_workflow_valid = not (
        effective_workflow is None
        or not isinstance(effective_workflow_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", effective_workflow_sha256) is None
        or _sha256(effective_workflow) != effective_workflow_sha256
    )
    if not effective_workflow_valid:
        mismatches["recipe.effective_workflow_sha256"] = {
            "expected": "sha256 of the inventoried effective_workflow.yaml",
            "actual": effective_workflow_sha256,
        }

    export_data = manifest.get("export")
    export_data = export_data if isinstance(export_data, Mapping) else {}
    manifest_model_name = export_data.get("model_name")
    model_name_allowed = manifest_model_name == recipe.export_name
    if not model_name_allowed:
        mismatches["export.model_name"] = {
            "expected": recipe.export_name,
            "actual": manifest_model_name,
        }

    effective_data = None
    if effective_workflow_valid and effective_workflow is not None:
        try:
            effective_data = yaml.safe_load(effective_workflow.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Failed to safely parse inventoried effective Workflow",
                model=recipe.model_id,
                path=str(effective_workflow),
                reason=str(exc),
            ) from exc
        if not isinstance(effective_data, Mapping):
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Inventoried effective Workflow must contain a mapping",
                model=recipe.model_id,
                path=str(effective_workflow),
            )
        workflow_export = effective_data.get("export")
        workflow_model = workflow_export.get("model") if isinstance(workflow_export, Mapping) else None
        if not isinstance(workflow_model, Mapping):
            mismatches["effective_workflow.export.model"] = {
                "expected": "mapping",
                "actual": workflow_model,
            }
        else:
            workflow_model_name = workflow_model.get("model_name")
            if workflow_model_name is not None or recipe.fixed_artifact_name is None:
                if workflow_model_name != manifest_model_name:
                    mismatches["effective_workflow.export.model.model_name"] = {
                        "expected": manifest_model_name,
                        "actual": workflow_model_name,
                    }
            if recipe.default_context_length is not None:
                expected_lengths = {
                    "context_max_length": parameters.get("context_length"),
                    "prefill_chunk_length": parameters.get("prefill_chunk_length"),
                    "max_pe_length": parameters.get("max_pe_length"),
                }
                for field, expected_value in expected_lengths.items():
                    actual_value = workflow_model.get(field)
                    if not _matches_manifest_length(actual_value, expected_value):
                        mismatches[f"effective_workflow.export.model.{field}"] = {
                            "expected": expected_value,
                            "actual": actual_value,
                        }

    expected_components = list(recipe.components + variant.extra_components)
    parameter_components = parameters.get("components")
    artifact_components = artifact.get("components")
    if parameter_components != expected_components:
        mismatches["parameters.components"] = {
            "expected": expected_components,
            "actual": parameter_components,
        }
    if artifact_components != expected_components:
        mismatches["artifact.components"] = {
            "expected": expected_components,
            "actual": artifact_components,
        }
    if parameter_components != artifact_components:
        mismatches["parameters.components/artifact.components"] = {
            "expected": parameter_components,
            "actual": artifact_components,
        }

    precision_data = parameters.get("precision")
    checkpoint_precision = precision_data.get("checkpoint") if isinstance(precision_data, Mapping) else None
    deployment_precision = precision_data.get("deployment") if isinstance(precision_data, Mapping) else None
    deployment_valid = isinstance(deployment_precision, Mapping)
    if deployment_valid:
        try:
            editable = {
                component for binding in recipe.deployment_precision_bindings for component in binding.components
            }
            resolved_deployment = resolve_deployment_precision(
                recipe, {key: value for key, value in deployment_precision.items() if key in editable}
            )
            deployment_valid = deployment_precision == resolved_deployment
            if effective_data is not None and recipe.deployment_precision_bindings:
                deployment_valid = deployment_valid and (
                    deployment_precision == deployment_precision_from_workflow(recipe, effective_data)
                )
        except ValueError:
            deployment_valid = False
    if not deployment_valid:
        mismatches["parameters.precision.deployment"] = {
            "expected": "registered component precisions matching the inventoried effective workflow",
            "actual": deployment_precision,
        }

    precision: ResolvedPrecision | None = None
    if not isinstance(checkpoint_precision, Mapping):
        mismatches["parameters.precision.checkpoint"] = {
            "expected": "resolved precision mapping",
            "actual": checkpoint_precision,
        }
    else:
        weight = checkpoint_precision.get("weight")
        activation = checkpoint_precision.get("activation")
        precision_components = checkpoint_precision.get("components", {})
        expected_precision_components = {} if weight == "w16" else recipe.default_precision.components
        if (
            not isinstance(weight, str)
            or not weight
            or not isinstance(activation, str)
            or not activation
            or not isinstance(precision_components, Mapping)
            or (weight, activation) not in recipe.supported_precisions
            or dict(precision_components) != expected_precision_components
        ):
            mismatches["parameters.precision.checkpoint"] = {
                "expected": {
                    "profiles": list(recipe.supported_precisions),
                    "components": expected_precision_components,
                },
                "actual": checkpoint_precision,
            }
        else:
            precision = ResolvedPrecision(
                weight=weight,
                activation=activation,
                components=dict(precision_components),
            )

    duplicated_parameters = {
        "parameters.max_pe_length": (parameters.get("max_pe_length"), recipe.max_pe_length),
        "parameters.golden_components": (
            parameters.get("golden_components"),
            list(recipe.golden_components),
        ),
        "parameters.vision_profile": (parameters.get("vision_profile"), recipe.vision_profile),
        "parameters.vision_gears": (parameters.get("vision_gears"), list(recipe.vision_gears)),
    }
    for field, (value, expected_value) in duplicated_parameters.items():
        if value != expected_value:
            mismatches[field] = {"expected": expected_value, "actual": value}

    profile = parameters.get("profile")
    if (recipe.profiles and profile not in recipe.profiles) or (not recipe.profiles and profile is not None):
        mismatches["parameters.profile"] = {
            "expected": list(recipe.profiles) if recipe.profiles else None,
            "actual": profile,
        }
    quant_method = parameters.get("quant_method")
    checkpoint_methods = set(recipe.quant_methods) | set(recipe.quant_checkpoint_methods)
    if quant_method is not None and quant_method not in checkpoint_methods:
        mismatches["parameters.quant_method"] = {
            "expected": [*sorted(checkpoint_methods), None],
            "actual": quant_method,
        }

    prefill = parameters.get("prefill_chunk_length")
    context = parameters.get("context_length")
    lengths_valid = True
    if recipe.default_context_length is not None:
        lengths_valid = (
            isinstance(prefill, int)
            and not isinstance(prefill, bool)
            and prefill > 0
            and isinstance(context, int)
            and not isinstance(context, bool)
            and context > 0
            and prefill <= context
            and isinstance(recipe.max_pe_length, int)
            and context <= recipe.max_pe_length
        )
        if not lengths_valid:
            mismatches["parameters.sequence_lengths"] = {
                "expected": "positive integer prefill/context lengths within the Recipe max_pe_length",
                "actual": {"prefill_chunk_length": prefill, "context_length": context},
            }
    elif prefill is not None or context is not None:
        lengths_valid = False
        mismatches["parameters.sequence_lengths"] = {
            "expected": {"prefill_chunk_length": None, "context_length": None},
            "actual": {"prefill_chunk_length": prefill, "context_length": context},
        }

    if model_name_allowed and precision is not None and lengths_valid:
        expected_artifact_name = _build_artifact_name(
            recipe,
            variant,
            manifest_model_name,
            precision,
            prefill,
            context,
        )
        if export_data.get("artifact_name") != expected_artifact_name:
            mismatches["export.artifact_name"] = {
                "expected": expected_artifact_name,
                "actual": export_data.get("artifact_name"),
            }
    if mismatches:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact contract does not match the installed model Recipe",
            model=recipe.model_id,
            variant=variant.name,
            mismatches=mismatches,
        )


def validate_manifest_files(
    root: Path,
    artifact: Mapping[str, Any],
    *,
    runtime_components: Sequence[str] = (),
) -> None:
    """Check delivered files; runtime may add regular cache/temporary files."""
    files = artifact.get("files")
    if not isinstance(files, list):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact Manifest files must be a list",
        )
    inventoried: set[str] = set()
    for entry in files:
        if not isinstance(entry, Mapping):
            raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Invalid Artifact file entry")
        path = confined_file(root, entry.get("path"), "artifact.files.path")
        relative = path.relative_to(root).as_posix()
        if relative in inventoried:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Artifact Manifest contains a duplicate file entry",
                path=relative,
            )
        inventoried.add(relative)
        mutable_runtime_file = _is_ephemeral_runtime_file(Path(relative), runtime_components)
        size = entry.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Manifest file entry is missing a valid size",
                path=entry.get("path"),
            )
        if not mutable_runtime_file and path.stat().st_size != size:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Artifact file size does not match Manifest",
                path=entry.get("path"),
            )
        checksum = entry.get("sha256")
        if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Manifest file entry is missing a valid sha256",
                path=entry.get("path"),
            )
        if not mutable_runtime_file and _sha256(path) != checksum:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Artifact file checksum does not match Manifest",
                path=entry.get("path"),
            )
    actual: set[str] = set()
    symlinks = []
    non_regular_nodes = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            symlinks.append(relative)
        elif stat.S_ISDIR(mode):
            continue
        elif stat.S_ISREG(mode):
            if relative != _MANIFEST_NAME:
                actual.add(relative)
        else:
            non_regular_nodes.append(relative)
    symlinks.sort()
    non_regular_nodes.sort()
    missing = sorted(inventoried - actual)
    if symlinks or non_regular_nodes or missing:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact file inventory contains missing files or unsafe filesystem nodes",
            missing_files=missing,
            symlinks=symlinks,
            non_regular_nodes=non_regular_nodes,
        )


def validate_format_artifact(
    recipe: ModelRecipe,
    root: Path,
    manifest: Mapping[str, Any],
    meta_file: Path,
) -> None:
    from .export_info import EXPORT_INFO_NAME, validate_export_info

    export_info = manifest.get("artifact", {}).get("export_meta_info")
    if export_info is not None:
        if export_info != EXPORT_INFO_NAME:
            raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Unexpected export inventory filename")
        path = artifact_file(root, manifest, export_info, "artifact.export_meta_info")
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Invalid export inventory JSON") from error
        validate_export_info(root, info, inventoried={entry["path"] for entry in manifest["artifact"]["files"]})
    format_spec = get_artifact_format_spec(recipe.artifact_format)
    if format_spec.validator is None:
        return
    module_name, separator, attribute = format_spec.validator.partition(":")
    if not separator:
        attribute = "validate_artifact"
    try:
        validator = getattr(importlib.import_module(module_name), attribute)
        validator(root=root, manifest=manifest, meta_file=meta_file, recipe=recipe)
    except ModelError:
        raise
    except Exception as error:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact does not satisfy its registered format contract",
            model=recipe.model_id,
            artifact_format=recipe.artifact_format,
            validator=format_spec.validator,
            reason=str(error),
        ) from error


def _is_ephemeral_runtime_file(relative: Path, components: Sequence[str]) -> bool:
    if re.fullmatch(r"\.xhquant_.*_sha256_.*\.json", relative.name):
        return False
    for index, part in enumerate(relative.parts[:-1]):
        if re.fullmatch(r"step_[0-9]+", part) and any(
            matches_component_directory(ancestor, component)
            for ancestor in relative.parts[:index]
            for component in components
        ):
            return True
    return False
