"""Runtime contract owned by the autoregressive integration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import artifact_file, path_contains_symlink
from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_catalog.identity import matches_manifest_length, source_constraint_mismatches

from .vision import validate_dynamic_vision


def validate_artifact(
    root: Path,
    manifest: Mapping[str, Any],
    meta_file: Path,
    recipe: ModelRecipe,
) -> None:
    try:
        metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Failed to parse autoregressive runtime metadata",
            path=str(meta_file),
            reason=str(exc),
        ) from exc
    if not isinstance(metadata, Mapping):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Autoregressive runtime metadata must contain a mapping",
            path=str(meta_file),
        )

    parameters = manifest.get("parameters", {})
    model_config = metadata.get("model_config")
    if not isinstance(parameters, Mapping) or not isinstance(model_config, Mapping):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Autoregressive runtime metadata is missing model_config length bindings",
            model=recipe.model_id,
            path=str(meta_file),
        )
    expected_lengths = {
        "context_max_length": parameters.get("context_length"),
        "prefill_chunk_length": parameters.get("prefill_chunk_length"),
        "max_pe_length": parameters.get("max_pe_length"),
    }
    length_mismatches = {
        field: {"expected": expected_value, "actual": model_config.get(field)}
        for field, expected_value in expected_lengths.items()
        if not matches_manifest_length(model_config.get(field), expected_value)
    }
    if length_mismatches:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Autoregressive runtime metadata lengths do not match Manifest parameters",
            model=recipe.model_id,
            path=str(meta_file),
            mismatches=length_mismatches,
        )

    directory_keys = {"hf_config", "hf_model", "fallback_hf_model"}
    file_keys = {
        "quant_embedding",
        "prefill_hmonnx",
        "decode_hmonnx",
        "prefill_onnx",
        "decode_onnx",
        "token_embedding_file",
        "per_layer_input_embedding",
        "gear_manifest",
        "hmonnx",
        "onnx",
    }

    def visit(value: Any, field: str) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{field}[{index}]")
            return
        if not isinstance(value, Mapping):
            return
        for key, item in value.items():
            child_field = f"{field}.{key}" if field else str(key)
            if key in directory_keys and isinstance(item, str) and item:
                _artifact_directory(root, manifest, item, child_field)
            elif (key in file_keys or key.endswith(("_hmonnx", "_onnx"))) and isinstance(item, str) and item:
                artifact_file(root, manifest, item, child_field)
            else:
                visit(item, child_field)

    visit(metadata, "runtime_metadata")
    hf_config_value = metadata.get("hf_config")
    if not isinstance(hf_config_value, str) or not hf_config_value:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Autoregressive runtime metadata must identify the packaged HF config",
            model=recipe.model_id,
        )
    hf_config = _artifact_directory(root, manifest, hf_config_value, "runtime_metadata.hf_config")
    config_path = hf_config / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Packaged HF config is missing or invalid",
            model=recipe.model_id,
            path=str(config_path),
            reason=str(exc),
        ) from exc
    if not isinstance(config, Mapping):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Packaged HF config must contain a mapping",
            model=recipe.model_id,
            path=str(config_path),
        )
    mismatches = source_constraint_mismatches(recipe, config)
    if mismatches:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Packaged HF config does not match the installed Registry identity",
            model=recipe.model_id,
            path=str(config_path),
            mismatches=mismatches,
        )

    if recipe.vision_gears:
        validate_dynamic_vision(
            root,
            metadata.get("visual_config"),
            model_id=recipe.model_id,
            vision_profile=recipe.vision_profile,
            vision_gears=recipe.vision_gears,
        )


def _artifact_directory(root: Path, manifest: Mapping[str, Any], value: str, field: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact directory path must remain inside the Artifact root",
            field=field,
            path=value,
        )
    candidate = root / relative
    if path_contains_symlink(candidate.absolute()):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact runtime directories must not use symlinks",
            field=field,
            path=value,
        )
    path = candidate.resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Artifact runtime directory is missing or escapes the Artifact root",
            field=field,
            path=value,
        )
    prefix = relative.as_posix().rstrip("/") + "/"
    files = manifest.get("artifact", {}).get("files", ())
    if not any(
        isinstance(entry, Mapping) and isinstance(entry.get("path"), str) and str(entry["path"]).startswith(prefix)
        for entry in files
    ):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Manifest v2 runtime directory contains no inventoried files",
            field=field,
            path=value,
        )
    return path


__all__ = ["matches_manifest_length", "validate_artifact"]
