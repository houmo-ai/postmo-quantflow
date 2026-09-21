"""Unified-API inference adapter for MiniMax-H3 HMONNX artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError, unsupported_combination
from xhmodel_merak.model_artifact.files import confined_file
from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_inference.contracts import InferenceResult
from xhmodel_merak.model_inference.inputs import require_local_file, validate_inputs

from .artifact import load_metadata, validate_runtime_contract


class _MiniMaxH3Adapter:
    def validate_artifact(
        self,
        *,
        recipe: ModelRecipe,
        root: Path,
        manifest: dict[str, Any],
        meta_file: Path,
        **_: Any,
    ) -> None:
        del manifest
        try:
            meta = load_metadata(meta_file)
            validate_runtime_contract(
                meta,
                components=recipe.components,
                resolve_file=lambda value, field: confined_file(root, value, field),
            )
        except ModelError:
            raise
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                f"MiniMax-H3 artifact contract invalid; re-export required: {error}",
                model=recipe.model_id,
            ) from error

    def run(
        self,
        *,
        recipe: ModelRecipe,
        root: Path,
        manifest: dict[str, Any],
        meta_file: Path,
        inputs: dict[str, Any],
        device: str,
        components: tuple[str, ...],
        selected_golden: tuple[str, ...],
        golden_artifacts: dict[str, tuple[str, ...]],
    ) -> InferenceResult:
        del manifest, golden_artifacts
        if selected_golden:
            raise unsupported_combination(
                recipe.model_id,
                "MiniMax-H3 unified inference does not support golden capture yet",
                golden=list(selected_golden),
            )
        normalized = validate_inputs(recipe, inputs)
        image = None
        if normalized.get("image"):
            image = require_local_file(recipe, "image", normalized["image"])
        workflow = str(normalized.get("workflow", "fl2va"))
        if workflow == "fl2va" and image is None:
            raise ModelError(
                ErrorCode.INVALID_PARAMETER,
                "MiniMax-H3 fl2va inference requires a local image path",
                model=recipe.model_id,
                field="image",
            )
        if workflow == "t2va" and image is not None:
            raise ModelError(
                ErrorCode.INVALID_PARAMETER,
                "MiniMax-H3 t2va inference does not accept image",
                model=recipe.model_id,
                field="image",
            )

        # Public unified inference currently packages/validates artifacts only.
        # End-to-end video generation still uses examples_merak/aigc/minimaxh3/hmonnx_demo.py
        # because InferenceResult has no video/audio AIGC payload type yet.
        raise unsupported_combination(
            recipe.model_id,
            (
                "MiniMax-H3 unified inference is not registered yet; "
                "use examples_merak.aigc.minimaxh3.hmonnx_demo with the published artifact"
            ),
            workflow=workflow,
            device=device,
            components=list(components),
            prompt=normalized["prompt"],
            image=str(image) if image is not None else None,
            output=normalized.get("output"),
            meta_file=str(meta_file),
            artifact_dir=str(root),
        )


adapter = _MiniMaxH3Adapter()


__all__ = ["adapter"]
