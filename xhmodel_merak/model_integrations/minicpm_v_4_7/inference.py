"""Unified-API adapter for MiniCPM-V-4.7 root-metadata HMONNX runtime."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import artifact_file
from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_inference.contracts import TextInferenceResult
from xhmodel_merak.model_inference.inputs import (
    require_local_file,
    validate_inputs,
)

from .artifact import load_metadata, validate_runtime_contract


def run_minicpm_v47_generation(
    meta_file: Path,
    *,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    device: str,
    downsample_mode: str = "4x",
) -> tuple[str, int, float, float, dict[str, Any]]:
    import torch

    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.inference import MiniCPMV47HMONNXRuntime

    requested_device = torch.device(device)
    runtime = MiniCPMV47HMONNXRuntime(meta_file, device=str(requested_device))
    messages = {
        "text": prompt,
        "image": str(image_path.resolve()),
    }
    if requested_device.type == "cuda":
        torch.cuda.synchronize(requested_device)
    started = time.perf_counter()
    output_text = runtime.generate(
        messages,
        downsample_mode=downsample_mode,
        max_new_tokens=max_new_tokens,
    )
    if requested_device.type == "cuda":
        torch.cuda.synchronize(requested_device)
    latency_s = time.perf_counter() - started
    # MiniCPM runtime returns decoded text only; approximate token count from whitespace.
    output_tokens = len(str(output_text).split())
    stats = {
        "downsample_mode": downsample_mode,
        "merak_model_type": runtime.meta.get("merak_model_type"),
    }
    return (
        str(output_text).strip(),
        output_tokens,
        latency_s,
        output_tokens / latency_s if latency_s > 0 else 0.0,
        stats,
    )


class _MiniCPMV47Adapter:
    def validate_artifact(
        self,
        *,
        recipe: ModelRecipe,
        root: Path,
        manifest: Mapping[str, Any],
        meta_file: Path,
        **_: Any,
    ) -> None:
        try:
            meta = load_metadata(meta_file)
            validate_runtime_contract(
                meta,
                components=recipe.components + next(v for v in recipe.variants if v.name == manifest.get("recipe", {}).get("variant", "full")).extra_components,
                resolve_file=lambda value, field: artifact_file(root, manifest, value, field),
            )
        except ModelError:
            raise
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                f"MiniCPM runtime contract invalid; re-export required: {error}",
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
    ) -> TextInferenceResult:
        del manifest
        if selected_golden:
            raise ModelError(
                ErrorCode.GOLDEN_NOT_SUPPORTED,
                "MiniCPM-V-4.7 unified inference does not dump golden via this adapter; "
                "use the workflow dump_golden path instead",
                model=recipe.model_id,
            )
        normalized = validate_inputs(recipe, inputs)
        image_path = require_local_file(recipe, "image", normalized["image"])
        output_text, output_tokens, latency_s, tokens_per_second, stats = run_minicpm_v47_generation(
            meta_file,
            image_path=image_path,
            prompt=str(normalized["prompt"]),
            max_new_tokens=int(normalized["max_new_tokens"]),
            device=device,
            downsample_mode=str(normalized.get("downsample_mode", "4x")),
        )
        return TextInferenceResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="text",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            latency_s=latency_s,
            components=components,
            golden_components=selected_golden,
            golden_artifacts=golden_artifacts,
            output_text=output_text,
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
            stats=stats,
        )


adapter = _MiniCPMV47Adapter()


__all__ = ["adapter", "run_minicpm_v47_generation"]
