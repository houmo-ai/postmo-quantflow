"""Unified API inference adapter for WeMM embedding artifacts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import invalid_parameter
from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_inference.contracts import InferenceResult
from xhmodel_merak.model_inference.inputs import require_local_file, scoped_environment, validate_inputs


@dataclass(frozen=True, kw_only=True)
class EmbeddingInferenceResult(InferenceResult):
    embedding: tuple[float, ...]


class _WeMMEmbeddingAdapter:
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
    ) -> EmbeddingInferenceResult:
        del manifest
        import torch

        from xhmodel_merak.xh_llm import AutoLLMHONNXModel

        normalized = validate_inputs(recipe, inputs)
        item = {name: normalized[name] for name in ("text", "image", "video") if normalized.get(name) is not None}
        if not item:
            raise invalid_parameter(recipe.model_id, "inputs", inputs, "requires text, image, or video")
        for name in ("image", "video"):
            if name in item:
                item[name] = str(require_local_file(recipe, name, item[name]))

        requested_device = torch.device(device)
        with scoped_environment({"ENABLE_HMINFERENCE_V2": "1"}):
            model = AutoLLMHONNXModel.from_pretrained(
                str(meta_file), device_map=[requested_device], enable_golden=False
            )
            if requested_device.type == "cuda":
                torch.cuda.synchronize(requested_device)
            started = time.perf_counter()
            vector = model.embed(item)
            if requested_device.type == "cuda":
                torch.cuda.synchronize(requested_device)
            latency_s = time.perf_counter() - started

        if vector.ndim != 2 or vector.shape[0] != 1 or not bool(torch.isfinite(vector).all()):
            raise ValueError(f"WeMM runtime must return one finite embedding row, got {tuple(vector.shape)}")
        embedding = tuple(float(value) for value in vector[0].detach().float().cpu().tolist())
        return EmbeddingInferenceResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="embedding",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            latency_s=latency_s,
            components=components,
            golden_components=selected_golden,
            golden_artifacts=golden_artifacts,
            embedding=embedding,
            stats={"dimension": len(embedding)},
        )


adapter = _WeMMEmbeddingAdapter()


__all__ = ["EmbeddingInferenceResult", "adapter"]
