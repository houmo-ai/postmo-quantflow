"""Validate user input and delegate all model behavior to the VL runtime."""

import time
from dataclasses import dataclass

from xhmodel_merak.model_inference.contracts import InferenceResult
from xhmodel_merak.model_inference.inputs import require_local_file, scoped_environment, validate_inputs


@dataclass(frozen=True, kw_only=True)
class RerankingResult(InferenceResult):
    scores: tuple[float, ...]
    ranking: tuple[int, ...]


class _RerankerAdapter:
    def run(self, *, recipe, root, manifest, meta_file, inputs, device, components, selected_golden, golden_artifacts):
        import torch

        from xhmodel_merak.xh_llm import AutoLLMHONNXModel
        from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import DEFAULT_INSTRUCTION, normalize_item

        del manifest
        normalized = validate_inputs(recipe, inputs)
        items = [normalize_item(normalized["query"]), *[normalize_item(item) for item in normalized["documents"]]]
        for item in items:
            item["image"] = [str(require_local_file(recipe, "image", path)) for path in item["image"]]
        with scoped_environment({"ENABLE_HMINFERENCE_V2": "1"}):
            model = AutoLLMHONNXModel.from_pretrained(str(meta_file), device_map=[device])
            model.to(device)
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            try:
                scores = tuple(
                    model.rerank(items[0], items[1:], instruction=normalized.get("instruction", DEFAULT_INSTRUCTION))
                )
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
            finally:
                model.release_inference_model()
        return RerankingResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="scores",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            latency_s=time.perf_counter() - started,
            components=components,
            golden_components=selected_golden,
            golden_artifacts=golden_artifacts,
            scores=scores,
            ranking=tuple(sorted(range(len(scores)), key=lambda i: -scores[i])),
        )


adapter = _RerankerAdapter()
