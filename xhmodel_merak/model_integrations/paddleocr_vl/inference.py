from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_inference.contracts import TextInferenceResult
from xhmodel_merak.model_inference.inputs import (
    collect_golden_dirs,
    require_local_file,
    scoped_environment,
    snapshot_golden_dirs,
    validate_inputs,
)


def run_paddleocr_generation(
    meta_file: Path,
    *,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    device: str,
    enable_golden: bool,
) -> tuple[str, int, float, float, dict[str, Any]]:
    import torch

    from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
    from xhmodel_merak.xh_llm.models.paddleocr_vl.processing import prepare_native_inputs

    requested_device = torch.device(device)
    device_context = torch.cuda.device(requested_device) if requested_device.type == "cuda" else nullcontext()
    with device_context, scoped_environment({"ENABLE_HMINFERENCE_V2": "1"}):
        model = AutoLLMHONNXModel.from_pretrained(
            str(meta_file),
            enable_golden=enable_golden,
            device_map=[requested_device],
        ).to(requested_device)
        processor = model.get_tf_processor()
        prepared = prepare_native_inputs(processor, image_path, prompt)
        inputs = {
            name: value.to(
                device=requested_device,
                dtype=torch.float16 if value.is_floating_point() else value.dtype,
            )
            if torch.is_tensor(value)
            else value
            for name, value in prepared.items()
        }
        prompt_tokens = int(inputs["input_ids"].shape[1])
        _synchronize(requested_device)
        started = time.perf_counter()
        with LLMInferenceContextManager(model, devices=[requested_device]):
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        _synchronize(requested_device)
        latency_s = time.perf_counter() - started
        sequence = generated.sequences if hasattr(generated, "sequences") else generated
        output_ids = sequence[0, prompt_tokens:]
        output_tokens = int(output_ids.numel())
        output_text = processor.tokenizer.decode(
            output_ids.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        grid = inputs.get("image_grid_thw")
        stats = {
            "prompt_tokens": prompt_tokens,
            "image_grid_thw": grid.tolist() if grid is not None else None,
        }
        return (
            output_text,
            output_tokens,
            latency_s,
            output_tokens / latency_s if latency_s > 0 else 0.0,
            stats,
        )


def _synchronize(device: Any) -> None:
    import torch

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class _PaddleOCRVLAdapter:
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
        normalized = validate_inputs(recipe, inputs)
        image_path = require_local_file(recipe, "image", normalized["image"])
        before = snapshot_golden_dirs(root, selected_golden) if selected_golden else {}
        output_text, output_tokens, latency_s, tokens_per_second, stats = run_paddleocr_generation(
            meta_file,
            image_path=image_path,
            prompt=str(normalized["prompt"]),
            max_new_tokens=int(normalized["max_new_tokens"]),
            device=device,
            enable_golden=bool(selected_golden),
        )
        produced_golden = (
            collect_golden_dirs(root, selected_golden, before=before)
            if selected_golden
            else golden_artifacts
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
            golden_artifacts=produced_golden,
            output_text=output_text,
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
            stats=stats,
        )


adapter = _PaddleOCRVLAdapter()


__all__ = ["adapter", "run_paddleocr_generation"]
