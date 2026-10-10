from __future__ import annotations

from pathlib import Path
from typing import Any

from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_inference.contracts import TextInferenceResult
from xhmodel_merak.model_inference.inputs import validate_inputs
from xhmodel_merak.model_integrations.autoregressive.inference import run_text_generation


class _Spark25Adapter:
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
        output_text, output_tokens, latency_s, tokens_per_second, stats = run_text_generation(
            meta_file,
            prompt=normalized["prompt"],
            max_new_tokens=int(normalized["max_new_tokens"]),
            device=device,
            device_map=True,
            pack_w4=False,
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


adapter = _Spark25Adapter()


__all__ = ["adapter"]
