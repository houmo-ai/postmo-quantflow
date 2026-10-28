"""Use the shared text runtime for Xing4.0 exported artifacts."""

from pathlib import Path
from typing import Any

from xhmodel_merak._environment import scoped_environment
from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_inference.contracts import TextInferenceResult
from xhmodel_merak.model_inference.inputs import validate_inputs
from xhmodel_merak.model_integrations.autoregressive.inference import run_text_generation


class _XingAdapter:
    def run(
        self, *, recipe: ModelRecipe, root: Path, manifest: dict[str, Any],
        meta_file: Path, inputs: dict[str, Any], device: str,
        components: tuple[str, ...], selected_golden: tuple[str, ...],
        golden_artifacts: dict[str, tuple[str, ...]],
    ) -> TextInferenceResult:
        normalized = validate_inputs(recipe, inputs)
        with scoped_environment({"XHQUANT_MOE_LOW_M_TRITON": "0"}):
            text, tokens, latency, rate, stats = run_text_generation(
                meta_file, prompt=normalized["prompt"],
                max_new_tokens=int(normalized["max_new_tokens"]), device=device,
                device_map=True, pack_w4=True,
            )
        return TextInferenceResult(
            model_id=recipe.model_id, task=recipe.task, output_type="text",
            artifact_dir=str(root), meta_file=str(meta_file), latency_s=latency,
            components=components, golden_components=selected_golden,
            golden_artifacts=golden_artifacts, stats=stats, output_text=text,
            output_tokens=tokens, tokens_per_second=rate,
        )


adapter = _XingAdapter()
