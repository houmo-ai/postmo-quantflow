from __future__ import annotations

from pathlib import Path
from typing import Any

from xhmodel_merak._environment import scoped_environment
from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_inference.contracts import TextInferenceResult
from xhmodel_merak.model_inference.inputs import require_local_file, validate_inputs


class _Qwen4ExpAdapter:
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
        image = normalized.get("image")
        image_path = str(require_local_file(recipe, "image", image)) if image is not None else None
        video = normalized.get("video")
        video_path = str(require_local_file(recipe, "video", video)) if video is not None else None
        with scoped_environment({"ENABLE_HMINFERENCE_V2": "1", "XHQUANT_PACKED_W4": "1"}):
            from xhmodel_merak.xh_llm.models.qwen4_exp.generation import run_text_generation

            kwargs = {}
            if image_path is not None:
                kwargs["image"] = image_path
            if video_path is not None:
                kwargs["video"] = video_path
            if selected_golden:
                import time

                kwargs.update(
                    golden_components=selected_golden, golden_dir=root / "golden" / f"request_{time.time_ns()}"
                )
            output_text, output_tokens, latency_s, tokens_per_second, stats = run_text_generation(
                meta_file,
                prompt=normalized["prompt"],
                max_new_tokens=int(normalized["max_new_tokens"]),
                device=device,
                **kwargs,
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
            golden_artifacts=stats.get("golden_artifacts", golden_artifacts),
            output_text=output_text,
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
            stats=stats,
        )


adapter = _Qwen4ExpAdapter()
