from __future__ import annotations

from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_inference.contracts import TextInferenceResult
from xhmodel_merak.model_inference.inputs import (
    collect_golden_dirs,
    require_local_file,
    scoped_environment,
    snapshot_golden_dirs,
    validate_inputs,
)


class _QwenAdapter:
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
        normalized = validate_inputs(recipe, inputs)
        image = normalized.get("image")
        image_path = str(require_local_file(recipe, "image", image)) if image is not None else None
        variant = str(manifest.get("recipe", {}).get("variant", recipe.variants[0].name))
        spec_mode = recipe.get_variant(variant).speculative_mode
        if spec_mode in {"mtp", "dflash"} and image_path is not None:
            raise ModelError(
                ErrorCode.UNSUPPORTED_COMBINATION,
                "The public speculative runtime accepts text input only",
                model=recipe.model_id,
                variant=variant,
            )
        if spec_mode in {"mtp", "dflash"} and selected_golden:
            raise ModelError(
                ErrorCode.GOLDEN_NOT_SUPPORTED,
                "Golden is unavailable for speculative draft sessions",
                model=recipe.model_id,
                variant=variant,
            )
        if "visual" in selected_golden and image_path is None:
            raise ModelError(
                ErrorCode.GOLDEN_NOT_SUPPORTED,
                "Visual Golden requires an image input",
                model=recipe.model_id,
            )
        max_new_tokens = int(normalized.get("max_new_tokens", 64))
        if "decode" in selected_golden and max_new_tokens < 2:
            raise ModelError(
                ErrorCode.UNSUPPORTED_COMBINATION,
                "Decode Golden requires max_new_tokens >= 2 so the decode graph executes",
                model=recipe.model_id,
                max_new_tokens=max_new_tokens,
            )

        runtime_kwargs: dict[str, Any] = {
            "prompt": normalized["prompt"],
            "image_path": image_path,
            "device": device,
            "max_new_tokens": max_new_tokens,
            "stream_output": False,
        }
        if selected_golden:
            runtime_kwargs["golden_components"] = selected_golden
        if spec_mode not in {"mtp", "dflash"}:
            runtime_kwargs["device_map"] = [device]

        golden_before = snapshot_golden_dirs(root, selected_golden) if selected_golden else {}
        with scoped_environment({"ENABLE_HMINFERENCE_V2": "1"}):
            from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import quick_test_hmonnx

            runtime_result = quick_test_hmonnx(str(meta_file), **runtime_kwargs)

        produced_golden = (
            collect_golden_dirs(root, selected_golden, before=golden_before) if selected_golden else golden_artifacts
        )
        return TextInferenceResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="text",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            latency_s=float(runtime_result.latency_s),
            components=components,
            golden_components=selected_golden,
            golden_artifacts=produced_golden,
            output_text=str(runtime_result.output_text),
            output_tokens=int(runtime_result.output_tokens),
            tokens_per_second=float(runtime_result.tokens_per_second),
            spec_decode_mode=getattr(runtime_result, "spec_decode_mode", None),
            stats=dict(getattr(runtime_result, "stats", {}) or {}),
        )


adapter = _QwenAdapter()


__all__ = ["adapter"]
