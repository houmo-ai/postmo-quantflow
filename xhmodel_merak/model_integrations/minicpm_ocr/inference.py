# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
import time

from xhmodel_merak.model_artifact.files import artifact_file
from xhmodel_merak.model_inference.contracts import TextInferenceResult
from xhmodel_merak.model_inference.inputs import require_local_file, validate_inputs
from .build import describe_artifact


class MiniCPMOcrAdapter:
    def validate_artifact(self, *, recipe, root, manifest, meta_file, **kwargs):
        describe_artifact(root, lambda value, field: artifact_file(root, manifest, value, field), recipe.components)

    def run(self, *, recipe, root, manifest, meta_file, inputs, device, components, selected_golden, golden_artifacts):
        from xhmodel_merak.xh_llm.models.minicpm_ocr.runtime import MiniCPMOcrRuntime

        normalized = validate_inputs(recipe, inputs)
        image = require_local_file(recipe, "image", normalized["image"])
        runtime = MiniCPMOcrRuntime(meta_file, device=device)
        try:
            started = time.perf_counter()
            output = runtime.generate(image, normalized["prompt"], normalized["max_new_tokens"])
            elapsed = time.perf_counter() - started
        finally:
            runtime.close()
        return TextInferenceResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="text",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            latency_s=elapsed,
            components=components,
            output_text=output["output_text"],
            output_tokens=output["output_tokens"],
            tokens_per_second=output["output_tokens"] / elapsed,
            stats={"prompt_tokens": output["prompt_tokens"], "token_ids": output["token_ids"]},
        )


adapter = MiniCPMOcrAdapter()
