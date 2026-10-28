# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
import time

from xhmodel_merak.model_inference.contracts import InferenceResult
from xhmodel_merak.model_inference.inputs import require_local_file, validate_inputs


class OcrRewardAdapter:
    def run(self, *, recipe, root, manifest, meta_file, inputs, device, components, selected_golden, golden_artifacts):
        from xhmodel_merak.xh_llm.models.ocr_rm.runtime import OcrRewardRuntime

        normalized = validate_inputs(recipe, inputs)
        original = require_local_file(recipe, "orig_image", normalized["orig_image"])
        rendered = require_local_file(recipe, "render_image", normalized["render_image"])
        runtime = OcrRewardRuntime(meta_file, device=device)
        try:
            started = time.perf_counter()
            scores = runtime.score(original, rendered, normalized["ocr_text"])
            elapsed = time.perf_counter() - started
        finally:
            runtime.close()
        return InferenceResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="classification",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            latency_s=elapsed,
            components=components,
            stats=scores,
        )


adapter = OcrRewardAdapter()
