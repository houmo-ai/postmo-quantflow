# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from ...workflows.base import BaseLLMWorkflow
from ...workflows.result import QuantResult


class Glm53Workflow(BaseLLMWorkflow):
    expected_model_config_cls_name = "XHGlm53ModelConfig"
    expected_model_cls_name = "XHGlm53Model"

    def quant(self, output_dir, device, config_overrides=None):
        config = self.workflow_config.with_overrides(config_overrides)
        if config.quant is None:
            return QuantResult(raw_model_dir=self.model_dir, skipped=True)
        from dataclasses import asdict
        from .quant_adapter import quantize_glm53

        result = quantize_glm53(model_dir=self.model_dir, output_dir=output_dir, device=device, config=config.quant)
        return QuantResult(raw_model_dir=self.model_dir, quanted_model_dir=result.output_dir, meta=asdict(result))
