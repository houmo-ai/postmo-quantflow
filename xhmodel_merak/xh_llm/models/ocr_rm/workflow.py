# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from xhmodel_merak.xh_llm.workflows.base import BaseLLMWorkflow


class OcrRewardWorkflow(BaseLLMWorkflow):
    """Quantize graph operators at export; never reinterpret RM as a CausalLM."""

    def export(self, quant_result, output_dir, device, config_overrides=None):
        from xhmodel_merak.xh_llm.workflows.runtime_config import finalize_merak_runtime_config
        from .precision import audit_sefp_h1, validate_sefp_h1_config

        validate_sefp_h1_config(self.workflow_config.with_overrides(config_overrides).build_export_dict()["model"])
        result = super().export(quant_result, output_dir, device, config_overrides)
        audit_sefp_h1(self._find_golden_meta_file(result))
        finalize_merak_runtime_config(result)
        return result
