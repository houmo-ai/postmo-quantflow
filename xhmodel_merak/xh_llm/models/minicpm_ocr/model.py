# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-OCR vision with the shared Qwen3.5 text export ABI."""

from xhmodel_merak.xh_llm.builder import register_llm_model
from xhmodel_merak.xh_llm.models.minicpm_v_4_6.text_model import MiniCPMV46TextModel


@register_llm_model("MiniCPMOcrForConditionalGeneration")
class XHMiniCPMOcrModel(MiniCPMV46TextModel):
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.minicpm_ocr.workflow:MiniCPMOcrWorkflow"
    transformers_min_version = "5.16.1"

    def __init__(self, config):
        from .native import register_native

        register_native()
        super().__init__(config)

    @classmethod
    def get_hf_model(cls, hf_model_dir, quant_weight=None, **kwargs):
        from .native import register_native

        register_native()
        model = super().get_hf_model(hf_model_dir, quant_weight=quant_weight, **kwargs)
        # Transformers 5.19 serializes this exclusively in rope_parameters;
        # Merak's existing Qwen wrapper reads the equivalent explicit field.
        for module in model.modules():
            config = getattr(module, "config", None)
            if config is not None and getattr(config, "model_type", None) == "qwen3_5_text":
                if not hasattr(config, "partial_rotary_factor"):
                    config.partial_rotary_factor = config.rope_parameters["partial_rotary_factor"]
        return model

    @classmethod
    def get_hf_auto_model_cls(cls):
        from .native import MiniCPMOcrNativeModel

        return MiniCPMOcrNativeModel

    def get_tf_processor(self):
        from .native import load_processor

        return load_processor(self.hf_model_dir)
