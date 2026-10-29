"""Standard export of an already converted Bonsai checkpoint."""

from ..qwen3_5.workflow import finalize_qwen35_merak_runtime_config
from ...workflows.base import BaseLLMWorkflow


class BonsaiWorkflow(BaseLLMWorkflow):
    expected_model_cls_name = "XHBonsaiModel"
    expected_model_config_cls_name = "XHBonsaiModelConfig"

    def export(self, quant_result, output_dir, device, config_overrides=None):
        result = super().export(quant_result, output_dir, device, config_overrides)
        finalize_qwen35_merak_runtime_config(result)
        return result
