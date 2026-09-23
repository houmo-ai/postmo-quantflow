from typing import Any

from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from ...builder import register_llm_model
from ...text_llm_model import TextLLMModel, TextLLMModelConfig
from .minicpm5_hmonnx_inference import XHMiniCPM5HMONNXModel


class XHMiniCPM5ModelConfig(TextLLMModelConfig):
    pass


@register_llm_model("LlamaForCausalLM")
class XHMiniCPM5Model(TextLLMModel):
    transformers_min_version = "4.51.0"
    HF_MODEL_CLS = LlamaForCausalLM
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    HMONNXINFERENCE_CLS = XHMiniCPM5HMONNXModel
    CONFIG_CLS = XHMiniCPM5ModelConfig
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.minicpm5_dense.workflow:XHMiniCPM5Workflow"

    def __init__(self, config: XHMiniCPM5ModelConfig):
        super().__init__(config)

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._model import register_wrap_modules

        register_wrap_modules()
        super().init_wrap_model(hf_model)


__all__ = ["XHMiniCPM5Model", "XHMiniCPM5ModelConfig"]
