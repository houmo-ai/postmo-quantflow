"""Qwen4-Exp registration for Merak workflow discovery."""

from ...base_llm_model import BaseLLMModel
from ...builder import register_llm_model
from ...types import VLLMModelMeta
from .configuration_qwen4_exp import XHQwen4ExpModelConfig
from .qwen4_exp_hmonnx_inference import XHQwen4ExpHMONNXModel


@register_llm_model("Qwen4ExpForConditionalGeneration")
@register_llm_model("qwen4_exp", master=False)  # Load pre-unified serving metadata.
class XHQwen4ExpModel(BaseLLMModel):
    """Workflow-owned Qwen4-Exp model entry.

    Qwen4-Exp cannot reuse the Qwen3.5 model wrapper because its cache ABI has
    PLE/ngram and QSA indexer state. Conversion is therefore owned by
    :class:`Qwen4ExpWorkflow`; this class supplies the standard Merak registry
    entry used to discover that workflow.
    """

    CONFIG_CLS = XHQwen4ExpModelConfig
    META_CLS = VLLMModelMeta
    HMONNXINFERENCE_CLS = XHQwen4ExpHMONNXModel
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.qwen4_exp.workflow:Qwen4ExpWorkflow"
    transformers_min_version = "5.16.0"

    def export_hmonnx(self, output_dir: str):
        del output_dir
        raise RuntimeError("Qwen4-Exp export is owned by Qwen4ExpWorkflow; use AutoLLMWorkflow")


__all__ = ["XHQwen4ExpModel"]
