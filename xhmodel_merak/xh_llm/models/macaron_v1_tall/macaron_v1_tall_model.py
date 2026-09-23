"""Macaron-V1-Tall registration over the Qwen3.5-MoE export model."""

from ...builder import register_llm_model
from ..qwen3_5_moe.qwen3_5_moe_model import XHQwen3_5MoeModel
from .macaron_v1_tall_hmonnx_inference import (
    XHMacaronV1TallHMONNXModel,
)


@register_llm_model("MacaronV1TallForConditionalGeneration")
class XHMacaronV1TallModel(XHQwen3_5MoeModel):
    """Select Macaron workflow/runtime while reusing Qwen conversion."""

    WORKFLOW_CLS = (
        "xhmodel_merak.xh_llm.models.macaron_v1_tall.workflow:"
        "MacaronV1TallWorkflow"
    )
    HMONNXINFERENCE_CLS = XHMacaronV1TallHMONNXModel


__all__ = ["XHMacaronV1TallModel"]
