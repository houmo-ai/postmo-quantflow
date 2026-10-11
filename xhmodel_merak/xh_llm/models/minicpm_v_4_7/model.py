"""Registered MiniCPM-V-4.7 Dense/MoE models for the Merak LLM workflow."""

from xhmodel_merak.xh_llm.builder import register_llm_model

from .text_model import MiniCPMV47MoeTextModel, MiniCPMV47TextModel


@register_llm_model("MiniCPMV47ForConditionalGeneration")
class XHMiniCPMV47Model(MiniCPMV47TextModel):
    """MiniCPM Vision plus Qwen3.5 Dense text adapter."""

    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.minicpm_v_4_7.workflow:MiniCPMV47Workflow"


@register_llm_model("MiniCPMV47MoeForConditionalGeneration")
class XHMiniCPMV47MoeModel(MiniCPMV47MoeTextModel):
    """MiniCPM Vision plus Qwen3.5 MoE text adapter."""

    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.minicpm_v_4_7.workflow:MiniCPMV47Workflow"


__all__ = ["XHMiniCPMV47Model", "XHMiniCPMV47MoeModel"]
