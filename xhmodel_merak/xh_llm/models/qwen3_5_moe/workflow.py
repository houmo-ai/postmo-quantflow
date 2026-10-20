"""Workflow specialization for the standalone Qwen3.5 MoE text model."""

from ..qwen3_5.workflow import Qwen35Workflow


class Qwen35MoeTextWorkflow(Qwen35Workflow):
    expected_model_cls_names = {"XHQwen3_5MoeCausalLMModel"}
