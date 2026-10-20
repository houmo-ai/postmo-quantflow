"""Merak workflow configuration for Qwen4-Exp."""

from ...types import BaseLLMModelConfig
from .precision import DEFAULT_PLE_QUANT_TYPE, resolve_ple_quant_type


class XHQwen4ExpModelConfig(BaseLLMModelConfig):
    """Route Qwen4-Exp configs through the model-specific workflow."""

    def __init__(self, *, ple_quant_type: str = DEFAULT_PLE_QUANT_TYPE, **kwargs):
        ple_quant_type = resolve_ple_quant_type(ple_quant_type)
        super().__init__(**kwargs)
        self.ple_quant_type = ple_quant_type


__all__ = ["XHQwen4ExpModelConfig"]
