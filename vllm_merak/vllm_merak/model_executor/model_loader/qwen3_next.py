# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Next loader for hybrid Merak HMONNX exports."""

from typing import Any

import torch
from xhmodel_merak.xh_llm import AutoLLMHONNXModel

from ..register import MERAK_MODEL_REGISTRY
from .qwen3_5 import Qwen3_5MerakForCausalLM


@MERAK_MODEL_REGISTRY.register_module("Qwen3NextForCausalLM")
class Qwen3NextMerakForCausalLM(Qwen3_5MerakForCausalLM):
    """Text-only native hybrid runtime with paged Full and isolated Linear state.

    Qwen3-Next and Ling/Bailing share Qwen3.5's split
    convolution/recurrent-state bridge, but have neither a visual tower nor
    M-RoPE. This loader keeps those capabilities accurate while reusing the
    common hybrid cache lifecycle.
    """

    supports_multimodal = False
    supports_multimodal_raw_input_only = False
    requires_raw_input_tokens = False
    supports_mrope = False

    def _prepare_runtime_data_processor_for_positions(
        self,
        data_preprocessor: Any,
        positions: torch.Tensor,
    ) -> None:
        # Qwen3-Next uses ordinary 1-D RoPE. Its HMONNX preprocessor derives
        # positions from past/current lengths and does not accept Qwen3.5's
        # three-axis M-RoPE override.
        del data_preprocessor, positions

    def _apply_runtime_positions(
        self,
        net_inputs: Any,
        positions: torch.Tensor,
        seq_len: int,
    ) -> Any:
        del positions, seq_len
        return net_inputs

    def _load_hmonnx_model(
        self,
        hmonnx_meta_file: str,
        runtime_kwargs: dict[str, Any],
    ) -> Any:
        return AutoLLMHONNXModel.from_pretrained(
            hmonnx_meta_file,
            **runtime_kwargs,
        )


__all__ = ["Qwen3NextMerakForCausalLM"]
