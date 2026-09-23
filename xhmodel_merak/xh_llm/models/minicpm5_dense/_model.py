import math
import types
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import Tensor
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import LLMCache, MaskedSoftmax, RMSNorm
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
from ..qwen3._model import (
    _Qwen3DecoderLayer,
    _Qwen3ForCausalLM,
    _Qwen3Model,
    _Qwen3RMSNorm,
    _Qwen3RotaryEmbedding,
)


@XHLLM_TRACEABLE_MODULES.register_module({LlamaRotaryEmbedding: "LlamaRotaryEmbedding"})
class _MiniCPM5RotaryEmbedding(_Qwen3RotaryEmbedding):
    pass


if TYPE_CHECKING:
    class _MiniCPM5AttentionBase(DynamicModule, LlamaAttention):
        ...
else:
    _MiniCPM5AttentionBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({LlamaAttention: "LlamaAttention"})
class _MiniCPM5Attention(_MiniCPM5AttentionBase):
    def rotate_half(self, x: Tensor) -> Tensor:
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def graph_forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, None, None]:
        batch_size, query_length, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states = self.rope(query_states, cos, sin)
        key_states = self.rope(key_states, cos, sin)
        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        query_states = query_states * self.kv_scale
        key_states = torch.repeat_interleave(key_states.transpose(2, 3), self.num_key_value_groups, dim=1)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
        attention_weights = torch.matmul(query_states, key_states)
        attention_weights: Optional[Tensor] = self.masked_softmax(attention_weights, past_seq_length)
        attention_output = torch.matmul(attention_weights, value_states).transpose(1, 2)
        attention_output = attention_output.reshape(batch_size, query_length, self.config.hidden_size)
        return self.o_proj(attention_output), None, None

    def _setup(self, cfg: ConfigDict | dict[str, Any]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.num_heads)
        self.rope = xhnn.Rope()
        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [2**63 - 1], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.use_cache = cfg.use_cache
        self.kv_scale = 1 / math.sqrt(self.head_dim)
        if self.use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCache(axis=cache_axis)
            self.v_cache = LLMCache(axis=cache_axis)
        else:
            self.k_cache = self.v_cache = None
        return self


@XHLLM_TRACEABLE_MODULES.register_module({LlamaDecoderLayer: "LlamaDecoderLayer"})
class _MiniCPM5DecoderLayer(_Qwen3DecoderLayer):
    pass


@XHLLM_TRACEABLE_MODULES.register_module({LlamaRMSNorm: "LlamaRMSNorm"})
class _MiniCPM5RMSNorm(_Qwen3RMSNorm):
    def _setup(self, cfg: dict | None = None):
        self.norm = RMSNorm(self.weight.shape[0], eps=self.variance_epsilon)
        self.norm.weight = torch.nn.Parameter(deepcopy(self.weight.data))
        return self


@XHLLM_TRACEABLE_MODULES.register_module({LlamaModel: "LlamaModel"})
class _MiniCPM5Model(_Qwen3Model):
    pass


@XHLLM_TRACEABLE_MODULES.register_module({LlamaForCausalLM: "LlamaForCausalLM"})
class _MiniCPM5ForCausalLM(_Qwen3ForCausalLM):
    pass


def register_wrap_modules(hf_model=None):
    return None
