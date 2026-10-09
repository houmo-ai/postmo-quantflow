# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Full-sequence Qwen3-VL feature graph and fixed-capacity session boundary."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


FEATURE_INPUT_NAMES = [
    "embeddings",
    "position_t",
    "position_h",
    "position_w",
    "past_length",
    "input_length",
    "deepstack_0",
    "deepstack_1",
    "deepstack_2",
]

CAPACITY_FEATURE_INPUT_NAMES = [
    "embeddings",
    "position_t",
    "position_h",
    "position_w",
    "input_length",
    "deepstack_0",
    "deepstack_1",
    "deepstack_2",
]


def load_feature_encoder(model_dir):
    """Load the official CPU encoder before adapting its weights to capacity graphs."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    # The native initializer reads this model's rope_parameters independently
    # of legacy default RoPE registrations installed by another model family.
    return Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32, device_map="cpu", local_files_only=True
    ).eval()


def _sdk_norm(source):
    from xhquant.nn import RMSNorm

    norm = RMSNorm(source.weight.numel(), source.variance_epsilon)
    norm.weight = nn.Parameter(source.weight.detach().clone())
    return norm


class CapacityTextBlock(nn.Module):
    def __init__(self, source):
        super().__init__()
        from xhquant.nn import FlashAttention, Rope

        attention = source.self_attn
        self.heads = attention.config.num_attention_heads
        self.kv_heads = attention.config.num_key_value_heads
        self.head_dim = attention.head_dim
        if self.head_dim not in (64, 128, 256, 512):
            raise ValueError("unsupported text FlashAttention head width")
        self.input_norm, self.post_norm = _sdk_norm(source.input_layernorm), _sdk_norm(source.post_attention_layernorm)
        self.q, self.k, self.v, self.out = (
            deepcopy(value) for value in (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj)
        )
        self.q_norm, self.k_norm = _sdk_norm(attention.q_norm), _sdk_norm(attention.k_norm)
        self.rotary = Rope()
        self.attention = FlashAttention(
            num_heads=self.heads, num_kv_heads=self.kv_heads, is_causal=True, scale=self.head_dim**-0.5
        )
        self.mlp = deepcopy(source.mlp)

    def forward(self, hidden, cos, sin, visible, length):
        batch, capacity, _ = hidden.shape
        x = self.input_norm(hidden)
        q = self.q_norm(self.q(x).reshape(batch, capacity, self.heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k(x).reshape(batch, capacity, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v(x).reshape(batch, capacity, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q, cos, sin), self.rotary(k, cos, sin)
        mask = visible.transpose(1, 2).unsqueeze(-1)
        out = self.attention(q * mask, k * mask, v * mask, current_input_length=length)
        out = out.transpose(1, 2).reshape(batch, capacity, self.heads * self.head_dim)
        hidden = (hidden + self.out(out)) * visible
        return (hidden + self.mlp(self.post_norm(hidden))) * visible


class CapacityTextFeatures(nn.Module):
    """Complete causal Qwen3-VL sequence with SDK norms/RoPE/FlashAttention."""

    def __init__(self, source, capacity, max_position):
        super().__init__()
        rotary = source.rotary_emb
        if rotary.rope_type != "default":
            raise ValueError("Qwen-Image feature export requires static default M-RoPE")
        frequency = torch.arange(max_position).float()[:, None] * rotary.inv_freq.float()[None]
        angle = torch.cat((frequency, frequency), dim=-1)
        self.register_buffer("cos_table", angle.cos() * rotary.attention_scaling)
        self.register_buffer("sin_table", angle.sin() * rotary.attention_scaling)
        axes = torch.zeros(frequency.shape[-1], dtype=torch.int64)
        for axis, offset in ((1, 1), (2, 2)):
            axes[offset : rotary.mrope_section[axis] * 3 : 3] = axis
        axes = torch.cat((axes, axes))
        for axis in range(3):
            self.register_buffer(f"axis_{axis}", (axes == axis).float())
        self.register_buffer("token_slots", torch.arange(capacity, dtype=torch.int32))
        self.blocks = nn.ModuleList(CapacityTextBlock(block) for block in source.layers)

    def forward(
        self, embeddings, position_t, position_h, position_w, input_length, deepstack_0, deepstack_1, deepstack_2
    ):
        cosine, sine = [], []
        for axis, positions in enumerate((position_t, position_h, position_w)):
            cosine.append(self.cos_table[positions.long()] * getattr(self, f"axis_{axis}"))
            sine.append(self.sin_table[positions.long()] * getattr(self, f"axis_{axis}"))
        cos = (cosine[0] + cosine[1] + cosine[2])[None, None]
        sin = (sine[0] + sine[1] + sine[2])[None, None]
        visible = (self.token_slots < input_length.reshape(1)).to(embeddings.dtype)[None, :, None]
        hidden = embeddings * visible
        deepstack = (deepstack_0, deepstack_1, deepstack_2)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, cos, sin, visible, input_length)
            if index < 3:
                hidden = (hidden + deepstack[index]) * visible
        # The official pipeline selects the hidden state before final RMSNorm.
        return hidden


class FullSequenceFeatures(nn.Module):
    """Adapt the existing embedding wrapper with cache disabled and pre-norm output.

    Construct the wrapped model with ``output_pre_norm_hidden_states=True``,
    ``num_logits_to_keep=0`` and ``use_cache=False``. SDK conversion must enter
    ``module_graph_forward_guard`` as for the existing Qwen3-VL wrapper.
    """

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(
        self,
        embeddings,
        position_t,
        position_h,
        position_w,
        past_length,
        input_length,
        deepstack_0,
        deepstack_1,
        deepstack_2,
    ):
        return self.wrapped(
            embeddings,
            position_t,
            position_h,
            position_w,
            past_length,
            input_length,
            deepstack_0,
            deepstack_1,
            deepstack_2,
            None,
            None,
        )


class FeatureSession:
    """Run one complete causal sequence; right padding cannot affect valid tokens.

    Inputs already contain vision embeddings and three scattered DeepStack
    features. Tokenization, vision execution and system-prefix removal belong
    to the processor. This interface performs neither pooling nor normalization.
    """

    def __init__(self, session, capacity, hidden_size, max_position, device="cuda:0"):
        for name, value in (("capacity", capacity), ("hidden_size", hidden_size), ("max_position", max_position)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if capacity > max_position:
            raise ValueError("feature capacity exceeds the exported RoPE table")
        self.session = session
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.max_position = max_position
        self.device = torch.device(device)

    def prepare_inputs(self, embeddings, positions, deepstack=None):
        if embeddings.ndim != 3 or embeddings.shape[0] != 1 or embeddings.shape[2] != self.hidden_size:
            raise ValueError("embeddings must have shape [1, tokens, hidden_size]")
        length = embeddings.shape[1]
        if not 0 < length <= self.capacity:
            raise ValueError("sequence length exceeds the feature gear or is empty")
        if positions.shape != (3, length) or positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must be integer [3, tokens]")
        if bool((positions < 0).any()) or bool((positions >= self.max_position).any()):
            raise ValueError("position exceeds the exported RoPE table")
        if deepstack is None:
            deepstack = [torch.zeros_like(embeddings) for _ in range(3)]
        if len(deepstack) != 3 or any(x.shape != embeddings.shape for x in deepstack):
            raise ValueError("three DeepStack tensors must match the embedding shape")
        padding = self.capacity - length

        def padded(x):
            return F.pad(x.to(device=self.device, dtype=torch.float16), (0, 0, 0, padding))

        # Use the same deterministic padding positions as export calibration.
        # SEFP block exponents can couple padded and valid activations even
        # though the floating-point causal attention excludes future keys.
        position_inputs = torch.arange(self.capacity, device=self.device, dtype=torch.int32).expand(3, -1).clone()
        position_inputs[:, :length] = positions.to(device=self.device, dtype=torch.int32)
        # Keep graph output shape fixed; the caller receives only valid tokens.
        return (
            padded(embeddings),
            *position_inputs.unbind(0),
            torch.tensor([0], device=self.device, dtype=torch.int32),
            torch.tensor([self.capacity], device=self.device, dtype=torch.int32),
            *(padded(x) for x in deepstack),
        )

    def __call__(self, embeddings, positions, deepstack=None):
        inputs = self.prepare_inputs(embeddings, positions, deepstack)
        output = self.session(*inputs)
        if output.shape != (1, self.capacity, self.hidden_size):
            raise ValueError("feature graph returned an incompatible output shape")
        return output[:, : embeddings.shape[1]].clone()


class CapacityFeatureSession(FeatureSession):
    def prepare_inputs(self, embeddings, positions, deepstack=None):
        values = super().prepare_inputs(embeddings, positions, deepstack)
        length = torch.tensor([embeddings.shape[1]], device=self.device, dtype=torch.int32)
        return (*values[:4], length, *values[6:])
