# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Eight-layer Qwen3-VL text features, retaining official padded query states."""

from copy import deepcopy

import torch
from torch import nn


TEXT_INPUT_NAMES = ["embeddings", "positions", "query_ranges", "query_mask", "key_mask", "input_length", "kv_length"]
OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


def sdk_norm(source):
    from xhquant.nn import RMSNorm

    norm = RMSNorm(source.weight.numel(), source.variance_epsilon)
    norm.weight = nn.Parameter(source.weight.detach().float().clone())
    return norm


class TextBlock(nn.Module):
    def __init__(self, source):
        super().__init__()
        from xhquant.nn import FlashAttention, Rope

        attention = source.self_attn
        self.heads = attention.config.num_attention_heads
        self.kv_heads = attention.config.num_key_value_heads
        self.dim = attention.head_dim
        self.norm1, self.norm2 = sdk_norm(source.input_layernorm), sdk_norm(source.post_attention_layernorm)
        self.q_norm, self.k_norm = sdk_norm(attention.q_norm), sdk_norm(attention.k_norm)
        self.q, self.k, self.v, self.out = [
            deepcopy(value) for value in (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj)
        ]
        self.mlp = deepcopy(source.mlp)
        self.rope = Rope()
        self.attention = FlashAttention(
            num_heads=self.heads, num_kv_heads=self.kv_heads, is_causal=False, scale=self.dim**-0.5
        )

    def forward(self, hidden, cos, sin, ranges, query_mask, key_mask, length, kv_length):
        x = self.norm1(hidden)
        q = self.q_norm(self.q(x).reshape(1, -1, self.heads, self.dim)).transpose(1, 2)
        k = self.k_norm(self.k(x).reshape(1, -1, self.kv_heads, self.dim)).transpose(1, 2)
        v = self.v(x).reshape(1, -1, self.kv_heads, self.dim).transpose(1, 2)
        q = self.rope(q, cos, sin) * query_mask.transpose(1, 2).unsqueeze(-1)
        k = self.rope(k, cos, sin) * key_mask.transpose(1, 2).unsqueeze(-1)
        v = v * key_mask.transpose(1, 2).unsqueeze(-1)
        value = self.attention(
            q, k, v, current_input_length=length, kv_valid_length=kv_length, query_kv_range_abs=ranges
        )
        value = value.transpose(1, 2).reshape(1, -1, self.heads * self.dim)
        hidden = (hidden + self.out(value)) * query_mask
        return (hidden + self.mlp(self.norm2(hidden))) * query_mask


class TextFeatures(nn.Module):
    def __init__(self, source, *, output_layers=OUTPUT_LAYERS, max_position=4096):
        super().__init__()
        self.output_layers = tuple(output_layers)
        if not self.output_layers or max(self.output_layers) > len(source.layers):
            raise ValueError("selected feature layers exceed the text model")
        self.blocks = nn.ModuleList(TextBlock(layer) for layer in source.layers[: max(self.output_layers)])
        angles = torch.arange(max_position).float()[:, None] * source.rotary_emb.inv_freq.detach().float()[None]
        angles = torch.cat((angles, angles), -1)
        self.register_buffer("cos_table", angles.cos() * source.rotary_emb.attention_scaling)
        self.register_buffer("sin_table", angles.sin() * source.rotary_emb.attention_scaling)

    def forward(self, embeddings, positions, query_ranges, query_mask, key_mask, input_length, kv_length):
        cos, sin = self.cos_table[positions.long()][:, None], self.sin_table[positions.long()][:, None]
        hidden, outputs = embeddings * query_mask, []
        for index, block in enumerate(self.blocks, 1):
            hidden = block(hidden, cos, sin, query_ranges, query_mask, key_mask, input_length, kv_length)
            if index in self.output_layers:
                outputs.append(hidden)
        return torch.cat(outputs, dim=-1)


def feature_inputs(embeddings, positions, real_length, semantic_length):
    capacity = embeddings.shape[1]
    slots = torch.arange(capacity, device=embeddings.device)
    ends = (slots + 1).clamp(max=real_length)
    ranges = torch.stack((torch.zeros_like(ends), ends), -1)[None].int()
    query_mask = (slots < semantic_length).to(embeddings.dtype)[None, :, None]
    key_mask = (slots < real_length).to(embeddings.dtype)[None, :, None]
    length = torch.tensor([semantic_length], dtype=torch.int32, device=embeddings.device)
    kv_length = torch.tensor([real_length], dtype=torch.int32, device=embeddings.device)
    return embeddings, positions, ranges, query_mask, key_mask, length, kv_length
