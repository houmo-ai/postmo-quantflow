# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Non-causal attention with explicit effective-prefix masks."""

import torch
from torch import nn


class LengthMaskedAttention(nn.Module):
    def __init__(self, query_capacity, kv_capacity, *, scale, use_flash_attention=False, num_heads=1):
        super().__init__()
        from xhquant.nn import Cast, Greater, MaskedAdd, MatMul

        self.scale = scale
        if type(use_flash_attention) is not bool:
            raise ValueError("use_flash_attention must be a bool")
        self.use_flash_attention = use_flash_attention
        self.register_buffer("query_positions", torch.arange(query_capacity, dtype=torch.int32))
        self.register_buffer("kv_positions", torch.arange(kv_capacity, dtype=torch.int32))
        self.valid = Greater()
        self.mask_cast = Cast(torch.float32)
        if use_flash_attention:
            from xhquant.nn import FlashAttention

            self.attention = FlashAttention(num_heads=num_heads, is_causal=False, scale=scale)
        else:
            self.qk, self.pv = MatMul(), MatMul()
            self.masked_add = MaskedAdd()
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, query_length, kv_length):
        valid_q = self.mask_cast(self.valid(query_length, self.query_positions))[None, None, :, None]
        valid_kv = self.mask_cast(self.valid(kv_length, self.kv_positions))[None, None, :, None]
        # Clear unused cache slots before SEFP quantization as well as Softmax.
        q, k, v = q * valid_q, k * valid_kv, v * valid_kv
        if self.use_flash_attention:
            output = self.attention(q, k, v, current_input_length=query_length, kv_valid_length=kv_length)
        else:
            mask = (1 - valid_kv.transpose(-1, -2)) * -65504.0
            # Match native FA: scale Q before SEFP quantization/QK FP16 writeback.
            scores = self.qk(q * self.scale, k.transpose(-1, -2))
            probabilities = self.softmax(self.masked_add(scores, mask))
            output = self.pv(probabilities, v)
        return output * valid_q
