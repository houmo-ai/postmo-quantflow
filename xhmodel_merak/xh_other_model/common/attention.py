# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Static attention graphs respecting the compiler's minimum FlashAttention KV."""

import torch
from torch import nn


MIN_FLASH_KV_LENGTH = 512


class CompilerAttention(nn.Module):
    """Keep SDK FlashAttention for large KV; express short KV using primitives.

    Shape discovery must call ``set_static_input_shapes`` before tracing. Fixed
    capacity callers may instead provide both capacities at construction.
    """

    def __init__(self, *, num_heads, scale, is_causal=False, num_kv_heads=None,
                 query_capacity=None, kv_capacity=None):
        super().__init__()
        from xhquant.nn import And, Equal, FlashAttention, Less, LessOrEqual, MaskedAdd, MatMul, Where

        self.scale, self.is_causal = scale, is_causal
        self.repeats = num_heads // (num_kv_heads or num_heads)
        self.flash = FlashAttention(num_heads=num_heads, num_kv_heads=num_kv_heads,
                                    scale=scale, is_causal=is_causal)
        self.qk, self.pv, self.masked_add = MatMul(), MatMul(), MaskedAdd()
        self.softmax = nn.Softmax(dim=-1)
        self.less, self.less_equal = Less(), LessOrEqual()
        self.mask_where, self.probability_where = Where(), Where()
        self.equal, self.and_op = Equal(), And()
        self.register_buffer('zero', torch.tensor(0.0))
        self.register_buffer('negative', torch.tensor(-65504.0))
        self.register_buffer('query_positions', torch.empty(0, dtype=torch.int32))
        self.register_buffer('key_positions', torch.empty(0, dtype=torch.int32))
        self.use_flash = True
        self.kv_capacity = None
        if query_capacity is not None and kv_capacity is not None:
            self._set_capacities(query_capacity, kv_capacity)

    def _set_capacities(self, query_capacity, kv_capacity):
        self.kv_capacity = kv_capacity
        self.use_flash = kv_capacity >= MIN_FLASH_KV_LENGTH
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with unset_fake_temporarily():
            self.key_positions = torch.arange(kv_capacity, dtype=torch.int32).reshape(1, 1, 1, -1)
            if not self.use_flash:
                self.query_positions = torch.arange(query_capacity, dtype=torch.int32).reshape(1, 1, -1, 1)

    def set_static_input_shapes(self, shapes):
        self._set_capacities(shapes[0][-2], shapes[1][-2])

    def forward(self, q, k, v, *, past_seq_length=None, current_input_length=None,
                kv_valid_length=None, key_mask=None):
        if self.use_flash:
            if not self.is_causal and self.kv_capacity is not None:
                if key_mask is None and (kv_valid_length is not None or current_input_length is not None):
                    length = kv_valid_length if kv_valid_length is not None else current_input_length
                    key_mask = self.mask_where(self.less(self.key_positions, length.reshape(-1, 1, 1, 1)),
                                          self.zero, self.negative)
                return self.flash(q, k, v, key_mask=key_mask)
            return self.flash(q, k, v, past_seq_length=past_seq_length,
                              current_input_length=current_input_length,
                              kv_valid_length=kv_valid_length, key_mask=key_mask)
        mask = key_mask
        if self.is_causal:
            past = self.zero if past_seq_length is None else past_seq_length.reshape(-1, 1, 1, 1)
            visible = self.less_equal(self.key_positions, self.query_positions + past)
            if current_input_length is not None:
                visible = self.and_op(
                    visible, self.less(self.key_positions, past + current_input_length.reshape(-1, 1, 1, 1))
                )
            mask = self.mask_where(visible, self.zero, self.negative)
        elif mask is None and (kv_valid_length is not None or current_input_length is not None):
            length = kv_valid_length if kv_valid_length is not None else current_input_length
            mask = self.mask_where(self.less(self.key_positions, length.reshape(-1, 1, 1, 1)), self.zero, self.negative)
        if self.repeats != 1:
            k, v = (x.repeat_interleave(self.repeats, dim=1) for x in (k, v))
        scores = self.qk(q * self.scale, k.transpose(-1, -2))
        probabilities = self.softmax(self.masked_add(scores, self.zero if mask is None else mask))
        # Finite mask sentinels produce a uniform row when every key is masked.
        # Zero those probabilities so the empty-key result remains exactly zero.
        if mask is not None:
            probabilities = self.probability_where(self.equal(mask, self.zero), probabilities, self.zero)
        return self.pv(probabilities, v)
