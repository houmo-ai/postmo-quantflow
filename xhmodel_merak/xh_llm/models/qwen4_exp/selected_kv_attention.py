"""QSA block selection using the XH2 SparseAttentionSelectKV operator.

This path consumes contiguous causal prefixes produced by the text export
wrapper. Pooling, indexer normalization and RoPE remain outside the fused op.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from xhquant.ops.xh.xh_pragram_op import xh_pragma_fx

from ...sparse_attention import StaticSparseAttention


@dataclass(frozen=True)
class Qwen4ExpQSAConfig:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    scale: float | None = None


class Qwen4ExpSelectedKVAttention(StaticSparseAttention):
    def __init__(
        self,
        config: Qwen4ExpQSAConfig,
        *,
        candidate_topk: int,
        index_head_dim: int,
        input_length: int,
        flatten_output: bool = False,
    ):
        import xhquant.nn as xhnn

        super().__init__(candidate_topk, 1.0 / math.sqrt(index_head_dim), input_length)
        self.config = config
        self.candidate_topk = int(candidate_topk)
        self.flatten_output = flatten_output
        if config.num_heads % config.num_kv_heads:
            raise ValueError("QSA query heads must be divisible by K/V heads")
        self.softmax = xhnn.Softmax(dim=-1)

    def forward(
        self, query, index_query, block_keys, key, value, current_key, current_value,
        key_tail, value_tail, attention_mask, past_seq_length, current_input_length,
    ):
        batch, heads, queries, dim = 1, self.config.num_heads, self.tail.input_length, self.config.head_dim
        selected_key, selected_value = self.select_kv(
            index_query.to(torch.float16),
            block_keys.to(torch.float16),
            key.to(torch.float16),
            past_seq_length.to(torch.int32),
            current_length=current_input_length.to(torch.int32),
            value_cache=value.to(torch.float16),
        )
        packed_key, next_key_tail = self.append_selected_tail(
            selected_key, current_key, key_tail, current_input_length, has_kv_heads=True,
        )
        packed_value, next_value_tail = self.append_selected_tail(
            selected_value, current_value, value_tail, current_input_length, has_kv_heads=True,
        )
        width = self.candidate_topk * 4 + 3
        if self.flatten_output:
            packed_query = query.transpose(1, 2)
            if queries != 1 or self.config.num_kv_heads != 1:
                packed_query = packed_query.reshape(
                    batch * queries, self.config.num_kv_heads, heads // self.config.num_kv_heads, dim,
                )
            attention = self._attend_grouped
        else:
            packed_query = query.transpose(1, 2).reshape(batch * queries, heads, 1, dim)
            attention = self.attend_packed
        packed_mask = attention_mask
        if queries != 1:
            packed_mask = packed_mask.reshape(batch * queries, 1, 1, width)
        output = attention(
            packed_query,
            packed_key.reshape(batch * queries, self.config.num_kv_heads, width, dim),
            packed_value.reshape(batch * queries, self.config.num_kv_heads, width, dim),
            packed_mask,
        )
        if self.flatten_output:
            output = output.reshape(batch, queries, -1)
        else:
            output = output.reshape(batch, queries, heads, dim).transpose(1, 2)
        return output, next_key_tail, next_value_tail

    def attend_packed(self, query, key, value, attention_mask):
        """GQA over fixed selected/tail segments without K/V replication."""
        batch, heads, dim = self.tail.input_length, self.config.num_heads, self.config.head_dim
        grouped_query = query.reshape(batch, self.config.num_kv_heads, heads // self.config.num_kv_heads, dim)
        return self._attend_grouped(grouped_query, key, value, attention_mask).reshape_as(query)

    def _attend_grouped(self, query, key, value, attention_mask):
        # Keep the former fused operator's 16-bit attention boundary even
        # when surrounding projections use an 8-bit activation scheme.
        xh_pragma_fx(
            query, {"type": "quanted", "action": "start", "qconfig": {"quant_type": "w16a16h1_sefp"}},
        )
        dim = self.config.head_dim
        scale = self.config.scale if self.config.scale is not None else dim**-0.5
        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        scores = self.mask_add(scores, attention_mask.to(query.dtype))
        output = torch.matmul(self.softmax(scores), value)
        xh_pragma_fx(output, {"type": "quanted", "action": "end"})
        return output
