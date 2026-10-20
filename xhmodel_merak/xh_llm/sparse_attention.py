"""Shared selection and streaming tail state for C4 sparse attention."""

from __future__ import annotations

import torch
from torch import nn

import xhquant.nn as xhnn


class SparseAttentionTail(nn.Module):
    """Keep the last uncompressed payload rows and form causal query windows.

    State and current payloads use [B,H,T,D].  Each query sees the last three
    rows in chronological order; the shared attention mask marks the rows that
    belong to its unfinished four-token group.  The new state uses only valid
    input rows, so padded prefill and speculative verification have the same
    transition as successive decode calls.
    """

    def __init__(self, input_length: int, pool_size: int = 4) -> None:
        super().__init__()
        self.width = pool_size - 1
        self.input_length = int(input_length)
        self.state_slice = xhnn.DynamicSlice([self.width], [2], [1])

    def forward(self, current, state, current_length):
        source = torch.cat((state, current), dim=2)
        windows = torch.stack(
            tuple(source[:, :, offset : offset + self.input_length] for offset in range(1, self.width + 1)),
            dim=3,
        ).transpose(1, 2)
        return windows, self.state_slice(source, current_length)


class StaticSparseAttention(nn.Module):
    """Common SelectKV and tail assembly for MLA and GQA attention."""

    def __init__(self, candidate_topk: int, score_scale: float, input_length: int, pool_size: int = 4) -> None:
        super().__init__()
        self.select_kv = xhnn.SparseAttentionSelectKV(
            candidate_topk=candidate_topk,
            score_scale=score_scale,
            pool_size=pool_size,
            compression_ratio=pool_size,
        )
        self.tail = SparseAttentionTail(input_length, pool_size)
        self.mask_add = xhnn.MaskedAdd(saturate=True)

    def append_selected_tail(self, selected, current, state, current_length, *, has_kv_heads: bool):
        windows, next_state = self.tail(current, state, current_length)
        if not has_kv_heads:
            windows = windows.squeeze(2)
        return torch.cat((selected.to(current.dtype), windows), dim=-2), next_state
