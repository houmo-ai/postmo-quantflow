# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""DSpark host sampling and ownership of the draft's sliding latent cache."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from xhquant.core import HybridCacheTensor

from .dspark import DSparkSpec
from .swa_layout import aligned_swa_attention_length


def dspark_attention_mask(spec: DSparkSpec, past_length: int, *, device=None,
                          dtype=torch.float16) -> Tensor:
    """Allow the complete query block and its position-dependent past window."""
    retained = min(int(past_length), spec.sliding_window - 1)
    width = aligned_swa_attention_length(spec.block_size, spec.sliding_window)
    keys = torch.arange(width, device=device).reshape(1, -1)
    query = torch.arange(spec.block_size, device=device).reshape(-1, 1)
    left = torch.clamp(retained + query - spec.sliding_window + 1, min=0)
    visible = (keys >= left) & (keys < retained + spec.block_size)
    return torch.where(visible, 0.0, -65504.0).to(dtype).reshape(1, 1, spec.block_size, width)


class DSparkSlidingCache:
    """Keep verified target features separate from transient draft-query KV."""

    def __init__(self, spec: DSparkSpec, *, device="cpu", dtype=torch.float16):
        self.spec = spec
        self.caches = tuple(HybridCacheTensor(torch.zeros(
            1, 1, spec.cache_backing_length, spec.head_dim, device=device, dtype=dtype
        )) for _ in range(spec.num_draft_layers))
        self.past_length = 0
        self._snapshot = None

    def begin_proposal(self):
        if self._snapshot is not None:
            raise RuntimeError("A DSpark proposal is already pending")
        self._snapshot = tuple((cache.data.clone(), cache.cache_valid_len) for cache in self.caches)

    def discard_proposal(self):
        if self._snapshot is None:
            raise RuntimeError("There is no pending DSpark proposal")
        for cache, (data, valid) in zip(self.caches, self._snapshot, strict=True):
            cache.data.copy_(data)
            cache.cache_valid_len = valid
        self._snapshot = None

    def commit_context(self, graph: Callable, target_hidden: Tensor,
                       *, current_length: int):
        if self._snapshot is not None:
            raise RuntimeError("Discard transient DSpark KV before committing target context")
        if not 0 < current_length <= target_hidden.shape[1]:
            raise ValueError("Invalid DSpark context prefix length")
        device = target_hidden.device
        graph(target_hidden, torch.tensor([self.past_length], dtype=torch.int32, device=device),
              torch.tensor([current_length], dtype=torch.int32, device=device), *self.caches)
        self.past_length += current_length

    def reset(self):
        for cache in self.caches:
            cache.zero_()
            cache.cache_valid_len = 0
        self.past_length = 0
        self._snapshot = None


def sample_dspark(base_logits: Tensor, head_hidden: Tensor, anchor: Tensor,
                   markov: Callable, *, temperature: float = 1.0,
                   generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, Tensor | None]:
    """Sample causal draft probabilities after applying each Markov correction.

    The returned probabilities are the distributions actually used to propose
    the tokens, and must be supplied to rejection sampling instead of the
    uncorrected parallel backbone logits.
    """
    if temperature < 0:
        raise ValueError("DSpark temperature cannot be negative")
    if base_logits.ndim != 3 or head_hidden.shape[:2] != base_logits.shape[:2]:
        raise ValueError("DSpark logits/hidden states have inconsistent block shapes")
    tokens, probabilities, confidences = [], [], []
    previous = anchor.to(device=base_logits.device, dtype=torch.int32).reshape(base_logits.shape[0], 1)
    for step in range(base_logits.shape[1]):
        outputs = markov(previous, head_hidden[:, step:step + 1])
        corrected = base_logits[:, step].float() + outputs[0][:, 0].float()
        if temperature == 0:
            token = corrected.argmax(-1, keepdim=True)
            probability = torch.zeros_like(corrected).scatter_(-1, token, 1)
        else:
            probability = torch.softmax(corrected / temperature, dim=-1)
            token = torch.multinomial(probability, 1, generator=generator)
        tokens.append(token)
        probabilities.append(probability)
        if len(outputs) > 1:
            confidences.append(outputs[1])
        previous = token.to(dtype=torch.int32)
    confidence = torch.cat(confidences, dim=1) if confidences else None
    return torch.cat(tokens, dim=1), torch.stack(probabilities, dim=1), confidence


__all__ = ["DSparkSlidingCache", "dspark_attention_mask", "sample_dspark"]
