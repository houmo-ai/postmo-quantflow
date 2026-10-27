# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Decode-only next-layer expert predictions; actual routing remains authoritative."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager

import torch
from torch import nn
from torch.nn import functional as F

LES_REFERENCE = (
    "https://github.com/tc-mb/llama.cpp-omni/blob/"
    "a64932ec615e1463158129feaf2f0752d0c07a2e/src/models/qwen35moe.cpp#L455-L465"
)


def validate_prediction_k(value: int, num_experts: int) -> int:
    if type(value) is not int or not 1 <= value <= num_experts:
        raise ValueError(f"LES prediction_k must be an integer in [1, {num_experts}]")
    return value


def expert_ids(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Rank logits directly, as in the reference prediction branch."""
    return logits.topk(k, dim=-1).indices


def prediction_counts(predicted: torch.Tensor, actual: torch.Tensor) -> dict[str, int]:
    """Count set overlap, complete coverage and exact Top-K set equality."""
    if predicted.shape[:-1] != actual.shape[:-1]:
        raise ValueError("LES predicted and actual token axes must match")
    hits = (actual.unsqueeze(-1) == predicted.unsqueeze(-2)).any(-1)
    count = hits.sum(-1)
    return {
        "tokens": count.numel(),
        "expert_hits": int(count.sum().item()),
        "actual_experts": hits.numel(),
        "predicted_experts": predicted.numel(),
        "complete_hits": int(hits.all(-1).sum().item()),
        "exact_hits": int(hits.all(-1).sum().item()) if predicted.shape[-1] == actual.shape[-1] else 0,
    }


def prediction_rates(counts: dict[str, int]) -> dict[str, float]:
    def ratio(numerator, denominator):
        return counts[numerator] / counts[denominator] if counts[denominator] else 0.0

    return {
        "expert_recall": ratio("expert_hits", "actual_experts"),
        "prediction_precision": ratio("expert_hits", "predicted_experts"),
        "complete_topk_coverage": ratio("complete_hits", "tokens"),
        "exact_topk_match": ratio("exact_hits", "tokens"),
    }


class LESObserver(AbstractContextManager):
    """Native MiniCPM observer and host scheduling boundary.

    ``prefetch(layer, ids)`` runs before the current layer's MoE. ``require``
    receives the next layer's *actual* IDs before its experts execute, so a
    backend can wait for hits and load misses. Neither callback may replace
    routing IDs/weights. Callbacks receive device tensors and own I/O policy.
    Predictions/records cover single-token decode only, never padded prefill
    or multi-token speculative verification. The context removes every hook.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        prediction_k: int = 8,
        *,
        prefetch: Callable | None = None,
        require: Callable | None = None,
    ):
        self.layers = list(layers)
        if len(self.layers) < 2 or any(not hasattr(layer.mlp, "gate") for layer in self.layers):
            raise ValueError("LES requires at least two MoE body layers")
        for layer in self.layers:
            gate = layer.mlp.gate
            if not hasattr(gate, "top_k"):
                raise ValueError("LESObserver requires native MiniCPM MoE routers")
            validate_prediction_k(prediction_k, gate.weight.shape[0])
        self.prediction_k = prediction_k
        self.prefetch = prefetch
        self.require = require
        self._handles = []
        self.reset()

    def reset(self):
        self.pending = {}
        self.records = []

    def __enter__(self):
        if self._handles:
            raise RuntimeError("LESObserver is already installed")
        for index, layer in enumerate(self.layers):
            self._handles.append(layer.mlp.gate.register_forward_pre_hook(self._predict(index)))
            self._handles.append(layer.mlp.gate.register_forward_hook(self._observe(index)))
        return self

    def _predict(self, index):
        @torch.no_grad()
        def hook(module, args):
            hidden = args[0]
            if index == 0:
                self.pending.clear()
            if hidden.numel() // hidden.shape[-1] != 1 or index + 1 == len(self.layers):
                return
            next_gate = self.layers[index + 1].mlp.gate
            logits = F.linear(hidden, next_gate.weight)
            predicted = expert_ids(logits, self.prediction_k).reshape(1, -1)
            self.pending[index + 1] = predicted
            if self.prefetch is not None:
                self.prefetch(index + 1, predicted.clone())

        return hook

    def _observe(self, index):
        def hook(module, args, output):
            if args[0].numel() // args[0].shape[-1] != 1:
                return
            # Native routers return (logits, normalized weights, actual IDs).
            actual = output[2].detach().reshape(1, -1)
            predicted = self.pending.pop(index, None)
            if predicted is not None:
                self.records.append((index, predicted, actual))
            if self.require is not None:
                self.require(index, actual.clone())

        return hook

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.pending.clear()
        return False
