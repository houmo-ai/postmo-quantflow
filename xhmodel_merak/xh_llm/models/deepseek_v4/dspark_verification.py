# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""C1 DSpark verification and bounded rollback of heterogeneous target caches."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .cache_abi import CSA, SLIDING


@dataclass(frozen=True)
class DSparkVerification:
    emitted_tokens: Tensor
    accepted_count: int
    all_accepted: bool


def verify_dspark(proposals: Tensor, proposal_probabilities: Tensor, target_logits: Tensor,
                   *, temperature: float = 1.0,
                   generator: torch.Generator | None = None) -> DSparkVerification:
    """Reject using p/q and sample from max(p-q,0) on the first rejection."""
    if proposals.ndim != 2 or proposals.shape[0] != 1:
        raise ValueError("DSpark accuracy verification currently requires one request")
    k = proposals.shape[1]
    if proposal_probabilities.shape[:2] != (1, k) or target_logits.shape != (
        1, k + 1, proposal_probabilities.shape[-1]
    ):
        raise ValueError("DSpark verification requires K proposals and K+1 target logits")
    if temperature < 0:
        raise ValueError("DSpark temperature cannot be negative")
    if not torch.isfinite(target_logits).all() or not torch.isfinite(proposal_probabilities).all():
        raise ValueError("DSpark verification probabilities/logits must be finite")
    if (proposal_probabilities < 0).any() or not torch.allclose(
        proposal_probabilities.sum(-1), torch.ones_like(proposal_probabilities[..., 0]), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("DSpark proposal probabilities must be normalized")
    logits = target_logits.float()
    probabilities = torch.softmax(logits / temperature, -1) if temperature else None
    for i in range(k):
        token = int(proposals[0, i])
        if temperature == 0:
            accepted = token == int(logits[0, i].argmax())
        else:
            q_token = proposal_probabilities[0, i, token]
            if q_token <= 0:
                raise ValueError("DSpark proposed a token with zero proposal probability")
            threshold = torch.minimum(probabilities[0, i, token] / q_token, q_token.new_ones(()))
            accepted = bool(torch.rand((), device=threshold.device, generator=generator) < threshold)
        if accepted:
            continue
        if temperature == 0:
            replacement = logits[:, i].argmax(-1, keepdim=True)
        else:
            residual = (probabilities[:, i] - proposal_probabilities[:, i]).clamp_min(0)
            mass = residual.sum(-1, keepdim=True)
            residual = torch.where(mass > 0, residual / mass.clamp_min(torch.finfo(residual.dtype).tiny),
                                    probabilities[:, i])
            replacement = torch.multinomial(residual, 1, generator=generator)
        return DSparkVerification(torch.cat((proposals[:, :i], replacement), dim=1), i, False)
    bonus = (logits[:, k].argmax(-1, keepdim=True) if temperature == 0
             else torch.multinomial(probabilities[:, k], 1, generator=generator))
    return DSparkVerification(torch.cat((proposals, bonus), dim=1), k, True)


class DSparkTargetTransaction:
    """Snapshot SWA/carry state and only compressed rows a verifier can overwrite.

    CSA's full 256K-context backing can be many gigabytes. A K+1-token verify
    can append only ceil((K+1)/ratio)+1 rows, so rollback never copies the
    full compressed cache. Tensor addresses remain stable for CUDA graphs.
    """

    def __init__(self, cache_mixin, *, past_length: int, input_length: int):
        if past_length < 0 or input_length <= 0:
            raise ValueError("Invalid DSpark target verification span")
        self._snapshots = []
        self._pending = True
        for kind, cache in zip(cache_mixin.abi.layer_types, cache_mixin.cache_inputs(), strict=True):
            ratio = 4 if kind == CSA else 128
            for name, tensor in zip(cache._fields, cache, strict=True):
                if kind == SLIDING or name == "swa_kv" or name.endswith("_state"):
                    selected = tensor.data
                else:
                    start = past_length // ratio
                    stop = min(tensor.shape[2], math.ceil((past_length + input_length) / ratio) + 1)
                    selected = tensor.data[:, :, start:stop]
                valid = getattr(tensor, "cache_valid_len", None)
                self._snapshots.append((selected, selected.clone(), tensor, valid))

    @property
    def snapshot_nbytes(self) -> int:
        return sum(before.numel() * before.element_size() for _, before, _, _ in self._snapshots)

    def rollback(self):
        if not self._pending:
            raise RuntimeError("DSpark target transaction is already closed")
        for selected, before, tensor, valid in self._snapshots:
            selected.copy_(before)
            if valid is not None:
                tensor.cache_valid_len = valid
        self._pending = False
        self._snapshots.clear()

    def commit(self):
        if not self._pending:
            raise RuntimeError("DSpark target transaction is already closed")
        self._pending = False
        self._snapshots.clear()


__all__ = ["DSparkVerification", "verify_dspark", "DSparkTargetTransaction"]
