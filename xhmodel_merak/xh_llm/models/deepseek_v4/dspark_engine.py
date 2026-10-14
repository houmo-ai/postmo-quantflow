# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Single-request DSpark decoding with target verification and committed features."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import torch
from torch import Tensor

from .dspark import DSparkSpec
from .dspark_runtime import DSparkSlidingCache, dspark_attention_mask, sample_dspark
from .dspark_verification import DSparkTargetTransaction, verify_dspark


@dataclass
class DSparkStatistics:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    replayed_target_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0


class DSparkEngine:
    """Run the four draft graphs against a multi-token target callback.

    ``target_forward(ids,past,prefill)`` returns all target verification logits
    and the three selected target features, and updates ``target_cache`` in
    place. Each draft graph slices its own absolute-position RoPE table.
    Rejected target suffixes are restored, then only the accepted prefix is
    replayed and committed to the draft context. This first implementation
    keeps request concurrency at one and starts fresh caches per generation.
    """

    def __init__(self, spec: DSparkSpec, *, target_forward: Callable, target_cache,
                 embedding: Callable, context: Callable,
                 context_decode: Callable, draft: Callable, markov: Callable,
                 device="cpu"):
        self.spec = spec
        self.target_forward = target_forward
        self.target_cache = target_cache
        self.embedding = embedding
        self.context = context
        self.context_decode = context_decode
        self.draft = draft
        self.markov = markov
        self.device = torch.device(device)
        self.cache = DSparkSlidingCache(spec, device=self.device)
        self.statistics = DSparkStatistics()

    @torch.no_grad()
    def generate(self, input_ids: Tensor, *, max_new_tokens: int, temperature: float = 1.0,
                 eos_token_ids: tuple[int, ...] = (), generator: torch.Generator | None = None) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("DSpark generation requires one nonempty prompt")
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("Invalid DSpark generation budget or temperature")
        ids = input_ids.to(self.device)
        if max_new_tokens == 0:
            return ids
        self.target_cache.cache_inputs()
        self.target_cache.reset_kv_cache()
        self.cache.reset()
        self.statistics = DSparkStatistics()
        past = 0
        logits = None
        chunk_length = self.spec.context_chunk_length
        for chunk in ids.split(chunk_length, dim=1):
            logits, features = self.target_forward(chunk, past, True)
            self.cache.commit_context(self.context, features.to(self.device),
                                       current_length=chunk.shape[1])
            past += chunk.shape[1]
        last = (logits[:, 0] if logits.shape[1] == 1 else logits[:, chunk.shape[1] - 1]).to(self.device)
        anchor = (last.argmax(-1, keepdim=True) if temperature == 0
                  else torch.multinomial(torch.softmax(last.float() / temperature, -1), 1, generator=generator))
        anchor = anchor.to(self.device)
        emitted = [anchor]
        remaining = max_new_tokens - 1
        while remaining and int(anchor.item()) not in eos_token_ids:
            query_ids = torch.full((1, self.spec.block_size), self.spec.noise_token_id,
                                    dtype=torch.int32, device=self.device)
            query_ids[:, :1] = anchor
            self.cache.begin_proposal()
            try:
                outputs = self.draft(self.embedding(query_ids),
                                      torch.tensor([past], dtype=torch.int32, device=self.device),
                                      torch.tensor([self.spec.block_size], dtype=torch.int32, device=self.device),
                                      dspark_attention_mask(self.spec, past, device=self.device),
                                      *self.cache.caches)
                outputs = tuple(outputs.values()) if isinstance(outputs, dict) else outputs
                candidates, q, _ = sample_dspark(*outputs, anchor, self.markov,
                                                 temperature=temperature, generator=generator)
            finally:
                self.cache.discard_proposal()
            candidate_count = min(self.spec.block_size, remaining)
            candidates, q = candidates[:, :candidate_count], q[:, :candidate_count]
            verify_ids = torch.cat((anchor, candidates), dim=1)
            transaction = DSparkTargetTransaction(self.target_cache, past_length=past,
                                                   input_length=verify_ids.shape[1])
            try:
                target_logits, features = self.target_forward(verify_ids, past, False)
                result = verify_dspark(candidates, q, target_logits[:, :candidate_count + 1].to(self.device),
                                        temperature=temperature, generator=generator)
            except BaseException:
                transaction.rollback()
                raise
            committed = 1 + result.accepted_count
            if result.all_accepted:
                transaction.commit()
            else:
                transaction.rollback()
                _, features = self.target_forward(verify_ids[:, :committed], past, False)
                self.statistics.replayed_target_tokens += committed
            self.cache.commit_context(self.context_decode, features.to(self.device),
                                       current_length=committed)
            past += committed
            self.statistics.rounds += 1
            self.statistics.proposed += candidate_count
            self.statistics.accepted += result.accepted_count
            block = result.emitted_tokens[:, :remaining]
            for i, token in enumerate(block[0]):
                if int(token) in eos_token_ids:
                    block = block[:, :i + 1]
                    break
            emitted.append(block)
            remaining -= block.shape[1]
            anchor = block[:, -1:]
        return torch.cat((ids, *emitted), dim=1)


__all__ = ["DSparkEngine", "DSparkStatistics"]
