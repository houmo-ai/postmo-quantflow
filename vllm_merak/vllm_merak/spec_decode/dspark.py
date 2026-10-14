# SPDX-License-Identifier: Apache-2.0
"""DSpark proposals and accepted-prefix transactions for serial Merak serving."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

DSPARK_PROPOSER = "vllm_merak.spec_decode.dspark.DeepSeekV4DSparkProposer"


def validate_dspark_service_config(config: Any) -> None:
    """Keep the heterogeneous runtime under one request and one cache owner."""
    spec = config.speculative_config
    if spec is None or spec.method != "custom_class" or spec.model != DSPARK_PROPOSER:
        raise ValueError(f"DeepSeek-V4 speculative serving requires custom_class {DSPARK_PROPOSER}")
    if spec.draft_sample_method != "probabilistic" or spec.rejection_sample_method != "standard":
        raise ValueError("DSpark requires probabilistic drafts and standard rejection sampling")
    if int(config.scheduler_config.max_num_seqs) != 1:
        raise ValueError("DSpark requires max_num_seqs=1")
    if bool(config.cache_config.enable_prefix_caching):
        raise ValueError("DSpark requires enable_prefix_caching=False")
    if bool(getattr(config.scheduler_config, "async_scheduling", False)):
        raise ValueError("DSpark cache commits require async_scheduling=False")


@dataclass
class _PendingVerify:
    transaction: Any
    input_ids: torch.Tensor
    hidden: torch.Tensor
    past: int


class DSparkServingState:
    """Reuse the exported graph callbacks while vLLM owns token sampling.

    The target transaction starts before verification. A rejection restores
    SWA/compressor/append state and replays only committed inputs; the draft
    cache receives the replayed features. The sampled recovery/bonus token
    remains uncached until the next scheduler step.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.owner: str | None = None
        self.past = 0
        self.proposal: torch.Tensor | None = None
        self.pending: _PendingVerify | None = None

    def forward_target(self, req_id: str, ids: torch.Tensor, *, past: int,
                       phase: str) -> torch.Tensor:
        from xhmodel_merak.xh_llm.models.deepseek_v4.dspark_verification import DSparkTargetTransaction

        engine = self.engine
        ids = ids.reshape(1, -1)
        if phase not in {"prefill", "decode", "verify"} or not req_id or not ids.numel():
            raise ValueError("DSpark requires a request, nonempty tokens and a valid phase")
        if past == 0:
            if phase != "prefill" or self.pending is not None:
                raise RuntimeError("Cannot reset DSpark with an uncommitted verification")
            engine.target_cache.cache_inputs()
            engine.target_cache.reset_kv_cache()
            engine.cache.reset()
            self.owner, self.past, self.proposal = req_id, 0, None
        if req_id != self.owner or past != self.past or engine.cache.past_length != past:
            raise RuntimeError("DSpark target/draft cache owner or length diverged")
        if self.pending is not None:
            raise RuntimeError("DSpark verification must be committed before another forward")
        if phase == "verify":
            if self.proposal is None or ids.shape[1] > self.proposal.shape[1] or not torch.equal(
                ids.to(self.proposal), self.proposal[:, :ids.shape[1]]
            ):
                raise RuntimeError("DSpark verify inputs do not match the outstanding proposal")
            transaction = DSparkTargetTransaction(engine.target_cache, past_length=past,
                                                  input_length=ids.shape[1])
            try:
                logits, hidden = engine.target_forward(ids, past, False)
                # CUDA graph outputs may be borrowed; replay must not overwrite
                # the logits currently being consumed by vLLM's sampler.
                self.pending = _PendingVerify(transaction, ids.clone(), hidden.clone(), past)
                return logits[:, :ids.shape[1]].clone()
            except BaseException:
                transaction.rollback()
                raise
        # The scheduler may discard a proposal near the output/context limit.
        self.proposal = None
        collected = []
        width = engine.spec.context_chunk_length if phase == "prefill" else engine.spec.block_size + 1
        for chunk in ids.split(width, dim=1):
            logits, hidden = engine.target_forward(chunk, self.past, phase == "prefill")
            engine.cache.commit_context(engine.context if phase == "prefill" else engine.context_decode,
                                        hidden.to(engine.device), current_length=chunk.shape[1])
            self.past += chunk.shape[1]
            if logits.shape[1] == 1 and phase == "prefill":
                # Production prefill exports only its final row; prompt
                # logprobs are rejected by the runner. vLLM samples only the
                # final position of the complete scheduled prefill span.
                collected.append(logits.expand(-1, chunk.shape[1], -1).clone())
            else:
                collected.append(logits[:, :chunk.shape[1]].clone())
        return torch.cat(collected, dim=1)

    def commit(self, req_id: str, accepted_steps: int, sampled: torch.Tensor) -> None:
        pending = self.pending
        if req_id != self.owner or pending is None:
            raise RuntimeError("DSpark has no matching pending target verification")
        length = pending.input_ids.shape[1]
        valid = sampled.reshape(-1)
        valid = valid[valid != -1]
        if not 1 <= accepted_steps <= length or valid.numel() != accepted_steps:
            raise ValueError("DSpark accepted target steps disagree with sampled tokens")
        if not torch.equal(valid[:accepted_steps - 1].to(pending.input_ids),
                           pending.input_ids[0, 1:accepted_steps]):
            raise RuntimeError("DSpark sampler accepted a prefix different from the proposal")
        engine = self.engine
        try:
            if accepted_steps == length:
                hidden = pending.hidden
            else:
                pending.transaction.rollback()
                self.pending = None
                _, hidden = engine.target_forward(pending.input_ids[:, :accepted_steps], pending.past, False)
                engine.statistics.replayed_target_tokens += accepted_steps
            engine.cache.commit_context(engine.context_decode, hidden.to(engine.device),
                                        current_length=accepted_steps)
            if accepted_steps == length:
                pending.transaction.commit()
        except BaseException:
            # A failed context graph may have written only some draft caches.
            # Drop this request rather than retain divergent cache lengths.
            self.release(req_id)
            raise
        engine.statistics.rounds += 1
        engine.statistics.proposed += length - 1
        engine.statistics.accepted += accepted_steps - 1
        self.past = pending.past + accepted_steps
        self.pending, self.proposal = None, None

    def propose(self, req_id: str, anchor: int, *, count: int, temperature: float,
                generator: torch.Generator | None) -> tuple[list[int], torch.Tensor | None]:
        from xhmodel_merak.xh_llm.models.deepseek_v4.dspark_runtime import dspark_attention_mask, sample_dspark

        engine = self.engine
        if req_id != self.owner or self.pending is not None:
            raise RuntimeError("DSpark proposal has no matching committed target cache")
        if self.proposal is not None:
            raise RuntimeError("DSpark already has an outstanding proposal")
        if self.past + engine.spec.block_size + 1 > engine.spec.max_context_length:
            return [], None
        query = torch.full((1, engine.spec.block_size), engine.spec.noise_token_id,
                           dtype=torch.int32, device=engine.device)
        query[:, 0] = anchor
        engine.cache.begin_proposal()
        try:
            outputs = engine.draft(engine.embedding(query),
                                   torch.tensor([self.past], dtype=torch.int32, device=engine.device),
                                   torch.tensor([engine.spec.block_size], dtype=torch.int32, device=engine.device),
                                   dspark_attention_mask(engine.spec, self.past, device=engine.device),
                                   *engine.cache.caches)
            outputs = tuple(outputs.values()) if isinstance(outputs, dict) else outputs
            tokens, probabilities, _ = sample_dspark(*outputs, query[:, :1], engine.markov,
                                                     temperature=temperature, generator=generator)
        finally:
            engine.cache.discard_proposal()
        tokens = tokens[:, :count]
        self.proposal = torch.cat((query[:, :1], tokens), dim=1)
        return tokens[0].tolist(), probabilities[:, :count]

    def abort(self) -> None:
        if self.pending is not None:
            self.pending.transaction.rollback()
        self.pending, self.proposal = None, None

    def release(self, req_id: str) -> None:
        if req_id == self.owner:
            self.abort()
            self.engine.cache.reset()
            self.owner, self.past = None, 0


class DeepSeekV4DSparkProposer:
    """vLLM custom proposer using the target package's four DSpark graphs."""

    def __init__(self, vllm_config: Any) -> None:
        validate_dspark_service_config(vllm_config)
        os.environ.setdefault("ENABLE_HMINFERENCE_V2", "1")
        self.vllm_config = vllm_config
        metadata = json.loads((Path(vllm_config.model_config.model) / "golden_meta_info.json").read_text())
        spec = metadata.get("dspark", {}).get("spec", {})
        self.num_draft_tokens = int(vllm_config.speculative_config.num_speculative_tokens)
        if not 1 <= self.num_draft_tokens <= int(spec.get("block_size", 0)):
            raise ValueError("DSpark draft count exceeds the exported graph capacity")
        self.runner: Any | None = None
        self.state: DSparkServingState | None = None
        self._last_draft_probs: torch.Tensor | None = None
        self._generators: dict[str, torch.Generator] = {}

    def attach_merak_runner(self, runner: Any) -> None:
        if self.runner is not None:
            if self.runner is not runner:
                raise RuntimeError("DSpark proposer cannot attach to two runners")
            return
        runtime = runner.model.inference_model
        devices = getattr(runtime, "_valid_devices", [runner.model.inference_device])
        device = devices[-1]
        device = torch.device(f"cuda:{device}") if isinstance(device, int) else torch.device(device)
        engine = runtime.build_dspark_engine(device=str(device))
        self.runner, self.state = runner, DSparkServingState(engine)

    def forward_target(self, *, input_ids: torch.Tensor, positions: torch.Tensor | None,
                       request_id: str, **kwargs: Any) -> torch.Tensor:
        assert self.runner is not None and self.state is not None
        if kwargs.get("inputs_embeds") is not None:
            raise ValueError("DSpark serving currently requires token IDs")
        model = self.runner.model
        flat = model._flatten_input_ids(input_ids)
        metadata = self.runner._get_attn_metadata()
        if metadata is None:
            past = int(positions.reshape(-1)[0]) if positions is not None else 0
            start, end = 0, flat.numel()
        else:
            if metadata.past_lens.numel() != 1:
                raise RuntimeError("DSpark requires exactly one scheduled request")
            past = int(metadata.past_lens[0])
            start, end = map(int, metadata.subsequence_begins.tolist())
        phase = ("verify" if request_id in self.runner._merak_verify_req_ids
                 else "prefill" if past == 0 or end - start > 1 else "decode")
        logits = self.state.forward_target(request_id, flat[start:end], past=past, phase=phase)
        output = torch.zeros(flat.numel(), logits.shape[-1], device=self.runner.device, dtype=logits.dtype)
        output[start:end].copy_(logits[0].to(output))
        model._cache_owner_request_id = request_id
        model._cache_owner_next_past_len = self.state.past
        if phase == "verify":
            self.runner._merak_pending_verify_req_ids.add(request_id)
        return output

    def on_target_accept(self, *, req_id: str, accepted_steps: int,
                         sampled_token_ids: torch.Tensor | Sequence[int]) -> None:
        assert self.state is not None and self.runner is not None
        self.state.commit(req_id, accepted_steps, torch.as_tensor(sampled_token_ids))
        self.runner.model._cache_owner_next_past_len = self.state.past

    def propose(self, sampled_token_ids: list[list[int]], num_tokens_no_spec: Any,
                token_ids_cpu: Any, *, slot_mappings: Any = None) -> list[list[int]]:
        del num_tokens_no_spec, token_ids_cpu, slot_mappings
        assert self.state is not None and self.runner is not None
        self._last_draft_probs = None
        req_ids = list(self.runner.input_batch.req_ids)
        if len(req_ids) != 1 or len(sampled_token_ids) != 1:
            raise RuntimeError("DSpark requires exactly one sampled request")
        valid = [int(x) for x in sampled_token_ids[0] if int(x) != -1]
        if not valid:
            return [[]]
        req_id = req_ids[0]
        batch = self.runner.input_batch
        idx = batch.req_id_to_index[req_id]
        temperature = 0.0 if req_id in batch.greedy_reqs else float(batch.temperature_cpu[idx])
        # The vLLM sampler's generator is on its primary device; the DSpark
        # graphs may live on another GPU. Keep a distinct seeded draft stream.
        generator = self._generators.get(req_id)
        if generator is None:
            source = getattr(batch, "generators", {}).get(idx)
            generator = torch.Generator(device=self.state.engine.device)
            if source is not None:
                # Distinct device generators with the same seed can repeat
                # random draws. Domain-separate the draft stream from the
                # target/rejection stream while preserving request seeding.
                generator.manual_seed((source.initial_seed() ^ 0x44535041524B) & ((1 << 63) - 1))
            else:
                generator.seed()
            self._generators[req_id] = generator
        tokens, q = self.state.propose(req_id, valid[-1], count=self.num_draft_tokens,
                                       temperature=temperature, generator=generator)
        if q is not None:
            self._last_draft_probs = q.to(self.runner.device).contiguous()
            self.runner._draft_probs = self._last_draft_probs
            self.runner._draft_prob_req_ids = [req_id]
        return [tokens]

    def take_last_draft_probs(self) -> torch.Tensor | None:
        return self._last_draft_probs

    def release_requests(self, req_ids: Sequence[str]) -> None:
        for req_id in req_ids:
            if self.state is not None:
                self.state.release(req_id)
            self._generators.pop(req_id, None)
        self._last_draft_probs = None

    def release_stale_requests(self, active_req_ids: Sequence[str]) -> None:
        if self.state is not None and self.state.owner not in active_req_ids:
            self.release_requests([self.state.owner] if self.state.owner else [])

    def abort_pending_verifies(self) -> None:
        if self.state is not None:
            self.state.abort()


__all__ = ["DeepSeekV4DSparkProposer", "DSparkServingState"]
