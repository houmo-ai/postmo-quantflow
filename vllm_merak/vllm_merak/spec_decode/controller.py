"""State machine shared by Merak HMONNX speculative-decoding backends.

The target model, vLLM rejection sampler, and HMONNX draft graphs have three
different notions of sequence length:

* ``target_visible_length`` is the number of tokens already materialized in
  the target cache.  The latest sampled token is deliberately not included.
* Qwen MTP's ``draft_context_length`` counts shifted ``(target_hidden,
  next_token)`` pairs and is therefore ``target_visible_length - 1``.
* Qwen DFlash's ``draft_context_length`` counts target hidden states and is
  therefore exactly ``target_visible_length``.
* Gemma4 shares the target KV read-only.  Its q=1 attention input uses
  ``target_visible_length - 1`` as a synthetic past length, while the latest
  sampled assistant input keeps RoPE position ``target_visible_length``.

Keeping those values explicit is what makes reject/rollback correct.  Dense
per-token KV may leave unreachable stale rows in place.  Paged SEFP V cache is
different because 64 tokens share one exponent: the runtime transaction
restores the pre-verify pages and rebuilds only the accepted prefix so rejected
rows cannot rescale committed values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

import torch

from ..determinism_trace import (
    emit_event as emit_determinism_event,
)
from ..determinism_trace import fingerprint as determinism_fingerprint
from ..determinism_trace import trace_enabled as determinism_trace_enabled
from .artifact import HMONNXSpecDecodeArtifact


TargetPhase = Literal["prefill", "decode", "verify"]


class HMONNXDraftBackend(Protocol):
    """Execution boundary implemented by the concrete HMONNX backend."""

    def append_mtp_context(
        self,
        *,
        req_id: str,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        past_length: int,
        next_token_embeddings: torch.Tensor | None = None,
    ) -> None: ...

    def append_dflash_context(
        self,
        *,
        req_id: str,
        hidden_states: torch.Tensor,
        past_length: int,
        decode_step: bool,
    ) -> None: ...

    def draft_mtp(
        self,
        *,
        req_id: str,
        current_token_id: int,
        target_hidden: torch.Tensor,
        past_length: int,
        num_draft_tokens: int,
    ) -> Sequence[int] | torch.Tensor: ...

    def commit_mtp_draft(
        self,
        *,
        req_id: str,
        accepted_steps: int,
    ) -> None: ...

    def draft_dflash(
        self,
        *,
        req_id: str,
        current_token_id: int,
        past_length: int,
        num_draft_tokens: int,
        temperature: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Sequence[int] | torch.Tensor: ...

    def can_draft_dflash(
        self,
        *,
        past_length: int,
        num_draft_tokens: int,
    ) -> bool: ...

    def take_last_dflash_draft_probs(self) -> torch.Tensor | None: ...

    def draft_gemma4(
        self,
        *,
        req_id: str,
        current_token_id: int,
        target_hidden: torch.Tensor,
        target_visible_length: int,
        num_draft_tokens: int,
        page_attention_group_contexts: dict[int, Any],
    ) -> Sequence[int] | torch.Tensor: ...

    def release_request(self, req_id: str) -> None: ...


@dataclass
class _PendingVerify:
    past_length: int
    input_ids: torch.Tensor
    hidden_states: torch.Tensor
    page_attention_group_contexts: dict[int, Any] | None


@dataclass
class _OutstandingProposal:
    current_token_id: int
    draft_token_ids: tuple[int, ...]


@dataclass
class _RequestState:
    target_visible_length: int = 0
    draft_context_length: int = 0
    pending_target_hidden: torch.Tensor | None = None
    page_attention_group_contexts: dict[int, Any] | None = None
    pending_verify: _PendingVerify | None = None
    outstanding_proposal: _OutstandingProposal | None = None


def _sequence_2d(
    value: torch.Tensor,
    *,
    name: str,
    expected_length: int | None = None,
) -> torch.Tensor:
    """Normalize ``[Q, D]``/``[1, Q, D]`` sequence outputs to ``[Q, D]``."""

    if value.ndim == 3:
        if value.shape[0] != 1:
            raise RuntimeError(
                f"{name} supports batch size 1, got shape={tuple(value.shape)}"
            )
        value = value[0]
    if value.ndim != 2:
        raise RuntimeError(
            f"{name} must have shape [Q, D] or [1, Q, D], got {tuple(value.shape)}"
        )
    if expected_length is not None and value.shape[0] != expected_length:
        raise RuntimeError(
            f"{name} sequence length mismatch: expected={expected_length}, "
            f"got={value.shape[0]}"
        )
    return value


def _token_sequence_1d(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 2:
        if value.shape[0] != 1:
            raise RuntimeError(
                f"{name} supports batch size 1, got shape={tuple(value.shape)}"
            )
        value = value[0]
    if value.ndim != 1:
        raise RuntimeError(
            f"{name} must have shape [Q] or [1, Q], got {tuple(value.shape)}"
        )
    return value


def _draft_ids(value: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu").reshape(-1).tolist()
    return tuple(int(token_id) for token_id in value)


class HMONNXSpecDecodeController:
    """Own per-request speculative state and enforce the artifact contract."""

    def __init__(
        self,
        artifact: HMONNXSpecDecodeArtifact,
        backend: HMONNXDraftBackend,
        *,
        num_draft_tokens: int | None = None,
    ) -> None:
        self.artifact = artifact
        self.backend = backend
        self.num_draft_tokens = (
            artifact.num_draft_tokens
            if num_draft_tokens is None
            else int(num_draft_tokens)
        )
        if not 1 <= self.num_draft_tokens <= artifact.num_draft_tokens:
            raise ValueError(
                "Runtime draft width must be within the exported HMONNX "
                f"capacity: runtime={self.num_draft_tokens}, "
                f"exported={artifact.num_draft_tokens}"
            )
        self._states: dict[str, _RequestState] = {}

    @property
    def active_request_ids(self) -> frozenset[str]:
        return frozenset(self._states)

    def _state(self, req_id: str) -> _RequestState:
        return self._states.setdefault(req_id, _RequestState())

    def _require_hidden(
        self,
        result: Any,
        *,
        req_id: str,
        phase: TargetPhase,
        query_length: int,
    ) -> torch.Tensor:
        hidden = getattr(result, "hidden_states", None)
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError(
                "Merak speculative target did not return the required hidden "
                f"states for request {req_id}"
            )
        normalized = _sequence_2d(
            hidden,
            name=f"target hidden for request {req_id}",
        )
        if self.artifact.is_gemma4 and phase == "prefill":
            if normalized.shape[0] != 1:
                raise RuntimeError(
                    "Gemma4 target prefill must return exactly its final "
                    "hidden state: "
                    f"request={req_id}, got={normalized.shape[0]}"
                )
            return normalized
        if normalized.shape[0] != query_length:
            raise RuntimeError(
                f"target hidden for request {req_id} sequence length "
                f"mismatch: expected={query_length}, "
                f"got={normalized.shape[0]}"
            )
        return normalized

    @staticmethod
    def _check_target_position(
        state: _RequestState,
        *,
        req_id: str,
        past_len: int,
    ) -> None:
        if past_len != state.target_visible_length:
            raise RuntimeError(
                "Merak speculative target/cache length diverged: "
                f"request={req_id}, runtime_past={past_len}, "
                f"controller_visible={state.target_visible_length}"
            )

    def _append_qwen_mtp_target_tokens(
        self,
        *,
        req_id: str,
        state: _RequestState,
        input_ids: torch.Tensor,
        input_embeddings: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> None:
        hidden_parts: list[torch.Tensor] = []
        token_parts: list[torch.Tensor] = []
        embedding_parts: list[torch.Tensor] = []
        if state.pending_target_hidden is not None:
            hidden_parts.append(state.pending_target_hidden)
            token_parts.append(input_ids[:1])
            if input_embeddings is not None:
                embedding_parts.append(input_embeddings[:1])
        if input_ids.numel() > 1:
            hidden_parts.append(hidden_states[:-1])
            token_parts.append(input_ids[1:])
            if input_embeddings is not None:
                embedding_parts.append(input_embeddings[1:])
        if hidden_parts:
            pair_hidden = torch.cat(hidden_parts, dim=0)
            pair_tokens = torch.cat(token_parts, dim=0)
            pair_embeddings = (
                torch.cat(embedding_parts, dim=0) if embedding_parts else None
            )
            self.backend.append_mtp_context(
                req_id=req_id,
                hidden_states=pair_hidden,
                next_token_ids=pair_tokens,
                next_token_embeddings=pair_embeddings,
                past_length=state.draft_context_length,
            )
            state.draft_context_length += int(pair_tokens.numel())
        state.pending_target_hidden = hidden_states[-1:].contiguous()

    def on_target_forward(
        self,
        *,
        req_id: str,
        phase: TargetPhase,
        past_len: int,
        input_ids: torch.Tensor | None,
        result: Any,
        input_embeddings: torch.Tensor | None = None,
        page_attention_group_contexts: dict[int, Any] | None = None,
    ) -> None:
        """Record one target invocation before vLLM performs acceptance."""

        if phase not in {"prefill", "decode", "verify"}:
            raise ValueError(f"Unsupported target phase: {phase!r}")
        if input_ids is None:
            raise RuntimeError(
                "HMONNX speculative decoding currently requires target input_ids"
            )
        tokens = _token_sequence_1d(
            input_ids,
            name=f"target input_ids for request {req_id}",
        )
        query_length = int(tokens.numel())
        if query_length <= 0:
            raise RuntimeError("Merak speculative target query cannot be empty")
        embeddings = (
            _sequence_2d(
                input_embeddings,
                name=f"target input embeddings for request {req_id}",
                expected_length=query_length,
            )
            if input_embeddings is not None
            else None
        )

        state = self._state(req_id)
        self._check_target_position(state, req_id=req_id, past_len=past_len)
        hidden = self._require_hidden(
            result,
            req_id=req_id,
            phase=phase,
            query_length=query_length,
        )
        if determinism_trace_enabled():
            logits = getattr(result, "logits", None)
            logits_argmax: list[int] | None = None
            if isinstance(logits, torch.Tensor) and logits.numel() > 0:
                flat_logits = logits.reshape(-1, logits.shape[-1])
                logits_argmax = [
                    int(token_id)
                    for token_id in flat_logits.argmax(dim=-1)
                    .detach()
                    .to(device="cpu")
                    .tolist()
                ]
            emit_determinism_event(
                "spec_target_forward",
                request_id=req_id,
                payload={
                    "mode": self.artifact.mode,
                    "phase": phase,
                    "past_len": past_len,
                    "input_ids": [int(token_id) for token_id in tokens.tolist()],
                    "logits_argmax": logits_argmax,
                    "hidden": determinism_fingerprint(hidden),
                },
            )

        if phase == "verify":
            if state.pending_verify is not None:
                raise RuntimeError(
                    f"Request {req_id} already has an uncommitted verify result"
                )
            proposal = state.outstanding_proposal
            if proposal is None:
                raise RuntimeError(
                    f"Request {req_id} entered verify without a draft proposal"
                )
            expected = (proposal.current_token_id, *proposal.draft_token_ids)
            actual = tuple(int(token_id) for token_id in tokens.tolist())
            if actual != expected[:query_length]:
                raise RuntimeError(
                    "Merak target verify tokens do not match the outstanding "
                    f"draft: request={req_id}, expected_prefix="
                    f"{expected[:query_length]}, actual={actual}"
                )
            state.pending_verify = _PendingVerify(
                past_length=past_len,
                input_ids=tokens,
                hidden_states=hidden,
                page_attention_group_contexts=page_attention_group_contexts,
            )
            return

        if state.outstanding_proposal is not None:
            raise RuntimeError(
                f"Request {req_id} ran {phase} before consuming its draft proposal"
            )

        if self.artifact.is_qwen_mtp:
            self._append_qwen_mtp_target_tokens(
                req_id=req_id,
                state=state,
                input_ids=tokens,
                input_embeddings=embeddings,
                hidden_states=hidden,
            )
        elif self.artifact.mode == "dflash":
            if state.draft_context_length != past_len:
                raise RuntimeError(
                    "Qwen DFlash context length diverged before target append: "
                    f"request={req_id}, draft={state.draft_context_length}, "
                    f"target={past_len}"
                )
            self.backend.append_dflash_context(
                req_id=req_id,
                hidden_states=hidden,
                past_length=past_len,
                decode_step=phase == "decode",
            )
            state.draft_context_length += query_length
            state.pending_target_hidden = hidden[-1:].contiguous()
        elif self.artifact.is_gemma4:
            state.pending_target_hidden = hidden[-1:].contiguous()
            state.page_attention_group_contexts = page_attention_group_contexts
        else:  # pragma: no cover - artifact parser rejects unknown ABIs.
            raise RuntimeError(f"Unsupported draft ABI: {self.artifact.draft_abi}")

        state.target_visible_length = past_len + query_length
        self._validate_committed_lengths(req_id, state)

    def _validate_committed_lengths(
        self,
        req_id: str,
        state: _RequestState,
    ) -> None:
        if self.artifact.is_qwen_mtp:
            expected = max(0, state.target_visible_length - 1)
            if state.draft_context_length != expected:
                raise RuntimeError(
                    "Qwen MTP shifted context length diverged: "
                    f"request={req_id}, draft={state.draft_context_length}, "
                    f"expected={expected}"
                )
        elif self.artifact.mode == "dflash":
            if state.draft_context_length != state.target_visible_length:
                raise RuntimeError(
                    "Qwen DFlash context length diverged: "
                    f"request={req_id}, draft={state.draft_context_length}, "
                    f"target={state.target_visible_length}"
                )

    def on_target_accept(
        self,
        *,
        req_id: str,
        accepted_steps: int,
        sampled_token_ids: torch.Tensor | Sequence[int],
    ) -> None:
        """Commit exactly the target snapshots selected by rejection sampling."""

        state = self._states.get(req_id)
        if state is None or state.pending_verify is None:
            raise RuntimeError(
                f"Request {req_id} has no pending target verify to accept"
            )
        pending = state.pending_verify
        query_length = int(pending.input_ids.numel())
        if not 1 <= accepted_steps <= query_length:
            raise RuntimeError(
                "Accepted target steps are outside the verify block: "
                f"request={req_id}, accepted={accepted_steps}, "
                f"query_length={query_length}"
            )

        if isinstance(sampled_token_ids, torch.Tensor):
            sampled = sampled_token_ids.detach().to(device="cpu").reshape(-1)
            valid_count = int((sampled != -1).sum().item())
        else:
            sampled = torch.as_tensor(tuple(sampled_token_ids), dtype=torch.long)
            valid_count = int((sampled != -1).sum().item())
        if valid_count != accepted_steps:
            raise RuntimeError(
                "vLLM sampled-token validity disagrees with accepted target "
                f"steps: request={req_id}, valid={valid_count}, "
                f"accepted={accepted_steps}"
            )

        selected_hidden = pending.hidden_states[
            accepted_steps - 1 : accepted_steps
        ].contiguous()
        if self.artifact.is_qwen_mtp:
            # The private MTP assistant writes K autoregressive cache rows
            # while proposing. Select the exact accepted snapshot before a
            # rejected row can remain in a shared-exponent SEFP V group.
            self.backend.commit_mtp_draft(
                req_id=req_id,
                accepted_steps=accepted_steps,
            )
            # The draft graph already wrote all K speculative entries in
            # place. If every draft is accepted, vLLM also returns the target
            # bonus token (K+1 valid steps), but the K-call autoregressive
            # drafter has not cached the final shifted pair
            # ``hidden(draft[K-1]) -> draft[K]``. Materialize that one pair
            # through the context graph before advancing the logical length.
            if accepted_steps == self.num_draft_tokens + 1:
                missing_pair_index = accepted_steps - 1
                self.backend.append_mtp_context(
                    req_id=req_id,
                    hidden_states=pending.hidden_states[
                        missing_pair_index - 1 : missing_pair_index
                    ].contiguous(),
                    next_token_ids=pending.input_ids[
                        missing_pair_index : missing_pair_index + 1
                    ].contiguous(),
                    past_length=(state.draft_context_length + self.num_draft_tokens),
                )
            state.draft_context_length += accepted_steps
        elif self.artifact.mode == "dflash":
            accepted_hidden = pending.hidden_states[:accepted_steps].contiguous()
            self.backend.append_dflash_context(
                req_id=req_id,
                hidden_states=accepted_hidden,
                past_length=pending.past_length,
                decode_step=True,
            )
            state.draft_context_length += accepted_steps
        elif self.artifact.is_gemma4:
            if pending.page_attention_group_contexts is None:
                raise RuntimeError(
                    "Gemma4 MTP verify did not retain target page contexts"
                )
            state.page_attention_group_contexts = pending.page_attention_group_contexts

        state.pending_target_hidden = selected_hidden
        state.target_visible_length = pending.past_length + accepted_steps
        state.pending_verify = None
        state.outstanding_proposal = None
        self._validate_committed_lengths(req_id, state)
        if determinism_trace_enabled():
            emit_determinism_event(
                "spec_target_accept",
                request_id=req_id,
                payload={
                    "mode": self.artifact.mode,
                    "accepted_steps": accepted_steps,
                    "sampled_token_ids": [
                        int(token_id) for token_id in sampled.tolist()
                    ],
                    "target_visible_length": state.target_visible_length,
                    "draft_context_length": state.draft_context_length,
                    "selected_hidden": determinism_fingerprint(selected_hidden),
                },
            )

    def propose_one(
        self,
        *,
        req_id: str,
        current_token_id: int,
        draft_temperature: float | None = None,
        draft_generator: torch.Generator | None = None,
    ) -> list[int]:
        state = self._states.get(req_id)
        if state is None or state.pending_target_hidden is None:
            return []
        if state.pending_verify is not None:
            raise RuntimeError(
                f"Request {req_id} cannot draft before verify acceptance"
            )
        if state.outstanding_proposal is not None:
            raise RuntimeError(f"Request {req_id} already has an outstanding proposal")

        num_drafts = self.num_draft_tokens
        if self.artifact.is_qwen_mtp:
            self._validate_committed_lengths(req_id, state)
            raw_drafts = self.backend.draft_mtp(
                req_id=req_id,
                current_token_id=current_token_id,
                target_hidden=state.pending_target_hidden,
                past_length=state.draft_context_length,
                num_draft_tokens=num_drafts,
            )
        elif self.artifact.mode == "dflash":
            self._validate_committed_lengths(req_id, state)
            # A DFlash checkpoint may use a wider fixed masked-token block
            # than the runtime verify width (for example 16 physical tokens
            # for K=9).  Near max_model_len, vLLM can still have room for K
            # outputs while the assistant's full block no longer fits its
            # RoPE/cache contract. Fall back to ordinary target decode rather
            # than executing an out-of-range draft graph.
            if not self.backend.can_draft_dflash(
                past_length=state.draft_context_length,
                num_draft_tokens=num_drafts,
            ):
                return []
            raw_drafts = self.backend.draft_dflash(
                req_id=req_id,
                current_token_id=current_token_id,
                past_length=state.draft_context_length,
                num_draft_tokens=num_drafts,
                temperature=draft_temperature,
                generator=draft_generator,
            )
        elif self.artifact.is_gemma4:
            contexts = state.page_attention_group_contexts
            if contexts is None:
                raise RuntimeError(
                    "Gemma4 MTP cannot draft without target page contexts"
                )
            raw_drafts = self.backend.draft_gemma4(
                req_id=req_id,
                current_token_id=current_token_id,
                target_hidden=state.pending_target_hidden,
                target_visible_length=state.target_visible_length,
                num_draft_tokens=num_drafts,
                page_attention_group_contexts=contexts,
            )
        else:  # pragma: no cover - artifact parser rejects unknown ABIs.
            raise RuntimeError(f"Unsupported draft ABI: {self.artifact.draft_abi}")

        drafts = _draft_ids(raw_drafts)
        if len(drafts) != num_drafts:
            raise RuntimeError(
                "HMONNX draft graph returned the wrong number of tokens: "
                f"request={req_id}, expected={num_drafts}, got={len(drafts)}"
            )
        state.outstanding_proposal = _OutstandingProposal(
            current_token_id=int(current_token_id),
            draft_token_ids=drafts,
        )
        if determinism_trace_enabled():
            emit_determinism_event(
                "spec_draft_proposal",
                request_id=req_id,
                payload={
                    "mode": self.artifact.mode,
                    "current_token_id": int(current_token_id),
                    "draft_token_ids": list(drafts),
                    "target_visible_length": state.target_visible_length,
                    "draft_context_length": state.draft_context_length,
                    "target_hidden": determinism_fingerprint(
                        state.pending_target_hidden
                    ),
                },
            )
        return list(drafts)

    def release_requests(self, req_ids: Sequence[str]) -> None:
        for req_id in req_ids:
            if self._states.pop(req_id, None) is not None:
                self.backend.release_request(req_id)

    def release_stale_requests(self, active_req_ids: Sequence[str]) -> None:
        active = set(active_req_ids)
        self.release_requests(
            [req_id for req_id in self._states if req_id not in active]
        )

    def abort_pending_verifies(self) -> None:
        """Drop requests whose mutable draft/target transaction is ambiguous."""

        self.release_requests(
            [
                req_id
                for req_id, state in self._states.items()
                if state.pending_verify is not None
                or state.outstanding_proposal is not None
            ]
        )


__all__ = [
    "HMONNXDraftBackend",
    "HMONNXSpecDecodeController",
    "TargetPhase",
]
