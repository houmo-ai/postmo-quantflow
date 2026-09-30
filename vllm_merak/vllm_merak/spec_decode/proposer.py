"""vLLM ``custom_class`` adapter for HMONNX speculative decoding."""

from __future__ import annotations

import os
from typing import Any, Sequence

import torch

from .artifact import load_hmonnx_spec_decode_artifact
from .controller import HMONNXSpecDecodeController, TargetPhase
from .hmonnx_backend import HMONNXDraftRuntimeBackend


class HMONNXSpecDecodeProposer:
    """Custom proposer selected through ``SpeculativeConfig.method``.

    Launch example::

        --speculative-config '{
          "method": "custom_class",
          "model":
            "vllm_merak.spec_decode.proposer.HMONNXSpecDecodeProposer",
          "num_speculative_tokens": 4
        }'
    """

    def __init__(self, vllm_config: Any) -> None:
        # The proposer is constructed before the target model is loaded.  Both
        # target and draft graphs must use V2 so their PageAttention cache
        # transaction shares one runtime contract; enabling it later in
        # ``attach_merak_runner`` is already too late for the target session.
        # Preserve an explicit user override so the backend's invariant check
        # still reports the unsupported choice clearly.
        os.environ.setdefault("ENABLE_HMINFERENCE_V2", "1")
        self.vllm_config = vllm_config
        model_path = vllm_config.model_config.model
        artifact = load_hmonnx_spec_decode_artifact(
            model_path,
            required=True,
        )
        assert artifact is not None
        self.artifact = artifact

        speculative_config = vllm_config.speculative_config
        if speculative_config is None:
            raise ValueError("HMONNXSpecDecodeProposer requires speculative_config")
        configured_tokens = int(speculative_config.num_speculative_tokens)
        if not 1 <= configured_tokens <= artifact.num_draft_tokens:
            raise ValueError(
                "vLLM num_speculative_tokens exceeds the HMONNX export "
                f"capacity: configured={configured_tokens}, "
                f"exported={artifact.num_draft_tokens}"
            )
        self.num_draft_tokens = configured_tokens
        draft_sample_method = getattr(
            speculative_config,
            "draft_sample_method",
            "greedy",
        )
        self._probabilistic_dflash2 = bool(
            artifact.is_qwen_dflash2
            and draft_sample_method == "probabilistic"
            and getattr(speculative_config, "rejection_sample_method", None)
            == "standard"
        )
        self._last_draft_probs: torch.Tensor | None = None
        max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
        if max_num_seqs != 1:
            raise ValueError(
                "The current serial HMONNX speculative runtime requires "
                f"max_num_seqs=1, got {max_num_seqs}. This prevents sharing "
                "one mutable Qwen draft cache across concurrent requests."
            )
        if bool(
            getattr(
                vllm_config.cache_config,
                "enable_prefix_caching",
                False,
            )
        ):
            raise ValueError(
                "The current HMONNX speculative runtime requires "
                "enable_prefix_caching=False: a reused target prefix does not "
                "carry the Qwen MTP/DFlash draft context needed to resume "
                "without recomputation."
            )
        if (
            artifact.is_qwen_hybrid_spec_decode
            and str(
                getattr(
                    vllm_config.cache_config,
                    "mamba_cache_mode",
                    "none",
                )
            )
            != "none"
        ):
            raise ValueError(
                "Qwen HMONNX speculative decoding requires "
                "mamba_cache_mode='none'. Merak owns the accepted-step "
                "Linear-state transaction; vLLM align/all postprocessing "
                "must not mutate the same checkpoint pages."
            )

        self.runner: Any | None = None
        self.backend: HMONNXDraftRuntimeBackend | None = None
        self.controller: HMONNXSpecDecodeController | None = None

    def attach_merak_runner(self, runner: Any) -> None:
        if self.runner is not None:
            if self.runner is not runner:
                raise RuntimeError("HMONNX proposer cannot attach to two Merak runners")
            return
        self.runner = runner
        self.backend = HMONNXDraftRuntimeBackend(
            runner=runner,
            artifact=self.artifact,
        )
        self.controller = HMONNXSpecDecodeController(
            self.artifact,
            self.backend,
            num_draft_tokens=self.num_draft_tokens,
        )

    def _controller(self) -> HMONNXSpecDecodeController:
        if self.controller is None:
            raise RuntimeError(
                "HMONNX proposer was used before Merak runner attachment"
            )
        return self.controller

    def _draft_temperature(self, req_id: str) -> float | None:
        if not self._probabilistic_dflash2 or self.runner is None:
            return None
        input_batch = self.runner.input_batch
        if req_id in getattr(input_batch, "greedy_reqs", ()):
            return None
        req_index = input_batch.req_id_to_index.get(req_id)
        if req_index is None:
            raise RuntimeError(f"DFlash2 request {req_id!r} is not in the input batch")
        temperature = float(input_batch.temperature_cpu[req_index])
        return temperature if temperature >= 1e-5 else None

    def _draft_generator(self, req_id: str) -> torch.Generator | None:
        if not self._probabilistic_dflash2 or self.runner is None:
            return None
        input_batch = self.runner.input_batch
        req_index = input_batch.req_id_to_index.get(req_id)
        if req_index is None:
            raise RuntimeError(f"DFlash2 request {req_id!r} is not in the input batch")
        return getattr(input_batch, "generators", {}).get(req_index)

    def take_last_draft_probs(self) -> torch.Tensor | None:
        """Expose proposal probabilities to vLLM's standard rejection sampler."""

        return self._last_draft_probs

    def on_target_forward(
        self,
        *,
        req_id: str,
        phase: TargetPhase,
        past_len: int,
        input_ids: torch.Tensor | None,
        result: Any,
        page_attention_group_contexts: dict[int, Any] | None,
        input_embeddings: torch.Tensor | None = None,
    ) -> None:
        self._controller().on_target_forward(
            req_id=req_id,
            phase=phase,
            past_len=past_len,
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            result=result,
            page_attention_group_contexts=page_attention_group_contexts,
        )

    def on_target_accept(
        self,
        *,
        req_id: str,
        accepted_steps: int,
        sampled_token_ids: torch.Tensor | Sequence[int],
    ) -> None:
        self._controller().on_target_accept(
            req_id=req_id,
            accepted_steps=accepted_steps,
            sampled_token_ids=sampled_token_ids,
        )

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        *,
        slot_mappings: Any = None,
    ) -> list[list[int]]:
        del num_tokens_no_spec, token_ids_cpu, slot_mappings
        if self.runner is None:
            raise RuntimeError("HMONNX proposer has no attached Merak runner")
        req_ids = list(self.runner.input_batch.req_ids)
        if len(sampled_token_ids) > len(req_ids):
            raise RuntimeError(
                "vLLM sampled-token rows exceed active request ids: "
                f"rows={len(sampled_token_ids)}, requests={len(req_ids)}"
            )
        proposals: list[list[int]] = []
        draft_probability_rows: list[torch.Tensor] = []
        draft_probability_req_ids: list[str] = []
        self._last_draft_probs = None
        controller = self._controller()
        backend = self.backend
        if backend is None:
            raise RuntimeError("HMONNX proposer has no attached draft backend")
        for req_id, sampled_row in zip(
            req_ids,
            sampled_token_ids,
            strict=False,
        ):
            valid = [int(token_id) for token_id in sampled_row if int(token_id) != -1]
            if not valid:
                proposals.append([])
                continue
            proposal = controller.propose_one(
                req_id=req_id,
                current_token_id=valid[-1],
                draft_temperature=self._draft_temperature(req_id),
                draft_generator=self._draft_generator(req_id),
            )
            proposals.append(proposal)
            take_draft_probs = getattr(backend, "take_last_dflash_draft_probs", None)
            draft_probs = take_draft_probs() if callable(take_draft_probs) else None
            if draft_probs is not None:
                if draft_probs.shape[0] != 1 or draft_probs.shape[1] != len(proposal):
                    raise RuntimeError(
                        "DFlash2 proposal probability shape does not match its tokens: "
                        f"shape={tuple(draft_probs.shape)}, tokens={len(proposal)}"
                    )
                draft_probability_rows.append(draft_probs[0])
                draft_probability_req_ids.append(req_id)
        if draft_probability_rows:
            self._last_draft_probs = torch.stack(draft_probability_rows, dim=0)
            # vLLM 0.24's custom_class branch does not yet pull probabilities
            # from the proposer. Publish them on its existing runner bridge so
            # standard rejection sampling remains lossless for temperature > 0.
            self.runner._draft_probs = self._last_draft_probs
            self.runner._draft_prob_req_ids = draft_probability_req_ids
        return proposals

    def release_requests(self, req_ids: Sequence[str]) -> None:
        self._controller().release_requests(req_ids)

    def release_stale_requests(
        self,
        active_req_ids: Sequence[str],
    ) -> None:
        self._controller().release_stale_requests(active_req_ids)

    def abort_pending_verifies(self) -> None:
        self._controller().abort_pending_verifies()


__all__ = ["HMONNXSpecDecodeProposer"]
