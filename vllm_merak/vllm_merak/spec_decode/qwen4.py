# SPDX-License-Identifier: Apache-2.0
"""Qwen4's native draft/cache callbacks under vLLM's serial scheduler."""

import torch

from .serial import SerialHMONNXProposer


class Qwen4ServingState:
    def __init__(self, model):
        from xhmodel_merak.xh_llm.models.qwen4_exp.generation import create_mtp_runtime

        self.model = model
        self.draft = create_mtp_runtime(model)
        self.owner = None
        self.past = 0
        self.hidden = None
        self.proposal = None
        self.pending = None
        self.saved_draft = None
        self.caches = []

    def _positions(self, past, count):
        return torch.arange(past, past + count, device=self.model.device).view(1, 1, -1).expand(3, 1, -1)

    def _forward(self, ids, past, *, prefill=False, verify=False):
        from xhmodel_merak.xh_llm.models.qwen4_exp.generation import _page_step

        model = self.model
        model.set_verify() if verify else model.set_prefill() if prefill else model.set_decode()
        logits = _page_step(
            model,
            self.caches,
            ids,
            past,
            position_ids=self._positions(past, ids.shape[1]),
            return_all=True,
        )
        return logits.clone(), model.last_hidden_states[:, : ids.shape[1]].clone()

    def _append_context(self, ids, hidden, past, *, prefill):
        embeddings = self.model.embed_tokens(ids)
        previous = self.hidden
        if prefill:
            start = 1 if previous is None else 0
            hiddens = hidden[:, :-1] if previous is None else torch.cat((previous, hidden[:, :-1]), 1)
            if start < ids.shape[1]:
                self.draft.prefill(
                    embeddings[:, start:],
                    hiddens,
                    position_ids=self._positions(past + start, ids.shape[1] - start),
                )
        else:
            for i in range(ids.shape[1]):
                self.draft.step(
                    embeddings[:, i : i + 1],
                    previous if i == 0 else hidden[:, i - 1 : i],
                    position_ids=self._positions(past + i, 1),
                )
        self.hidden = hidden[:, -1:].clone()

    def _snapshot(self):
        from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_snapshot import snapshot_state

        state = self.model._qwen4_state
        return (
            snapshot_state(
                state.tensors,
                self.caches,
                specs={s.name: s for s in state.contract.tensors},
                past_length=self.past,
                write_length=self.model._qwen4_verify_length,
            ),
            state.group_start_position.clone(),
        )

    def _restore(self, saved):
        state = self.model._qwen4_state
        saved[0].restore(state.tensors, self.caches)
        state.group_start_position.copy_(saved[1])
        self.model._data_processor = None

    @torch.inference_mode()
    def forward_target(self, req_id, ids, *, past, phase):
        ids = ids.reshape(1, -1).to(self.model.device)
        if past == 0:
            self.model.reset_internal_state()
            self.draft.reset()
            self.owner, self.past, self.hidden = req_id, 0, None
            self.stats = dict(
                target_verify_calls=0,
                target_replay_tokens=0,
                draft_tokens_total=0,
                accepted_drafts_total=0,
                rejected_drafts_total=0,
            )
            self.model.last_mtp_stats = self.stats
        if req_id != self.owner or past != self.past or self.pending is not None:
            raise RuntimeError("Qwen4 target cache owner or committed length diverged")
        if phase == "verify":
            if self.proposal is None or not torch.equal(ids, self.proposal[:, : ids.shape[1]]):
                raise RuntimeError("Qwen4 verify inputs differ from its outstanding proposal")
            saved = self._snapshot()
            try:
                logits, hidden = self._forward(ids, past, verify=True)
                self.pending = saved, ids.clone(), hidden
                return logits
            except BaseException:
                self._restore(saved)
                self.abort()
                raise
        # At an output/context limit vLLM can discard the proposed suffix.
        if self.saved_draft is not None:
            self.draft.restore(self.saved_draft)
            self.saved_draft = self.proposal = None
        collected = []
        width = self.model.meta_info.model_config.prefill_chunk_length if phase == "prefill" else 1
        for chunk in ids.split(width, 1):
            logits, hidden = self._forward(chunk, self.past, prefill=phase == "prefill")
            self._append_context(chunk, hidden, self.past, prefill=phase == "prefill")
            self.past += chunk.shape[1]
            collected.append(logits)
        return torch.cat(collected, 1)

    @torch.inference_mode()
    def propose(self, req_id, anchor, *, count):
        if req_id != self.owner or self.pending is not None or self.proposal is not None:
            raise RuntimeError("Qwen4 draft requires the committed target prefix")
        if self.past + self.model._qwen4_verify_length > self.model.meta_info.model_config.context_max_length:
            return []
        self.saved_draft = self.draft.snapshot(write_length=count + 1)
        token, hidden = int(anchor), self.hidden
        candidates = []
        for i in range(count):
            ids = torch.tensor([[token]], device=self.model.device)
            logits, hidden = self.draft.step(
                self.model.embed_tokens(ids),
                hidden,
                position_ids=self._positions(self.past + i, 1),
            )
            token = int(logits[:, -1].argmax(-1).item())
            candidates.append(token)
        self.proposal = torch.tensor([[anchor, *candidates]], device=self.model.device)
        return candidates

    @torch.inference_mode()
    def commit(self, req_id, accepted_steps, sampled):
        if req_id != self.owner or self.pending is None:
            raise RuntimeError("Qwen4 has no pending verification for this request")
        saved, ids, hidden = self.pending
        sampled = sampled.reshape(-1)
        sampled = sampled[sampled != -1]
        if not 1 <= accepted_steps <= ids.shape[1] or sampled.numel() != accepted_steps:
            raise ValueError("Qwen4 accepted prefix length differs from vLLM sampling")
        if not torch.equal(sampled[: accepted_steps - 1].to(ids), ids[0, 1:accepted_steps]):
            raise ValueError("Qwen4 accepted prefix differs from the verified proposal")
        committed = ids[:, :accepted_steps]
        if accepted_steps < ids.shape[1]:
            self._restore(saved)
            pieces = []
            for i in range(accepted_steps):
                _, hc = self._forward(committed[:, i : i + 1], self.past + i)
                pieces.append(hc)
            hidden = torch.cat(pieces, 1)
            self.stats["target_replay_tokens"] += accepted_steps
        if accepted_steps < self.proposal.shape[1]:
            self.draft.restore(self.saved_draft)
            self._append_context(committed, hidden, self.past, prefill=False)
        else:
            self.draft.step(
                self.model.embed_tokens(committed[:, -1:]),
                hidden[:, -2:-1],
                position_ids=self._positions(self.past + accepted_steps - 1, 1),
            )
            self.hidden = hidden[:, -1:].clone()
        self.stats["target_verify_calls"] += 1
        self.stats["draft_tokens_total"] += ids.shape[1] - 1
        self.stats["accepted_drafts_total"] += accepted_steps - 1
        self.stats["rejected_drafts_total"] += ids.shape[1] - accepted_steps
        self.past += accepted_steps
        self.pending = self.proposal = self.saved_draft = None

    def abort(self):
        if self.pending is not None:
            self._restore(self.pending[0])
        if self.saved_draft is not None:
            self.draft.restore(self.saved_draft)
        self.pending = self.proposal = self.saved_draft = None

    def release(self, req_id):
        if self.owner == req_id:
            self.abort()
            self.owner, self.past, self.hidden = None, 0, None


class Qwen4MTPProposer(SerialHMONNXProposer):
    def __init__(self, vllm_config):
        self.config = vllm_config
        self.num_draft_tokens = int(vllm_config.speculative_config.num_speculative_tokens)
        self.runner = self.state = None

    def attach_merak_runner(self, runner):
        self.runner = runner
        self.state = Qwen4ServingState(runner.model.inference_model)
        capacity = int(self.state.model.meta_info.spec_decode.num_draft_tokens)
        if not 1 <= self.num_draft_tokens <= capacity:
            raise ValueError("Qwen4 draft count exceeds the exported graph capacity")

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, *, slot_mappings=None):
        req_ids = list(self.runner.input_batch.req_ids)
        if len(req_ids) != 1 or len(sampled_token_ids) != 1:
            raise RuntimeError("Qwen4 MTP requires one sampled request")
        if req_ids[0] not in self.runner.input_batch.greedy_reqs:
            raise ValueError("Qwen4 MTP currently supports greedy sampling")
        valid = [int(t) for t in sampled_token_ids[0] if int(t) != -1]
        return [self.state.propose(req_ids[0], valid[-1], count=self.num_draft_tokens)] if valid else [[]]
