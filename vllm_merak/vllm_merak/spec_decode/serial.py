# SPDX-License-Identifier: Apache-2.0
"""Scheduler bridge for a runtime that owns a single request's cache."""

import torch


class SerialHMONNXProposer:
    def forward_target(self, *, input_ids, positions, request_id, **kwargs):
        if kwargs.get("inputs_embeds") is not None:
            raise ValueError("Serial speculative serving requires token IDs")
        model = self.runner.model
        flat = model._flatten_input_ids(input_ids)
        metadata = self.runner._get_attn_metadata()
        if metadata is None:
            past = int(positions.reshape(-1)[0]) if positions is not None else 0
            start, end = 0, flat.numel()
        else:
            if metadata.past_lens.numel() != 1:
                raise RuntimeError("Serial speculative serving requires one scheduled request")
            past = int(metadata.past_lens[0])
            start, end = map(int, metadata.subsequence_begins.tolist())
        phase = (
            "verify"
            if request_id in self.runner._merak_verify_req_ids
            else "prefill"
            if past == 0 or end - start > 1
            else "decode"
        )
        logits = self.state.forward_target(request_id, flat[start:end], past=past, phase=phase)
        output = torch.zeros(flat.numel(), logits.shape[-1], device=self.runner.device, dtype=logits.dtype)
        output[start:end].copy_(logits[0].to(output))
        model._cache_owner_request_id = request_id
        model._cache_owner_next_past_len = self.state.past
        if phase == "verify":
            self.runner._merak_pending_verify_req_ids.add(request_id)
        return output

    def on_target_accept(self, *, req_id, accepted_steps, sampled_token_ids):
        self.state.commit(req_id, accepted_steps, torch.as_tensor(sampled_token_ids))
        self.runner.model._cache_owner_next_past_len = self.state.past

    def release_requests(self, req_ids):
        for req_id in req_ids:
            if self.state is not None:
                self.state.release(req_id)

    def release_stale_requests(self, active_req_ids):
        if self.state is not None and self.state.owner not in active_req_ids:
            self.release_requests([self.state.owner] if self.state.owner else [])

    def abort_pending_verifies(self):
        if self.state is not None:
            self.state.abort()
