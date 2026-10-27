"""MiniCPM speculative loop.

The shared Qwen loop stops before emitting a recovery EOS. This copy emits that
token first and still commits only the anchor cache snapshot.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch

from ..qwen3_5._spec_decode_shared import SpecDecodeVerifyResult


def run_spec_decode_loop(
    *,
    max_new_tokens: int,
    initial_token_id: int,
    initial_past_seq_len: int,
    initial_last_hidden: Optional[torch.Tensor],
    initial_mtp_past_seq_len: int,
    eos_token_id: Optional[int],
    num_drafts: int,
    run_draft,
    verify_round,
    apply_verify_result,
    post_verify,
    on_token=None,
) -> Tuple[List[int], Dict[str, Union[int, float, List[int]]]]:
    generated_ids: List[int] = []

    def emit(token_id: int) -> None:
        generated_ids.append(token_id)
        if on_token is not None:
            on_token(token_id)

    current_token_id = initial_token_id
    past_seq_len = initial_past_seq_len
    last_hidden = initial_last_hidden
    mtp_past_seq_len = initial_mtp_past_seq_len
    emit(current_token_id)
    if len(generated_ids) >= max_new_tokens or (eos_token_id is not None and current_token_id == eos_token_id):
        return generated_ids, _stats(0, generated_ids, 0, 0, num_drafts, [])

    total_rounds = 0
    total_draft_tokens = 0
    total_accepted_tokens = 0
    accepted_per_round: List[int] = []
    while len(generated_ids) < max_new_tokens:
        total_rounds += 1
        draft_token_ids = run_draft(current_token_id, last_hidden, past_seq_len, mtp_past_seq_len, num_drafts)
        total_draft_tokens += len(draft_token_ids)
        verify_result = verify_round(current_token_id, draft_token_ids, past_seq_len)
        accepted_count = 0
        for predicted, drafted in zip(verify_result.predicted_token_ids, draft_token_ids, strict=False):
            if predicted != drafted:
                break
            accepted_count += 1
        total_accepted_tokens += accepted_count
        accepted_per_round.append(accepted_count)
        accepted_steps = accepted_count + 1
        apply_verify_result(verify_result, accepted_steps)
        past_seq_len = verify_result.initial_seq_len + accepted_steps
        stop = False
        for token_id in draft_token_ids[:accepted_count]:
            emit(token_id)
            if (eos_token_id is not None and token_id == eos_token_id) or len(generated_ids) >= max_new_tokens:
                stop = True
                break
        if stop:
            break
        next_token_id = verify_result.predicted_token_ids[
            accepted_count if accepted_count < len(draft_token_ids) else len(draft_token_ids)
        ]
        if eos_token_id is not None and next_token_id == eos_token_id:
            emit(next_token_id)
            break
        last_hidden, mtp_past_seq_len = post_verify(
            verify_result, accepted_count, accepted_steps, next_token_id, mtp_past_seq_len,
        )
        current_token_id = next_token_id
        emit(current_token_id)
    return generated_ids, _stats(
        total_rounds, generated_ids, total_draft_tokens, total_accepted_tokens, num_drafts, accepted_per_round,
    )


def _stats(rounds, generated, drafts, accepted, capacity, per_round):
    return {
        "num_rounds": rounds,
        "output_tokens": len(generated),
        "draft_tokens_total": drafts,
        "accepted_drafts_total": accepted,
        "avg_accepted_per_round": accepted / rounds if rounds else 0.0,
        "draft_capacity_per_round": capacity,
        "accepted_drafts_per_round": per_round,
    }


__all__ = ["SpecDecodeVerifyResult", "run_spec_decode_loop"]
