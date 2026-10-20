"""Host-side controls for Qwen4's contiguous-prefix sparse attention."""

from __future__ import annotations

import torch

from ...sparse_attention_controls import build_sparse_attention_controls


QSA_HOST_CONTROL_NAMES = (
    "qsa_pool_offset",
    "qsa_pool_new_count",
    "qsa_pool_write_start",
    "qsa_query_valid",
    "qsa_attention_mask",
)


def build_qsa_host_controls(
    past_length: int,
    current_length: int,
    input_length: int,
    context_length: int,
    candidate_topk: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Prepare the same pool, tail and mask controls used by GLM-5.3."""
    controls = build_sparse_attention_controls(
        past_length, current_length, input_length, context_length, candidate_topk,
        device=device, dtype=dtype,
    )
    return tuple(controls[name] for name in (
        "pool_offset", "pool_new_count", "pool_write_start", "query_validity", "attention_mask",
    ))


def qsa_host_control_spec(session) -> tuple[int, torch.dtype] | None:
    """Read the optional control ABI from the active serving graph."""
    names = set(session.get_input_names())
    present = names.intersection(QSA_HOST_CONTROL_NAMES)
    if not present:
        return None
    if present != set(QSA_HOST_CONTROL_NAMES):
        raise ValueError("QSA graph has an incomplete host-control ABI")
    info = session.get_input("qsa_attention_mask")
    width = int(info.shape[-1])
    if width < 7 or (width - 3) % 4:
        raise ValueError("QSA attention mask width must contain C4 blocks and three tail slots")
    return (width - 3) // 4, info.dtype
