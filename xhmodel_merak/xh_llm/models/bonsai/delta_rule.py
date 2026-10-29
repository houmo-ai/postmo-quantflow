"""Bonsai's FP32 delta-rule normalization, isolated from ordinary Qwen."""

import torch

from ..qwen3_5 import _delta_rule as qwen_delta


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _run_delta_rule(rule, query, key, value, **kwargs):
    dtype = query.dtype
    query, key, value = query.float(), key.float(), value.float()
    if kwargs.pop("use_qk_l2norm_in_kernel", False):
        query, key = l2norm(query), l2norm(key)
    for name in (
        "g", "beta", "initial_state", "mask_qkv", "chunk_eye_matrix",
        "chunk_eye_8_batched", "chunk_mask_incl", "chunk_mask_strict",
        "cumsum_matrix", "chunk_row_masks",
    ):
        if kwargs.get(name) is not None:
            kwargs[name] = kwargs[name].float()
    output, state = rule(query, key, value, use_qk_l2norm_in_kernel=False, **kwargs)
    return output.to(dtype), state


def torch_chunk_gated_delta_rule(query, key, value, **kwargs):
    return _run_delta_rule(qwen_delta.torch_chunk_gated_delta_rule, query, key, value, **kwargs)


def torch_recurrent_gated_delta_rule(query, key, value, **kwargs):
    return _run_delta_rule(qwen_delta.torch_recurrent_gated_delta_rule, query, key, value, **kwargs)
