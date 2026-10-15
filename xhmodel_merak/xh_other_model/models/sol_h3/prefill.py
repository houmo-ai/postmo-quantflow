# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Collect every feature row from cached prefill, with request-owned caches."""

import torch
from torch.nn import functional as F


def prefill_feature_graphs(hidden, cos, sin, additions, *, metadata, invoke, close, device):
    """Replay the published language groups for runtime or export calibration."""
    if not 0 < hidden.shape[1] <= metadata["text_capacity"]:
        raise ValueError("Qwen presentation exceeds the exported text capacity")
    for name in metadata["feature_graphs"]:
        start, end = metadata["graphs"][name]["text_layers"]
        try:
            hidden = prefill_feature_group(
                lambda *values, graph=name: invoke(graph, *values)[0],
                hidden,
                cos,
                sin,
                additions,
                chunk_length=metadata["text_prefill_chunk_length"],
                cache_shape=metadata["text_cache_shape"],
                num_layers=end - start,
                device=device,
            )
        finally:
            close()
    return hidden


def prefill_feature_group(
    invoke, hidden, cos, sin, additions, *, chunk_length, cache_shape, num_layers, device, cache_dtype=torch.float16
):
    """Execute all chunks of one layer group before releasing its weights.

    ``invoke`` accepts the flattened graph ABI and returns its feature tensor.
    K/V caches are fresh for each invocation of this function and remain on the
    execution device throughout the group. Padding is never returned as a feature.
    """
    from xhmodel_merak.exec_backends import CacheTensor

    if type(chunk_length) is not int or chunk_length <= 0 or type(num_layers) is not int or num_layers <= 0:
        raise ValueError("prefill chunk length and layer count must be positive integers")
    if (
        len(cache_shape) != 4
        or any(type(n) is not int or n <= 0 for n in cache_shape)
        or cache_shape[0] != 1
        or cache_shape[2] % chunk_length
    ):
        raise ValueError("prefill cache must be [1, KV heads, chunk-aligned capacity, head dimension]")
    if hidden.ndim != 3 or hidden.shape[0] != 1 or not 0 < hidden.shape[1] <= cache_shape[2]:
        raise ValueError("prefill features must be a nonempty batch-one sequence within cache capacity")
    if (
        cos.shape != sin.shape
        or cos.shape != (1, hidden.shape[1], cache_shape[-1])
        or len(additions) != 3
        or any(value.shape != hidden.shape for value in additions)
    ):
        raise ValueError("RoPE and all three dense DeepStack features must cover the input sequence")
    keys = [CacheTensor(torch.zeros(cache_shape, device=device, dtype=cache_dtype)) for _ in range(num_layers)]
    values = [CacheTensor(torch.zeros(cache_shape, device=device, dtype=cache_dtype)) for _ in range(num_layers)]
    results = []
    for start in range(0, hidden.shape[1], chunk_length):
        length = min(chunk_length, hidden.shape[1] - start)

        def chunk(value, start=start, length=length):
            return F.pad(value[:, start : start + length], (0, 0, 0, chunk_length - length))

        result = invoke(
            chunk(hidden),
            chunk(cos),
            chunk(sin),
            torch.tensor([length], dtype=torch.int32),
            torch.tensor([start], dtype=torch.int32),
            *(chunk(value) for value in additions),
            *keys,
            *values,
        )
        if result.shape != (1, chunk_length, hidden.shape[2]):
            raise ValueError("prefill graph must return every row of the current chunk")
        results.append(result[:, :length].detach().cpu())
    return torch.cat(results, dim=1)
