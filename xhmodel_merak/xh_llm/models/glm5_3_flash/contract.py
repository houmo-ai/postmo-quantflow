# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Host controls and cache layout for the released GLM-5.3-Flash text model."""

from dataclasses import dataclass

import torch


LINEAR = "linear_attention"
DSA = "deepseek_sparse_attention"
CONTROL_NAMES = (
    "past_seq_length",
    "current_input_length",
    "last_token_index",
    "pool_offset",
    "pool_new_count",
    "pool_write_start",
    "query_validity",
    "pool_validity",
    "tail_indices",
    "tail_validity",
)


@dataclass(frozen=True)
class Glm53CacheSpec:
    layer_types: tuple[str, ...]
    max_context_length: int
    linear_heads: int
    linear_dim: int
    conv_size: int
    latent_dim: int
    index_dim: int
    pool_size: int = 4
    index_topk: int = 2048

    def __post_init__(self):
        if not self.layer_types or any(k not in (LINEAR, DSA) for k in self.layer_types):
            raise ValueError("GLM cache requires KDA/DSA layer types")
        for name in (
            "max_context_length",
            "linear_heads",
            "linear_dim",
            "conv_size",
            "latent_dim",
            "index_dim",
            "pool_size",
            "index_topk",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pool_size != 4 or self.index_topk % self.pool_size:
            raise ValueError("GLM-5.3 requires C4 and an index_topk divisible by four")
        if self.max_context_length < self.pool_size:
            raise ValueError("context capacity must contain at least one complete pool")

    @classmethod
    def from_config(cls, config, max_context_length, max_layers=None):
        c = config.get_text_config() if hasattr(config, "get_text_config") else config
        count = c.num_hidden_layers if max_layers is None else int(max_layers)
        if not 1 <= count <= c.num_hidden_layers:
            raise ValueError("max_layers is outside the checkpoint layer range")
        if c.qk_rope_head_dim != 0 or not c.mla_use_nope or not c.mhc:
            raise ValueError("GLM-5.3 static graph requires NoPE MLA and mHC")
        if not c.index_kpool_compress or not c.index_kpool_always_select_tail:
            raise ValueError("GLM-5.3 requires learned k-pool compression and visible tails")
        if any(t != "full" for t in c.indexer_types[:count]):
            raise ValueError("shared indexers are outside the released GLM-5.3 contract")
        if max_context_length > c.max_position_embeddings:
            raise ValueError("context capacity exceeds checkpoint maximum")
        return cls(
            tuple(c.layer_types[:count]),
            int(max_context_length),
            c.linear_num_heads,
            c.linear_head_dim,
            c.linear_conv_kernel_dim,
            c.kv_lora_rank,
            c.index_head_dim,
            c.index_kpool,
            c.index_topk,
        )

    @property
    def pool_capacity(self):
        return self.max_context_length // self.pool_size

    @property
    def selected_pools(self):
        return min(self.index_topk // self.pool_size, self.pool_capacity)

    def cache_shapes(self, kind):
        if kind == LINEAR:
            return (
                (1, 3 * self.linear_heads * self.linear_dim, self.conv_size),
                (1, self.linear_heads, self.linear_dim, self.linear_dim),
            )
        return (
            (1, 1, self.max_context_length, self.latent_dim),
            (1, 1, self.pool_capacity, self.index_dim),
            (1, self.pool_size, self.index_dim),
            (1, self.pool_size, self.index_dim),
        )

    @staticmethod
    def cache_suffixes(kind):
        return ("conv", "recurrent") if kind == LINEAR else ("latent", "pool", "index_key", "index_score")

    def initial_caches(self, *, device="cpu", dtype=torch.float32):
        from xhquant.core import CacheTensor

        result = []
        for kind in self.layer_types:
            state = [torch.zeros(s, device=device, dtype=dtype) for s in self.cache_shapes(kind)]
            if kind == DSA:
                state[-1].fill_(-32768.0)
                state[0], state[1] = CacheTensor(state[0]), CacheTensor(state[1])
            result.extend(state)
        return tuple(result)

    def stage_caches(self, caches):
        """LLMCache writes in place; stage its two persistent stores before a call."""
        from xhquant.core import CacheTensor

        staged = list(caches)
        cursor = 0
        for kind in self.layer_types:
            if kind == DSA:
                for i in (cursor, cursor + 1):
                    staged[i] = CacheTensor(caches[i].data.clone())
            cursor += len(self.cache_suffixes(kind))
        return tuple(staged)

    def input_names(self):
        return [
            "inputs_embeds",
            *CONTROL_NAMES,
            *(
                f"layer_{i}_{suffix}_input"
                for i, kind in enumerate(self.layer_types)
                for suffix in self.cache_suffixes(kind)
            ),
        ]

    def output_names(self):
        return [
            "logits",
            *(
                f"layer_{i}_{suffix}_output"
                for i, kind in enumerate(self.layer_types)
                for suffix in self.cache_suffixes(kind)
            ),
        ]


def build_transition(spec, past_length, current_length, input_length, *, device="cpu", dtype=torch.float32):
    """One descriptor from committed length L and real length n; right padding only."""
    if past_length < 0 or not 0 < current_length <= input_length:
        raise ValueError("require L >= 0 and 0 < n <= fixed input length")
    if past_length + current_length > spec.max_context_length:
        raise ValueError("request exceeds the configured context capacity")
    r = spec.pool_size
    query_slots = torch.arange(input_length, device=device)
    real = query_slots < current_length
    visible = past_length + query_slots + 1
    complete = visible // r
    pool_valid = (torch.arange(spec.pool_capacity, device=device)[None, :] < complete[:, None]) & real[:, None]
    tail_slots = torch.arange(r - 1, device=device)
    tail_valid = (tail_slots[None, :] < (visible % r)[:, None]) & real[:, None]
    tail = complete[:, None] * r + tail_slots[None, :]
    tail = torch.where(tail_valid, tail, 0).to(torch.int32)

    def scalar(value):
        return torch.tensor([value], device=device, dtype=torch.int32)

    values = (
        scalar(past_length),
        scalar(current_length),
        scalar(current_length - 1),
        scalar(r - past_length % r),
        scalar((past_length % r + current_length) // r),
        scalar(past_length // r),
        real[None].to(dtype),
        pool_valid[None].to(dtype),
        tail[None],
        tail_valid[None].to(dtype),
    )
    return dict(zip(CONTROL_NAMES, values, strict=True))
