"""Shared cache-shape and ownership contracts for hybrid HMONNX models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...constants import (
    HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST,
    HYBRID_CACHE_POLICY_PAGED,
    SUPPORTED_HYBRID_CACHE_POLICIES,
)


def _positive_shape(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a four-dimensional cache shape")
    if len(value) != 4:
        raise ValueError(f"{field} must have shape [B, H, S, D], got {value!r}")
    if any(type(dimension) is not int or dimension <= 0 for dimension in value):
        raise ValueError(f"{field} must contain positive integers, got {value!r}")
    return tuple(value)


@dataclass(frozen=True)
class FullAttentionCacheShapes:
    key: tuple[int, ...]
    value: tuple[int, ...]
    cache_axis: int

    @property
    def max_cache_tokens(self) -> int:
        return self.key[self.cache_axis]

    @property
    def symmetric(self) -> bool:
        return self.key == self.value


def parse_full_attention_cache_shapes(
    kv_cache_meta: Mapping[str, Any],
) -> FullAttentionCacheShapes:
    """Parse both legacy symmetric and explicit ``[K_shape, V_shape]`` ABIs."""
    raw_shape = kv_cache_meta.get("kv_cache_shape")
    if (
        isinstance(raw_shape, Sequence)
        and not isinstance(raw_shape, (str, bytes))
        and len(raw_shape) == 2
        and all(
            isinstance(shape, Sequence) and not isinstance(shape, (str, bytes))
            for shape in raw_shape
        )
    ):
        key_shape = _positive_shape(raw_shape[0], field="kv_cache_shape[0]")
        value_shape = _positive_shape(raw_shape[1], field="kv_cache_shape[1]")
    else:
        key_shape = _positive_shape(raw_shape, field="kv_cache_shape")
        value_shape = key_shape

    cache_axis = kv_cache_meta.get("cache_axis", 2)
    if type(cache_axis) is not int or not 0 <= cache_axis < len(key_shape):
        raise ValueError(
            f"kv_cache.cache_axis must index the cache shape, got {cache_axis!r}"
        )
    if key_shape[0] != value_shape[0]:
        raise ValueError(
            "K/V cache batch dimensions must match: "
            f"K={key_shape}, V={value_shape}"
        )
    if key_shape[cache_axis] != value_shape[cache_axis]:
        raise ValueError(
            "K/V cache sequence capacities must match: "
            f"K={key_shape}, V={value_shape}, cache_axis={cache_axis}"
        )

    return FullAttentionCacheShapes(
        key=key_shape,
        value=value_shape,
        cache_axis=cache_axis,
    )


def resolve_hybrid_page_attention(
    *, cache_policy: str, requested: bool
) -> bool:
    if cache_policy not in SUPPORTED_HYBRID_CACHE_POLICIES:
        raise ValueError(f"Unsupported Merak hybrid cache policy {cache_policy!r}")
    if cache_policy == HYBRID_CACHE_POLICY_PAGED:
        return True
    if cache_policy == HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST:
        if requested:
            raise ValueError(
                "This Merak hybrid model owns an asymmetric contiguous HMONNX "
                "K/V cache and does not support PageAttention"
            )
        return False
    raise AssertionError(f"Unhandled Merak hybrid cache policy {cache_policy!r}")
