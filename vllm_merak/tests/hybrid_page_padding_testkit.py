"""Test-only oracles for hybrid KV physical-storage separation.

The production implementation deliberately does not import this module.  It
keeps independent byte arithmetic and recognizable state fixtures in one
place so allocator, planner, and lifecycle tests share exact expectations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import torch


FULL_BLOCK_SIZE = 64
LINEAR_PAGE_BYTES = 1_114_112
PHASE1_MAX_NUM_SEQS = 1
PHASE1_LINEAR_GROUPS = 3
PHASE1_RECOMMENDED_BLOCKS = 4_304


@dataclass(frozen=True)
class HybridModelGeometry:
    name: str
    full_layers: int
    linear_layers: int
    hmfp_page_bytes: int
    linear_page_bytes: int = LINEAR_PAGE_BYTES


QWEN35_9B = HybridModelGeometry(
    name="qwen3.5-9b",
    full_layers=8,
    linear_layers=24,
    hmfp_page_bytes=133_120,
)
QWEN36_35B = HybridModelGeometry(
    name="qwen3.6-35b-a3b",
    full_layers=10,
    linear_layers=30,
    hmfp_page_bytes=66_560,
)


def required_blocks(
    max_model_len: int,
    block_size: int = FULL_BLOCK_SIZE,
    linear_groups: int = PHASE1_LINEAR_GROUPS,
) -> int:
    """Return Full token blocks plus Phase-1 Mamba group demand."""
    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if linear_groups < 0:
        raise ValueError("linear_groups must be non-negative")
    return ceil(max_model_len / block_size) + linear_groups


def recommended_blocks(minimum_blocks: int, reserve_ratio: float = 0.05) -> int:
    if minimum_blocks <= 0:
        raise ValueError("minimum_blocks must be positive")
    if reserve_ratio < 0:
        raise ValueError("reserve_ratio must be non-negative")
    return ceil(minimum_blocks * (1.0 + reserve_ratio))


def virtual_scheduler_bytes(
    geometry: HybridModelGeometry,
    num_blocks: int,
) -> int:
    """Virtual bytes vLLM consumes to derive the global scalar block count.

    Hybrid grouping has one slot per Full layer.  The four logical cache
    groups reuse those slots, so the virtual bytes per global block are the
    unified page size times the Full-layer count, not all model layers.
    """
    return num_blocks * geometry.full_layers * geometry.linear_page_bytes


def exact_hmfp_bytes(geometry: HybridModelGeometry, num_blocks: int) -> int:
    return num_blocks * geometry.full_layers * geometry.hmfp_page_bytes


def phase1_linear_bytes(geometry: HybridModelGeometry) -> int:
    """One complete sparse checkpoint per Linear layer in serial Phase 1."""
    return geometry.linear_layers * geometry.linear_page_bytes


def phase1_physical_bytes(geometry: HybridModelGeometry, num_blocks: int) -> int:
    return exact_hmfp_bytes(geometry, num_blocks) + phase1_linear_bytes(geometry)


LayerState = tuple[torch.Tensor, ...]


def recognizable_states(value: int) -> LayerState:
    """Four tensors matching the split conv-q/k/v/recurrent state contract."""
    return (
        torch.full((2, 2), value, dtype=torch.float16),
        torch.full((2, 2), value + 1, dtype=torch.float16),
        torch.full((4, 2), value + 2, dtype=torch.float16),
        torch.full((2, 2, 2), value + 3, dtype=torch.float32),
    )


def assert_states_exact(
    actual: LayerState,
    expected: LayerState,
) -> None:
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor.dtype == expected_tensor.dtype
        assert actual_tensor.shape == expected_tensor.shape
        assert torch.equal(actual_tensor, expected_tensor)
