# SPDX-License-Identifier: Apache-2.0
"""Merak linear-attention state placeholder + custom SSM backend.

Qwen3.5 exports a split GatedDeltaNet linear-attention cache (conv_q, conv_k,
conv_v, recurrent). The HMONNX runtime owns the real compute; vLLM only needs
to *allocate* the per-layer state so request scheduling / preemption / resume
can be tracked through the official hybrid-cache machinery (``MambaSpec``).

This module provides:

* :class:`MerakLinearStateBackend` - a minimal SSM ``AttentionBackend`` so
  vLLM's ``_find_non_ssm_backend`` skips these layers and ``get_mamba_attn_backend``
  resolves without importing the CUDA-graph GDN builder.
* :class:`MerakLinearStatePlaceholder` - a :class:`MambaBase` subclass that
  registers itself in the static forward context and exposes the split
  state shapes/dtypes so vLLM emits one ``MambaSpec`` per linear layer.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadataBuilder
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
    register_backend,
)

logger = init_logger(__name__)


class _MerakLinearStateMetadataBuilder(AttentionMetadataBuilder):
    """No-op metadata builder.

    Merak drives the HMONNX runtime directly from ``MerakModelRunnerV1``; the
    linear-state layers never run vLLM attention kernels, so the builder only
    needs to satisfy the abstract interface and never actually build usable
    metadata.
    """

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self.reorder_batch_threshold = None

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        del common_prefix_len, common_attn_metadata, fast_build
        return None


class MerakLinearStateBackend(AttentionBackend):
    """Minimal SSM backend for Merak linear-attention placeholder layers."""

    @staticmethod
    def get_name() -> str:
        return "MERAK_LINEAR_STATE"

    @classmethod
    def is_ssm(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "MerakLinearStateBackend is a state-holder only; it has no attention impl."
        )

    @staticmethod
    def get_builder_cls():
        return _MerakLinearStateMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, *args, **kwargs):
        # MambaSpec-managed layers do not use this attention-style shape; the
        # real per-state shapes come from the placeholder's get_state_shape().
        del block_size, num_kv_heads, head_size, args, kwargs
        return (num_blocks,)


# Register the Merak SSM backend under the CUSTOM mamba enum slot so
# ``MambaBase.get_attn_backend`` -> ``get_mamba_attn_backend`` can resolve it
# without importing GDN's CUDA-graph builder on the Merak CPU platform.
register_backend(
    MambaAttentionBackendEnum.CUSTOM,
    f"{MerakLinearStateBackend.__module__}.{MerakLinearStateBackend.__qualname__}",
    is_mamba=True,
)


class MerakLinearStatePlaceholder(MambaBase, torch.nn.Module):
    """State-holder layer exposing a split-linear ``MambaSpec`` to vLLM.

    One instance per Qwen3.5 linear-attention layer. It performs no compute;
    vLLM's cache engine allocates ``kv_cache`` tensors matching
    ``get_state_shape()`` and attaches them here, which the runner then bridges
    to the HMONNX runtime's split conv/recurrent caches.
    """

    def __init__(
        self,
        shapes: tuple[tuple[int, ...], ...],
        dtypes: tuple[torch.dtype, ...],
        prefix: str,
    ) -> None:
        super().__init__()
        self._shapes = tuple(tuple(int(d) for d in shape) for shape in shapes)
        self._dtypes = tuple(dtypes)
        self.prefix = prefix
        self.layer_name = prefix
        # Populated by vLLM's initialize_kv_cache_tensors (bind_kv_cache).
        self.kv_cache = (torch.tensor([]),)

        vllm_config = get_current_vllm_config()
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_state_shape(self) -> Iterable[tuple[int, ...]]:
        return self._shapes

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return self._dtypes

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.CUSTOM

    def get_attn_backend(self) -> type[AttentionBackend]:
        return MerakLinearStateBackend

    def forward(self, *args, **kwargs):  # pragma: no cover - never invoked
        raise RuntimeError(
            "MerakLinearStatePlaceholder is a state holder and must not be called."
        )


def compute_merak_linear_state_shapes(
    linear_cfg: dict,
) -> tuple[tuple[int, ...], ...]:
    """Derive the four split GatedDeltaNet state shapes (excl. block dim).

    Layout (from HMONNX split_conv_cache export):
        value_dim = num_v_heads * head_v_dim
        key_dim   = (conv_dim - value_dim) // 2
        conv_q:    (key_dim,   conv_kernel_size)
        conv_k:    (key_dim,   conv_kernel_size)
        conv_v:    (value_dim, conv_kernel_size)
        recurrent: (num_v_heads, head_k_dim, head_v_dim)
    """
    conv_dim = int(linear_cfg["conv_dim"])
    conv_kernel_size = int(linear_cfg["conv_kernel_size"])
    num_v_heads = int(linear_cfg["num_v_heads"])
    head_k_dim = int(linear_cfg["head_k_dim"])
    head_v_dim = int(linear_cfg["head_v_dim"])

    value_dim = num_v_heads * head_v_dim
    key_dim = (conv_dim - value_dim) // 2

    return (
        (key_dim, conv_kernel_size),
        (key_dim, conv_kernel_size),
        (value_dim, conv_kernel_size),
        (num_v_heads, head_k_dim, head_v_dim),
    )


def get_merak_linear_kv_cache_config(vllm_config: VllmConfig) -> dict | None:
    """Fetch the linear_kv_cache_config dict off the MerakConfig, if present."""
    hf_config = vllm_config.model_config.hf_config
    linear_cfg = getattr(hf_config, "linear_kv_cache_config", None)
    if isinstance(linear_cfg, dict):
        return linear_cfg
    return None
