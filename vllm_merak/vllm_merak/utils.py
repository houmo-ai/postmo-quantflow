# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Merak vLLM backend."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from .constants import (
    MERAK_AUTO_OFFLOAD,
    MERAK_CUDA_GRAPH,
    MERAK_ENABLE_PAGE_ATTENTION,
    MERAK_PARALLEL_LINEAR_FUSION,
    MERAK_USE_V2,
)

if TYPE_CHECKING:
    from .config import MerakConfig


_CACHE_DTYPES: dict[str, torch.dtype] = {
    "u8": torch.uint8,
    "i8": torch.int8,
    "int8": torch.int8,
    "i16": torch.int16,
    "int16": torch.int16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "hmbf16": torch.int8,
    "f32": torch.float32,
    "fp32": torch.float32,
    "float32": torch.float32,
    "auto": torch.float16,
}


def resolve_torch_dtype(dtype_name: str | None) -> torch.dtype:
    """Map a Merak/HMONNX dtype string to a torch dtype with float16 fallback."""
    if dtype_name is None:
        return torch.float16
    normalized = dtype_name.lower()
    if normalized == "hmbf16":
        return torch.int8
    if normalized.startswith("sefp"):
        normalized = normalized.replace("sefp", "", 1).lstrip("_")
        if "kint16" in normalized or "vint16" in normalized:
            return torch.int16
        return torch.int8
    cached = _CACHE_DTYPES.get(normalized)
    if cached is not None:
        return cached
    return getattr(torch, normalized, torch.float16)


def env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _flag_enabled(config: "MerakConfig | None", attr_name: str, env_name: str) -> bool:
    if config is not None:
        value = getattr(config, attr_name, None)
        if value is not None:
            return bool(value)
    return env_flag_enabled(env_name)


def _use_v2_enabled(config: "MerakConfig | None" = None) -> bool:
    """Resolve V2 from Merak's flag or the underlying HMONNX flag."""

    if config is not None:
        value = getattr(config, "use_v2", None)
        if value is not None:
            return bool(value)
    if "ENABLE_HMINFERENCE_V2" in os.environ:
        return env_flag_enabled("ENABLE_HMINFERENCE_V2")
    return env_flag_enabled(MERAK_USE_V2) or env_flag_enabled(
        "ENABLE_HMINFERENCE_V2"
    )


def get_hmonnx_runtime_kwargs(config: "MerakConfig | None" = None) -> dict[str, bool | list[int]]:
    kwargs: dict[str, bool | list[int]] = {}
    if _flag_enabled(config, "cuda_graph", MERAK_CUDA_GRAPH):
        kwargs["enable_cuda_graph"] = True
    if _use_v2_enabled(config):
        os.environ["ENABLE_HMINFERENCE_V2"] = "1"
        fusion_value = os.environ.get(MERAK_PARALLEL_LINEAR_FUSION)
        kwargs["enable_parallel_linear_fusion"] = (
            True if fusion_value is None else env_flag_enabled(MERAK_PARALLEL_LINEAR_FUSION)
        )
        if not _flag_enabled(config, "auto_offload", MERAK_AUTO_OFFLOAD) and torch.cuda.is_available():
            # V2 without auto-offload must keep prefill/decode and their cache
            # tensors on one device.  Passing all visible GPUs here lets the
            # runtime choose different devices as memory availability changes,
            # which breaks cross-graph tensor/index contracts.  Multi-GPU V2
            # placement is opt-in through auto_offload below.
            kwargs["device_map"] = [torch.cuda.current_device()]
    if _flag_enabled(config, "auto_offload", MERAK_AUTO_OFFLOAD):
        kwargs["enable_auto_offload"] = True
    if (
        _use_v2_enabled(config)
        and _flag_enabled(config, "auto_offload", MERAK_AUTO_OFFLOAD)
        and _flag_enabled(config, "cuda_graph", MERAK_CUDA_GRAPH)
    ):
        # Make the runtime device set explicit while preserving all visible GPUs.
        kwargs["device_map"] = list(range(torch.cuda.device_count()))
    if _flag_enabled(config, "enable_page_attention", MERAK_ENABLE_PAGE_ATTENTION):
        kwargs["enable_page_attention"] = True
    return kwargs


def should_preserve_hmonnx_runtime_device_map(config: "MerakConfig | None" = None) -> bool:
    return _use_v2_enabled(config) and _flag_enabled(config, "auto_offload", MERAK_AUTO_OFFLOAD)


def should_enable_page_attention(config: "MerakConfig | None" = None) -> bool:
    if config is not None:
        value = getattr(config, "enable_page_attention", None)
        if value is not None:
            return bool(value)
    return env_flag_enabled(MERAK_ENABLE_PAGE_ATTENTION)


def resolve_hmonnx_primary_device(hmonnx_model: Any, fallback: torch.device) -> torch.device:
    runtime_device = getattr(hmonnx_model, "device", None)
    if runtime_device is None:
        return fallback
    return torch.device(runtime_device)


def move_hmonnx_auxiliary_tensors(hmonnx_model: Any, device: torch.device) -> None:
    embed_tokens = getattr(hmonnx_model, "embed_tokens", None)
    embed_tokens_to = getattr(embed_tokens, "to", None)
    if callable(embed_tokens_to):
        embed_tokens_to(device)

    get_kvcache_mixin = getattr(hmonnx_model, "get_kvcache_mixin", None)
    if callable(get_kvcache_mixin):
        kvcache_mixin = get_kvcache_mixin()
        kvcache_mixin_to = getattr(kvcache_mixin, "to", None)
        if callable(kvcache_mixin_to):
            kvcache_mixin_to(device)

    data_processor = getattr(hmonnx_model, "_data_processor", None)
    data_processor_to = getattr(data_processor, "to", None)
    if callable(data_processor_to):
        data_processor_to(device)

    preprocessor_embed_tokens = getattr(data_processor, "embed_tokens", None)
    preprocessor_embed_tokens_to = getattr(preprocessor_embed_tokens, "to", None)
    if callable(preprocessor_embed_tokens_to):
        preprocessor_embed_tokens_to(device)


def apply_hmonnx_runtime_options(hmonnx_model: Any, config: "MerakConfig | None" = None) -> None:
    if not _flag_enabled(config, "auto_offload", MERAK_AUTO_OFFLOAD):
        return
    if _use_v2_enabled(config):
        return
    enable_auto_offload = getattr(hmonnx_model, "enable_auto_offload", None)
    if callable(enable_auto_offload):
        enable_auto_offload()
