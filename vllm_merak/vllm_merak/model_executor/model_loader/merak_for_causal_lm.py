# SPDX-License-Identifier: Apache-2.0
"""Base class for Merak vLLM causal-LM models.

Implements the shared scheduling logic (paged ↔ contiguous KV cache copies,
HMONNX runtime invocation, attention metadata extraction) that both text and
multimodal Merak models depend on. Subclasses customise ``__init__`` (loading
the runtime, parsing arch-specific metadata) and override the multimodal
helpers when needed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

import torch
import torch.nn as nn
from torch import Tensor
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowSpec
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from ...attention.backends.merak import (
    MerakAttentionBackend,
    MerakAttentionMetadata,
    MerakPageAttentionGroupContext,
)
from ...config import MERAK_PREFERRED_BLOCK_SIZE
from ...determinism_trace import (
    emit_event as emit_determinism_event,
)
from ...determinism_trace import (
    fingerprint as determinism_fingerprint,
)
from ...determinism_trace import (
    fingerprint_named_tensors as determinism_fingerprint_named_tensors,
)
from ...determinism_trace import (
    trace_enabled as determinism_trace_enabled,
)
from ...hmonnx_op_trace import hmonnx_op_trace_scope
from ...gemma4_trace import GEMMA4_TRACE_ENV, emit_trace, trace_enabled
from ...hmfp_kv_cache import (
    HMFPFullAttentionSpec,
    HMFPPagedKVCache,
    HMFPSlidingWindowSpec,
    get_hmfp_vllm_backing_cache_dtype,
    is_hmfp_cache_dtype,
    write_hmfp_kv_cache_by_slot,
)
from ...kv_capacity import resolve_finalized_group_layer_specs
from ..register import MERAK_MODEL_REGISTRY
from .gemma4_cache_contract import (
    AttentionKind,
    Gemma4CacheContract,
    Gemma4LayerCacheBinding,
)

if TYPE_CHECKING:
    from xhquant.api import CacheTensor

    from .merak_linear_checkpoint_store import (
        LinearCheckpointSlot,
        MerakLinearCheckpointStore,
    )

logger = init_logger(__name__)

MerakExecutionPhase = Literal["prefill", "decode", "verify"]


@dataclass(frozen=True)
class MerakRequestResult:
    """Structured result retained across target verification and acceptance."""

    logits: torch.Tensor
    hidden_states: torch.Tensor | None = None
    output_names: tuple[str, ...] = ()
    raw_outputs: tuple[Any, ...] = ()


def _resolve_hmfp_block_size(cache_config: Any) -> int:
    block_size = getattr(cache_config, "block_size", None) if cache_config is not None else None
    if block_size is None:
        return MERAK_PREFERRED_BLOCK_SIZE

    block_size = int(block_size)
    if block_size <= 0 or block_size % MERAK_PREFERRED_BLOCK_SIZE != 0:
        raise ValueError(
            f"cache_config.block_size must be a positive multiple of {MERAK_PREFERRED_BLOCK_SIZE}, got {block_size}."
        )
    return block_size


class MerakAttention(Attention):
    def __init__(
        self,
        *args: Any,
        cache_config=None,
        attention_kind: AttentionKind = "full_attention",
        sliding_window: int | None = None,
        **kwargs: Any,
    ) -> None:
        if attention_kind not in {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported Merak attention kind {attention_kind!r}")
        if attention_kind == "full_attention" and sliding_window is not None:
            raise ValueError("Merak full_attention must not declare sliding_window")
        if attention_kind == "sliding_attention" and (type(sliding_window) is not int or sliding_window <= 0):
            raise ValueError("Merak sliding_attention requires a positive sliding_window")
        self.attention_kind = attention_kind
        self.kv_sharing_target_layer_name = kwargs.get("kv_sharing_target_layer_name")
        self.merak_logical_kv_cache_dtype = getattr(
            cache_config,
            "merak_logical_kv_cache_dtype",
            getattr(cache_config, "cache_dtype", "auto"),
        )
        if is_hmfp_cache_dtype(self.merak_logical_kv_cache_dtype) and cache_config is not None:
            cache_config = copy(cache_config)
            cache_config.cache_dtype = get_hmfp_vllm_backing_cache_dtype(self.merak_logical_kv_cache_dtype)
        # vLLM Attention owns ``self.sliding_window`` and assigns it during
        # super().__init__. Forward Gemma's per-layer value through the
        # official argument instead of assigning the attribute beforehand;
        # otherwise upstream overwrites every sliding placeholder with None.
        super().__init__(
            *args,
            cache_config=cache_config,
            per_layer_sliding_window=sliding_window,
            **kwargs,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if self.kv_sharing_target_layer_name is not None:
            return None
        if not is_hmfp_cache_dtype(self.merak_logical_kv_cache_dtype):
            spec = super().get_kv_cache_spec(vllm_config)
            if spec is not None and self.attention_kind == "sliding_attention":
                # Mixed Gemma4 keeps sliding-cache storage in raw FP. The
                # HMONNX PageKVCache backend indexes that storage through the
                # physical block stride, so vLLM may safely pad its scheduler
                # page to the larger HMFP full-attention page size.
                spec = replace(spec, indexes_kv_by_block_stride=True)
            return spec

        block_size = _resolve_hmfp_block_size(vllm_config.cache_config)
        spec_cls: type[HMFPFullAttentionSpec] | type[HMFPSlidingWindowSpec]
        extra: dict[str, int]
        if self.attention_kind == "full_attention":
            spec_cls = HMFPFullAttentionSpec
            extra = {}
        elif self.attention_kind == "sliding_attention":
            spec_cls = HMFPSlidingWindowSpec
            if type(self.sliding_window) is not int or self.sliding_window <= 0:
                raise RuntimeError(
                    f"Merak sliding placeholder lost its validated per-layer window: {self.sliding_window!r}"
                )
            extra = {"sliding_window": self.sliding_window}
        else:  # Defensive against post-construction mutation.
            raise ValueError(f"Unsupported Merak attention kind {self.attention_kind!r}")
        return spec_cls(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            sefp_cache_dtype=self.merak_logical_kv_cache_dtype,
            **extra,
        )


def bind_gemma4_cache_groups(
    contract: Gemma4CacheContract,
    kv_cache_groups: Any,
) -> tuple[Gemma4LayerCacheBinding, ...]:
    """Resolve Gemma bindings against finalized vLLM groups by exact layer name."""

    expected_by_name = {binding.layer_name: binding for binding in contract.bindings}
    if len(expected_by_name) != len(contract.bindings):
        raise RuntimeError("Gemma4 cache contract contains duplicate layer names")

    shared_layer_targets = {
        binding.layer_name: f"placeholder_attn_layers.{binding.cache_owner_layer_idx}"
        for binding in contract.bindings
        if binding.cache_owner_layer_idx is not None and binding.cache_owner_layer_idx != binding.layer_idx
    }
    resolved_by_name: dict[str, tuple[int, object]] = {}
    child_types_by_group: dict[int, dict[str, str]] = {}
    for group_idx, group in enumerate(kv_cache_groups):
        try:
            specs_by_layer = resolve_finalized_group_layer_specs(
                group,
                shared_layer_targets=shared_layer_targets,
            )
        except ValueError as exc:
            raise RuntimeError(f"Gemma4 KV group {group_idx} has invalid finalized specs: {exc}") from exc
        child_types_by_group[group_idx] = {
            layer_name: type(spec).__name__ for layer_name, spec in specs_by_layer.items()
        }
        for raw_layer_name in getattr(group, "layer_names", ()):
            layer_name = str(raw_layer_name)
            if layer_name in resolved_by_name:
                raise RuntimeError(f"Gemma4 KV layer {layer_name!r} appears more than once")
            if layer_name not in expected_by_name:
                raise RuntimeError(f"Gemma4 KV group contains unexpected layer {layer_name!r}")
            resolved_by_name[layer_name] = (
                group_idx,
                specs_by_layer[layer_name],
            )

    # vLLM intentionally omits modules with kv_sharing_target_layer_name
    # from finalized cache groups. Recover those logical Gemma layers from
    # the exact owner mapping retained by the export contract.
    for layer_name, owner_name in shared_layer_targets.items():
        if layer_name in resolved_by_name:
            continue
        owner_resolution = resolved_by_name.get(owner_name)
        if owner_resolution is None:
            raise RuntimeError(
                "Gemma4 shared KV owner is missing from finalized groups: "
                f"layer={layer_name!r}, owner={owner_name!r}"
            )
        resolved_by_name[layer_name] = owner_resolution

    missing = sorted(set(expected_by_name) - set(resolved_by_name))
    if missing:
        raise RuntimeError(f"Gemma4 KV groups are missing exact layers: {missing}")

    finalized: list[Gemma4LayerCacheBinding] = []
    cache_group_by_index: dict[int, int] = {}
    for binding in contract.bindings:
        group_idx, spec = resolved_by_name[binding.layer_name]
        expected_spec_type = (
            SlidingWindowSpec
            if contract.attention_lowering == "full_flash_attention"
            and binding.attention_kind == "sliding_attention"
            else (
                HMFPFullAttentionSpec
                if binding.attention_kind == "full_attention"
                else HMFPSlidingWindowSpec
            )
        )
        if type(spec) is not expected_spec_type:
            raise RuntimeError(
                "Gemma4 attention spec type mismatch for "
                f"{binding.layer_name}: expected {expected_spec_type.__name__}, "
                f"got {type(spec).__name__}; group_idx={group_idx}, "
                f"child_types={child_types_by_group[group_idx]}"
            )
        if binding.attention_kind == "sliding_attention" and int(getattr(spec, "sliding_window", -1)) != int(
            binding.sliding_window or -1
        ):
            raise RuntimeError(
                "Gemma4 sliding_window mismatch for "
                f"{binding.layer_name}: expected {binding.sliding_window}, "
                f"got {getattr(spec, 'sliding_window', None)}"
            )

        prior_group_idx = cache_group_by_index.setdefault(binding.cache_index, group_idx)
        if prior_group_idx != group_idx:
            raise RuntimeError(
                "Gemma4 shared cache_index maps to multiple KV groups: "
                f"cache_index={binding.cache_index}, groups={prior_group_idx}/{group_idx}"
            )
        scheduler_block_size = int(spec.block_size)
        if scheduler_block_size < MERAK_PREFERRED_BLOCK_SIZE or scheduler_block_size % MERAK_PREFERRED_BLOCK_SIZE != 0:
            raise RuntimeError(
                "Gemma4 finalized scheduler block size must be a positive "
                f"multiple of physical block size {MERAK_PREFERRED_BLOCK_SIZE}: "
                f"layer={binding.layer_name}, scheduler={scheduler_block_size}"
            )
        finalized.append(
            replace(
                binding,
                cache_group_idx=group_idx,
                scheduler_block_size=scheduler_block_size,
                physical_block_size=MERAK_PREFERRED_BLOCK_SIZE,
            )
        )

    return tuple(finalized)


@MERAK_MODEL_REGISTRY.register_module("MerakForCausalLM")
class MerakForCausalLM(nn.Module):
    """Base class for Merak HMONNX-backed causal-LM models.

    Subclasses are expected to populate the following attributes during
    ``__init__``: ``vllm_config``, ``inference_model``, ``inference_device``,
    ``input_embeddings``, ``placeholder_attn_layers``, ``logits_processor``,
    ``sampler``. The shared methods on this class assume those are present.
    """

    supports_v1: bool = True
    supports_lora: bool = False
    supports_pp: bool = False
    is_text_generation_model: bool = True

    # Multimodal subclasses set this so _run_request prefers passing raw
    # input_ids to the HMONNX runtime instead of pre-computed embeddings.
    _prefer_input_ids_runtime: bool = False

    def __init__(self, vllm_config: Optional[VllmConfig] = None, prefix: str = "") -> None:
        del vllm_config, prefix
        self.inference_model = None
        super().__init__()

    # ------------------------------------------------------------------
    # KV cache properties (delegate to the HMONNX runtime mixin)
    # ------------------------------------------------------------------
    @property
    def past_key_caches(self) -> list["CacheTensor"]:
        return self.inference_model.get_kvcache_mixin().past_key_caches  # type: ignore[attr-defined]

    @property
    def past_value_caches(self) -> list["CacheTensor"]:
        return self.inference_model.get_kvcache_mixin().past_value_caches  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Embedding (text-only default; multimodal subclasses override)
    # ------------------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        emb = self.input_embeddings  # type: ignore[attr-defined]
        return emb(input_ids.to(emb.weight.device)).to(input_ids.device)

    def embed_multimodal(self, **kwargs: object):  # type: ignore[override]
        del kwargs
        return []

    # ------------------------------------------------------------------
    # Block-table arithmetic
    # ------------------------------------------------------------------
    def _get_block_size(self) -> int:
        vllm_config = getattr(self, "vllm_config", None)
        cache_config = getattr(vllm_config, "cache_config", None)
        return _resolve_hmfp_block_size(cache_config)

    def _required_block_count(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        block_size = self._get_block_size()
        return (num_tokens + block_size - 1) // block_size

    def _copy_paged_to_contiguous(
        self,
        paged_cache: Tensor,
        block_ids: tuple[int, ...],
        dst_key_cache: Tensor,
        dst_value_cache: Tensor,
        start_pos: int,
        end_pos: int,
    ) -> None:
        if end_pos <= start_pos or paged_cache.numel() == 0:
            return

        required_blocks = self._required_block_count(end_pos)
        if len(block_ids) < required_blocks:
            raise ValueError(
                f"Block table only has {len(block_ids)} blocks but {required_blocks} are required for {end_pos} tokens."
            )

        block_size = self._get_block_size()
        key_blocks, value_blocks = paged_cache.unbind(0)
        start_block = start_pos // block_size
        for logical_block_idx in range(start_block, required_blocks):
            physical_block_id = block_ids[logical_block_idx]
            logical_block_start = logical_block_idx * block_size
            logical_block_end = logical_block_start + block_size
            copy_start = max(start_pos, logical_block_start)
            copy_end = min(end_pos, logical_block_end)
            src_start = copy_start - logical_block_start
            src_end = copy_end - logical_block_start

            dst_key_cache[0, :, copy_start:copy_end, :].copy_(key_blocks[physical_block_id, :, src_start:src_end, :])
            dst_value_cache[0, :, copy_start:copy_end, :].copy_(
                value_blocks[physical_block_id, :, src_start:src_end, :]
            )

    def _copy_sefp_paged_to_contiguous(
        self,
        paged_cache: HMFPPagedKVCache,
        block_ids: tuple[int, ...],
        dst_key_cache: Tensor,
        dst_value_cache: Tensor,
        start_pos: int,
        end_pos: int,
    ) -> None:
        if end_pos <= start_pos or paged_cache.numel() == 0:
            return

        required_blocks = self._required_block_count(end_pos)
        if len(block_ids) < required_blocks:
            raise ValueError(
                f"Block table only has {len(block_ids)} blocks but {required_blocks} are required for {end_pos} tokens."
            )

        block_size = self._get_block_size()
        start_block = start_pos // block_size
        for logical_block_idx in range(start_block, required_blocks):
            physical_block_id = block_ids[logical_block_idx]
            logical_block_start = logical_block_idx * block_size
            logical_block_end = logical_block_start + block_size
            copy_start = max(start_pos, logical_block_start)
            copy_end = min(end_pos, logical_block_end)
            src_start = copy_start - logical_block_start
            src_end = copy_end - logical_block_start

            key = paged_cache.k_qdata[physical_block_id, :, :, :, src_start:src_end].float()
            key_scale = torch.pow(
                2.0,
                paged_cache.k_scale_exp[physical_block_id, :, :, :, src_start:src_end].float(),
            )
            value = paged_cache.v_qdata[physical_block_id].float()
            value_scale = torch.pow(2.0, paged_cache.v_scale_exp[physical_block_id].float())

            num_tokens = copy_end - copy_start
            key_block = (key * key_scale).reshape(paged_cache.num_kv_heads, paged_cache.head_size, num_tokens)
            value_block = (value * value_scale).reshape(
                paged_cache.num_kv_heads,
                paged_cache.block_size,
                paged_cache.head_size_v,
            )[:, src_start:src_end, :]
            dst_key_cache[0, :, copy_start:copy_end, :].copy_(key_block.permute(0, 2, 1).to(dst_key_cache.dtype))
            dst_value_cache[0, :, copy_start:copy_end, :].copy_(value_block.to(dst_value_cache.dtype))

    def _copy_contiguous_to_paged(
        self,
        src_key_cache: Tensor,
        src_value_cache: Tensor,
        block_ids: tuple[int, ...],
        paged_cache: Tensor,
        start_pos: int,
        end_pos: int,
    ) -> None:
        if end_pos <= start_pos or paged_cache.numel() == 0:
            return

        required_blocks = self._required_block_count(end_pos)
        if len(block_ids) < required_blocks:
            raise ValueError(
                f"Block table only has {len(block_ids)} blocks but {required_blocks} are required for {end_pos} tokens."
            )

        block_size = self._get_block_size()
        key_blocks, value_blocks = paged_cache.unbind(0)
        start_block = start_pos // block_size
        for logical_block_idx in range(start_block, required_blocks):
            physical_block_id = block_ids[logical_block_idx]
            logical_block_start = logical_block_idx * block_size
            logical_block_end = logical_block_start + block_size
            copy_start = max(start_pos, logical_block_start)
            copy_end = min(end_pos, logical_block_end)
            dst_start = copy_start - logical_block_start
            dst_end = copy_end - logical_block_start

            key_blocks[physical_block_id, :, dst_start:dst_end, :].copy_(src_key_cache[0, :, copy_start:copy_end, :])
            value_blocks[physical_block_id, :, dst_start:dst_end, :].copy_(
                src_value_cache[0, :, copy_start:copy_end, :]
            )

    def _copy_contiguous_to_sefp_paged(
        self,
        src_key_cache: Tensor,
        src_value_cache: Tensor,
        block_ids: tuple[int, ...],
        paged_cache: HMFPPagedKVCache,
        start_pos: int,
        end_pos: int,
    ) -> None:
        if end_pos <= start_pos or paged_cache.numel() == 0:
            return

        required_blocks = self._required_block_count(end_pos)
        if len(block_ids) < required_blocks:
            raise ValueError(
                f"Block table only has {len(block_ids)} blocks but {required_blocks} are required for {end_pos} tokens."
            )

        block_size = self._get_block_size()
        start_block = start_pos // block_size
        for logical_block_idx in range(start_block, required_blocks):
            physical_block_id = block_ids[logical_block_idx]
            logical_block_start = logical_block_idx * block_size
            logical_block_end = logical_block_start + block_size
            copy_start = max(start_pos, logical_block_start)
            copy_end = min(end_pos, logical_block_end)
            dst_start = copy_start - logical_block_start
            dst_end = copy_end - logical_block_start

            key_tokens = src_key_cache[0, :, copy_start:copy_end, :].permute(1, 0, 2).contiguous()
            value_tokens = src_value_cache[0, :, copy_start:copy_end, :].permute(1, 0, 2).contiguous()
            slot_mapping = torch.arange(
                physical_block_id * block_size + dst_start,
                physical_block_id * block_size + dst_end,
                dtype=torch.long,
                device=key_tokens.device,
            )
            write_hmfp_kv_cache_by_slot(key_tokens, value_tokens, paged_cache, slot_mapping)

    def _reset_runtime_cache_like(self, cache: object) -> None:
        """Reset one HMONNX runtime cache object in-place.

        Qwen3.5 exports have two kinds of mutable runtime state:

        * full-attention KV caches, exposed as CacheTensor objects;
        * linear-attention caches, where a logical layer cache may be a nested
          tuple/list of CacheTensor or Tensor objects.

        Reset recursively so a fresh vLLM request cannot inherit linear-cache
        state from the previous request.
        """
        if cache is None:
            return

        reset = getattr(cache, "reset", None)
        if callable(reset):
            reset()
            return

        if isinstance(cache, Tensor):
            cache.zero_()
            return

        if isinstance(cache, (list, tuple)):
            for item in cache:
                self._reset_runtime_cache_like(item)

    def _get_kvcache_mixin(self) -> Any:
        get_kvcache_mixin = getattr(self.inference_model, "get_kvcache_mixin", None)  # type: ignore[attr-defined]
        if not callable(get_kvcache_mixin):
            return None
        return cast(Any, get_kvcache_mixin())

    def _clear_runtime_linear_caches(self) -> None:
        """Zero only the HMONNX linear-attention caches (conv q/k/v + recurrent).

        Split out of :meth:`_clear_runtime_request_caches` so the runner can
        control linear-state lifetime per request via the MambaSpec state
        blocks, independently of the full-attention KV reset.
        """
        kvcache_mixin = self._get_kvcache_mixin()
        if kvcache_mixin is None:
            return
        for cache in getattr(kvcache_mixin, "past_conv_caches", []) or []:
            self._reset_runtime_cache_like(cache)
        for cache in getattr(kvcache_mixin, "past_recurrent_states", []) or []:
            self._reset_runtime_cache_like(cache)

    def _clear_runtime_request_caches(self) -> None:
        for cache in self.past_key_caches:
            self._reset_runtime_cache_like(cache)
        for cache in self.past_value_caches:
            self._reset_runtime_cache_like(cache)

        # Linear caches are still cleared here for the legacy whole-request
        # reset path, but correctness for hybrid scheduling is now driven by the
        # runner through _clear_runtime_linear_caches / linear state transfer.
        self._clear_runtime_linear_caches()

    def _load_runtime_request_cache_from_paged(
        self,
        block_ids: tuple[int, ...],
        past_len: int,
    ) -> None:
        if past_len <= 0:
            self._clear_runtime_request_caches()
            return

        for layer_idx, attn_layer in enumerate(self.placeholder_attn_layers):  # type: ignore[attr-defined]
            paged_cache = attn_layer.kv_cache
            if isinstance(paged_cache, HMFPPagedKVCache):
                self._copy_sefp_paged_to_contiguous(
                    paged_cache,
                    block_ids,
                    self.past_key_caches[layer_idx],
                    self.past_value_caches[layer_idx],
                    0,
                    past_len,
                )
            else:
                self._copy_paged_to_contiguous(
                    cast(Tensor, paged_cache),
                    block_ids,
                    self.past_key_caches[layer_idx],
                    self.past_value_caches[layer_idx],
                    0,
                    past_len,
                )

    def _scatter_runtime_request_cache_to_paged(
        self,
        block_ids: tuple[int, ...],
        start_pos: int,
        end_pos: int,
    ) -> None:
        for layer_idx, attn_layer in enumerate(self.placeholder_attn_layers):  # type: ignore[attr-defined]
            paged_cache = attn_layer.kv_cache
            if isinstance(paged_cache, HMFPPagedKVCache):
                self._copy_contiguous_to_sefp_paged(
                    self.past_key_caches[layer_idx],
                    self.past_value_caches[layer_idx],
                    block_ids,
                    paged_cache,
                    start_pos,
                    end_pos,
                )
            else:
                self._copy_contiguous_to_paged(
                    self.past_key_caches[layer_idx],
                    self.past_value_caches[layer_idx],
                    block_ids,
                    cast(Tensor, paged_cache),
                    start_pos,
                    end_pos,
                )

    def _ensure_runtime_cache_tensors(self) -> None:
        get_kvcache_mixin = getattr(self.inference_model, "get_kvcache_mixin", None)  # type: ignore[attr-defined]
        if not callable(get_kvcache_mixin):
            return

        kvcache_mixin = cast(Any, get_kvcache_mixin())
        if hasattr(kvcache_mixin, "to"):
            kvcache_mixin.to(self.inference_device)  # type: ignore[attr-defined]

        past_key_caches = getattr(kvcache_mixin, "past_key_caches", None)
        past_value_caches = getattr(kvcache_mixin, "past_value_caches", None)
        if isinstance(past_key_caches, list) and isinstance(past_value_caches, list):
            if len(past_key_caches) == 0 and len(past_value_caches) == 0:
                kvcache_mixin.prepare_kv_cache()

        past_conv_caches = getattr(kvcache_mixin, "past_conv_caches", None)
        past_recurrent_states = getattr(kvcache_mixin, "past_recurrent_states", None)
        if isinstance(past_conv_caches, list) and isinstance(past_recurrent_states, list):
            if len(past_conv_caches) == 0 and len(past_recurrent_states) == 0:
                kvcache_mixin.prepare_other_cache()

    def _iter_runtime_linear_state_layers(self):
        """Yield HMONNX linear states in placeholder order without dense caches."""
        placeholders = getattr(self, "placeholder_linear_state_layers", None)
        if not placeholders:
            return
        kvcache_mixin = self._get_kvcache_mixin()
        if kvcache_mixin is None:
            return
        past_conv_caches = getattr(kvcache_mixin, "past_conv_caches", None)
        past_recurrent_states = getattr(kvcache_mixin, "past_recurrent_states", None)
        if not past_conv_caches or not past_recurrent_states:
            return
        if len(past_conv_caches) != len(placeholders) or len(past_recurrent_states) != len(placeholders):
            raise RuntimeError(
                "Merak HMONNX linear-state count mismatch: "
                f"{len(placeholders)} placeholders, {len(past_conv_caches)} conv caches, "
                f"{len(past_recurrent_states)} recurrent states."
            )

        for layer_idx, placeholder in enumerate(placeholders):
            conv_triple = past_conv_caches[layer_idx]
            recurrent = past_recurrent_states[layer_idx]
            if not isinstance(conv_triple, (tuple, list)) or len(conv_triple) != 3:
                raise RuntimeError(
                    "Merak split-conv linear cache expects a (q, k, v) triple per "
                    f"layer, got {type(conv_triple).__name__} at layer {layer_idx}."
                )
            yield str(placeholder.layer_name), conv_triple, recurrent

    def _iter_linear_state_layers(self):
        """Yield legacy dense-placeholder state alongside each HMONNX layer.

        ``state_tensors`` is the placeholder layer's four-tuple MambaSpec cache
        (conv_q, conv_k, conv_v, recurrent), each shaped ``[num_blocks, *state]``.
        """
        placeholders = getattr(self, "placeholder_linear_state_layers", None) or ()
        for layer_idx, (_layer_name, conv_triple, recurrent) in enumerate(self._iter_runtime_linear_state_layers()):
            state_tensors = placeholders[layer_idx].kv_cache
            if len(state_tensors) != 4:
                raise RuntimeError(
                    "Merak linear-state placeholder expects 4 MambaSpec tensors "
                    f"(conv_q, conv_k, conv_v, recurrent), got {len(state_tensors)} "
                    f"at layer {layer_idx}."
                )
            yield layer_idx, conv_triple, recurrent, state_tensors

    @staticmethod
    def _validate_linear_state_layer_order(
        runtime_layer_names: tuple[str, ...],
        checkpoint_layer_names: tuple[str, ...],
    ) -> None:
        runtime_set = set(runtime_layer_names)
        checkpoint_set = set(checkpoint_layer_names)
        if runtime_set != checkpoint_set:
            raise RuntimeError(
                "Merak linear-state layer set mismatch: "
                f"missing={sorted(runtime_set - checkpoint_set)}, "
                f"extra={sorted(checkpoint_set - runtime_set)}."
            )
        if runtime_layer_names != checkpoint_layer_names:
            raise RuntimeError(
                "Merak linear-state layer order mismatch: "
                f"runtime={runtime_layer_names}, checkpoint={checkpoint_layer_names}."
            )

    @staticmethod
    def _validate_linear_state_tensors(
        layer_name: str,
        source_states: tuple[Tensor, ...],
        destination_states: tuple[Tensor, ...],
    ) -> None:
        if len(source_states) != 4 or len(destination_states) != 4:
            raise RuntimeError(
                f"Merak linear-state layer {layer_name} requires exactly 4 states; "
                f"source={len(source_states)}, destination={len(destination_states)}."
            )
        for state_idx, (source, destination) in enumerate(zip(source_states, destination_states, strict=True)):
            if not isinstance(source, Tensor) or not isinstance(destination, Tensor):
                raise RuntimeError(f"Merak linear-state layer {layer_name} state {state_idx} must be a Tensor.")
            if source.shape != destination.shape:
                raise RuntimeError(
                    f"Merak linear-state layer {layer_name} state {state_idx} shape mismatch: "
                    f"source={tuple(source.shape)}, destination={tuple(destination.shape)}."
                )
            if source.dtype != destination.dtype:
                raise RuntimeError(
                    f"Merak linear-state layer {layer_name} state {state_idx} dtype mismatch: "
                    f"source={source.dtype}, destination={destination.dtype}."
                )

    def _load_runtime_linear_state_from_store(self, slot: "LinearCheckpointSlot") -> None:
        """Restore HMONNX linear state from one request's sparse checkpoint slot."""
        runtime_layers = list(self._iter_runtime_linear_state_layers())
        runtime_names = tuple(layer_name for layer_name, _conv, _recurrent in runtime_layers)
        checkpoint_names = tuple(slot.states_by_layer)
        self._validate_linear_state_layer_order(runtime_names, checkpoint_names)

        copies = []
        for layer_name, conv_triple, recurrent in runtime_layers:
            source_states = tuple(slot.states_by_layer[layer_name])
            destination_states = (
                conv_triple[0][0],
                conv_triple[1][0],
                conv_triple[2][0],
                recurrent[0],
            )
            self._validate_linear_state_tensors(layer_name, source_states, destination_states)
            copies.append((source_states, destination_states))

        for source_states, destination_states in copies:
            for source, destination in zip(source_states, destination_states, strict=True):
                destination.copy_(source)

    def _scatter_runtime_linear_state_to_store(
        self,
        store: "MerakLinearCheckpointStore",
        request_id: str,
    ) -> None:
        """Persist HMONNX linear state into a request's sparse checkpoint slot."""
        runtime_layers = list(self._iter_runtime_linear_state_layers())
        runtime_names = tuple(layer_name for layer_name, _conv, _recurrent in runtime_layers)
        self._validate_linear_state_layer_order(runtime_names, tuple(store.layer_names))

        states_by_layer = []
        for layer_name, conv_triple, recurrent in runtime_layers:
            runtime_states = (
                conv_triple[0][0],
                conv_triple[1][0],
                conv_triple[2][0],
                recurrent[0],
            )
            store_states = tuple(store.layer_state(request_id, layer_name))
            self._validate_linear_state_tensors(layer_name, runtime_states, store_states)
            states_by_layer.append((layer_name, runtime_states))

        for layer_name, runtime_states in states_by_layer:
            store.store_layer_state(request_id, layer_name, runtime_states)

    def _validate_runtime_linear_state_contract(self) -> None:
        placeholders = list(getattr(self, "placeholder_linear_state_layers", None) or ())
        if not placeholders:
            return
        kvcache_mixin = self._get_kvcache_mixin()
        if kvcache_mixin is None:
            raise RuntimeError("Merak linear-state placeholders require an HMONNX KV-cache mixin.")
        past_conv_caches = list(getattr(kvcache_mixin, "past_conv_caches", None) or ())
        past_recurrent_states = list(getattr(kvcache_mixin, "past_recurrent_states", None) or ())
        expected = len(placeholders)
        if len(past_conv_caches) != expected or len(past_recurrent_states) != expected:
            raise RuntimeError(
                "Merak HMONNX linear-state count mismatch: "
                f"{expected} placeholders, {len(past_conv_caches)} conv caches, "
                f"{len(past_recurrent_states)} recurrent states."
            )
        for layer_idx, conv_cache in enumerate(past_conv_caches):
            if not isinstance(conv_cache, (tuple, list)) or len(conv_cache) != 3:
                raise RuntimeError(
                    "Merak split-conv linear cache expects a (q, k, v) triple per layer, "
                    f"got {type(conv_cache).__name__} at linear layer index {layer_idx}."
                )

    def _resolve_state_indices(self, state_idx: "int | list[int]", num_layers: int) -> list[int]:
        if isinstance(state_idx, int):
            return [state_idx] * num_layers
        indices = list(state_idx)
        if len(indices) != num_layers:
            raise RuntimeError(
                f"Merak linear state_indices length ({len(indices)}) does not match linear layer count ({num_layers})."
            )
        return indices

    def _load_runtime_linear_state_from_vllm(self, state_idx: "int | list[int]") -> None:
        """Restore HMONNX linear caches from vLLM MambaSpec blocks.

        ``state_idx`` may be a single physical block index (applied to all
        layers) or a per-layer list of physical indices. vLLM's hybrid manager
        splits linear layers across several MambaSpec cache groups, each with an
        independent block table, so a decode step needs the physical index of
        the group that owns each specific layer.

        Copies (cpu vLLM block -> cuda HMONNX runtime) for every linear layer.
        """
        layers = list(self._iter_linear_state_layers())
        indices = self._resolve_state_indices(state_idx, len(layers))
        for (_layer_idx, conv_triple, recurrent, state_tensors), phys in zip(layers, indices):
            conv_q, conv_k, conv_v = conv_triple
            conv_q[0].copy_(state_tensors[0][phys])
            conv_k[0].copy_(state_tensors[1][phys])
            conv_v[0].copy_(state_tensors[2][phys])
            recurrent[0].copy_(state_tensors[3][phys])

    def _scatter_runtime_linear_state_to_vllm(self, state_idx: "int | list[int]") -> None:
        """Persist HMONNX linear caches into vLLM MambaSpec blocks.

        ``state_idx`` may be a single physical block index or a per-layer list
        of physical indices (one per linear layer, matching its cache group).

        Copies (cuda HMONNX runtime -> cpu vLLM block) for every linear layer.
        """
        layers = list(self._iter_linear_state_layers())
        indices = self._resolve_state_indices(state_idx, len(layers))
        for (_layer_idx, conv_triple, recurrent, state_tensors), phys in zip(layers, indices):
            conv_q, conv_k, conv_v = conv_triple
            state_tensors[0][phys].copy_(conv_q[0])
            state_tensors[1][phys].copy_(conv_k[0])
            state_tensors[2][phys].copy_(conv_v[0])
            state_tensors[3][phys].copy_(recurrent[0])

    def _set_runtime_page_attention_context(
        self,
        runtime_model: Any,
        block_ids: tuple[int, ...],
        slot_mapping: torch.Tensor,
        effective_kv_length: int | None = None,
    ) -> None:
        if self._uses_gemma4_page_attention_contract():
            raise RuntimeError("Gemma4 contract-v2 requires group-keyed PageAttention contexts")
        set_page_attention_context = getattr(runtime_model, "set_page_attention_context", None)
        if not callable(set_page_attention_context):
            raise NotImplementedError(
                "Merak page attention requires the HMONNX runtime to expose "
                "set_page_attention_context(paged_kv_caches, block_ids, slot_mapping, "
                "past_seq_length, query_length, block_size)."
            )

        block_ids_tensor = self._copy_to_page_attention_context_buffer(
            "_page_attention_block_ids_buffer",
            torch.as_tensor(block_ids, dtype=torch.int64),
            dtype=torch.int64,
        )
        slot_mapping = self._copy_to_page_attention_context_buffer(
            "_page_attention_slot_mapping_buffer",
            slot_mapping,
            dtype=torch.int64,
        )
        paged_kv_caches = tuple(attn_layer.kv_cache for attn_layer in self.placeholder_attn_layers)  # type: ignore[attr-defined]
        block_size = self._get_block_size()
        if not getattr(self, "_use_local_page_attention_context", False):
            try:
                set_page_attention_context(
                    paged_kv_caches=paged_kv_caches,
                    block_ids=block_ids_tensor,
                    slot_mapping=slot_mapping,
                    block_size=block_size,
                    effective_kv_length=effective_kv_length,
                )
                return
            except AttributeError as exc:
                if "graph_module" not in str(exc):
                    raise
                logger.warning_once(
                    "HMONNX runtime page-attention context API does not expose "
                    "graph_module; falling back to Merak local node_modules context wiring."
                )
                setattr(self, "_use_local_page_attention_context", True)

        self._set_runtime_page_attention_context_via_modules(
            runtime_model=runtime_model,
            paged_kv_caches=paged_kv_caches,
            block_ids=block_ids_tensor,
            slot_mapping=slot_mapping,
            block_size=block_size,
            effective_kv_length=effective_kv_length,
        )

    def _make_page_attention_v2_context(
        self,
        *,
        paged_kv_cache: Any,
        context: MerakPageAttentionGroupContext,
    ) -> Any:
        from xhquant.xhonnxruntime.parsers.page_attention import PageAttentionContext

        physical_block_size = (
            int(paged_kv_cache.shape[3])
            if isinstance(paged_kv_cache, torch.Tensor)
            else int(getattr(paged_kv_cache, "block_size", self._get_block_size()))
        )
        try:
            return PageAttentionContext(
                paged_kv_cache=paged_kv_cache,
                block_ids=context.block_ids,
                slot_mapping=context.slot_mapping,
                block_size=physical_block_size,
                kv_window_start_abs=context.kv_window_start_abs,
                seq_lens=context.seq_lens,
                mm_prefix_ranges=context.mm_prefix_ranges,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Gemma4 contract-v2 requires xhquant PageAttentionContext support "
                "for kv_window_start_abs, seq_lens, and mm_prefix_ranges"
            ) from exc

    def _set_runtime_page_attention_group_contexts(
        self,
        runtime_model: Any,
        group_contexts: dict[int, MerakPageAttentionGroupContext],
    ) -> None:
        bindings_by_group = self._gemma4_bindings_by_group()
        if set(group_contexts) != set(bindings_by_group):
            raise RuntimeError("Gemma4 PageAttention group contexts are incomplete")
        caches = self._page_attention_cache_by_index()
        contexts_by_cache_index: dict[int, Any] = {}
        cache_group_by_index: dict[int, int] = {}
        for group_idx, bindings in bindings_by_group.items():
            context = group_contexts[group_idx]
            if bindings[0].attention_kind == "full_attention" and (context.kv_window_start_abs is not None):
                raise RuntimeError(f"Gemma4 full cache_group_idx={group_idx} received sliding metadata")
            if bindings[0].attention_kind == "full_attention" and (context.mm_prefix_ranges is not None):
                raise RuntimeError(
                    f"Gemma4 full cache_group_idx={group_idx} received a visual "
                    "bidirectional overlay; Gemma4 full attention must remain causal"
                )
            for binding in bindings:
                prior_group_idx = cache_group_by_index.setdefault(binding.cache_index, group_idx)
                if prior_group_idx != group_idx:
                    raise RuntimeError(
                        "Gemma4 one cache_index cannot receive multiple group contexts: "
                        f"cache_index={binding.cache_index}, "
                        f"groups={prior_group_idx}/{group_idx}"
                    )
                if binding.cache_index in contexts_by_cache_index:
                    continue
                contexts_by_cache_index[binding.cache_index] = self._make_page_attention_v2_context(
                    paged_kv_cache=caches[binding.cache_index],
                    context=context,
                )

        set_page_attention_context = getattr(runtime_model, "set_page_attention_context", None)
        if not callable(set_page_attention_context):
            raise RuntimeError(
                "Gemma4 contract-v2 runtime must expose set_page_attention_context(contexts_by_cache_index=...)"
            )
        try:
            set_page_attention_context(
                contexts_by_cache_index=contexts_by_cache_index,
            )
            return
        except TypeError as exc:
            raise RuntimeError(
                "Gemma4 contract-v2 runtime does not accept contexts_by_cache_index; refusing the Qwen positional API"
            ) from exc
        except AttributeError as exc:
            if "graph_module" not in str(exc):
                raise

        for binding, page_attention_module in self._bound_page_attention_modules(runtime_model):
            context = group_contexts[cast(int, binding.cache_group_idx)]
            page_attention_module.set_context(
                self._make_page_attention_v2_context(
                    paged_kv_cache=caches[binding.cache_index],
                    context=context,
                )
            )

    def _set_runtime_page_attention_context_via_modules(
        self,
        *,
        runtime_model: Any,
        paged_kv_caches: tuple[Any, ...],
        block_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
        effective_kv_length: int | None = None,
    ) -> None:
        from xhquant.xhonnxruntime.parsers.page_attention import PageAttentionContext

        hmonnx_model = self._active_hmonnx_model(runtime_model)
        named_page_attention_modules = self._find_page_attention_modules(hmonnx_model)
        page_attention_modules = [
            entry[1] if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str) else entry
            for entry in named_page_attention_modules
        ]
        if not page_attention_modules:
            raise RuntimeError("Merak page attention runtime has no PageAttention modules to receive context.")
        if len(page_attention_modules) != len(paged_kv_caches):
            raise RuntimeError(
                "Merak page attention module/cache count mismatch: "
                f"{len(page_attention_modules)} modules vs {len(paged_kv_caches)} caches."
            )

        primary_device = torch.device(self.inference_device)  # type: ignore[attr-defined]
        cache_devices = [
            torch.device(getattr(paged_kv_cache, "device", primary_device)) for paged_kv_cache in paged_kv_caches
        ]
        distinct_cache_devices = set(cache_devices)
        if len(distinct_cache_devices) > 1:
            raise RuntimeError(
                "Merak multi-device PageAttention caches cannot use the primary-device-only "
                "local fallback; HMONNX set_page_attention_context must stage metadata per cache device."
            )
        metadata_device = next(iter(distinct_cache_devices), primary_device)
        if metadata_device != primary_device:
            raise RuntimeError(
                "Merak PageAttention local fallback cache device "
                f"{metadata_device} does not match primary inference device {primary_device}; "
                "per-cache-device metadata staging is required."
            )

        block_ids = block_ids.to(device=metadata_device, dtype=torch.int64, non_blocking=True)
        slot_mapping = slot_mapping.to(device=metadata_device, dtype=torch.int64, non_blocking=True)
        for paged_kv_cache, page_attention_module in zip(paged_kv_caches, page_attention_modules):
            page_attention_module.set_context(
                PageAttentionContext(
                    paged_kv_cache=paged_kv_cache,
                    block_ids=block_ids,
                    slot_mapping=slot_mapping,
                    block_size=int(block_size),
                    effective_kv_length=effective_kv_length,
                )
            )

    def _active_hmonnx_model(self, runtime_model: Any) -> Any:
        if bool(getattr(runtime_model, "_llm_prefill", True)):
            return getattr(runtime_model, "prefill_model", runtime_model)
        return getattr(runtime_model, "decode_model", runtime_model)

    def _find_page_attention_modules(self, hmonnx_model: Any) -> tuple[tuple[str, Any], ...]:
        from xhquant.xhonnxruntime.parsers.page_attention import PageAttention

        session = getattr(hmonnx_model, "hmonnx_session", hmonnx_model)
        candidates = [session]

        initialize = getattr(session, "initialize", None)
        if callable(initialize) and getattr(session, "_session", None) is None:
            initialize()

        inner_session = getattr(session, "_session", None)
        if inner_session is not None:
            candidates.append(inner_session)

        for candidate in candidates:
            graph_module = getattr(candidate, "graph_module", None)
            if graph_module is not None:
                modules: list[tuple[str, Any]] = []
                for node in graph_module.graph.nodes:
                    if node.op == "call_module":
                        module_name = str(node.target)
                        module = graph_module.get_submodule(module_name)
                        if isinstance(module, PageAttention):
                            modules.append((module_name, module))
                if modules:
                    return tuple(modules)

            node_modules = getattr(candidate, "node_modules", None)
            if node_modules is not None:
                modules = [
                    (f"node_modules.{module_idx}", module)
                    for module_idx, module in enumerate(node_modules)
                    if isinstance(module, PageAttention)
                ]
                if modules:
                    return tuple(modules)

        return ()

    @staticmethod
    def _page_attention_module_layer_idx(name: str, module: Any) -> int | None:
        del name
        layer_idx = getattr(module, "layer_idx", None)
        if type(layer_idx) is int:
            return layer_idx
        layer_name = getattr(module, "layer_name", None)
        if isinstance(layer_name, str):
            prefix = "placeholder_attn_layers."
            suffix = layer_name.removeprefix(prefix)
            if layer_name.startswith(prefix) and suffix.isdigit():
                return int(suffix)
        return None

    def _bound_page_attention_modules(self, runtime_model: Any) -> tuple[tuple[Gemma4LayerCacheBinding, Any], ...]:
        hmonnx_model = self._active_hmonnx_model(runtime_model)
        named_modules = self._find_page_attention_modules(hmonnx_model)
        modules_by_layer_idx: dict[int, tuple[str, Any]] = {}
        for name, module in named_modules:
            layer_idx = self._page_attention_module_layer_idx(name, module)
            if layer_idx is None:
                raise RuntimeError(f"Gemma4 PageAttention module {name!r} has no explicit layer identity")
            if layer_idx in modules_by_layer_idx:
                raise RuntimeError(f"Gemma4 PageAttention layer_idx={layer_idx} is duplicated")
            modules_by_layer_idx[layer_idx] = (name, module)

        contract = self.gemma4_cache_contract
        bindings = tuple(
            sorted(contract.context_bindings, key=lambda binding: binding.layer_idx)
        )
        expected_indices = {binding.layer_idx for binding in bindings}
        if set(modules_by_layer_idx) != expected_indices:
            raise RuntimeError(
                "Gemma4 PageAttention module/binding identity mismatch: "
                f"expected={sorted(expected_indices)}, "
                f"actual={sorted(modules_by_layer_idx)}"
            )
        bound = tuple(
            (binding, modules_by_layer_idx[binding.layer_idx][1])
            for binding in bindings
        )
        for binding, module in bound:
            module_cache_index = getattr(module, "cache_index", -1)
            if type(module_cache_index) is int and module_cache_index >= 0:
                if module_cache_index != binding.cache_index:
                    raise RuntimeError(
                        "Gemma4 page module/cache identity mismatch: "
                        f"layer={binding.layer_idx}, module_cache={module_cache_index}, "
                        f"binding_cache={binding.cache_index}"
                    )
        return bound

    def _writable_page_attention_modules(
        self,
        runtime_model: Any,
    ) -> tuple[Any, ...]:
        """Return target PageAttention writers in deterministic graph order."""

        hmonnx_model = self._active_hmonnx_model(runtime_model)
        modules = tuple(
            module
            for _name, module in self._find_page_attention_modules(hmonnx_model)
            if module.__class__.__name__ in {"PageAttention", "PageKVCache"}
        )
        if self.enable_page_attention and not modules:
            raise RuntimeError(
                "Merak target PageAttention is enabled but the active graph "
                "has no writable PageAttention modules"
            )
        return modules

    def _begin_page_attention_verify_transaction(
        self,
        runtime_model: Any,
        *,
        request_id: str,
        query_length: int,
    ) -> None:
        """Snapshot shared-exponent target KV before speculative verify."""

        pending_request = getattr(
            self,
            "_page_attention_verify_transaction_request_id",
            None,
        )
        if pending_request is not None:
            raise RuntimeError(
                "Merak supports only one uncommitted target PageAttention "
                f"verify: pending={pending_request!r}, next={request_id!r}"
            )
        modules = self._writable_page_attention_modules(runtime_model)
        started: list[Any] = []
        block_ids_by_slot_signature: dict[
            tuple[int, int, int],
            tuple[int, ...],
        ] = {}
        try:
            for module in modules:
                begin = getattr(module, "begin_cache_write_transaction", None)
                if not callable(begin):
                    raise RuntimeError(
                        "Speculative target PageAttention requires transactional "
                        "SEFP cache writes from the active xhquant runtime"
                    )
                slot_signature = getattr(
                    module,
                    "cache_write_transaction_slot_signature",
                    None,
                )
                physical_block_ids = (
                    block_ids_by_slot_signature.get(
                        tuple(slot_signature)
                    )
                    if slot_signature is not None
                    else None
                )
                if physical_block_ids is None:
                    begin(query_length=query_length)
                    if slot_signature is not None:
                        block_ids_by_slot_signature[
                            tuple(slot_signature)
                        ] = tuple(
                            module.cache_write_transaction_block_ids
                        )
                else:
                    # Layers bound to the same stable slot-mapping buffer can
                    # reuse its page resolution. Distinct Gemma cache groups
                    # retain independent resolutions.
                    begin(
                        query_length=query_length,
                        physical_block_ids=physical_block_ids,
                    )
                started.append(module)
        except Exception:
            for module in reversed(started):
                abort = getattr(module, "abort_cache_write_transaction", None)
                if callable(abort):
                    abort()
            raise
        self._page_attention_verify_transaction_modules = tuple(modules)
        self._page_attention_verify_transaction_request_id = request_id

    def _commit_page_attention_verify_transaction(
        self,
        *,
        request_id: str,
        accepted_steps: int,
    ) -> None:
        pending_request = getattr(
            self,
            "_page_attention_verify_transaction_request_id",
            None,
        )
        if pending_request != request_id:
            raise RuntimeError(
                "Merak target PageAttention transaction owner mismatch: "
                f"pending={pending_request!r}, commit={request_id!r}"
            )
        modules = tuple(
            getattr(
                self,
                "_page_attention_verify_transaction_modules",
                (),
            )
        )
        try:
            for module in modules:
                module.commit_cache_write_transaction(
                    accepted_steps=accepted_steps,
                )
        finally:
            self._page_attention_verify_transaction_modules = ()
            self._page_attention_verify_transaction_request_id = None

    def _abort_page_attention_verify_transaction(
        self,
        *,
        request_id: str | None = None,
    ) -> None:
        pending_request = getattr(
            self,
            "_page_attention_verify_transaction_request_id",
            None,
        )
        if pending_request is None:
            return
        if request_id is not None and pending_request != request_id:
            raise RuntimeError(
                "Merak target PageAttention abort owner mismatch: "
                f"pending={pending_request!r}, abort={request_id!r}"
            )
        modules = tuple(
            getattr(
                self,
                "_page_attention_verify_transaction_modules",
                (),
            )
        )
        try:
            for module in reversed(modules):
                module.abort_cache_write_transaction()
        finally:
            self._page_attention_verify_transaction_modules = ()
            self._page_attention_verify_transaction_request_id = None

    def _copy_to_page_attention_context_buffer(
        self,
        attr_name: str,
        source: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        device = self.inference_device  # type: ignore[attr-defined]
        length = int(source.numel())
        buffer = getattr(self, attr_name, None)
        if (
            not isinstance(buffer, torch.Tensor)
            or buffer.device != device
            or buffer.dtype != dtype
            or buffer.numel() < length
        ):
            buffer = torch.empty(length, dtype=dtype, device=device)
            setattr(self, attr_name, buffer)
            setattr(self, f"{attr_name}_view", None)

        view_attr_name = f"{attr_name}_view"
        view = getattr(self, view_attr_name, None)
        if not isinstance(view, torch.Tensor) or view.data_ptr() != buffer.data_ptr() or view.numel() != length:
            view = buffer[:length]
            setattr(self, view_attr_name, view)
        view.copy_(source.to(device=device, dtype=dtype, non_blocking=True).reshape(-1))
        return view

    def _pad_page_attention_slot_mapping(
        self,
        slot_mapping: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """Pad chunk-local slot mappings to the HMONNX runtime input length.

        Qwen3.5 HMONNX prefill pads a short final chunk to the exported
        ``input_sequence_length`` while keeping the real token count in its
        data preprocessor for some paths. CUDA-graph fallback/replay can still
        execute PageAttention with the padded length from the first captured
        chunk. Supplying ``-1`` slots for padded tokens makes the KV-cache write
        path ignore them and keeps PageAttention context shape compatible with
        either real or padded current lengths.
        """

        flat = slot_mapping.reshape(-1)
        if flat.numel() >= context_length:
            return flat[:context_length]

        padded = flat.new_full((context_length,), -1)
        if flat.numel() > 0:
            padded[: flat.numel()].copy_(flat)
        return padded

    @staticmethod
    def _pad_page_attention_block_ids(
        block_ids: tuple[int, ...],
        required_blocks: int,
    ) -> tuple[int, ...]:
        """Pad block ids so padded HMONNX chunks have a valid block table.

        The padded tail positions are marked invalid in slot_mapping and should
        not write KV cache. Some PageAttention execution paths nevertheless
        validate/read a block table for the full padded ``past + current``
        length, so repeat the last real block as a harmless valid placeholder.
        """

        if required_blocks <= len(block_ids):
            return block_ids
        if not block_ids:
            raise ValueError("page_attention_block_ids cannot be empty when PageAttention is enabled.")
        return block_ids + (block_ids[-1],) * (required_blocks - len(block_ids))

    def _page_attention_block_table_capacity(self, block_size: int, required_blocks: int) -> int:
        """Return a CUDA-graph-stable bucket for the PageAttention block table.

        xhquant PageAttention receives block ids out-of-band, so HMONNX's CUDA
        graph signature cannot see when the block table grows as context grows.
        Padding every request to ``max_model_len`` keeps replay safe but makes a
        256k deployment feed a 4096-entry block table to short prompts. Instead
        use power-of-two buckets and select the captured graph for the current
        bucket. Bucket choice must depend only on the current logical context;
        retaining a process high-water mark makes identical requests select
        different PageAttention kernels after a longer request has run.
        """

        required_blocks = max(int(required_blocks), 0)

        vllm_config = getattr(self, "vllm_config", None)
        model_config = getattr(vllm_config, "model_config", None)
        max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
        if max_model_len <= 0:
            return required_blocks

        max_blocks = (max_model_len + block_size - 1) // block_size
        if required_blocks <= 1:
            bucket_blocks = 1
        else:
            bucket_blocks = 1 << (required_blocks - 1).bit_length()
        bucket_blocks = min(max_blocks, max(required_blocks, bucket_blocks))
        return min(max_blocks, bucket_blocks)

    def _page_attention_max_block_table_capacity(self, block_size: int) -> int:
        vllm_config = getattr(self, "vllm_config", None)
        model_config = getattr(vllm_config, "model_config", None)
        max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
        if max_model_len <= 0:
            return 0
        return (max_model_len + block_size - 1) // block_size

    def _clear_runtime_cuda_graph_for_page_attention_capacity_change(
        self,
        runtime_model: Any,
        block_table_capacity: int,
    ) -> None:
        hmonnx_model = self._active_hmonnx_model(runtime_model)
        candidates = [hmonnx_model]
        hmonnx_session = getattr(hmonnx_model, "hmonnx_session", None)
        if hmonnx_session is not None:
            candidates.append(hmonnx_session)
            interpreter = getattr(hmonnx_session, "interpreter", None)
            if interpreter is not None:
                candidates.append(interpreter)

        cache_key = ("page_attention", "decode", block_table_capacity)
        for candidate in candidates:
            select_capture = getattr(candidate, "set_capture_cache_key", None)
            if callable(select_capture):
                select_capture(cache_key)
                return

        for candidate in candidates:
            clear = getattr(candidate, "clear", None)
            if callable(clear):
                clear(clear_disabled_reason=True)
                return
            clear_cuda_graph_state = getattr(candidate, "clear_cuda_graph_state", None)
            if callable(clear_cuda_graph_state):
                clear_cuda_graph_state(clear_disabled_reason=True)
                return

    # ------------------------------------------------------------------
    # Input shape helpers
    # ------------------------------------------------------------------
    def _flatten_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 2:
            if input_ids.shape[0] != 1:
                raise ValueError(f"Merak input_ids must have batch size 1, got shape {tuple(input_ids.shape)}")
            return input_ids.squeeze(0)
        if input_ids.dim() != 1:
            raise ValueError(
                f"Merak input_ids must have shape [num_tokens] or [batch, seq_len], got {tuple(input_ids.shape)}"
            )
        return input_ids

    def _flatten_inputs_embeds(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        if inputs_embeds.dim() == 3:
            if inputs_embeds.shape[0] != 1:
                raise ValueError(f"Merak inputs_embeds must have batch size 1, got shape {tuple(inputs_embeds.shape)}")
            return inputs_embeds.squeeze(0)
        if inputs_embeds.dim() != 2:
            raise ValueError(
                "Merak inputs_embeds must have shape [num_tokens, dim] or [batch, seq_len, dim], "
                f"got {tuple(inputs_embeds.shape)}"
            )
        return inputs_embeds

    def _restore_runtime_batch_shape(self, request_embeds: torch.Tensor) -> torch.Tensor:
        if request_embeds.dim() == 2:
            return request_embeds.unsqueeze(0)
        if request_embeds.dim() == 3:
            return request_embeds
        raise ValueError(
            "Merak request_embeds must have shape [num_tokens, dim] or [batch, seq_len, dim], "
            f"got {tuple(request_embeds.shape)}"
        )

    def _prepare_request_embeds(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Merak request needs either input_ids or inputs_embeds.")

        if inputs_embeds is not None:
            return self._flatten_inputs_embeds(inputs_embeds)

        assert input_ids is not None
        flat_input_ids = self._flatten_input_ids(input_ids)
        return self.embed_input_ids(flat_input_ids)

    # ------------------------------------------------------------------
    # Output extraction
    # ------------------------------------------------------------------
    def _extract_last_token_logits(self, model_output: Any) -> torch.Tensor:
        logits = model_output
        if isinstance(model_output, dict):
            logits = model_output["logits"]
        elif hasattr(model_output, "logits"):
            logits = model_output.logits
        elif isinstance(model_output, (list, tuple)):
            logits = model_output[0]

        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Unexpected Merak runtime output type: {type(logits).__name__}")
        vocab_size = self.logits_processor.vocab_size  # type: ignore[attr-defined]
        if logits.dim() == 1:
            return logits[:vocab_size]
        if logits.dim() == 2:
            return logits[-1, :vocab_size]
        return logits.reshape(-1, logits.shape[-1])[-1, :vocab_size]

    @staticmethod
    def _runtime_output_names(
        runtime_model: Any,
        output_count: int,
    ) -> tuple[str, ...]:
        is_prefill_fn = getattr(runtime_model, "is_prefill", None)
        if callable(is_prefill_fn):
            is_prefill = bool(is_prefill_fn())
        else:
            is_prefill = bool(getattr(runtime_model, "_llm_prefill", True))
        active_model = getattr(
            runtime_model,
            "prefill_model" if is_prefill else "decode_model",
            None,
        )
        session = getattr(active_model, "hmonnx_session", None)
        for owner in (session, active_model):
            if owner is None:
                continue
            get_names = getattr(owner, "get_output_names", None)
            if callable(get_names):
                names = tuple(str(name) for name in get_names())
                if len(names) == output_count:
                    return names
            names = getattr(owner, "output_names", None)
            if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
                normalized = tuple(str(name) for name in names)
                if len(normalized) == output_count:
                    return normalized
        return ()

    @staticmethod
    def _flatten_runtime_sequence_output(
        value: torch.Tensor,
    ) -> torch.Tensor:
        if value.ndim == 1:
            return value.unsqueeze(0)
        if value.ndim == 2:
            return value
        return value.reshape(-1, value.shape[-1])

    def _extract_request_result(
        self,
        runtime_model: Any,
        model_output: Any,
        *,
        current_input_length: int,
        hidden_output_name: str | None,
        require_all_logits: bool,
        allow_last_hidden_only: bool = False,
    ) -> MerakRequestResult:
        output_map: dict[str, Any] = {}
        if isinstance(model_output, Mapping):
            output_map = {str(name): value for name, value in model_output.items()}
            output_names = tuple(output_map)
            raw_outputs = tuple(output_map.values())
        else:
            raw_outputs = (
                tuple(model_output)
                if isinstance(model_output, (list, tuple))
                else (model_output,)
            )
            output_names = self._runtime_output_names(
                runtime_model,
                len(raw_outputs),
            )
            if output_names:
                output_map = dict(zip(output_names, raw_outputs, strict=True))

        logits_value = output_map.get("logits")
        if logits_value is None and hasattr(model_output, "logits"):
            logits_value = model_output.logits
        if logits_value is None and raw_outputs:
            logits_value = raw_outputs[0]
        if not isinstance(logits_value, torch.Tensor):
            raise TypeError(
                "Merak runtime did not return tensor logits: "
                f"{type(logits_value).__name__}"
            )
        vocab_size = self.logits_processor.vocab_size  # type: ignore[attr-defined]
        logits = self._flatten_runtime_sequence_output(logits_value)[:, :vocab_size]
        if require_all_logits and logits.shape[0] < current_input_length:
            raise RuntimeError(
                "Merak target verify graph must return one logits row per "
                f"input token: got {logits.shape[0]}, "
                f"required={current_input_length}"
            )
        if logits.shape[0] >= current_input_length:
            logits = logits[:current_input_length]

        hidden_states = None
        if hidden_output_name:
            hidden_value = output_map.get(hidden_output_name)
            if hidden_value is None:
                raise RuntimeError(
                    "Merak target graph does not expose required speculative "
                    f"hidden output {hidden_output_name!r}; "
                    f"outputs={list(output_names) if output_names else '<unnamed>'}"
                )
            if not isinstance(hidden_value, torch.Tensor):
                raise TypeError(
                    "Merak target speculative hidden output is not a tensor: "
                    f"name={hidden_output_name!r}, "
                    f"type={type(hidden_value).__name__}"
                )
            hidden_states = self._flatten_runtime_sequence_output(hidden_value)
            if hidden_states.shape[0] < current_input_length:
                if not (allow_last_hidden_only and hidden_states.shape[0] == 1):
                    raise RuntimeError(
                        "Merak target hidden output is shorter than the "
                        f"request: got {hidden_states.shape[0]}, "
                        f"required={current_input_length}"
                    )
            else:
                hidden_states = hidden_states[:current_input_length]

        return MerakRequestResult(
            logits=logits,
            hidden_states=hidden_states,
            output_names=output_names,
            raw_outputs=raw_outputs,
        )

    # ------------------------------------------------------------------
    # Multimodal helpers — text-only default
    # ------------------------------------------------------------------
    def _extract_request_multimodal_kwargs(
        self,
        *,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        image_grid_offset: int,
        video_grid_thw: Optional[torch.Tensor],
        video_grid_offset: int,
    ) -> tuple[dict[str, torch.Tensor], int, int]:
        del input_ids, inputs_embeds, image_grid_thw, video_grid_thw
        return {}, image_grid_offset, video_grid_offset

    def _prepare_runtime_data_processor_for_positions(
        self,
        data_preprocessor: Any,
        positions: torch.Tensor,
    ) -> None:
        del data_preprocessor, positions

    def _apply_runtime_positions(
        self,
        net_inputs: Any,
        positions: torch.Tensor,
        seq_len: int,
    ) -> Any:
        del positions, seq_len
        return net_inputs

    # ------------------------------------------------------------------
    # Attention metadata
    # ------------------------------------------------------------------
    def _uses_gemma4_page_attention_contract(self) -> bool:
        contract = getattr(self, "gemma4_cache_contract", None)
        return isinstance(contract, Gemma4CacheContract) and contract.version == 2

    def _gemma4_bindings_by_group(
        self,
    ) -> dict[int, tuple[Gemma4LayerCacheBinding, ...]]:
        if not self._uses_gemma4_page_attention_contract():
            raise RuntimeError("Gemma4 PageAttention routing requires contract-v2")
        bindings = tuple(getattr(self, "page_attention_bindings", ()))
        if not bindings:
            bindings = self.gemma4_cache_contract.bindings  # type: ignore[attr-defined]
        grouped: dict[int, list[Gemma4LayerCacheBinding]] = {}
        for binding in bindings:
            if binding.cache_group_idx is None:
                raise RuntimeError(f"Gemma4 binding {binding.layer_name} has no finalized cache_group_idx")
            grouped.setdefault(binding.cache_group_idx, []).append(binding)
        return {group_idx: tuple(values) for group_idx, values in grouped.items()}

    @staticmethod
    def _attention_metadata_compatible(
        left: MerakAttentionMetadata,
        right: MerakAttentionMetadata,
    ) -> bool:
        for field_name in (
            "past_lens",
            "subsequence_begins",
            "block_indices",
            "block_indices_begins",
            "slot_mapping",
        ):
            left_value = getattr(left, field_name)
            right_value = getattr(right, field_name)
            if (
                left_value.shape != right_value.shape
                or left_value.dtype != right_value.dtype
                or left_value.device != right_value.device
                or not torch.equal(left_value, right_value)
            ):
                return False
        return True

    def _get_attn_metadata_by_group(
        self,
    ) -> dict[int, MerakAttentionMetadata]:
        """Resolve Gemma4 attention metadata through immutable layer bindings."""

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            raise RuntimeError("Gemma4 contract-v2 requires layer-keyed attention metadata")

        metadata_by_group: dict[int, MerakAttentionMetadata] = {}
        for group_idx, bindings in self._gemma4_bindings_by_group().items():
            candidates: list[tuple[str, MerakAttentionMetadata]] = []
            for binding in bindings:
                candidate = attn_metadata.get(binding.layer_name)
                if candidate is None:
                    raise RuntimeError(
                        "Gemma4 attention metadata is missing exact binding: "
                        f"cache_group_idx={group_idx}, layer={binding.layer_name}"
                    )
                if not isinstance(candidate, MerakAttentionMetadata):
                    raise RuntimeError(
                        f"Gemma4 attention metadata type mismatch for {binding.layer_name}: {type(candidate).__name__}"
                    )
                candidates.append((binding.layer_name, candidate))
            if not candidates:
                raise RuntimeError(
                    f"Gemma4 attention metadata is missing exact binding for cache_group_idx={group_idx}"
                )
            canonical_name, canonical = candidates[0]
            for layer_name, candidate in candidates[1:]:
                if not self._attention_metadata_compatible(canonical, candidate):
                    raise RuntimeError(
                        "Gemma4 cache-group metadata mismatch: "
                        f"cache_group_idx={group_idx}, canonical={canonical_name}, "
                        f"incompatible={layer_name}"
                    )
            metadata_by_group[group_idx] = canonical
        return metadata_by_group

    def _build_page_attention_group_contexts(
        self,
        *,
        metadata_by_group: dict[int, MerakAttentionMetadata],
        query_lengths: torch.Tensor,
        visual_ranges: torch.Tensor | None = None,
    ) -> dict[int, MerakPageAttentionGroupContext]:
        """Build exact full/sliding PageAttention context tensors per group."""

        bindings_by_group = self._gemma4_bindings_by_group()
        if set(metadata_by_group) != set(bindings_by_group):
            missing = sorted(set(bindings_by_group) - set(metadata_by_group))
            extras = sorted(set(metadata_by_group) - set(bindings_by_group))
            raise RuntimeError(f"Gemma4 PageAttention metadata/group mismatch: missing={missing}, unexpected={extras}")

        query_lengths = query_lengths.reshape(-1).to(dtype=torch.int64)
        normalized_ranges: torch.Tensor | None = None
        if visual_ranges is not None:
            normalized_ranges = visual_ranges.to(dtype=torch.int32)
            if normalized_ranges.ndim == 2:
                normalized_ranges = normalized_ranges.unsqueeze(0)
            if normalized_ranges.ndim != 3 or normalized_ranges.shape[-1] != 2:
                raise ValueError("Gemma4 visual_ranges must have shape [batch, max_ranges, 2]")

        contexts: dict[int, MerakPageAttentionGroupContext] = {}
        for group_idx, bindings in bindings_by_group.items():
            metadata = metadata_by_group[group_idx]
            past_lens = metadata.past_lens.reshape(-1).to(dtype=torch.int64)
            if past_lens.numel() != query_lengths.numel():
                raise RuntimeError(f"Gemma4 cache_group_idx={group_idx} batch/query length mismatch")
            seq_lens = past_lens + query_lengths.to(device=past_lens.device)
            kinds = {binding.attention_kind for binding in bindings}
            windows = {binding.sliding_window for binding in bindings}
            if len(kinds) != 1 or len(windows) != 1:
                raise RuntimeError(f"Gemma4 cache_group_idx={group_idx} has inconsistent bindings")
            attention_kind = bindings[0].attention_kind
            scheduler_block_sizes = {binding.scheduler_block_size for binding in bindings}
            physical_block_sizes = {binding.physical_block_size for binding in bindings}
            page_ratios = {binding.physical_pages_per_scheduler_block for binding in bindings}
            if len(scheduler_block_sizes) != 1 or len(physical_block_sizes) != 1 or len(page_ratios) != 1:
                raise RuntimeError(
                    f"Gemma4 cache group has inconsistent virtual/physical page geometry: cache_group_idx={group_idx}"
                )
            scheduler_block_size = next(iter(scheduler_block_sizes))
            physical_block_size = next(iter(physical_block_sizes))
            page_ratio = next(iter(page_ratios))
            assert scheduler_block_size is not None

            fixed_physical_pages = bindings[0].fixed_physical_pages
            if attention_kind == "sliding_attention":
                if fixed_physical_pages is None or fixed_physical_pages <= 0:
                    raise RuntimeError(
                        f"Gemma4 sliding cache group has no fixed physical page capacity: cache_group_idx={group_idx}"
                    )
                if past_lens.numel() != 1:
                    raise RuntimeError(
                        "Gemma4 fixed sliding cache supports exactly one active sequence per HMONNX artifact"
                    )
                # vLLM's block table contains scheduler-pool IDs.  Those IDs
                # are global across heterogeneous cache groups and therefore
                # cannot index the compact (window + prefill chunk) backing.
                # Preserve the logical table length required by PageAttention,
                # but map each absolute logical page into the fixed ring.
                physical_block_ids = torch.arange(
                    metadata.block_indices.numel(),
                    dtype=torch.int64,
                    device=metadata.block_indices.device,
                ).remainder(fixed_physical_pages)

                past_values = past_lens.to(device="cpu").tolist()
                query_values = query_lengths.to(device="cpu").tolist()
                physical_slots = []
                for past_value, query_value in zip(past_values, query_values, strict=True):
                    positions = torch.arange(
                        int(past_value),
                        int(past_value) + int(query_value),
                        dtype=torch.int64,
                        device=metadata.slot_mapping.device,
                    )
                    physical_slots.append(
                        positions.div(physical_block_size, rounding_mode="floor").remainder(fixed_physical_pages)
                        * physical_block_size
                        + positions.remainder(physical_block_size)
                    )
                physical_slot_mapping = torch.cat(physical_slots)
            else:
                # MultiGroupBlockTable already maps each scheduler block to
                # ``page_ratio`` kernel blocks when scheduler_block_size differs
                # from the backend's physical block size.  Both block_indices
                # and slot_mapping therefore arrive in physical coordinates.
                physical_block_ids = metadata.block_indices.to(dtype=torch.int64).clone()
                physical_slot_mapping = metadata.slot_mapping.to(dtype=torch.int64).clone()

            # The flattened slot number is likewise already the exact physical
            # slot.  Keep it unchanged, but fail closed if the finalized
            # geometry ever stops satisfying the scheduler/kernel identity.
            if scheduler_block_size != page_ratio * physical_block_size:
                raise RuntimeError(
                    "Gemma4 slot mapping cannot be projected from scheduler to "
                    f"physical pages: cache_group_idx={group_idx}"
                )

            kv_window_start_abs: torch.Tensor | None = None
            mm_prefix_ranges: torch.Tensor | None = None
            # Gemma4's Transformers 5.13 mask contract applies bidirectional
            # visual blocks only inside sliding-attention layers.  Full
            # attention remains causal, so never publish the ranges to its
            # PageAttention context (including the legacy context fallback).
            if normalized_ranges is not None and attention_kind == "sliding_attention":
                ranges = normalized_ranges.to(device=past_lens.device)
                if ranges.shape[0] not in (1, past_lens.numel()):
                    raise ValueError("Gemma4 visual_ranges batch dimension does not match metadata")
                if ranges.shape[0] == 1 and past_lens.numel() != 1:
                    ranges = ranges.expand(past_lens.numel(), -1, -1)
                starts = ranges[..., 0].to(torch.int64)
                ends = ranges[..., 1].to(torch.int64)
                sentinels = (starts == 0) & (ends == 0)
                query_ends = past_lens + query_lengths.to(past_lens.device) - 1
                intersects = ~sentinels & (ends >= past_lens[:, None]) & (starts <= query_ends[:, None])
                filtered = torch.where(
                    intersects[..., None],
                    ranges,
                    torch.zeros_like(ranges),
                ).contiguous()
                if bool(intersects.any()):
                    mm_prefix_ranges = filtered
            if attention_kind == "sliding_attention":
                sliding_window = bindings[0].sliding_window
                if type(sliding_window) is not int or sliding_window <= 0:
                    raise RuntimeError(f"Gemma4 cache_group_idx={group_idx} has invalid sliding_window")
                kv_window_start_abs = torch.clamp(past_lens - sliding_window + 1, min=0)

            contexts[group_idx] = MerakPageAttentionGroupContext(
                block_ids=physical_block_ids,
                slot_mapping=physical_slot_mapping,
                seq_lens=seq_lens,
                kv_window_start_abs=kv_window_start_abs,
                mm_prefix_ranges=mm_prefix_ranges,
            )
        return contexts

    @staticmethod
    def _common_page_attention_metadata(
        metadata_by_group: dict[int, MerakAttentionMetadata],
    ) -> MerakAttentionMetadata:
        if not metadata_by_group:
            raise RuntimeError("Gemma4 contract-v2 has no metadata groups")
        canonical_group = min(metadata_by_group)
        canonical = metadata_by_group[canonical_group]
        for group_idx, metadata in metadata_by_group.items():
            for field_name in ("past_lens", "subsequence_begins"):
                if not torch.equal(getattr(canonical, field_name), getattr(metadata, field_name)):
                    raise RuntimeError(
                        "Gemma4 request layout differs across cache groups: "
                        f"canonical_group={canonical_group}, group={group_idx}, "
                        f"field={field_name}"
                    )
        return canonical

    @staticmethod
    def _slice_page_attention_metadata(
        metadata: MerakAttentionMetadata,
        req_idx: int,
    ) -> tuple[MerakAttentionMetadata, int]:
        query_bounds = metadata.subsequence_begins.to(device="cpu").tolist()
        block_bounds = metadata.block_indices_begins.to(device="cpu").tolist()
        query_start = int(query_bounds[req_idx])
        query_end = int(query_bounds[req_idx + 1])
        block_start = int(block_bounds[req_idx])
        block_end = int(block_bounds[req_idx + 1])
        query_length = query_end - query_start
        return (
            MerakAttentionMetadata(
                past_lens=metadata.past_lens[req_idx : req_idx + 1],
                subsequence_begins=torch.tensor(
                    [0, query_length],
                    dtype=metadata.subsequence_begins.dtype,
                    device=metadata.subsequence_begins.device,
                ),
                block_indices=metadata.block_indices[block_start:block_end],
                block_indices_begins=torch.tensor(
                    [0, block_end - block_start],
                    dtype=metadata.block_indices_begins.dtype,
                    device=metadata.block_indices_begins.device,
                ),
                max_context_len=metadata.max_context_len,
                slot_mapping=metadata.slot_mapping[query_start:query_end],
                sampled_token_indices=metadata.sampled_token_indices,
                seq_lens=(metadata.seq_lens[req_idx : req_idx + 1] if metadata.seq_lens is not None else None),
            ),
            query_length,
        )

    def _page_attention_cache_by_index(self) -> dict[int, Any]:
        placeholders = {str(layer.layer_name): layer for layer in getattr(self, "placeholder_attn_layers", ())}
        caches: dict[int, Any] = {}
        for bindings in self._gemma4_bindings_by_group().values():
            for binding in bindings:
                placeholder = placeholders.get(binding.layer_name)
                if placeholder is None:
                    raise RuntimeError(f"Gemma4 cache binding has no placeholder: {binding.layer_name}")
                cache = placeholder.kv_cache
                prior = caches.setdefault(binding.cache_index, cache)
                if prior is not cache:
                    raise RuntimeError(
                        "Gemma4 repeated cache_index does not share one cache handle: "
                        f"cache_index={binding.cache_index}"
                    )
        return caches

    @staticmethod
    def _page_attention_capacity(length: int) -> int:
        if length <= 1:
            return max(length, 1)
        return 1 << (length - 1).bit_length()

    def _clear_runtime_cuda_graph_for_group_buffer_growth(self) -> None:
        """Invalidate every graph that captured shared PageAttention buffers.

        Prefill and decode use distinct HMONNX interpreters, but their
        PageAttention modules consume the same Merak-owned staging tensors.
        Growing a tensor while prefill is active therefore invalidates both the
        active prefill graph and any decode graph captured by an earlier
        request.  Clearing only the active phase leaves decode replaying freed
        block-table addresses once a later, longer prompt crosses a capacity
        bucket.
        """

        runtime_model = cast(Any, self.inference_model)  # type: ignore[attr-defined]
        phase_models: list[Any] = []
        prefill_models = getattr(runtime_model, "prefill_models", None)
        if isinstance(prefill_models, Mapping):
            phase_models.extend(prefill_models.values())
        for attribute in ("prefill_model", "decode_model"):
            phase_model = getattr(runtime_model, attribute, None)
            if phase_model is not None:
                phase_models.append(phase_model)
        if not phase_models:
            phase_models.append(runtime_model)

        seen_phase_models: set[int] = set()
        for phase_model in phase_models:
            if id(phase_model) in seen_phase_models:
                continue
            seen_phase_models.add(id(phase_model))
            session = getattr(phase_model, "hmonnx_session", None)
            candidates = [
                phase_model,
                session,
                getattr(session, "interpreter", None) if session is not None else None,
            ]
            for candidate in candidates:
                if candidate is None:
                    continue
                clear = getattr(candidate, "clear", None)
                if callable(clear):
                    clear(clear_disabled_reason=True)
                    break
                clear_graph = getattr(candidate, "clear_cuda_graph_state", None)
                if callable(clear_graph):
                    clear_graph(clear_disabled_reason=True)
                    break

    def _stage_group_contexts(
        self,
        contexts: dict[int, MerakPageAttentionGroupContext],
    ) -> dict[int, MerakPageAttentionGroupContext]:
        """Copy group metadata into stable power-of-two device buffers."""

        buffers = getattr(self, "_page_attention_group_buffers", None)
        if not isinstance(buffers, dict):
            buffers = {}
            self._page_attention_group_buffers = buffers
        caches = self._page_attention_cache_by_index()
        group_bindings = self._gemma4_bindings_by_group()
        primary_device = torch.device(getattr(self, "inference_device", "cpu"))
        device_by_group: dict[int, torch.device] = {}
        for group_idx, bindings in group_bindings.items():
            owner_binding = bindings[0]
            owner_device = torch.device(getattr(caches[owner_binding.cache_index], "device", primary_device))
            for binding in bindings[1:]:
                binding_device = torch.device(getattr(caches[binding.cache_index], "device", primary_device))
                if binding_device != owner_device:
                    raise RuntimeError(
                        "Gemma4 one cache group cannot span multiple PageAttention devices: "
                        f"cache_group_idx={group_idx}, owner_cache_index="
                        f"{owner_binding.cache_index} on {owner_device}, cache_index="
                        f"{binding.cache_index} on {binding_device}"
                    )
            device_by_group[group_idx] = owner_device
        grew_existing_buffer = False

        def stage(
            group_idx: int,
            field_name: str,
            source: torch.Tensor,
            dtype: torch.dtype,
            pad_value: int,
        ) -> torch.Tensor:
            nonlocal grew_existing_buffer
            device = device_by_group[group_idx]
            flat_source = source.reshape(-1)
            length = int(flat_source.numel())
            capacity = self._page_attention_capacity(length)
            key = (group_idx, field_name)
            buffer = buffers.get(key)
            if (
                not isinstance(buffer, torch.Tensor)
                or buffer.device != device
                or buffer.dtype != dtype
                or buffer.numel() < capacity
            ):
                if isinstance(buffer, torch.Tensor):
                    grew_existing_buffer = True
                buffer = torch.empty(capacity, dtype=dtype, device=device)
                buffers[key] = buffer
            buffer.fill_(pad_value)
            buffer[:length].copy_(flat_source.to(device=device, dtype=dtype, non_blocking=True))
            # Multimodal ranges preserve their semantic [B, R, 2] shape while
            # using the same fixed-address CUDA Graph path as text.  Other
            # inputs expose the full bucket so both pointer and tensor metadata
            # remain stable until capacity growth.
            if source.ndim > 1:
                return buffer[:length].view(source.shape)
            return buffer

        staged: dict[int, MerakPageAttentionGroupContext] = {}
        for group_idx, context in contexts.items():
            staged[group_idx] = MerakPageAttentionGroupContext(
                block_ids=stage(group_idx, "block_ids", context.block_ids, torch.int64, 0),
                slot_mapping=stage(group_idx, "slot_mapping", context.slot_mapping, torch.int64, -1),
                seq_lens=stage(group_idx, "seq_lens", context.seq_lens, torch.int64, 0),
                kv_window_start_abs=(
                    stage(
                        group_idx,
                        "kv_window_start_abs",
                        context.kv_window_start_abs,
                        torch.int64,
                        0,
                    )
                    if context.kv_window_start_abs is not None
                    else None
                ),
                mm_prefix_ranges=(
                    stage(
                        group_idx,
                        "mm_prefix_ranges",
                        context.mm_prefix_ranges,
                        torch.int32,
                        0,
                    )
                    if context.mm_prefix_ranges is not None
                    else None
                ),
            )
        if grew_existing_buffer:
            self._clear_runtime_cuda_graph_for_group_buffer_growth()
        return staged

    @staticmethod
    def _page_attention_execution_mode(
        contexts: dict[int, MerakPageAttentionGroupContext],
    ) -> str:
        del contexts
        return "cuda_graph"

    def _select_page_attention_execution_mode(
        self,
        runtime_model: Any,
        contexts: dict[int, MerakPageAttentionGroupContext],
    ) -> str:
        mode = self._page_attention_execution_mode(contexts)
        setter = getattr(runtime_model, "set_page_attention_execution_mode", None)
        if callable(setter):
            setter(mode)
        else:
            # Sibling HMONNX contract-v2 runtimes inspect this per-step marker;
            # older runtimes that predate execution-mode routing ignore it.
            setattr(runtime_model, "_merak_page_attention_execution_mode", mode)
        return mode

    def _page_attention_visual_ranges(
        self,
        input_ids: torch.Tensor | None,
        *,
        past_len: int,
    ) -> torch.Tensor | None:
        """Return absolute inclusive multimodal ranges for one request."""

        if input_ids is None:
            return None
        config = getattr(self, "config", None)
        token_ids = {
            int(value)
            for value in (
                getattr(self, "image_token_id", None),
                getattr(self, "video_token_id", None),
                getattr(config, "image_token_id", None),
                getattr(config, "video_token_id", None),
            )
            if type(value) is int and value >= 0
        }
        if not token_ids:
            return None
        flat_ids = input_ids.reshape(-1)
        mask = torch.zeros_like(flat_ids, dtype=torch.bool)
        for token_id in token_ids:
            mask |= flat_ids == token_id
        flags = mask.to(device="cpu").tolist()
        ranges: list[tuple[int, int]] = []
        range_start: int | None = None
        for token_idx, is_visual in enumerate(flags):
            if is_visual and range_start is None:
                range_start = token_idx
            if range_start is not None and (token_idx == len(flags) - 1 or not flags[token_idx + 1]):
                ranges.append((past_len + range_start, past_len + token_idx))
                range_start = None
        if not ranges:
            return None
        return torch.tensor([ranges], dtype=torch.int32, device=flat_ids.device)

    def _gemma4_mm_token_type_ids(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct Gemma4's source-model multimodal type IDs.

        The exported preprocessor uses 1 for image, 2 for video and 0 for
        ordinary/audio tokens.  Keep this separate from visual-range planning:
        the former is the HMONNX input ABI, while the latter controls vLLM's
        atomic chunk and cache metadata.
        """

        flat_ids = input_ids.reshape(-1)
        token_types = torch.zeros_like(flat_ids, dtype=torch.long)
        image_token_id = int(getattr(self, "image_token_id", -1))
        video_token_id = int(getattr(self, "video_token_id", -1))
        if image_token_id >= 0:
            token_types.masked_fill_(flat_ids == image_token_id, 1)
        if video_token_id >= 0:
            token_types.masked_fill_(flat_ids == video_token_id, 2)
        return token_types.unsqueeze(0)

    @staticmethod
    def _gemma4_visual_ranges_from_group_contexts(
        contexts: dict[int, MerakPageAttentionGroupContext],
    ) -> tuple[tuple[int, int], ...]:
        """Read one canonical set of absolute ranges from group contexts."""

        canonical: tuple[tuple[int, int], ...] | None = None
        for context in contexts.values():
            tensor = context.mm_prefix_ranges
            if tensor is None:
                continue
            values = tuple(
                (int(start), int(end))
                for start, end in tensor.detach().to(device="cpu").reshape(-1, 2).tolist()
                if not (int(start) == 0 and int(end) == 0)
            )
            if canonical is None:
                canonical = values
            elif values != canonical:
                raise RuntimeError("Gemma4 visual ranges differ across PageAttention cache groups")
        return canonical or ()

    def _gemma4_max_mm_ranges_per_chunk(
        self,
        contexts: dict[int, MerakPageAttentionGroupContext] | None,
    ) -> int:
        configured = getattr(self, "max_mm_ranges_per_chunk", None)
        if configured is not None:
            capacity = int(configured)
            if capacity <= 0:
                raise ValueError(f"Gemma4 max_mm_ranges_per_chunk must be positive, got {configured!r}")
            return capacity
        # Compatibility for directly constructed test models. Production Gemma4
        # loaders always export max_mm_ranges_per_chunk from golden metadata.
        return max(
            (
                int(context.mm_prefix_ranges.shape[-2])
                for context in (contexts or {}).values()
                if context.mm_prefix_ranges is not None
            ),
            default=1,
        )

    def _plan_gemma4_atomic_request(
        self,
        *,
        past_len: int,
        num_tokens: int,
        visual_ranges: Sequence[tuple[int, int]] | torch.Tensor | None,
        page_attention_group_contexts: (dict[int, MerakPageAttentionGroupContext] | None) = None,
    ) -> tuple[
        tuple[tuple[int, int], ...],
        dict[tuple[int, int], torch.Tensor | None],
    ]:
        """Validate and plan one Gemma4 request without runtime side effects."""

        if not self._uses_gemma4_page_attention_contract():
            raise RuntimeError("Gemma4 atomic planning requires contract-v2")
        from .gemma4 import (
            _visual_ranges_for_chunk,
            plan_gemma4_prefill_chunks,
        )

        if isinstance(visual_ranges, torch.Tensor):
            absolute_ranges = tuple(
                (int(start), int(end))
                for start, end in visual_ranges.detach().to(device="cpu").reshape(-1, 2).tolist()
                if not (int(start) == 0 and int(end) == 0)
            )
        elif visual_ranges is not None:
            absolute_ranges = tuple(visual_ranges)
        elif page_attention_group_contexts is not None:
            absolute_ranges = self._gemma4_visual_ranges_from_group_contexts(page_attention_group_contexts)
        else:
            absolute_ranges = ()

        use_prefill = num_tokens > 1 or past_len == 0
        if use_prefill:
            query_end_abs = past_len + num_tokens
            local_ranges: list[tuple[int, int]] = []
            for visual_start, visual_end in absolute_ranges:
                if visual_start < past_len or visual_end >= query_end_abs:
                    raise RuntimeError(
                        "Gemma4 visual span must remain atomic; an upstream "
                        "scheduler cut the request at "
                        f"[{past_len}, {query_end_abs}) across inclusive range "
                        f"{(visual_start, visual_end)}"
                    )
                local_ranges.append((visual_start - past_len, visual_end - past_len))
            chunk_plan = plan_gemma4_prefill_chunks(
                num_tokens,
                self.gemma4_cache_contract.prefill_chunk_size,  # type: ignore[attr-defined]
                local_ranges,
            )
        else:
            chunk_plan = ((0, num_tokens),)

        max_ranges = self._gemma4_max_mm_ranges_per_chunk(page_attention_group_contexts)
        chunk_visual_ranges: dict[tuple[int, int], torch.Tensor | None] = {}
        for chunk_start, chunk_end in chunk_plan:
            chunk_start_abs = past_len + chunk_start
            chunk_end_abs = past_len + chunk_end
            chunk_visual_ranges[(chunk_start, chunk_end)] = _visual_ranges_for_chunk(
                chunk_start_abs,
                chunk_end_abs,
                absolute_ranges,
                max_ranges,
            )
        return chunk_plan, chunk_visual_ranges

    def _chunk_page_attention_group_contexts(
        self,
        contexts: dict[int, MerakPageAttentionGroupContext],
        *,
        chunk_start: int,
        chunk_end: int,
        chunk_past_len: int,
        context_length: int,
        chunk_visual_ranges: torch.Tensor | None,
    ) -> dict[int, MerakPageAttentionGroupContext]:
        block_size = self._get_block_size()
        real_chunk_length = chunk_end - chunk_start
        bindings_by_group = self._gemma4_bindings_by_group()
        chunk_contexts: dict[int, MerakPageAttentionGroupContext] = {}
        for group_idx, context in contexts.items():
            required_blocks = (chunk_past_len + context_length + block_size - 1) // block_size
            actual_required_blocks = (chunk_past_len + real_chunk_length + block_size - 1) // block_size
            block_ids = context.block_ids.reshape(-1)
            if block_ids.numel() < actual_required_blocks:
                raise RuntimeError(
                    "Gemma4 physical block projection is shorter than the HMONNX "
                    "request; repeating the final page would alias KV data: "
                    f"cache_group_idx={group_idx}, physical_blocks="
                    f"{block_ids.numel()}, required={actual_required_blocks}"
                )
            if block_ids.numel() < required_blocks:
                # A short final prefill chunk is padded to the fixed HMONNX
                # input length and all padded slot mappings are -1.  Extending
                # metadata for those unreachable lanes is safe; extending past
                # the real sequence length above is forbidden.
                padding = block_ids[-1:].expand(required_blocks - block_ids.numel())
                block_ids = torch.cat((block_ids, padding))
            slot_mapping = self._pad_page_attention_slot_mapping(
                context.slot_mapping.reshape(-1)[chunk_start:chunk_end],
                context_length,
            )
            bindings = bindings_by_group[group_idx]
            kv_window_start_abs: torch.Tensor | None = None
            mm_prefix_ranges = (
                None
                if (
                    chunk_visual_ranges is None
                    or bindings[0].attention_kind != "sliding_attention"
                )
                else chunk_visual_ranges.to(
                    device=context.seq_lens.device,
                    dtype=torch.int32,
                ).contiguous()
            )
            if bindings[0].attention_kind == "sliding_attention":
                sliding_window = cast(int, bindings[0].sliding_window)
                kv_window_start_abs = torch.tensor(
                    [max(0, chunk_past_len - sliding_window + 1)],
                    dtype=torch.int64,
                    device=context.seq_lens.device,
                )
            chunk_contexts[group_idx] = MerakPageAttentionGroupContext(
                block_ids=block_ids,
                slot_mapping=slot_mapping,
                seq_lens=torch.tensor(
                    [chunk_past_len + real_chunk_length],
                    dtype=torch.int64,
                    device=context.seq_lens.device,
                ),
                kv_window_start_abs=kv_window_start_abs,
                mm_prefix_ranges=mm_prefix_ranges,
            )
        return chunk_contexts

    def _get_attn_metadata(self) -> Optional[MerakAttentionMetadata]:
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, list):
            if not attn_metadata:
                return None
            attn_metadata = attn_metadata[0]

        if isinstance(attn_metadata, dict):
            layer_name = str(self.placeholder_attn_layers[0].layer_name)  # type: ignore[attr-defined]
            layer_metadata = attn_metadata.get(layer_name)
            if isinstance(layer_metadata, MerakAttentionMetadata):
                return layer_metadata
            return None

        if isinstance(attn_metadata, MerakAttentionMetadata):
            return attn_metadata
        return None

    # ------------------------------------------------------------------
    # Placeholder attention registration
    # ------------------------------------------------------------------
    def _register_placeholder_attention_layers(self, hf_model_config) -> None:
        if self.vllm_config is None:  # type: ignore[attr-defined]
            return

        num_heads = int(hf_model_config.num_attention_heads)
        num_kv_heads = int(getattr(hf_model_config, "num_key_value_heads", num_heads))
        head_size = int(getattr(hf_model_config, "head_dim", hf_model_config.hidden_size // num_heads))
        head_size_v = head_size
        # Cache width can differ from the source attention head width for an
        # absorbed MLA graph. Keep the model's original attention scaling;
        # exported cache metadata controls storage only.
        scale = 1.0 / math.sqrt(head_size)
        num_layers = int(hf_model_config.num_hidden_layers)
        contract = getattr(self, "gemma4_cache_contract", None)
        if isinstance(contract, Gemma4CacheContract):
            if len(contract.bindings) != num_layers:
                raise RuntimeError(
                    "Gemma4 placeholder count must match HF num_hidden_layers: "
                    f"{len(contract.bindings)} != {num_layers}"
                )
            for expected_layer_idx, binding in enumerate(contract.bindings):
                if binding.layer_idx != expected_layer_idx:
                    raise RuntimeError(
                        "Gemma4 cache bindings must be in exact layer order: "
                        f"expected layer {expected_layer_idx}, got {binding.layer_idx}"
                    )

            owner_layer_name_by_cache_index = {
                binding.cache_index: (
                    f"placeholder_attn_layers.{binding.cache_owner_layer_idx}"
                    if binding.cache_owner_layer_idx is not None
                    else binding.layer_name
                )
                for binding in contract.bindings
                if binding.cache_owner_layer_idx == binding.layer_idx or binding.cache_owner_layer_idx is None
            }
            missing_owners = sorted(set(contract.cache_shapes) - set(owner_layer_name_by_cache_index))
            if missing_owners:
                raise RuntimeError(f"Gemma4 cache bindings have no physical owner layers: {missing_owners}")

            def make_placeholder(binding: Gemma4LayerCacheBinding) -> MerakAttention:
                if len(binding.kv_shape) != 4:
                    raise RuntimeError(
                        "Gemma4 placeholder KV shape must be [B, H, S, D]: "
                        f"layer={binding.layer_name}, shape={binding.kv_shape}"
                    )
                binding_num_kv_heads = int(binding.kv_shape[1])
                binding_head_size = int(binding.kv_shape[-1])
                owner_layer_name = owner_layer_name_by_cache_index[binding.cache_index]
                sharing_target = None if binding.layer_name == owner_layer_name else owner_layer_name
                cache_config = self.vllm_config.cache_config  # type: ignore[attr-defined]
                if (
                    contract.attention_lowering == "full_flash_attention"
                    and binding.attention_kind == "sliding_attention"
                ):
                    cache_config = copy(cache_config)
                    # Sliding attention remains an ordinary FP MatMul/Softmax
                    # graph. Keep its paged storage in the same FP format so
                    # the cache-only bridge does not add an HMFP dequant pass.
                    cache_config.merak_logical_kv_cache_dtype = "float16"
                    cache_config.cache_dtype = "float16"
                return MerakAttention(
                    num_heads=num_heads,
                    num_kv_heads=binding_num_kv_heads,
                    head_size=binding_head_size,
                    scale=1.0 / math.sqrt(binding_head_size),
                    cache_config=cache_config,
                    prefix=binding.layer_name,
                    attn_backend=MerakAttentionBackend,
                    attention_kind=binding.attention_kind,
                    sliding_window=binding.sliding_window,
                    kv_sharing_target_layer_name=sharing_target,
                )

            self.placeholder_attn_layers = nn.ModuleList(make_placeholder(binding) for binding in contract.bindings)
            actual_layer_names = tuple(str(layer.layer_name) for layer in self.placeholder_attn_layers)
            expected_layer_names = tuple(binding.layer_name for binding in contract.bindings)
            if actual_layer_names != expected_layer_names:
                raise RuntimeError(
                    "Gemma4 placeholder layer names do not match cache contract: "
                    f"expected {expected_layer_names}, got {actual_layer_names}"
                )
            self.page_attention_bindings = contract.bindings
            logger.info(
                "Registered %d exact Gemma4 placeholder attention layers.",
                len(self.placeholder_attn_layers),
            )
            return

        layer_types = getattr(hf_model_config, "layer_types", None)
        if getattr(type(self), "is_hybrid", False):
            layer_types = self._resolve_hybrid_layer_types(hf_model_config)
        if isinstance(layer_types, list) and len(layer_types) == num_layers:
            attn_layer_indices = [
                layer_idx
                for layer_idx, layer_type in enumerate(layer_types)
                if str(layer_type).lower() == "full_attention"
            ]
            if not attn_layer_indices and not getattr(type(self), "is_hybrid", False):
                attn_layer_indices = list(range(num_layers))
        else:
            attn_layer_indices = list(range(num_layers))
        exported_key_shape = getattr(self, "key_cache_shape", None)
        exported_value_shape = getattr(self, "value_cache_shape", None)
        if isinstance(exported_key_shape, tuple) and isinstance(exported_value_shape, tuple):
            if len(exported_key_shape) != 4 or len(exported_value_shape) != 4:
                raise RuntimeError("Merak exported K/V cache shapes must be [B,H,S,D]")
            if exported_key_shape[:3] != exported_value_shape[:3]:
                raise RuntimeError("Merak exported K/V caches must share batch/head/sequence dimensions")
            num_kv_heads = int(exported_key_shape[1])
            head_size = int(exported_key_shape[-1])
            head_size_v = int(exported_value_shape[-1])

        self.placeholder_attn_layers = nn.ModuleList(
            MerakAttention(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                head_size_v=head_size_v,
                scale=scale,
                cache_config=self.vllm_config.cache_config,  # type: ignore[attr-defined]
                prefix=f"model.layers.{layer_idx}.attn",
                attn_backend=MerakAttentionBackend,
            )
            for layer_idx in attn_layer_indices
        )
        logger.info(
            f"Registered {len(self.placeholder_attn_layers)} placeholder attention layers for Merak model."  # type: ignore[attr-defined]
        )

    # ------------------------------------------------------------------
    # Runtime invocation
    # ------------------------------------------------------------------
    def _register_placeholder_linear_state_layers(self, hf_model_config) -> None:
        """Register MambaSpec state-holder layers for linear-attention layers.

        Only runs for hybrid Merak models (``is_hybrid=True``) that provide the
        ``get_mamba_state_shape_from_config`` / ``get_mamba_state_dtype_from_config``
        classmethods. Each linear-attention layer gets one
        ``MerakLinearStatePlaceholder`` so vLLM allocates a matching ``MambaSpec``
        state block, in HMONNX runtime layer order.
        """
        self.placeholder_linear_state_layers = nn.ModuleList()
        if self.vllm_config is None:  # type: ignore[attr-defined]
            return
        if not getattr(type(self), "is_hybrid", False):
            return

        get_shapes = getattr(type(self), "get_mamba_state_shape_from_config", None)
        get_dtypes = getattr(type(self), "get_mamba_state_dtype_from_config", None)
        if not callable(get_shapes) or not callable(get_dtypes):
            return

        layer_types = self._resolve_hybrid_layer_types(hf_model_config)
        linear_layer_indices = [
            layer_idx
            for layer_idx, layer_type in enumerate(layer_types)
            if layer_type == "linear_attention"
        ]

        if not linear_layer_indices:
            logger.info("No linear-attention layers detected for Merak hybrid model.")
            return

        from .merak_linear_state import MerakLinearStatePlaceholder

        shapes = tuple(get_shapes(self.vllm_config))  # type: ignore[attr-defined]
        dtypes = tuple(get_dtypes(self.vllm_config))  # type: ignore[attr-defined]

        self.placeholder_linear_state_layers = nn.ModuleList(
            MerakLinearStatePlaceholder(
                shapes=shapes,
                dtypes=dtypes,
                prefix=f"model.layers.{layer_idx}.linear_attn",
            )
            for layer_idx in linear_layer_indices
        )
        logger.info(
            f"Registered {len(self.placeholder_linear_state_layers)} placeholder linear-state "
            f"layers for Merak hybrid model (state tensors per layer: {len(shapes)})."
        )
        self._validate_hybrid_layer_contract(hf_model_config)

    def _resolve_hybrid_layer_types(self, hf_model_config: Any) -> list[str]:
        """Resolve one authoritative Full/Linear type for every hybrid layer.

        Explicit HF ``layer_types`` always wins and is validated strictly. A
        concrete loader may derive the list only when the source config omits
        the field; this keeps architecture-specific legacy metadata out of the
        shared Qwen hybrid path.
        """

        num_layers = int(hf_model_config.num_hidden_layers)
        layer_types = getattr(hf_model_config, "layer_types", None)
        derived = False
        if layer_types is None:
            derive_layer_types = getattr(
                type(self),
                "derive_missing_hybrid_layer_types",
                None,
            )
            if callable(derive_layer_types):
                layer_types = derive_layer_types(hf_model_config)
                derived = True

        if not isinstance(layer_types, list) or len(layer_types) != num_layers:
            raise RuntimeError(
                "Merak hybrid initialization requires HF layer_types for every layer; "
                f"expected {num_layers}, got "
                f"{len(layer_types) if isinstance(layer_types, list) else 'missing'}."
            )

        normalized_types = [str(layer_type).lower() for layer_type in layer_types]
        unknown_types = sorted(
            set(normalized_types) - {"full_attention", "linear_attention"}
        )
        if unknown_types:
            raise RuntimeError(
                "Merak hybrid HF metadata contains unsupported layer type(s): "
                + ", ".join(unknown_types)
            )
        if derived:
            logger.info(
                "Derived %s hybrid layer_types from architecture-owned metadata: "
                "Full=%d, Linear=%d.",
                type(self).__name__,
                normalized_types.count("full_attention"),
                normalized_types.count("linear_attention"),
            )
        return normalized_types

    def _validate_hybrid_layer_contract(self, hf_model_config: Any) -> None:
        normalized_types = self._resolve_hybrid_layer_types(hf_model_config)

        expected_full_names = [
            f"model.layers.{layer_idx}.attn"
            for layer_idx, layer_type in enumerate(normalized_types)
            if layer_type == "full_attention"
        ]
        expected_linear_names = [
            f"model.layers.{layer_idx}.linear_attn"
            for layer_idx, layer_type in enumerate(normalized_types)
            if layer_type == "linear_attention"
        ]
        actual_full_names = [str(layer.layer_name) for layer in self.placeholder_attn_layers]  # type: ignore[attr-defined]
        actual_linear_names = [str(layer.layer_name) for layer in self.placeholder_linear_state_layers]
        if actual_full_names != expected_full_names:
            raise RuntimeError(
                "Merak Full placeholder order does not match HF layer_types: "
                f"expected {expected_full_names}, got {actual_full_names}."
            )
        if actual_linear_names != expected_linear_names:
            raise RuntimeError(
                "Merak Linear placeholder order does not match HF layer_types: "
                f"expected {expected_linear_names}, got {actual_linear_names}."
            )

        golden_full_count = int(getattr(self, "num_layers", len(expected_full_names)))
        linear_config = getattr(getattr(self, "config", None), "linear_kv_cache_config", None)
        golden_linear_count = (
            int(linear_config.get("num_layers", len(expected_linear_names)))
            if isinstance(linear_config, dict)
            else len(expected_linear_names)
        )
        if golden_full_count != len(expected_full_names) or golden_linear_count != len(expected_linear_names):
            raise RuntimeError(
                "Merak HF/golden hybrid layer count mismatch: "
                f"HF Full={len(expected_full_names)}, Linear={len(expected_linear_names)}; "
                f"golden Full={golden_full_count}, Linear={golden_linear_count}."
            )

    def _run_request(
        self,
        past_len: int,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        video_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        page_attention_block_ids: tuple[int, ...] | None = None,
        page_attention_slot_mapping: torch.Tensor | None = None,
        page_attention_group_contexts: dict[int, MerakPageAttentionGroupContext] | None = None,
        visual_ranges: Sequence[tuple[int, int]] | torch.Tensor | None = None,
        request_id: str | None = None,
        execution_phase: MerakExecutionPhase | None = None,
        return_result: bool = False,
        target_hidden_output_name: str | None = None,
    ) -> torch.Tensor | MerakRequestResult:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Merak request needs either input_ids or inputs_embeds.")

        trace_prefill_request = determinism_trace_enabled() and past_len == 0
        if trace_prefill_request:
            emit_determinism_event(
                "merak_request_inputs",
                request_id=request_id,
                payload={
                    "past_len": past_len,
                    "input_ids": determinism_fingerprint(input_ids),
                    "inputs_embeds": determinism_fingerprint(inputs_embeds),
                    "positions": determinism_fingerprint(positions),
                    "image_embeds": determinism_fingerprint(image_embeds),
                    "image_grid_thw": determinism_fingerprint(image_grid_thw),
                    "page_attention_block_ids": list(page_attention_block_ids or ()),
                    "page_attention_slot_mapping": determinism_fingerprint(
                        page_attention_slot_mapping
                    ),
                },
            )

        runtime_model = cast(Any, self.inference_model)  # type: ignore[attr-defined]
        runtime_dtype = self.input_embeddings.weight.dtype  # type: ignore[attr-defined]
        runtime_device = self.inference_device  # type: ignore[attr-defined]

        flat_input_ids = self._flatten_input_ids(input_ids) if input_ids is not None else None
        use_raw_input_ids = flat_input_ids is not None and (
            self._prefer_input_ids_runtime
            or image_embeds is not None
            or video_embeds is not None
            or audio_embeds is not None
            or image_grid_thw is not None
            or video_grid_thw is not None
            or inputs_embeds is None
        )

        if use_raw_input_ids:
            assert flat_input_ids is not None
            num_tokens = flat_input_ids.shape[0]
            # Return logits/hidden states to the caller's original device even
            # though the HMONNX runtime may execute on a different device.
            output_device = flat_input_ids.device
        else:
            assert inputs_embeds is not None
            num_tokens = self._flatten_inputs_embeds(inputs_embeds).shape[0]

        if execution_phase is None:
            execution_phase = "prefill" if num_tokens > 1 or past_len == 0 else "decode"
        if execution_phase not in {"prefill", "decode", "verify"}:
            raise ValueError(f"Unsupported Merak execution phase: {execution_phase!r}")
        use_prefill = execution_phase == "prefill"
        if execution_phase == "verify" and past_len <= 0:
            raise ValueError("Merak target verify requires a non-empty prefix")
        uses_gemma4_atomic_chunks = self._uses_gemma4_page_attention_contract()
        trace_gemma4 = uses_gemma4_atomic_chunks and trace_enabled(GEMMA4_TRACE_ENV)
        if uses_gemma4_atomic_chunks:
            if page_attention_block_ids is not None:
                raise RuntimeError(
                    "Gemma4 group contexts cannot be combined with the Qwen positional PageAttention ABI"
                )
            chunk_plan, gemma_chunk_visual_ranges = self._plan_gemma4_atomic_request(
                past_len=past_len,
                num_tokens=num_tokens,
                visual_ranges=visual_ranges,
                page_attention_group_contexts=page_attention_group_contexts,
            )
            if trace_gemma4:
                if isinstance(visual_ranges, torch.Tensor):
                    if visual_ranges.device.type != "cpu":
                        raise RuntimeError("Gemma4 visual range evidence must remain host-resident")
                    inclusive_mm_ranges = [
                        [int(start), int(end)]
                        for start, end in visual_ranges.detach().reshape(-1, 2).tolist()
                        if not (int(start) == 0 and int(end) == 0)
                    ]
                elif visual_ranges is not None:
                    inclusive_mm_ranges = [[int(start), int(end)] for start, end in visual_ranges]
                elif page_attention_group_contexts is not None:
                    inclusive_mm_ranges = [
                        [int(start), int(end)]
                        for start, end in self._gemma4_visual_ranges_from_group_contexts(page_attention_group_contexts)
                    ]
                else:
                    inclusive_mm_ranges = []
                emit_trace(
                    GEMMA4_TRACE_ENV,
                    component="model",
                    event="request_plan",
                    payload={
                        "request_id": request_id or "__direct__",
                        "phase": execution_phase,
                        "P": past_len,
                        "Q": num_tokens,
                        "chunk_plan": [list(chunk) for chunk in chunk_plan],
                        "inclusive_mm_ranges": inclusive_mm_ranges,
                    },
                )
        else:
            chunk_plan = ()
            gemma_chunk_visual_ranges = {}

        request_embeds: Optional[torch.Tensor] = None
        if not use_raw_input_ids:
            request_embeds = self._prepare_request_embeds(input_ids=input_ids, inputs_embeds=inputs_embeds)
            output_device = request_embeds.device

        if use_prefill:
            runtime_model.set_prefill()
        else:
            runtime_model.set_decode()

        runtime_input_sequence_length = max(1, int(runtime_model.get_input_sequence_length()))
        if uses_gemma4_atomic_chunks:
            max_chunk_len = 320
            if use_prefill and runtime_input_sequence_length < max_chunk_len:
                raise ValueError(
                    "Gemma4 atomic visual prefill requires runtime input sequence "
                    f"length >= 320, got {runtime_input_sequence_length}"
                )
        else:
            max_chunk_len = runtime_input_sequence_length
            chunk_plan = tuple(
                (chunk_start, min(chunk_start + max_chunk_len, num_tokens))
                for chunk_start in range(0, num_tokens, max_chunk_len)
            )
        last_token_logits: torch.Tensor | None = None
        result_logits: list[torch.Tensor] = []
        result_hidden_states: list[torch.Tensor] = []
        last_request_result: MerakRequestResult | None = None
        self._ensure_runtime_cache_tensors()
        data_preprocessor = runtime_model.get_data_preprocessor()
        routed_embeddings = {
            "image": (
                int(getattr(self, "image_token_id", -1)),
                image_embeds,
            ),
            "video": (
                int(getattr(self, "video_token_id", -1)),
                video_embeds,
            ),
            "audio": (
                int(getattr(self, "audio_token_id", -1)),
                audio_embeds,
            ),
        }
        embedding_offsets = {modality: 0 for modality in routed_embeddings}
        for chunk_start, chunk_end in chunk_plan:
            chunk_past_len = past_len + chunk_start
            execution_mode: str | None = None
            staged_group_contexts: dict[int, MerakPageAttentionGroupContext] | None = None
            chunk_positions = positions[..., chunk_start:chunk_end] if positions is not None else None
            data_batch: dict[str, Any] = {
                "past_seq_length": chunk_past_len,
            }
            if page_attention_group_contexts is not None:
                # PageAttention metadata follows the physical HMONNX graph
                # capacity, not only the logical number of tokens in this
                # scheduler step.  Speculative target artifacts reuse their
                # fixed K+1 decode graph for ordinary Q=1 decode (for example
                # when no draft fits near max_model_len).  Keep the unused
                # tail as invalid slots so cache writers never read past a
                # one-element mapping and never commit padded KV entries.
                context_length = max_chunk_len
                chunk_group_contexts = self._chunk_page_attention_group_contexts(
                    page_attention_group_contexts,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    chunk_past_len=chunk_past_len,
                    context_length=context_length,
                    chunk_visual_ranges=gemma_chunk_visual_ranges[(chunk_start, chunk_end)],
                )
                staged_group_contexts = self._stage_group_contexts(chunk_group_contexts)
                execution_mode = self._select_page_attention_execution_mode(runtime_model, staged_group_contexts)
                self._set_runtime_page_attention_group_contexts(
                    runtime_model,
                    staged_group_contexts,
                )
            elif page_attention_block_ids is not None:
                if page_attention_slot_mapping is None:
                    raise ValueError(
                        "page_attention_slot_mapping is required when page_attention_block_ids is set."
                    )
                # See the group-context branch above: a fixed-width target
                # graph still produces K+1 physical KV rows during logical
                # Q=1 decode, so slot_mapping must cover that full capacity
                # with -1 padding.
                context_length = max_chunk_len
                block_size = self._get_block_size()
                required_blocks = (chunk_past_len + context_length + block_size - 1) // block_size
                if use_prefill:
                    # Prefill has padded HMONNX chunks and is not the stable decode
                    # replay path. Keep the historical fixed table here; bucketed
                    # prefill tables make 256k short-prompt TTFT regress badly.
                    block_table_capacity = max(
                        required_blocks,
                        self._page_attention_max_block_table_capacity(block_size),
                    )
                else:
                    block_table_capacity = self._page_attention_block_table_capacity(block_size, required_blocks)
                    self._clear_runtime_cuda_graph_for_page_attention_capacity_change(
                        runtime_model,
                        block_table_capacity,
                    )
                context_block_ids = self._pad_page_attention_block_ids(page_attention_block_ids, block_table_capacity)
                context_slot_mapping = self._pad_page_attention_slot_mapping(
                    page_attention_slot_mapping[chunk_start:chunk_end],
                    context_length,
                )
                self._set_runtime_page_attention_context(
                    runtime_model,
                    context_block_ids,
                    context_slot_mapping,
                    effective_kv_length=chunk_past_len + (chunk_end - chunk_start),
                )
            if use_raw_input_ids:
                assert flat_input_ids is not None
                chunk_input_ids = flat_input_ids[chunk_start:chunk_end].to(device=runtime_device)
                data_batch["input_ids"] = chunk_input_ids.unsqueeze(0)
                if uses_gemma4_atomic_chunks:
                    # Gemma4's compact Flash/PageAttention ABI derives the
                    # bidirectional image/video ranges from
                    # ``mm_token_type_ids`` inside its data preprocessor.
                    # vLLM normally carries only placeholder token IDs and
                    # merged embeddings at this point, so reconstruct the
                    # exact source-model type contract here.  Omitting this
                    # tensor silently turns every visual range into [0, 0]:
                    # the image features are then present, but sliding layers
                    # attend to them causally instead of blockwise.
                    data_batch["mm_token_type_ids"] = (
                        self._gemma4_mm_token_type_ids(chunk_input_ids)
                    )
                for modality, (token_id, embeddings) in routed_embeddings.items():
                    if embeddings is None:
                        continue
                    num_chunk_tokens = int((chunk_input_ids == token_id).sum().item()) if token_id >= 0 else 0
                    if num_chunk_tokens > 0:
                        offset = embedding_offsets[modality]
                        next_offset = offset + num_chunk_tokens
                        chunk_embeddings = embeddings[offset:next_offset]
                        if chunk_embeddings.shape[0] != num_chunk_tokens:
                            raise ValueError(
                                f"Merak {modality} embedding/token count mismatch while routing "
                                f"prefill chunk [{chunk_start}:{chunk_end}]."
                            )
                        data_batch[f"{modality}_embeds"] = chunk_embeddings.to(
                            device=runtime_device, dtype=runtime_dtype
                        )
                        embedding_offsets[modality] = next_offset
                if chunk_positions is None and chunk_past_len == 0:
                    if image_grid_thw is not None:
                        data_batch["image_grid_thw"] = image_grid_thw.to(device=runtime_device)
                    if video_grid_thw is not None:
                        data_batch["video_grid_thw"] = video_grid_thw.to(device=runtime_device)
            else:
                assert request_embeds is not None
                chunk_embeds = request_embeds[chunk_start:chunk_end].to(device=runtime_device, dtype=runtime_dtype)
                data_batch["inputs_embeds"] = self._restore_runtime_batch_shape(chunk_embeds)

            if chunk_positions is not None:
                self._prepare_runtime_data_processor_for_positions(
                    data_preprocessor,
                    chunk_positions,
                )
            net_inputs = data_preprocessor(data_batch)
            if chunk_positions is not None:
                net_inputs = self._apply_runtime_positions(
                    net_inputs,
                    chunk_positions,
                    chunk_end - chunk_start,
                )

            trace_prefill_chunk = trace_prefill_request and chunk_past_len == 0
            if trace_prefill_chunk:
                emit_determinism_event(
                    "hmonnx_prefill_inputs",
                    request_id=request_id,
                    payload={
                        "chunk": [chunk_start, chunk_end],
                        "data_batch": determinism_fingerprint(data_batch),
                        "net_inputs": determinism_fingerprint_named_tensors(net_inputs),
                    },
                )
            # The runtime has two physical HMONNX graphs (prefill/decode),
            # while ``verify`` is a logical execution phase on the decode
            # graph.  Per-node tracing is installed against the physical
            # graph name, so normalize verify here instead of silently
            # recording an empty trace.
            hmonnx_trace_stage = (
                "decode" if execution_phase == "verify" else execution_phase
            )
            page_verify_started = False
            if execution_phase == "verify" and self.enable_page_attention:
                if chunk_start != 0 or chunk_end != num_tokens:
                    raise RuntimeError(
                        "Merak target PageAttention verify must execute as one "
                        "atomic HMONNX chunk"
                    )
                if request_id is None:
                    raise RuntimeError(
                        "Merak target PageAttention verify requires request ownership"
                    )
                self._begin_page_attention_verify_transaction(
                    runtime_model,
                    request_id=request_id,
                    query_length=chunk_end - chunk_start,
                )
                page_verify_started = True
            try:
                with hmonnx_op_trace_scope(
                    hmonnx_trace_stage,
                    request_id=request_id,
                ):
                    model_output = runtime_model.forward(*net_inputs)
                chunk_result = self._extract_request_result(
                    runtime_model,
                    model_output,
                    current_input_length=chunk_end - chunk_start,
                    hidden_output_name=target_hidden_output_name,
                    require_all_logits=execution_phase == "verify",
                    # Gemma4's target prefill ABI deliberately exports only the
                    # final target hidden state.  That is sufficient for its
                    # read-only MTP draft, which consumes the target
                    # PageAttention cache directly. Verify still must return
                    # all K+1 rows so acceptance can select the committed
                    # hidden state.
                    allow_last_hidden_only=(
                        uses_gemma4_atomic_chunks
                        and execution_phase == "prefill"
                    ),
                )
            except Exception:
                if page_verify_started:
                    self._abort_page_attention_verify_transaction(
                        request_id=request_id,
                    )
                raise
            last_request_result = chunk_result
            last_token_logits = chunk_result.logits[-1]
            if return_result:
                result_logits.append(chunk_result.logits)
                if chunk_result.hidden_states is not None:
                    result_hidden_states.append(chunk_result.hidden_states)
            if trace_prefill_chunk:
                emit_determinism_event(
                    "hmonnx_prefill_outputs",
                    request_id=request_id,
                    payload={
                        "model_output": determinism_fingerprint(model_output),
                        "last_token_logits": determinism_fingerprint(last_token_logits),
                        "argmax_token_id": int(last_token_logits.argmax().item()),
                    },
                )
            if trace_gemma4 and staged_group_contexts is not None:
                runtime_scalar_inputs = {
                    str(index): tensor.detach().reshape(-1).tolist()
                    for index, tensor in enumerate(net_inputs)
                    if isinstance(tensor, torch.Tensor) and not tensor.dtype.is_floating_point and tensor.numel() <= 4
                }
                graph_runtime: dict[str, Any] = {}
                execution_state = getattr(runtime_model, "get_page_attention_execution_state", None)
                if callable(execution_state):
                    reported_state = execution_state()
                    if isinstance(reported_state, Mapping):
                        graph_runtime.update(
                            {
                                str(key): value
                                for key, value in reported_state.items()
                                if isinstance(value, (bool, int, float, str)) or value is None
                            }
                        )
                runtime_owners = [
                    runtime_model,
                    getattr(runtime_model, "hmonnx_session", None),
                    getattr(runtime_model, "prefill_model", None),
                    getattr(runtime_model, "decode_model", None),
                ]
                graph_fields = (
                    "clear_count",
                    "capture_count",
                    "replay_count",
                    "cuda_graph_capture_count",
                    "cuda_graph_replay_count",
                    "graph_capture_count",
                    "graph_replay_count",
                    "cuda_graph_state",
                )
                for owner in runtime_owners:
                    if owner is None:
                        continue
                    for field_name in graph_fields:
                        if field_name in graph_runtime:
                            continue
                        try:
                            value = getattr(owner, field_name)
                        except (AttributeError, RuntimeError):
                            continue
                        if isinstance(value, (bool, int, float, str)) or value is None:
                            graph_runtime[field_name] = value

                bindings_by_group = self._gemma4_bindings_by_group()
                caches = self._page_attention_cache_by_index()
                for group_idx, context in staged_group_contexts.items():
                    bindings = bindings_by_group[group_idx]
                    cache_indices = sorted({int(binding.cache_index) for binding in bindings})
                    backing_ptrs = {}
                    backing_bytes = {}
                    for cache_index in cache_indices:
                        cache = caches[cache_index]
                        tensors = tuple(getattr(cache, "tensors", lambda: ())())
                        if tensors:
                            backing_ptrs[str(cache_index)] = int(tensors[0].data_ptr())
                            backing_bytes[str(cache_index)] = sum(
                                tensor.numel() * tensor.element_size() for tensor in tensors
                            )

                    mm_prefix_ranges = None if context.mm_prefix_ranges is None else context.mm_prefix_ranges.tolist()

                    emit_trace(
                        GEMMA4_TRACE_ENV,
                        component="model",
                        event="chunk_context",
                        payload={
                            "request_id": request_id or "__direct__",
                            "phase": execution_phase,
                            "P": chunk_past_len,
                            "Q": chunk_end - chunk_start,
                            "chunk": [chunk_start, chunk_end],
                            "chunk_plan": [list(chunk) for chunk in chunk_plan],
                            "cache_group_idx": group_idx,
                            "cache_indices": cache_indices,
                            "attention_kind": bindings[0].attention_kind,
                            "sliding_window": bindings[0].sliding_window,
                            "kv_window_start_abs": (
                                None if context.kv_window_start_abs is None else context.kv_window_start_abs.tolist()
                            ),
                            "seq_lens": context.seq_lens.tolist(),
                            "block_ids_summary": {
                                "count": int(context.block_ids.numel()),
                                "head": context.block_ids[:4].tolist(),
                                "tail": context.block_ids[-4:].tolist(),
                            },
                            "slot_mapping_summary": {
                                "count": int(context.slot_mapping.numel()),
                                "valid_count": int(torch.count_nonzero(context.slot_mapping >= 0).item()),
                                "head": context.slot_mapping[:4].tolist(),
                                "tail": context.slot_mapping[-4:].tolist(),
                            },
                            "runtime_scalar_inputs": runtime_scalar_inputs,
                            "mm_prefix_ranges": mm_prefix_ranges,
                            "execution_mode": execution_mode,
                            "metadata_ptrs": {
                                "block_ids": int(context.block_ids.data_ptr()),
                                "slot_mapping": int(context.slot_mapping.data_ptr()),
                                "seq_lens": int(context.seq_lens.data_ptr()),
                                "kv_window_start_abs": (
                                    None
                                    if context.kv_window_start_abs is None
                                    else int(context.kv_window_start_abs.data_ptr())
                                ),
                                "mm_prefix_ranges": (
                                    None
                                    if context.mm_prefix_ranges is None
                                    else int(context.mm_prefix_ranges.data_ptr())
                                ),
                            },
                            "backing_ptrs": backing_ptrs,
                            "backing_bytes": backing_bytes,
                            "graph_runtime": graph_runtime,
                        },
                    )

        for modality, (_, embeddings) in routed_embeddings.items():
            if embeddings is not None and embedding_offsets[modality] != embeddings.shape[0]:
                raise ValueError(
                    f"Merak did not route all {modality} embeddings across prefill "
                    f"chunks: routed={embedding_offsets[modality]}, "
                    f"total={embeddings.shape[0]}."
                )

        if last_token_logits is None:
            raise RuntimeError(
                "Merak runtime produced no logits for the current request."
            )
        if not return_result:
            return last_token_logits.to(device=output_device)
        assert last_request_result is not None
        if (
            uses_gemma4_atomic_chunks
            and execution_phase == "prefill"
            and last_request_result.hidden_states is not None
        ):
            result_hidden = last_request_result.hidden_states[-1:]
        else:
            result_hidden = (
                torch.cat(result_hidden_states, dim=0) if result_hidden_states else None
            )
        return MerakRequestResult(
            logits=torch.cat(result_logits, dim=0).to(device=output_device),
            hidden_states=(
                result_hidden.to(device=output_device)
                if result_hidden is not None
                else None
            ),
            output_names=last_request_result.output_names,
            raw_outputs=last_request_result.raw_outputs,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("MerakForCausalLM.forward needs either input_ids or inputs_embeds.")

        del intermediate_tensors

        uses_gemma4_group_contexts = self._uses_gemma4_page_attention_contract()
        metadata_by_group: dict[int, MerakAttentionMetadata] = {}
        if uses_gemma4_group_contexts:
            metadata_by_group = self._get_attn_metadata_by_group()
            attn_metadata = self._common_page_attention_metadata(metadata_by_group)
        else:
            attn_metadata = self._get_attn_metadata()
        flat_input_ids = self._flatten_input_ids(input_ids) if input_ids is not None else None
        prevalidated_visual_ranges: dict[int, torch.Tensor | None] = {}
        gemma_past_lens: list[int] | None = None
        gemma_query_bounds: list[int] | None = None
        if uses_gemma4_group_contexts and attn_metadata is not None:
            gemma_past_lens = [int(value) for value in attn_metadata.past_lens.to(device="cpu").tolist()]
            gemma_query_bounds = [int(value) for value in attn_metadata.subsequence_begins.to(device="cpu").tolist()]
            for req_idx, past_len in enumerate(gemma_past_lens):
                query_start = gemma_query_bounds[req_idx]
                query_end = gemma_query_bounds[req_idx + 1]
                query_len = query_end - query_start
                if query_len <= 0:
                    continue
                request_input_ids = flat_input_ids[query_start:query_end] if flat_input_ids is not None else None
                request_ranges = self._page_attention_visual_ranges(
                    request_input_ids,
                    past_len=past_len,
                )
                self._plan_gemma4_atomic_request(
                    past_len=past_len,
                    num_tokens=query_len,
                    visual_ranges=request_ranges,
                )
                prevalidated_visual_ranges[req_idx] = request_ranges
        flat_inputs_embeds = self._prepare_request_embeds(input_ids=input_ids, inputs_embeds=inputs_embeds)

        device = flat_inputs_embeds.device
        flat_logits = torch.zeros(
            flat_inputs_embeds.shape[0],
            self.logits_processor.vocab_size,  # type: ignore[attr-defined]
            device=device,
            dtype=self.input_embeddings.weight.dtype,  # type: ignore[attr-defined]
        )

        self._ensure_runtime_cache_tensors()

        if len(self.placeholder_attn_layers) == 0:  # type: ignore[attr-defined]
            return flat_logits

        first_layer_kv_cache = cast(Tensor, self.placeholder_attn_layers[0].kv_cache)  # type: ignore[attr-defined]
        if attn_metadata is None or first_layer_kv_cache.numel() == 0:
            return flat_logits

        image_grid_thw = kwargs.get("image_grid_thw")
        video_grid_thw = kwargs.get("video_grid_thw")
        image_grid_offset = 0
        video_grid_offset = 0

        num_reqs = int(attn_metadata.past_lens.shape[0])
        # Hoist all D2H syncs out of the per-request loop — one .tolist() per
        # tensor instead of 6 .item() calls per request.
        past_lens_list = gemma_past_lens if gemma_past_lens is not None else attn_metadata.past_lens.tolist()
        subsequence_begins_list = (
            gemma_query_bounds if gemma_query_bounds is not None else attn_metadata.subsequence_begins.tolist()
        )
        block_indices_begins_list = attn_metadata.block_indices_begins.tolist()
        block_indices_list = attn_metadata.block_indices.tolist()
        slot_mapping = attn_metadata.slot_mapping

        for req_idx in range(num_reqs):
            past_len = int(past_lens_list[req_idx])
            query_start = int(subsequence_begins_list[req_idx])
            query_end = int(subsequence_begins_list[req_idx + 1])
            query_len = query_end - query_start
            if query_len <= 0:
                continue

            block_begin = int(block_indices_begins_list[req_idx])
            block_end = int(block_indices_begins_list[req_idx + 1])
            block_ids = tuple(int(block_id) for block_id in block_indices_list[block_begin:block_end])
            page_attention_block_ids = block_ids if getattr(self, "enable_page_attention", False) else None
            page_attention_slot_mapping = slot_mapping[query_start:query_end] if page_attention_block_ids else None
            if page_attention_block_ids is None:
                self._load_runtime_request_cache_from_paged(block_ids, past_len)

            request_input_ids = flat_input_ids[query_start:query_end] if flat_input_ids is not None else None
            request_embeds = flat_inputs_embeds[query_start:query_end]
            request_positions = positions[..., query_start:query_end] if positions is not None else None
            request_mm_kwargs, image_grid_offset, video_grid_offset = self._extract_request_multimodal_kwargs(
                input_ids=request_input_ids,
                inputs_embeds=request_embeds,
                image_grid_thw=cast(Optional[torch.Tensor], image_grid_thw),
                image_grid_offset=image_grid_offset,
                video_grid_thw=cast(Optional[torch.Tensor], video_grid_thw),
                video_grid_offset=video_grid_offset,
            )
            page_attention_group_contexts: dict[int, MerakPageAttentionGroupContext] | None = None
            request_visual_ranges = prevalidated_visual_ranges.get(req_idx) if uses_gemma4_group_contexts else None
            if uses_gemma4_group_contexts and page_attention_block_ids is not None:
                request_metadata_by_group: dict[int, MerakAttentionMetadata] = {}
                for group_idx, group_metadata in metadata_by_group.items():
                    sliced_metadata, group_query_length = self._slice_page_attention_metadata(group_metadata, req_idx)
                    if group_query_length != query_len:
                        raise RuntimeError(
                            f"Gemma4 query length differs across cache groups: cache_group_idx={group_idx}"
                        )
                    request_metadata_by_group[group_idx] = sliced_metadata
                page_attention_group_contexts = self._build_page_attention_group_contexts(
                    metadata_by_group=request_metadata_by_group,
                    query_lengths=torch.tensor([query_len], dtype=torch.int64, device=device),
                    visual_ranges=request_visual_ranges,
                )
            last_token_logits = self._run_request(
                past_len,
                input_ids=request_input_ids,
                inputs_embeds=request_embeds,
                positions=request_positions,
                page_attention_block_ids=(page_attention_block_ids if not uses_gemma4_group_contexts else None),
                page_attention_slot_mapping=(page_attention_slot_mapping if not uses_gemma4_group_contexts else None),
                page_attention_group_contexts=page_attention_group_contexts,
                visual_ranges=request_visual_ranges,
                request_id=f"__merak_model_req_{req_idx}",
                **request_mm_kwargs,
            )
            if page_attention_block_ids is None:
                self._scatter_runtime_request_cache_to_paged(block_ids, past_len, past_len + query_len)
            flat_logits[query_end - 1].copy_(last_token_logits.to(flat_logits.dtype))

        return flat_logits

    # ------------------------------------------------------------------
    # Logits / sampling
    # ------------------------------------------------------------------
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata] = None,
    ) -> Optional[torch.Tensor]:
        if sampling_metadata is None:
            return hidden_states
        return self.logits_processor(None, hidden_states, sampling_metadata)  # type: ignore[attr-defined]

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.sampler(logits, sampling_metadata)  # type: ignore[attr-defined]
