from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterator, Literal, Mapping

AttentionKind = Literal["full_attention", "sliding_attention"]
_SUPPORTED_ATTENTION_KINDS = frozenset({"full_attention", "sliding_attention"})


class _FrozenCacheShapes(Mapping[int, tuple[int, ...]]):
    def __init__(self, values: Mapping[int, tuple[int, ...]]) -> None:
        self._items = tuple(sorted(values.items()))

    def __getitem__(self, key: int) -> tuple[int, ...]:
        for cache_index, shape in self._items:
            if cache_index == key:
                return shape
        raise KeyError(key)

    def __iter__(self) -> Iterator[int]:
        return (cache_index for cache_index, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __deepcopy__(self, memo) -> "_FrozenCacheShapes":
        del memo
        return self


@dataclass(frozen=True)
class Gemma4LayerCacheBinding:
    layer_idx: int
    layer_name: str
    attention_kind: AttentionKind
    cache_group_idx: int | None
    cache_index: int
    kv_shape: tuple[int, ...]
    sliding_window: int | None
    cache_owner_layer_idx: int | None = None
    # vLLM may enlarge a smaller physical page's scheduler block so every
    # hybrid group has the same byte-sized virtual page.  HMONNX keeps its
    # exported 64-token physical pages, so finalized bindings must retain both
    # coordinate systems explicitly.
    scheduler_block_size: int | None = None
    physical_block_size: int = 64

    @property
    def physical_pages_per_scheduler_block(self) -> int:
        scheduler_block_size = self.scheduler_block_size
        if scheduler_block_size is None:
            raise RuntimeError(
                f"Gemma4 binding {self.layer_name} is not finalized against vLLM"
            )
        if (
            self.physical_block_size <= 0
            or scheduler_block_size < self.physical_block_size
            or scheduler_block_size % self.physical_block_size != 0
        ):
            raise RuntimeError(
                "Gemma4 scheduler/physical block sizes are incompatible: "
                f"layer={self.layer_name}, scheduler={scheduler_block_size}, "
                f"physical={self.physical_block_size}"
            )
        return scheduler_block_size // self.physical_block_size

    @property
    def fixed_physical_pages(self) -> int | None:
        """Return the exported fixed backing for a sliding-attention layer.

        Full-attention storage follows vLLM's global scheduler capacity.  A
        Gemma4 sliding layer instead receives the moving ``window + prefill``
        cache exported by HMONNX, whose sequence axis is already recorded in
        ``kv_shape``.
        """

        if self.attention_kind != "sliding_attention":
            return None
        if len(self.kv_shape) != 4:
            raise RuntimeError(
                f"Gemma4 sliding binding {self.layer_name} must have [B, H, S, D] shape"
            )
        sequence_length = int(self.kv_shape[2])
        if sequence_length <= 0 or sequence_length % self.physical_block_size != 0:
            raise RuntimeError(
                "Gemma4 sliding cache sequence length must be divisible by its "
                f"physical block size: layer={self.layer_name}, "
                f"sequence={sequence_length}, block={self.physical_block_size}"
            )
        return sequence_length // self.physical_block_size


@dataclass(frozen=True)
class Gemma4CacheContract:
    version: int
    prefill_chunk_size: int
    bindings: tuple[Gemma4LayerCacheBinding, ...]
    cache_shapes: Mapping[int, tuple[int, ...]]
    attention_lowering: str = "flash_attention"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cache_shapes", _FrozenCacheShapes(dict(self.cache_shapes))
        )

    def binding_for_layer(self, layer_name: str) -> Gemma4LayerCacheBinding:
        for binding in self.bindings:
            if binding.layer_name == layer_name:
                return binding
        raise KeyError(f"Unknown Gemma4 cache layer: {layer_name}")

    def bindings_for_group(
        self, cache_group_idx: int
    ) -> tuple[Gemma4LayerCacheBinding, ...]:
        return tuple(
            binding
            for binding in self.bindings
            if binding.cache_group_idx == cache_group_idx
        )

    def shape_for_cache(self, cache_index: int) -> tuple[int, ...]:
        try:
            return self.cache_shapes[cache_index]
        except KeyError as exc:
            raise KeyError(f"Unknown Gemma4 cache index: {cache_index}") from exc

    @property
    def context_bindings(self) -> tuple[Gemma4LayerCacheBinding, ...]:
        if self.attention_lowering == "flash_attention":
            return self.bindings
        if self.attention_lowering == "full_flash_attention":
            return tuple(
                binding
                for binding in self.bindings
                if binding.attention_kind == "full_attention"
                or binding.cache_owner_layer_idx == binding.layer_idx
            )
        raise RuntimeError(
            f"Unsupported Gemma4 attention_lowering={self.attention_lowering!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "prefill_chunk_size": self.prefill_chunk_size,
            "bindings": [asdict(binding) for binding in self.bindings],
            "cache_shapes": dict(self.cache_shapes),
            "attention_lowering": self.attention_lowering,
        }


def _require_list(meta: Mapping[str, Any], field: str) -> list[Any]:
    value = meta.get(field)
    if not isinstance(value, list):
        raise ValueError(f"Gemma4 {field} must be a list")
    return value


def _require_int(
    meta: Mapping[str, Any], field: str, *, minimum: int | None = None
) -> int:
    value = meta.get(field)
    if type(value) is not int:
        raise ValueError(f"Gemma4 {field} must be an integer")
    normalized = value
    if minimum is not None and normalized < minimum:
        raise ValueError(f"Gemma4 {field} must be at least {minimum}")
    return normalized


def _get_config_value(config: Any, field: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(field)
    return getattr(config, field, None)


def _get_hf_text_config(hf_config: Any) -> Any:
    return _get_config_value(hf_config, "text_config") or hf_config


def _validate_mtp_page_cache_contract(
    hmonnx_meta: Mapping[str, Any],
    *,
    sliding_window: int,
    prefill_chunk_size: int,
) -> None:
    """Fail closed unless MTP uses the read-only paged-cache ABI.

    Target verification writes speculative K/V into the normal target pages.
    Rejection is handled by vLLM's logical sequence length, so the assistant
    must only *read* those pages and the fixed sliding backing must retain at
    least ``K`` overwrite slots beyond the visible window.
    """

    spec = hmonnx_meta.get("spec_decode")
    if not isinstance(spec, Mapping):
        raise ValueError("Gemma4 contract-v2 MTP requires a spec_decode contract")
    if spec.get("runtime_contract_version") != 2:
        raise ValueError(
            "Gemma4 contract-v2 MTP requires spec_decode.runtime_contract_version=2"
        )
    if str(spec.get("mode", "")).strip().lower() != "mtp":
        raise ValueError("Gemma4 contract-v2 MTP requires spec_decode.mode='mtp'")

    num_draft_tokens = spec.get("num_draft_tokens", spec.get("block_size"))
    if type(num_draft_tokens) is not int or num_draft_tokens <= 0:
        raise ValueError(
            "Gemma4 contract-v2 MTP requires positive spec_decode.num_draft_tokens"
        )
    verify_length = spec.get("verify_length")
    if verify_length != num_draft_tokens + 1:
        raise ValueError("Gemma4 MTP verify_length must equal num_draft_tokens + 1")

    draft = spec.get("draft")
    if not isinstance(draft, Mapping):
        raise ValueError("Gemma4 contract-v2 MTP requires spec_decode.draft")
    draft_abi = draft.get("abi")
    if draft_abi not in {
        "gemma4_mtp_readonly_page_attention_v1",
        "gemma4_mtp_readonly_page_attention_v2",
        "gemma4_mtp_readonly_mixed_attention_v3",
    }:
        raise ValueError(
            "Gemma4 contract-v2 MTP requires the read-only PageAttention ABI"
        )
    if (
        draft_abi == "gemma4_mtp_readonly_page_attention_v2"
        and draft.get("attention_lowering") != "causal"
    ):
        raise ValueError(
            "Gemma4 read-only PageAttention v2 requires causal lowering"
        )
    if (
        draft_abi == "gemma4_mtp_readonly_mixed_attention_v3"
        and draft.get("attention_lowering") != "full_page_sliding_unfused"
    ):
        raise ValueError(
            "Gemma4 read-only mixed attention v3 requires full-page/sliding-unfused lowering"
        )
    if draft.get("cache_mutation") != "read_only":
        raise ValueError("Gemma4 MTP assistant cache_mutation must be read_only")
    if draft.get("cache_binding") != "target_attention_type_owner":
        raise ValueError(
            "Gemma4 MTP assistant must bind target attention-type cache owners"
        )

    exported_sliding_capacity = ((sliding_window + prefill_chunk_size + 15) // 16) * 16
    if exported_sliding_capacity < sliding_window + num_draft_tokens:
        raise ValueError(
            "Gemma4 sliding cache does not retain enough speculative overwrite slack"
        )


def _normalize_shape(value: Any, cache_index: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(
            f"Gemma4 layer_kv_shapes[{cache_index}] must be a shape with a sequence axis"
        )
    shape: list[int] = []
    for dimension in value:
        if type(dimension) is not int:
            raise ValueError(
                f"Gemma4 layer_kv_shapes[{cache_index}] must contain positive integers"
            )
        normalized = dimension
        if normalized <= 0:
            raise ValueError(
                f"Gemma4 layer_kv_shapes[{cache_index}] must contain positive integers"
            )
        shape.append(normalized)
    return tuple(shape)


def parse_gemma4_cache_contract(
    *, hf_config: Any, hmonnx_meta: Mapping[str, Any]
) -> Gemma4CacheContract:
    if not isinstance(hmonnx_meta, Mapping):
        raise ValueError("Gemma4 HMONNX metadata must be a mapping")

    version = _require_int(hmonnx_meta, "attention_contract_version")
    if version != 2:
        raise ValueError("Gemma4 cache parsing requires exact contract-v2 metadata")
    attention_lowering = str(
        hmonnx_meta.get("attention_lowering", "flash_attention")
    ).strip().lower()
    if attention_lowering not in {"flash_attention", "full_flash_attention"}:
        raise ValueError(
            f"Gemma4 contract-v2 has unsupported attention_lowering={attention_lowering!r}"
        )
    expected_sliding_flash = attention_lowering == "flash_attention"
    if (
        "uses_sliding_flash_attention_v2" in hmonnx_meta
        and hmonnx_meta.get("uses_sliding_flash_attention_v2") is not expected_sliding_flash
    ):
        raise ValueError(
            "Gemma4 uses_sliding_flash_attention_v2 does not match attention_lowering"
        )

    raw_layer_types = _require_list(hmonnx_meta, "layer_types")
    layer_types = tuple(str(value).strip().lower() for value in raw_layer_types)
    unknown_types = sorted(set(layer_types) - _SUPPORTED_ATTENTION_KINDS)
    if unknown_types:
        raise ValueError(f"Unsupported Gemma4 layer_types: {unknown_types}")
    if "sliding_attention" not in layer_types:
        raise ValueError("Gemma4 contract-v2 requires sliding_attention")

    if hmonnx_meta.get("sliding_kv_cache_input_mode") != "slice_window":
        raise ValueError(
            "Gemma4 contract-v2 requires sliding_kv_cache_input_mode='slice_window'"
        )

    model_config = hmonnx_meta.get("model_config", {})
    if not isinstance(model_config, Mapping):
        raise ValueError("Gemma4 model_config must be a mapping")

    if hmonnx_meta.get("bidirectional_vision_attention") is not True:
        raise ValueError(
            "Gemma4 visual variants require bidirectional_vision_attention=true"
        )

    _require_int(hmonnx_meta, "max_mm_ranges_per_chunk", minimum=1)
    sliding_window = _require_int(hmonnx_meta, "sliding_window", minimum=1)
    prefill_chunk_size = _require_int(hmonnx_meta, "prefill_chunk_length", minimum=280)
    if str(model_config.get("spec_decode_mode", "none")).strip().lower() == "mtp":
        _validate_mtp_page_cache_contract(
            hmonnx_meta,
            sliding_window=sliding_window,
            prefill_chunk_size=prefill_chunk_size,
        )

    text_config = _get_hf_text_config(hf_config)
    num_hidden_layers = _get_config_value(text_config, "num_hidden_layers")
    if isinstance(num_hidden_layers, bool) or num_hidden_layers is None:
        raise ValueError("Gemma4 HF num_hidden_layers must be an integer")
    try:
        num_hidden_layers = int(num_hidden_layers)
    except (TypeError, ValueError) as exc:
        raise ValueError("Gemma4 HF num_hidden_layers must be an integer") from exc
    if num_hidden_layers != len(layer_types):
        raise ValueError(
            "Gemma4 HF num_hidden_layers does not match metadata layer_types: "
            f"{num_hidden_layers} != {len(layer_types)}"
        )

    hf_layer_types = _get_config_value(text_config, "layer_types")
    if hf_layer_types is not None:
        if not isinstance(hf_layer_types, (list, tuple)):
            raise ValueError("Gemma4 HF layer_types must be a list")
        normalized_hf_types = tuple(
            str(value).strip().lower() for value in hf_layer_types
        )
        if normalized_hf_types != layer_types:
            raise ValueError("Gemma4 HF layer_types do not match export layer_types")

    raw_cache_types = _require_list(hmonnx_meta, "layer_cache_types")
    cache_types = tuple(str(value).strip().lower() for value in raw_cache_types)
    unknown_cache_types = sorted(set(cache_types) - _SUPPORTED_ATTENTION_KINDS)
    if unknown_cache_types:
        raise ValueError(f"Unsupported Gemma4 layer_cache_types: {unknown_cache_types}")

    raw_shapes = _require_list(hmonnx_meta, "layer_kv_shapes")
    if len(cache_types) != len(raw_shapes):
        raise ValueError(
            "Gemma4 layer_cache_types and layer_kv_shapes must have equal lengths"
        )
    if not cache_types:
        raise ValueError("Gemma4 layer_cache_types must not be empty")
    cache_shapes = {
        cache_index: _normalize_shape(shape, cache_index)
        for cache_index, shape in enumerate(raw_shapes)
    }
    expected_sliding_cache_tokens = (
        (sliding_window + prefill_chunk_size + 15) // 16
    ) * 16
    for cache_index, (cache_kind, shape) in enumerate(
        zip(cache_types, cache_shapes.values(), strict=True)
    ):
        if (
            cache_kind == "sliding_attention"
            and shape[2] != expected_sliding_cache_tokens
        ):
            raise ValueError(
                f"Gemma4 layer_kv_shapes[{cache_index}] sequence length must equal "
                f"align16(sliding_window + prefill_chunk_length)={expected_sliding_cache_tokens}, "
                f"got {shape[2]}"
            )

    raw_cache_indices = _require_list(hmonnx_meta, "layer_cache_indices")
    if len(raw_cache_indices) != len(layer_types):
        raise ValueError(
            "Gemma4 layer_cache_indices length must equal layer_types length"
        )
    cache_indices: list[int] = []
    for layer_idx, value in enumerate(raw_cache_indices):
        if type(value) is not int:
            raise ValueError(
                f"Gemma4 layer_cache_indices[{layer_idx}] must be an integer"
            )
        cache_index = value
        if not 0 <= cache_index < len(cache_types):
            raise ValueError(
                f"Gemma4 layer_cache_indices[{layer_idx}]={cache_index} is out of range"
            )
        cache_indices.append(cache_index)
    if set(cache_indices) != set(range(len(cache_types))):
        raise ValueError(
            "Gemma4 layer_cache_indices must reference every declared cache index"
        )

    raw_owner_indices = _require_list(hmonnx_meta, "layer_cache_owner_indices")
    if len(raw_owner_indices) != len(cache_types):
        raise ValueError(
            "Gemma4 layer_cache_owner_indices length must equal layer_cache_types length"
        )
    cache_owner_layer_indices: list[int] = []
    for cache_index, value in enumerate(raw_owner_indices):
        if type(value) is not int:
            raise ValueError(
                f"Gemma4 layer_cache_owner_indices[{cache_index}] must be an integer"
            )
        owner_layer_idx = value
        if not 0 <= owner_layer_idx < len(layer_types):
            raise ValueError(
                "Gemma4 cache owner layer is out of range: "
                f"cache_index={cache_index}, owner_layer_idx={owner_layer_idx}"
            )
        if cache_indices[owner_layer_idx] != cache_index:
            raise ValueError(
                "Gemma4 cache owner layer does not reference its owned cache: "
                f"cache_index={cache_index}, owner_layer_idx={owner_layer_idx}, "
                f"owner_cache_index={cache_indices[owner_layer_idx]}"
            )
        canonical_owner_layer_idx = cache_indices.index(cache_index)
        if owner_layer_idx != canonical_owner_layer_idx:
            raise ValueError(
                "Gemma4 cache owner must be the first layer that references the "
                f"cache: cache_index={cache_index}, "
                f"expected_owner={canonical_owner_layer_idx}, got={owner_layer_idx}"
            )
        if layer_types[owner_layer_idx] != cache_types[cache_index]:
            raise ValueError(
                "Gemma4 layer_cache_types entry does not match its cache owner: "
                f"cache_index={cache_index}, owner_layer_idx={owner_layer_idx}"
            )
        cache_owner_layer_indices.append(owner_layer_idx)

    bindings: list[Gemma4LayerCacheBinding] = []
    for layer_idx, (attention_kind, cache_index) in enumerate(
        zip(layer_types, cache_indices, strict=True)
    ):
        cache_kind = cache_types[cache_index]
        if cache_kind != attention_kind:
            raise ValueError(
                "Gemma4 layer_cache_types does not match layer_types for "
                f"layer {layer_idx}: {cache_kind!r} != {attention_kind!r}"
            )
        shape = cache_shapes[cache_index]
        bindings.append(
            Gemma4LayerCacheBinding(
                layer_idx=layer_idx,
                layer_name=f"placeholder_attn_layers.{layer_idx}",
                attention_kind=attention_kind,  # type: ignore[arg-type]
                cache_group_idx=None,
                cache_index=cache_index,
                kv_shape=shape,
                sliding_window=(
                    sliding_window if attention_kind == "sliding_attention" else None
                ),
                cache_owner_layer_idx=cache_owner_layer_indices[cache_index],
            )
        )

    return Gemma4CacheContract(
        version=version,
        prefill_chunk_size=prefill_chunk_size,
        bindings=tuple(bindings),
        cache_shapes=cache_shapes,
        attention_lowering=attention_lowering,
    )


__all__ = [
    "AttentionKind",
    "Gemma4CacheContract",
    "Gemma4LayerCacheBinding",
    "parse_gemma4_cache_contract",
]
