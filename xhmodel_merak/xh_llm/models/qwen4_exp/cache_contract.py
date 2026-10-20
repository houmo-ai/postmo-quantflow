"""Qwen4Exp HMONNX cache contract metadata.

This module intentionally describes cache ABI only.  It does not implement a
Qwen4Exp model wrapper or an exporter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


QWEN4_EXP_LLM_CACHE_ABI = "llm_cache_inplace_v1"
QWEN4_EXP_PAGE_CACHE_ABI = "page_kv_v1"


def resolve_qwen4_exp_cache_abi(value=None) -> str:
    """Missing metadata identifies legacy PageKV packages."""
    value = QWEN4_EXP_PAGE_CACHE_ABI if value is None else value
    if value not in (QWEN4_EXP_LLM_CACHE_ABI, QWEN4_EXP_PAGE_CACHE_ABI):
        raise ValueError(f"Unsupported Qwen4 cache ABI: {value!r}")
    return value


LINEAR_ATTENTION = "linear_attention"
QWEN_SPARSE_ATTENTION = "qwen_sparse_attention"
FULL_ATTENTION = "full_attention"

DELTA_CONV_STATE_IDX = 0
PLE_CONV_STATE_IDX = 1
PLE_NGRAM_STATE_IDX = 2


def qwen4_exp_cache_input_specs(contract):
    """Map exported ordinal K/V names to the logical layer state specs."""
    ordinals = {layer.layer_idx: ordinal for ordinal, layer in enumerate(
        layer for layer in contract.layers if any(spec.state_kind == "kv_key" for spec in layer.tensors)
    )}
    return {
        (f"past_key_cache_{ordinals[spec.layer_idx]}" if spec.state_kind == "kv_key" else
         f"past_value_cache_{ordinals[spec.layer_idx]}" if spec.state_kind == "kv_value" else spec.name): spec
        for spec in contract.tensors
    }


def _config_get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_int(config: Any, name: str, default: Any | None = None) -> int:
    value = _config_get(config, name, default)
    if value is None:
        raise ValueError(f"Qwen4Exp text_config is missing required field {name!r}")
    return int(value)


def _as_int_list(config: Any, name: str, default: tuple[int, ...] = ()) -> tuple[int, ...]:
    value = _config_get(config, name, default)
    if value is None:
        return default
    return tuple(int(item) for item in value)


def _shape_of(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"cache tensor has no shape attribute: {type(value).__name__}")
    return tuple(int(dim) for dim in shape)


def resolve_qwen4_exp_cache_shape(spec, *, batch_size: int, context_length: int) -> tuple[int, ...]:
    sizes = {"batch": batch_size, "cache_length": context_length, "qsa_pool_length": context_length // 4}
    return tuple(int(sizes.get(dim, dim)) for dim in spec.shape)


@dataclass(frozen=True)
class Qwen4ExpTensorSpec:
    """One named tensor in the Qwen4Exp cache ABI."""

    name: str
    layer_idx: int
    state_kind: str
    shape: tuple[int | str, ...]
    dtype: str
    state_idx: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer_idx": self.layer_idx,
            "state_kind": self.state_kind,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "state_idx": self.state_idx,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Qwen4ExpTensorSpec":
        return cls(
            name=str(data["name"]),
            layer_idx=int(data["layer_idx"]),
            state_kind=str(data["state_kind"]),
            shape=tuple(data["shape"]),
            dtype=str(data["dtype"]),
            state_idx=None if data.get("state_idx") is None else int(data["state_idx"]),
        )


@dataclass(frozen=True)
class Qwen4ExpLayerCacheContract:
    layer_idx: int
    layer_type: str
    tensors: tuple[Qwen4ExpTensorSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "layer_type": self.layer_type,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Qwen4ExpLayerCacheContract":
        return cls(
            layer_idx=int(data["layer_idx"]),
            layer_type=str(data["layer_type"]),
            tensors=tuple(Qwen4ExpTensorSpec.from_dict(item) for item in data["tensors"]),
        )


@dataclass(frozen=True)
class Qwen4ExpCacheContract:
    """Shape/key contract for Qwen4Exp text cache export.

    Qwen4Exp uses Transformers ``DynamicCache`` with heterogeneous layer
    classes.  Linear layers own DeltaNet state ``0``.  PLE layers add dilated
    convolution state ``1`` and ngram token-history state ``2``.  Sparse/full
    attention layers own ordinary K/V, pooled indexer keys, a four-token
    indexer tail, and three recent rows for each K/V tail. RoPE positions are
    prepared on the host and are not part of the model cache ABI.
    """

    num_hidden_layers: int
    layer_types: tuple[str, ...]
    ple_layer_indices: tuple[int, ...]
    num_key_value_heads: int
    head_dim: int
    indexer_head_dim: int
    linear_conv_dim: int
    linear_conv_kernel_dim: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    ple_hidden_size: int
    ple_short_conv_state_len: int
    ngram_context_len: int
    layers: tuple[Qwen4ExpLayerCacheContract, ...]
    model_tensors: tuple[Qwen4ExpTensorSpec, ...]

    @classmethod
    def from_text_config(cls, text_config: Any) -> "Qwen4ExpCacheContract":
        num_hidden_layers = _as_int(text_config, "num_hidden_layers")
        layer_types_value = _config_get(text_config, "layer_types")
        if layer_types_value is None:
            raise ValueError("Qwen4Exp text_config is missing required field 'layer_types'")
        layer_types = tuple(
            QWEN_SPARSE_ATTENTION if str(item) == FULL_ATTENTION else str(item)
            for item in layer_types_value
        )
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"Qwen4Exp layer_types length must match num_hidden_layers: {len(layer_types)} != {num_hidden_layers}"
            )

        hidden_size = _as_int(text_config, "hidden_size")
        hc_count = _as_int(text_config, "hc_count")
        num_key_value_heads = _as_int(text_config, "num_key_value_heads")
        head_dim = _as_int(text_config, "head_dim", hidden_size // _as_int(text_config, "num_attention_heads"))
        indexer_head_dim = _as_int(text_config, "indexer_head_dim")
        linear_key_heads = _as_int(text_config, "linear_num_key_heads")
        linear_value_heads = _as_int(text_config, "linear_num_value_heads")
        linear_key_head_dim = _as_int(text_config, "linear_key_head_dim")
        linear_value_head_dim = _as_int(text_config, "linear_value_head_dim")
        linear_conv_kernel_dim = _as_int(text_config, "linear_conv_kernel_dim")
        ple_conv_kernel_size = _as_int(text_config, "ple_conv_kernel_size")
        ngram_size = _as_int(text_config, "ngram_size")
        ple_layer_indices = tuple(layer_id - 1 for layer_id in _as_int_list(text_config, "ple_layer_ids"))
        invalid_ple = [idx for idx in ple_layer_indices if idx < 0 or idx >= num_hidden_layers]
        if invalid_ple:
            raise ValueError(f"Qwen4Exp ple_layer_ids resolve outside layer range: {invalid_ple}")

        linear_conv_dim = linear_key_heads * linear_key_head_dim * 2 + linear_value_heads * linear_value_head_dim
        recurrent_shape: tuple[int | str, ...] = (
            "batch",
            linear_value_heads,
            linear_key_head_dim,
            linear_value_head_dim,
        )
        layers: list[Qwen4ExpLayerCacheContract] = []

        for layer_idx, layer_type in enumerate(layer_types):
            tensors: list[Qwen4ExpTensorSpec] = []
            if layer_type == LINEAR_ATTENTION:
                tensors.extend(
                    [
                        Qwen4ExpTensorSpec(
                            name=f"delta_conv_state_{layer_idx}",
                            layer_idx=layer_idx,
                            state_kind="delta_conv",
                            state_idx=DELTA_CONV_STATE_IDX,
                            shape=("batch", linear_conv_dim, linear_conv_kernel_dim),
                            dtype="floating",
                        ),
                        Qwen4ExpTensorSpec(
                            name=f"delta_recurrent_state_{layer_idx}",
                            layer_idx=layer_idx,
                            state_kind="delta_recurrent",
                            state_idx=DELTA_CONV_STATE_IDX,
                            shape=recurrent_shape,
                            dtype="floating",
                        ),
                    ]
                )
            elif layer_type in {QWEN_SPARSE_ATTENTION, FULL_ATTENTION}:
                tensors.extend(
                    [
                        Qwen4ExpTensorSpec(
                            name=f"key_cache_{layer_idx}",
                            layer_idx=layer_idx,
                            state_kind="kv_key",
                            shape=("batch", num_key_value_heads, "cache_length", head_dim),
                            dtype="floating",
                        ),
                        Qwen4ExpTensorSpec(
                            name=f"value_cache_{layer_idx}",
                            layer_idx=layer_idx,
                            state_kind="kv_value",
                            shape=("batch", num_key_value_heads, "cache_length", head_dim),
                            dtype="floating",
                        ),
                        Qwen4ExpTensorSpec(
                            name=f"qsa_indexer_key_cache_{layer_idx}",
                            layer_idx=layer_idx,
                            state_kind="qsa_indexer_key",
                            shape=("batch", 1, "qsa_pool_length", indexer_head_dim),
                            dtype="floating",
                        ),
                        Qwen4ExpTensorSpec(
                            name=f"qsa_indexer_key_tail_{layer_idx}",
                            layer_idx=layer_idx,
                            state_kind="qsa_indexer_tail",
                            shape=("batch", 4, indexer_head_dim),
                            dtype="floating",
                        ),
                        Qwen4ExpTensorSpec(
                            name=f"qsa_key_tail_{layer_idx}",
                            layer_idx=layer_idx,
                            state_kind="qsa_key_tail",
                            shape=("batch", num_key_value_heads, 3, head_dim),
                            dtype="floating",
                        ),
                        Qwen4ExpTensorSpec(
                            name=f"qsa_value_tail_{layer_idx}",
                            layer_idx=layer_idx,
                            state_kind="qsa_value_tail",
                            shape=("batch", num_key_value_heads, 3, head_dim),
                            dtype="floating",
                        ),
                    ]
                )
            else:
                raise ValueError(f"unsupported Qwen4Exp layer_type at layer {layer_idx}: {layer_type!r}")

            if layer_idx in ple_layer_indices:
                tensors.extend(
                    [
                        Qwen4ExpTensorSpec(
                            name=f"ple_conv_state_{layer_idx}",
                            layer_idx=layer_idx,
                            state_kind="ple_conv",
                            state_idx=PLE_CONV_STATE_IDX,
                            shape=("batch", hidden_size * hc_count, (ple_conv_kernel_size - 1) * ngram_size),
                            dtype="floating",
                        ),
                        Qwen4ExpTensorSpec(
                            name=f"ple_ngram_history_{layer_idx}",
                            layer_idx=layer_idx,
                            state_kind="ple_ngram_history",
                            state_idx=PLE_NGRAM_STATE_IDX,
                            shape=("batch", ngram_size - 1),
                            dtype="integer",
                        ),
                    ]
                )
            layers.append(
                Qwen4ExpLayerCacheContract(layer_idx=layer_idx, layer_type=layer_type, tensors=tuple(tensors))
            )

        model_tensors = ()
        return cls(
            num_hidden_layers=num_hidden_layers,
            layer_types=layer_types,
            ple_layer_indices=ple_layer_indices,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            indexer_head_dim=indexer_head_dim,
            linear_conv_dim=linear_conv_dim,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_num_value_heads=linear_value_heads,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            ple_hidden_size=hidden_size * hc_count,
            ple_short_conv_state_len=(ple_conv_kernel_size - 1) * ngram_size,
            ngram_context_len=ngram_size - 1,
            layers=tuple(layers),
            model_tensors=model_tensors,
        )

    @property
    def tensors(self) -> tuple[Qwen4ExpTensorSpec, ...]:
        return self.model_tensors + tuple(tensor for layer in self.layers for tensor in layer.tensors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "qwen4_exp_text",
            "num_hidden_layers": self.num_hidden_layers,
            "layer_types": list(self.layer_types),
            "ple_layer_indices": list(self.ple_layer_indices),
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "indexer_head_dim": self.indexer_head_dim,
            "linear_conv_dim": self.linear_conv_dim,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "linear_num_value_heads": self.linear_num_value_heads,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "ple_hidden_size": self.ple_hidden_size,
            "ple_short_conv_state_len": self.ple_short_conv_state_len,
            "ngram_context_len": self.ngram_context_len,
            "model_tensors": [tensor.to_dict() for tensor in self.model_tensors],
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Qwen4ExpCacheContract":
        return cls(
            num_hidden_layers=int(data["num_hidden_layers"]),
            layer_types=tuple(str(item) for item in data["layer_types"]),
            ple_layer_indices=tuple(int(item) for item in data["ple_layer_indices"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            head_dim=int(data["head_dim"]),
            indexer_head_dim=int(data["indexer_head_dim"]),
            linear_conv_dim=int(data["linear_conv_dim"]),
            linear_conv_kernel_dim=int(data["linear_conv_kernel_dim"]),
            linear_num_value_heads=int(data["linear_num_value_heads"]),
            linear_key_head_dim=int(data["linear_key_head_dim"]),
            linear_value_head_dim=int(data["linear_value_head_dim"]),
            ple_hidden_size=int(data["ple_hidden_size"]),
            ple_short_conv_state_len=int(data["ple_short_conv_state_len"]),
            ngram_context_len=int(data["ngram_context_len"]),
            layers=tuple(Qwen4ExpLayerCacheContract.from_dict(item) for item in data["layers"]),
            model_tensors=tuple(Qwen4ExpTensorSpec.from_dict(item) for item in data["model_tensors"]),
        )

    def expected_keys(self) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in self.tensors)

    def validate_cache_shapes(self, cache_tensors: Mapping[str, Any]) -> None:
        missing = [name for name in self.expected_keys() if name not in cache_tensors]
        if missing:
            raise ValueError(f"missing Qwen4Exp cache tensor(s): {missing}")
        unexpected = sorted(set(cache_tensors).difference(self.expected_keys()))
        if unexpected:
            raise ValueError(f"unexpected Qwen4Exp cache tensor(s): {unexpected}")

        dims: dict[str, int] = {}
        for spec in self.tensors:
            actual = _shape_of(cache_tensors[spec.name])
            if len(actual) != len(spec.shape):
                raise ValueError(
                    f"{spec.name} rank mismatch: expected {len(spec.shape)} dims {spec.shape}, got {actual}"
                )
            for axis, (expected, actual_dim) in enumerate(zip(spec.shape, actual, strict=True)):
                if isinstance(expected, int):
                    if actual_dim != expected:
                        raise ValueError(
                            f"{spec.name} shape mismatch at dim {axis}: expected {expected}, got {actual_dim}"
                        )
                    continue
                if expected in dims and dims[expected] != actual_dim:
                    raise ValueError(
                        f"{spec.name} shape mismatch for symbolic dim {expected!r}: "
                        f"expected {dims[expected]}, got {actual_dim}"
                    )
                dims.setdefault(expected, actual_dim)
        if "qsa_pool_length" in dims and "cache_length" in dims:
            if dims["qsa_pool_length"] != dims["cache_length"] // 4:
                raise ValueError("QSA pool cache length must be one quarter of context length")
