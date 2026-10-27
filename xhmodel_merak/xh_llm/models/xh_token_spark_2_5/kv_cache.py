"""Spark's mixed full/sliding cache layout, shared by export and runtime."""

import torch

from xhquant.core import CacheTensor, HybridCacheTensor

from ...kv_cache_mixin import KVCacheMixin


def build_layer_kv_shapes(layer_types, full_shape, sliding_window, prefill_chunk_length):
    if sliding_window <= 0 or prefill_chunk_length <= 0:
        raise ValueError("Spark sliding_window and prefill_chunk_length must be positive")
    # Use the slice-window storage contract: W + the largest input chunk.
    # LLMCache's attention output is separately aligned to W + current Q - 1.
    sliding_capacity = ((sliding_window + prefill_chunk_length + 15) // 16) * 16
    shapes = []
    for layer_type in layer_types:
        if layer_type not in {"sliding_attention", "full_attention"}:
            raise ValueError(f"Unsupported Spark layer type: {layer_type!r}")
        shape = list(full_shape)
        if layer_type == "sliding_attention":
            shape[2] = sliding_capacity
        shapes.append(shape)
    return shapes


class SparkKVCacheMixin(KVCacheMixin):
    def __init__(
        self, config, layer_types, layer_kv_shapes, *, cache_type=CacheTensor, sliding_cache_type=HybridCacheTensor
    ):
        super().__init__(config)
        if config.cache_axis != 2:
            raise ValueError("Spark KV cache requires cache_axis=2")
        if len(layer_types) != config.num_layers or len(layer_kv_shapes) != config.num_layers:
            raise ValueError("Spark layer types and KV shapes must match num_layers")
        self.layer_types = list(layer_types)
        self.layer_kv_shapes = [list(shape) for shape in layer_kv_shapes]
        self.CACHCE_TENSOR_TYPE = cache_type
        self.sliding_cache_type = sliding_cache_type

    def prepare_kv_cache(self):
        if self._cache_initialized or not self.use_cache:
            return
        for layer_type, shape in zip(self.layer_types, self.layer_kv_shapes, strict=True):
            cache_type = self.sliding_cache_type if layer_type == "sliding_attention" else self.CACHCE_TENSOR_TYPE
            self.past_key_caches.append(cache_type(torch.zeros(shape, dtype=self.kvcache_config.cache_torch_dtype)))
            self.past_value_caches.append(cache_type(torch.zeros(shape, dtype=self.kvcache_config.cache_torch_dtype)))
        self._cache_initialized = True
