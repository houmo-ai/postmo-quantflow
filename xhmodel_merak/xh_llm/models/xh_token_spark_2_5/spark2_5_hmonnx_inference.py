from xhmodel_merak.xh_llm.hmonnx import TextLLMHMONNXModel
from xhmodel_merak.xh_llm.llm_data_processor import (
    BaseLLMInputProcessor,
    SlidingWindowInputProcessorConfig,
    SlidingWindowLLMInputProcessor,
)
from xhmodel_merak.xh_llm.types import LLMModelMeta

from .kv_cache import SparkKVCacheMixin, build_layer_kv_shapes


class XHSpark2_5HMONNXModel(TextLLMHMONNXModel):  # noqa: N801 - Spark 2_5 public name
    def __init__(self, meta_info: LLMModelMeta, **kwargs):
        if kwargs.get("enable_page_attention"):
            raise ValueError("Spark sliding-window processing does not support PageAttention")
        layer_types = getattr(meta_info, "layer_types", None)
        layer_kv_shapes = getattr(meta_info, "layer_kv_shapes", None)
        if meta_info.kv_cache.use_cache:
            if layer_types is None or layer_kv_shapes is None:
                raise ValueError("Spark requires per-layer KV cache metadata; re-export the model")
            expected = build_layer_kv_shapes(
                layer_types,
                meta_info.kv_cache.kv_cache_shape,
                meta_info.model_config.sliding_window,
                meta_info.model_config.prefill_chunk_length,
            )
            if layer_kv_shapes != expected or len(layer_types) != meta_info.kv_cache.num_layers:
                raise ValueError("Spark per-layer KV cache metadata is inconsistent; re-export the model")
        super().__init__(meta_info, **kwargs)
        self.sliding_window = meta_info.model_config.sliding_window  # type: ignore
        if self.use_cache:
            self._kvcache_mixin = SparkKVCacheMixin(
                self.kvcache_config,
                layer_types,
                layer_kv_shapes,
                cache_type=self._backend.resolve_class("CacheTensor"),
                sliding_cache_type=self._backend.resolve_class("HybridCacheTensor"),
            )

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        assert self.sliding_window > 0
        preprocessor = SlidingWindowLLMInputProcessor(
            SlidingWindowInputProcessorConfig(
                embed_tokens=self.get_input_embeddings(),
                input_sequence_length=self.get_input_sequence_length(),
                past_key_caches=self.past_key_caches,
                past_value_caches=self.past_value_caches,
                enable_page_attention=self.enable_page_attention,
                pad_token_id=self.pad_token_id,
                sliding_window=self.sliding_window,
            )
        )
        return preprocessor


__all__ = [
    "XHSpark2_5HMONNXModel",
]
