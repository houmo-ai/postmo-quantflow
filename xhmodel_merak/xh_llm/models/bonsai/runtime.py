"""Qwen's hybrid cache runtime specialized for a text-only Bonsai artifact."""

from ...hmonnx.base_llm_hmonnx_model import BaseLLMHMONNXModel
from ..qwen3_5.data_preprocess import Qwen3_5_DataPreprocess
from ..qwen3_5.qwen3_5_hmonnx_inference import Qwen3_5HMONNXKVCacheMixin, XHQwen3_5_HMONNXModel


class XHBonsaiHMONNXModel(XHQwen3_5_HMONNXModel):
    """Inherit Qwen's hybrid state handling without constructing a visual runtime."""

    def __init__(self, meta_info, **kwargs):
        BaseLLMHMONNXModel.__init__(self, meta_info, **kwargs)
        self._kvcache_mixin = Qwen3_5HMONNXKVCacheMixin(self.kvcache_config)
        self._kvcache_mixin.split_conv_cache = bool(meta_info.model_config.split_conv_cache)
        self._sync_page_attention_mode_to_kvcache()

    # The Qwen overrides also move/configure a visual tower. Bonsai uses the
    # ordinary text lifecycle and keeps the inherited hybrid-cache forward.
    _set_device = BaseLLMHMONNXModel._set_device
    _set_dtype = BaseLLMHMONNXModel._set_dtype
    _set_enable_golden = BaseLLMHMONNXModel._set_enable_golden
    get_tf_processor = BaseLLMHMONNXModel.get_tf_processor

    def _get_data_preprocessor(self):
        return Qwen3_5_DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.get_input_sequence_length(),
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            enable_page_attention=self.enable_page_attention,
        )
