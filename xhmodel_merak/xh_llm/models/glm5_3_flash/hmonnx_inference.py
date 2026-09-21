# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from ...hmonnx.text_llm_hmonnx_model import TextLLMHMONNXModel
from .contract import Glm53CacheSpec
from .runtime import Glm53CacheMixin, Glm53DataPreprocess
from .vision import create_glm53_processor, load_vision_encoder


class XHGlm53HMONNXModel(TextLLMHMONNXModel):
    def __init__(self, meta_info, **kwargs):
        if kwargs.get("enable_page_attention", False):
            raise ValueError("GLM-5.3 does not use PageAttention caches")
        if getattr(meta_info, "glm53_state_abi_version", None) != 1:
            raise ValueError("unsupported GLM-5.3 state ABI")
        super().__init__(meta_info, **kwargs)
        self.cache_spec = Glm53CacheSpec(**meta_info.glm53_cache_spec)
        self._kvcache_mixin = Glm53CacheMixin(self.cache_spec, self.kvcache_config)
        self._kvcache_mixin.to(device=self.device, dtype=self.dtype)
        self._data_processor = None
        self.vision_encoder = None
        vision = getattr(meta_info, "glm53_vision", None)
        if vision is not None:
            root = Path(getattr(meta_info, "_meta_path_", Path(meta_info.hf_config).parent / "meta.json")).parent
            self.vision_encoder = load_vision_encoder(vision, root, device=self.device)

    def _get_data_preprocessor(self):
        return Glm53DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.get_input_sequence_length(),
            cache_mixin=self._kvcache_mixin,
            pad_token_id=self.pad_token_id,
        )

    def get_processor(self):
        capacity = self.vision_encoder.capacity if self.vision_encoder is not None else 1024
        return create_glm53_processor(self.hf_model_dir, capacity)

    def forward(self, *args):
        return self._kvcache_mixin.commit(super().forward(*args))
