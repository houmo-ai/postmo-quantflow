# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceV2
from xhquant.xhonnxruntime.parsers.llm_cache import LLMCache

from ...hmonnx.text_llm_hmonnx_model import TextLLMHMONNXModel
from .contract import DSA, Glm53CacheSpec, validate_state_abi, validate_state_graph
from .runtime import Glm53CacheMixin, Glm53DataPreprocess
from .vision import create_glm53_processor, load_vision_encoder


def _enable_inplace_dsa_caches(prefill_session, decode_session, spec):
    mode = os.environ.get("XHMODEL_GLM53_INPLACE_CACHE_UPDATES", "1")
    if mode not in ("0", "1"):
        raise ValueError("XHMODEL_GLM53_INPLACE_CACHE_UPDATES must be 0 or 1")
    if mode == "0":
        return False

    sessions = (("prefill", prefill_session), ("decode", decode_session))
    is_v2 = tuple(isinstance(session, HMONNXInferenceV2) for _, session in sessions)
    if not any(is_v2):
        return False  # Both graphs use a backend without the V2 cache writer.
    if not all(is_v2):
        raise RuntimeError("GLM-5.3 prefill and decode must use the same HMONNX backend")

    # Check both graphs before changing either one. The SDK's enable method
    # mutates LLMCache modules and cannot report a mismatch without doing so.
    # Each DSA layer owns exactly two persistent writers: latent and pool.
    expected = 2 * spec.layer_types.count(DSA)
    writers = []
    for name, session in sessions:
        session.initialize()
        eligible = tuple(
            module
            for module in session.graph_module.modules()
            if isinstance(module, LLMCache) and not module.only_handle_old_cache and module.attention_max_length < 0
        )
        if len(eligible) != expected:
            raise RuntimeError(
                f"GLM-5.3 {name} cache writer count mismatch: expected={expected}, found={len(eligible)}"
            )
        writers.extend((module, module.reuse_output_storage) for module in eligible)

    try:
        alias_stages = []
        for name, session in sessions:
            enabled = session.enable_inplace_cache_updates()
            if enabled != expected:
                raise RuntimeError(
                    f"GLM-5.3 {name} cache writer count mismatch: expected={expected}, enabled={enabled}"
                )
            configure_aliases = getattr(session, "enable_reused_cache_output_aliases", None)
            alias_stages.append(configure_aliases() if callable(configure_aliases) else 0)
            configure_static_rhs = getattr(session, "enable_constant_matmul_rhs_cache", None)
            if callable(configure_static_rhs):
                configure_static_rhs(min_numel=4_000_000)
    except Exception:
        for module, reuse_output_storage in writers:
            module.reuse_output_storage = reuse_output_storage
        raise
    return all(alias_stages)


class XHGlm53HMONNXModel(TextLLMHMONNXModel):
    def __init__(self, meta_info, **kwargs):
        if kwargs.get("enable_page_attention", False):
            raise ValueError("GLM-5.3 does not use PageAttention caches")
        import onnx

        validate_state_abi(getattr(meta_info, "glm53_state_abi_version", None))
        spec = Glm53CacheSpec(**meta_info.glm53_cache_spec)
        for stage in ("prefill", "decode"):
            graph = onnx.load(getattr(meta_info, stage + "_hmonnx"), load_external_data=False)
            validate_state_graph(graph.graph, spec, stage)
        super().__init__(meta_info, **kwargs)
        self.cache_spec = spec
        # GLM begins each request with zeroed caches and only appends valid
        # rows, so their backing storage is the full-capacity graph output.
        inplace_step_staging = _enable_inplace_dsa_caches(
            self.prefill_model.hmonnx_session,
            self.decode_model.hmonnx_session,
            self.cache_spec,
        )
        self._kvcache_mixin = Glm53CacheMixin(
            self.cache_spec, self.kvcache_config, inplace_step_staging=inplace_step_staging
        )
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
        try:
            return self._kvcache_mixin.commit(super().forward(*args))
        except Exception:
            self._kvcache_mixin.rollback_step()
            raise
