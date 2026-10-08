# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 Flash loader for the heterogeneous HMONNX cache ABI."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Optional, cast

import torch
from torch import nn
from transformers import AutoProcessor
from vllm.config import ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.glm4_1v import Glm4vDummyInputsBuilder, Glm4vMultiModalProcessor, Glm4vProcessingInfo
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, _require_is_multimodal
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageSize
from vllm.v1.sample.sampler import Sampler

from xhmodel_merak.xh_llm import AutoLLMHONNXModel
from xhmodel_merak.xh_llm.models.glm5_3_flash import XHGlm53HMONNXModel

from ...attention.backends.merak import MerakAttentionBackend
from ...config import MerakConfig
from ...constants import HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST
from ...utils import (
    apply_hmonnx_runtime_options,
    get_hmonnx_runtime_kwargs,
    move_hmonnx_auxiliary_tensors,
    resolve_hmonnx_primary_device,
    resolve_torch_dtype,
    should_preserve_hmonnx_runtime_device_map,
)
from ..register import MERAK_MODEL_REGISTRY
from ._contiguous_hmonnx import ContiguousHMONNXForCausalLM
from .hf_assets import get_model_trust_remote_code, load_hf_config, load_hf_pretrained
from .merak_for_causal_lm import MerakAttention


logger = init_logger(__name__)


class Glm53MerakProcessingInfo(Glm4vProcessingInfo):
    @cached_property
    def vision_metadata(self):
        meta = json.loads((Path(self.model_id) / "golden_meta_info.json").read_text())
        if not meta.get("glm53_vision"):
            raise ValueError("GLM image input requires an exported vision component.")
        return meta["glm53_vision"]

    @cached_property
    def source_config(self):
        return load_hf_config(self.model_id, trust_remote_code=get_model_trust_remote_code(self.ctx.model_config))

    def get_hf_config(self):
        return self.source_config

    @cached_property
    def processor(self):
        processor = load_hf_pretrained(
            AutoProcessor, self.model_id, trust_remote_code=get_model_trust_remote_code(self.ctx.model_config)
        )
        processor.image_processor.max_image_tokens = self.get_max_image_tokens()
        return processor

    def get_hf_processor(self, **kwargs):
        requested = kwargs.get("max_image_tokens", self.get_max_image_tokens())
        if int(requested) > self.get_max_image_tokens():
            raise ValueError("max_image_tokens exceeds the HMONNX vision capacity.")
        return self.processor

    def get_supported_mm_limits(self):
        return {"image": None}

    def get_num_frames_with_most_features(self, seq_len, mm_counts):
        # The inherited dummy builder requests a frame count even with zero videos.
        return 1

    def get_max_image_tokens(self):
        vision = self.vision_metadata
        return int(vision["max_patches_per_frame"]) // int(vision["vision_config"]["spatial_merge_size"]) ** 2

    def _get_image_max_pixels(self):
        config = self.get_hf_config().vision_config
        return self.get_max_image_tokens() * config.temporal_patch_size * (config.patch_size * config.spatial_merge_size) ** 2

    def _get_vision_info(self, *, image_width, image_height, num_frames=16, do_resize=True, max_image_pixels=None):
        from transformers.models.glm5_next.image_processing_glm5_next import smart_resize

        config = self.get_hf_config().vision_config
        factor = config.patch_size * config.spatial_merge_size
        height, width = image_height, image_width
        if do_resize:
            tokens = self.get_max_image_tokens()
            if max_image_pixels is not None:
                tokens = min(tokens, max_image_pixels // (config.temporal_patch_size * factor**2))
            height, width = smart_resize(
                num_frames, height, width, temporal_factor=config.temporal_patch_size,
                factor=factor, min_pixels=1, max_pixels=tokens,
            )
        grid_t = max(1, (num_frames + config.temporal_patch_size - 1) // config.temporal_patch_size)
        return ImageSize(width=width, height=height), grid_t * height * width // factor**2


class Glm53MerakMultiModalProcessor(Glm4vMultiModalProcessor):
    def _hf_processor_applies_updates(self, prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs):
        # Transformers' GLM processor already expands every image placeholder.
        return True


@MULTIMODAL_REGISTRY.register_processor(
    Glm53MerakMultiModalProcessor, info=Glm53MerakProcessingInfo, dummy_inputs=Glm4vDummyInputsBuilder
)
@MERAK_MODEL_REGISTRY.register_module("Glm5NextForConditionalGeneration")
class Glm53MerakForCausalLM(ContiguousHMONNXForCausalLM):
    """Run GLM-5.3 through its request-local KDA/DSA cache.

    GLM-5.3 Flash does not expose ordinary K/V tensors per attention
    layer.  Its HMONNX runtime owns linear-attention and compressed-sparse
    states together, so vLLM must schedule one request at a time
    while the runtime retains the cache between calls.
    """

    hybrid_cache_policy = HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST
    uses_internal_hmonnx_cache = True
    supports_multimodal = True
    supports_multimodal_raw_input_only = True
    requires_raw_input_tokens = True
    supports_mrope = False
    _prefer_input_ids_runtime = True

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        kv_cache_dtype: str = "auto",
        *,
        vllm_config: Optional[VllmConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if vllm_config is not None:
            model_config = vllm_config.model_config
        if model_config is None:
            raise ValueError("Glm53MerakForCausalLM requires either model_config or vllm_config.")

        self.vllm_config = vllm_config
        self.prefix = prefix
        config = model_config.hf_config
        if not isinstance(config, MerakConfig):
            raise TypeError(
                "Glm53MerakForCausalLM expects a MerakConfig as hf_config, "
                "got "
                f"{type(config).__name__}. Did you forget to set "
                "config_format='merak_llm' on the LLM?"
            )
        self.config: MerakConfig = config
        if bool(getattr(config, "enable_page_attention", False)):
            raise ValueError("GLM-5.3 Flash owns KDA/DSA state in HMONNX and does not support PageAttention.")

        max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1))
        if max_num_seqs != 1:
            raise ValueError(
                "GLM-5.3 Flash Merak runtime requires max_num_seqs=1 because "
                "its heterogeneous HMONNX cache is request-local."
            )
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "GLM-5.3 Flash Merak runtime does not support speculative "
                "decoding because KDA/DSA state has no rollback contract."
            )
        normalized_cache_dtype = str(kv_cache_dtype).lower()
        if normalized_cache_dtype not in {"auto", "float16", "fp16", "torch.float16"}:
            raise ValueError(
                f"GLM-5.3 Flash HMONNX cache is exported as float16; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        model_path = getattr(model_config, "model", None)
        name_or_path = getattr(config, "_name_or_path", None)
        config_dir_candidates = [Path(path) for path in (model_path, name_or_path) if path]
        config_dir = next(
            (candidate for candidate in config_dir_candidates if candidate.is_dir()),
            Path(name_or_path or model_path or "."),
        )
        hf_config_dir = config_dir / "hf_config"
        hf_config_file = hf_config_dir / "config.json"
        hf_config = json.loads(hf_config_file.read_text(encoding="utf-8"))
        architectures = hf_config.get("architectures") or []
        if not architectures or str(architectures[0]) != "Glm5NextForConditionalGeneration":
            raise ValueError(
                "GLM-5.3 Merak loader requires hf_config/config.json with "
                f"architectures=['Glm5NextForConditionalGeneration'], got {architectures!r}."
            )

        self.device = torch.device("cpu")
        self.inference_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        xh_model = config.xh_model
        if not isinstance(xh_model, dict) or str(xh_model.get("model_type", "")).lower() != "hmonnx":
            raise ValueError("GLM-5.3 Merak loader requires xh_model.model_type='hmonnx'.")

        hmonnx_meta_file = str(config_dir / "golden_meta_info.json")
        runtime_kwargs = get_hmonnx_runtime_kwargs(config)
        if should_preserve_hmonnx_runtime_device_map(config):
            runtime_kwargs["device_map"] = list(range(torch.cuda.device_count()))
        hmonnx_model = AutoLLMHONNXModel.from_pretrained(
            hmonnx_meta_file,
            **runtime_kwargs,
        )
        if not isinstance(hmonnx_model, XHGlm53HMONNXModel):
            raise TypeError(f"GLM-5.3 Merak loader expected XHGlm53HMONNXModel, got {type(hmonnx_model).__name__}.")
        self.prefill_chunk_length = int(hmonnx_model.meta_info.model_config.prefill_chunk_length)
        if self.prefill_chunk_length <= 0:
            raise ValueError("GLM-5.3 HMONNX prefill chunk length must be positive.")
        exported_context_length = int(hmonnx_model.cache_spec.max_context_length)
        configured_context_length = int(model_config.max_model_len)
        if configured_context_length > exported_context_length:
            raise ValueError(
                "vLLM max_model_len exceeds the GLM-5.3 HMONNX export: "
                f"configured={configured_context_length}, exported={exported_context_length}."
            )
        apply_hmonnx_runtime_options(hmonnx_model, config)
        if not should_preserve_hmonnx_runtime_device_map(config):
            hmonnx_model.to(self.inference_device)
        self.inference_device = resolve_hmonnx_primary_device(hmonnx_model, self.inference_device)
        self.inference_model = hmonnx_model
        move_hmonnx_auxiliary_tensors(hmonnx_model, self.inference_device)
        vision_encoder = getattr(hmonnx_model, "vision_encoder", None)
        if vision_encoder is not None:
            vision_encoder.to(self.inference_device)

        source_config = load_hf_config(
            hf_config_dir,
            trust_remote_code=get_model_trust_remote_code(model_config),
        )
        text_config = source_config.get_text_config()
        self.logits_processor = LogitsProcessor(int(text_config.vocab_size), logits_as_input=True)
        self.input_embeddings = cast(nn.Embedding, hmonnx_model.get_input_embeddings())
        self.sampler = Sampler()
        self.enable_page_attention = False
        self.cache_dtype = "float16"
        self.cache_torch_dtype = resolve_torch_dtype(self.cache_dtype)
        self.num_layers = 0
        self.key_cache_config: list[list[int]] = []
        self.value_cache_config: list[list[int]] = []
        self._cache_owner_request_id: str | None = None
        self._cache_owner_next_past_len = 0

        # vLLM still needs one tiny scheduler-visible KV group to produce
        # request metadata and block tables.  The GLM-5.3 runtime does not
        # read this placeholder; its real heterogeneous caches live in
        # ``inference_model.get_kvcache_mixin()``.
        self.placeholder_attn_layers = nn.ModuleList(
            [
                MerakAttention(
                    num_heads=1,
                    num_kv_heads=1,
                    head_size=1,
                    scale=1.0,
                    cache_config=self.vllm_config.cache_config,
                    prefix="model.layers.0.attn",
                    attn_backend=MerakAttentionBackend,
                )
            ]
        )
        logger.info(
            "Loaded GLM-5.3 Flash with request-local heterogeneous HMONNX cache; "
            "vLLM scheduler KV placeholder is intentionally not used by the runtime."
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        if modality != "image":
            raise ValueError(f"Unsupported GLM Merak modality: {modality}")
        return "<|begin_of_image|><|image|><|end_of_image|>"

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        pixels, grid = kwargs.get("pixel_values"), kwargs.get("image_grid_thw")
        if pixels is None and grid is None:
            return []
        if not isinstance(pixels, torch.Tensor) or not isinstance(grid, torch.Tensor):
            raise TypeError("GLM images require pixel_values and image_grid_thw tensors.")
        encoder = getattr(self.inference_model, "vision_encoder", None)
        if encoder is None:
            raise ValueError("GLM image input requires an exported vision component.")
        features = encoder(pixels, grid)
        counts = (grid.prod(dim=-1) // encoder.config.spatial_merge_size**2).tolist()
        return tuple(features.split(counts, dim=0))

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        embeds = super().embed_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return embeds
        return _merge_multimodal_embeddings(
            inputs_embeds=embeds,
            multimodal_embeddings=tuple(x.to(device=embeds.device, dtype=embeds.dtype) for x in multimodal_embeddings),
            is_multimodal=_require_is_multimodal(is_multimodal),
        )


__all__ = ["Glm53MerakForCausalLM"]
