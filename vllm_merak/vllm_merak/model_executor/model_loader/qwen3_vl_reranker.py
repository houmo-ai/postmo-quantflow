# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL relevance scoring with the existing Prefill-only HMONNX runtime."""

from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.activations import PoolerClassify
from vllm.model_executor.layers.pooler.seqwise import ClassifierPoolerHead, SequencePooler
from vllm.model_executor.layers.pooler.seqwise.methods import LastPool
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLMultiModalProcessor
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import PromptUpdateDetails

from xhmodel_merak.model_integrations.qwen3_vl_reranker.artifact import validate_score_contract
from xhmodel_merak.xh_llm import AutoLLMHONNXModel
from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import prefill_chunks
from xhmodel_merak.xh_llm.models.wemm_embedding.processor import video_frame_grids

from ...attention.backends.merak import MerakAttentionBackend
from ...config import MerakConfig
from ...constants import HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST
from ...utils import apply_hmonnx_runtime_options, get_hmonnx_runtime_kwargs
from ..register import MERAK_MODEL_REGISTRY
from ._contiguous_hmonnx import ContiguousHMONNXForCausalLM
from .hf_assets import get_model_trust_remote_code, load_hf_pretrained
from .merak_for_causal_lm import MerakAttention
from .qwen3_5 import (
    MerakQwen3_5ProcessingInfo,
    Qwen3_5MerakForCausalLM,
    Qwen3VLDummyInputsBuilder,
    _load_merak_visual_config,
    _max_visual_patch_capacity,
    _uses_visual_token_gears,
)


def build_reranker_pooler() -> DispatchPooler:
    return DispatchPooler(
        {
            "classify": SequencePooler(
                pooling=LastPool(),
                head=ClassifierPoolerHead(head_dtype=torch.float32, activation=PoolerClassify(num_labels=1)),
            )
        }
    )


class RerankerProcessingInfo(MerakQwen3_5ProcessingInfo):
    def get_hf_processor(self, **kwargs):
        # Preserve source assets. Qwen3.5's loader changes image resize limits
        # to fit export gears, whereas the official reranker resizes with
        # qwen_vl_utils first and disables HF resizing afterwards.
        return _load_reranker_processor(
            self.model_id,
            use_fast=bool(kwargs.get("use_fast", True)),
            trust_remote_code=get_model_trust_remote_code(self.ctx.model_config),
        )

    def get_supported_mm_limits(self):
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": self.get_max_image_tokens(), "video": self.get_max_video_tokens(seq_len, mm_counts)}


@lru_cache(maxsize=8)
def _load_reranker_processor(model_id, *, use_fast, trust_remote_code):
    from transformers import Qwen3VLProcessor

    return load_hf_pretrained(
        Qwen3VLProcessor, model_id, use_fast=use_fast, padding_side="left", trust_remote_code=trust_remote_code
    )


class RerankerMultiModalProcessor(Qwen3VLMultiModalProcessor):
    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        if self.info.ctx.get_mm_config().video_pruning_rate:
            raise ValueError("Reranker official preprocessing does not prune video tokens")
        visual = _load_merak_visual_config(
            self.info.model_id,
            trust_remote_code=get_model_trust_remote_code(self.info.ctx.model_config),
        )
        if _uses_visual_token_gears(visual):
            capacity = _max_visual_patch_capacity(visual)
            for modality in ("image", "video"):
                for item in out_mm_kwargs.get(modality, []):
                    grid = item[f"{modality}_grid_thw"].data
                    patches = int((grid if modality == "image" else grid[1:]).prod())
                    if patches > capacity:
                        raise ValueError(
                            f"Official {modality} preprocessing needs {patches} patches "
                            "and exceeds exported visual capacity "
                            f"{capacity}; export a larger visual gear to preserve the official input"
                        )
        updates = super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        config = self.info.get_hf_config()
        result = []
        for update in updates:
            if update.modality == "video":
                original = update.replacement

                def with_outer_boundaries(item_idx, replacement=original):
                    details = replacement(item_idx)
                    # HF replaces video_pad only, retaining its outer wrapper.
                    # Native vLLM matches the whole wrapper; restore it here.
                    return PromptUpdateDetails.select_token_id(
                        [config.vision_start_token_id, *details.full, config.vision_end_token_id], config.video_token_id
                    )

                update = replace(update, replacement=with_outer_boundaries)
            result.append(update)
        return result


@MULTIMODAL_REGISTRY.register_processor(
    RerankerMultiModalProcessor, info=RerankerProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
@MERAK_MODEL_REGISTRY.register_module("Qwen3VLRerankerMerakForSequenceClassification")
class Qwen3VLRerankerMerakForSequenceClassification(ContiguousHMONNXForCausalLM):
    is_pooling_model = True
    is_text_generation_model = False
    score_type = "cross-encoder"
    default_seq_pooling_type = "LAST"
    supports_multimodal = True
    supports_multimodal_raw_input_only = True
    requires_raw_input_tokens = True
    supports_mrope = True
    uses_internal_hmonnx_cache = True
    hybrid_cache_policy = HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST
    _language_model_names: list[str] = []
    _tower_model_names: list[str] = []
    _has_oov_mm_tokens = False

    # These methods only depend on the source geometry, not the Qwen3.5 GDN.
    get_mrope_input_positions = Qwen3_5MerakForCausalLM.get_mrope_input_positions
    get_num_mm_encoder_tokens = Qwen3_5MerakForCausalLM.get_num_mm_encoder_tokens
    get_num_mm_connector_tokens = Qwen3_5MerakForCausalLM.get_num_mm_connector_tokens
    _compute_deepstack_embeds = Qwen3VLForConditionalGeneration._compute_deepstack_embeds

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality not in {"image", "video"}:
            raise ValueError("Reranker Merak supports image and video inputs")
        return Qwen3_5MerakForCausalLM.get_placeholder_str(modality, i)

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        kv_cache_dtype="auto",
        *,
        vllm_config: VllmConfig | None = None,
        prefix="",
    ):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if vllm_config is None:
            raise ValueError("Reranker requires vllm_config")
        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        config = model_config.hf_config
        if not isinstance(config, MerakConfig) or model_config.runner_type != "pooling":
            raise ValueError("Reranker requires MerakConfig and runner=pooling")
        if config.num_labels != 1 or model_config.pooler_config.get_seq_pooling_type() != "LAST":
            raise ValueError("Reranker requires num_labels=1 and LAST pooling")
        if any(getattr(model_config.pooler_config, field) is not None for field in ("logit_mean", "logit_sigma")):
            raise ValueError("Reranker scoring does not apply additional logit calibration")
        if vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError("Reranker internal HMONNX cache requires max_num_seqs=1")
        if vllm_config.speculative_config is not None or config.enable_page_attention:
            raise ValueError("Reranker Prefill-only runtime does not support speculative decoding or PageAttention")
        if str(kv_cache_dtype) not in {"auto", "float16", "fp16"}:
            raise ValueError("Reranker HMONNX cache is float16")
        self.config = config
        root = Path(model_config.model)
        metadata = json.loads((root / "golden_meta_info.json").read_text())
        validate_score_contract(root, metadata)
        if metadata["visual_config"].get("visual_input_mode") != "patches":
            raise ValueError("Reranker Merak requires the dynamic patch-token visual export")
        source = json.loads((root / metadata["hf_config"] / "config.json").read_text())
        if source.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
            raise ValueError("Reranker Merak requires Qwen3-VL source architecture")
        self.inference_device = torch.device("cuda:0")
        self.device = torch.device("cpu")
        runtime_kwargs = get_hmonnx_runtime_kwargs(config)
        runtime_kwargs["device_map"] = [self.inference_device]
        self.inference_model = AutoLLMHONNXModel.from_pretrained(str(root / "golden_meta_info.json"), **runtime_kwargs)
        apply_hmonnx_runtime_options(self.inference_model, config)
        self.input_embeddings = self.inference_model.get_input_embeddings()
        self.visual_dim = self.input_embeddings.embedding_dim
        self.deepstack_num_level = 3
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        self._deepstack_input_embeds = None
        self.prefill_chunk_length = int(metadata["model_config"]["prefill_chunk_length"])
        self.max_cache_tokens = int(metadata["model_config"]["context_max_length"])
        if model_config.max_model_len > self.max_cache_tokens:
            raise ValueError("max_model_len exceeds exported Reranker context capacity")
        self.spatial_merge_size = int(metadata["visual_config"]["spatial_merge_size"])
        self.image_token_id = int(config.image_token_id)
        self.video_token_id = int(config.video_token_id)
        self.vision_start_token_id = int(config.vision_start_token_id)
        self.vision_end_token_id = int(config.vision_end_token_id)
        self.enable_page_attention = False
        self.num_layers = 0
        self.key_cache_config = []
        self.value_cache_config = []
        self.cache_dtype = "float16"
        self.cache_torch_dtype = torch.float16
        self._cache_owner_request_id = None
        self._cache_owner_next_past_len = 0
        self.pooler = build_reranker_pooler()
        # Follow the existing contiguous-cache loaders: one tiny scheduler
        # group; actual KV tensors belong to HMONNX, without a duplicate copy.
        self.placeholder_attn_layers = nn.ModuleList(
            [
                MerakAttention(
                    num_heads=1,
                    num_kv_heads=1,
                    head_size=1,
                    scale=1.0,
                    cache_config=vllm_config.cache_config,
                    prefix="model.layers.0.attn",
                    attn_backend=MerakAttentionBackend,
                )
            ]
        )

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        self._deepstack_input_embeds = None
        embeds = super().embed_input_ids(input_ids)
        if multimodal_embeddings is None or not len(multimodal_embeddings):
            return embeds
        if is_multimodal is None:
            raise ValueError("Multimodal inputs require a visual-token mask")
        # Video placeholders can include timestamps and vision delimiters.
        # Only the actual image/video pad positions receive visual features.
        is_multimodal = is_multimodal & ((input_ids == self.image_token_id) | (input_ids == self.video_token_id)).to(
            is_multimodal.device
        )
        self._deepstack_input_embeds, images = self._compute_deepstack_embeds(
            inputs_embeds=embeds, multimodal_embeddings=multimodal_embeddings, is_multimodal=is_multimodal
        )
        return _merge_multimodal_embeddings(
            inputs_embeds=embeds, multimodal_embeddings=images, is_multimodal=is_multimodal
        )

    def embed_multimodal(self, **kwargs):
        outputs = []
        # vLLM orders modality kwargs to match the encoder cache's items.
        # Preserve that order instead of always processing images first.
        for key in kwargs:
            if key not in {"pixel_values", "pixel_values_videos"} or kwargs[key] is None:
                continue
            video = key == "pixel_values_videos"
            pixels = kwargs[key]
            grids = kwargs.get("video_grid_thw" if video else "image_grid_thw")
            if grids is None:
                raise ValueError("Visual pixels require their image/video grid_thw")
            frame_grids = video_frame_grids(pixels, grids, self.spatial_merge_size) if video else grids
            if not video and bool((grids[:, 0] != 1).any()):
                raise ValueError("Image grid_thw must have one temporal unit")
            frames = []
            for image, grid in zip(
                torch.split(pixels, [int(grid.prod()) for grid in frame_grids]), frame_grids, strict=True
            ):
                embeds, deepstack = self.inference_model.visual.encode(image, grid.reshape(1, 3))
                frames.append(torch.cat([embeds, *deepstack], dim=-1).squeeze(0))
            if video:
                offset = 0
                for temporal_units in grids[:, 0].tolist():
                    count = int(temporal_units)
                    outputs.append(torch.cat(frames[offset : offset + count], dim=0))
                    offset += count
            else:
                outputs.extend(frames)
        return tuple(outputs)

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        request_id=None,
        **kwargs: Any,
    ):
        del intermediate_tensors, kwargs
        if input_ids is None:
            raise ValueError("Reranker requires input_ids")
        ids = input_ids.reshape(-1)
        metadata = self._get_attn_metadata()
        if metadata is None:
            # vLLM memory profiling does not have real scheduler metadata.
            return torch.zeros((ids.numel(), 1), dtype=torch.float32, device=ids.device)
        past_lens = metadata.past_lens.tolist()
        bounds = metadata.subsequence_begins.tolist()
        if len(past_lens) != 1 or bounds != [0, ids.numel()]:
            raise RuntimeError("Reranker internal cache requires exactly one request per step")
        past = int(past_lens[0])
        owner = request_id or "__direct__"
        if past and (owner != self._cache_owner_request_id or past != self._cache_owner_next_past_len):
            raise RuntimeError("Reranker cannot resume a missing or mismatched request cache owner")
        if past + ids.numel() > self.max_cache_tokens:
            raise ValueError("Reranker input exceeds exported context capacity")
        runtime = self.inference_model
        runtime.set_prefill()
        if past == 0:
            cache = runtime.get_kvcache_mixin()
            cache.clear_kv_cache()
        with torch.device(self.inference_device):
            self._ensure_runtime_cache_tensors()
        self._cache_owner_request_id = owner
        embeds = inputs_embeds if inputs_embeds is not None else super().embed_input_ids(ids)
        embeds = embeds.to(self.inference_device, self.input_embeddings.weight.dtype)
        if positions is None or tuple(positions.shape) != (3, ids.numel()):
            raise ValueError("Reranker requires global M-RoPE positions with shape [3, tokens]")
        positions = positions.to(self.inference_device, torch.int64)
        deepstack = self._deepstack_input_embeds
        self._deepstack_input_embeds = None
        if deepstack is None:
            deepstack = embeds.new_zeros((3, ids.numel(), self.visual_dim))
        full = (
            embeds.unsqueeze(0),
            *positions.unbind(),
            torch.tensor([past], dtype=torch.int32, device=self.inference_device),
            torch.tensor([ids.numel()], dtype=torch.int32, device=self.inference_device),
            *[feature.unsqueeze(0).to(embeds) for feature in deepstack],
            runtime.past_key_caches,
            runtime.past_value_caches,
        )
        try:
            for chunk in prefill_chunks(
                full, chunk_length=self.prefill_chunk_length, context_length=self.max_cache_tokens - past
            ):
                chunk = list(chunk)
                chunk[4] = chunk[4] + past
                logits = runtime.forward(*chunk)
        except Exception:
            self.release_internal_cache_owner()
            raise
        self._cache_owner_next_past_len = past + ids.numel()
        output = torch.zeros((ids.numel(), 1), device=ids.device, dtype=torch.float32)
        output[-1, 0] = logits.reshape(-1)[0].float().to(ids.device)
        return output
