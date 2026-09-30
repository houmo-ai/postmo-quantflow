from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, cast

import torch
from torch import nn
from transformers.models.gemma4 import Gemma4Config, Gemma4Processor
from transformers.models.gemma4_unified.configuration_gemma4_unified import (
    Gemma4UnifiedConfig,
)
from transformers.models.gemma4_unified.processing_gemma4_unified import (
    Gemma4UnifiedProcessor,
)
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.gemma4_mm import (
    Gemma4DummyInputsBuilder,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
)
from vllm.model_executor.models.gemma4_unified import (
    Gemma4UnifiedProcessingInfo,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    _require_is_multimodal,
)
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing.processor import PromptUpdateDetails
from vllm.v1.sample.sampler import Sampler
from xhmodel_merak.xh_llm import AutoLLMHONNXModel

from ...config import MerakConfig
from ...hmfp_kv_cache import is_hmfp_cache_dtype
from ...utils import (
    apply_hmonnx_runtime_options,
    get_hmonnx_runtime_kwargs,
    move_hmonnx_auxiliary_tensors,
    resolve_hmonnx_primary_device,
    resolve_torch_dtype,
    should_enable_page_attention,
    should_preserve_hmonnx_runtime_device_map,
)
from ..register import MERAK_MODEL_REGISTRY
from .hf_assets import (
    get_model_trust_remote_code,
    load_hf_config,
    load_hf_pretrained,
)
from .gemma4_cache_contract import (
    Gemma4CacheContract,
    parse_gemma4_cache_contract,
)
from .merak_for_causal_lm import MerakForCausalLM

logger = init_logger(__name__)

_GEMMA4_TOWER_SOURCE_ARCHITECTURES = frozenset(
    {"Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"}
)
_GEMMA4_UNIFIED_SOURCE_ARCHITECTURES = frozenset(
    {"Gemma4UnifiedForConditionalGeneration"}
)
_GEMMA4_PREFILL_CHUNK_SIZE = 320
_GEMMA4_MAX_VISUAL_TOKENS = 280


def _validated_visual_ranges(
    visual_ranges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    for visual_range in visual_ranges:
        if len(visual_range) != 2:
            raise ValueError(
                "Gemma4 visual ranges must be inclusive (start, end) pairs"
            )
        start, end = visual_range
        if type(start) is not int or type(end) is not int or start > end:
            raise ValueError(
                f"Gemma4 visual range {visual_range!r} is not a valid inclusive range"
            )
        if start < 0:
            raise ValueError(
                f"Gemma4 visual range starts before token zero: {visual_range}"
            )
        length = end - start + 1
        if length > _GEMMA4_MAX_VISUAL_TOKENS:
            raise ValueError(
                f"Gemma4 visual range {visual_range} has {length} tokens; "
                f"maximum is {_GEMMA4_MAX_VISUAL_TOKENS}"
            )
        normalized.append((start, end))

    normalized.sort()
    for previous, current in zip(normalized, normalized[1:]):
        if current[0] <= previous[1]:
            raise ValueError(
                f"Gemma4 inclusive visual ranges overlap: {previous} and {current}"
            )
    return tuple(normalized)


def plan_gemma4_prefill_chunks(
    total_tokens: int,
    chunk_size: int,
    visual_ranges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Plan half-open 320-token chunks without splitting inclusive visual spans."""

    if type(total_tokens) is not int or total_tokens < 0:
        raise ValueError(
            f"Gemma4 total_tokens must be non-negative, got {total_tokens!r}"
        )
    if chunk_size != _GEMMA4_PREFILL_CHUNK_SIZE:
        raise ValueError(
            f"Gemma4 atomic prefill requires chunk_size=320, got {chunk_size!r}"
        )
    ranges = _validated_visual_ranges(visual_ranges)
    for start, end in ranges:
        if start >= total_tokens:
            raise ValueError(
                f"Gemma4 visual range {(start, end)} is outside {total_tokens} tokens"
            )
        if end >= total_tokens:
            raise RuntimeError(
                "Gemma4 visual span must remain atomic; an upstream scheduler "
                f"cut inclusive range {(start, end)} at token {total_tokens}"
            )

    chunks: list[tuple[int, int]] = []
    start = 0
    while start < total_tokens:
        for visual_start, visual_end in ranges:
            if visual_start < start <= visual_end:
                raise RuntimeError(
                    "Gemma4 visual span must remain atomic; an upstream scheduler "
                    f"started a chunk at {start} inside {(visual_start, visual_end)}"
                )

        end = min(start + chunk_size, total_tokens)
        for visual_start, visual_end in ranges:
            if start < visual_start < end <= visual_end:
                end = visual_start
                break
        if end <= start:
            raise RuntimeError("Gemma4 atomic chunk planner made no forward progress")
        chunks.append((start, end))
        start = end
    return tuple(chunks)


def _visual_ranges_for_chunk(
    start: int,
    end: int,
    ranges: Sequence[tuple[int, int]],
    max_ranges: int,
) -> torch.Tensor:
    """Return fixed-capacity absolute inclusive ranges intersecting one query."""

    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        raise ValueError(
            f"Gemma4 chunk must be a valid half-open range, got {(start, end)}"
        )
    if type(max_ranges) is not int or max_ranges <= 0:
        raise ValueError(f"Gemma4 max_ranges must be positive, got {max_ranges!r}")
    normalized = _validated_visual_ranges(ranges)
    selected = [
        (visual_start, visual_end)
        for visual_start, visual_end in normalized
        if not (visual_end < start or visual_start >= end)
    ]
    if len(selected) > max_ranges:
        raise ValueError(
            f"Gemma4 chunk has {len(selected)} visual ranges; capacity={max_ranges}"
        )
    padded = selected + [(0, 0)] * (max_ranges - len(selected))
    return torch.tensor(padded, dtype=torch.int32).unsqueeze(0)


def _resolve_model_dir(model_config: ModelConfig, config: MerakConfig) -> Path:
    candidates = [
        getattr(model_config, "model", None),
        getattr(config, "_name_or_path", None),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    raise ValueError("Gemma4 Merak loader requires a readable model directory")


def _resolve_meta_path(config_dir: Path, xh_model: dict) -> Path:
    meta_path = Path(str(xh_model.get("meta_info") or "golden_meta_info.json"))
    if not meta_path.is_absolute():
        meta_path = config_dir / meta_path
    if not meta_path.is_file():
        raise FileNotFoundError(f"Gemma4 HMONNX metadata does not exist: {meta_path}")
    return meta_path


@lru_cache(maxsize=8)
def _load_merak_gemma4_source_config(
    model_id: str,
    *,
    trust_remote_code: bool,
) -> Gemma4Config | Gemma4UnifiedConfig:
    return cast(
        Gemma4Config | Gemma4UnifiedConfig,
        load_hf_config(
            model_id,
            trust_remote_code=trust_remote_code,
        ),
    )


@lru_cache(maxsize=8)
def _load_merak_gemma4_unified_processor(
    model_id: str,
    *,
    trust_remote_code: bool,
) -> Gemma4UnifiedProcessor:
    return load_hf_pretrained(
        Gemma4UnifiedProcessor,
        model_id,
        trust_remote_code=trust_remote_code,
    )


@lru_cache(maxsize=8)
def _load_merak_gemma4_meta(model_id: str) -> dict:
    model_dir = Path(model_id)
    merak_config = json.loads(
        (model_dir / "merak_config.json").read_text(encoding="utf-8")
    )
    xh_model = merak_config.get("xh_model", {})
    meta_path = Path(str(xh_model.get("meta_info") or "golden_meta_info.json"))
    if not meta_path.is_absolute():
        meta_path = model_dir / meta_path
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _augment_hmonnx_visual_inputs(
    outputs: Any,
    *,
    visual_token_count: int,
    pooling_kernel_size: int,
) -> Any:
    """Pad HF image patches and build the four exported visual inputs."""

    pixel_values = outputs.get("pixel_values")
    position_ids = outputs.get("pixel_position_ids")
    if pixel_values is None or position_ids is None:
        return outputs
    if not isinstance(pixel_values, torch.Tensor) or not isinstance(
        position_ids, torch.Tensor
    ):
        raise TypeError("Gemma4 HMONNX visual inputs must be tensors")
    if position_ids.dim() == 2:
        position_ids = position_ids.unsqueeze(0)
    if pixel_values.dim() == 2:
        pixel_values = pixel_values.unsqueeze(0)
    kernel_area = pooling_kernel_size * pooling_kernel_size
    target_patches = visual_token_count * kernel_area
    current_patches = int(position_ids.shape[-2])
    if current_patches > target_patches:
        raise ValueError(
            "Gemma4 processor exceeds exported visual patch capacity: "
            f"current={current_patches}, target={target_patches}"
        )
    if current_patches < target_patches:
        pad = target_patches - current_patches
        pixel_values = torch.nn.functional.pad(pixel_values, (0, 0, 0, pad))
        position_ids = torch.nn.functional.pad(position_ids, (0, 0, 0, pad), value=-1)

    valid_mask = ~(position_ids == -1).all(dim=-1)
    valid_counts = valid_mask.sum(dim=-1)
    if torch.any(valid_counts % kernel_area):
        raise ValueError(
            "Gemma4 visual patch count must be divisible by pooling kernel area"
        )
    normalized_positions = position_ids.clone()
    normalized_positions[~valid_mask] = 0
    batch = int(position_ids.shape[0])
    pooling_matrix = torch.zeros(
        (batch, visual_token_count, target_patches),
        dtype=torch.float16,
        device=position_ids.device,
    )
    for batch_idx in range(batch):
        valid_rows = torch.nonzero(valid_mask[batch_idx], as_tuple=False).flatten()
        if valid_rows.numel() == 0:
            continue
        valid_positions = position_ids[batch_idx, valid_rows]
        pooled_width = (
            int(valid_positions[:, 0].max().item()) + 1
        ) // pooling_kernel_size
        kernel_xy = torch.div(
            valid_positions, pooling_kernel_size, rounding_mode="floor"
        )
        kernel_indices = kernel_xy[:, 0] + pooled_width * kernel_xy[:, 1]
        for output_idx in range(int(valid_rows.numel()) // kernel_area):
            selected = valid_rows[kernel_indices == output_idx]
            if selected.numel() != kernel_area:
                raise ValueError(
                    "Gemma4 visual pooling expects "
                    f"{kernel_area} patches for output {output_idx}, "
                    f"got {selected.numel()}"
                )
            pooling_matrix[batch_idx, output_idx, selected] = 1.0 / kernel_area
    negative = torch.tensor(
        torch.finfo(torch.float16).min,
        dtype=torch.float16,
        device=valid_mask.device,
    )
    visual_attention_mask = torch.zeros(
        (batch, 1, 1, target_patches),
        dtype=torch.float16,
        device=valid_mask.device,
    ).masked_fill(~valid_mask[:, None, None, :], negative)
    outputs["pixel_values"] = pixel_values
    outputs["pixel_position_ids"] = normalized_positions.to(torch.int32)
    outputs["pooling_matrix"] = pooling_matrix
    outputs["visual_attention_mask"] = visual_attention_mask
    outputs["image_soft_token_count"] = (valid_counts // kernel_area).to(torch.long)
    return outputs


@lru_cache(maxsize=8)
def _load_merak_gemma4_processor(
    model_id: str,
    *,
    trust_remote_code: bool,
) -> Gemma4Processor:
    processor = load_hf_pretrained(
        Gemma4Processor,
        model_id,
        trust_remote_code=trust_remote_code,
    )
    visual = _load_merak_gemma4_meta(model_id).get("visual_config")
    if not isinstance(visual, dict):
        raise ValueError("Gemma4 multimodal processor requires visual_config")
    image_seq_length = visual.get("num_image_tokens")
    if type(image_seq_length) is not int or not 0 < image_seq_length <= 280:
        raise ValueError(
            "Gemma4 visual_config.num_image_tokens must be in [1, 280], "
            f"got {image_seq_length!r}"
        )
    processor.image_seq_length = image_seq_length
    image_processor = processor.image_processor
    image_processor.max_soft_tokens = image_seq_length
    return processor


class MerakGemma4ProcessingInfo(Gemma4ProcessingInfo):
    def get_hf_config(self) -> Gemma4Config:
        return cast(
            Gemma4Config,
            _load_merak_gemma4_source_config(
                self.model_id,
                trust_remote_code=get_model_trust_remote_code(self.ctx.model_config),
            ),
        )

    def get_hf_processor(self, **kwargs: object) -> Gemma4Processor:
        del kwargs
        return _load_merak_gemma4_processor(
            self.model_id,
            trust_remote_code=get_model_trust_remote_code(self.ctx.model_config),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # The current Merak loader routes the exported four-input image VIT.
        # Do not inherit upstream video/audio budgets until their distinct
        # HMONNX input contracts are wired into embed_multimodal().
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None:
        del seq_len, mm_counts
        visual = _load_merak_gemma4_meta(self.model_id).get("visual_config")
        if not isinstance(visual, dict):
            raise ValueError("Gemma4 multimodal processor requires visual_config")
        visual_token_count = visual.get("num_image_tokens")
        if (
            type(visual_token_count) is not int
            or not 0 < visual_token_count <= _GEMMA4_MAX_VISUAL_TOKENS
        ):
            raise ValueError(
                "Gemma4 visual_config.num_image_tokens must be in [1, 280], "
                f"got {visual_token_count!r}"
            )
        # vLLM budgets encoder *outputs*, not the padded VIT input patches.
        # Gemma4 feeds up to 280 * 9 patches to HMONNX, pools them to at most
        # 280 embeddings, then trims those embeddings to the actual count.
        return {"image": visual_token_count}


class MerakGemma4UnifiedProcessingInfo(Gemma4UnifiedProcessingInfo):
    """Use the source checkpoint processor and checkpoint-owned MM budgets."""

    def get_hf_config(self) -> Gemma4UnifiedConfig:
        config = _load_merak_gemma4_source_config(
            self.model_id,
            trust_remote_code=get_model_trust_remote_code(self.ctx.model_config),
        )
        if not isinstance(config, Gemma4UnifiedConfig):
            raise TypeError(
                "Gemma4 Unified processing requires Gemma4UnifiedConfig, got "
                f"{type(config).__name__}"
            )
        return config

    def get_hf_processor(self, **kwargs: object) -> Gemma4UnifiedProcessor:
        del kwargs
        return _load_merak_gemma4_unified_processor(
            self.model_id,
            trust_remote_code=get_model_trust_remote_code(self.ctx.model_config),
        )

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None:
        tokens = dict(super().get_mm_max_tokens_per_item(seq_len, mm_counts) or {})
        # The exported static image graph is sized by the checkpoint config.
        # In particular, do not let request-level max_soft_tokens silently
        # change the 12B contract from 280 to another upstream-supported size.
        tokens["image"] = int(self.get_hf_config().vision_config.num_soft_tokens)
        return tokens

    def get_audio_repl(
        self,
        *,
        audio_len: int,
        processor: Gemma4UnifiedProcessor | None,
    ) -> PromptUpdateDetails[list[int]]:
        """Expand raw Unified audio frames, not the legacy tower subsampler."""

        if processor is None:
            processor = self.get_hf_processor()
        config = self.get_hf_config()
        samples_per_token = int(config.audio_config.audio_samples_per_token)
        num_tokens = min(
            (int(audio_len) + samples_per_token - 1) // samples_per_token,
            int(processor.audio_seq_length),
        )
        token_ids = (
            [int(config.boa_token_id)]
            + [int(processor.audio_token_id)] * num_tokens
            + [int(getattr(config, "eoa_token_id", config.eoa_token_index))]
        )
        return PromptUpdateDetails.select_token_id(
            token_ids, int(processor.audio_token_id)
        )


class MerakGemma4MultiModalProcessor(Gemma4MultiModalProcessor):
    """Use the exported Gemma visual preprocessing tensors with vLLM routing."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        outputs = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
        visual = _load_merak_gemma4_meta(self.info.model_id).get("visual_config")
        if not isinstance(visual, dict):
            raise ValueError("Gemma4 multimodal processor requires visual_config")
        visual_token_count = int(visual["num_image_tokens"])
        pooling_kernel_size = int(visual.get("pooling_kernel_size", 3))
        return _augment_hmonnx_visual_inputs(
            outputs,
            visual_token_count=visual_token_count,
            pooling_kernel_size=pooling_kernel_size,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        for name in (
            "pooling_matrix",
            "visual_attention_mask",
            "image_soft_token_count",
        ):
            if name in hf_inputs:
                fields[name] = MultiModalFieldConfig.batched("image")
        return fields


class MerakGemma4UnifiedMultiModalProcessor(Gemma4MultiModalProcessor):
    """Keep vLLM preprocessing aligned with the static Unified side graphs."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        for scope in ("images_kwargs", "videos_kwargs"):
            scoped = mm_kwargs.get(scope)
            if isinstance(scoped, Mapping) and "max_soft_tokens" in scoped:
                raise ValueError(
                    f"Gemma4 Unified {scope}.max_soft_tokens is checkpoint-owned"
                )
        if "max_soft_tokens" in mm_kwargs:
            raise ValueError("Gemma4 Unified max_soft_tokens is checkpoint-owned")

        outputs = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
        config = cast(Gemma4UnifiedConfig, self.info.get_hf_config())
        image_tokens = int(config.vision_config.num_soft_tokens)
        image_positions = outputs.get("pixel_position_ids")
        if (
            isinstance(image_positions, torch.Tensor)
            and int(image_positions.shape[-2]) != image_tokens
        ):
            raise ValueError(
                "Gemma4 Unified processor does not match the exported image "
                f"capacity: expected {image_tokens}, got "
                f"{int(image_positions.shape[-2])}"
            )

        processor = cast(Gemma4UnifiedProcessor, self.info.get_hf_processor())
        video_tokens = int(processor.video_processor.max_soft_tokens)
        video_positions = outputs.get("pixel_position_ids_videos")
        if (
            isinstance(video_positions, torch.Tensor)
            and int(video_positions.shape[-2]) != video_tokens
        ):
            raise ValueError(
                "Gemma4 Unified processor does not match the exported video "
                f"capacity: expected {video_tokens}, got "
                f"{int(video_positions.shape[-2])}"
            )

        audio_tokens = int(processor.audio_seq_length)
        features = outputs.get("input_features_padded")
        masks = outputs.get("input_features_mask")
        if isinstance(features, list) and isinstance(masks, list):
            padded_features: list[torch.Tensor] = []
            padded_masks: list[torch.Tensor] = []
            for feature, mask in zip(features, masks, strict=True):
                current = int(feature.shape[-2])
                if current > audio_tokens:
                    raise ValueError(
                        "Gemma4 Unified audio exceeds the exported capacity: "
                        f"{current} > {audio_tokens}"
                    )
                pad = audio_tokens - current
                padded_features.append(torch.nn.functional.pad(feature, (0, 0, 0, pad)))
                padded_masks.append(
                    torch.nn.functional.pad(mask.to(torch.bool), (0, pad), value=False)
                )
            outputs["input_features_padded"] = padded_features
            outputs["input_features_mask"] = padded_masks
        return outputs


@MULTIMODAL_REGISTRY.register_processor(
    MerakGemma4MultiModalProcessor,
    info=MerakGemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
@MERAK_MODEL_REGISTRY.register_module(list(_GEMMA4_TOWER_SOURCE_ARCHITECTURES))
class Gemma4MerakForCausalLM(MerakForCausalLM):
    supports_multimodal = True
    supports_multimodal_raw_input_only = True
    requires_raw_input_tokens = True
    supports_encoder_tp_data = False
    _prefer_input_ids_runtime = True
    _processor_factory: Any
    _source_architectures = _GEMMA4_TOWER_SOURCE_ARCHITECTURES

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
            raise ValueError(
                "Gemma4MerakForCausalLM requires either model_config or vllm_config"
            )

        config = model_config.hf_config
        if not isinstance(config, MerakConfig):
            raise TypeError(
                "Gemma4MerakForCausalLM expects MerakConfig as hf_config; "
                "set config_format='merak_llm'"
            )
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.config = config
        self.enable_page_attention = should_enable_page_attention(config)

        source_architectures = set(getattr(config, "source_architectures", ()))
        if not source_architectures.intersection(self._source_architectures):
            raise ValueError(
                f"{type(self).__name__} requires one of the exact source "
                f"architectures {sorted(self._source_architectures)}"
            )

        xh_model = config.xh_model
        if (
            not isinstance(xh_model, dict)
            or str(xh_model.get("model_type", "")).lower() != "hmonnx"
        ):
            raise ValueError(
                "Gemma4MerakForCausalLM requires xh_model.model_type='hmonnx'"
            )

        config_dir = _resolve_model_dir(model_config, config)
        meta_path = _resolve_meta_path(config_dir, xh_model)
        hmonnx_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        contract = getattr(config, "gemma4_cache_contract", None)
        if not isinstance(contract, Gemma4CacheContract):
            contract = parse_gemma4_cache_contract(
                hf_config=config,
                hmonnx_meta=hmonnx_meta,
            )
        if contract.prefill_chunk_size != _GEMMA4_PREFILL_CHUNK_SIZE:
            raise ValueError(
                "Gemma4 atomic visual prefill requires prefill_chunk_size=320, "
                f"got {contract.prefill_chunk_size}"
            )
        self.gemma4_cache_contract = contract
        max_mm_ranges_per_chunk = hmonnx_meta.get("max_mm_ranges_per_chunk")
        if type(max_mm_ranges_per_chunk) is not int or max_mm_ranges_per_chunk <= 0:
            raise ValueError(
                "Gemma4 max_mm_ranges_per_chunk must be a positive integer, "
                f"got {max_mm_ranges_per_chunk!r}"
            )
        self.max_mm_ranges_per_chunk = max_mm_ranges_per_chunk

        self.device = torch.device("cpu")
        self.inference_device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        hmonnx_model = AutoLLMHONNXModel.from_pretrained(
            str(meta_path),
            **get_hmonnx_runtime_kwargs(config),
        )
        runtime_input_sequence_length = int(hmonnx_model.get_input_sequence_length())
        if runtime_input_sequence_length < _GEMMA4_PREFILL_CHUNK_SIZE:
            raise ValueError(
                "Gemma4 atomic visual prefill requires runtime input sequence "
                f"length >= 320, got {runtime_input_sequence_length}"
            )
        apply_hmonnx_runtime_options(hmonnx_model, config)
        if not should_preserve_hmonnx_runtime_device_map(config):
            hmonnx_model.to(self.inference_device)
        self.inference_device = resolve_hmonnx_primary_device(
            hmonnx_model, self.inference_device
        )
        self.inference_model = hmonnx_model
        self._runner_device = self.inference_device
        move_hmonnx_auxiliary_tensors(hmonnx_model, self.inference_device)

        self.num_layers = len(contract.bindings)
        self.cache_axis = 2
        self.key_cache_config = [
            list(binding.kv_shape) for binding in contract.bindings
        ]
        self.value_cache_config = [
            list(binding.kv_shape) for binding in contract.bindings
        ]
        requested_cache_dtype = (
            kv_cache_dtype.strip().lower()
            if isinstance(kv_cache_dtype, str)
            else "auto"
        )
        cache_dtype = (
            requested_cache_dtype
            if is_hmfp_cache_dtype(requested_cache_dtype)
            else getattr(config, "kv_cache_format", "float16")
        )
        self.cache_dtype = cache_dtype
        self.cache_torch_dtype = resolve_torch_dtype(cache_dtype)
        self.max_cache_tokens = max(
            shape[self.cache_axis] for shape in contract.cache_shapes.values()
        )

        # Task 3 owns vLLM placeholder-group construction and exact group-order
        # resolution. Keep the loader boundary usable without predicting that order.
        self.placeholder_attn_layers = nn.ModuleList()
        text_config = config.get_text_config()
        self.logits_processor = LogitsProcessor(
            int(text_config.vocab_size), logits_as_input=True
        )
        self.input_embeddings = cast(
            nn.Embedding, self.inference_model.get_input_embeddings()
        )
        source_config = _load_merak_gemma4_source_config(
            str(config_dir),
            trust_remote_code=get_model_trust_remote_code(model_config),
        )
        self.image_token_id = int(getattr(source_config, "image_token_id", -1))
        self.video_token_id = int(getattr(source_config, "video_token_id", -1))
        self.audio_token_id = int(getattr(source_config, "audio_token_id", -1))
        self.sampler = Sampler()

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality == "image":
            return "<image_soft_token>"
        if modality == "audio":
            return "<audio_soft_token>"
        if modality == "video":
            return "<|video|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def set_runner_device(self, device: torch.device | str) -> None:
        self._runner_device = torch.device(device)

    @staticmethod
    def _normalize_visual_embeds(output: Any) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Unexpected Gemma4 visual output: {type(output).__name__}")
        if output.dim() == 3:
            if output.shape[0] != 1:
                raise ValueError(
                    f"Unexpected Gemma4 visual shape {tuple(output.shape)}"
                )
            output = output.squeeze(0)
        if output.dim() != 2:
            raise ValueError(f"Unexpected Gemma4 visual shape {tuple(output.shape)}")
        return output

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:  # type: ignore[override]
        names = (
            "pixel_values",
            "pixel_position_ids",
            "pooling_matrix",
            "visual_attention_mask",
        )
        values = [kwargs.get(name) for name in names]
        if all(value is None for value in values):
            return []
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError(
                "Gemma4 visual execution requires pixel_values, "
                "pixel_position_ids, pooling_matrix, and visual_attention_mask"
            )
        tensors = cast(list[torch.Tensor], values)
        counts = kwargs.get("image_soft_token_count")
        if not isinstance(counts, torch.Tensor):
            raise TypeError("Gemma4 visual execution requires image_soft_token_count")
        flat_counts = counts.reshape(-1).tolist()
        batch = int(tensors[0].shape[0])
        if any(int(tensor.shape[0]) != batch for tensor in tensors):
            raise ValueError("Gemma4 visual inputs disagree on image batch size")
        if len(flat_counts) != batch:
            raise ValueError("Gemma4 visual token counts disagree with image batch")
        visual = cast(Any, getattr(self.inference_model, "visual"))
        result: list[torch.Tensor] = []
        for image_idx, count in enumerate(flat_counts):
            visual_inputs = [
                tensor[image_idx : image_idx + 1].to(
                    device=visual.device,
                    dtype=(
                        visual.dtype if tensor.is_floating_point() else tensor.dtype
                    ),
                )
                for tensor in tensors
            ]
            output = self._normalize_visual_embeds(visual.forward(*visual_inputs))
            result.append(
                output[: int(count)].to(
                    device=self._runner_device,
                    dtype=self.input_embeddings.weight.dtype,
                )
            )
        return tuple(result)

    def embed_input_ids(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = super().embed_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        staged = tuple(
            embedding.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            for embedding in multimodal_embeddings
        )
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=staged,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )

    def _extract_request_multimodal_kwargs(  # type: ignore[override]
        self,
        *,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        image_grid_offset: int,
        video_grid_thw: Optional[torch.Tensor],
        video_grid_offset: int,
    ) -> tuple[dict[str, torch.Tensor], int, int]:
        del image_grid_thw, video_grid_thw
        if input_ids is None or inputs_embeds is None or self.image_token_id < 0:
            return {}, image_grid_offset, video_grid_offset
        flat_ids = self._flatten_input_ids(input_ids)
        mask = flat_ids == self.image_token_id
        if not torch.any(mask):
            return {}, image_grid_offset, video_grid_offset
        flat_embeds = self._flatten_inputs_embeds(inputs_embeds)
        return (
            {"image_embeds": flat_embeds[mask]},
            image_grid_offset,
            video_grid_offset,
        )


@MULTIMODAL_REGISTRY.register_processor(
    MerakGemma4UnifiedMultiModalProcessor,
    info=MerakGemma4UnifiedProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
@MERAK_MODEL_REGISTRY.register_module(list(_GEMMA4_UNIFIED_SOURCE_ARCHITECTURES))
class Gemma4UnifiedMerakForCausalLM(Gemma4MerakForCausalLM):
    """Gemma4 12B encoder-free image/video/audio HMONNX deployment adapter."""

    _source_architectures = _GEMMA4_UNIFIED_SOURCE_ARCHITECTURES

    @staticmethod
    def _graph_input(
        tensor: torch.Tensor,
        graph: Any,
    ) -> torch.Tensor:
        return tensor.to(
            device=graph.device,
            dtype=graph.dtype if tensor.is_floating_point() else tensor.dtype,
        )

    def _run_unified_visual_graph(
        self,
        *,
        graph: Any,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if position_ids.dim() != 3 or position_ids.shape[-1] != 2:
            raise ValueError(
                "Gemma4 Unified visual position IDs must have shape [B, T, 2], "
                f"got {tuple(position_ids.shape)}"
            )
        if int(position_ids.shape[0]) != 1:
            raise ValueError("Gemma4 Unified side graphs execute one item at a time")
        valid = ~(position_ids == -1).all(dim=-1).squeeze(0)
        normalized_positions = position_ids.clone()
        normalized_positions[normalized_positions < 0] = 0
        output = self._normalize_visual_embeds(
            graph.forward(
                self._graph_input(pixel_values, graph),
                self._graph_input(normalized_positions.to(torch.int32), graph),
            )
        )
        return output[valid.to(device=output.device)]

    def _process_unified_images(
        self, kwargs: Mapping[str, object]
    ) -> list[torch.Tensor]:
        pixel_values = kwargs.get("pixel_values")
        position_ids = kwargs.get("pixel_position_ids")
        if not isinstance(pixel_values, torch.Tensor):
            return []
        if not isinstance(position_ids, torch.Tensor):
            raise TypeError(
                "Gemma4 Unified image execution requires pixel_position_ids"
            )
        if pixel_values.shape[0] != position_ids.shape[0]:
            raise ValueError("Gemma4 Unified image inputs disagree on batch size")
        graph = getattr(self.inference_model, "visual", None)
        if graph is None:
            raise RuntimeError("Gemma4 Unified image HMONNX graph is not loaded")
        return [
            self._run_unified_visual_graph(
                graph=graph,
                pixel_values=pixel_values[index : index + 1],
                position_ids=position_ids[index : index + 1],
            )
            for index in range(int(pixel_values.shape[0]))
        ]

    def _process_unified_videos(
        self, kwargs: Mapping[str, object]
    ) -> list[torch.Tensor]:
        pixel_values = kwargs.get("pixel_values_videos")
        position_ids = kwargs.get("pixel_position_ids_videos")
        frame_counts = kwargs.get("video_frame_counts")
        if not isinstance(pixel_values, torch.Tensor):
            return []
        if not isinstance(position_ids, torch.Tensor) or not isinstance(
            frame_counts, torch.Tensor
        ):
            raise TypeError(
                "Gemma4 Unified video execution requires position IDs and frame counts"
            )
        graph = getattr(self.inference_model, "video_visual", None)
        if graph is None:
            raise RuntimeError("Gemma4 Unified video HMONNX graph is not loaded")
        counts = [int(value) for value in frame_counts.reshape(-1).tolist()]
        if sum(counts) != int(pixel_values.shape[0]):
            raise ValueError(
                "Gemma4 Unified video frame counts do not match the frame batch"
            )
        if pixel_values.shape[0] != position_ids.shape[0]:
            raise ValueError("Gemma4 Unified video inputs disagree on frame count")
        per_frame = [
            self._run_unified_visual_graph(
                graph=graph,
                pixel_values=pixel_values[index : index + 1],
                position_ids=position_ids[index : index + 1],
            )
            for index in range(int(pixel_values.shape[0]))
        ]
        per_video: list[torch.Tensor] = []
        offset = 0
        for count in counts:
            frames = per_frame[offset : offset + count]
            if not frames:
                raise ValueError("Gemma4 Unified video must contain at least one frame")
            per_video.append(torch.cat(frames, dim=0))
            offset += count
        return per_video

    def _process_unified_audio(
        self, kwargs: Mapping[str, object]
    ) -> list[torch.Tensor]:
        input_features = kwargs.get("input_features_padded")
        input_mask = kwargs.get("input_features_mask")
        if not isinstance(input_features, torch.Tensor):
            return []
        if not isinstance(input_mask, torch.Tensor):
            raise TypeError(
                "Gemma4 Unified audio execution requires input_features_mask"
            )
        if input_features.dim() == 4 and input_features.shape[1] == 1:
            input_features = input_features.squeeze(1)
        if input_mask.dim() == 3 and input_mask.shape[1] == 1:
            input_mask = input_mask.squeeze(1)
        if input_features.dim() != 3 or input_mask.dim() != 2:
            raise ValueError(
                "Gemma4 Unified audio tensors must have shapes [B, T, F] and [B, T]"
            )
        if input_features.shape[:2] != input_mask.shape:
            raise ValueError("Gemma4 Unified audio feature and mask shapes disagree")
        graph = getattr(self.inference_model, "audio", None)
        if graph is None:
            raise RuntimeError("Gemma4 Unified audio HMONNX graph is not loaded")
        per_audio: list[torch.Tensor] = []
        for index in range(int(input_features.shape[0])):
            output = self._normalize_visual_embeds(
                graph.forward(
                    self._graph_input(input_features[index : index + 1], graph)
                )
            )
            valid = input_mask[index].to(device=output.device, dtype=torch.bool)
            per_audio.append(output[valid])
        return per_audio

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:  # type: ignore[override]
        result: list[torch.Tensor] = []
        handled: set[str] = set()
        for key in kwargs:
            if key == "pixel_values" and "image" not in handled:
                result.extend(self._process_unified_images(kwargs))
                handled.add("image")
            elif key == "pixel_values_videos" and "video" not in handled:
                result.extend(self._process_unified_videos(kwargs))
                handled.add("video")
            elif key == "input_features_padded" and "audio" not in handled:
                result.extend(self._process_unified_audio(kwargs))
                handled.add("audio")
        return tuple(
            embedding.to(
                device=self._runner_device,
                dtype=self.input_embeddings.weight.dtype,
            )
            for embedding in result
        )

    def _extract_request_multimodal_kwargs(  # type: ignore[override]
        self,
        *,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        image_grid_offset: int,
        video_grid_thw: Optional[torch.Tensor],
        video_grid_offset: int,
    ) -> tuple[dict[str, torch.Tensor], int, int]:
        del image_grid_thw, video_grid_thw
        if input_ids is None or inputs_embeds is None:
            return {}, image_grid_offset, video_grid_offset
        flat_ids = self._flatten_input_ids(input_ids)
        flat_embeds = self._flatten_inputs_embeds(inputs_embeds)
        extracted: dict[str, torch.Tensor] = {}
        for modality in ("image", "video", "audio"):
            token_id = int(getattr(self, f"{modality}_token_id", -1))
            if token_id < 0:
                continue
            mask = flat_ids == token_id
            if torch.any(mask):
                extracted[f"{modality}_embeds"] = flat_embeds[mask]
        return extracted, image_grid_offset, video_grid_offset


def get_model(
    vllm_config: VllmConfig,
    kv_cache_dtype: str,
    **kwargs,
) -> torch.nn.Module:
    del kwargs
    with cast(AbstractContextManager[None], set_current_vllm_config(vllm_config)):
        source_architectures = set(
            getattr(vllm_config.model_config.hf_config, "source_architectures", ())
        )
        model_cls = (
            Gemma4UnifiedMerakForCausalLM
            if source_architectures.intersection(_GEMMA4_UNIFIED_SOURCE_ARCHITECTURES)
            else Gemma4MerakForCausalLM
        )
        return model_cls(
            vllm_config=vllm_config,
            kv_cache_dtype=kv_cache_dtype,
        )


__all__ = [
    "Gemma4MerakForCausalLM",
    "Gemma4UnifiedMerakForCausalLM",
    "MerakGemma4UnifiedProcessingInfo",
    "get_model",
    "plan_gemma4_prefill_chunks",
]
