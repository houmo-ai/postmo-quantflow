# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, cast

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers.image_utils import to_numpy_array
from transformers.models.qwen3_vl import Qwen3VLProcessor
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen2_vl import _create_qwen2vl_field_factory
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ProcessingInfo as VllmQwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
)
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config
from vllm.v1.sample.sampler import Sampler
from xhmodel_merak.xh_llm import AutoLLMHONNXModel
from xhmodel_merak.xh_llm.builder import get_model_class as get_xhmodel_class
from xhmodel_merak.xh_llm.hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
from xhmodel_merak.xh_llm.models.qwen3_5.image_processing_qwen3_5 import (
    Qwen3_5ImageProcessor,
)
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import (
    Qwen3_5HMONNXKVCacheMixin,
    XHQwen3_5_HMONNXModel,
)
from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_hmonnx_inference import (
    XHQwen3_5MoeHMONNXModel,
)

from ...config import MerakConfig
from ...constants import HYBRID_CACHE_POLICY_PAGED
from ...determinism_trace import (
    emit_event as emit_determinism_event,
)
from ...determinism_trace import (
    fingerprint as determinism_fingerprint,
)
from ...determinism_trace import (
    trace_enabled as determinism_trace_enabled,
)
from ...hmfp_kv_cache import is_hmfp_cache_dtype
from ...hmonnx_op_trace import (
    hmonnx_op_trace_scope,
    install_hmonnx_op_tracing,
)
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
from .hybrid_cache_policy import (
    parse_full_attention_cache_shapes,
    resolve_hybrid_page_attention,
)
from .merak_for_causal_lm import MerakForCausalLM
from .merak_linear_state import (
    compute_merak_linear_state_shapes,
    get_merak_linear_kv_cache_config,
)
from .qwen_hybrid_cache_contract import QwenHybridCacheContract

logger = init_logger(__name__)


def _repair_text_only_runtime_model_config(meta_info: Any) -> None:
    """Fill legacy early-snapshot fields needed by the text preprocessor."""

    model_config = getattr(meta_info, "model_config", None)
    if model_config is None:
        return
    if _coerce_optional_int(getattr(model_config, "spatial_merge_size", None)):
        return

    nested_visual = getattr(model_config, "visual_config", None)
    if isinstance(nested_visual, Mapping):
        raw_spatial_merge_size = nested_visual.get("spatial_merge_size")
    else:
        raw_spatial_merge_size = getattr(
            nested_visual,
            "spatial_merge_size",
            None,
        )
    spatial_merge_size = _coerce_optional_int(raw_spatial_merge_size) or 2
    model_config.spatial_merge_size = spatial_merge_size


class _TextOnlyQwen3_5RuntimeMixin:
    """Build the Qwen3.5 language runtime without constructing a visual graph."""

    def __init__(self, meta_info: Any, **kwargs: Any) -> None:
        _repair_text_only_runtime_model_config(meta_info)
        VisonLLMHMONNXModel.__init__(self, meta_info, **kwargs)
        nested_visual = getattr(meta_info.model_config, "visual_config", None)
        if isinstance(nested_visual, Mapping):
            nested_visual = SimpleNamespace(**nested_visual)
        self.visual_meta = nested_visual or SimpleNamespace(
            max_size_w=448,
            max_size_h=448,
            patch_size=16,
        )
        self._kvcache_mixin = Qwen3_5HMONNXKVCacheMixin(self.kvcache_config)
        self._kvcache_mixin.split_conv_cache = bool(
            getattr(meta_info.model_config, "split_conv_cache", False)
        )
        self._sync_page_attention_mode_to_kvcache()

    def _set_device(self, device: torch.device):
        return VisonLLMHMONNXModel._set_device(self, device)

    def _set_dtype(self, dtype: torch.dtype):
        return VisonLLMHMONNXModel._set_dtype(self, dtype)

    def _set_enable_golden(self, enable: bool) -> None:
        del enable


class _TextOnlyXHQwen3_5HMONNXModel(
    _TextOnlyQwen3_5RuntimeMixin,
    XHQwen3_5_HMONNXModel,
):
    pass


class _TextOnlyXHQwen3_5MoeHMONNXModel(
    _TextOnlyQwen3_5RuntimeMixin,
    XHQwen3_5MoeHMONNXModel,
):
    pass


def _load_text_only_qwen3_5_hmonnx_model(meta_path: str, **runtime_kwargs: Any):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    meta_dict["_meta_path_"] = meta_path
    model_config = meta_dict.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Qwen3.5 HMONNX metadata is missing model_config.")

    xhmodel_cls = get_xhmodel_class(model_config)
    meta_info = xhmodel_cls.META_CLS.from_dict(meta_dict)
    upstream_runtime_cls = xhmodel_cls.HMONNXINFERENCE_CLS
    if issubclass(upstream_runtime_cls, XHQwen3_5MoeHMONNXModel):
        runtime_cls = _TextOnlyXHQwen3_5MoeHMONNXModel
    elif issubclass(upstream_runtime_cls, XHQwen3_5_HMONNXModel):
        runtime_cls = _TextOnlyXHQwen3_5HMONNXModel
    else:
        raise TypeError(
            f"Unsupported Qwen3.5 text-only HMONNX runtime class: {upstream_runtime_cls.__name__}"
        )
    runtime_cls.LLM_MODEL_CLS = xhmodel_cls
    return runtime_cls(meta_info, **runtime_kwargs)


def _coerce_image_item_to_pil(image: object) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    array = to_numpy_array(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 0))

    if array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)

    return Image.fromarray(array).convert("RGB")


def _coerce_optional_int(value: object) -> int | None:
    if value is None or isinstance(value, Mapping):
        return None
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=8)
def _load_merak_source_hf_config(
    model_id: str,
    *,
    trust_remote_code: bool,
):
    return cast(
        Qwen3_5Config,
        load_hf_config(
            model_id,
            trust_remote_code=trust_remote_code,
        ),
    )


@lru_cache(maxsize=8)
def _load_merak_visual_config(
    model_id: str,
    *,
    trust_remote_code: bool,
) -> dict[str, Any]:
    meta_file = Path(model_id) / "golden_meta_info.json"
    if meta_file.is_file():
        with meta_file.open("r", encoding="utf-8") as f:
            meta_info = json.load(f)

        visual_config = meta_info.get("visual_config")
        if isinstance(visual_config, dict):
            return visual_config

        model_config = meta_info.get("model_config", {})
        if isinstance(model_config, dict):
            visual_config = model_config.get("visual_config")
            if isinstance(visual_config, dict):
                return {
                    "image_size_w": visual_config.get("max_size_w", 448),
                    "image_size_h": visual_config.get("max_size_h", 448),
                    "patch_size": visual_config.get("patch_size", 16),
                    "temporal_patch_size": visual_config.get("temporal_patch_size", 2),
                    "spatial_merge_size": model_config.get("spatial_merge_size", 2),
                }

    hf_config = _load_merak_source_hf_config(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    vision_config = getattr(hf_config, "vision_config", None)
    return {
        "image_size_w": 448,
        "image_size_h": 448,
        "patch_size": getattr(vision_config, "patch_size", 16),
        "temporal_patch_size": getattr(vision_config, "temporal_patch_size", 2),
        "spatial_merge_size": getattr(vision_config, "spatial_merge_size", 2),
    }


def _uses_visual_token_gears(visual_config: Mapping[str, Any]) -> bool:
    return visual_config.get("visual_input_mode") == "patches" and bool(
        visual_config.get("gears")
        or visual_config.get("image_token_gears")
        or visual_config.get("image_token_capacity")
    )


def _max_visual_patch_capacity(visual_config: Mapping[str, Any]) -> int:
    capacities: list[int] = []
    gears = visual_config.get("gears")
    if isinstance(gears, Sequence) and not isinstance(gears, (str, bytes)):
        for gear in gears:
            if not isinstance(gear, Mapping):
                continue
            patch_capacity = _coerce_optional_int(gear.get("patch_token_capacity"))
            if patch_capacity is not None:
                capacities.append(patch_capacity)
                continue
            image_capacity = _coerce_optional_int(gear.get("image_token_capacity"))
            if image_capacity is not None:
                merge_size = int(visual_config.get("spatial_merge_size", 2))
                capacities.append(image_capacity * merge_size * merge_size)

    top_level_capacity = _coerce_optional_int(visual_config.get("patch_token_capacity"))
    if top_level_capacity is not None:
        capacities.append(top_level_capacity)
    if not capacities:
        image_capacity = _coerce_optional_int(visual_config.get("image_token_capacity"))
        if image_capacity is not None:
            merge_size = int(visual_config.get("spatial_merge_size", 2))
            capacities.append(image_capacity * merge_size * merge_size)

    if not capacities or min(capacities) <= 0:
        raise ValueError(
            "Qwen3.5 visual token-gear metadata is missing a positive patch capacity."
        )
    return max(capacities)


def _configure_merak_image_processor(
    image_processor: Any,
    visual_config: Mapping[str, Any],
) -> None:
    if _uses_visual_token_gears(visual_config):
        patch_size = int(visual_config.get("patch_size", 16))
        max_pixels = _max_visual_patch_capacity(visual_config) * patch_size * patch_size
        current_size = dict(image_processor.size)
        current_min_pixels = _coerce_optional_int(current_size.get("shortest_edge"))
        if current_min_pixels is None:
            merge_size = int(visual_config.get("spatial_merge_size", 2))
            current_min_pixels = (patch_size * merge_size) ** 2
        min_pixels = min(current_min_pixels, max_pixels)
    else:
        max_pixels = int(visual_config.get("image_size_w", 448)) * int(
            visual_config.get("image_size_h", 448)
        )
        min_pixels = max_pixels

    image_processor.size = {
        **dict(image_processor.size),
        "shortest_edge": min_pixels,
        "longest_edge": max_pixels,
    }
    # Older Qwen image processors consume these attributes directly, while
    # current Transformers versions use ``size``.  Keep both contracts in sync.
    image_processor.min_pixels = min_pixels
    image_processor.max_pixels = max_pixels


@lru_cache(maxsize=8)
def _load_merak_qwen3_5_processor(
    model_id: str,
    use_fast: bool = True,
    *,
    trust_remote_code: bool,
) -> Qwen3VLProcessor:
    processor = cast(
        Qwen3VLProcessor,
        load_hf_pretrained(
            Qwen3VLProcessor,
            model_id,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        ),
    )
    visual_config = _load_merak_visual_config(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    _configure_merak_image_processor(processor.image_processor, visual_config)
    return processor


@lru_cache(maxsize=8)
def _load_merak_hm_image_processor(
    model_id: str,
    *,
    trust_remote_code: bool,
) -> Qwen3_5ImageProcessor:
    base_processor = _load_merak_qwen3_5_processor(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    hm_processor = Qwen3_5ImageProcessor(**vars(base_processor.image_processor))

    visual_config = _load_merak_visual_config(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    _configure_merak_image_processor(hm_processor, visual_config)
    return hm_processor


def _build_merak_hm_pixel_values(
    images: Sequence[object],
    model_id: str,
    *,
    trust_remote_code: bool,
) -> torch.Tensor:
    hm_processor = _load_merak_hm_image_processor(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    hm_patches = hm_processor._hm_preprocess(
        images=images,
        do_resize=False,
        resample=hm_processor.resample,
        do_convert_rgb=hm_processor.do_convert_rgb,
    )
    hm_tensor = torch.from_numpy(hm_patches)
    hm_tensor = hm_tensor.unsqueeze(2).repeat(
        1, 1, hm_processor.temporal_patch_size, 1, 1
    )
    return hm_tensor.unsqueeze(1)


def _create_merak_qwen3_5_field_factory(
    spatial_merge_size: int,
):
    qwen2vl_field_factory = _create_qwen2vl_field_factory(spatial_merge_size)

    def _merak_field_config(hf_inputs: Mapping[str, torch.Tensor]):
        field_config = dict(qwen2vl_field_factory(hf_inputs))
        if "hm_pixel_values" in hf_inputs:
            field_config["hm_pixel_values"] = MultiModalFieldConfig.batched("image")
        return field_config

    return _merak_field_config


class MerakQwen3_5MultiModalProcessor(Qwen3VLMultiModalProcessor):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        mm_data = dict(mm_data)
        images = mm_data.get("images")
        trust_remote_code = get_model_trust_remote_code(self.info.ctx.model_config)
        visual_config = _load_merak_visual_config(
            self.info.model_id,
            trust_remote_code=trust_remote_code,
        )
        if _uses_visual_token_gears(visual_config):
            # The HF processor already emits Qwen's native flattened patches
            # and image_grid_thw.  Preserving the source aspect ratio here is
            # what lets the runtime select the smallest fitting static gear.
            return super()._call_hf_processor(
                prompt=prompt,
                mm_data=mm_data,
                mm_kwargs=mm_kwargs,
                tok_kwargs=tok_kwargs,
            )

        target_width = int(visual_config.get("image_size_w", 448))
        target_height = int(visual_config.get("image_size_h", 448))
        resized_images: list[Image.Image] = []
        if isinstance(images, list) and images:
            for image in images:
                resized_images.append(
                    _coerce_image_item_to_pil(image).resize(
                        (target_width, target_height)
                    )
                )
            mm_data["images"] = resized_images

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        if resized_images:
            processed_outputs["hm_pixel_values"] = _build_merak_hm_pixel_values(
                resized_images,
                self.info.model_id,
                trust_remote_code=trust_remote_code,
            )
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_processor_mm_kwargs
        hf_config = cast(Qwen3_5Config, self.info.get_hf_config())
        vision_config = cast(Any, hf_config.vision_config)
        return _create_merak_qwen3_5_field_factory(
            int(getattr(vision_config, "spatial_merge_size", 2)),
        )(hf_inputs)


class MerakQwen3_5ProcessingInfo(VllmQwen3_5ProcessingInfo):
    def get_hf_config(self) -> Qwen3_5Config:
        return _load_merak_source_hf_config(
            self.model_id,
            trust_remote_code=get_model_trust_remote_code(self.ctx.model_config),
        )

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        use_fast = kwargs.pop("use_fast", True)
        return _load_merak_qwen3_5_processor(
            self.model_id,
            use_fast=bool(use_fast),
            trust_remote_code=get_model_trust_remote_code(self.ctx.model_config),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        del seq_len, mm_counts
        return {"image": self.get_max_image_tokens()}


@dataclass(frozen=True)
class _MerakProcessorRegistration:
    processor: Any
    info: Any
    dummy_inputs: Any


@dataclass(frozen=True)
class _MerakArchitectureSpec:
    source_architectures: frozenset[str]
    supports_multimodal: bool = False
    supports_multimodal_raw_input_only: bool = False
    requires_raw_input_tokens: bool = False
    supports_mrope: bool = False
    processor_registration: _MerakProcessorRegistration | None = None


_MERAK_NATIVE_HYBRID_TEXT_SOURCE_ARCHITECTURES = frozenset(
    {
        "Qwen3NextForCausalLM",
        "BailingMoeV3ForCausalLM",
    }
)

_MERAK_TEXT_ARCHITECTURE_SPEC = _MerakArchitectureSpec(
    source_architectures=frozenset(
        {
            "Qwen3ForCausalLM",
            "Qwen2ForCausalLM",
        }
    )
    | _MERAK_NATIVE_HYBRID_TEXT_SOURCE_ARCHITECTURES,
)

_MERAK_QWEN3_5_MULTIMODAL_SPEC = _MerakArchitectureSpec(
    source_architectures=frozenset(
        {
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        }
    ),
    supports_multimodal=True,
    supports_multimodal_raw_input_only=True,
    requires_raw_input_tokens=True,
    supports_mrope=True,
    processor_registration=_MerakProcessorRegistration(
        processor=MerakQwen3_5MultiModalProcessor,
        info=MerakQwen3_5ProcessingInfo,
        dummy_inputs=Qwen3VLDummyInputsBuilder,
    ),
)

_MERAK_QWEN3_5_TEXT_SPEC = _MerakArchitectureSpec(
    source_architectures=_MERAK_QWEN3_5_MULTIMODAL_SPEC.source_architectures,
    requires_raw_input_tokens=True,
    supports_mrope=True,
)

_MERAK_ARCHITECTURE_SPECS: tuple[_MerakArchitectureSpec, ...] = (
    _MERAK_TEXT_ARCHITECTURE_SPEC,
    _MERAK_QWEN3_5_MULTIMODAL_SPEC,
)


def _resolve_merak_architecture_spec(
    source_architecture: str,
) -> _MerakArchitectureSpec:
    for architecture_spec in _MERAK_ARCHITECTURE_SPECS:
        if source_architecture in architecture_spec.source_architectures:
            return architecture_spec

    supported_architectures = sorted(
        architecture
        for architecture_spec in _MERAK_ARCHITECTURE_SPECS
        for architecture in architecture_spec.source_architectures
    )
    raise ValueError(
        f"Unsupported architecture {source_architecture} for Merak model. "
        f"Supported architectures: {supported_architectures}."
    )


@MULTIMODAL_REGISTRY.register_processor(
    MerakQwen3_5MultiModalProcessor,
    info=MerakQwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
@MERAK_MODEL_REGISTRY.register_module(
    ["Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"]
)
class Qwen3_5MerakForCausalLM(MerakForCausalLM):
    supports_multimodal = True
    supports_multimodal_raw_input_only = True
    requires_raw_input_tokens = True
    supports_mrope = True
    supports_encoder_tp_data = False
    # Hybrid / inner-state declaration so vLLM allocates MambaSpec state for the
    # Qwen3.5 linear-attention (split GatedDeltaNet) layers alongside the
    # full-attention paged KV cache.
    is_hybrid = True
    has_inner_state = True
    hybrid_cache_policy = HYBRID_CACHE_POLICY_PAGED
    _processor_factory: Any
    _language_model_names: list[str] = []
    _tower_model_names: list[str] = []
    _has_oov_mm_tokens = False

    def _load_hmonnx_model(
        self,
        hmonnx_meta_file: str,
        runtime_kwargs: dict[str, Any],
    ) -> Any:
        """Load the runtime selected by the concrete source architecture.

        Qwen3.5 text exports still carry a multimodal source architecture, so
        they need the text-only wrapper that suppresses the missing visual
        component. Other hybrid text architectures, such as Qwen3-Next,
        override this hook and load their native HMONNX runtime directly.
        """

        if type(self).supports_multimodal:
            return AutoLLMHONNXModel.from_pretrained(
                hmonnx_meta_file,
                **runtime_kwargs,
            )

        logger.warning(
            "Qwen3.5 export has no visual HMONNX model; image input is "
            "unsupported and the vLLM service will run in text-only mode."
        )
        return _load_text_only_qwen3_5_hmonnx_model(
            hmonnx_meta_file,
            **runtime_kwargs,
        )

    @classmethod
    def register_hf_architecture_processor(
        cls, architecture_spec: _MerakArchitectureSpec
    ):
        processor_registration = architecture_spec.processor_registration
        if processor_registration is None:
            raise ValueError(
                "Cannot register a multimodal processor for a text-only Merak architecture spec."
            )

        return MULTIMODAL_REGISTRY.register_processor(
            processor_registration.processor,
            info=processor_registration.info,
            dummy_inputs=processor_registration.dummy_inputs,
        )

    def _sync_runtime_processor_metadata(self) -> None:
        get_data_preprocessor = getattr(
            self.inference_model, "get_data_preprocessor", None
        )
        if not callable(get_data_preprocessor):
            return

        runtime_data_preprocessor = get_data_preprocessor()
        self.runtime_data_preprocessor = runtime_data_preprocessor

        image_token_id = _coerce_optional_int(
            getattr(runtime_data_preprocessor, "image_token_id", None)
        )
        if image_token_id is not None:
            self.image_token_id = image_token_id

        video_token_id = _coerce_optional_int(
            getattr(runtime_data_preprocessor, "video_token_id", None)
        )
        if video_token_id is not None:
            self.video_token_id = video_token_id

        vision_start_token_id = _coerce_optional_int(
            getattr(runtime_data_preprocessor, "vision_start_token_id", None)
        )
        if vision_start_token_id is not None:
            self.vision_start_token_id = vision_start_token_id

        vision_end_token_id = _coerce_optional_int(
            getattr(runtime_data_preprocessor, "vision_end_token_id", None)
        )
        if vision_end_token_id is not None:
            self.vision_end_token_id = vision_end_token_id

        spatial_merge_size = _coerce_optional_int(
            getattr(runtime_data_preprocessor, "spatial_merge_size", None)
        )
        if spatial_merge_size is not None:
            self.spatial_merge_size = spatial_merge_size

        image_size_w = _coerce_optional_int(
            getattr(runtime_data_preprocessor, "image_size_w", None)
        )
        if image_size_w is not None:
            self.visual_export_config.setdefault("image_size_w", image_size_w)

        image_size_h = _coerce_optional_int(
            getattr(runtime_data_preprocessor, "image_size_h", None)
        )
        if image_size_h is not None:
            self.visual_export_config.setdefault("image_size_h", image_size_h)

        patch_size = _coerce_optional_int(
            getattr(runtime_data_preprocessor, "patch_size", None)
        )
        if patch_size is not None:
            self.visual_export_config.setdefault("patch_size", patch_size)

        self.visual_export_config.setdefault(
            "spatial_merge_size", self.spatial_merge_size
        )

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
                "MerakForCausalLM requires either model_config or vllm_config."
            )

        self.vllm_config = vllm_config
        self.prefix = prefix
        config = model_config.hf_config
        if not isinstance(config, MerakConfig):
            raise TypeError(
                "MerakForCausalLM expects a MerakConfig as hf_config, got "
                f"{type(config).__name__}. Did you forget to set "
                "config_format='merak_llm' on the LLM?"
            )
        self.config: MerakConfig = config
        xh_model = self.config.xh_model
        requested_page_attention = should_enable_page_attention(self.config)
        # Hybrid Merak models (full-attention + linear-attention) are scheduled
        # concurrently by vLLM's hybrid KV-cache manager. Correct concurrent
        # serving requires per-request isolation of BOTH cache families:
        #   * linear-attention state -> MambaSpec block transfer (this class),
        #   * full-attention KV       -> paged save/restore (page attention).
        # With page attention off, the full-attention KV cache is a single
        # shared runtime buffer, so a concurrent request's prefill overwrites
        # another's context (cross-request state bleed). Force it on for hybrid
        # models so a stale ``enable_page_attention=false`` in merak_config.json
        # cannot silently reintroduce crosstalk.
        self.enable_page_attention = resolve_hybrid_page_attention(
            cache_policy=type(self).hybrid_cache_policy,
            requested=requested_page_attention,
        )
        if self.enable_page_attention and not requested_page_attention:
            logger.warning(
                "Merak hybrid model requires paged full-attention KV cache for "
                "correct concurrent serving; overriding enable_page_attention "
                "False -> True (config value ignored for hybrid models)."
            )
            self.enable_page_attention = True
        config_dir_candidates: list[Path] = []
        model_path = getattr(model_config, "model", None)
        if model_path:
            config_dir_candidates.append(Path(model_path))
        name_or_path = getattr(config, "_name_or_path", None)
        if name_or_path:
            config_dir_candidates.append(Path(name_or_path))

        config_dir = next(
            (candidate for candidate in config_dir_candidates if candidate.is_dir()),
            Path(name_or_path or model_path or "."),
        )
        hf_config_dir = config_dir / "hf_config"
        hf_config_file = hf_config_dir / "config.json"
        hf_config = json.load(hf_config_file.open("r", encoding="utf-8"))
        architectures = hf_config.get("architectures") or []
        if not architectures:
            raise ValueError(
                f"Merak model at {config_dir} is missing HF architectures metadata."
            )

        architecture = str(architectures[0])
        self.architecture_spec = _resolve_merak_architecture_spec(architecture)
        if (
            not type(self).supports_multimodal
            and self.architecture_spec.supports_multimodal
        ):
            self.architecture_spec = _MERAK_QWEN3_5_TEXT_SPEC
        logger.info(
            f"Loading Merak model with architecture {architecture} from {config_dir}."
        )
        self.source_architecture = architecture
        self.device = torch.device("cpu")
        self.inference_device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.image_token_id = int(hf_config.get("image_token_id", -1))
        self.video_token_id = int(hf_config.get("video_token_id", -1))
        self.vision_start_token_id = int(hf_config.get("vision_start_token_id", -1))
        self.vision_end_token_id = int(hf_config.get("vision_end_token_id", -1))
        vision_config = hf_config.get("vision_config", {})
        self.spatial_merge_size = int(vision_config.get("spatial_merge_size", 2))
        self.visual_export_config: dict[str, Any] = {}
        self._prefer_input_ids_runtime = False
        self.runtime_data_preprocessor: Any | None = None

        xh_model_type = xh_model.get("model_type", None)
        if xh_model_type is None:
            raise ValueError(
                f"Merak model config at {config_dir} is missing 'model_type' field."
            )

        self.key_cache_config: list[list[int]] = []
        self.value_cache_config: list[list[int]] = []
        cache_dtype = "float16"

        self.num_layers = 0
        self.kv_cache_shape: tuple[int, ...] | None = None
        self.cache_axis = 2
        requested_cache_dtype = (
            kv_cache_dtype.strip().lower()
            if isinstance(kv_cache_dtype, str)
            else "auto"
        )
        if xh_model_type.lower() == "hmonnx":
            hmonnx_meta_file = str(config_dir / "golden_meta_info.json")
            runtime_kwargs = get_hmonnx_runtime_kwargs(self.config)
            # ``get_hmonnx_runtime_kwargs`` derives page attention from the
            # (possibly stale) config, but hybrid models force it on above via
            # ``self.enable_page_attention``. Propagate the resolved value so the
            # backend actually runs ``convert_to_page_attention``; otherwise the
            # runtime keeps a single shared full-attention KV buffer and the
            # per-request ``block_ids`` we pass later have no paged graph to act
            # on, silently reintroducing cross-request crosstalk.
            if self.enable_page_attention:
                runtime_kwargs["enable_page_attention"] = True
            hmonnx_model = self._load_hmonnx_model(
                hmonnx_meta_file,
                runtime_kwargs,
            )
            apply_hmonnx_runtime_options(hmonnx_model, self.config)
            install_hmonnx_op_tracing(hmonnx_model)
            preserve_runtime_device_map = should_preserve_hmonnx_runtime_device_map(
                self.config
            )
            if not preserve_runtime_device_map:
                hmonnx_model.to(self.inference_device)
            self.inference_device = resolve_hmonnx_primary_device(
                hmonnx_model, self.inference_device
            )
            self.inference_model = hmonnx_model
            self._runner_device = self.inference_device
            if type(self).supports_multimodal:
                visual = getattr(hmonnx_model, "visual", None)
                if visual is None:
                    raise RuntimeError(
                        "Qwen3.5 visual HMONNX export did not create a visual runtime component."
                    )
                visual_device = self._validate_visual_gpu_runtime(visual)
                logger.info(
                    "Qwen3.5 visual HMONNX runtime is loaded on GPU device %s.",
                    visual_device,
                )
                shared_weight_summary = getattr(visual, "shared_weight_summary", None)
                if shared_weight_summary is not None:
                    logger.info(
                        "Qwen3.5 visual token-gear shared-weight pool: %s.",
                        shared_weight_summary,
                    )
            move_hmonnx_auxiliary_tensors(hmonnx_model, self.inference_device)
            with open(hmonnx_meta_file, "r", encoding="utf-8") as f:
                hmonnx_meta_info = json.load(f)
                self.qwen_hybrid_cache_contract = QwenHybridCacheContract.from_metadata(
                    hmonnx_meta_info
                )
                self.qwen_hybrid_cache_contract.validate_graphs(hmonnx_model)
                kv_cache_meta = hmonnx_meta_info["kv_cache"]
                model_meta = hmonnx_meta_info.get("model_config", {})
                if isinstance(model_meta, dict):
                    for field_name in (
                        "image_token_id",
                        "video_token_id",
                        "vision_start_token_id",
                        "vision_end_token_id",
                        "spatial_merge_size",
                    ):
                        field_value = _coerce_optional_int(model_meta.get(field_name))
                        if field_value is not None:
                            setattr(self, field_name, field_value)
                    nested_visual_config = model_meta.get("visual_config")
                    if (
                        isinstance(nested_visual_config, dict)
                        and not self.visual_export_config
                    ):
                        self.visual_export_config = {
                            "image_size_w": nested_visual_config.get("max_size_w", 448),
                            "image_size_h": nested_visual_config.get("max_size_h", 448),
                            "patch_size": nested_visual_config.get("patch_size", 16),
                            "temporal_patch_size": nested_visual_config.get(
                                "temporal_patch_size", 2
                            ),
                            "spatial_merge_size": self.spatial_merge_size,
                        }
                visual_config = hmonnx_meta_info.get("visual_config")
                if isinstance(visual_config, dict):
                    self.visual_export_config = visual_config
                cache_shapes = parse_full_attention_cache_shapes(kv_cache_meta)
                num_layers = int(kv_cache_meta["num_layers"])
                self.num_layers = num_layers
                self.key_cache_shape = cache_shapes.key
                self.value_cache_shape = cache_shapes.value
                # Keep the legacy attribute as the key shape for consumers
                # that only need the shared sequence axis/capacity.
                self.kv_cache_shape = cache_shapes.key
                self.cache_axis = cache_shapes.cache_axis
                for _ in range(num_layers):
                    self.key_cache_config.append(list(cache_shapes.key))
                    self.value_cache_config.append(list(cache_shapes.value))
                cache_dtype = kv_cache_meta.get("cache_dtype", "float16")
                if is_hmfp_cache_dtype(requested_cache_dtype):
                    cache_dtype = requested_cache_dtype
            self._sync_runtime_processor_metadata()
            move_hmonnx_auxiliary_tensors(hmonnx_model, self.inference_device)
        self.cache_dtype = cache_dtype
        self.cache_torch_dtype = resolve_torch_dtype(cache_dtype)
        if self.cache_axis != 2:
            raise NotImplementedError(
                f"Merak HMONNX cache_axis={self.cache_axis} is not supported; expected sequence axis at 2."
            )
        if self.kv_cache_shape is None:
            raise RuntimeError(
                f"Failed to determine KV cache shape for Merak model at {config_dir}."
            )

        self.max_cache_tokens = int(self.kv_cache_shape[self.cache_axis])

        hf_model_config = load_hf_config(
            hf_config_dir,
            trust_remote_code=get_model_trust_remote_code(model_config),
        )
        if hasattr(hf_model_config, "text_config"):
            hf_model_config = hf_model_config.text_config
        self._register_placeholder_attention_layers(hf_model_config)
        self._register_placeholder_linear_state_layers(hf_model_config)
        self.logits_processor = LogitsProcessor(
            hf_model_config.vocab_size, logits_as_input=True
        )
        self.input_embeddings = cast(
            nn.Embedding, self.inference_model.get_input_embeddings()
        )
        self._configure_hf_architecture_runtime()

        self.sampler = Sampler()

    def _validate_runtime_linear_state_contract(self) -> None:
        super()._validate_runtime_linear_state_contract()
        contract = getattr(self, "qwen_hybrid_cache_contract", None)
        if isinstance(contract, QwenHybridCacheContract):
            contract.validate_prepared_runtime_state(self.inference_model)

    def _configure_hf_architecture_runtime(self) -> None:
        architecture_spec = self.architecture_spec
        self.supports_multimodal = architecture_spec.supports_multimodal
        self.supports_multimodal_raw_input_only = (
            architecture_spec.supports_multimodal_raw_input_only
        )
        self.requires_raw_input_tokens = architecture_spec.requires_raw_input_tokens
        self.supports_mrope = architecture_spec.supports_mrope

        if self.supports_multimodal:
            if not hasattr(self.inference_model, "visual"):
                raise ValueError(
                    "Merak multimodal models require a visual runtime component."
                )
            self._prefer_input_ids_runtime = True

    def set_runner_device(self, device: torch.device | str) -> None:
        self._runner_device = torch.device(device)

    def _prepare_runtime_data_processor_for_positions(
        self,
        data_preprocessor: Any,
        positions: torch.Tensor,
    ) -> None:
        # The plugin replaces the processor's generated position tensors with
        # vLLM's request-scoped M-RoPE positions below. Keep only a neutral
        # placeholder here so the upstream processor can traverse resumed
        # chunks without depending on mutable rope state from another request.
        processor_device = getattr(data_preprocessor, "_device", None)
        if processor_device is None:
            processor_device = positions.device
        data_preprocessor.rope_deltas = torch.zeros(
            (1, 1),
            device=processor_device,
            dtype=torch.long,
        )

    def _apply_runtime_positions(
        self,
        net_inputs: Any,
        positions: torch.Tensor,
        seq_len: int,
    ) -> tuple[Any, ...]:
        if not isinstance(net_inputs, (list, tuple)) or len(net_inputs) < 4:
            raise TypeError(
                "Qwen3.5 HMONNX preprocessor must return embeddings followed by time/height/width position tensors."
            )
        if positions.dim() != 2 or positions.shape[0] != 3:
            raise ValueError(
                f"Qwen3.5 M-RoPE positions must have shape [3, num_tokens], got {tuple(positions.shape)}."
            )
        if positions.shape[1] != seq_len:
            raise ValueError(
                f"Qwen3.5 M-RoPE position/token count mismatch: positions={positions.shape[1]}, tokens={seq_len}."
            )

        routed_inputs = list(net_inputs)
        for axis in range(3):
            padded_positions = routed_inputs[axis + 1]
            if not isinstance(padded_positions, torch.Tensor):
                raise TypeError("Qwen3.5 HMONNX position inputs must be tensors.")
            chunk_axis_positions = positions[axis].to(
                device=padded_positions.device,
                dtype=padded_positions.dtype,
            )
            routed_inputs[axis + 1] = torch.cat(
                (chunk_axis_positions, padded_positions[seq_len:]),
                dim=0,
            )
        return tuple(routed_inputs)

    @staticmethod
    def _validate_visual_gpu_runtime(visual: Any) -> torch.device:
        visual_device = torch.device(getattr(visual, "device", "cpu"))
        if visual_device.type != "cuda":
            raise RuntimeError(
                f"Qwen3.5 visual HMONNX must run on a GPU, but the runtime reported device {visual_device}."
            )
        return visual_device

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality == "image":
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality == "video":
            return "<|vision_start|><|video_pad|><|vision_end|>"
        return None

    # ------------------------------------------------------------------
    # Hybrid (linear-attention) state shape/dtype — Merak split GDN layout.
    # ------------------------------------------------------------------
    @classmethod
    def _get_linear_kv_cache_config(cls, vllm_config: VllmConfig) -> dict:
        linear_cfg = get_merak_linear_kv_cache_config(vllm_config)
        if linear_cfg is None:
            raise ValueError(
                "Qwen3_5MerakForCausalLM requires 'linear_kv_cache_config' on the "
                "MerakConfig for hybrid state allocation. Check golden_meta_info.json "
                "propagation in config.py."
            )
        return linear_cfg

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], ...]:
        linear_cfg = cls._get_linear_kv_cache_config(vllm_config)
        return compute_merak_linear_state_shapes(linear_cfg)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        linear_cfg = cls._get_linear_kv_cache_config(vllm_config)
        cache_dtype = str(linear_cfg.get("cache_dtype", "float16"))
        torch_dtype = resolve_torch_dtype(cache_dtype)
        # Four split states: conv_q, conv_k, conv_v, recurrent.
        return (torch_dtype, torch_dtype, torch_dtype, torch_dtype)

    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
        if not self.supports_multimodal:
            raise NotImplementedError(
                f"Merak architecture {self.source_architecture} does not support multimodal encoder tokens."
            )
        return num_image_tokens * (self.spatial_merge_size**2)

    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int:
        if not self.supports_multimodal:
            raise NotImplementedError(
                f"Merak architecture {self.source_architecture} does not support multimodal connector tokens."
            )
        return num_vision_tokens

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[Any],
    ) -> tuple[torch.Tensor, int]:
        if not self.supports_mrope:
            raise NotImplementedError(
                f"Merak architecture {self.source_architecture} does not support M-RoPE input positions."
            )

        return Qwen3VLForConditionalGeneration._get_mrope_input_positions(
            input_tokens=input_tokens,
            mm_features=cast(Any, mm_features),
            config=cast(Any, self.config),
        )

    def _embed_text_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.input_embeddings(
            input_ids.to(self.input_embeddings.weight.device)
        ).to(input_ids.device)

    def embed_input_ids(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        if not self.supports_multimodal:
            raise NotImplementedError(
                f"Merak architecture {self.source_architecture} does not support multimodal embeddings."
            )

        staged_embeddings = self._stage_multimodal_embeddings_for_merge(
            inputs_embeds,
            multimodal_embeddings,
        )
        merged = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=staged_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )
        if determinism_trace_enabled():
            emit_determinism_event(
                "merged_input_embeddings",
                payload={
                    "input_ids": determinism_fingerprint(input_ids),
                    "multimodal_embeddings": determinism_fingerprint(staged_embeddings),
                    "is_multimodal": determinism_fingerprint(
                        _require_is_multimodal(is_multimodal)
                    ),
                    "merged": determinism_fingerprint(merged),
                },
            )
        return merged

    @staticmethod
    def _stage_multimodal_embeddings_for_merge(
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
    ) -> MultiModalEmbeddings:
        if isinstance(multimodal_embeddings, torch.Tensor):
            return multimodal_embeddings.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
        return tuple(
            embedding.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            for embedding in multimodal_embeddings
        )

    def _normalize_visual_embeds(self, visual_embeds: Any) -> torch.Tensor:
        if isinstance(visual_embeds, (list, tuple)):
            visual_embeds = visual_embeds[0]
        if not isinstance(visual_embeds, torch.Tensor):
            raise TypeError(
                f"Unexpected Merak visual output type: {type(visual_embeds).__name__}"
            )
        if visual_embeds.dim() == 3:
            if visual_embeds.shape[0] != 1:
                raise ValueError(
                    f"Unexpected Merak visual output shape: {tuple(visual_embeds.shape)}"
                )
            return visual_embeds.squeeze(0)
        if visual_embeds.dim() != 2:
            raise ValueError(
                f"Unexpected Merak visual output shape: {tuple(visual_embeds.shape)}"
            )
        return visual_embeds

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:  # type: ignore[override]
        if not self.supports_multimodal:
            if any(
                kwargs.get(key) is not None
                for key in (
                    "hm_pixel_values",
                    "pixel_values",
                    "image_grid_thw",
                    "pixel_values_videos",
                    "video_grid_thw",
                )
            ):
                raise NotImplementedError(
                    f"Merak architecture {self.source_architecture} does not support multimodal inputs."
                )
            return []

        if kwargs.get("pixel_values_videos") is not None or kwargs.get("video_grid_thw") is not None:
            pixels, grids = kwargs.get("pixel_values_videos"), kwargs.get("video_grid_thw")
            if not isinstance(pixels, torch.Tensor) or not isinstance(grids, torch.Tensor):
                raise ValueError("Merak video inputs require pixel_values_videos and video_grid_thw tensors together")
            visual = cast(Any, getattr(self.inference_model, "visual"))
            encode_videos = getattr(visual, "encode_videos", None)
            if not callable(encode_videos):
                raise ValueError("Merak video inputs require the patch-token visual runtime")
            image_kwargs = {key: value for key, value in kwargs.items()
                            if key not in {"pixel_values_videos", "video_grid_thw"}}
            images = tuple(self.embed_multimodal(**image_kwargs))
            with hmonnx_op_trace_scope("visual", request_id=None):
                outputs = encode_videos(pixels, grids)
            videos = tuple(self._normalize_visual_embeds(output).to(
                device=self._runner_device, dtype=self.input_embeddings.weight.dtype,
            ) for output in outputs)
            if len(videos) != grids.shape[0]:
                raise ValueError("Merak visual runtime must return one tensor per video")
            # vLLM's multimodal field ordering determines the embedding order.
            first = next(key for key in kwargs if kwargs[key] is not None and key in {
                "pixel_values", "image_grid_thw", "hm_pixel_values", "pixel_values_videos", "video_grid_thw",
            })
            return videos + images if first in {"pixel_values_videos", "video_grid_thw"} else images + videos

        visual = cast(Any, getattr(self.inference_model, "visual"))
        hm_pixel_values = kwargs.get("hm_pixel_values")
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        encode_many = getattr(visual, "encode_many", None)
        if callable(encode_many):
            if pixel_values is None and image_grid_thw is None:
                if hm_pixel_values is not None:
                    raise ValueError(
                        "Qwen3.5 visual token gears require native flattened "
                        "pixel_values and image_grid_thw; hm_pixel_values is the "
                        "legacy fixed-resolution input."
                    )
                return []
            if pixel_values is None or image_grid_thw is None:
                raise ValueError(
                    "Qwen3.5 visual token gears require pixel_values and "
                    "image_grid_thw together."
                )
            if not isinstance(pixel_values, torch.Tensor):
                raise TypeError(
                    "Merak pixel_values must be a tensor, got "
                    f"{type(pixel_values).__name__}"
                )
            if not isinstance(image_grid_thw, torch.Tensor):
                raise TypeError(
                    "Merak image_grid_thw must be a tensor, got "
                    f"{type(image_grid_thw).__name__}"
                )
            if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
                raise ValueError(
                    "Merak image_grid_thw must have shape [num_images, 3], got "
                    f"{tuple(image_grid_thw.shape)}"
                )
            expected_patches = sum(
                int(torch.prod(grid).item()) for grid in image_grid_thw
            )
            if pixel_values.ndim != 2 or pixel_values.shape[0] != expected_patches:
                raise ValueError(
                    "Merak flattened pixel_values/image_grid_thw patch count "
                    f"mismatch: pixels={tuple(pixel_values.shape)}, "
                    f"expected_patches={expected_patches}."
                )

            if determinism_trace_enabled():
                emit_determinism_event(
                    "hmonnx_visual_token_gear_input",
                    payload={
                        "pixels": determinism_fingerprint(pixel_values),
                        "image_grid_thw": determinism_fingerprint(image_grid_thw),
                    },
                )
            with hmonnx_op_trace_scope("visual", request_id=None):
                visual_outputs = encode_many(pixel_values, image_grid_thw)
            if not isinstance(visual_outputs, (list, tuple)):
                raise TypeError(
                    "Qwen3.5 visual token-gear encode_many must return one "
                    f"tensor per image, got {type(visual_outputs).__name__}."
                )
            if len(visual_outputs) != image_grid_thw.shape[0]:
                raise ValueError(
                    "Qwen3.5 visual token-gear output count mismatch: "
                    f"outputs={len(visual_outputs)}, images={image_grid_thw.shape[0]}."
                )

            image_embeddings = tuple(
                self._normalize_visual_embeds(output).to(
                    device=self._runner_device,
                    dtype=self.input_embeddings.weight.dtype,
                )
                for output in visual_outputs
            )
            if determinism_trace_enabled():
                emit_determinism_event(
                    "hmonnx_visual_token_gear_output",
                    payload={
                        "outputs": [
                            determinism_fingerprint(output)
                            for output in image_embeddings
                        ]
                    },
                )
            return image_embeddings

        if hm_pixel_values is not None:
            image_embeddings: list[torch.Tensor] = []
            if isinstance(hm_pixel_values, torch.Tensor):
                if hm_pixel_values.dim() == 6:
                    hm_pixel_value_items = list(hm_pixel_values.unbind(0))
                elif hm_pixel_values.dim() == 5:
                    if hm_pixel_values.shape[0] == 1:
                        hm_pixel_value_items = [hm_pixel_values]
                    else:
                        hm_pixel_value_items = [
                            item.unsqueeze(0) for item in hm_pixel_values.unbind(0)
                        ]
                else:
                    raise ValueError(
                        f"Unexpected Merak hm_pixel_values tensor shape: {tuple(hm_pixel_values.shape)}"
                    )
            else:
                hm_pixel_value_items = list(cast(Any, hm_pixel_values))

            for image_pixels in hm_pixel_value_items:
                if not isinstance(image_pixels, torch.Tensor):
                    raise TypeError(
                        f"Merak hm_pixel_values items must be tensors, got {type(image_pixels).__name__}"
                    )
                staged_pixels = image_pixels.to(
                    device=visual.device, dtype=visual.dtype
                )
                if determinism_trace_enabled():
                    emit_determinism_event(
                        "hmonnx_visual_input",
                        payload={"pixels": determinism_fingerprint(staged_pixels)},
                    )
                with hmonnx_op_trace_scope("visual", request_id=None):
                    visual_output = visual.forward(staged_pixels)
                normalized_output = self._normalize_visual_embeds(visual_output).to(
                    device=self._runner_device,
                    dtype=self.input_embeddings.weight.dtype,
                )
                if determinism_trace_enabled():
                    emit_determinism_event(
                        "hmonnx_visual_output",
                        payload={"output": determinism_fingerprint(normalized_output)},
                    )
                image_embeddings.append(normalized_output)

            return tuple(image_embeddings)

        if pixel_values is None or image_grid_thw is None:
            return []
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(
                f"Merak pixel_values must be a tensor, got {type(pixel_values).__name__}"
            )
        if not isinstance(image_grid_thw, torch.Tensor):
            raise TypeError(
                f"Merak image_grid_thw must be a tensor, got {type(image_grid_thw).__name__}"
            )

        patch_counts = [int(torch.prod(grid).item()) for grid in image_grid_thw]
        image_embeddings: list[torch.Tensor] = []
        for image_pixels in pixel_values.split(patch_counts, dim=0):
            staged_pixels = image_pixels.to(device=visual.device, dtype=visual.dtype)
            if determinism_trace_enabled():
                emit_determinism_event(
                    "hmonnx_visual_input",
                    payload={"pixels": determinism_fingerprint(staged_pixels)},
                )
            with hmonnx_op_trace_scope("visual", request_id=None):
                visual_output = visual.forward(staged_pixels)
            normalized_output = self._normalize_visual_embeds(visual_output).to(
                device=self._runner_device,
                dtype=self.input_embeddings.weight.dtype,
            )
            if determinism_trace_enabled():
                emit_determinism_event(
                    "hmonnx_visual_output",
                    payload={"output": determinism_fingerprint(normalized_output)},
                )
            image_embeddings.append(normalized_output)

        return tuple(image_embeddings)

    def _count_multimodal_items(
        self, input_ids: Optional[torch.Tensor]
    ) -> tuple[int, int]:
        if input_ids is None or self.vision_start_token_id < 0:
            return 0, 0

        flat_input_ids = self._flatten_input_ids(input_ids)
        if flat_input_ids.numel() < 2:
            return 0, 0

        vision_start_indices = torch.argwhere(
            flat_input_ids[:-1] == self.vision_start_token_id
        ).squeeze(1)
        if vision_start_indices.numel() == 0:
            return 0, 0

        vision_tokens = flat_input_ids[vision_start_indices + 1]
        num_images = (
            int((vision_tokens == self.image_token_id).sum().item())
            if self.image_token_id >= 0
            else 0
        )
        num_videos = (
            int((vision_tokens == self.video_token_id).sum().item())
            if self.video_token_id >= 0
            else 0
        )
        return num_images, num_videos

    def _extract_request_image_embeds(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if input_ids is None or inputs_embeds is None or self.image_token_id < 0:
            return None

        flat_input_ids = self._flatten_input_ids(input_ids)
        flat_inputs_embeds = self._flatten_inputs_embeds(inputs_embeds)
        image_mask = flat_input_ids == self.image_token_id
        if not torch.any(image_mask):
            return None

        return flat_inputs_embeds[image_mask]

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
        request_kwargs: dict[str, torch.Tensor] = {}
        num_images, num_videos = self._count_multimodal_items(input_ids)
        flat_input_ids = (
            self._flatten_input_ids(input_ids) if input_ids is not None else None
        )
        num_image_tokens = (
            int((flat_input_ids == self.image_token_id).sum().item())
            if flat_input_ids is not None and self.image_token_id >= 0
            else 0
        )

        next_image_offset = image_grid_offset + num_images

        # Each video contributes one timestamped span per temporal group. The
        # batch grid still has one row per video; offsets here count group rows.
        next_video_offset = video_grid_offset
        if flat_input_ids is not None and self.video_token_id >= 0:
            video_mask = flat_input_ids == self.video_token_id
            if torch.any(video_mask):
                if inputs_embeds is None:
                    raise ValueError("Merak multimodal request is missing merged video embeddings")
                request_kwargs["video_embeds"] = self._flatten_inputs_embeds(inputs_embeds)[video_mask]
        if num_videos and video_grid_thw is not None:
            if video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3:
                raise ValueError("Merak video_grid_thw must have shape [num_videos, 3]")
            if video_grid_thw.dtype not in (torch.int32, torch.int64) or (video_grid_thw <= 0).any():
                raise ValueError("Merak video_grid_thw must contain positive integer dimensions")
            groups = video_grid_thw.repeat_interleave(video_grid_thw[:, 0], dim=0).clone()
            groups[:, 0] = 1
            next_video_offset = video_grid_offset + num_videos
            request_grids = groups[video_grid_offset:next_video_offset]
            if request_grids.shape[0] != num_videos:
                raise ValueError(f"Merak multimodal request expected {num_videos} video group grids")
            request_kwargs["video_grid_thw"] = request_grids
        # Cached/resumed chunks may contain video tokens without a new grid or
        # vision-start token. Their merged features and vLLM M-RoPE are sufficient.

        if num_images == 0 and num_image_tokens == 0:
            return request_kwargs, next_image_offset, next_video_offset

        request_image_embeds = self._extract_request_image_embeds(
            input_ids, inputs_embeds
        )
        if request_image_embeds is None:
            raise ValueError(
                "Merak multimodal request is missing merged image embeddings."
            )
        request_kwargs["image_embeds"] = request_image_embeds

        # vLLM may resume a chunk in the middle of an image placeholder span.
        # In that case the current slice has image tokens but no vision-start
        # token or new grid entry. The already-computed M-RoPE positions are
        # passed separately, so only the corresponding embedding rows are needed.
        if num_images == 0:
            return request_kwargs, next_image_offset, next_video_offset

        if image_grid_thw is None:
            # Raw multimodal processor fields are emitted only for
            # ``scheduled_new_reqs``.  A long prefill can resume on a later
            # scheduler step whose slice starts at another image placeholder;
            # vLLM still supplies the cached image embeddings and the already
            # computed M-RoPE positions, but no longer repeats image_grid_thw.
            # The grid was consumed by the visual encoder and is unnecessary
            # for this LLM slice, so do not consume a batch-level grid offset.
            return request_kwargs, image_grid_offset, next_video_offset

        request_image_grid_thw = image_grid_thw[image_grid_offset:next_image_offset]
        if request_image_grid_thw.shape[0] != num_images:
            raise ValueError(
                f"Merak multimodal request expected {num_images} image grids, got {request_image_grid_thw.shape[0]}."
            )

        request_kwargs["image_grid_thw"] = request_image_grid_thw
        return request_kwargs, next_image_offset, next_video_offset


@MERAK_MODEL_REGISTRY.register_module("Qwen3_5MerakTextForCausalLM")
class Qwen3_5MerakTextForCausalLM(Qwen3_5MerakForCausalLM):
    supports_multimodal = False
    supports_multimodal_raw_input_only = False
    requires_raw_input_tokens = True
    supports_mrope = True


def get_model(
    vllm_config: VllmConfig,
    kv_cache_dtype: str,
    **kwargs,
) -> torch.nn.Module:
    del kwargs
    with cast(AbstractContextManager[None], set_current_vllm_config(vllm_config)):
        loader_architecture = getattr(
            vllm_config.model_config.hf_config,
            "merak_loader_architecture",
            Qwen3_5MerakForCausalLM.__name__,
        )
        model_cls = (
            Qwen3_5MerakTextForCausalLM
            if loader_architecture == Qwen3_5MerakTextForCausalLM.__name__
            else Qwen3_5MerakForCausalLM
        )
        return model_cls(vllm_config=vllm_config, kv_cache_dtype=kv_cache_dtype)
