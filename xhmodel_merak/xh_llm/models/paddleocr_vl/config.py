# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from xhmodel_merak.configuration_utils import HFModelConfig

from ...vision_llm_model import VisionLLMModelConfig
from .visual_token_gears import VISUAL_INPUT_PATCHES, normalize_image_token_gears


def _require_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


class XHPaddleOCRVLVisualConfig(HFModelConfig):
    def __init__(
        self,
        *,
        max_size_h=896,
        max_size_w=896,
        patch_size=14,
        spatial_merge_size=2,
        visual_input_mode=VISUAL_INPUT_PATCHES,
        image_token_gears=None,
        image_token_capacity=None,
        rotary_cache_length=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = _require_int(patch_size, "patch_size")
        self.spatial_merge_size = _require_int(spatial_merge_size, "spatial_merge_size")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        if visual_input_mode != VISUAL_INPUT_PATCHES:
            raise ValueError(
                f"PaddleOCR visual_input_mode only supports {VISUAL_INPUT_PATCHES!r}, got {visual_input_mode!r}"
            )
        self.visual_input_mode = VISUAL_INPUT_PATCHES
        for name, value in (("max_size_h", max_size_h), ("max_size_w", max_size_w)):
            value = _require_int(value, name)
            if value <= 0 or value % (self.patch_size * self.spatial_merge_size):
                raise ValueError(f"{name} must be positive and divisible by patch_size * spatial_merge_size")
            setattr(self, name, int(value))
        if image_token_gears is None:
            image_token_gears = [1280]
        normalized_gears = normalize_image_token_gears(image_token_gears)
        if image_token_capacity is None:
            image_token_capacity = normalized_gears[-1]
        image_token_capacity = _require_int(image_token_capacity, "image_token_capacity")
        if image_token_capacity < normalized_gears[-1]:
            raise ValueError(
                "image_token_capacity must cover image_token_gears: "
                f"got {image_token_capacity}, largest gear is {normalized_gears[-1]}"
            )
        minimum_rope_cache_length = normalized_gears[-1] * self.spatial_merge_size
        if rotary_cache_length is None:
            rotary_cache_length = minimum_rope_cache_length
        rotary_cache_length = _require_int(rotary_cache_length, "rotary_cache_length")
        if rotary_cache_length < minimum_rope_cache_length:
            raise ValueError(
                "rotary_cache_length must cover the largest PaddleOCR visual gear: "
                f"got {rotary_cache_length}, need at least {minimum_rope_cache_length}"
            )
        self.image_token_gears = normalized_gears
        self.image_token_capacity = int(image_token_capacity)
        self.rotary_cache_length = int(rotary_cache_length)


class XHPaddleOCRVLModelConfig(VisionLLMModelConfig):
    def __init__(self, *, visual_config=None, **kwargs):
        kwargs.setdefault("context_max_length", 16384)
        kwargs.setdefault("prefill_chunk_length", 256)
        super().__init__(**kwargs)
        if self.batch_size != 1:
            raise ValueError("PaddleOCR deployment currently requires batch_size=1")
        if visual_config is None:
            visual_config = {}
        if isinstance(visual_config, dict):
            visual_config = dict(visual_config)
            visual_config.setdefault("model_name", f"{self.model_name}_visual")
            visual_config.setdefault("model_type", "PaddleOCRVLForConditionalGeneration_visual")
            visual_config.setdefault("hf_model", self.hf_model)
            visual_config.setdefault("chip_arch", self.chip_arch)
            visual_config.setdefault("dtype", self.dtype)
            visual_config.setdefault("quant_scheme", self.quant_scheme)
            visual_config = XHPaddleOCRVLVisualConfig(**visual_config)
        self.visual_config = visual_config
