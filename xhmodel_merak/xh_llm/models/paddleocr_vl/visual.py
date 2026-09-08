# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import torch
from transformers import AutoModelForImageTextToText, PaddleOCRVLForConditionalGeneration

from xhquant.api import FrontendType, get_xhquant_logger, to_frontend_graph
from xhquant.frontend.utils import module_graph_forward_guard

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...llm_data_processor import BaseVisualProcessor
from ...types import LLMModelState, VisualModelMeta
from ._vision_model_impl import build_static_image_embedding_model
from .config import XHPaddleOCRVLVisualConfig
from .hf import load_paddleocr_config, load_paddleocr_model
from .visual_token_gears import (
    VISUAL_ATTENTION_MASK_FORMAT,
    VISUAL_ATTENTION_MASK_OPERATOR,
    VISUAL_ATTENTION_MASK_SHAPE,
    VISUAL_INPUT_PATCHES,
    VISUAL_POSITION_FORMAT,
    VISUAL_ROTARY_POSITION_FORMAT,
    build_visual_gear_manifest,
    build_visual_token_gear_inputs,
    factor_image_token_grid,
    patch_token_capacity,
)


PADDLEOCR_VL_VISUAL_MODEL_TYPE = "PaddleOCRVLForConditionalGeneration_visual"
PADDLEOCR_VISUAL_INPUT_NAMES = (
    "pixel_values",
    "position_ids",
    "position_weights",
    "rotary_position_ids",
    "attention_mask",
)


class _PaddleOCRVisualTokenInputProcessor(BaseVisualProcessor):
    def forward(self, data: dict) -> tuple[torch.Tensor, ...]:
        missing = [name for name in PADDLEOCR_VISUAL_INPUT_NAMES if name not in data]
        if missing:
            raise ValueError(f"missing PaddleOCR visual token inputs: {missing}")
        return tuple(data[name] for name in PADDLEOCR_VISUAL_INPUT_NAMES)


def _select_native_model_root(hf_model: Any) -> Any:
    if isinstance(hf_model, PaddleOCRVLForConditionalGeneration):
        return hf_model.model
    if hasattr(hf_model, "visual") and hasattr(hf_model, "projector"):
        return hf_model
    raise TypeError(
        "PaddleOCR-VL visual export requires a PaddleOCRVLForConditionalGeneration "
        "or its inner model with visual and projector modules."
    )


def _snapshot_native_visual_source(native_root: Any) -> Any:
    return SimpleNamespace(
        visual=copy.deepcopy(native_root.visual),
        projector=copy.deepcopy(native_root.projector),
        config=native_root.config,
        # Parent wrapping and visual export cast the shared modules in place.
        # Keep the small frequency module at its loaded precision so rebuilding
        # another grid does not derive RoPE constants from rounded frequencies.
        rotary_pos_emb=copy.deepcopy(native_root.visual.vision_model.encoder.rotary_pos_emb),
    )


@register_llm_model("PaddleOCRVLForConditionalGeneration_visual", master=False)
class XHPaddleOCRVLVisualModel(BaseVisionModel):
    transformers_min_version = "5.16.1"
    transformers_max_version = "5.16.1"
    HF_MODEL_CLS = PaddleOCRVLForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    HF_MODEL_DTYPE = torch.bfloat16
    BUILD_HF_COMPATIBLE_FUNC = None
    HMONNXINFERENCE_CLS = None
    META_CLS = VisualModelMeta
    CONFIG_CLS = XHPaddleOCRVLVisualConfig

    def __init__(self, config: XHPaddleOCRVLVisualConfig) -> None:
        super().__init__(config)
        self.config = cast(XHPaddleOCRVLVisualConfig, self.config)
        if self.config.model_type is None:
            self.config.model_type = PADDLEOCR_VL_VISUAL_MODEL_TYPE

    @staticmethod
    def _raise_checkpoint_owner_error() -> NoReturn:
        raise RuntimeError(
            "XHPaddleOCRVLVisualModel expects the main PaddleOCR-VL checkpoint "
            "owner to pass a loaded native model into init_wrap_model()."
        )

    def get_native_model(self) -> Any:
        if self.hf_model_dir is None:
            self._raise_checkpoint_owner_error()
        return load_paddleocr_model(self.hf_model_dir)

    def get_empty_native_model(self) -> Any:
        self._raise_checkpoint_owner_error()

    def get_compatible_native_model(self) -> Any:
        return self.get_native_model()

    def to_wrap(self, hf_model: Any = None):
        if hf_model is None:
            hf_model = getattr(self, "_native_visual_source", None) or getattr(self, "_native_model_owner", None)
        return super().to_wrap(hf_model)

    def init_wrap_model(self, hf_model: Any = None) -> Any:
        native_visual_source = getattr(self, "_native_visual_source", None)
        if hf_model is None:
            hf_model = native_visual_source or getattr(self, "_native_model_owner", None)
        if hf_model is None:
            hf_model = self.get_native_model()
        if hf_model is native_visual_source:
            native_root = native_visual_source
        else:
            native_root = _select_native_model_root(hf_model)
            self._native_model_owner = hf_model
            self._native_model_root = native_root
            self._native_visual_source = _snapshot_native_visual_source(native_root)
        profile_source = copy.deepcopy(self._native_visual_source)
        grid_thw = factor_image_token_grid(self.config.image_token_capacity, self.config.spatial_merge_size)
        self._wrap_model = build_static_image_embedding_model(
            profile_source,
            grid_thw,
            self.config.image_token_capacity,
            merge_major_input=True,
        )
        self._visual_num_positions = int(self._wrap_model.num_positions)
        self._wrap_model.to(dtype=self.dtype)
        return self._wrap_model

    def _to_fronted(self, wrap_model):
        dummy_inputs = [
            value.float().cpu() if torch.is_floating_point(value) else value.cpu()
            for value in self.get_dummy_inputs().values()
        ]
        get_xhquant_logger().info("Converting PaddleOCR-VL patch-token visual model through TorchFX frontend")
        return to_frontend_graph(wrap_model.float().cpu(), FrontendType.TorchFX, dummy_inputs)

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        image_capacity = int(self.config.image_token_capacity)
        patch_count = patch_token_capacity(image_capacity, self.config.spatial_merge_size)
        grid_thw = factor_image_token_grid(image_capacity, self.config.spatial_merge_size)
        dummy = {
            "pixel_values": torch.zeros(
                1,
                patch_count,
                3,
                int(self.config.patch_size),
                int(self.config.patch_size),
                dtype=torch.float32,
            )
        }
        num_positions = int(getattr(getattr(self, "_wrap_model", None), "num_positions", 4096))
        rotary_cache_length = int(
            getattr(getattr(self, "_wrap_model", None), "rotary_cache_length", self.config.rotary_cache_length)
        )
        dummy.update(
            build_visual_token_gear_inputs(
                grid_thw,
                patch_capacity=patch_count,
                num_positions=num_positions,
                spatial_merge_size=self.config.spatial_merge_size,
                dtype=torch.float32,
                rotary_cache_length=rotary_cache_length,
            )
        )
        return {name: cast(torch.Tensor, dummy[name]) for name in PADDLEOCR_VISUAL_INPUT_NAMES}

    def _get_data_preprocessor(self):
        return _PaddleOCRVisualTokenInputProcessor()

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {
            "input_names": list(PADDLEOCR_VISUAL_INPUT_NAMES),
            "output_names": ["image_embeds"],
        }

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._state == LLMModelState.WRAP:
            with module_graph_forward_guard(self._inference_model):
                return self._inference_model(*args, **kwargs)
        return self._inference_model(*args, **kwargs)

    def _output_hidden_size(self) -> int:
        for candidate in (getattr(self, "_wrap_model", None), getattr(self, "_inference_model", None)):
            if hasattr(candidate, "output_hidden_size"):
                return int(candidate.output_hidden_size)
        native_source = getattr(self, "_native_visual_source", None)
        if native_source is not None and hasattr(native_source, "config"):
            return int(native_source.config.text_config.hidden_size)
        native_root = getattr(self, "_native_model_root", None)
        if native_root is not None and hasattr(native_root, "config"):
            return int(native_root.config.text_config.hidden_size)
        if self.hf_model_dir is not None and (Path(self.hf_model_dir) / "config.json").is_file():
            return int(load_paddleocr_config(self.hf_model_dir).text_config.hidden_size)
        return 1024

    def _metadata_for_profile(
        self,
        output_dir: str,
        profile: dict[str, Any] | None = None,
    ) -> VisualModelMeta:
        del output_dir
        if profile is None:
            profile = {"image_token_capacity": int(self.config.image_token_capacity), "name": "default"}
        image_token_capacity = int(profile.get("image_token_capacity", self.config.image_token_capacity))
        patch_capacity = patch_token_capacity(image_token_capacity, self.config.spatial_merge_size)
        spatial_merge_size = int(self.config.spatial_merge_size)
        meta_info = self.get_export_metadata_cls()()
        meta_info = cast(VisualModelMeta, meta_info)
        meta_info.profile_name = str(profile.get("name", "default"))
        meta_info.patch_size = int(self.config.patch_size)
        meta_info.spatial_merge_size = spatial_merge_size
        meta_info.visual_input_mode = VISUAL_INPUT_PATCHES
        meta_info.routing_policy = "smallest_fit"
        meta_info.overflow_policy = "reject"
        meta_info.image_token_capacity = image_token_capacity
        meta_info.patch_token_capacity = patch_capacity
        meta_info.image_token_gears = list(self.config.image_token_gears)
        meta_info.input_names = list(PADDLEOCR_VISUAL_INPUT_NAMES)
        meta_info.position_format = VISUAL_POSITION_FORMAT
        meta_info.rotary_position_format = VISUAL_ROTARY_POSITION_FORMAT
        meta_info.attention_mask_format = VISUAL_ATTENTION_MASK_FORMAT
        meta_info.attention_mask_shape = VISUAL_ATTENTION_MASK_SHAPE
        meta_info.attention_mask_operator = VISUAL_ATTENTION_MASK_OPERATOR
        meta_info.num_positions = int(
            getattr(
                self,
                "_visual_num_positions",
                getattr(getattr(self, "_wrap_model", None), "num_positions", 4096),
            )
        )
        meta_info.rotary_cache_length = int(
            getattr(getattr(self, "_wrap_model", None), "rotary_cache_length", self.config.rotary_cache_length)
        )
        meta_info.input_shape = [1, patch_capacity, 3, int(self.config.patch_size), int(self.config.patch_size)]
        meta_info.output_shape = [image_token_capacity, self._output_hidden_size()]
        return meta_info

    def create_export_metadata(self, output_dir: str) -> VisualModelMeta:
        return self._metadata_for_profile(output_dir)

    def _export_single_profile_hmonnx(self, output_dir: str, profile: dict[str, Any]) -> VisualModelMeta:
        original_capacity = self.config.image_token_capacity
        original_work_dir = self.config.work_dir
        try:
            self.config.image_token_capacity = int(profile["image_token_capacity"])
            self.config.work_dir = output_dir
            hmonnx = str(super()._export_hmonnx(output_dir))
            meta_info = self._metadata_for_profile(output_dir, profile)
            meta_info.hmonnx = hmonnx
            return meta_info
        finally:
            self.config.image_token_capacity = original_capacity
            self.config.work_dir = original_work_dir

    def _profile_manifest(self, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        return build_visual_gear_manifest(
            artifacts,
            spatial_merge_size=self.config.spatial_merge_size,
            rotary_cache_length=int(
                getattr(getattr(self, "_wrap_model", None), "rotary_cache_length", self.config.rotary_cache_length)
            ),
        )

    def _reset_compiled_state_for_profile(self) -> None:
        self._wrap_model = None
        self._frontend_model = None
        self._quanted_model = None
        self._exported_model = None
        self._inference_model = None
        self.hf_compatible_model = None
        self._state = LLMModelState.NONE

    def _profile_matches_active_config(self, profile: dict[str, Any]) -> bool:
        return int(profile["image_token_capacity"]) == int(self.config.image_token_capacity)

    def export_hmonnx(self, output_dir: str) -> VisualModelMeta:
        profiles = [
            {"image_token_capacity": int(gear), "name": f"m{int(gear)}"} for gear in self.config.image_token_gears
        ]
        if len(profiles) == 1:
            if self._state != LLMModelState.NONE and not self._profile_matches_active_config(profiles[0]):
                self._reset_compiled_state_for_profile()
            meta_info = self._export_single_profile_hmonnx(output_dir, profiles[0])
            return meta_info

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metadata_root = output_path.parent if output_path.name == "visual" else output_path
        artifacts = []
        for profile in profiles:
            image_capacity = int(profile["image_token_capacity"])
            profile_dir = output_path / f"visual_m{image_capacity}"
            if self._state != LLMModelState.NONE:
                self._reset_compiled_state_for_profile()
            profile_meta = self._export_single_profile_hmonnx(str(profile_dir), profile)
            artifacts.append(
                {
                    "name": str(profile.get("name", f"m{image_capacity}")),
                    "image_token_capacity": image_capacity,
                    "patch_token_capacity": int(profile_meta.patch_token_capacity),
                    "input_shape": list(profile_meta.input_shape),
                    "output_shape": list(profile_meta.output_shape),
                    "hmonnx": Path(profile_meta.hmonnx).relative_to(metadata_root).as_posix(),
                }
            )
        manifest_path = output_path / "visual_gears.json"
        manifest = self._profile_manifest(artifacts)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        meta_info = self._metadata_for_profile(output_dir, profiles[-1])
        meta_info.hmonnx = str(metadata_root / artifacts[-1]["hmonnx"])
        meta_info.gears = artifacts
        meta_info.gear_manifest = str(manifest_path.relative_to(metadata_root))
        meta_info.routing_policy = manifest["routing_policy"]
        meta_info.image_token_gears = [artifact["image_token_capacity"] for artifact in artifacts]
        return meta_info
