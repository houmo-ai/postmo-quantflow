# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

import torch

from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader

from ...hmonnx.base_llm_hmonnx_model import BaseLLMHMONNXModel
from ...hmonnx.hmonnx_model import HMONNXModel
from ...llm_data_processor import BaseInputProcessorConfig
from ...utils import unfold_args
from .data import PaddleOCRLLMInputProcessor
from .generation import PaddleOCRGenerationMixin, _validate_visual_pixel_values_shape
from .hf import load_paddleocr_config, load_paddleocr_processor
from .visual_token_gears import (
    VISUAL_INPUT_PATCHES,
    build_visual_token_gear_inputs,
    pad_patch_images,
    patch_token_capacity,
    select_image_token_gear,
)


PADDLEOCR_VISUAL_INPUT_NAMES = (
    "pixel_values",
    "position_ids",
    "position_weights",
    "rotary_position_ids",
    "attention_mask",
)


def _grid_key(grid_thw) -> tuple[int, int, int]:
    grid = torch.as_tensor(grid_thw, dtype=torch.long)
    if grid.shape == (1, 3):
        grid = grid[0]
    if grid.shape != (3,):
        raise ValueError(f"PaddleOCR image_grid_thw must have shape [1, 3] or [3], got {tuple(grid.shape)}")
    return tuple(int(value) for value in grid.tolist())


def _patch_size_from_meta(visual_meta) -> int:
    patch_size = getattr(visual_meta, "patch_size", None)
    if patch_size is None:
        raise ValueError("PaddleOCR visual metadata must include patch_size")
    return int(patch_size)


class PaddleOCRVLVisualHMONNXRouter:
    def __init__(self, visual_meta, *, device_map, enable_golden: bool = False):
        self.visual_meta = visual_meta
        self.patch_size = _patch_size_from_meta(visual_meta)
        self._enable_golden = False
        gears = list(getattr(visual_meta, "gears", None) or [])
        if not gears:
            gears = [visual_meta]
        has_token_gears = all(hasattr(gear, "image_token_capacity") for gear in gears)
        self.token_gear_mode = has_token_gears and getattr(
            visual_meta, "visual_input_mode", VISUAL_INPUT_PATCHES
        ) == VISUAL_INPUT_PATCHES
        self.models: dict[tuple[int, int, int] | int, HMONNXModel] = {}
        if self.token_gear_mode:
            if not HMONNXModel._is_inference_v2_env_enabled():
                raise RuntimeError(
                    "multi-gear PaddleOCR visual runtime requires ENABLE_HMINFERENCE_V2=1 "
                    "so the gear graphs can share GPU weights"
                )
            self.gears = tuple(sorted(int(gear.image_token_capacity) for gear in gears))
            if len(set(self.gears)) != len(gears):
                raise ValueError("duplicate PaddleOCR visual token gear in metadata")
            graph_files = {f"m{int(gear.image_token_capacity)}": str(gear.hmonnx) for gear in gears}
            self._loader = MultiHMONNXLoader(graph_files)
            for gear in gears:
                image_capacity = int(gear.image_token_capacity)
                expected_patch_capacity = patch_token_capacity(image_capacity, int(visual_meta.spatial_merge_size))
                if int(gear.patch_token_capacity) != expected_patch_capacity:
                    raise ValueError(
                        f"PaddleOCR visual gear {image_capacity} declares patch capacity "
                        f"{gear.patch_token_capacity}, expected {expected_patch_capacity}"
                    )
                self.models[image_capacity] = HMONNXModel(
                    gear.hmonnx,
                    onnx_graph=self._loader.graphs[f"m{image_capacity}"],
                    enable_golden=enable_golden,
                    device_map=device_map,
                )
        else:
            self.gears = ()
            for gear in gears:
                grid = _grid_key(getattr(gear, "image_grid_thw", None))
                if grid in self.models:
                    raise ValueError(f"duplicate PaddleOCR visual grid profile in metadata: {list(grid)}")
                self.models[grid] = HMONNXModel(
                    gear.hmonnx,
                    enable_golden=enable_golden,
                    device_map=device_map,
                )
        self._device = next(iter(self.models.values())).device
        if enable_golden:
            self.enable_golden = True

    def _set_device(self, device):
        self._device = device
        for model in self.models.values():
            model.to(device)
        return self

    def to(self, device):
        return self._set_device(device)

    @property
    def device(self):
        return self._device

    @property
    def shared_weight_summary(self):
        loader = getattr(self, "_loader", None)
        return loader.initializer_pool.summary() if loader is not None else None

    @property
    def enable_golden(self) -> bool:
        return self._enable_golden

    @enable_golden.setter
    def enable_golden(self, enable: bool) -> None:
        for model in self.models.values():
            model.enable_golden = enable
        self._enable_golden = enable

    def update_step(self) -> None:
        for model in self.models.values():
            model.update_step()

    @staticmethod
    def _session_input_dtype(model: HMONNXModel, name: str, fallback: torch.dtype) -> torch.dtype:
        session = getattr(model, "hmonnx_session", None)
        get_input = getattr(session, "get_input", None)
        if callable(get_input):
            return get_input(name).dtype
        return fallback

    def _coerce_inputs(self, model: HMONNXModel, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        inputs = []
        for name in PADDLEOCR_VISUAL_INPUT_NAMES:
            fallback = torch.float16 if name in {"pixel_values", "position_weights", "attention_mask"} else torch.long
            dtype = self._session_input_dtype(model, name, fallback)
            inputs.append(values[name].to(device=model.device, dtype=dtype))
        return tuple(inputs)

    def _forward_token_gear(self, pixel_values, image_grid_thw):
        grid = torch.as_tensor(image_grid_thw, dtype=torch.long).reshape(1, 3)
        grid_tuple = _grid_key(grid)
        _validate_visual_pixel_values_shape(pixel_values, grid_tuple, self.patch_size)
        merge_size = int(self.visual_meta.spatial_merge_size)
        valid_patch_tokens = int(grid.prod().item())
        valid_image_tokens = valid_patch_tokens // (merge_size * merge_size)
        gear = select_image_token_gear(valid_image_tokens, self.gears)
        model = self.models[gear]
        capacity = patch_token_capacity(gear, merge_size)
        padded_pixels, tensor_patch_tokens = pad_patch_images(
            pixel_values,
            capacity,
            grid,
            spatial_merge_size=merge_size,
        )
        if tensor_patch_tokens != valid_patch_tokens:
            raise ValueError(
                f"pixel_values has {tensor_patch_tokens} patches but image_grid_thw declares {valid_patch_tokens}"
            )
        position_dtype = self._session_input_dtype(model, "position_weights", torch.float16)
        geometry = build_visual_token_gear_inputs(
            grid,
            patch_capacity=capacity,
            num_positions=int(getattr(self.visual_meta, "num_positions", 4096)),
            spatial_merge_size=merge_size,
            dtype=position_dtype,
            rotary_cache_length=int(getattr(self.visual_meta, "rotary_cache_length", gear * merge_size)),
        )
        values = {"pixel_values": padded_pixels}
        values.update({name: geometry[name] for name in PADDLEOCR_VISUAL_INPUT_NAMES if name != "pixel_values"})
        output = model.forward(*self._coerce_inputs(model, values))
        if output.ndim == 3:
            return output[:, :valid_image_tokens]
        if output.ndim == 2:
            return output[:valid_image_tokens]
        raise ValueError(f"unexpected PaddleOCR visual output shape: {tuple(output.shape)}")

    def forward(self, pixel_values, image_grid_thw):
        if self.token_gear_mode:
            return self._forward_token_gear(pixel_values, image_grid_thw)
        grid = _grid_key(image_grid_thw)
        model = self.models.get(grid)
        if model is None:
            supported = ", ".join(str(list(candidate)) for candidate in sorted(self.models))
            raise ValueError(f"Unsupported PaddleOCR image_grid_thw {list(grid)}; exported grids: {supported}")
        _validate_visual_pixel_values_shape(pixel_values, grid, self.patch_size)
        return model.forward(pixel_values)


class XHPaddleOCRVLHMONNXModel(PaddleOCRGenerationMixin, BaseLLMHMONNXModel):
    def __init__(self, meta, **kwargs):
        super().__init__(meta, **kwargs)
        self.paddleocr_hf_config = load_paddleocr_config(self.hf_model_dir)
        self.visual = PaddleOCRVLVisualHMONNXRouter(
            meta.visual_config,
            enable_golden=kwargs.get("enable_golden", False),
            device_map=self._valid_devices,
        )
        self.enable_golden = bool(kwargs.get("enable_golden", False))

    def _set_dtype(self, dtype):
        if dtype != torch.float16:
            raise ValueError("PaddleOCR HMONNX requires the float16 KV-cache ABI")
        self._dtype = dtype
        self.embed_tokens.to(dtype=dtype)
        for model in self._models.values():
            model.to(dtype=dtype)
        return self

    def _set_device(self, device):
        super()._set_device(device)
        if hasattr(self, "visual"):
            self.visual.to(device)
        return self

    def _set_enable_golden(self, enable: bool) -> None:
        self.visual.enable_golden = enable

    def get_tf_processor(self):
        return load_paddleocr_processor(self.hf_model_dir)

    def _get_data_preprocessor(self):
        return PaddleOCRLLMInputProcessor(
            BaseInputProcessorConfig(
                self.get_input_embeddings(),
                self.get_input_sequence_length(),
                self.past_key_caches,
                self.past_value_caches,
                pad_token_id=self.pad_token_id,
            )
        )

    def forward(self, *args):
        return super().forward(*[arg.to(torch.int32) if arg.dtype == torch.int64 else arg for arg in unfold_args(args)])
