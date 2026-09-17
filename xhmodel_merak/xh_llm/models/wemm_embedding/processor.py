"""Processor variant that omits the generation prompt for embeddings."""

from __future__ import annotations

from typing import Any

import torch
from qwen_vl_utils import process_vision_info

from xhmodel_merak.xh_llm.models.qwen3_5.processing_qwen3_5 import Qwen3_5Processor
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_processor import XHQwen3_5Processor
from xhmodel_merak.xh_llm.models.qwen3_5.visual_token_gears import VISUAL_INPUT_PATCHES


def video_frame_grids(pixel_values: torch.Tensor, grid_thw: torch.Tensor, merge_size: int) -> torch.Tensor:
    """Expand each video's grid into one row per visual graph invocation."""

    if pixel_values.ndim != 2 or grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError("WeMM video inputs require flattened patches and [video, 3] grids")
    if grid_thw.numel() == 0 or bool((grid_thw <= 0).any()):
        raise ValueError("WeMM video grids must have positive temporal and spatial dimensions")
    if bool((grid_thw[:, 1:] % merge_size).any()):
        raise ValueError(f"WeMM video grids must be divisible by spatial_merge_size={merge_size}")
    expected_patches = int(grid_thw.prod(dim=1).sum().item())
    if pixel_values.shape[0] != expected_patches:
        raise ValueError(
            f"Video patches and video_grid_thw do not match: patches={pixel_values.shape[0]}, grid={expected_patches}"
        )
    frame_grids = torch.repeat_interleave(grid_thw, grid_thw[:, 0], dim=0).clone()
    frame_grids[:, 0] = 1
    return frame_grids


class XHWeMMEmbeddingProcessor(XHQwen3_5Processor):  # noqa: N801
    """Qwen3.5 processor with a controllable ``add_generation_prompt`` flag."""

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        enable_thinking: bool = False,
        add_generation_prompt: bool = False,
    ):
        text = Qwen3_5Processor.apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.config.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if video_inputs is not None:
            video_inputs, video_metadata = zip(*video_inputs, strict=True)
            video_inputs, video_metadata = list(video_inputs), list(video_metadata)
        else:
            video_metadata = None
        if self.config.visual_input_mode != VISUAL_INPUT_PATCHES:
            raise ValueError("WeMM HMONNX visual runtime requires patch-token gears")
        if self.config.max_pixels is not None:
            self.image_processor.max_pixels = int(self.config.max_pixels)

        model_inputs = self(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadata,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        if "hm_pixel_values" in model_inputs:
            model_inputs.pop("hm_pixel_values")
        return model_inputs


__all__ = ["XHWeMMEmbeddingProcessor"]
