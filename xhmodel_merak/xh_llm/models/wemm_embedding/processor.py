"""Processor variant that omits the generation prompt for embeddings."""

from __future__ import annotations

from typing import Any

import torch
from qwen_vl_utils import process_vision_info

from xhmodel_merak.xh_llm.models.qwen3_5.processing_qwen3_5 import Qwen3_5Processor
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_processor import XHQwen3_5Processor
from xhmodel_merak.xh_llm.models.qwen3_5.visual_token_gears import VISUAL_INPUT_PATCHES


EMBEDDING_TOKEN = "<embedding>"


def normalize_embedding_pooling_anchor(model_inputs: Any, tokenizer: Any) -> None:
    """Ensure one WeMM pooling anchor at the last valid position.

    WeMM's official template provides ``<embedding>``, while minimal or custom
    templates may omit it and Transformers 5.16 may retain a turn terminator
    after it.  Normalize token ids after rendering so both forms have the same
    pooling contract without introducing an extra whitespace token.
    """

    input_ids = model_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("WeMM processor must return rank-2 input_ids")
    anchor_ids = tokenizer.encode(EMBEDDING_TOKEN, add_special_tokens=False)
    if len(anchor_ids) != 1:
        raise ValueError(f"Expected {EMBEDDING_TOKEN!r} to encode as one token, got {anchor_ids!r}")
    anchor_id = int(anchor_ids[0])
    attention_mask = model_inputs.get("attention_mask")
    normalized_rows: list[torch.Tensor] = []
    for row_index, row in enumerate(input_ids):
        if isinstance(attention_mask, torch.Tensor):
            valid_positions = torch.nonzero(attention_mask[row_index], as_tuple=False).flatten()
        else:
            valid_positions = torch.arange(row.numel(), device=row.device)
        if valid_positions.numel() == 0:
            raise ValueError(f"WeMM processor returned an empty sequence at row {row_index}")
        valid_ids = row[valid_positions]
        anchor_mask = valid_ids == anchor_id
        normalized_rows.append(
            torch.cat(
                (
                    valid_ids[~anchor_mask],
                    torch.tensor([anchor_id], dtype=row.dtype, device=row.device),
                )
            )
        )

    target_width = max(input_ids.shape[1], *(row.numel() for row in normalized_rows))
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    normalized_input_ids = input_ids.new_full(input_ids.shape[:-1] + (target_width,), int(pad_token_id))
    normalized_attention_mask = None
    if isinstance(attention_mask, torch.Tensor):
        normalized_attention_mask = attention_mask.new_zeros(attention_mask.shape[:-1] + (target_width,))
    padding_side = getattr(tokenizer, "padding_side", "right")
    if padding_side not in {"left", "right"}:
        raise ValueError(f"Unsupported tokenizer padding_side={padding_side!r}")
    for row_index, normalized in enumerate(normalized_rows):
        start = target_width - normalized.numel() if padding_side == "left" else 0
        end = start + normalized.numel()
        normalized_input_ids[row_index, start:end] = normalized
        if normalized_attention_mask is not None:
            normalized_attention_mask[row_index, start:end] = 1
    model_inputs["input_ids"] = normalized_input_ids
    if normalized_attention_mask is not None:
        model_inputs["attention_mask"] = normalized_attention_mask


def validate_embedding_pooling_anchor(model_inputs: Any, tokenizer: Any) -> None:
    """Require every processed sequence to end at WeMM's pooling anchor."""

    input_ids = model_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("WeMM processor must return rank-2 input_ids")
    anchor_ids = tokenizer.encode(EMBEDDING_TOKEN, add_special_tokens=False)
    if len(anchor_ids) != 1:
        raise ValueError(f"Expected {EMBEDDING_TOKEN!r} to encode as one token, got {anchor_ids!r}")
    attention_mask = model_inputs.get("attention_mask")
    for row_index, row in enumerate(input_ids):
        if isinstance(attention_mask, torch.Tensor):
            valid_positions = torch.nonzero(attention_mask[row_index], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                raise ValueError(f"WeMM processor returned an empty sequence at row {row_index}")
            valid_ids = row[valid_positions]
            last_index = int(valid_positions[-1].item())
        else:
            if row.numel() == 0:
                raise ValueError(f"WeMM processor returned an empty sequence at row {row_index}")
            valid_ids = row
            last_index = int(row.numel() - 1)
        actual = int(row[last_index].item())
        if actual != anchor_ids[0]:
            raise ValueError(
                "WeMM embedding input must end with the <embedding> pooling "
                f"token (expected id {anchor_ids[0]}, got {actual})"
            )
        anchor_count = int((valid_ids == anchor_ids[0]).sum().item())
        if anchor_count != 1:
            raise ValueError(
                "WeMM embedding input must contain exactly one <embedding> "
                f"pooling token, got {anchor_count} at row {row_index}"
            )


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
        normalize_embedding_pooling_anchor(model_inputs, self.tokenizer)
        validate_embedding_pooling_anchor(model_inputs, self.tokenizer)
        return model_inputs


__all__ = [
    "EMBEDDING_TOKEN",
    "XHWeMMEmbeddingProcessor",
    "normalize_embedding_pooling_anchor",
    "validate_embedding_pooling_anchor",
]
