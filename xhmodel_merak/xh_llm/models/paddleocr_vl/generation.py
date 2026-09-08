# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Single-image greedy generation with chunked M-RoPE prefill."""

import torch
from transformers.models.paddleocr_vl.modeling_paddleocr_vl import PaddleOCRVLModel

from .visual_token_gears import (
    VISUAL_INPUT_PATCHES,
    build_visual_token_gear_inputs,
    pad_patch_images,
    patch_token_capacity,
    select_image_token_gear,
)


class PaddleOCRPositionIndexer:
    get_rope_index = PaddleOCRVLModel.get_rope_index
    get_vision_position_ids = PaddleOCRVLModel.get_vision_position_ids

    def __init__(self, config):
        self.config = config


def _normalize_grid(image_grid_thw) -> list[int]:
    if image_grid_thw is None:
        raise ValueError("image_grid_thw is required when pixel_values are provided")
    grid = torch.as_tensor(image_grid_thw, dtype=torch.long)
    if grid.shape == (1, 3):
        grid = grid[0]
    if grid.shape != (3,):
        raise ValueError(f"image_grid_thw must have shape [1, 3] or [3], got {tuple(grid.shape)}")
    return [int(value) for value in grid.tolist()]


def _exported_visual_grids(visual_meta) -> set[tuple[int, int, int]]:
    gears = getattr(visual_meta, "gears", None)
    if gears:
        return {tuple(_normalize_grid(gear.image_grid_thw)) for gear in gears}
    image_grid_thw = getattr(visual_meta, "image_grid_thw", None)
    if image_grid_thw is not None:
        return {tuple(_normalize_grid(image_grid_thw))}
    patch_size = int(visual_meta.patch_size)
    profiles = getattr(visual_meta, "resolution_profiles", None)
    if profiles:
        return {(1, int(profile["height"]) // patch_size, int(profile["width"]) // patch_size) for profile in profiles}
    return {(1, int(visual_meta.max_size_h) // patch_size, int(visual_meta.max_size_w) // patch_size)}


def _grid_from_compiled_visual(candidate) -> tuple[int, int, int] | None:
    if candidate is None:
        return None
    grid_thw = getattr(candidate, "grid_thw", None)
    if grid_thw is not None:
        return tuple(_normalize_grid(grid_thw))
    for child_name in ("embeddings", "projector"):
        child = getattr(candidate, child_name, None)
        grid_thw = getattr(child, "grid_thw", None)
        if grid_thw is not None:
            return tuple(_normalize_grid(grid_thw))
    return None


def _live_visual_grids(visual, visual_meta) -> set[tuple[int, int, int]]:
    for attr_name in ("_wrap_model", "_inference_model"):
        grid = _grid_from_compiled_visual(getattr(visual, attr_name, None))
        if grid is not None:
            return {grid}
    visual_config = getattr(visual, "config", visual_meta)
    patch_size = int(getattr(visual_config, "patch_size", visual_meta.patch_size))
    return {
        (
            1,
            int(getattr(visual_config, "max_size_h", visual_meta.max_size_h)) // patch_size,
            int(getattr(visual_config, "max_size_w", visual_meta.max_size_w)) // patch_size,
        )
    }


def _supported_visual_grids(visual, visual_meta) -> set[tuple[int, int, int]]:
    router_models = getattr(visual, "models", None)
    if router_models is not None:
        return set(router_models)
    if hasattr(visual, "visual_meta"):
        return _exported_visual_grids(visual_meta)
    return _live_visual_grids(visual, visual_meta)


def _validate_token_gear_grid(grid: tuple[int, int, int], visual_meta) -> None:
    merge_size = int(getattr(visual_meta, "spatial_merge_size", 2))
    valid_patch_tokens = int(grid[0]) * int(grid[1]) * int(grid[2])
    merge_unit = merge_size * merge_size
    if valid_patch_tokens <= 0 or valid_patch_tokens % merge_unit:
        raise ValueError(f"PaddleOCR image_grid_thw {list(grid)} does not align to spatial_merge_size={merge_size}")
    gears = getattr(visual_meta, "image_token_gears", None)
    if not gears:
        gears = [int(gear.image_token_capacity) for gear in getattr(visual_meta, "gears", ())]
    select_image_token_gear(valid_patch_tokens // merge_unit, gears)


def _visual_patch_size(visual, visual_meta) -> int:
    visual_config = getattr(visual, "config", None)
    if visual_config is not None and hasattr(visual_config, "patch_size"):
        return int(visual_config.patch_size)
    return int(visual_meta.patch_size)


def _validate_visual_pixel_values_shape(pixel_values, grid: tuple[int, int, int], patch_size: int) -> None:
    expected = (1, int(grid[0]) * int(grid[1]) * int(grid[2]), 3, patch_size, patch_size)
    if tuple(pixel_values.shape) != expected:
        raise ValueError(
            "PaddleOCR pixel_values shape must match image_grid_thw exactly: "
            f"expected {expected} for grid {list(grid)}, got {tuple(pixel_values.shape)}"
        )


class PaddleOCRGenerationMixin:
    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        *,
        pixel_values=None,
        image_grid_thw=None,
        attention_mask=None,
        mm_token_type_ids=None,
        max_new_tokens=1024,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=None,
        streamer=None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported PaddleOCR generation arguments: {sorted(kwargs)}")
        if do_sample:
            raise ValueError("PaddleOCR deployment currently supports greedy generation only")
        if self.dtype != torch.float16:
            raise ValueError("PaddleOCR chunked generation requires the float16 KV-cache ABI")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("Expected one non-empty OCR prompt")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError("Batched/padded OCR prompts are not supported")
        deployment_config = self.meta_info.model_config if hasattr(self, "meta_info") else self.config
        prompt_length = input_ids.shape[1]
        if max_new_tokens <= 0 or prompt_length + max_new_tokens > deployment_config.context_max_length:
            raise ValueError("prompt_tokens + max_new_tokens must fit context_max_length")
        input_ids = input_ids.to(self.device)
        self.prepare_for_inference()
        hf_config = self.paddleocr_hf_config
        embedding = self.get_input_embeddings().to(self.device)
        embeddings = embedding(input_ids).to(self.dtype)
        image_mask = input_ids.eq(hf_config.image_token_id)
        if pixel_values is not None:
            runtime_meta = getattr(self, "meta_info", None)
            visual_meta = getattr(runtime_meta, "visual_config", deployment_config.visual_config)
            actual_grid = tuple(_normalize_grid(image_grid_thw))
            if getattr(visual_meta, "visual_input_mode", None) == VISUAL_INPUT_PATCHES:
                _validate_token_gear_grid(actual_grid, visual_meta)
            else:
                supported_grids = _supported_visual_grids(self.visual, visual_meta)
                if actual_grid not in supported_grids:
                    supported = ", ".join(str(list(grid)) for grid in sorted(supported_grids))
                    raise ValueError(
                        f"Unsupported PaddleOCR image_grid_thw {list(actual_grid)}; exported grids: {supported}"
                    )
            if pixel_values.ndim == 4:
                pixel_values = pixel_values.unsqueeze(0)
            _validate_visual_pixel_values_shape(pixel_values, actual_grid, _visual_patch_size(self.visual, visual_meta))
            visual_pixels = pixel_values.to(device=self.device, dtype=self.dtype)
            if hasattr(self.visual, "visual_meta"):
                image_embeddings = self.visual.forward(visual_pixels, image_grid_thw)
            elif getattr(visual_meta, "visual_input_mode", None) == VISUAL_INPUT_PATCHES:
                visual_config = self.visual.config
                capacity = int(visual_config.image_token_capacity)
                merge_size = int(visual_config.spatial_merge_size)
                patch_capacity = patch_token_capacity(capacity, merge_size)
                visual_pixels, _ = pad_patch_images(
                    visual_pixels,
                    patch_capacity,
                    image_grid_thw,
                    spatial_merge_size=merge_size,
                )
                wrap_model = getattr(self.visual, "_wrap_model", None)
                geometry = build_visual_token_gear_inputs(
                    image_grid_thw,
                    patch_capacity=patch_capacity,
                    num_positions=int(getattr(wrap_model, "num_positions", 4096)),
                    spatial_merge_size=merge_size,
                    dtype=self.dtype,
                    device=self.device,
                    rotary_cache_length=int(
                        getattr(wrap_model, "rotary_cache_length", visual_config.rotary_cache_length)
                    ),
                )
                image_embeddings = self.visual.forward(
                    visual_pixels,
                    geometry["position_ids"],
                    geometry["position_weights"],
                    geometry["rotary_position_ids"],
                    geometry["attention_mask"],
                )[: geometry["valid_image_tokens"]]
            else:
                image_embeddings = self.visual.forward(visual_pixels)
            if isinstance(image_embeddings, (tuple, list)):
                image_embeddings = image_embeddings[0]
            image_embeddings = image_embeddings.reshape(-1, embeddings.shape[-1])
            if int(image_mask.sum()) != image_embeddings.shape[0]:
                raise ValueError("Image placeholder count does not match visual output tokens")
            embeddings[image_mask] = image_embeddings.to(embeddings)
        elif bool(image_mask.any()):
            raise ValueError("Image tokens require pixel_values")
        if mm_token_type_ids is None:
            mm_token_type_ids = image_mask.to(torch.int64)
        positions, delta = PaddleOCRPositionIndexer(hf_config).get_rope_index(
            input_ids,
            mm_token_type_ids.to(self.device),
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        eos = eos_token_id if eos_token_id is not None else self.get_tokenizer().eos_token_id
        eos_ids = {int(value) for value in (eos if isinstance(eos, (list, tuple)) else [eos]) if value is not None}
        generated = []
        with self.get_kvcache_mixin().kv_cache_scope(device=self.device):
            self.set_prefill()
            chunk_length = deployment_config.prefill_chunk_length
            processor = self.get_data_preprocessor()
            processor.past_key_caches = self.past_key_caches
            processor.past_value_caches = self.past_value_caches
            processor.input_sequence_length = chunk_length
            for start in range(0, prompt_length, chunk_length):
                end = min(start + chunk_length, prompt_length)
                inputs = processor(
                    {
                        "inputs_embeds": embeddings[:, start:end],
                        "position_ids": positions[:, :, start:end],
                        "past_seq_length": start,
                    }
                )
                logits = self.forward(*inputs)
            next_token = logits[:, -1].argmax(-1).reshape(1, 1)
            self.set_decode()
            processor.input_sequence_length = 1
            for step in range(max_new_tokens):
                generated.append(next_token)
                if streamer is not None:
                    streamer.put(next_token.cpu())
                if int(next_token.item()) in eos_ids or step + 1 == max_new_tokens:
                    break
                past_length = prompt_length + step
                decode_positions = (delta + past_length).expand(3, 1)
                inputs = processor(
                    {"input_ids": next_token, "position_ids": decode_positions, "past_seq_length": past_length}
                )
                logits = self.forward(*inputs)
                next_token = logits[:, -1].argmax(-1).reshape(1, 1)
        if streamer is not None:
            streamer.end()
        return torch.cat([input_ids, *generated], dim=1)
