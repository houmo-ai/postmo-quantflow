# Copyright 2025 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
# Modifications by HOUMO AI for Merak static visual graph export.
# Derived from Hugging Face Transformers PaddleOCR-VL implementation:
# https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/paddleocr_vl/modeling_paddleocr_vl.py
# The upstream implementation notes that it is based on EleutherAI GPT-NeoX and
# Transformers GPT-NeoX/OPT implementations, adapted for PaddleOCR-VL.
# See licenses/transformers/LICENSE for the upstream Apache-2.0 license text.

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from xhquant import nn as xhnn

from .visual_token_gears import build_visual_token_gear_inputs, patch_token_capacity


def _to_grid_tuple(grid_thw: torch.Tensor | list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    grid = torch.as_tensor(grid_thw, dtype=torch.long)
    if grid.shape == (1, 3):
        grid = grid[0]
    if grid.shape != (3,):
        raise ValueError(f"PaddleOCR-VL visual grid must have shape (3,) or (1, 3), got {tuple(grid.shape)}")
    return int(grid[0].item()), int(grid[1].item()), int(grid[2].item())


def _validate_grid(grid_thw: torch.Tensor, spatial_merge_size: int) -> tuple[int, int, int]:
    grid_t, grid_h, grid_w = _to_grid_tuple(grid_thw)
    if grid_t != 1:
        raise ValueError(f"PaddleOCR-VL static visual wrapper supports one image frame, got grid_t={grid_t}")
    if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
        raise ValueError("PaddleOCR-VL visual grid must be divisible by spatial_merge_size")
    return grid_t, grid_h, grid_w


class PaddleOCRVLStaticEmbeddings(nn.Module):
    def __init__(self, native_embeddings: nn.Module, patch_token_capacity: int):
        super().__init__()
        self.patch_embedding = native_embeddings.patch_embedding
        self.position_embedding = native_embeddings.position_embedding
        self.embed_dim = int(native_embeddings.embed_dim)
        self.patch_size = int(native_embeddings.patch_size)
        self.num_channels = int(native_embeddings.patch_embedding.in_channels)
        self.patch_token_capacity = int(patch_token_capacity)
        self.num_positions = int(native_embeddings.position_embedding.weight.shape[0])

    def forward(self, pixel_values: Tensor, position_ids: Tensor, position_weights: Tensor) -> Tensor:
        patches = pixel_values.reshape(
            self.patch_token_capacity,
            self.num_channels,
            self.patch_size,
            self.patch_size,
        )
        embeddings = self.patch_embedding(patches).flatten(-2).squeeze(-1)
        embeddings = embeddings.reshape(1, self.patch_token_capacity, self.embed_dim)
        position_values = self.position_embedding(position_ids)
        position = (position_values * position_weights.unsqueeze(-1)).sum(0).unsqueeze(0)
        return embeddings + position


class PaddleOCRVLStaticAttention(nn.Module):
    def __init__(self, native_attention: nn.Module, patch_token_capacity: int):
        super().__init__()
        self.q_proj = native_attention.q_proj
        self.k_proj = native_attention.k_proj
        self.v_proj = native_attention.v_proj
        self.out_proj = native_attention.out_proj
        self.num_heads = int(native_attention.num_heads)
        self.head_dim = int(native_attention.head_dim)
        self.scaling = float(native_attention.scaling)
        self.patch_token_capacity = int(patch_token_capacity)
        self.rope = xhnn.Rope()
        self.masked_add = xhnn.MaskedAdd()
        self.softmax = xhnn.Softmax(dim=-1)

    def forward(
        self, hidden_states: Tensor, position_embeddings: tuple[Tensor, Tensor], attention_mask: Tensor
    ) -> Tensor:
        query = self.q_proj(hidden_states).view(1, self.patch_token_capacity, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(1, self.patch_token_capacity, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(1, self.patch_token_capacity, self.num_heads, self.head_dim)
        cos, sin = position_embeddings
        cos = cos.unsqueeze(0).unsqueeze(-2)
        sin = sin.unsqueeze(0).unsqueeze(-2)
        query = self.rope(query, cos, sin)
        key = self.rope(key, cos, sin)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        weights = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        weights = self.masked_add(weights.to(query.dtype), attention_mask)
        weights = self.softmax(weights)
        output = torch.matmul(weights, value)
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(1, self.patch_token_capacity, self.num_heads * self.head_dim).contiguous()
        return self.out_proj(output)


class PaddleOCRVLStaticEncoderLayer(nn.Module):
    def __init__(self, native_layer: nn.Module, patch_token_capacity: int):
        super().__init__()
        self.layer_norm1 = native_layer.layer_norm1
        self.self_attn = PaddleOCRVLStaticAttention(native_layer.self_attn, patch_token_capacity)
        self.layer_norm2 = native_layer.layer_norm2
        self.mlp = native_layer.mlp

    def forward(
        self, hidden_states: Tensor, position_embeddings: tuple[Tensor, Tensor], attention_mask: Tensor
    ) -> Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states), position_embeddings, attention_mask
        )
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class PaddleOCRVLStaticVisionProjector(nn.Module):
    def __init__(
        self, native_projector: nn.Module, patch_token_capacity: int, grid_thw: torch.Tensor, merge_major_input: bool
    ):
        super().__init__()
        self.pre_norm = native_projector.pre_norm
        self.linear_1 = native_projector.linear_1
        self.act = native_projector.act
        self.linear_2 = native_projector.linear_2
        self.merge_kernel_size = tuple(int(v) for v in native_projector.merge_kernel_size)
        self.hidden_size = int(native_projector.pre_norm.normalized_shape[0])
        self.patch_token_capacity = int(patch_token_capacity)
        self.merge_major_input = bool(merge_major_input)
        self.register_buffer("grid_thw", torch.as_tensor(grid_thw, dtype=torch.long).reshape(1, 3), persistent=False)

    def forward(self, image_features: Tensor) -> Tensor:
        merge_h, merge_w = self.merge_kernel_size
        hidden_size = self.hidden_size
        image_features = self.pre_norm(image_features)
        if self.merge_major_input:
            image_features = image_features.reshape(
                1,
                self.patch_token_capacity // (merge_h * merge_w),
                merge_h * merge_w,
                hidden_size,
            )
        else:
            _, grid_h, grid_w = _validate_grid(self.grid_thw, merge_h)
            image_features = image_features.reshape(
                1,
                grid_h // merge_h,
                merge_h,
                grid_w // merge_w,
                merge_w,
                hidden_size,
            )
            image_features = image_features.transpose(2, 3)
        image_features = image_features.reshape(
            self.patch_token_capacity // (merge_h * merge_w),
            merge_h * merge_w * hidden_size,
        )
        return self.linear_2(self.act(self.linear_1(image_features)))


class PaddleOCRVLStaticImageEmbeddingModel(nn.Module):
    def __init__(
        self,
        native_model: nn.Module,
        grid_thw: torch.Tensor,
        image_token_capacity: int | None = None,
        *,
        merge_major_input: bool = False,
    ):
        super().__init__()
        grid_thw = torch.as_tensor(grid_thw, dtype=torch.long).reshape(1, 3)
        _, grid_h, grid_w = _validate_grid(grid_thw, native_model.config.vision_config.spatial_merge_size)
        self.visual = native_model.visual
        self.spatial_merge_size = int(native_model.config.vision_config.spatial_merge_size)
        if image_token_capacity is None:
            image_token_capacity = grid_h * grid_w // (self.spatial_merge_size * self.spatial_merge_size)
        self.image_token_capacity = int(image_token_capacity)
        self.patch_token_capacity = patch_token_capacity(self.image_token_capacity, self.spatial_merge_size)
        if grid_h * grid_w > self.patch_token_capacity:
            raise ValueError(
                f"PaddleOCR grid {[1, grid_h, grid_w]} needs {grid_h * grid_w} patches, "
                f"which exceeds gear capacity {self.patch_token_capacity}"
            )
        self.merge_major_input = bool(merge_major_input)
        self.projector = PaddleOCRVLStaticVisionProjector(
            native_model.projector,
            self.patch_token_capacity,
            grid_thw,
            merge_major_input=self.merge_major_input,
        )
        vision_model = native_model.visual.vision_model
        self.embeddings = PaddleOCRVLStaticEmbeddings(vision_model.embeddings, self.patch_token_capacity)
        self.layers = nn.ModuleList(
            PaddleOCRVLStaticEncoderLayer(layer, self.patch_token_capacity)
            for layer in vision_model.encoder.layers
        )
        self.post_layernorm = vision_model.post_layernorm
        head_dim = int(vision_model.encoder.config.hidden_size // vision_model.encoder.config.num_attention_heads)
        rotary_pos_emb = getattr(native_model, "rotary_pos_emb", vision_model.encoder.rotary_pos_emb)
        rotary_cache_length = max(grid_h, grid_w, self.image_token_capacity * self.spatial_merge_size)
        rotary_position_ids = torch.arange(rotary_cache_length, device=rotary_pos_emb.inv_freq.device)
        rotary = rotary_pos_emb(rotary_position_ids)
        self.register_buffer("rotary_cos_cached", rotary.cos(), persistent=False)
        self.register_buffer("rotary_sin_cached", rotary.sin(), persistent=False)
        self.rotary_gather = xhnn.Gather(axis=0)
        self.register_buffer("grid_thw", grid_thw, persistent=False)
        geometry = build_visual_token_gear_inputs(
            grid_thw,
            patch_capacity=self.patch_token_capacity,
            num_positions=self.embeddings.num_positions,
            spatial_merge_size=self.spatial_merge_size,
            dtype=torch.float32,
            rotary_cache_length=rotary_cache_length,
            merge_major=self.merge_major_input,
        )
        self.register_buffer("_default_position_ids", geometry["position_ids"], persistent=False)
        self.register_buffer("_default_position_weights", geometry["position_weights"], persistent=False)
        self.register_buffer("_default_rotary_position_ids", geometry["rotary_position_ids"], persistent=False)
        self.register_buffer("_default_attention_mask", geometry["attention_mask"], persistent=False)
        self.output_hidden_size = int(native_model.config.text_config.hidden_size)
        self.head_dim = head_dim
        self.num_positions = self.embeddings.num_positions
        self.rotary_cache_length = rotary_cache_length

    def _position_embeddings(self, rotary_position_ids: Tensor) -> tuple[Tensor, Tensor]:
        rotary_cos = self.rotary_gather(self.rotary_cos_cached, rotary_position_ids)
        rotary_sin = self.rotary_gather(self.rotary_sin_cached, rotary_position_ids)
        cos_half = rotary_cos.permute(1, 0, 2).reshape(self.patch_token_capacity, -1)
        sin_half = rotary_sin.permute(1, 0, 2).reshape(self.patch_token_capacity, -1)
        return cos_half.repeat(1, 2), sin_half.repeat(1, 2)

    def forward(
        self,
        pixel_values: Tensor,
        position_ids: Tensor | None = None,
        position_weights: Tensor | None = None,
        rotary_position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if position_ids is None:
            position_ids = self._default_position_ids
        if position_weights is None:
            position_weights = self._default_position_weights
        if rotary_position_ids is None:
            rotary_position_ids = self._default_rotary_position_ids
        if attention_mask is None:
            attention_mask = self._default_attention_mask
        hidden_states = self.embeddings(pixel_values, position_ids, position_weights)
        position_embeddings = self._position_embeddings(rotary_position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask)
        hidden_states = self.post_layernorm(hidden_states)
        return self.projector(hidden_states)


def build_static_image_embedding_model(
    native_model: nn.Module,
    grid_thw: torch.Tensor,
    image_token_capacity: int | None = None,
    *,
    merge_major_input: bool = False,
) -> PaddleOCRVLStaticImageEmbeddingModel:
    return PaddleOCRVLStaticImageEmbeddingModel(
        native_model,
        grid_thw,
        image_token_capacity,
        merge_major_input=merge_major_input,
    )


def default_grid_from_config(config: Any) -> torch.Tensor:
    patch_size = int(config.patch_size)
    return torch.tensor(
        [[1, int(config.max_size_h) // patch_size, int(config.max_size_w) // patch_size]],
        dtype=torch.long,
    )
