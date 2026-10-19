# Copyright 2026 the HuggingFace Team. All rights reserved.
# Copyright 2026 HOUMO AI (static graph adaptation)
# SPDX-License-Identifier: Apache-2.0
# Reference: transformers 5.16.1, models/glm5_next/modeling_glm5_next.py.
# See licenses/transformers/LICENSE and THIRD_PARTY_NOTICES.
"""Frame-local vision graphs with Host-owned grid positions and padding."""

import torch
from torch import nn
from xhquant.nn import FlashAttention

from .modeling import StaticRMSNorm


def _conv_as_linear(source):
    result = nn.Linear(source.weight[0].numel(), source.out_channels, bias=source.bias is not None)
    result.weight = nn.Parameter(source.weight.flatten(1), requires_grad=False)
    result.bias = source.bias
    return result


class StaticVisionStem(nn.Module):
    def __init__(self, source):
        super().__init__()
        # HF patchify already packs [channel, temporal, patch_y, patch_x].
        self.proj = _conv_as_linear(source.proj)

    def forward(self, pixel_values):
        return self.proj(pixel_values)


class StaticVisionBlock(nn.Module):
    def __init__(self, source, num_patches):
        super().__init__()
        self.length = num_patches
        self.heads, self.dim = source.attn.num_heads, source.attn.head_dim
        self.norm1, self.norm2 = StaticRMSNorm(source.norm1), StaticRMSNorm(source.norm2)
        self.q_norm, self.k_norm = StaticRMSNorm(source.attn.q_norm), StaticRMSNorm(source.attn.k_norm)
        self.qkv, self.proj, self.mlp = source.attn.qkv, source.attn.proj, source.mlp
        self.attention = FlashAttention(
            num_heads=self.heads,
            scale=self.dim**-0.5,
            is_causal=False,
            q_bits=16,
            k_bits=16,
            v_bits=16,
            p_bits=16,
        )

    def _rotate(self, x, cos, sin):
        half = self.dim // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return (x.float() * cos.float() + rotated.float() * sin.float()).to(x.dtype)

    def forward(self, hidden_states, rotary_cos, rotary_sin, valid_length):
        qkv = self.qkv(self.norm1(hidden_states)).reshape(1, self.length, 3, self.heads, self.dim)
        q = self.q_norm(qkv[:, :, 0]).transpose(1, 2)
        k = self.k_norm(qkv[:, :, 1]).transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        q = self._rotate(q, rotary_cos, rotary_sin)
        k = self._rotate(k, rotary_cos, rotary_sin)
        # Host packing guarantees a valid prefix followed by padded keys.
        attended = self.attention(q, k, v, current_input_length=valid_length, kv_valid_length=valid_length)
        attended = attended.transpose(1, 2).reshape(1, self.length, self.heads * self.dim)
        hidden_states = hidden_states + self.proj(attended)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class StaticVisionMerger(nn.Module):
    def __init__(self, source, num_patches):
        super().__init__()
        self.norm = StaticRMSNorm(source.post_layernorm)
        self.merge = source.spatial_merge_size
        self.groups = num_patches // self.merge**2
        self.width = source.config.hidden_size
        self.downsample = _conv_as_linear(source.downsample)
        self.merger = source.merger

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        # Native Conv2d sees channels first within each consecutive merge group.
        hidden_states = hidden_states.reshape(1, self.groups, self.merge**2, self.width)
        hidden_states = hidden_states.transpose(2, 3).reshape(1, self.groups, self.width * self.merge**2)
        return self.merger(self.downsample(hidden_states))


class StaticVisionModel(nn.Module):
    """Numerical reference wrapper; export materializes each component separately."""

    def __init__(self, source, num_patches):
        super().__init__()
        self.stem = StaticVisionStem(source.patch_embed)
        self.blocks = nn.ModuleList([StaticVisionBlock(block, num_patches) for block in source.blocks])
        self.merger = StaticVisionMerger(source, num_patches)

    def forward(self, pixel_values, rotary_cos, rotary_sin, valid_length):
        hidden_states = self.stem(pixel_values)
        for block in self.blocks:
            hidden_states = block(hidden_states, rotary_cos, rotary_sin, valid_length)
        return self.merger(hidden_states)


def prepare_vision_frame(patches, height, width, config, capacity, *, device="cpu", dtype=torch.float16):
    """Keep native merge order, mask padded keys and crop after spatial merging."""
    from transformers.vision_utils import get_vision_position_ids

    merge = config.spatial_merge_size
    count = height * width
    patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
    if height <= 0 or width <= 0 or height % merge or width % merge:
        raise ValueError("vision grid dimensions must be positive multiples of spatial_merge_size")
    if count > capacity or patches.shape != (count, patch_dim):
        raise ValueError(f"vision frame exceeds {capacity} patches or has an incompatible patch layout")
    pixels = torch.zeros(1, capacity, patch_dim, device=device, dtype=dtype)
    pixels[:, :count] = patches.to(device=device, dtype=dtype)
    positions = get_vision_position_ids(torch.tensor([[1, height, width]], device=device), merge)
    rotary_dim = config.hidden_size // config.num_heads // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    freq = (positions.unsqueeze(-1) * inv_freq).flatten(1)
    freq = torch.cat((freq, freq), dim=-1)
    cos = torch.ones(1, 1, capacity, rotary_dim * 2, device=device, dtype=dtype)
    sin = torch.zeros_like(cos)
    cos[:, :, :count], sin[:, :, :count] = freq.cos(), freq.sin()
    mask = torch.full((1, 1, 1, capacity), -32768.0, device=device, dtype=dtype)
    mask[..., :count] = 0
    return pixels, cos, sin, mask


def vision_grid_frames(pixels, grid, config, capacity):
    """Validate the entire packed input before yielding independent temporal groups."""
    if pixels is None or grid is None or grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] == 0:
        raise ValueError("pixel values and a nonempty [items,3] grid must be supplied together")
    if grid.is_floating_point() or grid.dtype == torch.bool:
        raise ValueError("vision grids must contain integers")
    grids = grid.cpu().tolist()
    merge = config.spatial_merge_size
    patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
    for t, h, w in grids:
        if min(t, h, w) <= 0 or h % merge or w % merge or h * w > capacity:
            raise ValueError(f"invalid vision grid or frame exceeds {capacity} patches")
    if pixels.ndim != 2 or pixels.shape != (sum(t * h * w for t, h, w in grids), patch_dim):
        raise ValueError("packed vision pixels do not match grid_thw")
    frames, offset = [], 0
    for t, h, w in grids:
        for _ in range(t):
            frames.append((pixels[offset : offset + h * w], h, w))
            offset += h * w
    return frames


class Glm53VisionEncoder:
    def __init__(self, stem, blocks, merger, config, capacity, *, device="cpu", dtype=torch.float16, abi_version=2):
        self.stem, self.blocks, self.merger = stem, blocks, merger
        self.config, self.capacity, self.device, self.dtype = config, capacity, device, dtype
        if abi_version not in (1, 2):
            raise ValueError("unsupported GLM-5.3 vision ABI")
        self.abi_version = abi_version

    def to(self, device):
        for component in (self.stem, *self.blocks, self.merger):
            component.to(device)
        self.device = torch.device(device)
        return self

    @staticmethod
    def _tensor(value):
        return value if isinstance(value, torch.Tensor) else value[0]

    @torch.inference_mode()
    def __call__(self, pixels, grid):
        frames = vision_grid_frames(pixels, grid, self.config, self.capacity)
        features = []
        for patches, h, w in frames:
            values, cos, sin, mask = prepare_vision_frame(
                patches, h, w, self.config, self.capacity, device=self.device, dtype=self.dtype
            )
            hidden = self._tensor(self.stem(values))
            attention_input = mask if self.abi_version == 1 else torch.tensor([h * w], device=self.device, dtype=torch.int32)
            for block in self.blocks:
                hidden = self._tensor(block(hidden, cos, sin, attention_input))
            output = self._tensor(self.merger(hidden))
            features.append(output[0, : h * w // self.config.spatial_merge_size**2])
        return torch.cat(features, dim=0)


def fuse_multimodal_embeddings(input_ids, embedding, encoder, config, **values):
    """Use the native image-token/video-span convention before text chunking."""
    embeds = embedding(input_ids.to(embedding.weight.device))
    video_depth = (input_ids == config.video_start_token_id).int().cumsum(1)
    video_depth -= (input_ids == config.video_end_token_id).int().cumsum(1)
    if (video_depth < 0).any() or (video_depth > 1).any() or video_depth[0, -1] != 0:
        raise ValueError("video start/end markers must form complete, unnested spans")
    mm = input_ids == config.image_token_id
    for pixels_name, grid_name, mask in (
        ("pixel_values", "image_grid_thw", mm & (video_depth == 0)),
        ("pixel_values_videos", "video_grid_thw", mm & (video_depth > 0)),
    ):
        pixels, grid = values.get(pixels_name), values.get(grid_name)
        count = int(mask.sum())
        if pixels is None and grid is None and not count:
            continue
        if encoder is None:
            raise ValueError("this artifact has no vision component; use a multimodal export")
        frames = vision_grid_frames(pixels, grid, encoder.config, encoder.capacity)
        expected = sum(h * w // encoder.config.spatial_merge_size**2 for _, h, w in frames)
        if count != expected:
            raise ValueError(f"{grid_name}: {count} placeholder tokens != {expected} vision features")
        features = encoder(pixels, grid)
        embeds[mask.to(embeds.device)] = features.to(device=embeds.device, dtype=embeds.dtype)
    return embeds


def load_vision_encoder(vision, root, *, device="cuda:0"):
    """Load a portable vision component bundle without loading language weights."""
    from pathlib import Path

    from transformers.models.glm5_next import Glm5NextVisionConfig

    from ....utils import calculate_file_md5
    from ...hmonnx.hmonnx_model import HMONNXModel

    if vision.get("abi_version") not in (1, 2) or vision.get("dtype") != "float16":
        raise ValueError("unsupported GLM-5.3 vision ABI or dtype")
    config = Glm5NextVisionConfig(**vision["vision_config"])
    expected = ["stem", *[f"block_{i:02d}" for i in range(config.depth)], "merger"]
    if [c["name"] for c in vision["components"]] != expected:
        raise ValueError("incomplete or unordered GLM vision components")
    capacity = vision["max_patches_per_frame"]
    if capacity <= 0 or capacity % config.spatial_merge_size**2:
        raise ValueError("invalid vision patch capacity")
    paths = [Path(root) / vision["directory"] / component["hmonnx"] for component in vision["components"]]
    for path, component in zip(paths, vision["components"], strict=True):
        if calculate_file_md5(str(path)) != component["md5"]:
            raise ValueError(f"vision graph checksum mismatch: {path}")
    components = [HMONNXModel(str(path), device_map=[device]) for path in paths]
    return Glm53VisionEncoder(
        components[0], components[1:-1], components[-1], config, capacity,
        device=device, abi_version=vision["abi_version"],
    )


def create_glm53_processor(model_dir, capacity):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir)
    tokens = capacity // processor.image_processor.merge_size**2
    processor.image_processor.max_image_tokens = tokens
    processor.image_processor.min_image_tokens = min(processor.image_processor.min_image_tokens, tokens)
    # Native video budget counts merged tokens over the entire clip. This
    # default bounds every frame and leaves room for multiple media in a prompt.
    processor.video_processor.max_image_tokens = tokens
    processor.video_processor.min_image_tokens = min(processor.video_processor.min_image_tokens, tokens)
    return processor


def load_glm53_video(processor, path):
    """Decode with PyAV and preserve GLM's original sampling, including repeats."""
    import numpy as np
    from transformers.video_utils import load_video

    sampler = processor.video_processor.sample_frames

    def unique_indices(metadata, **kwargs):
        return np.unique(sampler(metadata=metadata))

    # HF's PyAV reader decodes unique frame indices. Expand repeats afterward
    # so a low-fps clip retains both the sampled pixel count and its timestamps.
    frames, metadata = load_video(str(path), backend="pyav", sample_indices_fn=unique_indices)
    requested = sampler(metadata=metadata)
    lookup = {int(index): offset for offset, index in enumerate(metadata.frames_indices)}
    frames = frames[[lookup[int(index)] for index in requested]]
    metadata.frames_indices = requested
    return frames, metadata
