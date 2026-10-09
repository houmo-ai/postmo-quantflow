# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Packed Qwen3-VL patches with actual geometry and fixed FlashAttention capacity."""

from copy import deepcopy

import torch
from torch import nn


VISION_INPUT_NAMES = ["patches", "position_indices", "position_weights", "cos", "sin", "patch_mask", "input_length"]


def geometry_inputs(grid_hw, capacity, grid_side, head_dim, *, device, dtype):
    """Host geometry in the processor's merge-major 2x2 patch order."""
    height, width = map(int, grid_hw)
    if height <= 0 or width <= 0 or height % 2 or width % 2 or height * width > capacity:
        raise ValueError("vision patch grid must be even and fit the patch capacity")
    if capacity % 4 or head_dim % 4 or grid_side <= 0:
        raise ValueError("invalid Qwen3-VL geometry configuration")
    count = height * width
    order = torch.arange(count).reshape(height // 2, 2, width // 2, 2).permute(0, 2, 1, 3).flatten()
    h = torch.linspace(0, grid_side - 1, height)
    w = torch.linspace(0, grid_side - 1, width)
    hf, wf = h.int(), w.int()
    hc, wc = (hf + 1).clamp(max=grid_side - 1), (wf + 1).clamp(max=grid_side - 1)
    dh, dw = h - hf, w - wf
    indices = torch.stack(
        [(a[:, None] * grid_side + b[None, :]).flatten()[order] for a, b in ((hf, wf), (hf, wc), (hc, wf), (hc, wc))]
    )
    weights = torch.stack(
        [
            (a[:, None] * b[None, :]).flatten()[order]
            for a, b in ((1 - dh, 1 - dw), (1 - dh, dw), (dh, 1 - dw), (dh, dw))
        ]
    )
    row = torch.arange(height)[:, None].expand(height, width).flatten()[order]
    col = torch.arange(width)[None, :].expand(height, width).flatten()[order]
    inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2)))
    angle = torch.cat((row[:, None] * inv, col[:, None] * inv), dim=-1)
    angle = torch.cat((angle, angle), dim=-1)
    pad = capacity - count
    indices = torch.nn.functional.pad(indices, (0, pad)).to(device=device, dtype=torch.int32)
    weights = torch.nn.functional.pad(weights, (0, pad)).to(device=device, dtype=dtype)
    cos = torch.nn.functional.pad(angle.cos(), (0, 0, 0, pad), value=1).to(device=device, dtype=dtype)
    sin = torch.nn.functional.pad(angle.sin(), (0, 0, 0, pad)).to(device=device, dtype=dtype)
    mask = (torch.arange(capacity, device=device) < count).to(dtype).reshape(1, capacity, 1)
    return (
        indices,
        weights,
        cos[None, None],
        sin[None, None],
        mask,
        torch.tensor([count], device=device, dtype=torch.int32),
    )


class PackedVisionBlock(nn.Module):
    def __init__(self, source):
        super().__init__()
        from xhquant.nn import FlashAttention, Rope

        self.heads, self.head_dim = source.attn.num_heads, source.attn.head_dim
        self.physical_dim = next((dim for dim in (64, 128, 256, 512) if dim >= self.head_dim), None)
        if self.physical_dim is None or self.head_dim % 4:
            raise ValueError("Qwen3-VL vision head cannot fit supported FlashAttention channels")
        self.norm1, self.norm2 = deepcopy(source.norm1), deepcopy(source.norm2)
        self.qkv, self.proj = deepcopy(source.attn.qkv), deepcopy(source.attn.proj)
        self.mlp = deepcopy(source.mlp)
        self.rotary = Rope()
        self.attention = FlashAttention(num_heads=self.heads, is_causal=False, scale=self.head_dim**-0.5)

    def forward(self, hidden, cos, sin, mask, length):
        batch, capacity, _ = hidden.shape
        packed = self.qkv(self.norm1(hidden)).reshape(batch, capacity, 3, self.heads, self.head_dim)
        packed = packed.permute(2, 0, 3, 1, 4)
        q, k, v = packed[0], packed[1], packed[2]
        q, k = self.rotary(q, cos, sin), self.rotary(k, cos, sin)
        visible = mask.transpose(1, 2).unsqueeze(-1)
        q, k, v = q * visible, k * visible, v * visible
        if self.physical_dim > self.head_dim:
            pad = (0, self.physical_dim - self.head_dim)
            q, k, v = (torch.nn.functional.pad(x, pad) for x in (q, k, v))
        out = self.attention(q, k, v, current_input_length=length)[..., : self.head_dim]
        out = out.transpose(1, 2).reshape(batch, capacity, self.heads * self.head_dim)
        hidden = (hidden + self.proj(out)) * mask
        return (hidden + self.mlp(self.norm2(hidden))) * mask


class PackedVision(nn.Module):
    def __init__(self, source):
        super().__init__()
        config = source.config
        if (config.patch_size, config.temporal_patch_size, config.spatial_merge_size, config.in_channels) != (
            16,
            2,
            2,
            3,
        ):
            raise ValueError("Qwen-Image vision requires RGB patch16, temporal2, merge2")
        if len(source.deepstack_visual_indexes) != 3:
            raise ValueError("Qwen-Image requires three vision DeepStack outputs")
        projection = source.patch_embed.proj
        self.patch_projection = nn.Linear(
            projection.weight[0].numel(), projection.out_channels, bias=projection.bias is not None
        )
        self.patch_projection.weight = nn.Parameter(projection.weight.detach().flatten(1).clone())
        if projection.bias is not None:
            self.patch_projection.bias = nn.Parameter(projection.bias.detach().clone())
        self.position_embedding = deepcopy(source.pos_embed)
        self.blocks = nn.ModuleList(PackedVisionBlock(block) for block in source.blocks)
        self.merger = deepcopy(source.merger)
        self.deep_mergers = deepcopy(source.deepstack_merger_list)
        self.deep_indexes = tuple(source.deepstack_visual_indexes)

    def forward(self, patches, indices, weights, cos, sin, mask, length):
        position = self.position_embedding(indices.long()) * weights[:, :, None]
        position = position[0] + position[1] + position[2] + position[3]
        hidden = (self.patch_projection(patches * mask) + position[None]) * mask
        merged_mask = mask.reshape(1, -1, 4, 1)[:, :, 0]
        deep = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, cos, sin, mask, length)
            if index in self.deep_indexes:
                value = self.deep_mergers[self.deep_indexes.index(index)](hidden).unsqueeze(0)
                deep.append(value * merged_mask)
        return (self.merger(hidden).unsqueeze(0) * merged_mask, *deep)


class PackedVisionSession:
    def __init__(self, session, capacity, grid_side, head_dim, *, device, dtype=torch.float16):
        self.session, self.capacity = session, capacity
        self.grid_side, self.head_dim, self.device = grid_side, head_dim, device
        self.dtype = dtype

    def __call__(self, patches, grid_thw):
        time, height, width = map(int, grid_thw)
        if time != 1 or patches.shape != (height * width, 1536):
            raise ValueError("vision session requires one image's processor patches")
        geometry = geometry_inputs(
            (height, width), self.capacity, self.grid_side, self.head_dim, device=self.device, dtype=self.dtype
        )
        patches = patches.to(device=self.device, dtype=self.dtype)
        patches = torch.nn.functional.pad(patches, (0, 0, 0, self.capacity - patches.shape[0]))[None]
        output = self.session(patches, *geometry)
        return tuple(value[:, : height * width // 4].clone() for value in output)
