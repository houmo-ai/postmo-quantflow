"""Shared precomputed rotary tables and interleaved three-axis RoPE."""

from __future__ import annotations

import torch
from torch import Tensor, nn

import xhquant.nn as xhnn


MROPE_CURRENT_POSITION_NAMES = ("time_position_ids", "height_position_ids", "width_position_ids")
MROPE_POOL_POSITION_NAMES = ("pool_time_position_ids", "pool_height_position_ids", "pool_width_position_ids")


def compute_rotary_cache(inv_freq: Tensor, attention_scaling: float, max_seq_len: int) -> tuple[Tensor, Tensor]:
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(1, max_seq_len, -1, 1)
    positions = torch.arange(max_seq_len, device=inv_freq.device, dtype=torch.float32)[None, :, None, None]
    device_type = inv_freq.device.type if inv_freq.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded @ positions).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
    return cos[0].to(dtype=inv_freq.dtype), sin[0].to(dtype=inv_freq.dtype)


def combine_mrope_axes(time, height, width, time_mask, height_mask, width_mask):
    return time * time_mask + height * height_mask + width * width_mask


def prepare_mrope_position_inputs(group_start, position_ids, past_length, input_length, current_length):
    """Prepare current and C4 group-start T/H/W positions, retaining the open group's start."""
    current = group_start.new_zeros((*group_start.shape[:2], input_length))
    current[..., : position_ids.shape[-1]] = position_ids.to(device=current.device, dtype=current.dtype)
    group_count = (int(input_length) + 3) // 4
    offsets = torch.arange(group_count, device=current.device) * 4 - int(past_length) % 4
    group = current.index_select(-1, offsets.clamp_min(0))
    if int(past_length) % 4:
        group[..., :1] = group_start
    remaining = (int(past_length) + int(current_length)) % 4
    if remaining == 0:
        next_group_start = torch.zeros_like(group_start)
    elif remaining <= current_length:
        index = current_length - remaining
        next_group_start = current[..., index:index + 1].clone()
    else:
        next_group_start = group_start.clone()
    return current, group, next_group_start


class StaticMRotaryEmbedding(nn.Module):
    """Look up only requested T/H/W positions in FP32-built RoPE tables."""

    def __init__(self, inv_freq: Tensor, attention_scaling: float, max_seq_len: int, sections) -> None:
        super().__init__()
        cos, sin = compute_rotary_cache(inv_freq.float(), attention_scaling, max_seq_len)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        axis = torch.zeros(inv_freq.numel(), device=inv_freq.device, dtype=torch.int64)
        axis[1 : int(sections[1]) * 3 : 3] = 1
        axis[2 : int(sections[2]) * 3 : 3] = 2
        for index, name in enumerate(("time_mask", "height_mask", "width_mask")):
            mask = (axis == index).to(dtype=inv_freq.dtype)
            self.register_buffer(name, torch.cat((mask, mask)).view(1, 1, 1, -1), persistent=False)
        self.gather = xhnn.Gather(axis=0)

    def forward(self, x: Tensor, time: Tensor, height: Tensor, width: Tensor) -> tuple[Tensor, Tensor]:
        masks = (self.time_mask.to(x.dtype), self.height_mask.to(x.dtype), self.width_mask.to(x.dtype))
        cos = combine_mrope_axes(
            self.gather(self.cos_cached, time).to(x.dtype),
            self.gather(self.cos_cached, height).to(x.dtype),
            self.gather(self.cos_cached, width).to(x.dtype),
            *masks,
        )
        sin = combine_mrope_axes(
            self.gather(self.sin_cached, time).to(x.dtype),
            self.gather(self.sin_cached, height).to(x.dtype),
            self.gather(self.sin_cached, width).to(x.dtype),
            *masks,
        )
        return cos.squeeze(2), sin.squeeze(2)
