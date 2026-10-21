# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity complete flow-map DiT with explicit per-mode lengths."""

import math
from copy import deepcopy

import torch
from torch import nn


MODES = ("image", "image_cond", "text")
# Native BF16 text projections exceed FP16 (observed >115k). Store every
# residual stream divided by this power of two, and compensate its updates and
# LayerNorm epsilon. Q/K/V, modulation and the final prediction retain their
# original mathematical values.
RESIDUAL_SCALE = 1024.0
DIT_INPUT_NAMES = [
    "target",
    "references",
    "text_features",
    "time_features",
    "next_time_features",
    "cos",
    "sin",
    "mode_masks",
    "mode_lengths",
    "joint_order",
    "joint_mask",
    "joint_length",
]


class Block(nn.Module):
    def __init__(self, source):
        super().__init__()
        from xhquant.nn import FlashAttention, RMSNorm, Rope

        self.heads, self.hidden = source.heads, source.q_proj.out_features
        self.dim = self.hidden // self.heads
        self.norm = nn.LayerNorm(self.hidden, eps=1e-6 / RESIDUAL_SCALE**2, elementwise_affine=False)
        for name in ("q_proj", "k_proj", "v_proj", "mlp_in", "attn_out", "mlp_out"):
            setattr(self, name, deepcopy(getattr(source, name)))
        with torch.no_grad():
            for projection in (self.attn_out, self.mlp_out):
                projection.weight.div_(RESIDUAL_SCALE)
                if projection.bias is not None:
                    projection.bias.div_(RESIDUAL_SCALE)
        self.q_norm, self.k_norm = RMSNorm(self.dim, 1e-6), RMSNorm(self.dim, 1e-6)
        self.q_norm.weight = nn.Parameter(source.norm.query_norm.scale.detach().float().clone())
        self.k_norm.weight = nn.Parameter(source.norm.key_norm.scale.detach().float().clone())
        self.rope = Rope()
        self.attention = FlashAttention(num_heads=self.heads, is_causal=False, scale=self.dim**-0.5)

    def forward(self, hidden, mod, cos, sin, mask, length):
        shift, scale, gate = mod
        x = self.norm(hidden) * (1 + scale) + shift
        q = self.q_norm(self.q_proj(x).reshape(1, -1, self.heads, self.dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).reshape(1, -1, self.heads, self.dim)).transpose(1, 2)
        v = self.v_proj(x).reshape(1, -1, self.heads, self.dim).transpose(1, 2)
        # The source rotates adjacent pairs; SDK Rope rotates channel halves.
        q = torch.cat((q[..., 0::2], q[..., 1::2]), -1)
        k = torch.cat((k[..., 0::2], k[..., 1::2]), -1)
        visible = mask.transpose(1, 2).unsqueeze(-1)
        q, k, v = self.rope(q, cos, sin) * visible, self.rope(k, cos, sin) * visible, v * visible
        attention = self.attention(q, k, v, current_input_length=length)
        attention = attention.transpose(1, 2).reshape(1, -1, self.hidden)
        a, b = self.mlp_in(x).chunk(2, -1)
        update = self.attn_out(attention) + self.mlp_out(torch.nn.functional.silu(a) * b)
        return (hidden + gate * update) * mask


class Modulation(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.lin = deepcopy(source.lin)

    def forward(self, value):
        shift, scale, gate = self.lin(torch.nn.functional.silu(value)).chunk(3, -1)
        return shift, scale, gate


class Denoiser(nn.Module):
    def __init__(self, source, capacities):
        super().__init__()
        if not source.flow_map:
            raise ValueError("FLUX 3 Klein deployment requires the distilled flow-map checkpoint")
        self.capacities = tuple(capacities)
        self.time, self.ref_time = deepcopy(source.time_in), deepcopy(source.ref_time_in)
        self.inputs = nn.ModuleList(
            [deepcopy(source.emb_in["image"]), deepcopy(source.emb_in["image_cond"]), deepcopy(source.txt_in)]
        )
        with torch.no_grad():
            for projection in self.inputs:
                projection.weight.div_(RESIDUAL_SCALE)
                if projection.bias is not None:
                    projection.bias.div_(RESIDUAL_SCALE)
        self.early_mod = nn.ModuleDict(
            {key: Modulation(value) for key, value in source.early_stream_modulations.items()}
        )
        self.single_mod = nn.ModuleDict(
            {key: Modulation(value) for key, value in source.single_stream_modulations.items()}
        )
        self.early = nn.ModuleList(
            [
                nn.ModuleList(Block(block) for block in source.content_mode_blocks["image"]),
                nn.ModuleList(Block(block) for block in source.content_mode_blocks["image_cond"]),
                nn.ModuleList(Block(block) for block in source.txt_mode_blocks),
            ]
        )
        self.joint = nn.ModuleList(Block(block) for block in source.single_blocks)
        self.final_mod = deepcopy(source.final_layer["image"].adaLN_modulation)
        self.final_norm = nn.LayerNorm(
            source.txt_in.out_features, eps=1e-6 / RESIDUAL_SCALE**2, elementwise_affine=False
        )
        self.final_out = deepcopy(source.final_layer["image"].linear)

    def forward(
        self,
        target,
        references,
        text_features,
        time_features,
        next_time_features,
        cos,
        sin,
        mode_masks,
        mode_lengths,
        joint_order,
        joint_mask,
        joint_length,
    ):
        times = self.time(time_features) + self.ref_time(next_time_features)
        values, single, offset = [], ([], [], []), 0
        for index, (key, tokens) in enumerate(
            zip(("image", "image_cond", "txt"), (target, references, text_features), strict=True)
        ):
            capacity = self.capacities[index]
            mask = mode_masks[:, offset : offset + capacity]
            c, s = cos[:, :, offset : offset + capacity], sin[:, :, offset : offset + capacity]
            vec = times[index : index + 1, None]
            early_mod, single_mod = self.early_mod[key](vec), self.single_mod[key](vec)
            hidden = self.inputs[index](tokens) * mask
            for block in self.early[index]:
                hidden = block(hidden, early_mod, c, s, mask, mode_lengths[index : index + 1])
            values.append(hidden)
            for j in range(3):
                single[j].append(single_mod[j].expand(1, capacity, -1))
            offset += capacity
        hidden = torch.cat(values, 1).index_select(1, joint_order.long())
        mods = tuple(torch.cat(parts, 1).index_select(1, joint_order.long()) for parts in single)
        c, s = cos.index_select(2, joint_order.long()), sin.index_select(2, joint_order.long())
        for block in self.joint:
            hidden = block(hidden, mods, c, s, joint_mask, joint_length)
        shift, scale = self.final_mod(times[0:1, None]).chunk(2, -1)
        hidden = self.final_norm(hidden[:, : self.capacities[0]]) * (1 + scale) + shift
        return self.final_out(hidden)


def time_features(timestep, *, device="cpu", dtype=torch.float32):
    t = torch.as_tensor(timestep, device=device, dtype=dtype).reshape(-1, 1)
    frequency = torch.exp(-math.log(10000.0) * torch.arange(128, device=device).float() / 128)
    angle = (1000 * t).float() * frequency[None]
    return torch.cat((angle.cos(), angle.sin()), -1).to(dtype)


def rotary(ids, axes_dim=(32, 32, 32, 32)):
    angles = []
    for axis, dim in enumerate(axes_dim):
        frequency = 1 / 10000 ** (torch.arange(0, dim, 2, device=ids.device).double() / dim)
        angles.append(ids[:, axis : axis + 1].double() * frequency[None])
    angle = torch.cat(angles, -1).float()
    return torch.cat((angle.cos(), angle.cos()), -1)[None, None], torch.cat((angle.sin(), angle.sin()), -1)[None, None]


def image_ids(height, width, *, reference=None, device="cpu"):
    row, col = torch.meshgrid(
        torch.arange(height // 16, device=device), torch.arange(width // 16, device=device), indexing="ij"
    )
    zero = torch.zeros_like(row)
    return (
        torch.stack((zero, row * 0.5, col * 0.5, zero if reference is None else zero - reference - 1), -1)
        .reshape(-1, 4)
        .float()
    )


def denoise_inputs(target, references, text, target_hw, reference_hws, capacities, timestep, next_timestep):
    from torch.nn import functional as F

    device, dtype = target.device, target.dtype
    lengths = [target.shape[1], references.shape[1], text.shape[1]]
    ids = [image_ids(*target_hw, device=device)]
    refs = [image_ids(*size, reference=index, device=device) for index, size in enumerate(reference_hws)]
    ids.append(torch.cat(refs) if refs else torch.empty(0, 4, device=device))
    position = torch.arange(lengths[2], device=device)
    ids.append(torch.stack((torch.zeros_like(position),) * 3 + (position,), -1).float())
    padded, padded_ids, masks, valid, invalid, offset = [], [], [], [], [], 0
    for value, positions, count, capacity in zip((target, references, text), ids, lengths, capacities, strict=True):
        if count > capacity:
            raise ValueError("mode tokens exceed the graph capacity")
        padded.append(F.pad(value, (0, 0, 0, capacity - count)))
        padded_ids.append(F.pad(positions, (0, 0, 0, capacity - count)))
        masks.append((torch.arange(capacity, device=device) < count).to(dtype)[None, :, None])
        valid.append(torch.arange(offset, offset + count, device=device))
        invalid.append(torch.arange(offset + count, offset + capacity, device=device))
        offset += capacity
    order = torch.cat((*valid, *invalid)).int()
    cos, sin = rotary(torch.cat(padded_ids))
    joint_mask = (torch.arange(offset, device=device) < sum(lengths)).to(dtype)[None, :, None]
    return (
        *padded,
        time_features([timestep, 0, 0], device=device, dtype=dtype),
        time_features([next_timestep], device=device, dtype=dtype),
        cos.to(dtype),
        sin.to(dtype),
        torch.cat(masks, 1),
        torch.tensor(lengths, device=device, dtype=torch.int32),
        order,
        joint_mask,
        torch.tensor([sum(lengths)], device=device, dtype=torch.int32),
    )
