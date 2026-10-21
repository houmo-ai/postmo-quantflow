# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Spatial canvas VAE, including valid-region GroupNorm and native latent packing."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


VAE_INPUT_NAMES = [
    "vae_input",
    "mask_1",
    "mask_2",
    "mask_4",
    "mask_8",
    "mask_16",
    "inverse_fraction",
    "attention_order",
    "attention_restore",
    "attention_length",
]


def spatial_inputs(actual_hw, canvas_hw, *, device="cpu", dtype=torch.float32):
    h, w = actual_hw
    ch, cw = canvas_hw
    if any(type(n) is not int or n <= 0 or n % 16 for n in (*actual_hw, *canvas_hw)) or h > ch or w > cw:
        raise ValueError("VAE actual/canvas sizes must be positive aligned rectangles within capacity")
    masks = []
    for factor in (1, 2, 4, 8, 16):
        row = torch.arange(ch // factor, device=device)[:, None] < h // factor
        col = torch.arange(cw // factor, device=device)[None] < w // factor
        masks.append((row & col).to(dtype)[None, None])
    slots = torch.arange(ch * cw // 64, device=device).reshape(ch // 8, cw // 8)
    valid = masks[3].flatten().bool()
    order = torch.cat((slots[: h // 8, : w // 8].flatten(), slots.flatten()[~valid])).int()
    return (
        *masks,
        torch.tensor([ch * cw / (h * w)], device=device, dtype=dtype),
        order,
        torch.argsort(order).int(),
        torch.tensor([h * w // 64], dtype=torch.int32, device=device),
    )


class MaskedGroupNorm(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.groups, self.channels, self.eps = source.num_groups, source.num_channels, source.eps
        self.weight = nn.Parameter(source.weight.detach().float().reshape(1, -1, 1, 1).clone())
        self.bias = nn.Parameter(source.bias.detach().float().reshape(1, -1, 1, 1).clone())

    def forward(self, value, mask, inverse_fraction):
        batch, channels, height, width = value.shape
        x = (value * mask).reshape(batch, self.groups, self.channels // self.groups, height, width)
        visible = mask.reshape(batch, 1, 1, height, width)
        fraction = inverse_fraction.reshape(1, 1, 1, 1, 1)
        mean = x.mean((2, 3, 4), keepdim=True) * fraction
        centred = (x - mean) * visible
        variance = (centred * centred).mean((2, 3, 4), keepdim=True) * fraction
        normal = centred * torch.rsqrt(variance + self.eps)
        return (normal.reshape(batch, channels, height, width) * self.weight + self.bias) * mask


class Residual(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.norm1, self.norm2 = MaskedGroupNorm(source.norm1), MaskedGroupNorm(source.norm2)
        self.conv1, self.conv2 = deepcopy(source.conv1), deepcopy(source.conv2)
        self.shortcut = deepcopy(getattr(source, "nin_shortcut", nn.Identity()))

    def forward(self, value, mask, fraction):
        value = value * mask
        hidden = self.conv1(F.silu(self.norm1(value, mask, fraction))) * mask
        hidden = self.conv2(F.silu(self.norm2(hidden, mask, fraction))) * mask
        return (self.shortcut(value) + hidden) * mask


class Attention(nn.Module):
    def __init__(self, source):
        super().__init__()
        from xhquant.nn import FlashAttention

        self.norm = MaskedGroupNorm(source.norm)
        self.logical_dim = source.q.out_channels
        # The SDK's wide-head kernel uses a bounded-memory tile. Padding keeps
        # the complete single-head dot product and its original scale intact.
        self.physical_dim = 768 if self.logical_dim == 512 else self.logical_dim
        self.q, self.k, self.v, self.out = [
            deepcopy(value) for value in (source.q, source.k, source.v, source.proj_out)
        ]
        self.attention = FlashAttention(num_heads=1, is_causal=False, scale=source.q.out_channels**-0.5)

    def forward(self, value, mask, fraction, order, restore, length):
        x = self.norm(value, mask, fraction)
        batch, channels, h, w = value.shape

        def pack(projection):
            tensor = (projection(x) * mask).flatten(2).index_select(2, order.long())
            return F.pad(tensor.transpose(1, 2)[:, None], (0, self.physical_dim - self.logical_dim))

        result = self.attention(pack(self.q), pack(self.k), pack(self.v), current_input_length=length)
        result = (
            result[:, 0, :, : self.logical_dim]
            .transpose(1, 2)
            .index_select(2, restore.long())
            .reshape(batch, channels, h, w)
        )
        return (value + self.out(result * mask)) * mask


class CanvasVAE(nn.Module):
    def __init__(self, source, *, encode):
        super().__init__()
        self.encode = encode
        network = source.encoder if encode else source.decoder
        self.channels = source.decoder.conv_in.in_channels
        self.conv_in, self.conv_out = deepcopy(network.conv_in), deepcopy(network.conv_out)
        self.norm_out = MaskedGroupNorm(network.norm_out)
        self.mid1, self.mid2 = Residual(network.mid.block_1), Residual(network.mid.block_2)
        self.attention = Attention(network.mid.attn_1)
        self.blocks = nn.ModuleList()
        self.resample = nn.ModuleList()
        for _level, group in enumerate(network.down if encode else network.up):
            self.blocks.append(nn.ModuleList(Residual(block) for block in group.block))
            module = getattr(group, "downsample" if encode else "upsample", None)
            self.resample.append(deepcopy(module.conv) if module is not None else nn.Identity())
        self.quant = deepcopy(network.quant_conv if encode else network.post_quant_conv)
        scale = torch.sqrt(source.bn.running_var.detach().float() + source.bn_eps).reshape(1, -1, 1, 1)
        shift = source.bn.running_mean.detach().float().reshape(1, -1, 1, 1)
        self.register_buffer("bn_scale", scale)
        self.register_buffer("bn_shift", shift)

    def forward(
        self,
        vae_input,
        mask_1,
        mask_2,
        mask_4,
        mask_8,
        mask_16,
        inverse_fraction,
        attention_order,
        attention_restore,
        attention_length,
    ):
        masks = (mask_1, mask_2, mask_4, mask_8, mask_16)
        if self.encode:
            hidden = self.conv_in(vae_input * mask_1) * mask_1
            for level, blocks in enumerate(self.blocks):
                for block in blocks:
                    hidden = block(hidden, masks[level], inverse_fraction)
                if level < len(self.blocks) - 1:
                    hidden = self.resample[level](F.pad(hidden * masks[level], (0, 1, 0, 1))) * masks[level + 1]
        else:
            batch, _, h, w = vae_input.shape
            latent = (vae_input * self.bn_scale + self.bn_shift) * mask_16
            latent = latent.reshape(batch, self.channels, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
            latent = latent.reshape(batch, self.channels, h * 2, w * 2)
            hidden = self.conv_in(self.quant(latent * mask_8) * mask_8) * mask_8
        hidden = self.mid1(hidden, mask_8, inverse_fraction)
        hidden = self.attention(hidden, mask_8, inverse_fraction, attention_order, attention_restore, attention_length)
        hidden = self.mid2(hidden, mask_8, inverse_fraction)
        if not self.encode:
            for level in range(len(self.blocks) - 1, -1, -1):
                for block in self.blocks[level]:
                    hidden = block(hidden, masks[level], inverse_fraction)
                if level > 0:
                    hidden = F.interpolate(hidden * masks[level], scale_factor=2.0, mode="nearest") * masks[level - 1]
                    hidden = self.resample[level](hidden) * masks[level - 1]
        mask = mask_8 if self.encode else mask_1
        hidden = self.conv_out(F.silu(self.norm_out(hidden, mask, inverse_fraction))) * mask
        if self.encode:
            mean = self.quant(hidden).chunk(2, 1)[0] * mask_8
            batch, _, h, w = mean.shape
            packed = mean.reshape(batch, self.channels, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
            packed = packed.reshape(batch, 4 * self.channels, h // 2, w // 2)
            return ((packed - self.bn_shift) / self.bn_scale) * mask_16
        return hidden
