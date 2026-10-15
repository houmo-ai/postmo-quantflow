# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""One H3 reference encoder with masked domains and native reflection boundaries."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class MaskedSpatialGroupNorm(nn.Module):
    """Native per-frame GroupNorm, excluding the static canvas's padded pixels."""

    def __init__(self, source):
        super().__init__()
        self.groups, self.eps = source.num_groups, source.eps
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        self.norm = None

    def set_static_shape(self, shape):
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        from xhquant.nn import RMSNorm

        width = shape[1] // self.groups * shape[3] * shape[4]
        if self.norm is None:
            with unset_fake_temporarily():
                self.norm = RMSNorm(width, self.eps).requires_grad_(False)

    def forward(self, value, mask, area_ratio, area_scale, inverse_scale):
        batch, channels, frames, height, width = value.shape
        grouped = value.permute(0, 2, 1, 3, 4).reshape(batch, frames, self.groups, -1, height, width)
        valid = mask.permute(0, 2, 1, 3, 4).unsqueeze(2)
        mean = (grouped * valid).mean((3, 4, 5), keepdim=True) * area_ratio
        centered = ((grouped - mean) * valid * area_scale).reshape(batch, frames, self.groups, -1)
        # Multiplying by sqrt(padded_area / valid_area) lets the ordinary
        # wide-accumulation RMS primitive retain the original epsilon exactly.
        if self.norm is None:
            wide = centered.float()
            normalized = (wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + self.eps)).to(value.dtype)
        else:
            normalized = self.norm(centered)
        normalized = (normalized * inverse_scale).reshape(batch, frames, channels, height, width)
        normalized = normalized.permute(0, 2, 1, 3, 4)
        return normalized * self.weight.reshape(1, -1, 1, 1, 1) + self.bias.reshape(1, -1, 1, 1, 1)


class MaskedReferenceConv(nn.Module):
    def __init__(self, source, *, downsample=False):
        super().__init__()
        from xhquant.nn import Gather

        if getattr(source, "spatial_padding_mode", "reflect") != "reflect":
            raise ValueError("shared H3 reference encoder requires native reflect spatial padding")
        self.padding = getattr(source, "spatial_padding", 0)
        self.temporal_padding = getattr(source, "temporal_padding", 0)
        self.downsample = downsample
        self.height_gather, self.width_gather = Gather(axis=3), Gather(axis=4)
        self.conv = nn.Conv3d(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            source.stride,
            padding=0,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
            device="meta",
        )
        self.conv.weight = source.weight
        if source.bias is not None:
            self.conv.bias = source.bias

    def forward(self, value, mask, height_indices, width_indices):
        # Only temporal validity is needed here: reflection gathers select
        # spatial values inside the native rectangle, never the padded tail.
        value = value * mask[:, :, :, :1, :1]
        if self.padding or self.downsample:
            if self.padding not in (0, 1):
                raise ValueError("shared reference encoder expects a one-pixel reflection boundary")
            if self.downsample:
                height_indices, width_indices = height_indices[1:], width_indices[1:]
            value = self.height_gather(value, height_indices)
            value = self.width_gather(value, width_indices)
        if self.temporal_padding:
            value = F.pad(value, (0, 0, 0, 0, self.temporal_padding, 0))
        return self.conv(value)


class MaskedReferenceResnet(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.norm1, self.norm2 = MaskedSpatialGroupNorm(source.norm1), MaskedSpatialGroupNorm(source.norm2)
        self.conv1, self.conv2 = MaskedReferenceConv(source.conv1), MaskedReferenceConv(source.conv2)
        self.shortcut = MaskedReferenceConv(source.conv_shortcut) if source.conv_shortcut is not None else None

    def forward(self, value, mask, height_indices, width_indices, *area):
        hidden = self.conv1(F.silu(self.norm1(value, mask, *area)), mask, height_indices, width_indices)
        hidden = self.conv2(F.silu(self.norm2(hidden, mask, *area)), mask, height_indices, width_indices)
        residual = self.shortcut(value, mask, height_indices, width_indices) if self.shortcut else value
        return residual + hidden


class MaskedReferenceEncodeGraph(nn.Module):
    spatial_padding_mode = "reflect"

    def __init__(self, source):
        super().__init__()
        encoder = source.encoder
        self.input, self.output = MaskedReferenceConv(encoder.conv_in), MaskedReferenceConv(encoder.conv_out)
        self.norm = MaskedSpatialGroupNorm(encoder.norm_out)
        self.quant = source.quant_conv
        self.resnets, self.downsamplers = nn.ModuleList(), nn.ModuleList()
        self.levels, self.block_levels = [(1, 1)], []
        self.downsample_levels = []
        for block in encoder.down_blocks:
            self.block_levels.append(len(self.levels) - 1)
            self.resnets.append(nn.ModuleList(MaskedReferenceResnet(resnet) for resnet in block.resnets))
            samplers, levels = nn.ModuleList(), []
            for sampler in block.downsamplers or []:
                levels.append(len(self.levels) - 1)
                samplers.append(MaskedReferenceConv(sampler.conv, downsample=sampler.spatial_stride == 2))
                st, sh, sw = sampler.conv.stride
                if sh != sw:
                    raise ValueError("shared reference encoder requires equal spatial strides")
                self.levels.append((self.levels[-1][0] * st, self.levels[-1][1] * sh))
            self.downsamplers.append(samplers)
            self.downsample_levels.append(levels)

    def forward(self, pixels, area_ratio, area_scale, inverse_scale, *domains):
        area = (area_ratio, area_scale, inverse_scale)
        value = self.input(pixels, *domains[:3])
        for block, samplers, level, sampler_levels in zip(
            self.resnets, self.downsamplers, self.block_levels, self.downsample_levels, strict=True
        ):
            domain = domains[level * 3 : level * 3 + 3]
            for resnet in block:
                value = resnet(value, *domain, *area)
            for sampler, level in zip(samplers, sampler_levels, strict=True):
                value = sampler(value, *domains[level * 3 : level * 3 + 3])
        domain = domains[-3:]
        value = self.output(F.silu(self.norm(value, domain[0], *area)), *domain)
        return self.quant(value)


def reference_mask_names(levels, spatial_padding_mode="reflect"):
    # Missing policy in already published packages means the original reflect ABI.
    if spatial_padding_mode not in ("reflect", "constant"):
        raise ValueError("unsupported reference spatial padding mode")
    kinds = ("valid_mask", "height_indices", "width_indices") if spatial_padding_mode == "reflect" else ("valid_mask",)
    return [
        "pixels",
        "area_ratio",
        "area_scale",
        "inverse_scale",
        *[f"{kind}_{i}" for i in range(len(levels)) for kind in kinds],
    ]


def reference_mask_inputs(pixels, shape, levels, spatial_padding_mode="reflect"):
    if spatial_padding_mode not in ("reflect", "constant"):
        raise ValueError("unsupported reference spatial padding mode")
    frames, height, width = pixels.shape[2:]
    mt, mh, mw = shape
    if (
        pixels.ndim != 5
        or pixels.shape[:2] != (1, 3)
        or any(n > maximum for n, maximum in zip((frames, height, width), shape, strict=True))
        or height % levels[-1][1]
        or width % levels[-1][1]
    ):
        raise ValueError("reference tile is outside the shared encoder canvas")
    ratio = mh * mw / (height * width)
    args = [F.pad(pixels, (0, mw - width, 0, mh - height, 0, mt - frames))]
    args.extend(pixels.new_tensor([value]) for value in (ratio, math.sqrt(ratio), 1 / math.sqrt(ratio)))
    for temporal, spatial in levels:
        t, h, w = (mt - 1) // temporal + 1, mh // spatial, mw // spatial
        valid_t, valid_h, valid_w = (frames - 1) // temporal + 1, height // spatial, width // spatial
        if spatial_padding_mode == "reflect" and min(valid_h, valid_w) <= 1:
            raise ValueError("reference reflection requires at least two pixels on the coarsest axis")
        mask = pixels.new_zeros((1, 1, t, h, w))
        mask[:, :, :valid_t, :valid_h, :valid_w] = 1
        args.append(mask)
        if spatial_padding_mode == "constant":
            continue
        for maximum, valid in ((h, valid_h), (w, valid_w)):
            index = torch.arange(-1, maximum + 1, device=pixels.device) % (2 * (valid - 1))
            args.append(torch.minimum(index, 2 * (valid - 1) - index).to(torch.int32))
    return tuple(args)
