# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Deterministic conv video VAE, with pixel normalization kept as an SDK RMSNorm."""

from __future__ import annotations

import math

import torch
from torch import nn


class ChannelPixelNorm(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        self.norm = None
        self.channels = None
        self.bounded = False

    def set_static_shape(self, shape):
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        from xhquant.nn import RMSNorm

        if self.norm is None:
            with unset_fake_temporarily():
                self.channels = shape[1]
                self.bounded = self.eps < 65504.0**-2
                if self.bounded:
                    from .precision import StableRMSNorm

                    # Residual32 scales PixelNorm epsilon by 1/1024. The SDK
                    # reciprocal narrows to half even with FP32 sums; keep it
                    # bounded using the existing augmented-coordinate graph.
                    source = nn.RMSNorm(shape[1], eps=self.eps * 32**2)
                    self.norm = StableRMSNorm(source, input_divisor=32, sdk_ops=True)
                else:
                    self.norm = RMSNorm(shape[1], self.eps, reduce_dim=1).requires_grad_(False)
        elif self.channels != shape[1]:
            raise ValueError("shared PixelNorm has incompatible channel counts")

    def forward(self, x):
        if self.norm is not None:
            if self.bounded:
                return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            return self.norm(x)
        wide = x.float()
        return (wide * torch.rsqrt(wide.square().mean(1, keepdim=True) + self.eps)).to(x.dtype)


class CausalPadConv(nn.Module):
    def __init__(self, source, *, static_causal=None):
        super().__init__()
        self.convolution = source.conv
        self.time_padding = source.time_kernel_size - 1
        self.static_causal = static_causal

    def forward(self, value, causal=True):
        if self.static_causal is not None and causal != self.static_causal:
            raise ValueError("causal mode differs from the static decoder graph")
        if causal:
            prefix = value[:, :, :1].repeat(1, 1, self.time_padding, 1, 1)
            value = torch.cat((prefix, value), 2)
        else:
            half = self.time_padding // 2
            prefix = value[:, :, :1].repeat(1, 1, half, 1, 1)
            suffix = value[:, :, -1:].repeat(1, 1, half, 1, 1)
            value = torch.cat((prefix, value, suffix), 2)
        return self.convolution(value)


class DepthExpansion(nn.Module):
    """Expand channel offsets into time/space using explicit tensor operations."""

    def __init__(self, source):
        super().__init__()
        self.conv = source.conv
        self.factors = tuple(source.stride)
        self.residual = source.residual
        self.repeats = self.factors[0] * self.factors[1] * self.factors[2] // source.out_channels_reduction_factor

    def expand(self, x):
        batch, channels, time, height, width = x.shape
        dt, dh, dw = self.factors
        result = x.reshape(batch, channels // (dt * dh * dw), dt, dh, dw, time, height, width)
        result = result.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(
            batch, channels // (dt * dh * dw), time * dt, height * dh, width * dw
        )
        return result[:, :, 1:] if dt == 2 else result

    def forward(self, x, causal=True):
        result = self.expand(self.conv(x, causal=causal))
        return result + self.expand(x).repeat(1, self.repeats, 1, 1, 1) if self.residual else result


class DepthReduction(nn.Module):
    """Pack temporal/spatial offsets into channels for native residual encode."""

    def __init__(self, source):
        super().__init__()
        self.conv = source.conv
        self.factors = tuple(source.stride)
        self.group_size = source.group_size

    def pack(self, x):
        batch, channels, frames, height, width = x.shape
        dt, dh, dw = self.factors
        return (
            x.reshape(batch, channels, frames // dt, dt, height // dh, dh, width // dw, dw)
            .permute(0, 1, 3, 5, 7, 2, 4, 6)
            .reshape(batch, channels * dt * dh * dw, frames // dt, height // dh, width // dw)
        )

    def forward(self, x, causal=True):
        if self.factors[0] == 2:
            x = torch.cat((x[:, :, :1], x), dim=2)
        skip = self.pack(x)
        batch, channels, frames, height, width = skip.shape
        skip = skip.reshape(batch, channels // self.group_size, self.group_size, frames, height, width).mean(dim=2)
        return self.pack(self.conv(x, causal=causal)) + skip


class SpatialUnpatchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        batch, packed, frames, height, width = x.shape
        p = self.patch_size
        channels = packed // (p * p)
        x = x.reshape(batch, channels, p, p, frames, height, width)
        # The checkpoint packs width-offset before height-offset.
        return x.permute(0, 1, 4, 5, 3, 6, 2).reshape(batch, channels, frames, height * p, width * p)


def _adapt_vae_norms_and_convolutions(source, *, static_causal=None):
    from ltx_core.model.common.normalization import PixelNorm
    from ltx_core.model.video_vae.convolution import CausalConv3d
    from ltx_core.model.video_vae.sampling import SpaceToDepthDownsample

    for name, module in list(source.named_modules()):
        if isinstance(module, PixelNorm):
            source.set_submodule(name, ChannelPixelNorm(module.eps))
    for name, module in list(source.named_modules()):
        if isinstance(module, CausalConv3d):
            source.set_submodule(name, CausalPadConv(module, static_causal=static_causal))
    for name, module in list(source.named_modules()):
        if isinstance(module, SpaceToDepthDownsample):
            source.set_submodule(name, DepthReduction(module))


@torch.no_grad()
def _scale_encoder_residuals(source, scale):
    """Keep the linear residual stream at x/scale; normalize in original units.

    Downsample skip branches are linear, so their convolution biases alone
    change. A normalized residual branch instead scales its final convolution.
    Compensating epsilon preserves PixelNorm/GroupNorm's mathematical value.
    """
    from ltx_core.model.video_vae.convolution import CausalConv3d
    from ltx_core.model.video_vae.resnet import ResnetBlock3D, UNetMidBlock3D
    from ltx_core.model.video_vae.sampling import SpaceToDepthDownsample

    if scale == 1:
        return
    if any(
        not isinstance(block, (ResnetBlock3D, UNetMidBlock3D, SpaceToDepthDownsample, CausalConv3d))
        for block in source.down_blocks
    ):
        raise ValueError("scaled LTX encode requires normalized residual and space-to-depth blocks")

    def convolution(module, *, weight=True):
        conv = module.conv if hasattr(module, "conv") else module
        if weight:
            conv.weight.div_(scale)
        if conv.bias is not None:
            conv.bias.div_(scale)

    def normalization(module):
        module.eps /= scale * scale

    convolution(source.conv_in)
    for block in source.down_blocks:
        if isinstance(block, CausalConv3d):
            convolution(block, weight=False)
    for block in source.modules():
        if isinstance(block, ResnetBlock3D):
            if block.inject_noise or block.timestep_conditioning:
                raise ValueError("scaled LTX encode requires deterministic residual blocks")
            normalization(block.norm1)
            convolution(block.conv2)
            if not isinstance(block.norm3, nn.Identity):
                normalization(block.norm3)
                convolution(block.conv_shortcut)
        elif isinstance(block, SpaceToDepthDownsample):
            convolution(block.conv, weight=False)
    normalization(source.conv_norm_out)


class VideoEncodeGraph(nn.Module):
    """Deterministic normalized LTX image/video means with safe PixelNorm.

    Keeps the checkpoint's width-before-height patch order and discards only
    unused posterior variance channels. References use valid temporal lengths
    (one frame for a keyframe), so no implicit frame cropping is performed.
    """

    def __init__(self, source, *, residual_scale=32.0, normalization_epsilon=None):
        super().__init__()
        from . import source_io

        if not math.isfinite(residual_scale) or residual_scale < 1 or math.log2(residual_scale) % 1:
            raise ValueError("LTX encoder residual scale must be a finite power of two >= 1")
        if normalization_epsilon is not None and (
            type(normalization_epsilon) not in (int, float)
            or not math.isfinite(normalization_epsilon)
            or normalization_epsilon <= 0
        ):
            raise ValueError("LTX encoder normalization epsilon must be finite and positive or None")
        source = source_io.prepare_video_encoder(source, residual_scale, normalization_epsilon)
        self.residual_scale = residual_scale
        self.patch_size, self.temporal_factor = source.patch_size, source.video_scale_factors.time
        self.channels = source.latent_channels
        self.input, self.blocks = source.conv_in, source.down_blocks
        self.norm, self.activation, self.output = source.conv_norm_out, source.conv_act, source.conv_out
        for target, name in (("latent_mean", "mean-of-means"), ("latent_std", "std-of-means")):
            self.register_buffer(
                target, source.per_channel_statistics.get_buffer(name).detach().reshape(1, -1, 1, 1, 1)
            )

    def forward(self, pixels):
        batch, channels, frames, height, width = pixels.shape
        p = self.patch_size
        if (frames - 1) % self.temporal_factor or height % p or width % p:
            raise ValueError("LTX encoder input must have valid temporal and spatial dimensions")
        value = pixels.reshape(batch, channels, frames, height // p, p, width // p, p)
        value = value.permute(0, 1, 6, 4, 2, 3, 5).reshape(batch, channels * p * p, frames, height // p, width // p)
        value = self.input(value)
        for block in self.blocks:
            value = block(value)
        means = self.output(self.activation(self.norm(value)))[:, : self.channels]
        return (means - self.latent_mean.to(means)) / self.latent_std.to(means)


class VideoDecodeGraph(nn.Module):
    def __init__(self, source):
        super().__init__()
        if source.timestep_conditioning:
            raise ValueError("this static VAE does not accept stochastic timestep conditioning")
        from . import source_io

        source = source_io.prepare_video_decoder(source)
        self.register_buffer(
            "latent_std", source.per_channel_statistics.get_buffer("std-of-means").detach().reshape(1, -1, 1, 1, 1)
        )
        self.register_buffer(
            "latent_mean", source.per_channel_statistics.get_buffer("mean-of-means").detach().reshape(1, -1, 1, 1, 1)
        )
        self.input = source.conv_in
        self.blocks = source.up_blocks
        self.norm, self.activation, self.output = source.conv_norm_out, source.conv_act, source.conv_out
        self.patch_size, self.causal = source.patch_size, source.causal
        self.unpatchify = SpatialUnpatchify(self.patch_size)

    def forward(self, latent):
        value = self.input(latent * self.latent_std + self.latent_mean, causal=self.causal)
        for block in self.blocks:
            if type(block).__name__ in ("UNetMidBlock3D", "ResnetBlock3D"):
                value = block(value, causal=self.causal, generator=None)
            else:
                value = block(value, causal=self.causal)
        value = self.output(self.activation(self.norm(value)), causal=self.causal)
        return self.unpatchify(value)


class MaskedVideoDecodeGraph(VideoDecodeGraph):
    """Share spatial tile gears while preserving each convolution's zero boundary."""

    def __init__(self, source):
        super().__init__(source)
        # PixelNorm is local to a pixel. Spatial GroupNorm would instead need
        # valid-domain statistics and cannot use this boundary-only rewrite.
        for module in self.modules():
            if isinstance(module, nn.GroupNorm):
                raise ValueError("masked LTX decode requires the checkpoint's PixelNorm architecture")
            if isinstance(module, nn.Conv3d) and module.padding_mode != "zeros":
                raise ValueError("masked LTX decode requires zero spatial convolution padding")
            if getattr(module, "inject_noise", False):
                raise ValueError("masked LTX decode requires deterministic residual blocks")
        scales = [(1, 1)]
        for block in self.blocks:
            if isinstance(block, DepthExpansion):
                scale = tuple(a * b for a, b in zip(scales[-1], block.factors[1:], strict=True))
                if scale != scales[-1]:
                    scales.append(scale)
            elif type(block).__name__ not in ("UNetMidBlock3D", "ResnetBlock3D"):
                raise ValueError(f"unsupported masked LTX block: {type(block).__name__}")
        self.mask_scales = tuple(scales)

    def _resnet(self, block, value, mask):
        hidden = block.conv1(block.non_linearity(block.norm1(value)) * mask, causal=self.causal)
        hidden = block.dropout(block.non_linearity(block.norm2(hidden)))
        hidden = block.conv2(hidden * mask, causal=self.causal)
        return block.conv_shortcut(block.norm3(value) * mask) + hidden

    def forward(self, latent, *masks):
        mask_index = 0
        mask = masks[mask_index]
        value = self.input((latent * self.latent_std + self.latent_mean) * mask, causal=self.causal)
        for block in self.blocks:
            if isinstance(block, DepthExpansion):
                value = block(value * mask, causal=self.causal)
                if block.factors[1:] != (1, 1):
                    mask_index += 1
                    mask = masks[mask_index]
            elif type(block).__name__ == "UNetMidBlock3D":
                for resnet in block.res_blocks:
                    value = self._resnet(resnet, value, mask)
            else:
                value = self._resnet(block, value, mask)
        value = self.output(self.activation(self.norm(value)) * mask, causal=self.causal)
        return self.unpatchify(value)


def decoder_mask_inputs(latent, shape, scales):
    """Pad one native tile and describe its valid rectangle at every resolution."""
    frames, height, width = map(int, shape)
    if (
        latent.ndim != 5
        or latent.shape[2] != frames
        or any(n > limit for n, limit in zip(latent.shape[-2:], (height, width), strict=True))
    ):
        raise ValueError("decoder tile exceeds the shared graph or changes its temporal boundary")
    h, w = latent.shape[-2:]
    masks = []
    for sy, sx in scales:
        mask = latent.new_zeros((1, 1, 1, height * sy, width * sx))
        mask[..., : h * sy, : w * sx] = 1
        masks.append(mask)
    padded = torch.nn.functional.pad(latent, (0, width - w, 0, height - h))
    return (padded, *masks)
