# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Single-frame VAE tensor boundaries, retaining native RGBA and residual semantics."""

from copy import deepcopy

import torch
from torch import nn


class ChannelNormalize(nn.Module):
    """Traceable FP32 L2 normalization with the native clamp epsilon."""

    def __init__(self, source):
        super().__init__()
        self.gamma = source.gamma
        self.bias = source.bias
        self.scale = source.scale
        self.axis = 1 if source.channel_first else -1

    def forward(self, value):
        normalized = torch.nn.functional.normalize(value.float(), dim=self.axis).type_as(value)
        return normalized * self.scale * self.gamma + self.bias


class SpatialChannelNormalize(nn.Module):
    """Fuse channel normalization using the existing RMSNorm and a small epsilon."""

    def __init__(self, source):
        super().__init__()
        from xhquant.nn import RMSNorm

        if not source.channel_first or isinstance(source.bias, nn.Parameter):
            raise ValueError("image VAE requires channel-first normalization without affine bias")
        # Accept the small epsilon difference from native L2 normalization.
        # A 1e-8 variance floor also keeps the FP16 reciprocal finite at padding.
        self.norm = RMSNorm(source.gamma.numel(), eps=1e-8, reduce_dim=1)
        self.norm.weight = nn.Parameter(source.gamma.detach().reshape(-1).clone())

    def forward(self, value):
        return self.norm(value)


class SpatialConv2d(nn.Module):
    """Image convolution with the original explicit spatial padding and 4D ABI."""

    def __init__(self, source):
        super().__init__()
        self.padding = tuple(source._padding)
        self.conv = nn.Conv2d(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            source.stride,
            0,
            source.dilation,
            source.groups,
            bias=source.bias is not None,
        )
        self.conv.weight, self.conv.bias = source.weight, source.bias

    def forward(self, value):
        return self.conv(torch.nn.functional.pad(value, self.padding))


class FrameAttention(nn.Module):
    """Single-head image attention with explicit tensor operations for export."""

    def __init__(self, source):
        super().__init__()
        self.channels = source.dim
        self.norm = ChannelNormalize(source.norm)
        self.qkv = source.to_qkv
        self.projection = source.proj

    def forward(self, value):
        batch, channels, frames, height, width = value.shape
        image = value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        q, k, v = self.qkv(self.norm(image)).flatten(2).transpose(1, 2).chunk(3, -1)
        attention = ((q @ k.transpose(1, 2)) * self.channels**-0.5).softmax(-1) @ v
        image = attention.transpose(1, 2).reshape(batch * frames, channels, height, width)
        output = self.projection(image).reshape(batch, frames, channels, height, width)
        return value + output.permute(0, 2, 1, 3, 4)


class SpatialZeroPad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = tuple(padding)

    def forward(self, value):
        return torch.nn.functional.pad(value, self.padding)


class FrameConv2d(nn.Module):
    """Express the image-specialized causal convolution as a standard Conv2d."""

    def __init__(self, source):
        super().__init__()
        left, right, top, bottom = source._padding
        if left != right or top != bottom:
            raise ValueError("image VAE convolution requires symmetric spatial padding")
        self.padding = (left, right, top, bottom)
        self.conv = nn.Conv2d(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            source.stride,
            0,
            source.dilation,
            source.groups,
            bias=source.bias is not None,
        )
        self.conv.weight = source.weight
        self.conv.bias = source.bias

    def forward(self, value, cache_x=None):
        if cache_x is not None:
            raise ValueError("single-frame VAE convolution cannot consume temporal cache")
        # Preserve the native padded input shape as well as the mathematical
        # operation: folding pad into Conv2d can select different TF32 kernels.
        image = torch.nn.functional.pad(value.squeeze(2), self.padding)
        return self.conv(image).unsqueeze(2)


def _prepare(module):
    # Own the module tree; replacing normalization must not mutate a shared HF VAE.
    result = deepcopy(module)

    def replace(parent):
        for name, child in list(parent.named_children()):
            if type(child).__name__ == "QwenImage21CausalConv3d":
                setattr(parent, name, FrameConv2d(child))
            elif type(child).__name__ == "QwenImage21AttentionBlock":
                setattr(parent, name, FrameAttention(child))
            elif type(child).__name__ == "QwenImage21RMS_norm":
                setattr(parent, name, ChannelNormalize(child))
            elif isinstance(child, nn.ZeroPad2d):
                setattr(parent, name, SpatialZeroPad(child.padding))
            else:
                replace(child)

    replace(result)
    return result.eval()


class VaeEncode(nn.Module):
    """RGBA [1,4,1,H,W] to posterior mean, before Host latent normalization."""

    def __init__(self, source):
        super().__init__()
        if source.config.patch_size is not None:
            raise ValueError("Qwen-Image-2.1 image VAE requires patch_size=None")
        self.encoder = _prepare(source.encoder)
        self.projection = FrameConv2d(deepcopy(source.quant_conv))
        self.latent_channels = source.config.z_dim

    def forward(self, image):
        moments = self.projection(self.encoder(image, feat_cache=None, feat_idx=[0]))
        return moments[:, : self.latent_channels]


class VaeDecode(nn.Module):
    """Unnormalized [1,C,1,H,W] latent to RGBA in [-1,1], for a single frame."""

    def __init__(self, source):
        super().__init__()
        if source.config.patch_size is not None:
            raise ValueError("Qwen-Image-2.1 image VAE requires patch_size=None")
        self.decoder = _prepare(source.decoder)
        self.projection = FrameConv2d(deepcopy(source.post_quant_conv))

    def forward(self, latent):
        return self.decoder(self.projection(latent), feat_cache=None, feat_idx=[0], first_chunk=True).clamp(-1, 1)
