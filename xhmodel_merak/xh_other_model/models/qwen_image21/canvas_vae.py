# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Fixed spatial canvas VAE graphs with request-supplied masks and permutations."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .vae_graph import SpatialChannelNormalize, SpatialConv2d, SpatialZeroPad


VAE_INPUT_NAMES = [
    "vae_input",
    "mask_1",
    "mask_2",
    "mask_4",
    "mask_8",
    "mask_16",
    "attention_order",
    "attention_restore",
    "attention_length",
]


def spatial_inputs(actual_hw, canvas_hw, *, device="cpu", dtype=torch.float16):
    """Host-only geometry preparation; every returned tensor has a fixed shape."""
    if any(type(n) is not int or n <= 0 or n % 32 for n in (*actual_hw, *canvas_hw)):
        raise ValueError("actual and canvas dimensions must be positive multiples of 32")
    height, width = actual_hw
    cap_h, cap_w = canvas_hw
    if height > cap_h or width > cap_w:
        raise ValueError("actual dimensions exceed the spatial canvas")
    masks = []
    for factor in (1, 2, 4, 8, 16):
        rows = torch.arange(cap_h // factor, device=device)[:, None] < height // factor
        cols = torch.arange(cap_w // factor, device=device)[None, :] < width // factor
        masks.append((rows & cols).to(dtype)[None, None])
    slots = torch.arange(cap_h * cap_w // 256, device=device).reshape(cap_h // 16, cap_w // 16)
    valid = slots[: height // 16, : width // 16].flatten()
    visible = masks[-1].flatten().bool()
    order = torch.cat((valid, slots.flatten()[~visible])).to(torch.int32)
    restore = torch.argsort(order).to(torch.int32)
    length = torch.tensor([height * width // 256], dtype=torch.int32, device=device)
    return (*masks, order, restore, length)


class CanvasConv(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.conv = SpatialConv2d(deepcopy(source))

    def forward(self, value, mask):
        return self.conv(value * mask) * mask


class CanvasResidual(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.norm1, self.norm2 = SpatialChannelNormalize(source.norm1), SpatialChannelNormalize(source.norm2)
        self.conv1, self.conv2 = CanvasConv(source.conv1), CanvasConv(source.conv2)
        self.activation = deepcopy(source.nonlinearity)
        self.dropout = deepcopy(source.dropout)
        self.shortcut = None if isinstance(source.conv_shortcut, nn.Identity) else CanvasConv(source.conv_shortcut)

    def forward(self, value, mask):
        value = value * mask
        shortcut = value if self.shortcut is None else self.shortcut(value, mask)
        hidden = self.conv1(self.activation(self.norm1(value)), mask)
        hidden = self.conv2(self.dropout(self.activation(self.norm2(hidden))), mask)
        return (hidden + shortcut) * mask


class CanvasResample(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.upsample = source.mode.startswith("upsample")
        if source.mode not in ("upsample2d", "upsample3d", "downsample2d", "downsample3d"):
            raise ValueError("unsupported canvas VAE resampling mode")
        resize = source.resample[0]
        self.resize = SpatialZeroPad(resize.padding) if isinstance(resize, nn.ZeroPad2d) else deepcopy(resize)
        self.conv = deepcopy(source.resample[1])

    def forward(self, value, input_mask, output_mask):
        image = value * input_mask
        image = self.resize(image)
        if self.upsample:
            image = image * output_mask
        # Downsampling preserves the original right/bottom ZeroPad2d before
        # stride2 Conv2d. It does not replace this with symmetric padding.
        return self.conv(image) * output_mask


class SpatialDownShortcut(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.factor_s, self.factor_t = source.factor_s, source.factor_t
        self.out_channels, self.group_size = source.out_channels, source.group_size

    def forward(self, value):
        batch, channels, height, width = value.shape
        s = self.factor_s
        blocks = value.reshape(batch, channels, height // s, s, width // s, s)
        blocks = blocks.permute(0, 1, 3, 5, 2, 4).reshape(batch, channels, s * s, height // s, width // s)
        # Native single-frame temporal downsampling prepends zero frames.
        if self.factor_t > 1:
            blocks = F.pad(blocks, (0, 0, 0, 0, (self.factor_t - 1) * s * s, 0))
        return blocks.reshape(batch, self.out_channels, self.group_size, height // s, width // s).mean(2)


class SpatialUpShortcut(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.factor_s, self.factor_t = source.factor_s, source.factor_t
        self.out_channels, self.repeats = source.out_channels, source.repeats

    def forward(self, value):
        batch, _, height, width = value.shape
        s = self.factor_s
        repeated = value.repeat_interleave(self.repeats, dim=1)
        blocks = repeated.reshape(batch, self.out_channels, self.factor_t, s, s, height, width)
        # first_chunk=True retains only the last duplicate of the single frame.
        blocks = blocks[:, :, self.factor_t - 1]
        return blocks.permute(0, 1, 4, 2, 5, 3).reshape(batch, self.out_channels, height * s, width * s)


class CanvasDownBlock(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.blocks = nn.ModuleList([CanvasResidual(block) for block in source.resnets])
        self.shortcut = SpatialDownShortcut(source.avg_shortcut)
        self.down = CanvasResample(source.downsampler) if source.downsampler is not None else None

    def forward(self, value, input_mask, output_mask):
        value = value * input_mask
        shortcut = self.shortcut(value) * output_mask
        for block in self.blocks:
            value = block(value, input_mask)
        if self.down is not None:
            value = self.down(value, input_mask, output_mask)
        return (value + shortcut) * output_mask


class CanvasUpBlock(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.blocks = nn.ModuleList([CanvasResidual(block) for block in source.resnets])
        self.shortcut = SpatialUpShortcut(source.avg_shortcut) if source.avg_shortcut is not None else None
        self.up = CanvasResample(source.upsampler) if source.upsampler is not None else None

    def forward(self, value, input_mask, output_mask):
        value = value * input_mask
        shortcut = self.shortcut(value) * output_mask if self.shortcut is not None else None
        for block in self.blocks:
            value = block(value, input_mask)
        if self.up is not None:
            value = self.up(value, input_mask, output_mask)
        if shortcut is not None:
            value = value + shortcut
        return value * output_mask


class CanvasAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        from xhquant.nn import FlashAttention, Gather

        self.channels = source.dim
        self.physical_channels = max(64, self.channels)
        if self.physical_channels not in (64, 128, 256, 512, 768, 1152):
            raise ValueError("unsupported VAE single-head FlashAttention width")
        self.norm = SpatialChannelNormalize(source.norm)
        self.qkv = deepcopy(source.to_qkv)
        self.projection = deepcopy(source.proj)
        self.attention = FlashAttention(num_heads=1, is_causal=False, scale=self.channels**-0.5)
        self.gather = Gather(axis=2)

    def forward(self, value, mask, order, restore, length):
        batch, channels, height, width = value.shape
        image = value * mask
        qkv = self.qkv(self.norm(image)) * mask
        q, k, v = qkv.reshape(batch, 1, 3 * channels, height * width).transpose(-1, -2).chunk(3, -1)
        q, k, v = (self.gather(part, order) for part in (q, k, v))
        q, k, v = (F.pad(part, (0, self.physical_channels - self.channels)) for part in (q, k, v))
        attended = self.attention(q, k, v, current_input_length=length)[..., : self.channels]
        attended = self.gather(attended, restore).squeeze(1).transpose(1, 2).reshape(batch, channels, height, width)
        attended = attended * mask
        output = self.projection(attended) * mask
        return (value + output) * mask


class CanvasMid(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.blocks = nn.ModuleList([CanvasResidual(block) for block in source.resnets])
        self.attentions = nn.ModuleList([CanvasAttention(attention) for attention in source.attentions])

    def forward(self, value, mask, order, restore, length):
        value = self.blocks[0](value, mask)
        for index, attention in enumerate(self.attentions):
            value = attention(value, mask, order, restore, length)
            value = self.blocks[index + 1](value, mask)
        return value


def _validate_source(source):
    if source.config.patch_size is not None or not source.config.is_residual or len(source.config.dim_mult) != 5:
        raise ValueError("canvas VAE requires the Qwen-Image-2.1 residual, patch_size=None, spatial16 architecture")


class CanvasVaeEncode(nn.Module):
    def __init__(self, source):
        super().__init__()
        _validate_source(source)
        encoder = source.encoder
        self.conv_in = CanvasConv(encoder.conv_in)
        self.down = nn.ModuleList([CanvasDownBlock(block) for block in encoder.down_blocks])
        self.mid = CanvasMid(encoder.mid_block)
        self.norm = SpatialChannelNormalize(encoder.norm_out)
        self.activation = deepcopy(encoder.nonlinearity)
        self.conv_out = CanvasConv(encoder.conv_out)
        self.projection = CanvasConv(source.quant_conv)
        self.latent_channels = source.config.z_dim

    def forward(
        self, image, mask_1, mask_2, mask_4, mask_8, mask_16, attention_order, attention_restore, attention_length
    ):
        masks = (mask_1, mask_2, mask_4, mask_8, mask_16)
        value = self.conv_in(image, mask_1)
        for index, block in enumerate(self.down):
            value = block(value, masks[index], masks[min(index + 1, 4)])
        value = self.mid(value, mask_16, attention_order, attention_restore, attention_length)
        value = self.conv_out(self.activation(self.norm(value)), mask_16)
        return self.projection(value, mask_16)[:, : self.latent_channels]


class CanvasVaeDecode(nn.Module):
    def __init__(self, source):
        super().__init__()
        _validate_source(source)
        decoder = source.decoder
        self.projection = CanvasConv(source.post_quant_conv)
        self.conv_in = CanvasConv(decoder.conv_in)
        self.mid = CanvasMid(decoder.mid_block)
        self.up = nn.ModuleList([CanvasUpBlock(block) for block in decoder.up_blocks])
        self.norm = SpatialChannelNormalize(decoder.norm_out)
        self.activation = deepcopy(decoder.nonlinearity)
        self.conv_out = CanvasConv(decoder.conv_out)

    def forward(
        self, latent, mask_1, mask_2, mask_4, mask_8, mask_16, attention_order, attention_restore, attention_length
    ):
        masks = (mask_1, mask_2, mask_4, mask_8, mask_16)
        value = self.conv_in(self.projection(latent, mask_16), mask_16)
        value = self.mid(value, mask_16, attention_order, attention_restore, attention_length)
        for index, block in enumerate(self.up):
            value = block(value, masks[max(4 - index, 0)], masks[max(3 - index, 0)])
        return self.conv_out(self.activation(self.norm(value)), mask_1).clamp(-1, 1) * mask_1


class CanvasVaeSession:
    """Host padding, per-request masks and cropping for one published canvas graph."""

    def __init__(self, session, canvas_hw, *, encode, token_capacity):
        self.session, self.canvas_hw, self.encode, self.token_capacity = (
            session,
            tuple(canvas_hw),
            encode,
            token_capacity,
        )

    def __call__(self, value):
        if value.ndim != 4 or value.shape[0] != 1:
            raise ValueError("image VAE requires a batch-one four-dimensional tensor")
        height, width = value.shape[-2:]
        actual = (height, width) if self.encode else (height * 16, width * 16)
        if actual[0] * actual[1] // 256 > self.token_capacity:
            raise ValueError("VAE input exceeds the published token capacity")
        inputs = spatial_inputs(actual, self.canvas_hw, device=value.device, dtype=value.dtype)
        factor = 1 if self.encode else 16
        padded = F.pad(value, (0, self.canvas_hw[1] // factor - width, 0, self.canvas_hw[0] // factor - height))
        output = self.session(padded, *inputs)
        out_h, out_w = (height // 16, width // 16) if self.encode else actual
        return output[..., :out_h, :out_w].clone()
