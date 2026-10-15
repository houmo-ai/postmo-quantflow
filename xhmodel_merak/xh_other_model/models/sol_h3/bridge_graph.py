# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Static x2 upscaler segments and frozen 37-to-17-frame latent alignment."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .conditioning import validate_fp16_constants


def upscaler_segments(frames=37, *, chunk=32, overlap=5):
    """The author's temporal chunk windows and their complementary blend ramps."""
    result = []
    for start in range(0, frames, chunk):
        end = min(frames, start + chunk)
        output_start, output_end = max(0, start - overlap), min(frames, end + overlap)
        low, high = max(0, output_start - overlap), min(frames + 2 * overlap, output_end + overlap)
        crop = output_start + overlap - low
        weights = torch.ones(output_end - output_start, dtype=torch.float32)
        left, right = start - output_start, output_end - end
        if left:
            weights[:left] = torch.arange(1, left + 1) / (left + 1)
        if right:
            weights[-right:] = torch.arange(right, 0, -1) / (right + 1)
        result.append(
            dict(
                input_start=low,
                input_end=high,
                crop_start=crop,
                crop_end=crop + weights.numel(),
                output_start=output_start,
                output_end=output_end,
                weights=weights.tolist(),
            )
        )
    return result


class FixedScaleResidual(nn.Module):
    def __init__(self, source, embedding):
        super().__init__()
        self.input, self.norm, self.output, self.skip = (
            source.in_layers,
            source.out_norm,
            source.out_layers,
            source.skip,
        )
        with torch.no_grad():
            scale, shift = source.emb_layers(embedding).chunk(2, 1)
        scale, shift = (value.detach().float().reshape(1, -1, 1, 1, 1) for value in (1 + scale, shift))
        validate_fp16_constants(scale, name="upscaler scale")
        validate_fp16_constants(shift, name="upscaler shift")
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def forward(self, x):
        return self.skip(x) + self.output(self.norm(self.input(x)) * self.scale + self.shift)


class UpscalerSegment(nn.Module):
    """Temporal interpolation is identity; use equal framewise bilinear resizing."""

    def __init__(self, source):
        super().__init__()
        with torch.no_grad():
            embedding = source.embed(next(source.parameters()).new_ones((1, 1)))

        def blocks(items):
            return nn.ModuleList(
                [FixedScaleResidual(item, embedding) if hasattr(item, "emb_layers") else item for item in items]
            )

        self.input, self.output, self.norm = source.conv_in, source.conv_out, source.norm_out
        self.before, self.after = blocks(source.in_blocks), blocks(source.out_blocks)
        self.static_shape = None
        from xhquant.nn import Resize

        self.resize = Resize(scale_factor=[1, 1, 2, 2], resize_mode="bilinear", align_corners=False)

    def set_static_shape(self, shape):
        self.static_shape = (shape[0], self.input.out_channels, *shape[2:])

    def forward(self, latent):
        x = self.input(latent)
        for block in self.before:
            x = block(x)
        batch, channels, frames, height, width = self.static_shape if self.static_shape is not None else x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        x = self.resize(x)
        x = x.reshape(batch, frames, channels, height * 2, width * 2).permute(0, 2, 1, 3, 4)
        for block in self.after:
            x = block(x)
        return self.output(F.silu(self.norm(x)))


class TemporalAlignment(nn.Module):
    def __init__(self, *, pixel_frames=124, slots=3, sdk_ops=False):
        super().__init__()
        chunks = (pixel_frames + 16) // 17
        source = torch.tensor(
            [min(chunk * 17 + token * 4, pixel_frames - 1) for chunk in range(chunks) for token in range(5)][:-3],
            dtype=torch.float32,
        )
        padded_frames = ((pixel_frames - 1 + 7) // 8) * 8 + 1
        target = torch.arange((padded_frames - 1) // 8 + 1, dtype=torch.float32) * 8
        right = torch.searchsorted(source, target).clamp(max=source.numel() - 1)
        left = (right - 1).clamp(min=0)
        difference = source[right] - source[left]
        alpha = torch.where(difference > 0, (target - source[left]) / difference, torch.zeros_like(target)).clamp(0, 1)
        nearest = (source[:, None] - target[None]).abs().argmin(1)
        packed, occupied = [], []
        for frame in range(target.numel()):
            selected = torch.where(nearest == frame)[0].tolist()
            if len(selected) > slots:
                raise ValueError("frozen temporal pack exceeds the checkpoint's three slots")
            packed.extend(selected + [0] * (slots - len(selected)))
            occupied.extend([1] * len(selected) + [0] * (slots - len(selected)))
        for name, value in (("left", left), ("right", right), ("packed", torch.tensor(packed))):
            self.register_buffer(name, value.to(torch.int32))
        self.register_buffer("alpha", alpha.reshape(1, 1, -1, 1, 1))
        self.register_buffer("occupied", torch.tensor(occupied, dtype=torch.float32).reshape(1, 1, -1, 1, 1))
        self.frames, self.slots = target.numel(), slots
        self.static_shape = None
        self.sdk_ops = sdk_ops
        if sdk_ops:
            from xhquant.nn import Gather

            self.gather = Gather(axis=2)

    def set_static_shape(self, shape):
        self.static_shape = shape

    def select(self, x, indices):
        return self.gather(x, indices) if self.sdk_ops else x.index_select(2, indices.long())

    def forward(self, latent):
        left, right = self.select(latent, self.left), self.select(latent, self.right)
        linear = left + (right - left) * self.alpha
        batch, channels, _, height, width = self.static_shape if self.static_shape is not None else latent.shape
        packed = self.select(latent, self.packed) * self.occupied
        packed = packed.reshape(batch, channels, self.frames, self.slots, height, width)
        packed = packed.permute(0, 3, 1, 2, 4, 5).reshape(batch, self.slots * channels, self.frames, height, width)
        combined = torch.cat((linear, packed), 1)
        channels = (self.slots + 1) * channels
        # Pixel-unshuffle keeps channel, height-offset, width-offset order.
        combined = combined.reshape(batch, channels, self.frames, height // 2, 2, width // 2, 2)
        return combined.permute(0, 1, 4, 6, 2, 3, 5).reshape(batch, channels * 4, self.frames, height // 2, width // 2)


class AdapterResidual(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.norm = source.norm
        self.spatial, self.temporal = source.spatial, source.temporal
        self.projection, self.output = source.in_proj, source.out_proj

    def forward(self, x):
        h = self.temporal(self.spatial(F.silu(self.norm(x))))
        value, gate = self.projection(F.silu(h)).chunk(2, 1)
        return x + self.output(value * F.silu(gate))


class AdapterGraph(nn.Module):
    def __init__(self, source, *, sdk_ops=False):
        super().__init__()
        self.alignment = TemporalAlignment(sdk_ops=sdk_ops)
        self.skip, self.stem = source.skip, source.stem
        self.blocks = nn.ModuleList([AdapterResidual(block) for block in source.blocks])
        self.norm, self.head = source.final_norm, source.head

    def forward(self, upscaled_h3):
        # Preserve the seventeenth adapter output's convolution context before cropping.
        aligned = self.alignment(upscaled_h3)
        residual = self.stem(aligned)
        for block in self.blocks:
            residual = block(residual)
        result = self.skip(aligned) + self.head(F.silu(self.norm(residual)))
        return result[:, :, :16].contiguous()
