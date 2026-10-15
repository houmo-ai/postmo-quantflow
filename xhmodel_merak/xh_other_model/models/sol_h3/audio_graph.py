# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""LTX audio encoder with static normalization and no patchifier objects in graph."""

from __future__ import annotations

from torch import nn

from .conditioning import validate_fp16_constants


class AudioEncodeGraph(nn.Module):
    def __init__(self, source):
        super().__init__()
        from . import source_io

        source = source_io.prepare_audio_encoder(source)
        self.input, self.down, self.mid = source.conv_in, source.down, source.mid
        self.norm, self.activation, self.output = source.norm_out, source.non_linearity, source.conv_out
        self.layers = source.num_res_blocks
        self.channels = 8
        for name, field in (("mean", "mean-of-means"), ("std", "std-of-means")):
            value = source.per_channel_statistics.get_buffer(field).detach().float().reshape(1, 8, 1, 16)
            validate_fp16_constants(value, name=f"audio latent {name}")
            if name == "std" and bool((value <= 0).any()):
                raise ValueError("audio latent standard deviations must be positive")
            self.register_buffer(name, value)

    def forward(self, spectrogram):
        value = self.input(spectrogram)
        for i, stage in enumerate(self.down):
            for j in range(self.layers):
                value = stage.block[j](value, temb=None)
                if stage.attn:
                    value = stage.attn[j](value)
            if i + 1 < len(self.down):
                value = stage.downsample(value)
        value = self.mid.block_1(value, temb=None)
        value = self.mid.attn_1(value)
        value = self.mid.block_2(value, temb=None)
        value = self.output(self.activation(self.norm(value)))[:, : self.channels]
        return (value - self.mean) / self.std
