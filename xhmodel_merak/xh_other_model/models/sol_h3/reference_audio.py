# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Length-aware native H3 reference audio encoder, without duplicated time gears."""

import json
from math import prod
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .feature_graph import load_component_weights


def load_reference_audio(model_dir):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio

    from ..minimax_h3.audio_vae_decoder_wrapper import _materialize_weight_norm

    root = Path(model_dir) / "audio_vae"
    config = json.loads((root / "config.json").read_text())
    with torch.device("meta"):
        source = AutoencoderKLMiniMaxH3Audio.from_config(config)
    source.decoder = source.dec_in_proj = source.logs_proj = nn.Identity()
    load_component_weights(source, root, "", weights_name="diffusion_pytorch_model.safetensors")
    _materialize_weight_norm(source)
    return source.eval().requires_grad_(False), config


class _MaskedConv(nn.Module):
    def __init__(self, conv, capacity, factor):
        super().__init__()
        self.conv = conv
        stride, padding, dilation, kernel = (
            value[0] for value in (conv.stride, conv.padding, conv.dilation, conv.kernel_size)
        )

        def output(length):
            return (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

        if factor % stride or output(factor) != factor // stride or output(capacity) != capacity // stride:
            raise ValueError("reference audio convolution must preserve integral latent frame lengths")
        self.capacity, self.factor = capacity // stride, factor // stride

    def forward(self, value, frames):
        value = self.conv(value)
        mask = frames.unsqueeze(-1).expand(1, 1, -1, self.factor).flatten(2)
        return value * mask


class _MaskedStack(nn.Module):
    def __init__(self, sequence, capacity, factor):
        super().__init__()
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
            MiniMaxH3AudioEncoderBlock,
            MiniMaxH3AudioResidualUnit,
            MiniMaxH3AudioSnake1d,
        )

        layers = []
        for layer in sequence:
            if isinstance(layer, nn.Conv1d):
                layer = _MaskedConv(layer, capacity, factor)
            elif isinstance(layer, MiniMaxH3AudioResidualUnit):
                layer = _MaskedResidual(layer.block, capacity, factor)
            elif isinstance(layer, MiniMaxH3AudioEncoderBlock):
                layer = _MaskedStack(layer.block, capacity, factor)
            elif not isinstance(layer, MiniMaxH3AudioSnake1d):
                raise ValueError(f"unsupported reference audio encoder layer: {type(layer).__name__}")
            if isinstance(layer, (_MaskedConv, _MaskedStack, _MaskedResidual)):
                capacity, factor = layer.capacity, layer.factor
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.capacity, self.factor = capacity, factor

    def forward(self, value, frames):
        for layer in self.layers:
            if isinstance(layer, (_MaskedConv, _MaskedStack, _MaskedResidual)):
                value = layer(value, frames)
            else:
                value = layer(value)
        return value


class _MaskedResidual(nn.Module):
    def __init__(self, sequence, capacity, factor):
        super().__init__()
        self.block = _MaskedStack(sequence, capacity, factor)
        self.capacity, self.factor = self.block.capacity, self.block.factor
        if (self.capacity, self.factor) != (capacity, factor):
            raise ValueError("reference audio residual must preserve temporal length")

    def forward(self, value, frames):
        return value + self.block(value, frames)


class ReferenceAudioGraph(nn.Module):
    def __init__(self, source, max_frames):
        super().__init__()
        from ..minimax_h3.audio_vae_decoder_wrapper import _materialize_weight_norm
        from .h3_vae_wrappers import wrap_audio_vae_encoder

        if type(max_frames) is not int or max_frames <= 0:
            raise ValueError("reference audio max_frames must be positive")
        _materialize_weight_norm(source)
        self.hop_length, self.max_frames = source.hop_length, max_frames
        self.encoder = _MaskedStack(source.encoder.block, max_frames * self.hop_length, self.hop_length)
        # Inspect native containers before DynamicModule changes their classes;
        # the reused Snake/projection objects still receive the existing wrappers.
        source = wrap_audio_vae_encoder(source)
        self.pre_block, self.mean_proj = source.pre_block, source.mean_proj

    def forward(self, waveform, frame_mask):
        mask = frame_mask.unsqueeze(-1).expand(1, 1, -1, self.hop_length).flatten(2)
        hidden = self.encoder(waveform * mask, frame_mask)
        hidden = self.pre_block(hidden.transpose(1, 2)).transpose(1, 2)
        mean = self.mean_proj(hidden)
        return mean * frame_mask


class _MeanPosterior:
    def __init__(self, mean):
        self.mean = mean

    def mode(self):
        return self.mean


class ReferenceAudioAdapter:
    """The native reference path consumes posterior.mode(), never a sample."""

    def __init__(self, config, invoke, *, max_frames):
        from types import SimpleNamespace

        self.config = SimpleNamespace(**config)
        self.hop_length = prod(config["encoder_rates"])
        self.max_frames, self.invoke = max_frames, invoke

    def encode(self, waveform, return_dict=True):
        from types import SimpleNamespace

        if waveform.ndim != 3 or waveform.shape[:2] != (2, 1) or waveform.shape[-1] <= 0:
            raise ValueError("reference audio expects two mono batches [2, 1, samples]")
        count = (waveform.shape[-1] + self.hop_length - 1) // self.hop_length
        if count > self.max_frames:
            raise ValueError("reference audio exceeds the exported frame capacity")
        padded = F.pad(waveform.float(), (0, self.max_frames * self.hop_length - waveform.shape[-1]))
        mask = (torch.arange(self.max_frames) < count).float().reshape(1, 1, -1)
        mean = self.invoke(padded, mask)[:, :, :count].float()
        posterior = _MeanPosterior(mean)
        return SimpleNamespace(latent_dist=posterior) if return_dict else (posterior,)
