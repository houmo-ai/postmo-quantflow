# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Sol-specific H3 VAE export adaptations over the unchanged MiniMax wrappers."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..minimax_h3 import audio_vae_encoder_wrapper as _audio
from ..minimax_h3 import vae_encoder_wrapper as _video
from ..minimax_h3.vae_encoder_wrapper import MiniMaxH3VAEEncoderTileWrapper


class _SolH3VideoCausalConv3d(_video._MiniMaxH3VideoCausalConv3d):
    def set_static_shape(self, shape):
        """For one frame, all causal history taps multiply constant zeros."""
        if isinstance(self.conv, nn.Conv2d):
            if shape[2] != 1:
                raise ValueError("a single-frame causal convolution cannot share a video gear")
            return
        source = self.conv
        if shape[2] != 1 or self.temporal_padding != (source.kernel_size[0] - 1) * source.dilation[0]:
            return
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        # Shape discovery runs in FakeTensorMode. Keep the checkpoint slice real
        # so the ordinary SDK Conv2d quantizer consumes the original coefficients.
        with unset_fake_temporarily():
            conv = nn.Conv2d(
                source.in_channels,
                source.out_channels,
                source.kernel_size[1:],
                stride=source.stride[1:],
                padding=source.padding[1:],
                dilation=source.dilation[1:],
                groups=source.groups,
                bias=source.bias is not None,
                device=source.weight.device,
                dtype=source.weight.dtype,
            )
            conv.weight = nn.Parameter(source.weight[:, :, -1].detach().clone(), requires_grad=False)
            if source.bias is not None:
                conv.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        self.conv = conv

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_padding > 0:
            padding = self.spatial_padding
            hidden_states = F.pad(
                hidden_states,
                (padding, padding, padding, padding, 0, 0),
                mode=self.spatial_padding_mode,
            )
        if isinstance(self.conv, nn.Conv2d):
            return self.conv(hidden_states.squeeze(2)).unsqueeze(2)
        if self.temporal_padding > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0))
        return self.conv(hidden_states)


class _SolH3AudioSnake1d(_audio._MiniMaxH3AudioSnake1d):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sine = torch.sin(self.alpha * hidden_states)
        return hidden_states + (self.alpha + 1e-9).reciprocal() * (sine * sine)


class _SolH3AudioCausalAttention(_audio._MiniMaxH3AudioCausalAttention):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        from ...common.attention import CompilerAttention as FlashAttention

        projection = nn.Linear(
            self.qkv.in_features,
            self.qkv.out_features,
            bias=True,
            device=self.qkv.weight.device,
            dtype=self.qkv.weight.dtype,
        )
        projection.weight = nn.Parameter(self.qkv.weight.detach().clone())
        projection.bias = nn.Parameter(torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)).detach().clone())
        self.qkv = projection
        self.physical_dim = next((dim for dim in (64, 128, 256, 512) if dim >= self.head_dim), None)
        if self.physical_dim is None:
            raise ValueError("H3 audio attention head exceeds supported FlashAttention dimensions")
        self.attention = FlashAttention(num_heads=self.num_heads, is_causal=True, scale=self.head_dim**-0.5)
        # Adaptive pooling acts only on channels. Fold its exact interval-average
        # matrix into the following projection, including overlapping bins.
        pooling = self.proj.weight.new_zeros(self.out_dim, self.head_dim)
        for index in range(self.out_dim):
            start = index * self.head_dim // self.out_dim
            stop = math.ceil((index + 1) * self.head_dim / self.out_dim)
            pooling[index, start:stop] = 1 / (stop - start)
        projection = nn.Linear(
            self.head_dim,
            self.out_dim,
            bias=self.proj.bias is not None,
            device=self.proj.weight.device,
            dtype=self.proj.weight.dtype,
        )
        projection.weight = nn.Parameter(
            (self.proj.weight.detach().float() @ pooling.float()).to(self.proj.weight.dtype)
        )
        if self.proj.bias is not None:
            projection.bias = nn.Parameter(self.proj.bias.detach().clone())
        self.proj = projection
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        packed = qkv.reshape(batch_size, sequence_length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = packed[0], packed[1], packed[2]
        pad = self.physical_dim - self.head_dim
        query, key, value = (F.pad(tensor, (0, pad)) for tensor in (query, key, value))
        hidden_states = self.attention(query, key, value)[..., : self.head_dim].mean(dim=1)
        return self.proj(hidden_states)


def wrap_vae_encoder(vae: nn.Module) -> nn.Module:
    """Specialize only Sol's causal convolutions, reusing the other wrappers."""
    for _, module in list(vae.encoder.named_modules()):
        if type(module) is _video.MiniMaxH3VideoCausalConv3d:
            _SolH3VideoCausalConv3d.convert(module)
    return _video.wrap_vae_encoder(vae)


def wrap_audio_vae_encoder(audio_vae: nn.Module) -> nn.Module:
    """Keep Sol's export-safe activation and attention local to this model."""
    wrappers = {
        _audio.MiniMaxH3AudioSnake1d: _SolH3AudioSnake1d,
        _audio.MiniMaxH3AudioCausalAttention: _SolH3AudioCausalAttention,
    }
    modules = list(audio_vae.encoder.named_modules()) + list(audio_vae.pre_block.named_modules())
    for _, module in modules:
        wrapper = wrappers.get(type(module))
        if wrapper is not None:
            wrapper.convert(module)
    return _audio.wrap_audio_vae_encoder(audio_vae)


class SolH3VAEEncoderWrapper(_video.MiniMaxH3VAEEncoderWrapper):
    def __init__(self, vae, tile_encoder=None, *, wrap_modules=True):
        if wrap_modules:
            vae = wrap_vae_encoder(vae)
        super().__init__(vae, tile_encoder, wrap_modules=False)


__all__ = [
    "MiniMaxH3VAEEncoderTileWrapper",
    "SolH3VAEEncoderWrapper",
    "wrap_audio_vae_encoder",
    "wrap_vae_encoder",
]
