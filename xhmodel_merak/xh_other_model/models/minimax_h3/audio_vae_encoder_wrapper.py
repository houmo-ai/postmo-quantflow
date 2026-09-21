import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
    MiniMaxH3AudioAttnProjection,
    MiniMaxH3AudioCausalAttention,
    MiniMaxH3AudioEncoder,
    MiniMaxH3AudioEncoderBlock,
    MiniMaxH3AudioGeGluMlp,
    MiniMaxH3AudioResidualUnit,
    MiniMaxH3AudioSnake1d,
)
from diffusers.models.modeling_utils import get_parameter_dtype

from xhquant.utils.registry.dynamic_module import DynamicModule


class _MiniMaxH3AudioSnake1d(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * hidden_states).pow(2)


class _MiniMaxH3AudioResidualUnit(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.block(hidden_states)
        pad = (hidden_states.shape[-1] - residual.shape[-1]) // 2
        if pad > 0:
            hidden_states = hidden_states[..., pad:-pad]
        return hidden_states + residual


class _MiniMaxH3AudioEncoderBlock(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class _MiniMaxH3AudioEncoder(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class _MiniMaxH3AudioGeGluMlp(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.act(self.w0(hidden_states)) * self.w1(hidden_states)
        return self.w2(hidden_states)


class _MiniMaxH3AudioCausalAttention(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.scale = self.head_dim**-0.5
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = F.linear(
            hidden_states,
            self.qkv.weight,
            torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        )
        query, key, value = (
            qkv.reshape(batch_size, sequence_length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        )
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=hidden_states.device,
        ).triu(1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        hidden_states = torch.matmul(attention, value).transpose(1, 2)
        hidden_states = torch.mean(hidden_states, dim=2)
        hidden_states = F.adaptive_avg_pool1d(hidden_states, self.out_dim)
        return self.proj(hidden_states)


class _MiniMaxH3AudioAttnProjection(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(self.norm3(hidden_states)) + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


_AUDIO_ENCODER_WRAPPERS = {
    MiniMaxH3AudioSnake1d: _MiniMaxH3AudioSnake1d,
    MiniMaxH3AudioResidualUnit: _MiniMaxH3AudioResidualUnit,
    MiniMaxH3AudioEncoderBlock: _MiniMaxH3AudioEncoderBlock,
    MiniMaxH3AudioEncoder: _MiniMaxH3AudioEncoder,
    MiniMaxH3AudioGeGluMlp: _MiniMaxH3AudioGeGluMlp,
    MiniMaxH3AudioCausalAttention: _MiniMaxH3AudioCausalAttention,
    MiniMaxH3AudioAttnProjection: _MiniMaxH3AudioAttnProjection,
}


def wrap_audio_vae_encoder(audio_vae: nn.Module) -> nn.Module:
    modules = list(audio_vae.encoder.named_modules()) + list(audio_vae.pre_block.named_modules())
    for _, module in modules:
        dynamic_cls = _AUDIO_ENCODER_WRAPPERS.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            dynamic_cls.convert(module)
    return audio_vae


class MiniMaxH3AudioVAEEncoderWrapper(nn.Module):
    def __init__(self, audio_vae: nn.Module):
        super().__init__()
        self.audio_vae = wrap_audio_vae_encoder(audio_vae)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        right_pad = math.ceil(waveform.shape[-1] / self.audio_vae.hop_length) * self.audio_vae.hop_length
        right_pad -= waveform.shape[-1]
        if right_pad > 0:
            waveform = F.pad(waveform, (0, right_pad))
        encoder_dtype = get_parameter_dtype(self.audio_vae.encoder)
        hidden_states = self.audio_vae.encoder(waveform.to(encoder_dtype))
        hidden_states = self.audio_vae.pre_block(hidden_states.transpose(1, 2)).transpose(1, 2)
        mean = self.audio_vae.mean_proj(hidden_states)
        log_std = self.audio_vae.logs_proj(hidden_states)
        if encoder_dtype != torch.float32:
            mean, log_std = mean.float(), log_std.float()
        return mean, log_std


__all__ = ["MiniMaxH3AudioVAEEncoderWrapper", "wrap_audio_vae_encoder"]
