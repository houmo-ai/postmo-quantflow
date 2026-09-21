import math

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
    MiniMaxH3VideoAttention,
    MiniMaxH3VideoRotaryPosEmbed,
    MiniMaxH3VideoTransformerBlock,
    MiniMaxH3VideoViTDecoder3d,
)

from xhquant import nn as xhnn
from xhquant.utils.registry.dynamic_module import DynamicModule


class _MiniMaxH3VideoRotaryPosEmbed(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


class _MiniMaxH3VideoAttention(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.scale = self.dim_head**-0.5
        self.rotary_dim = self.dim_head * 3 // 4
        self.qkv_reshape = xhnn.Reshape()
        self.norm_q_h = xhnn.RMSNorm(self.norm_q.normalized_shape[0], eps=self.norm_q.eps)
        self.norm_k_h = xhnn.RMSNorm(self.norm_k.normalized_shape[0], eps=self.norm_k.eps)
        self.norm_q_h.weight = (
            nn.Parameter((self.norm_q.weight.data))
            if self.norm_q.weight is not None
            else nn.Parameter(torch.ones(self.norm_q.normalized_shape[0], dtype=torch.float16))
        )
        self.norm_k_h.weight = (
            nn.Parameter((self.norm_k.weight.data))
            if self.norm_k.weight is not None
            else nn.Parameter(torch.ones(self.norm_k.normalized_shape[0], dtype=torch.float16))
        )

        self.hidden_rotary_slice = xhnn.Slice([0], [self.rotary_dim], [-1], [1])
        self.hidden_pass_slice = xhnn.Slice([self.rotary_dim], [self.dim_head], [-1], [1])
        self.rope = xhnn.Rope()
        self.output_concat = xhnn.Concat(dim=-1)

        del self.norm_q, self.norm_k
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length = hidden_states.shape[:2]
        qkv_shape = (batch_size, sequence_length, self.heads, self.dim_head)
        query = self.qkv_reshape(self.to_q(hidden_states), qkv_shape)
        key = self.qkv_reshape(self.to_k(hidden_states), qkv_shape)
        value = self.qkv_reshape(self.to_v(hidden_states), qkv_shape)
        query = self.norm_q_h(query)
        key = self.norm_k_h(key)

        if rotary_emb is not None:
            cos, sin = rotary_emb
            query = self.output_concat(
                self.rope(self.hidden_rotary_slice(query), cos, sin),
                self.hidden_pass_slice(query),
            )
            key = self.output_concat(
                self.rope(self.hidden_rotary_slice(key), cos, sin),
                self.hidden_pass_slice(key),
            )

        attention = torch.matmul(query.transpose(1, 2), key.transpose(1, 2).transpose(-1, -2)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        hidden_states = torch.matmul(attention, value.transpose(1, 2))
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        return self.to_out[0](hidden_states)


class _MiniMaxH3VideoTransformerBlock(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.norm1_h = xhnn.RMSNorm(self.norm1.normalized_shape[0], eps=self.norm1.eps)
        self.norm2_h = xhnn.RMSNorm(self.norm2.normalized_shape[0], eps=self.norm2.eps)
        self.norm1_h.weight = (
            nn.Parameter(self.norm1.weight.data)
            if self.norm1.weight is not None
            else nn.Parameter(torch.ones(self.norm1.normalized_shape[0], dtype=torch.float16))
        )
        self.norm2_h.weight = (
            nn.Parameter(self.norm2.weight.data)
            if self.norm2.weight is not None
            else nn.Parameter(torch.ones(self.norm2.normalized_shape[0], dtype=torch.float16))
        )
        del self.norm1, self.norm2
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1_h(hidden_states)
        hidden_states = hidden_states + self.attn(norm_hidden_states, rotary_emb) * self.scale1
        norm_hidden_states = self.norm2_h(hidden_states)
        return hidden_states + self.ff(norm_hidden_states) * self.scale2


class _MiniMaxH3VideoFeedForward(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        inner_dim = self.net[0].proj.out_features // 2
        self.hidden_slice = xhnn.Slice([0], [inner_dim], [-1], [1])
        self.gate_slice = xhnn.Slice([inner_dim], [2 * inner_dim], [-1], [1])
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        swiglu = self.net[0]
        projected = swiglu.proj(hidden_states)
        hidden = self.hidden_slice(projected)
        gate = self.gate_slice(projected)
        hidden_states = hidden * swiglu.activation(gate)
        return self.net[2](hidden_states)


class _MiniMaxH3VideoViTDecoder3d(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape  # torch.Size([1, 24, 7, 16, 16])
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
            batch_size, num_frames * height * width, num_channels
        )
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        hidden_states = torch.cat((hidden_states, register_tokens, torch.zeros_like(hidden_states[:, :1])), dim=1)

        rotary_emb = (rotary_cos, rotary_sin)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)
        hidden_states = self.proj_out(self.norm_out(hidden_states))[:, :num_patches]

        patch_size, patch_size_t = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.view(
            batch_size,
            num_frames,
            height,
            width,
            self.out_channels,
            patch_size_t,
            patch_size,
            patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        )


def build_vae_decoder_rotary_emb(
    decoder: nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, num_frames, height, width = hidden_states.shape
    grids = [
        2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
        for size in (num_frames, height, width)
    ]
    position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
    suffix_ids = position_ids.new_zeros((batch_size, decoder.num_register_tokens + 1, 3))
    return decoder.rope(torch.cat((position_ids, suffix_ids), dim=1))


_DECODER_WRAPPERS = {
    MiniMaxH3VideoRotaryPosEmbed: _MiniMaxH3VideoRotaryPosEmbed,
    MiniMaxH3VideoAttention: _MiniMaxH3VideoAttention,
    MiniMaxH3VideoTransformerBlock: _MiniMaxH3VideoTransformerBlock,
    MiniMaxH3VideoViTDecoder3d: _MiniMaxH3VideoViTDecoder3d,
}


def wrap_vae_decoder(vae: nn.Module) -> nn.Module:
    for block in vae.decoder.transformer_blocks:
        if type(block.ff) is FeedForward:
            _MiniMaxH3VideoFeedForward.convert(block.ff)
    for _, module in list(vae.decoder.named_modules()):
        dynamic_cls = _DECODER_WRAPPERS.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            dynamic_cls.convert(module)
    return vae


class MiniMaxH3VAEDecoderWrapper(nn.Module):
    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = wrap_vae_decoder(vae)

    def forward(
        self,
        latents: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.vae.post_quant_conv(latents)
        return self.vae.decoder(hidden_states, rotary_cos, rotary_sin)


__all__ = ["MiniMaxH3VAEDecoderWrapper", "build_vae_decoder_rotary_emb", "wrap_vae_decoder"]
