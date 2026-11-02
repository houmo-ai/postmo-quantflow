# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# Copyright 2026 HOUMO AI (graph adaptation and host integration)
# SPDX-License-Identifier: Apache-2.0
# Adapted from diffusers 0.41.0 autoencoder_kl_minimax_h3.py.
# Source: https://github.com/huggingface/diffusers
# See licenses/diffusers/LICENSE.
"""Native H3 ViT decoder tiles, with Diffusers owning temporal/spatial stitching.

Graphs consume raw (denormalized) BCTHW latents and return ImageNet-normalized
pixels. ``decode`` preserves that native contract; ``forward`` converts pipeline
normalized latents to float32 RGB in [0, 1]. No LTX or MiniMax wrapper is involved.
"""

import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .feature_graph import load_component_weights
from .precision import StableRMSNorm


def _shell(config):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    with torch.device("meta"):
        vae = AutoencoderKLMiniMaxH3.from_config(config)
    vae.encoder = vae.quant_conv = nn.Identity()
    return vae


def load_native_video_vae(model_dir):
    """Load decoder-only checkpoint tensors; never materialize the CNN encoder."""
    root = Path(model_dir) / "vae"
    config = json.loads((root / "config.json").read_text())
    vae = _shell(config)
    load_component_weights(vae, root, "", weights_name="diffusion_pytorch_model.safetensors")
    # Rotary frequencies are non-persistent buffers, absent from the checkpoint.
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed

    vae.decoder.rope = MiniMaxH3VideoRotaryPosEmbed(
        int(config["decoder_attention_head_dim"] * config.get("decoder_rope_dim_ratio", 0.75)),
        theta=config.get("decoder_rope_theta", 100.0),
    )
    return vae.eval().requires_grad_(False), config


def _norm(source, sdk_ops):
    if source.weight is None:
        source = nn.RMSNorm(source.normalized_shape, eps=source.eps)
    return StableRMSNorm(source, sdk_ops=sdk_ops, bounded_reciprocal=False)


class _Attention(nn.Module):
    def __init__(self, source, sdk_ops):
        super().__init__()
        self.to_q, self.to_k, self.to_v = source.to_q, source.to_k, source.to_v
        self.output = source.to_out[0]
        self.heads, self.head_dim = source.heads, source.dim_head
        self.q_norm, self.k_norm = _norm(source.norm_q, sdk_ops), _norm(source.norm_k, sdk_ops)
        self.sdk_ops = sdk_ops
        if sdk_ops:
            from xhquant.nn import Rope

            from ...common.attention import CompilerAttention as FlashAttention

            self.attention = FlashAttention(num_heads=self.heads, is_causal=False, scale=self.head_dim**-0.5)
            self.rope = Rope()

    def forward(self, value, rotary_emb):
        def heads(projection):
            return projection(value).reshape(value.shape[0], -1, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(self.to_q), heads(self.to_k), heads(self.to_v)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = (v.transpose(1, 2).to(q.dtype) for v in rotary_emb)
        width = cos.shape[-1]

        def rotate(x):
            part = x[..., :width]
            if self.sdk_ops:
                part = self.rope(part, cos, sin)
            else:
                a, b = part.chunk(2, -1)
                part = part * cos + torch.cat((-b, a), -1) * sin
            return torch.cat((part, x[..., width:]), -1)

        q, k = rotate(q), rotate(k)
        attended = self.attention(q, k, v) if self.sdk_ops else F.scaled_dot_product_attention(q, k, v)
        return self.output(attended.transpose(1, 2).flatten(2))


class NativeVideoDecodeGraph(nn.Module):
    """One exact latent tile; Conv3d lowering is owned by the SDK SliceConvAdd pass.

    Native register/zero tokens, patch rearrangement, FP32
    residuals, and epsilon are retained. Only attention/norm graph primitives
    are substituted. Input dtype follows ``post_quant_conv.weight.dtype``.
    """

    def __init__(self, source, tile_shape, *, sdk_ops=False):
        super().__init__()
        self.tile_shape = _shape(tile_shape)
        self.post_quant_conv = copy.deepcopy(source.post_quant_conv)
        self.decoder = copy.deepcopy(source.decoder)
        grids = [2 * (torch.arange(n, dtype=torch.float32) + 0.5) / n - 1 for n in self.tile_shape]
        positions = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).reshape(1, -1, 3)
        positions = torch.cat((positions, positions.new_zeros(1, self.decoder.num_register_tokens + 1, 3)), dim=1)
        cos, sin = self.decoder.rope(positions)
        self.register_buffer("rotary_cos", cos)
        self.register_buffer("rotary_sin", sin)
        for block in self.decoder.transformer_blocks:
            block.attn = _Attention(block.attn, sdk_ops)
            block.norm1, block.norm2 = _norm(block.norm1, sdk_ops), _norm(block.norm2, sdk_ops)

    def forward(self, latent):
        if tuple(latent.shape[2:]) != self.tile_shape:
            raise ValueError("native video tile does not match the exported shape")
        value = self.post_quant_conv(latent)
        batch, channels, frames, height, width = value.shape
        value = self.decoder.proj_in(value.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels))
        patches = frames * height * width
        value = torch.cat(
            (value, self.decoder.register_tokens.expand(batch, -1, -1).to(value.dtype), torch.zeros_like(value[:, :1])),
            dim=1,
        )
        for block in self.decoder.transformer_blocks:
            value = block(value, (self.rotary_cos, self.rotary_sin))
        value = self.decoder.norm_out(value.float()).to(self.decoder.proj_out.weight.dtype)
        value = self.decoder.proj_out(value)[:, :patches]
        spatial, temporal = self.decoder.patch_size, self.decoder.patch_size_t
        value = value.reshape(batch, frames, height, width, self.decoder.out_channels, temporal, spatial, spatial)
        return value.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            batch, self.decoder.out_channels, frames * temporal, height * spatial, width * spatial
        )


def _shape(shape):
    if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("native video latent shape must contain positive integer T, H, W")
    return tuple(shape)


class _TileSession(nn.Module):
    def __init__(self, invoke, dtype):
        super().__init__()
        self.invoke, self.dtype = invoke, dtype

    def forward(self, value):
        return self.invoke(tuple(value.shape[2:]), value.to(self.dtype)).float()


class NativeVideoDecodeAdapter(nn.Module):
    """Weight-free native chunk scheduling around ``invoke((T,H,W), latent)``."""

    def __init__(self, config, invoke, *, tile_size=(256, 256), overlap=(64, 64), dtype=torch.float16):
        super().__init__()
        self.vae = _shell(config)
        self.vae.decoder = _TileSession(invoke, dtype)
        self.vae.post_quant_conv = nn.Identity()
        ratio = self.vae.spatial_compression_ratio
        if (
            len(tile_size) != 2
            or len(overlap) != 2
            or any(type(n) is not int or n <= 0 or n % ratio for n in tile_size)
            or any(type(n) is not int or n < 0 or n % ratio for n in overlap)
            or any(o >= t for o, t in zip(overlap, tile_size, strict=True))
        ):
            raise ValueError("native video tiles/overlaps must be latent-aligned with positive stride")
        self.vae.enable_tiling(tile_size[0], tile_size[1], overlap[0], overlap[1])
        self.vae.tile_sample_min_overlap_height, self.vae.tile_sample_min_overlap_width = overlap
        self.config, self.dtype = self.vae.config, dtype

    def tile_shapes(self, latent_shapes):
        """Enumerate exact native temporal and spatial gears without allocating pixels."""
        shapes = set()
        ratio, temporal = self.vae.spatial_compression_ratio, self.vae.temporal_compression_ratio

        def capture(shape, value):
            shapes.add(shape)
            t, h, w = shape
            return value.new_empty((1, self.config.out_channels, t * temporal, h * ratio, w * ratio))

        planner = NativeVideoDecodeAdapter(
            dict(self.config),
            capture,
            tile_size=(self.vae.tile_sample_min_height, self.vae.tile_sample_min_width),
            overlap=(self.vae.tile_sample_min_overlap_height, self.vae.tile_sample_min_overlap_width),
        )
        for shape in latent_shapes:
            planner.decode(torch.empty((1, self.config.latent_channels, *_shape(shape)), device="meta"))
        return sorted(shapes)

    def decode(self, latent, return_dict=True):
        from diffusers.models.autoencoders.vae import DecoderOutput

        if latent.ndim != 5 or latent.shape[:2] != (1, self.config.latent_channels):
            raise ValueError("native video decoder requires batch-one BCTHW latents")
        _shape(tuple(latent.shape[2:]))
        if latent.shape[2] + self.config.token_drop <= self.vae.tokens_chunk_size:
            raise ValueError("native video decoder needs enough latent frames for a complete temporal chunk")
        # _decode avoids the public native decode's dtype lookup on Identity.
        pixels = self.vae._decode(latent.float())
        return DecoderOutput(sample=pixels) if return_dict else (pixels,)

    def forward(self, normalized_latent):
        mean = torch.tensor(self.config.latents_mean, device=normalized_latent.device, dtype=torch.float32).view(
            1, -1, 1, 1, 1
        )
        std = torch.tensor(self.config.latents_std, device=normalized_latent.device, dtype=torch.float32).view(
            1, -1, 1, 1, 1
        )
        pixels = self.decode(normalized_latent.float() * std + mean, return_dict=False)[0]
        pixel_mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        pixel_std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        return (pixels * pixel_std + pixel_mean).clamp(0, 1)
