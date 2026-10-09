# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Explicit tensor interfaces for Qwen-Image-2.1 DiT block export (batch 1)."""

import torch
from torch import nn


class BlockCore(nn.Module):
    """Reuse native weights, with modulation and prefix-cache ownership outside the block."""

    def __init__(self, source, clip_fp16=False, *, sdk_ops=False):
        super().__init__()
        self.norm1, self.norm2 = source.img_norm1, source.img_norm2
        self.q, self.k, self.v = source.attn.to_q, source.attn.to_k, source.attn.to_v
        self.out = source.attn.to_out[0]
        self.q_weight, self.k_weight = source.attn.norm_q.weight, source.attn.norm_k.weight
        self.eps = source.attn.norm_q.eps
        self.mlp_gate, self.mlp_in, self.mlp_out = source.img_mlp.gate_layer, source.img_mlp.proj, source.img_mlp.out
        self.heads = source.attn.heads
        self.head_dim = self.q.out_features // self.heads
        self.dim = self.q.in_features
        self.clip_fp16 = clip_fp16
        self.sdk_ops = sdk_ops
        if sdk_ops:
            from xhquant.nn import FlashAttention, RMSNorm, Rope

            self.q_norm = RMSNorm(self.head_dim, source.attn.norm_q.eps)
            self.k_norm = RMSNorm(self.head_dim, source.attn.norm_k.eps)
            self.q_norm.weight = self.q_weight
            self.k_norm.weight = self.k_weight
            self.rotary = Rope()
            self.attention = FlashAttention(num_heads=self.heads, is_causal=False, scale=self.head_dim**-0.5)

    def rms(self, x, weight):
        wide = x.float()
        return (wide * torch.rsqrt((wide * wide).mean(-1, keepdim=True) + self.eps) * weight.float()).type_as(x)

    def rope(self, x, cos, sin):
        if self.sdk_ops:
            # Qwen-Image rotates adjacent pairs; SDK Rope rotates two halves.
            # Convert to halves and restore pairs so the cache ABI is unchanged.
            halves = x.reshape(1, self.heads, -1, self.head_dim // 2, 2).transpose(-1, -2).flatten(-2)
            rotated = self.rotary(halves, torch.cat((cos, cos), -1), torch.cat((sin, sin), -1))
            return rotated.reshape(1, self.heads, -1, 2, self.head_dim // 2).transpose(-1, -2).flatten(-2)
        even, odd = x[..., 0::2].float(), x[..., 1::2].float()
        return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2).type_as(x)

    def project(self, x, scale, cos, sin):
        h = self.norm1(x) * scale
        q = self.q(h).reshape(1, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(h).reshape(1, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(h).reshape(1, -1, self.heads, self.head_dim).transpose(1, 2)
        if self.sdk_ops:
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.rms(q, self.q_weight), self.rms(k, self.k_weight)
        return self.rope(q, cos, sin), self.rope(k, cos, sin), v

    def finish(self, x, q, k, v, gate1, scale2, gate2, mask):
        if self.sdk_ops:
            attended = self.attention(q, k, v, query_kv_range_abs=mask)
        else:
            scores = (q @ k.transpose(-1, -2)) * (self.head_dim**-0.5)
            probs = (scores + mask).softmax(-1)
            attended = probs @ v
        attention = attended.transpose(1, 2).reshape(1, -1, self.dim)
        x = x + gate1 * self.out(attention)
        h = self.norm2(x) * scale2
        h = torch.nn.functional.silu(self.mlp_gate(h)) * self.mlp_in(h)
        y = x + gate2 * self.mlp_out(h)
        return y.clamp(-65504, 65504) if self.clip_fp16 else y


class ExtractBlock(nn.Module):
    def __init__(self, core, prefix_length):
        super().__init__()
        if not isinstance(prefix_length, int) or isinstance(prefix_length, bool) or prefix_length < 0:
            raise ValueError("prefix_length must be a nonnegative integer")
        self.core = core
        self.prefix_length = prefix_length

    def forward(self, x, scale1, gate1, scale2, gate2, cos, sin, mask):
        q, k, v = self.core.project(x, scale1, cos, sin)
        y = self.core.finish(x, q, k, v, gate1, scale2, gate2, mask)
        return y, k[:, :, : self.prefix_length], v[:, :, : self.prefix_length]


class CachedBlock(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x, scale1, gate1, scale2, gate2, cos, sin, mask, prefix_k, prefix_v):
        q, k, v = self.core.project(x, scale1, cos, sin)
        k = torch.cat((prefix_k, k), dim=2)
        v = torch.cat((prefix_v, v), dim=2)
        return self.core.finish(x, q, k, v, gate1, scale2, gate2, mask)


class SharedModulation(nn.Module):
    """Compute the shared four parameters once per time row, before token broadcasting."""

    def __init__(self, modulation):
        super().__init__()
        self.modulation = modulation

    def forward(self, time_embedding):
        scale1, gate1, scale2, gate2 = self.modulation(time_embedding).chunk(4, -1)
        return 1 + scale1, gate1.tanh(), 1 + scale2, gate2.tanh()


class ExtractStack(nn.Module):
    """Full joint sequence; output order is hidden, layer0 K/V, layer1 K/V, ... ."""

    def __init__(self, cores, prefix_length, target_length):
        super().__init__()
        if not cores or target_length <= 0:
            raise ValueError("a nonempty block stack and positive target_length are required")
        self.blocks = nn.ModuleList([ExtractBlock(core, prefix_length) for core in cores])
        self.prefix_length = prefix_length
        self.target_length = target_length

    def expand_rows(self, rows):
        # Native time embedding order: current time first, fixed t=0 second.
        return torch.cat(
            (rows[1:2, None].expand(1, self.prefix_length, -1), rows[:1, None].expand(1, self.target_length, -1)), dim=1
        )

    def forward(self, hidden, scale1, gate1, scale2, gate2, cos, sin, mask):
        parameters = [self.expand_rows(p) for p in (scale1, gate1, scale2, gate2)]
        cache = []
        for block in self.blocks:
            hidden, key, value = block(hidden, *parameters, cos, sin, mask)
            cache.extend((key, value))
        return (hidden, *cache)


class CachedStack(nn.Module):
    """Target-only graph; read one immutable K/V pair per block."""

    def __init__(self, cores):
        super().__init__()
        if not cores:
            raise ValueError("a nonempty block stack is required")
        self.blocks = nn.ModuleList([CachedBlock(core) for core in cores])

    def forward(self, hidden, scale1, gate1, scale2, gate2, cos, sin, mask, prefix_cache):
        parameters = [p[:1, None] for p in (scale1, gate1, scale2, gate2)]
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, *parameters, cos, sin, mask, prefix_cache[2 * index], prefix_cache[2 * index + 1])
        return hidden


class SharedConditioning(nn.Module):
    """Time-embedding rows to all block parameters and the final normalization scale."""

    def __init__(self, source):
        super().__init__()
        self.activation = source.modulation[0]
        self.block_projection = source.modulation[1]
        self.output_projection = source.norm_out.linear

    def forward(self, time_embedding):
        activated = self.activation(time_embedding)
        scale1, gate1, scale2, gate2 = self.block_projection(activated).chunk(4, -1)
        return 1 + scale1, gate1.tanh(), 1 + scale2, gate2.tanh(), 1 + self.output_projection(activated)


class OutputHead(nn.Module):
    """Project target hidden states only; prefix output is never consumed by the scheduler."""

    def __init__(self, source):
        super().__init__()
        self.norm = source.norm_out.norm
        self.projection = source.proj_out

    def forward(self, target_hidden, output_scale):
        return self.projection(self.norm(target_hidden) * output_scale[:1, None])


class TimeConditioning(nn.Module):
    """Time MLP and raw shared projections; Host prepares scales and tanh gates."""

    def __init__(self, source):
        super().__init__()
        self.embedding = source.time_text_embed.timestep_embedder
        self.activation = source.modulation[0]
        self.block_projection = source.modulation[1]
        self.output_projection = source.norm_out.linear

    def forward(self, sinusoidal_features):
        activated = self.activation(self.embedding(sinusoidal_features))
        return self.block_projection(activated), self.output_projection(activated)


class ImageProjection(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.projection = source.img_in

    def forward(self, latents):
        return self.projection(latents)


class TextProjection(nn.Module):
    """Feature projection with checkpoint's zero-centered RMSNorm scale."""

    def __init__(self, source, norm=None):
        super().__init__()
        self.norm = norm
        self.norm_weight = source.txt_in.text_norm.weight
        self.eps = source.txt_in.text_norm.eps
        self.input_projection = source.txt_in.in_layer
        self.activation = source.txt_in.act
        self.output_projection = source.txt_in.out_layer

    def forward(self, features):
        if self.norm is not None:
            normalized = self.norm(features)
        else:
            wide = features.float()
            normalized = (
                wide * torch.rsqrt((wide * wide).mean(-1, keepdim=True) + self.eps) * (1 + self.norm_weight.float())
            ).type_as(features)
        return self.output_projection(self.activation(self.input_projection(normalized)))
