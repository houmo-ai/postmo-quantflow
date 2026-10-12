# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Complete segmented DiT stacks and historical extract/cached graph adapters."""

import torch
from torch import nn

from .graph import BlockCore, ImageProjection, OutputHead, TimeConditioning
from .length_attention import LengthMaskedAttention


class SegmentDenoiser(nn.Module):
    """One complete causal text or bidirectional image stack with mutable KVCache."""

    def __init__(self, source, capacity, *, text_projection=None, cache_capacity=None,
                 use_noncausal_flash_attention=False):
        super().__init__()
        from xhquant.nn import FlashAttention, Greater, LLMCacheV2

        self.causal = text_projection is not None
        self.projection = text_projection if self.causal else ImageProjection(source)
        self.time = TimeConditioning(source)
        self.head = None if self.causal else OutputHead(source)
        self.blocks = nn.ModuleList([BlockCore(block, sdk_ops=True) for block in source.transformer_blocks])
        self.k_cache = nn.ModuleList([LLMCacheV2(axis=2) for _ in self.blocks])
        self.v_cache = nn.ModuleList([LLMCacheV2(axis=2) for _ in self.blocks])
        for core in self.blocks:
            if self.causal:
                core.attention = FlashAttention(num_heads=core.heads, is_causal=True, scale=core.head_dim**-0.5)
            else:
                core.attention = LengthMaskedAttention(
                    capacity, cache_capacity, scale=core.head_dim**-0.5,
                    use_flash_attention=use_noncausal_flash_attention, num_heads=core.heads,
                )
        self.visible = Greater()
        self.register_buffer("positions", torch.arange(capacity, dtype=torch.int32)[None])

    def forward(self, tokens, time_features, cos, sin, past_seq_length, current_input_length, caches):
        raw, output = self.time(time_features)
        # Explicit sections avoid TCIM 1.5 treating num_outputs as a chunk size.
        scale1, gate1, scale2, gate2 = torch.split(
            raw, [self.time.block_projection.out_features // 4] * 4, dim=-1
        )
        scale1, gate1, scale2, gate2 = (
            1 + scale1[:, None],
            2 * torch.sigmoid(2 * gate1[:, None]) - 1,
            1 + scale2[:, None],
            2 * torch.sigmoid(2 * gate2[:, None]) - 1,
        )
        hidden = self.projection(tokens)
        visible = self.visible(current_input_length, self.positions)[:, None, :, None].to(hidden.dtype)
        for index, core in enumerate(self.blocks):
            q, k, v = core.project(hidden, scale1, cos, sin)
            q, k, v = q * visible, k * visible, v * visible
            k = self.k_cache[index](k, past_seq_length, current_input_length, caches[2 * index])
            v = self.v_cache[index](v, past_seq_length, current_input_length, caches[2 * index + 1])
            if self.causal:
                attended = core.attention(
                    q, k, v, past_seq_length=past_seq_length, current_input_length=current_input_length
                )
            else:
                attended = core.attention(
                    q, k, v, current_input_length, past_seq_length + current_input_length
                )
            attention = attended.transpose(1, 2).reshape(1, -1, core.dim)
            hidden = hidden + gate1 * core.out(attention)
            h = core.norm2(hidden) * scale2
            h = torch.nn.functional.silu(core.mlp_gate(h)) * core.mlp_in(h)
            hidden = hidden + gate2 * core.mlp_out(h)
        return hidden if self.causal else self.head(hidden, 1 + output)


class FullDenoiser(nn.Module):
    """The cached graph replaces target K/V; only prefix K/V is read from history.

    Cache ports are layer0 K, layer0 V, layer1 K, layer1 V, ... . Both graphs
    return noise followed by full-capacity cache tensors. The runtime owns the
    buffers and copies outputs before invoking another request on the session.
    """

    def __init__(self, source, text_projection, prefix_length, target_length, *, cached=False):
        super().__init__()
        from xhquant.nn import Greater

        self.visible = Greater()
        self.text = text_projection
        self.image = ImageProjection(source)
        self.time = TimeConditioning(source)
        self.head = OutputHead(source)
        self.blocks = nn.ModuleList([BlockCore(block, sdk_ops=True) for block in source.transformer_blocks])
        self.prefix_length = prefix_length
        self.target_length = target_length
        self.cached = cached

    def forward(self, text, latents, text_mask, time_features, cos, sin, mask, caches):
        raw, output = self.time(time_features)
        scale1, gate1, scale2, gate2 = raw.chunk(4, -1)
        # Exact real-number identity, lowered through the SDK's Sigmoid path.
        parameters = (1 + scale1, 2 * torch.sigmoid(2 * gate1) - 1, 1 + scale2, 2 * torch.sigmoid(2 * gate2) - 1)
        hidden = self.image(latents)
        if self.cached:
            parameters = tuple(p[:1, None] for p in parameters)
        else:
            prefix = hidden[:, : self.prefix_length]
            prefix = self.text(text) * text_mask + prefix * (1 - text_mask)
            hidden = torch.cat((prefix, hidden[:, self.prefix_length :]), dim=1)
            parameters = tuple(
                torch.cat(
                    (p[1:2, None].expand(1, self.prefix_length, -1), p[:1, None].expand(1, self.target_length, -1)),
                    dim=1,
                )
                for p in parameters
            )
        scale1, gate1, scale2, gate2 = parameters
        updated = []
        for index, core in enumerate(self.blocks):
            q, k, v = core.project(hidden, scale1, cos, sin)
            # Clear invisible rows before SEFP score and transposed-V groups
            # choose their exponents. A visibility range alone is insufficient
            # if padding K/V contains projection biases.
            visible = self.visible(mask[..., 1], mask[..., 0])[:, None, :, None].to(k.dtype)
            q, k, v = q * visible, k * visible, v * visible
            if self.cached:
                k = torch.cat((caches[2 * index][:, :, : self.prefix_length], k), dim=2)
                v = torch.cat((caches[2 * index + 1][:, :, : self.prefix_length], v), dim=2)
            updated.extend((k, v))
            hidden = core.finish(hidden, q, k, v, gate1, scale2, gate2, mask)
        noise = self.head(hidden[:, -self.target_length :], 1 + output)
        return (noise, *updated)


class ExtractGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text, latents, text_mask, time_features, cos, sin, mask):
        return self.model(text, latents, text_mask, time_features, cos, sin, mask, ())


class CachedGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latents, time_features, cos, sin, mask, caches):
        return self.model(None, latents, None, time_features, cos, sin, mask, caches)
