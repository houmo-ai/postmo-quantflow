# Derived from Qwen2.5-VL / Wall-X, Apache-2.0; see licenses/wall-x/LICENSE.
# Copyright 2026 HOUMO AI (packed static visual adapter)
"""One patch-capacity graph with actual window ranges and inverse merge order."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .graph import rotate


class VisionBlock(nn.Module):
    def __init__(self, source, heads, dim, *, sdk):
        super().__init__()
        self.norm1, self.norm2 = deepcopy(source.norm1), deepcopy(source.norm2)
        self.qkv, self.proj, self.mlp = deepcopy(source.attn.qkv), deepcopy(source.attn.proj), deepcopy(source.mlp)
        self.heads, self.dim, self.sdk = heads, dim, sdk
        if sdk:
            from xhquant.nn import FlashAttention, RMSNorm, Rope

            for name in ("norm1", "norm2"):
                old = getattr(self, name)
                new = RMSNorm(old.weight.numel(), old.variance_epsilon)
                new.weight = old.weight
                setattr(self, name, new)
            self.rope = Rope()
            self.attention = FlashAttention(num_heads=heads, is_causal=False, scale=dim**-0.5)
            self.physical_dim = next(d for d in (64, 128, 256) if d >= dim)

    def forward(self, hidden, cos, sin, mask, ranges):
        qkv = self.qkv(self.norm1(hidden)).reshape(1, -1, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        rope = self.rope if self.sdk else rotate
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        valid = mask.transpose(1, 2).unsqueeze(-1)
        q, k, v = q * valid, k * valid, v * valid
        if self.sdk:
            q, k, v = (F.pad(x, (0, self.physical_dim - self.dim)) for x in (q, k, v))
            out = self.attention(q, k, v, query_kv_range_abs=ranges)[..., : self.dim]
        else:
            pos = torch.arange(hidden.shape[1], device=hidden.device)
            allowed = (pos[None, None, :] >= ranges[:, :, 0, None]) & (pos[None, None, :] < ranges[:, :, 1, None])
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None], scale=self.dim**-0.5)
        hidden = (hidden + self.proj(out.transpose(1, 2).reshape(1, -1, self.heads * self.dim))) * mask
        return (hidden + self.mlp(self.norm2(hidden))) * mask


class PackedVision(nn.Module):
    def __init__(self, source, capacity, *, sdk=True):
        super().__init__()
        vc = source.config
        self.capacity, self.full = capacity, tuple(vc.fullatt_block_indexes)
        self.patch = nn.Linear(source.patch_embed.proj.weight[0].numel(), vc.hidden_size, bias=False)
        self.patch.weight = nn.Parameter(source.patch_embed.proj.weight.detach().flatten(1).clone())
        self.blocks = nn.ModuleList(
            [VisionBlock(b, vc.num_heads, vc.hidden_size // vc.num_heads, sdk=sdk) for b in source.blocks]
        )
        self.merger = deepcopy(source.merger)
        if sdk:
            from xhquant.nn import RMSNorm

            old = self.merger.ln_q
            new = RMSNorm(old.weight.numel(), old.variance_epsilon)
            new.weight = old.weight
            self.merger.ln_q = new

    def forward(self, patches, order, reverse, cos, sin, mask, full_range, window_range):
        hidden = self.patch(patches[:, order.long()]) * mask
        for i, block in enumerate(self.blocks):
            hidden = block(hidden, cos, sin, mask, full_range if i in self.full else window_range)
        result = self.merger(hidden).reshape(1, self.capacity // 4, -1)
        result = result * mask.reshape(1, self.capacity // 4, 4, 1)[:, :, 0]
        return result[:, reverse.long()]
