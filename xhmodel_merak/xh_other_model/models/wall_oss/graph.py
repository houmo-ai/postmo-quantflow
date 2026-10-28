# Derived from X-Square-Robot/wall-x, Apache-2.0; see licenses/wall-x/LICENSE.
# Copyright 2026 HOUMO AI (static graph and cache adapters)
"""Separate routed experts with graph-owned KV writes and complete action projections."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


class ExpertStack(nn.Module):
    def __init__(self, source, *, expert, query_capacity, cache_capacity, sdk=True):
        super().__init__()
        self.experts = nn.ModuleList([deepcopy(layer[expert]) for layer in source.layers])
        self.heads, self.kvheads, self.dim = source.heads, source.kvheads, source.dim
        self.sdk, self.query_capacity, self.cache_capacity = sdk, query_capacity, cache_capacity
        self.register_buffer("positions", torch.arange(query_capacity, dtype=torch.int32).reshape(1, query_capacity, 1))
        if sdk:
            from xhquant.nn import FlashAttention, Greater, LLMCacheV2, RMSNorm, Rope

            self.rotary, self.visible = Rope(), Greater()
            self.attention = FlashAttention(
                num_heads=self.heads, num_kv_heads=self.kvheads, is_causal=True, scale=self.dim**-0.5
            )
            self.kcache = nn.ModuleList([LLMCacheV2(axis=2) for _ in self.experts])
            self.vcache = nn.ModuleList([LLMCacheV2(axis=2) for _ in self.experts])
            for e in self.experts:
                for name in ("norm1", "norm2"):
                    old = getattr(e, name)
                    new = RMSNorm(old.weight.numel(), old.eps)
                    new.weight = old.weight
                    setattr(e, name, new)

    def forward(self, hidden, cos, sin, past, current, caches):
        if self.sdk:
            mask = self.visible(current.reshape(1, 1, 1), self.positions).to(hidden.dtype)
        else:
            mask = (self.positions < current.reshape(1, 1, 1)).to(hidden.dtype)
        presents = []
        for i, e in enumerate(self.experts):
            q, k, v = e.qkv(e.norm1(hidden)).split(
                (self.heads * self.dim, self.kvheads * self.dim, self.kvheads * self.dim), -1
            )
            q = q.reshape(1, self.query_capacity, self.heads, self.dim).transpose(1, 2)
            k = k.reshape(1, self.query_capacity, self.kvheads, self.dim).transpose(1, 2)
            v = v.reshape(1, self.query_capacity, self.kvheads, self.dim).transpose(1, 2)
            rope = self.rotary if self.sdk else rotate
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            visible = mask.transpose(1, 2).unsqueeze(-1)
            q, k, v = q * visible, k * visible, v * visible
            if self.sdk:
                k = self.kcache[i](k, past, current, caches[2 * i])
                v = self.vcache[i](v, past, current, caches[2 * i + 1])
                a = self.attention(q, k, v, past_seq_length=past, current_input_length=current)
            else:
                offset, length = int(past[0]), int(current[0])
                pk, pv = caches[2 * i].clone(), caches[2 * i + 1].clone()
                pk[:, :, offset : offset + length] = k[:, :, :length]
                pv[:, :, offset : offset + length] = v[:, :, :length]
                k, v = pk, pv
                presents.extend((pk, pv))
                keys = torch.arange(self.cache_capacity, device=hidden.device)[None, None, None, :]
                queries = (torch.arange(self.query_capacity, device=hidden.device) + offset)[None, None, :, None]
                valid = (
                    (keys <= queries)
                    & (keys < offset + length)
                    & (torch.arange(self.query_capacity, device=hidden.device)[None, None, :, None] < length)
                )
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=valid, enable_gqa=True)
            a = a.transpose(1, 2).reshape(1, self.query_capacity, self.heads * self.dim)
            hidden = (hidden + e.out(a)) * mask
            hidden = (hidden + e.mlp(e.norm2(hidden))) * mask
        return hidden if self.sdk else (hidden, tuple(presents))


class ActionDenoiser(nn.Module):
    def __init__(self, source, *, query_capacity, cache_capacity, sdk=True):
        super().__init__()
        self.stack = ExpertStack(
            source, expert=1, query_capacity=query_capacity, cache_capacity=cache_capacity, sdk=sdk
        )
        self.w1, self.w2, self.w3 = deepcopy(source.w1), deepcopy(source.w2), deepcopy(source.w3)
        self.norm, self.proj = deepcopy(source.action_norm), deepcopy(source.action_out)
        self.sdk = sdk
        if sdk:
            from xhquant.nn import RMSNorm

            norm = RMSNorm(self.norm.weight.numel(), self.norm.eps)
            norm.weight = self.norm.weight
            self.norm = norm

    def forward(self, action, dof, time, cos, sin, past, current, caches):
        h = self.w1(torch.cat((action, dof), -1))
        h = self.w3(F.silu(self.w2(torch.cat((h, time[:, None].expand(-1, self.stack.query_capacity, -1)), -1))))
        result = self.stack(h, cos, sin, past, current, caches)
        if self.sdk:
            return self.proj(self.norm(result))
        hidden, presents = result
        return self.proj(self.norm(hidden)), presents
