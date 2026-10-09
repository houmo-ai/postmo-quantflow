# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Ordered DiT prefix initialization and request-owned in-place image cache."""

import torch
from torch.nn import functional as F

from .runtime import timestep_features


class SegmentDenoiseRequest:
    def __init__(
        self, layout, compact_rope, text_features, reference_latents, *, text, image, gear, layers, heads, head_dim
    ):
        self.layout, self.gear = layout, gear
        self.text_graph, self.image_graph = text, image
        self.layers = layers
        self.cache_shape = (1, heads, gear.cache_tokens, head_dim)
        self.channels = reference_latents.shape[-1]
        self.caches = None
        self.initialized = False
        if layout.prefix_tokens + layout.target_tokens > gear.cache_tokens:
            raise ValueError("actual joint sequence exceeds the cache capacity")
        if layout.target_tokens > gear.target_tokens:
            raise ValueError("actual target exceeds the image graph capacity")
        gap = layout.prefix_capacity - layout.prefix_tokens
        positions = [index - gap for index in layout.text_destinations]
        image_positions = [index - gap for index in layout.image_destinations[: -layout.target_tokens]]
        if tuple(reference_latents.shape[:2]) != (1, len(image_positions)):
            raise ValueError("reference latents disagree with the compact prefix")
        self.text = text_features.new_zeros(1, layout.prefix_tokens, text_features.shape[-1])
        self.text[:, positions] = text_features[:, layout.text_sources]
        self.reference = reference_latents.new_zeros(1, layout.prefix_tokens, self.channels)
        self.reference[:, image_positions] = reference_latents
        if text_features.dtype != reference_latents.dtype or text_features.device != reference_latents.device:
            raise ValueError("text and reference boundaries must share dtype/device")
        if compact_rope.shape != (layout.prefix_tokens + layout.target_tokens, head_dim // 2):
            raise ValueError("RoPE must retain the complete original token layout")
        self.cos, self.sin = [
            x[None, None].to(text_features).contiguous() for x in (compact_rope.real, compact_rope.imag)
        ]
        ids = layout.image_ids[gap : gap + layout.prefix_tokens].tolist()
        self.segments = []
        start = 0
        for end in range(1, len(ids) + 1):
            if end == len(ids) or ids[end] != ids[start]:
                self.segments.append((start, end, "text" if ids[start] < 0 else "image"))
                start = end

    def arguments(self, tokens, start, timestep, *, text):
        length = tokens.shape[1]
        capacity = self.gear.text_tokens if text else self.gear.image_tokens
        if not 0 < length <= capacity:
            raise ValueError("segment length exceeds its exported graph capacity")
        padded = F.pad(tokens, (0, 0, 0, capacity - length))
        cos = F.pad(self.cos[:, :, start : start + length], (0, 0, 0, capacity - length), value=1)
        sin = F.pad(self.sin[:, :, start : start + length], (0, 0, 0, capacity - length))
        time = timestep_features(timestep, device=tokens.device, dtype=tokens.dtype)[:1]
        past = torch.tensor([start], dtype=torch.int32, device=tokens.device)
        current = torch.tensor([length], dtype=torch.int32, device=tokens.device)
        return padded, time, cos, sin, past, current, self.caches

    def allocate_caches(self):
        from xhquant.core.cache_tensor import CacheTensor

        self.caches = tuple(CacheTensor(self.text.new_zeros(self.cache_shape)) for _ in range(2 * self.layers))

    @torch.no_grad()
    def initialize_prefix(self):
        if self.initialized:
            return
        self.allocate_caches()
        try:
            for start, end, role in self.segments:
                tokens = self.text[:, start:end] if role == "text" else self.reference[:, start:end]
                graph = self.text_graph if role == "text" else self.image_graph
                graph(*self.arguments(tokens, start, 0.0, text=role == "text"))
        except Exception:
            self.caches = None
            raise
        self.initialized = True

    @torch.no_grad()
    def __call__(self, target, timestep):
        if tuple(target.shape) != (1, self.layout.target_tokens, self.channels):
            raise ValueError("target latents disagree with the request geometry")
        if target.dtype != self.text.dtype or target.device != self.text.device:
            raise ValueError("target dtype/device disagree with request")
        self.initialize_prefix()
        noise = self.image_graph(*self.arguments(target, self.layout.prefix_tokens, timestep, text=False))
        if tuple(noise.shape) != (1, self.gear.image_tokens, self.channels):
            raise ValueError("image DiT graph returned an invalid prediction")
        return noise[:, : self.layout.target_tokens].clone()
