# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Request-owned replacement buffers for complete DiT graphs."""

import torch

from .runtime import timestep_features


class FullDenoiseRequest:
    def __init__(
        self, layout, compact_rope, text_features, reference_latents, *, extract, cached, layers, heads, head_dim
    ):
        self.layout = layout
        self.extract_graph, self.cached_graph = extract, cached
        self.layers = layers
        self.cache_shape = (1, heads, layout.sequence_length, head_dim)
        self.channels = reference_latents.shape[-1]
        count = len(layout.image_destinations) - layout.target_tokens
        if reference_latents.shape[:2] != (1, count):
            raise ValueError("reference latents disagree with layout")
        self.text = text_features.new_zeros(1, layout.prefix_capacity, text_features.shape[-1])
        self.text[:, layout.text_destinations] = text_features[:, layout.text_sources]
        self.text_mask = text_features.new_zeros(1, layout.prefix_capacity, 1)
        self.text_mask[:, layout.text_destinations] = 1
        self.reference = reference_latents.new_zeros(1, layout.prefix_capacity, self.channels)
        self.reference[:, layout.image_destinations[:count]] = reference_latents
        self.cos, self.sin = [value.to(text_features).contiguous() for value in layout.pad_rope(compact_rope)]
        self.mask = layout.attention_ranges(device=text_features.device)
        self.caches = None

    @torch.no_grad()
    def __call__(self, target, timestep):
        if tuple(target.shape) != (1, self.layout.target_tokens, self.channels):
            raise ValueError("target latents disagree with deployed gear")
        if target.dtype != self.text.dtype or target.device != self.text.device:
            raise ValueError("target dtype/device disagree with request")
        time = torch.as_tensor(timestep, device=target.device, dtype=target.dtype)
        features = timestep_features(time, device=target.device, dtype=target.dtype)
        padded = torch.nn.functional.pad(target, (0, 0, 0, self.layout.target_capacity - self.layout.target_tokens))
        p = self.layout.prefix_capacity
        if self.caches is None:
            result = self.extract_graph(
                self.text,
                torch.cat((self.reference, padded), dim=1),
                self.text_mask,
                features,
                self.cos,
                self.sin,
                self.mask,
            )
        else:
            result = self.cached_graph(
                padded, features, self.cos[:, :, p:], self.sin[:, :, p:], self.mask[:, p:], self.caches
            )
        if not isinstance(result, (tuple, list)) or len(result) != 1 + 2 * self.layers:
            raise ValueError("complete DiT graph returned an invalid cache ABI")
        noise, *updated = result
        if tuple(noise.shape) != (1, self.layout.target_capacity, self.channels):
            raise ValueError("complete DiT graph returned an invalid prediction")
        if any(tuple(value.shape) != self.cache_shape for value in updated):
            raise ValueError("complete DiT graph returned an invalid cache shape")
        if self.caches is None:
            self.caches = tuple(value.detach().clone() for value in updated)
        else:
            # Prefix is immutable. Replace the same target region on every step;
            # buffers are never shared between requests or CFG branches.
            for cache, value in zip(self.caches, updated, strict=True):
                cache[:, :, p:].copy_(value[:, :, p:])
        return noise[:, : self.layout.target_tokens].clone()
