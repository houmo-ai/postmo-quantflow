# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Request-local execution of DiT extract/cached graph pairs."""

import math
from dataclasses import dataclass

import torch


def timestep_features(timestep, *, device=None, dtype=torch.float32):
    """Build [current time, zero-prefix time] features without rounding frequencies.

    The scheduler supplies normalized time (the native model multiplies by 1000).
    Keep phase arithmetic in FP32 and cast only the completed graph input.
    """
    time = torch.as_tensor(timestep, device=device, dtype=torch.float32)
    if time.numel() != 1 or not torch.isfinite(time).all():
        raise ValueError("timestep must contain one finite scalar")
    rows = torch.stack((time.reshape(()), torch.zeros((), device=time.device, dtype=torch.float32)))
    # Native initialization computes this buffer on CPU before moving the model.
    frequencies = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128).to(time.device)
    phase = (1000 * rows[:, None]) * frequencies[None]
    return torch.cat((phase.cos(), phase.sin()), dim=-1).to(dtype=dtype)


def prepare_conditioning(block_parameters, output_parameters):
    """Apply small per-step transforms outside HMONNX V2 (which lacks Tanh)."""
    if output_parameters.ndim != 2 or output_parameters.shape[0] != 2:
        raise ValueError("output conditioning must have shape [2, D]")
    if block_parameters.shape != (2, 4 * output_parameters.shape[1]):
        raise ValueError("block conditioning must have shape [2, 4*D]")
    scale1, gate1, scale2, gate2 = block_parameters.chunk(4, -1)
    return 1 + scale1, gate1.tanh(), 1 + scale2, gate2.tanh(), 1 + output_parameters


@dataclass(frozen=True)
class PrefixState:
    """State belongs to one chain/request; no global cache or cross-request reuse."""

    owner: object
    tensors: tuple[torch.Tensor, ...]


class DenoiseStackRunner:
    """Run callable tensor graphs (eager or HMONNX), preserving the cache ABI."""

    def __init__(self, extract, cached, *, layers, heads, head_dim, prefix_length, target_length):
        for name, value in dict(layers=layers, heads=heads, head_dim=head_dim, target_length=target_length).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(prefix_length, int) or isinstance(prefix_length, bool) or prefix_length < 0:
            raise ValueError("prefix_length must be a nonnegative integer")
        self.extract_graph = extract
        self.cached_graph = cached
        self.layers = layers
        self.cache_shape = (1, heads, prefix_length, head_dim)
        self.prefix_length = prefix_length
        self.target_length = target_length
        self.dim = heads * head_dim
        self._owner = object()

    def _validate_hidden(self, hidden, length):
        if not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != (1, length, self.dim):
            raise ValueError(f"hidden must have shape {(1, length, self.dim)}")

    @torch.no_grad()
    def extract(self, hidden, *parameters):
        self._validate_hidden(hidden, self.prefix_length + self.target_length)
        result = self.extract_graph(hidden, *parameters)
        if not isinstance(result, (tuple, list)) or len(result) != 1 + 2 * self.layers:
            raise ValueError("extract graph must return hidden followed by one K/V pair per layer")
        output, *values = result
        self._validate_hidden(output, self.prefix_length + self.target_length)
        cache = []
        for value in values:
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != self.cache_shape:
                raise ValueError(f"cache tensor must have shape {self.cache_shape}")
            # Compact ownership is essential: a view may otherwise pin all target K/V.
            cache.append(value.detach().clone(memory_format=torch.contiguous_format))
        return output[:, -self.target_length :], PrefixState(self._owner, tuple(cache))

    @torch.no_grad()
    def cached(self, state, hidden, *parameters):
        self._validate_hidden(hidden, self.target_length)
        if not isinstance(state, PrefixState) or state.owner is not self._owner:
            raise ValueError("prefix cache belongs to another request/runner")
        if len(state.tensors) != 2 * self.layers or any(tuple(t.shape) != self.cache_shape for t in state.tensors):
            raise ValueError("invalid prefix cache layout")
        output = self.cached_graph(hidden, *parameters, state.tensors)
        self._validate_hidden(output, self.target_length)
        return output


class DenoiseRequest:
    """One fixed prompt/reference set, with extract once then target-only steps.

    Callables may be eager modules or deployed graph sessions. A fresh instance is
    required for every prompt, reference set, or classifier-free guidance branch.
    The caller owns scheduler updates and VAE encode/decode.
    """

    @torch.no_grad()
    def __init__(
        self,
        layout,
        compact_rope,
        text_features,
        reference_latents,
        *,
        text_projection,
        image_projection,
        time_conditioning,
        output_head,
        extract,
        cached,
        layers,
        heads,
        head_dim,
    ):
        expected_references = len(layout.image_destinations) - layout.target_tokens
        if reference_latents.ndim != 3 or reference_latents.shape[:2] != (1, expected_references):
            raise ValueError("reference latents disagree with the token layout")
        if text_features.device != reference_latents.device or text_features.dtype != reference_latents.dtype:
            raise ValueError("text features and reference latents must share dtype and device")
        self.layout = layout
        self.image_projection = image_projection
        self.time_conditioning = time_conditioning
        self.output_head = output_head
        self.runner = DenoiseStackRunner(
            extract,
            cached,
            layers=layers,
            heads=heads,
            head_dim=head_dim,
            prefix_length=layout.prefix_capacity,
            target_length=layout.target_tokens,
        )
        # Graph sessions may reuse output buffers on their next invocation.
        self.text = text_projection(text_features).detach().clone()
        self.references = (
            image_projection(reference_latents).detach().clone()
            if expected_references
            else self.text.new_empty(1, 0, heads * head_dim)
        )
        self.latent_channels = reference_latents.shape[-1]
        self.cos, self.sin = [
            value.to(device=self.text.device, dtype=self.text.dtype).contiguous()
            for value in layout.pad_rope(compact_rope)
        ]
        self.mask = layout.attention_bias(device=self.text.device, dtype=self.text.dtype)
        self.state = None

    @torch.no_grad()
    def __call__(self, target_latents, timestep):
        expected = (1, self.layout.target_tokens, self.latent_channels)
        if tuple(target_latents.shape) != expected:
            raise ValueError(f"target latents must have shape {expected}")
        if target_latents.device != self.text.device or target_latents.dtype != self.text.dtype:
            raise ValueError("target latents must match the request dtype and device")
        time = torch.as_tensor(timestep, device=target_latents.device, dtype=target_latents.dtype)
        features = timestep_features(time, device=target_latents.device, dtype=target_latents.dtype)
        parameters = prepare_conditioning(*self.time_conditioning(features))
        target = self.image_projection(target_latents)
        if self.state is None:
            joint = self.layout.assemble(self.text, torch.cat((self.references, target), dim=1))
            hidden, state = self.runner.extract(joint, *parameters[:4], self.cos, self.sin, self.mask)
        else:
            prefix = self.layout.prefix_capacity
            hidden = self.runner.cached(
                self.state,
                target,
                *parameters[:4],
                self.cos[:, :, prefix:],
                self.sin[:, :, prefix:],
                self.mask[:, :, prefix:],
            )
            state = self.state
        noise = self.output_head(hidden, parameters[4])
        if tuple(noise.shape) != expected:
            raise ValueError(f"output head must return noise with shape {expected}")
        self.state = state
        return noise


class LayerwiseDenoiser:
    """Adapt per-block graph sessions to the stack ABI without fusing all weights.

    Each extract block returns hidden, K, V; each cached block consumes one K/V
    pair. This permits exporting one layer at a time to bound conversion memory.
    """

    def __init__(self, extract_blocks, cached_blocks, *, prefix_length, target_length):
        if not extract_blocks or len(extract_blocks) != len(cached_blocks):
            raise ValueError("one extract and cached graph are required for each layer")
        if prefix_length < 0 or target_length <= 0:
            raise ValueError("invalid token lengths")
        self.extract_blocks = tuple(extract_blocks)
        self.cached_blocks = tuple(cached_blocks)
        self.prefix_length = prefix_length
        self.target_length = target_length

    def extract(self, hidden, scale1, gate1, scale2, gate2, cos, sin, mask):
        parameters = [
            torch.cat(
                (p[1:2, None].expand(1, self.prefix_length, -1), p[:1, None].expand(1, self.target_length, -1)), dim=1
            )
            for p in (scale1, gate1, scale2, gate2)
        ]
        cache = []
        for block in self.extract_blocks:
            hidden, key, value = block(hidden, *parameters, cos, sin, mask)
            cache.extend((key, value))
        return (hidden, *cache)

    def cached(self, hidden, scale1, gate1, scale2, gate2, cos, sin, mask, prefix_cache):
        if len(prefix_cache) != 2 * len(self.cached_blocks):
            raise ValueError("one K/V pair is required for each cached layer")
        parameters = [p[:1, None] for p in (scale1, gate1, scale2, gate2)]
        for index, block in enumerate(self.cached_blocks):
            hidden = block(hidden, *parameters, cos, sin, mask, *prefix_cache[2 * index : 2 * index + 2])
        return hidden


class TokenProjection:
    """Pad a token-local projection to its exported gear and trim the result."""

    def __init__(self, graph, *, capacity, input_channels):
        for value in (capacity, input_channels):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("projection capacity and channels must be positive integers")
        self.graph = graph
        self.capacity = capacity
        self.input_channels = input_channels

    def __call__(self, inputs):
        if inputs.ndim != 3 or inputs.shape[0] != 1 or inputs.shape[-1] != self.input_channels:
            raise ValueError("projection input must have shape [1, tokens, input_channels]")
        tokens = inputs.shape[1]
        if not 0 < tokens <= self.capacity:
            raise ValueError("projection token count exceeds its gear or is empty")
        padded = inputs.new_zeros(1, self.capacity, self.input_channels)
        padded[:, :tokens] = inputs
        output = self.graph(padded)
        if output.ndim != 3 or tuple(output.shape[:2]) != (1, self.capacity):
            raise ValueError("projection output does not match its token gear")
        return output[:, :tokens]
