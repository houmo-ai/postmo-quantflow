# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Static DSpark components using the existing DeepSeek-V4 graph primitives.

The context graphs append only verified target features. Draft queries write
transient KV, so the runtime restores their cache snapshot before committing
the next verified context. The anchor is the first prediction position.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy

import torch
from torch import Tensor, nn

from xhquant import nn as xhnn

from .full_model import _build_rope_cache
from ._trace import is_fx_proxy
from .attention import InterleavedPartialRope
from .decoder import StaticSWADecoderLayer
from .mhc import BoundHCHead
from .norm import rms_norm_from_hf
from .swa import StaticSWAUpdate
from .swa_layout import aligned_swa_backing_length


def _share_operator_tensors(module: nn.Module) -> nn.Module:
    memo = {id(value): nn.Parameter(value.detach(), requires_grad=value.requires_grad)
            for value in module.parameters()}
    memo.update({id(value): value for value in module.buffers()})
    return copy.deepcopy(module, memo)


@dataclass(frozen=True)
class DSparkSpec:
    target_layer_ids: tuple[int, ...] = (40, 41, 42)
    num_draft_layers: int = 3
    block_size: int = 5
    context_chunk_length: int = 256
    sliding_window: int = 128
    hidden_size: int = 4096
    head_dim: int = 512
    rope_dim: int = 64
    hc_mult: int = 4
    markov_rank: int = 256
    vocab_size: int = 129280
    noise_token_id: int = 128799
    head_weight_bits: int = 5
    max_context_length: int = 256 * 1024

    def __post_init__(self):
        if type(self.num_draft_layers) is not int or self.num_draft_layers != 3:
            raise ValueError("DeepSeek-V4 DSpark requires its three draft layers")
        if not self.target_layer_ids or len(set(self.target_layer_ids)) != len(self.target_layer_ids):
            raise ValueError("DSpark requires distinct target feature layer ids")
        if any(type(value) is not int or value < 0 for value in self.target_layer_ids):
            raise ValueError("DSpark target feature layer ids must be nonnegative integers")
        dimensions = (self.block_size, self.context_chunk_length, self.sliding_window,
                      self.hidden_size, self.head_dim, self.rope_dim, self.hc_mult,
                      self.markov_rank, self.vocab_size, self.max_context_length)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("DSpark dimensions must be positive integers")
        if self.rope_dim > self.head_dim or self.rope_dim % 2:
            raise ValueError("DSpark RoPE dimension must be even and fit within the head")
        if type(self.noise_token_id) is not int or not 0 <= self.noise_token_id < self.vocab_size:
            raise ValueError("DSpark noise token is outside the target vocabulary")
        if type(self.head_weight_bits) is not int or self.head_weight_bits not in (4, 5, 8):
            raise ValueError("DSpark shared head supports W4, W5 and W8")

    @property
    def cache_backing_length(self) -> int:
        return aligned_swa_backing_length(
            max(self.context_chunk_length, self.block_size + 1), self.sliding_window
        )

    def quant_scheme(self, activation_bits: int) -> dict:
        """W4 routed experts, W8 dense/shared experts, and one target-owned head.

        MoeBlock contains only routed experts. Shared experts and the router
        are separate Linear operators and retain the global W8 setting.
        The LM head uses the same W5A8 (or requested W4/W8) configuration in
        target and draft graphs, including when draft activations use A16.
        """
        if activation_bits not in (8, 16):
            raise ValueError("DSpark activations must use A8 or A16")
        return {
            "quant_type": f"w8a{activation_bits}h1_sefp",
            "w_schema": {"nshare": 64},
            "ops_cfg": {
                "MoeBlock": {"w_schema": {"bits": 4, "nshare": 64}},
            },
            "nodes_cfg": {
                "lm_head": {"quant_type": f"w{self.head_weight_bits}a8h1_sefp"},
            },
        }


class DSparkPositionRope(nn.Module):
    """Slice checkpoint RoPE by absolute position, including padded final rows."""

    def __init__(self, rotary_emb: nn.Module, spec: DSparkSpec, input_sequence_length: int):
        super().__init__()
        length = spec.max_context_length + max(spec.context_chunk_length, spec.block_size + 1) - 1
        cos, sin = _build_rope_cache(
            rotary_emb.main_inv_freq, rotary_emb.main_attention_scaling,
            torch.arange(length, device=rotary_emb.main_inv_freq.device))
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.slice = xhnn.DynamicSlice(input_sequence_length, 1, 1)

    def for_length(self, length: int):
        result = _share_operator_tensors(self)
        result.slice = xhnn.DynamicSlice(length, 1, 1)
        return result

    def forward(self, past_length: Tensor):
        return self.slice(self.cos_cached, past_length), self.slice(self.sin_cached, past_length)


class DSparkContext(nn.Module):
    """Project selected target layers and append each draft layer's latent KV."""

    def __init__(self, main_proj: nn.Module, main_norm: nn.Module,
                 draft_layers: list[StaticSWADecoderLayer], spec: DSparkSpec,
                 *, input_sequence_length: int, position_rope: DSparkPositionRope):
        super().__init__()
        if len(draft_layers) != spec.num_draft_layers:
            raise ValueError("DSpark context projection requires all draft layers")
        self.spec = spec
        self.position_rope = position_rope.for_length(input_sequence_length)
        self.main_proj = _share_operator_tensors(main_proj)
        self.main_norm = _share_operator_tensors(main_norm)
        self.kv_projections = nn.ModuleList([_share_operator_tensors(layer.attention.projection.kv_proj)
                                            for layer in draft_layers])
        self.kv_norms = nn.ModuleList([_share_operator_tensors(layer.attention.projection.kv_norm)
                                     for layer in draft_layers])
        self.rope = InterleavedPartialRope(spec.rope_dim)
        self.cache_updates = nn.ModuleList([
            StaticSWAUpdate(input_sequence_length=input_sequence_length,
                            window_size=spec.sliding_window,
                            backing_length=spec.cache_backing_length)
            for _ in draft_layers
        ])

    def forward(self, target_hidden: Tensor, past_length: Tensor, current_length: Tensor,
                cache_0: Tensor, cache_1: Tensor, cache_2: Tensor):
        if not is_fx_proxy(target_hidden):
            width = self.spec.hidden_size * len(self.spec.target_layer_ids)
            if target_hidden.ndim != 3 or target_hidden.shape[-1] != width:
                raise ValueError(f"DSpark target features must be [B,P,{width}]")
        cos, sin = self.position_rope(past_length)
        context = self.main_norm(self.main_proj(target_hidden))
        caches = (cache_0, cache_1, cache_2)
        output = []
        for i in range(self.spec.num_draft_layers):
            kv = self.kv_norms[i](self.kv_projections[i](context))
            kv = self.rope(kv.unsqueeze(2), cos, sin).squeeze(2)
            output.append(self.cache_updates[i](kv, past_length, current_length, caches[i]).physical_kv)
        return tuple(output)


class DSparkDraft(nn.Module):
    """Three parallel MoE/mHC blocks, draft HC head and the shared target LM head."""

    def __init__(self, layers: list[StaticSWADecoderLayer], hc_head: nn.Module,
                 norm: nn.Module, lm_head: nn.Module, spec: DSparkSpec,
                 position_rope: DSparkPositionRope):
        super().__init__()
        if len(layers) != spec.num_draft_layers:
            raise ValueError("DSpark draft requires all three trained blocks")
        self.spec = spec
        if any(layer.mlp.router.is_hash for layer in layers):
            raise ValueError("DSpark draft requires learned routing without token IDs")
        self.position_rope = position_rope
        self.layers = nn.ModuleList(layers)
        self.hc_head = hc_head
        self.norm = norm
        self.lm_head = lm_head

    @classmethod
    def from_hf(cls, draft_model: nn.Module, target_lm_head: nn.Module, spec: DSparkSpec):
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding

        base = draft_model.model if hasattr(draft_model, "model") else draft_model
        # Reconstruct FP32 frequencies from checkpoint configuration: model.half()
        # may already have rounded the HF module's inverse-frequency buffers.
        rotary = DeepseekV4RotaryEmbedding(draft_model.config)
        layers = [StaticSWADecoderLayer.from_hf(
            layer, input_sequence_length=spec.block_size,
            swa_backing_length=spec.cache_backing_length,
        ) for layer in base.layers]
        # Frontend/PTQ replaces operator instances in place. Keep independent
        # operator metadata for target/draft shapes while sharing its tensors.
        shared_head = _share_operator_tensors(target_lm_head)
        return cls(layers, BoundHCHead.from_hf(base.hc_head),
                   rms_norm_from_hf(base.norm), shared_head, spec,
                   DSparkPositionRope(rotary, spec, spec.block_size))

    def forward(self, noise_embedding: Tensor,
                past_length: Tensor, current_length: Tensor, attn_mask: Tensor,
                cache_0: Tensor, cache_1: Tensor, cache_2: Tensor):
        if not is_fx_proxy(noise_embedding) and noise_embedding.shape[1] != self.spec.block_size:
            raise ValueError("DSpark draft input must retain the configured block size")
        cos, sin = self.position_rope(past_length)
        hidden = noise_embedding.unsqueeze(2).expand(-1, -1, self.spec.hc_mult, -1)
        caches = (cache_0, cache_1, cache_2)
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, None, past_length, current_length,
                           attn_mask, caches[i], cos, sin).hidden_states
        head_hidden = self.hc_head(hidden)
        # Every position predicts a candidate, including the anchor position.
        return self.lm_head(self.norm(head_hidden)), head_hidden


class DSparkMarkov(nn.Module):
    """One sequential transition correction; the host samples the next token."""

    def __init__(self, token_codebook: Tensor, bias_weight: Tensor,
                 *, confidence_weight: Tensor | None = None):
        super().__init__()
        if token_codebook.ndim != 2 or bias_weight.shape != token_codebook.shape:
            raise ValueError("DSpark Markov weights must both be [vocab,rank]")
        vocab, rank = token_codebook.shape
        self.token_codebook = nn.Embedding(vocab, rank, device="meta", dtype=token_codebook.dtype)
        self.token_codebook.weight = nn.Parameter(token_codebook, requires_grad=False)
        self.bias_proj = nn.Linear(rank, vocab, bias=False, device="meta", dtype=bias_weight.dtype)
        self.bias_proj.weight = nn.Parameter(bias_weight, requires_grad=False)
        self.confidence_proj = None
        if confidence_weight is not None:
            self.confidence_proj = nn.Linear(confidence_weight.shape[1], 1, bias=False,
                                             device="meta", dtype=confidence_weight.dtype)
            self.confidence_proj.weight = nn.Parameter(confidence_weight, requires_grad=False)

    def forward(self, previous_token: Tensor, head_hidden: Tensor):
        markov_embedding = self.token_codebook(previous_token)
        bias = self.bias_proj(markov_embedding)
        if self.confidence_proj is None:
            return (bias,)
        confidence = torch.sigmoid(self.confidence_proj(torch.cat((head_hidden, markov_embedding), dim=-1)))
        return bias, confidence


__all__ = ["DSparkSpec", "DSparkContext", "DSparkDraft", "DSparkMarkov"]
