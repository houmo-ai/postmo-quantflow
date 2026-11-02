# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL's first 50 text layers, without a final norm or an LM head."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .precision import ATTENTION_VALUE_DIVISOR, StableRMSNorm, scaled_linear


TEXT_FEATURE_DIVISOR = 256.0
TEXT_PRODUCT_DIVISOR = 512.0


def load_text_source(model_dir, *, layers=50, quantized_dir=None):
    """Only materialize the used language prefix and embeddings from local shards."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding

    if quantized_dir is not None:
        from gptqmodel import GPTQModel
        from gptqmodel.utils.backend import BACKEND

        from xhmodel_merak.xh_llm._dequant_converter import replace_gptqmodel_quant_linears_with_packed
        from xhmodel_merak.xh_llm.big_hf_model_helper import GPTQModelQuantizedModelPreprocessor

        root = Path(quantized_dir)
        receipt = json.loads((root / "qwen3_vl_feature_quantization.json").read_text())
        if receipt.get("feature_layer") != layers or receipt.get("bits") != 4 or not receipt.get("reload_validated"):
            raise ValueError("Qwen3-VL checkpoint must be validated W4 for the requested feature layer")
        loaded = GPTQModel.load(str(root), device="cpu", backend=BACKEND.TORCH)
        source = loaded.model.model.language_model
        config = source.config.to_dict()
        source.layers = nn.ModuleList(list(source.layers[:layers]))
        # Follow Merak's existing checkpoint handoff: v2 needs metadata only,
        # never an extra arithmetic offset on the already-canonical zeros.
        GPTQModelQuantizedModelPreprocessor.prepare_loaded_module(source, loaded.quantize_config)
        replaced = replace_gptqmodel_quant_linears_with_packed(source)
        if len(replaced) != layers * 7:
            raise ValueError("Qwen3-VL feature prefix must contain seven packed projections per layer")
        return source.eval().requires_grad_(False), config
    root = Path(model_dir) / "text_encoder"
    data = json.loads((root / "config.json").read_text())["text_config"]
    if not 0 < layers < data["num_hidden_layers"]:
        raise ValueError("H3 conditioning must stop before the native text tower's final normalization")
    config = Qwen3VLTextConfig.from_dict(data)
    config.num_hidden_layers = layers
    with torch.device("meta"):
        source = Qwen3VLTextModel(config)
    load_component_weights(source, root, "model.language_model.")
    source.rotary_emb = Qwen3VLTextRotaryEmbedding(config)
    return source.eval().requires_grad_(False), data


def load_component_weights(source, root, prefix, *, weights_name="model.safetensors"):
    """Load only one component's tensors from shared HF shards, with strict keys."""
    from safetensors import safe_open

    targets, state = set(source.state_dict()), {}
    index_path = root / f"{weights_name}.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(str(root / weights_name), framework="pt", device="cpu") as checkpoint:
            index = {key: weights_name for key in checkpoint.keys()}
    needed = {key: shard for key, shard in index.items() if key.startswith(prefix) and key[len(prefix) :] in targets}
    for shard in sorted(set(needed.values())):
        with safe_open(str(root / shard), framework="pt", device="cpu") as checkpoint:
            for key in (name for name, file in needed.items() if file == shard):
                state[key[len(prefix) :]] = checkpoint.get_tensor(key)
    source.load_state_dict(state, strict=True, assign=True)


class TextFeatureBlock(nn.Module):
    def __init__(self, source, *, sdk_ops=False, divisor=TEXT_FEATURE_DIVISOR):
        super().__init__()
        attention, mlp = source.self_attn, source.mlp
        self.norm1 = StableRMSNorm(
            source.input_layernorm, input_divisor=divisor, sdk_ops=sdk_ops, bounded_reciprocal=divisor != 16
        )
        self.norm2 = StableRMSNorm(
            source.post_attention_layernorm, input_divisor=divisor, sdk_ops=sdk_ops, bounded_reciprocal=divisor != 16
        )
        self.q, self.k = attention.q_proj, attention.k_proj
        self.v = scaled_linear(attention.v_proj, output_divisor=ATTENTION_VALUE_DIVISOR)
        self.head_dim = attention.head_dim
        self.heads = self.q.out_features // self.head_dim
        self.kv_heads = self.k.out_features // self.head_dim
        self.q_norm, self.k_norm = (
            StableRMSNorm(attention.q_norm, sdk_ops=sdk_ops),
            StableRMSNorm(attention.k_norm, sdk_ops=sdk_ops),
        )
        self.output = scaled_linear(attention.o_proj, input_multiplier=ATTENTION_VALUE_DIVISOR, output_divisor=divisor)
        self.gate = mlp.gate_proj
        self.up = scaled_linear(mlp.up_proj, output_divisor=TEXT_PRODUCT_DIVISOR)
        self.down = scaled_linear(mlp.down_proj, input_multiplier=TEXT_PRODUCT_DIVISOR, output_divisor=divisor)
        self.sdk_ops = sdk_ops
        self.use_cache = False
        if sdk_ops:
            from xhquant.nn import Rope

            from ...common.attention import CompilerAttention as FlashAttention

            self.rotary = Rope()
            self.attention = FlashAttention(
                num_heads=self.heads, num_kv_heads=self.kv_heads, scale=attention.scaling, is_causal=True
            )
            self.register_buffer("past", torch.zeros(1, dtype=torch.int32))

    def rotate(self, x, cos, sin):
        if self.sdk_ops:
            return self.rotary(x, cos[:, None], sin[:, None])
        first, second = x.chunk(2, -1)
        return x * cos[:, None] + torch.cat((-second, first), -1) * sin[:, None]

    def forward(self, hidden, cos, sin, valid_length):
        return self._forward(hidden, cos, sin, valid_length)

    def _forward(self, hidden, cos, sin, valid_length, past_length=None, key_cache=None, value_cache=None):
        x = self.norm1(hidden)
        q = self.q(x).reshape(1, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(1, -1, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(1, -1, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotate(self.q_norm(q), cos, sin), self.rotate(self.k_norm(k), cos, sin)
        if self.use_cache:
            k = self.k_cache(k, past_length, valid_length, key_cache)
            v = self.v_cache(v, past_length, valid_length, value_cache)
        if self.sdk_ops:
            attended = self.attention(
                q, k, v, past_seq_length=past_length if self.use_cache else self.past, current_input_length=valid_length
            )
        else:
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        hidden = hidden + self.output(attended.transpose(1, 2).flatten(2))
        x = self.norm2(hidden)
        return hidden + self.down(F.silu(self.gate(x)) * self.up(x))


class TextFeatureStack(nn.Module):
    def __init__(self, layers, *, sdk_ops=False, divisor=TEXT_FEATURE_DIVISOR):
        super().__init__()
        self.layers = nn.ModuleList([TextFeatureBlock(layer, sdk_ops=sdk_ops, divisor=divisor) for layer in layers])

    def forward(self, hidden, cos, sin, valid_length):
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, valid_length)
        return hidden


class MultimodalTextFeatureStack(TextFeatureStack):
    """Dense DeepStack additions supplied by Host at the first three decoder layers.

    Host scatters each visual output into its image/video pad positions and applies
    the same residual divisor as token embeddings. RoPE is provided after the
    native Qwen3-VL multimodal position calculation.
    """

    def __init__(self, layers, *, start_layer=0, sdk_ops=False, divisor=TEXT_FEATURE_DIVISOR):
        super().__init__(layers, sdk_ops=sdk_ops, divisor=divisor)
        self.start_layer = start_layer

    def forward(self, hidden, cos, sin, valid_length, deepstack_0, deepstack_1, deepstack_2):
        additions = (deepstack_0, deepstack_1, deepstack_2)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, valid_length)
            absolute = self.start_layer + index
            if absolute < len(additions):
                hidden = hidden + additions[absolute]
        return hidden


class CachedTextFeatureBlock(TextFeatureBlock):
    """Chunk prefill using Merak's existing SDK cache operator and cache ownership.

    Each layer owns separate K/V caches. Host supplies the absolute past length,
    excludes padding from current length and slices this layer's dense DeepStack
    residual at the same chunk offset. Cache values retain the attention value
    divisor used by the non-cached feature graph.
    """

    def __init__(self, source, *, divisor=TEXT_FEATURE_DIVISOR):
        from xhquant.nn import LLMCacheV2

        super().__init__(source, sdk_ops=True, divisor=divisor)
        self.use_cache = True
        self.k_cache, self.v_cache = LLMCacheV2(axis=2), LLMCacheV2(axis=2)

    def forward(self, hidden, cos, sin, valid_length, past_length, key_cache, value_cache, deepstack):
        return self._forward(hidden, cos, sin, valid_length, past_length, key_cache, value_cache) + deepstack


class CachedMultimodalTextFeatureStack(nn.Module):
    """A registered layer group with separate cache tensors for every layer."""

    def __init__(self, layers, *, start_layer=0, divisor=TEXT_FEATURE_DIVISOR):
        super().__init__()
        self.layers = nn.ModuleList([CachedTextFeatureBlock(layer, divisor=divisor) for layer in layers])
        self.start_layer = start_layer

    def forward(
        self,
        hidden,
        cos,
        sin,
        valid_length,
        past_length,
        deepstack_0,
        deepstack_1,
        deepstack_2,
        key_caches,
        value_caches,
    ):
        additions = (deepstack_0, deepstack_1, deepstack_2)
        for index, layer in enumerate(self.layers):
            hidden = layer._forward(hidden, cos, sin, valid_length, past_length, key_caches[index], value_caches[index])
            absolute = self.start_layer + index
            if absolute < len(additions):
                hidden = hidden + additions[absolute]
        return hidden
