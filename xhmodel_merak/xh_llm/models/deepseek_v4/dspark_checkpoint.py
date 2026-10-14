# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Consume the three-block BF16 draft checkpoint produced by GPTQModel."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from transformers import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer, DeepseekV4HyperHead, DeepseekV4RMSNorm,
)

from .dspark import DSparkDraft, DSparkMarkov, DSparkSpec
from .norm import build_xh_rms_norm


_RENAMES = (
    ("attn.", "self_attn."), ("ffn.", "mlp."),
    ("attn_norm.", "input_layernorm."), ("ffn_norm.", "post_attention_layernorm."),
    ("hc_attn_fn", "attn_hc.fn"), ("hc_attn_base", "attn_hc.base"), ("hc_attn_scale", "attn_hc.scale"),
    ("hc_ffn_fn", "ffn_hc.fn"), ("hc_ffn_base", "ffn_hc.base"), ("hc_ffn_scale", "ffn_hc.scale"),
    ("self_attn.attn_sink", "self_attn.sinks"),
    ("self_attn.wq_a.", "self_attn.q_a_proj."), ("self_attn.wq_b.", "self_attn.q_b_proj."),
    ("self_attn.q_norm.", "self_attn.q_a_norm."), ("self_attn.wkv.", "self_attn.kv_proj."),
    ("self_attn.wo_a.", "self_attn.o_a_proj."), ("self_attn.wo_b.", "self_attn.o_b_proj."),
    ("mlp.gate.bias", "mlp.gate.e_score_correction_bias"),
    ("mlp.shared_experts.w1.", "mlp.shared_experts.gate_proj."),
    ("mlp.shared_experts.w3.", "mlp.shared_experts.up_proj."),
    ("mlp.shared_experts.w2.", "mlp.shared_experts.down_proj."),
)


def load_dspark_checkpoint(path: str | Path, target_lm_head: nn.Module,
                            *, context_chunk_length: int = 256,
                            block_size: int | None = None,
                            head_weight_bits: int = 5,
                            target_layer_ids: tuple[int, ...] | None = None,
                            max_context_length: int = 256 * 1024):
    """Load learned-router draft blocks without allocating a duplicate target head."""
    root = Path(path)
    manifest = json.loads((root / "dspark_bf16_preparation.json").read_text())
    if target_layer_ids is not None and tuple(manifest["target_layer_ids"]) != tuple(target_layer_ids):
        raise ValueError("DSpark checkpoint feature layers do not match the exported target")
    config = DeepseekV4Config.from_pretrained(root)
    if config.num_hidden_layers != 3 or config.layer_types != ["sliding_attention"] * 3:
        raise ValueError("DSpark preparation must contain three sliding-only blocks")
    if config.mlp_layer_types != ["moe"] * 3:
        raise ValueError("DSpark draft blocks must use learned routers")
    spec = DSparkSpec(
        target_layer_ids=tuple(manifest["target_layer_ids"]),
        block_size=int(block_size or config.dspark_block_size),
        context_chunk_length=context_chunk_length, max_context_length=max_context_length,
        sliding_window=config.sliding_window, hidden_size=config.hidden_size,
        head_dim=config.head_dim, rope_dim=config.qk_rope_head_dim, hc_mult=config.hc_mult,
        markov_rank=config.dspark_markov_rank, vocab_size=config.vocab_size,
        noise_token_id=config.dspark_noise_token_id, head_weight_bits=head_weight_bits,
    )
    if (target_lm_head.in_features, target_lm_head.out_features) != (spec.hidden_size, spec.vocab_size):
        raise ValueError("DSpark and target shared head shapes differ")
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    layers = []
    for layer_index in range(3):
        prefix = f"layers.{layer_index}."
        native = {}
        for filename in sorted({filename for key, filename in index.items() if key.startswith(prefix)}):
            with safe_open(root / filename, framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    if key in index and key.startswith(prefix):
                        native[key[len(prefix):]] = reader.get_tensor(key)
        state = {}
        for key, value in native.items():
            if re.match(r"mlp\.experts\.", key) or re.match(r"ffn\.experts\.", key):
                continue
            for original, renamed in _RENAMES:
                if key.startswith(original):
                    key = renamed + key[len(original):]
                    break
            # Attention leaves need a second rename after the namespace rename.
            for original, renamed in _RENAMES[10:]:
                if key.startswith(original):
                    key = renamed + key[len(original):]
                    break
            state[key] = value
        n, h, intermediate = config.n_routed_experts, config.hidden_size, config.moe_intermediate_size
        gate_up = torch.empty(n, 2 * intermediate, h, dtype=torch.bfloat16)
        down = torch.empty(n, h, intermediate, dtype=torch.bfloat16)
        for expert in range(n):
            gate_up[expert, :intermediate].copy_(native[f"ffn.experts.{expert}.w1.weight"])
            gate_up[expert, intermediate:].copy_(native[f"ffn.experts.{expert}.w3.weight"])
            down[expert].copy_(native[f"ffn.experts.{expert}.w2.weight"])
        state["mlp.experts.gate_up_proj"] = gate_up
        state["mlp.experts.down_proj"] = down
        with torch.device("meta"):
            layer = DeepseekV4DecoderLayer(config, layer_index)
        layer.load_state_dict(state, strict=True, assign=True)
        layers.append(layer)
        del native, state, gate_up, down
    heads = load_file(root / manifest["heads"])
    projection = nn.Linear(spec.hidden_size * len(spec.target_layer_ids), spec.hidden_size, bias=False,
                            device="meta", dtype=torch.float16)
    projection.weight = nn.Parameter(heads["main_proj.weight"].half(), requires_grad=False)
    main_norm = build_xh_rms_norm(spec.hidden_size, config.rms_norm_eps,
                                  weight=heads["main_norm.weight"].half())
    global_tensors = {}
    for filename in sorted({filename for key, filename in index.items() if not key.startswith("layers.")}):
        with safe_open(root / filename, framework="pt", device="cpu") as reader:
            for key in index:
                if index[key] == filename and not key.startswith("layers."):
                    global_tensors[key] = reader.get_tensor(key)
    with torch.device("meta"):
        hc_head = DeepseekV4HyperHead(config)
        norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
    hc_head.load_state_dict({"hc_fn": global_tensors["hc_head_fn"],
                             "hc_base": global_tensors["hc_head_base"],
                             "hc_scale": global_tensors["hc_head_scale"]}, strict=True, assign=True)
    norm.load_state_dict({"weight": global_tensors["norm.weight"]}, strict=True, assign=True)
    container = SimpleNamespace(config=config, model=SimpleNamespace(layers=layers, hc_head=hc_head, norm=norm))
    draft = DSparkDraft.from_hf(container, target_lm_head, spec).half().eval()
    markov = DSparkMarkov(heads["markov_head.markov_w1.weight"].half(),
                          heads["markov_head.markov_w2.weight"].half(),
                          confidence_weight=heads.get("confidence_head.proj.weight"))
    markov.half().eval()
    return spec, draft, projection, main_norm, markov


__all__ = ["load_dspark_checkpoint"]
