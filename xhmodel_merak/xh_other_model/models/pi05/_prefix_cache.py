# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# Copyright 2026 HOUMO AI (modifications)
# Licensed under the Apache License, Version 2.0.
# See licenses/lerobot/LICENSE.
# Source: LeRobot v0.6.0, lerobot/policies/pi_gemma.py (PiGemmaModel.forward).
"""Collect native PiGemma prefix KV using an already constructed attention mask.

The export calibration path owns its complete four-dimensional prefix mask.
LeRobot 0.6.0's full model wrapper still passes the removed cache_position
argument to Transformers.create_causal_mask. Use the native decoder layers
with the provided mask, preserving their actual attention/norm/cache behavior
without changing either library's functions or the policy's forward method.
"""

import torch
from transformers.cache_utils import DynamicCache


@torch.no_grad()
def collect_prefix_cache(model, embeddings, attention_mask, position_ids):
    if attention_mask.ndim != 4 or attention_mask.shape[-2:] != (embeddings.shape[1], embeddings.shape[1]):
        raise ValueError("Pi0.5 prefix cache requires a complete square 4D attention mask")
    if position_ids.shape != embeddings.shape[:2]:
        raise ValueError("Pi0.5 prefix position IDs must match the embedding batch and length")
    if not model.layers:
        raise ValueError("Pi0.5 prefix cache requires at least one decoder layer")
    hidden = embeddings.to(model.layers[0].self_attn.q_proj.weight.dtype)
    rotary = model.rotary_emb(hidden, position_ids)
    cache = DynamicCache()
    for layer in model.layers[:model.config.num_hidden_layers]:
        hidden = layer(hidden, attention_mask=attention_mask.to(hidden.dtype), position_ids=position_ids,
                       past_key_values=cache, use_cache=True, position_embeddings=rotary, adarms_cond=None)
    # Final norm affects returned hidden states, but cannot change stored KV.
    return cache
