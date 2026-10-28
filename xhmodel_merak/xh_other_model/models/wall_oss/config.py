# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Validate Wall's two-expert architecture and compatible tensor dimensions."""


def validate_source_config(config):
    if config.get("model_type") != "qwen2_5_vl" or config.get("num_experts") != 2:
        raise ValueError("requires the Wall-OSS two-expert architecture")
    for key in ("attention_moe", "mlp_moe", "norm_moe", "causal_action_attention_mask"):
        if config.get(key) is not True:
            raise ValueError(f"Wall requires {key}=True")
    for key in ("use_x_pred", "use_adarms", "use_sliding_window"):
        if config.get(key, False):
            raise ValueError(f"unsupported Wall variant: {key}")
    if not config.get("proj_with_mask", True):
        raise ValueError("requires action/DOF-mask projection")
    experts = config.get("experts", [])
    dims = config.get("dim_inputs", [])
    if (
        len(experts) != 2
        or len(dims) != 2
        or [e.get("hidden_size") for e in experts] != dims
        or dims[0] != config.get("hidden_size")
    ):
        raise ValueError("expert hidden dimensions disagree")
    for value in [
        *dims,
        config.get("num_hidden_layers"),
        config.get("num_attention_heads"),
        config.get("num_key_value_heads"),
    ]:
        if type(value) is not int or value <= 0:
            raise ValueError("invalid Wall architecture dimension")
    if config["hidden_size"] % config["num_attention_heads"]:
        raise ValueError("hidden size must divide evenly into attention heads")
    if config.get("action_hidden_size", dims[1]) != dims[1]:
        raise ValueError("action projection and expert hidden dimensions disagree")
    if any(e.get("hidden_act", "silu") != "silu" for e in experts):
        raise ValueError("Wall experts require SiLU")
    head = config["hidden_size"] // config["num_attention_heads"]
    if head not in (64, 128) or config["num_attention_heads"] % config["num_key_value_heads"]:
        raise ValueError("unsupported grouped attention geometry")
    sections = config.get("rope_scaling", {}).get("mrope_section", [])
    if len(sections) != 3 or sum(sections) != head // 2:
        raise ValueError("M-RoPE sections disagree with attention head dimension")
    vision = config.get("vision_config", {})
    if any(
        vision.get(k) != v
        for k, v in dict(patch_size=14, temporal_patch_size=2, spatial_merge_size=2, window_size=112).items()
    ):
        raise ValueError("unexpected Wall patch/window architecture")
    if vision.get("out_hidden_size") != dims[0] or vision.get("hidden_size", 0) % vision.get("num_heads", 1):
        raise ValueError("vision output/head dimensions disagree")
    return config
