from __future__ import annotations

import re

import pytest

from xhmodel_merak.xh_llm.models.qwen4_exp.precision import (
    QWEN4_EXP_PRECISION_RULES,
    QWEN4_EXP_PROFILE,
    high_precision_patterns,
    validate_qwen4_exp_config,
)


def _config() -> dict:
    layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "qwen_sparse_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "qwen_sparse_attention",
    ]
    return {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "num_hidden_layers": len(layer_types),
            "layer_types": layer_types,
            "full_attention_interval": 4,
            "hc_count": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "ple_layer_ids": [2],
            "ngram_size": 3,
            "heads_per_ngram": 16,
            "ngram_vocab_size_base": 20_000_000,
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 128,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 4,
        },
    }


def _match(name: str):
    for rule in QWEN4_EXP_PRECISION_RULES:
        if re.search(rule.pattern, name):
            return rule
    return None


@pytest.mark.parametrize("qwen4", [False, True])
def test_model_config_does_not_mutate_reused_quant_scheme(qwen4):
    import copy

    from xhmodel_merak.xh_llm.models.qwen4_exp.configuration_qwen4_exp import XHQwen4ExpModelConfig
    from xhmodel_merak.xh_llm.types import BaseLLMModelConfig

    scheme = {"quant_type": "w8a8h1_sefp", "ops": {"MoeBlock": {"quant_type": "w4a8h0_ssfp"}}}
    original = copy.deepcopy(scheme)
    cls = XHQwen4ExpModelConfig if qwen4 else BaseLLMModelConfig
    first = cls(model_name="config_ownership", quant_scheme=scheme)
    second = cls(model_name="config_ownership", quant_scheme=scheme)
    assert scheme == original
    assert first.quant_scheme.to_dict() == second.quant_scheme.to_dict()
    if qwen4:
        assert first._export_quant_config == second._export_quant_config == original


def test_precision_contract_matches_qwen35_like_target():
    assert QWEN4_EXP_PROFILE.base_bits == 8
    assert QWEN4_EXP_PROFILE.expert_bits == 4
    assert QWEN4_EXP_PROFILE.activation_bits == 8

    routed = _match("model.language_model.layers.0.mlp.experts.3.gate_proj")
    assert routed is not None
    assert (routed.bits, routed.activation_bits) == (4, 8)

    fused_routed = _match("model.language_model.layers.0.mlp.experts.gate_up_proj")
    assert fused_routed is not None
    assert (fused_routed.bits, fused_routed.activation_bits) == (4, 8)

    shared = _match("model.language_model.layers.0.mlp.shared_expert.down_proj")
    assert shared is not None
    assert (shared.bits, shared.activation_bits) == (8, 8)

    full_attn = _match("model.language_model.layers.3.self_attn.q_proj")
    linear_attn = _match("model.language_model.layers.0.linear_attn.in_proj_qkv")
    assert full_attn is not None
    assert linear_attn is not None
    assert (full_attn.bits, full_attn.activation_bits) == (8, 8)
    assert (linear_attn.bits, linear_attn.activation_bits) == (8, 8)


def test_high_precision_rules_cover_qsa_ple_router_hyper_and_state():
    assert _match("model.language_model.layers.3.self_attn.indexer.index_qk_proj").bits == 16
    assert _match("model.language_model.layers.0.linear_attn.conv1d").bits == 16
    assert _match("model.language_model.layers.0.attn_hyper_connection.up_proj").bits == 16
    assert _match("model.language_model.layers.0.mlp.gate").bits == 16
    assert _match("model.ngram_embedding.weight").bits == 16
    assert any("hyper" in pattern for pattern in high_precision_patterns())


def test_validate_accepts_flat_qwen4_text_schema_inside_outer_config():
    summary = validate_qwen4_exp_config(_config())

    assert summary["model_type"] == "qwen4_exp"
    assert summary["text_model_type"] == "qwen4_exp_text"
    assert summary["full_attention_layers"] == 2
    assert summary["linear_attention_layers"] == 6
    assert summary["hc_count"] == 4
    assert summary["num_experts_per_tok"] == 10
    assert summary["ple_layer_ids"] == (2,)


def test_validate_accepts_raw_full_attention_spelling():
    config = _config()
    config["text_config"]["layer_types"] = [
        "full_attention" if layer_type == "qwen_sparse_attention" else layer_type
        for layer_type in config["text_config"]["layer_types"]
    ]

    summary = validate_qwen4_exp_config(config)

    assert summary["full_attention_layers"] == 2


def test_validate_rejects_zero_indexed_ple_layers():
    config = _config()
    config["text_config"]["ple_layer_ids"] = [0]

    with pytest.raises(ValueError, match="one-indexed"):
        validate_qwen4_exp_config(config)


def test_validate_rejects_incomplete_qsa_fields():
    config = _config()
    del config["text_config"]["indexer_budget"]

    with pytest.raises(ValueError, match="indexer_budget"):
        validate_qwen4_exp_config(config)


def test_validate_rejects_broken_layer_interval():
    config = _config()
    config["text_config"]["layer_types"][3] = "linear_attention"

    with pytest.raises(ValueError, match="both linear_attention and full/QSA attention|full_attention_interval"):
        validate_qwen4_exp_config(config)
