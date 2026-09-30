import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_merak.config import MerakConfigParser
from vllm_merak.constants import (
    HYBRID_CACHE_POLICY_PAGED,
)
from vllm_merak.model_executor.model_loader.ling_3_flash import (
    Ling3FlashMerakForCausalLM,
)
from vllm_merak.model_executor.model_loader.merak_for_causal_lm import MerakForCausalLM
from vllm_merak.model_executor.model_loader.qwen3_next import (
    Qwen3NextMerakForCausalLM,
)
from vllm_merak.model_executor.model_loader.qwen3_5 import (
    _resolve_merak_architecture_spec,
)


def _write_qwen3_next_export(
    tmp_path, *, source_architecture: str = "Qwen3NextForCausalLM"
) -> None:
    (tmp_path / "hf_config").mkdir()
    (tmp_path / "hf_config" / "config.json").write_text(
        json.dumps(
            {
                "architectures": [source_architecture],
                "model_type": "qwen3_next",
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "max_position_embeddings": 262144,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "merak_config.json").write_text(
        json.dumps({"xh_model": {"model_type": "hmonnx"}}),
        encoding="utf-8",
    )
    (tmp_path / "golden_meta_info.json").write_text(
        json.dumps(
            {
                "model_config": {
                    "model_type": "Qwen3NextForCausalLM",
                    "context_max_length": 262144,
                },
                "kv_cache": {
                    "linear_kv_cache_config": {
                        "num_layers": 3,
                        "cache_dtype": "float16",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("source_architecture", "loader_architecture", "cache_policy"),
    [
        (
            "Qwen3NextForCausalLM",
            "Qwen3NextMerakForCausalLM",
            HYBRID_CACHE_POLICY_PAGED,
        ),
        (
            "BailingMoeV3ForCausalLM",
            "Ling3FlashMerakForCausalLM",
            HYBRID_CACHE_POLICY_PAGED,
        ),
    ],
)
def test_native_hybrid_export_selects_dedicated_loader(
    tmp_path,
    source_architecture: str,
    loader_architecture: str,
    cache_policy: str,
) -> None:
    _write_qwen3_next_export(tmp_path, source_architecture=source_architecture)

    config_dict, config = MerakConfigParser().parse(
        tmp_path,
        trust_remote_code=False,
    )

    assert config_dict["source_architectures"] == [source_architecture]
    assert config_dict["architectures"] == [loader_architecture]
    assert config.merak_loader_architecture == loader_architecture
    assert config.merak_hybrid_cache_policy == cache_policy
    assert config.max_position_embeddings == 262144
    assert config.linear_kv_cache_config["num_layers"] == 3

    architecture_spec = _resolve_merak_architecture_spec(source_architecture)
    assert architecture_spec.supports_multimodal is False
    assert architecture_spec.requires_raw_input_tokens is False
    assert architecture_spec.supports_mrope is False


def test_qwen3_next_capabilities_match_text_only_hybrid_runtime() -> None:
    assert Qwen3NextMerakForCausalLM.is_hybrid is True
    assert Qwen3NextMerakForCausalLM.has_inner_state is True
    assert Qwen3NextMerakForCausalLM.supports_multimodal is False
    assert Qwen3NextMerakForCausalLM.requires_raw_input_tokens is False
    assert Qwen3NextMerakForCausalLM.supports_mrope is False
    assert Qwen3NextMerakForCausalLM.hybrid_cache_policy == HYBRID_CACHE_POLICY_PAGED
    assert Ling3FlashMerakForCausalLM.hybrid_cache_policy == HYBRID_CACHE_POLICY_PAGED


def test_absorbed_mla_placeholder_uses_exported_cache_widths_and_source_scale(monkeypatch) -> None:
    import vllm_merak.model_executor.model_loader.merak_for_causal_lm as merak_base

    captured = []

    class FakeAttention(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.append(kwargs)

    monkeypatch.setattr(merak_base, "MerakAttention", FakeAttention)
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    nn.Module.__init__(model)
    model.vllm_config = SimpleNamespace(cache_config=SimpleNamespace())
    model.key_cache_shape = (1, 1, 262144, 576)
    model.value_cache_shape = (1, 1, 262144, 512)
    hf_config = SimpleNamespace(
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=192,
        hidden_size=6144,
        num_hidden_layers=2,
        layer_types=["full_attention", "linear_attention"],
    )

    model._register_placeholder_attention_layers(hf_config)

    assert len(captured) == 1
    assert captured[0]["num_kv_heads"] == 1
    assert captured[0]["head_size"] == 576
    assert captured[0]["head_size_v"] == 512
    assert captured[0]["scale"] == pytest.approx(1.0 / math.sqrt(192))


def _ling_contract_model(
    *,
    golden_full_count: int = 7,
    golden_linear_count: int = 35,
):
    model = Ling3FlashMerakForCausalLM.__new__(Ling3FlashMerakForCausalLM)
    model.num_layers = golden_full_count
    model.config = SimpleNamespace(
        linear_kv_cache_config={"num_layers": golden_linear_count},
    )
    full_indices = [5, 11, 17, 23, 29, 35, 41]
    linear_indices = [
        layer_idx for layer_idx in range(42) if layer_idx not in full_indices
    ]
    model.placeholder_attn_layers = [
        SimpleNamespace(layer_name=f"model.layers.{layer_idx}.attn")
        for layer_idx in full_indices
    ]
    model.placeholder_linear_state_layers = [
        SimpleNamespace(layer_name=f"model.layers.{layer_idx}.linear_attn")
        for layer_idx in linear_indices
    ]
    return model


def test_ling_derives_bailing_grouped_hybrid_layer_types() -> None:
    model = _ling_contract_model()
    hf_config = SimpleNamespace(
        num_hidden_layers=42,
        layer_group_size=6,
    )

    layer_types = model._resolve_hybrid_layer_types(hf_config)

    assert len(layer_types) == 42
    assert [
        layer_idx
        for layer_idx, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    ] == [5, 11, 17, 23, 29, 35, 41]
    assert layer_types.count("linear_attention") == 35
    model._validate_hybrid_layer_contract(hf_config)


def test_ling_grouped_pattern_matches_bailing_trailing_group_rule() -> None:
    hf_config = SimpleNamespace(
        num_hidden_layers=8,
        layer_group_size=6,
    )

    layer_types = Ling3FlashMerakForCausalLM.derive_missing_hybrid_layer_types(
        hf_config
    )

    assert layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "full_attention",
        "full_attention",
    ]


@pytest.mark.parametrize("group_size", [None, 0, -1, True, 6.0])
def test_ling_rejects_malformed_layer_group_size(group_size) -> None:
    model = _ling_contract_model()
    hf_config = SimpleNamespace(
        num_hidden_layers=42,
        layer_group_size=group_size,
    )

    with pytest.raises(RuntimeError, match="positive integer layer_group_size"):
        model._resolve_hybrid_layer_types(hf_config)


def test_ling_does_not_replace_malformed_explicit_layer_types() -> None:
    model = _ling_contract_model()
    hf_config = SimpleNamespace(
        num_hidden_layers=42,
        layer_group_size=6,
        layer_types=["linear_attention"] * 41,
    )

    with pytest.raises(RuntimeError, match="expected 42, got 41"):
        model._resolve_hybrid_layer_types(hf_config)


@pytest.mark.parametrize(
    ("golden_full_count", "golden_linear_count"),
    [(8, 35), (7, 36)],
)
def test_ling_rejects_derived_hf_and_golden_count_mismatch(
    golden_full_count: int,
    golden_linear_count: int,
) -> None:
    model = _ling_contract_model(
        golden_full_count=golden_full_count,
        golden_linear_count=golden_linear_count,
    )
    hf_config = SimpleNamespace(
        num_hidden_layers=42,
        layer_group_size=6,
    )

    with pytest.raises(RuntimeError, match="HF/golden hybrid layer count mismatch"):
        model._validate_hybrid_layer_contract(hf_config)


def test_qwen3_next_missing_explicit_layer_types_remains_fail_closed() -> None:
    model = Qwen3NextMerakForCausalLM.__new__(Qwen3NextMerakForCausalLM)
    hf_config = SimpleNamespace(num_hidden_layers=4)

    with pytest.raises(RuntimeError, match="expected 4, got missing"):
        model._resolve_hybrid_layer_types(hf_config)


def test_qwen3_next_does_not_apply_qwen35_mrope_positions() -> None:
    model = Qwen3NextMerakForCausalLM.__new__(Qwen3NextMerakForCausalLM)
    net_inputs = (torch.zeros(1), torch.ones(1))

    result = model._apply_runtime_positions(
        net_inputs,
        torch.arange(16),
        16,
    )

    assert result is net_inputs
