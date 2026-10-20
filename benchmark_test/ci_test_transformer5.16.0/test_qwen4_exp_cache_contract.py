from __future__ import annotations

from types import SimpleNamespace

import pytest

from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import (
    DELTA_CONV_STATE_IDX,
    LINEAR_ATTENTION,
    PLE_CONV_STATE_IDX,
    PLE_NGRAM_STATE_IDX,
    QWEN_SPARSE_ATTENTION,
    Qwen4ExpCacheContract,
)


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def make_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=4,
        layer_types=[
            LINEAR_ATTENTION,
            LINEAR_ATTENTION,
            LINEAR_ATTENTION,
            QWEN_SPARSE_ATTENTION,
        ],
        hidden_size=2560,
        hc_count=4,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=256,
        indexer_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        ple_conv_kernel_size=4,
        ngram_size=3,
        ple_layer_ids=[2],
    )


def test_builds_four_layer_sparse_pattern_with_ple_state_indices() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(make_config())

    assert contract.layer_types == (
        LINEAR_ATTENTION,
        LINEAR_ATTENTION,
        LINEAR_ATTENTION,
        QWEN_SPARSE_ATTENTION,
    )
    assert contract.ple_layer_indices == (1,)
    assert contract.linear_conv_dim == 10240
    assert contract.ple_hidden_size == 10240
    assert contract.ple_short_conv_state_len == 9
    assert contract.ngram_context_len == 2

    layer1 = contract.layers[1]
    by_kind = {tensor.state_kind: tensor for tensor in layer1.tensors}
    assert by_kind["delta_conv"].state_idx == DELTA_CONV_STATE_IDX
    assert by_kind["delta_recurrent"].state_idx == DELTA_CONV_STATE_IDX
    assert by_kind["ple_conv"].state_idx == PLE_CONV_STATE_IDX
    assert by_kind["ple_ngram_history"].state_idx == PLE_NGRAM_STATE_IDX
    assert by_kind["ple_conv"].shape == ("batch", 10240, 9)
    assert by_kind["ple_ngram_history"].shape == ("batch", 2)

    layer3 = contract.layers[3]
    assert [tensor.state_kind for tensor in layer3.tensors] == [
        "kv_key",
        "kv_value",
        "qsa_indexer_key",
        "qsa_indexer_tail",
        "qsa_key_tail",
        "qsa_value_tail",
    ]
    assert layer3.tensors[0].shape == ("batch", 2, "cache_length", 256)
    assert layer3.tensors[2].shape == ("batch", 1, "qsa_pool_length", 128)
    assert layer3.tensors[3].shape == ("batch", 4, 128)


def test_checkpoint_and_transformers_configs_have_the_same_cache_contract() -> None:
    from transformers import Qwen4ExpTextConfig

    raw = vars(make_config()).copy()
    raw["layer_types"] = [*raw["layer_types"][:-1], "full_attention"]
    raw.update(indexer_n_heads=4, indexer_kv_heads=1, indexer_budget=2048, indexer_compress_ratio=4, eos_token_id=0)
    raw["rope_parameters"] = {"rope_type": "default", "partial_rotary_factor": 0.25,
                              "mrope_section": [11, 11, 10]}
    loaded = Qwen4ExpTextConfig(**{**raw, "layer_types": list(raw["layer_types"])})
    assert raw["layer_types"][-1] == "full_attention"
    assert loaded.layer_types[-1] == QWEN_SPARSE_ATTENTION
    assert Qwen4ExpCacheContract.from_text_config(raw).to_dict() == (
        Qwen4ExpCacheContract.from_text_config(loaded).to_dict()
    )


def test_round_trip_serialization_preserves_contract() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(make_config())

    restored = Qwen4ExpCacheContract.from_dict(contract.to_dict())

    assert restored == contract
    assert "past_position_ids" not in restored.expected_keys()


def test_validate_cache_shapes_accepts_complete_cache() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(make_config())
    cache_tensors = {
        "delta_conv_state_0": FakeTensor((2, 10240, 4)),
        "delta_recurrent_state_0": FakeTensor((2, 48, 128, 128)),
        "delta_conv_state_1": FakeTensor((2, 10240, 4)),
        "delta_recurrent_state_1": FakeTensor((2, 48, 128, 128)),
        "ple_conv_state_1": FakeTensor((2, 10240, 9)),
        "ple_ngram_history_1": FakeTensor((2, 2)),
        "delta_conv_state_2": FakeTensor((2, 10240, 4)),
        "delta_recurrent_state_2": FakeTensor((2, 48, 128, 128)),
        "key_cache_3": FakeTensor((2, 2, 7, 256)),
        "value_cache_3": FakeTensor((2, 2, 7, 256)),
        "qsa_indexer_key_cache_3": FakeTensor((2, 1, 1, 128)),
        "qsa_indexer_key_tail_3": FakeTensor((2, 4, 128)),
        "qsa_key_tail_3": FakeTensor((2, 2, 3, 256)),
        "qsa_value_tail_3": FakeTensor((2, 2, 3, 256)),
    }

    contract.validate_cache_shapes(cache_tensors)


def test_validate_cache_shapes_rejects_missing_key() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(make_config())

    with pytest.raises(ValueError, match="missing Qwen4Exp cache tensor"):
        contract.validate_cache_shapes({})


def test_validate_cache_shapes_rejects_illegal_shape() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(make_config())
    cache_tensors = {name: FakeTensor((3, 2, 7)) for name in contract.expected_keys()}
    cache_tensors.update(
        {
            "delta_conv_state_0": FakeTensor((2, 10240, 4)),
            "delta_recurrent_state_0": FakeTensor((2, 48, 128, 128)),
            "delta_conv_state_1": FakeTensor((2, 10240, 4)),
            "delta_recurrent_state_1": FakeTensor((2, 48, 128, 128)),
            "ple_conv_state_1": FakeTensor((2, 10240, 8)),
            "ple_ngram_history_1": FakeTensor((2, 2)),
            "delta_conv_state_2": FakeTensor((2, 10240, 4)),
            "delta_recurrent_state_2": FakeTensor((2, 48, 128, 128)),
            "key_cache_3": FakeTensor((2, 2, 7, 256)),
            "value_cache_3": FakeTensor((2, 2, 7, 256)),
            "qsa_indexer_key_cache_3": FakeTensor((2, 1, 1, 128)),
            "qsa_indexer_key_tail_3": FakeTensor((2, 4, 128)),
            "qsa_key_tail_3": FakeTensor((2, 2, 3, 256)),
            "qsa_value_tail_3": FakeTensor((2, 2, 3, 256)),
        }
    )

    with pytest.raises(ValueError, match="ple_conv_state_1 shape mismatch"):
        contract.validate_cache_shapes(cache_tensors)


def test_validate_cache_shapes_rejects_inconsistent_symbolic_dimensions() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(make_config())
    cache_tensors = {
        "delta_conv_state_0": FakeTensor((2, 10240, 4)),
        "delta_recurrent_state_0": FakeTensor((2, 48, 128, 128)),
        "delta_conv_state_1": FakeTensor((2, 10240, 4)),
        "delta_recurrent_state_1": FakeTensor((2, 48, 128, 128)),
        "ple_conv_state_1": FakeTensor((2, 10240, 9)),
        "ple_ngram_history_1": FakeTensor((2, 2)),
        "delta_conv_state_2": FakeTensor((2, 10240, 4)),
        "delta_recurrent_state_2": FakeTensor((2, 48, 128, 128)),
        "key_cache_3": FakeTensor((2, 2, 7, 256)),
        "value_cache_3": FakeTensor((2, 2, 7, 256)),
        "qsa_indexer_key_cache_3": FakeTensor((1, 1, 1, 128)),
        "qsa_indexer_key_tail_3": FakeTensor((2, 4, 128)),
        "qsa_key_tail_3": FakeTensor((2, 2, 3, 256)),
        "qsa_value_tail_3": FakeTensor((2, 2, 3, 256)),
    }

    with pytest.raises(ValueError, match="symbolic dim 'batch'"):
        contract.validate_cache_shapes(cache_tensors)
