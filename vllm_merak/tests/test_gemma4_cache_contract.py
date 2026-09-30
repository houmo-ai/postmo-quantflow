import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.attention import Attention
from vllm.v1.kv_cache_interface import SlidingWindowSpec, UniformTypeKVCacheSpecs
from vllm_merak.attention.backends.merak import MerakAttentionBackend
from vllm_merak.config import MerakConfigParser
from vllm_merak.hmfp_kv_cache import (
    HMFPFullAttentionSpec,
    HMFPSlidingWindowSpec,
)
from vllm_merak.model_executor.model_loader.gemma4_cache_contract import (
    parse_gemma4_cache_contract,
)
from vllm_merak.model_executor.model_loader.merak_for_causal_lm import (
    MerakAttention,
    MerakForCausalLM,
    bind_gemma4_cache_groups,
)
from vllm_merak.model_executor.register import get_registered_model_cls


def _hf(num_layers: int = 3, *, layer_types=None):
    if layer_types is None:
        layer_types = ["full_attention", "sliding_attention", "sliding_attention"]
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=num_layers,
            layer_types=layer_types,
            num_attention_heads=4,
            num_key_value_heads=4,
            hidden_size=512,
        )
    )


def _meta(**overrides):
    meta = {
        "attention_contract_version": 2,
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention"],
        "layer_cache_types": ["full_attention", "sliding_attention"],
        "layer_cache_indices": [0, 1, 1],
        "layer_cache_owner_indices": [0, 1],
        "layer_kv_shapes": [[1, 4, 262464, 128], [1, 4, 1344, 128]],
        "sliding_window": 1024,
        "sliding_kv_cache_input_mode": "slice_window",
        "prefill_chunk_length": 320,
        "bidirectional_vision_attention": True,
        "max_mm_ranges_per_chunk": 1,
        "model_config": {"spec_decode_mode": "none"},
    }
    meta.update(overrides)
    return meta


def test_contract_maps_layers_to_shared_heterogeneous_caches() -> None:
    contract = parse_gemma4_cache_contract(hf_config=_hf(), hmonnx_meta=_meta())

    assert [
        (binding.layer_idx, binding.cache_group_idx, binding.cache_index)
        for binding in contract.bindings
    ] == [(0, None, 0), (1, None, 1), (2, None, 1)]
    assert contract.prefill_chunk_size == 320
    assert contract.shape_for_cache(0)[2] == 262464
    assert contract.shape_for_cache(1)[2] == 1344
    assert contract.binding_for_layer("placeholder_attn_layers.2").cache_index == 1
    assert contract.bindings[0].sliding_window is None
    assert contract.bindings[1].sliding_window == 1024
    assert contract.bindings[1].kv_shape is contract.bindings[2].kv_shape
    assert [binding.cache_owner_layer_idx for binding in contract.bindings] == [
        0,
        1,
        1,
    ]


def test_mixed_contract_binds_full_layers_and_only_physical_sliding_owners() -> None:
    contract = parse_gemma4_cache_contract(
        hf_config=_hf(),
        hmonnx_meta=_meta(
            attention_lowering="full_flash_attention",
            uses_sliding_flash_attention_v2=False,
        ),
    )

    assert contract.attention_lowering == "full_flash_attention"
    assert [binding.layer_idx for binding in contract.context_bindings] == [0, 1]
    assert contract.context_bindings[0].attention_kind == "full_attention"
    assert contract.context_bindings[1].attention_kind == "sliding_attention"

    with pytest.raises(ValueError, match="uses_sliding_flash_attention_v2"):
        parse_gemma4_cache_contract(
            hf_config=_hf(),
            hmonnx_meta=_meta(
                attention_lowering="full_flash_attention",
                uses_sliding_flash_attention_v2=True,
            ),
        )


def test_contract_derives_sliding_cache_width_from_exported_window() -> None:
    contract = parse_gemma4_cache_contract(
        hf_config=_hf(),
        hmonnx_meta=_meta(
            sliding_window=512,
            layer_kv_shapes=[[1, 4, 131392, 128], [1, 4, 832, 128]],
        ),
    )

    assert contract.shape_for_cache(1)[2] == 832
    assert contract.bindings[1].sliding_window == 512


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("attention_contract_version", 1, "contract-v2"),
        ("sliding_kv_cache_input_mode", "legacy_full", "slice_window"),
        ("prefill_chunk_length", 279, "at least 280"),
        ("bidirectional_vision_attention", False, "bidirectional"),
        ("max_mm_ranges_per_chunk", 0, "max_mm_ranges_per_chunk"),
        ("layer_types", ["full_attention", "mystery"], "mystery"),
        ("layer_types", ["full_attention", "linear_attention"], "linear_attention"),
        ("layer_types", ["full_attention", "full_attention"], "sliding_attention"),
        ("sliding_window", 0, "sliding_window"),
    ],
)
def test_contract_rejects_invalid_export(field, value, message) -> None:
    with pytest.raises((ValueError, NotImplementedError), match=message):
        parse_gemma4_cache_contract(
            hf_config=_hf(2, layer_types=_meta(**{field: value}).get("layer_types")),
            hmonnx_meta=_meta(**{field: value}),
        )


def test_contract_accepts_readonly_page_attention_mtp() -> None:
    meta = _meta(model_config={"spec_decode_mode": "mtp"})
    meta["spec_decode"] = {
        "runtime_contract_version": 2,
        "mode": "mtp",
        "num_draft_tokens": 4,
        "verify_length": 5,
        "draft": {
            "abi": "gemma4_mtp_readonly_page_attention_v1",
            "cache_mutation": "read_only",
            "cache_binding": "target_attention_type_owner",
        },
    }

    contract = parse_gemma4_cache_contract(
        hf_config=_hf(),
        hmonnx_meta=meta,
    )

    assert contract.version == 2
    assert contract.shape_for_cache(1)[2] == 1344


def test_contract_accepts_causal_readonly_page_attention_mtp() -> None:
    meta = _meta(model_config={"spec_decode_mode": "mtp"})
    meta["spec_decode"] = {
        "runtime_contract_version": 2,
        "mode": "mtp",
        "num_draft_tokens": 4,
        "verify_length": 5,
        "draft": {
            "abi": "gemma4_mtp_readonly_page_attention_v2",
            "attention_lowering": "causal",
            "cache_mutation": "read_only",
            "cache_binding": "target_attention_type_owner",
        },
    }

    contract = parse_gemma4_cache_contract(
        hf_config=_hf(),
        hmonnx_meta=meta,
    )

    assert contract.version == 2
    assert contract.shape_for_cache(1)[2] == 1344


def test_contract_accepts_mixed_readonly_page_attention_mtp() -> None:
    meta = _meta(
        attention_lowering="full_flash_attention",
        uses_sliding_flash_attention_v2=False,
        model_config={"spec_decode_mode": "mtp"},
    )
    meta["spec_decode"] = {
        "runtime_contract_version": 2,
        "mode": "mtp",
        "num_draft_tokens": 4,
        "verify_length": 5,
        "draft": {
            "abi": "gemma4_mtp_readonly_mixed_attention_v3",
            "attention_lowering": "full_page_sliding_unfused",
            "cache_mutation": "read_only",
            "cache_binding": "target_attention_type_owner",
        },
    }

    contract = parse_gemma4_cache_contract(hf_config=_hf(), hmonnx_meta=meta)

    assert contract.attention_lowering == "full_flash_attention"

    meta["spec_decode"]["draft"]["attention_lowering"] = "causal"
    with pytest.raises(ValueError, match="full-page/sliding-unfused"):
        parse_gemma4_cache_contract(hf_config=_hf(), hmonnx_meta=meta)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (None, "spec_decode contract"),
        (
            {
                "runtime_contract_version": 2,
                "mode": "mtp",
                "num_draft_tokens": 4,
                "verify_length": 5,
                "draft": {
                    "abi": "gemma4_mtp_dense_shared_kv_v1",
                    "cache_mutation": "read_only",
                    "cache_binding": "target_attention_type_owner",
                },
            },
            "read-only PageAttention ABI",
        ),
    ],
)
def test_contract_rejects_unsafe_mtp_cache_abi(mutation, message) -> None:
    meta = _meta(model_config={"spec_decode_mode": "mtp"})
    if mutation is not None:
        meta["spec_decode"] = mutation

    with pytest.raises(ValueError, match=message):
        parse_gemma4_cache_contract(hf_config=_hf(), hmonnx_meta=meta)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"layer_cache_indices": None}, "layer_cache_indices"),
        ({"layer_cache_indices": [0, 2, 1]}, "layer_cache_indices"),
        ({"layer_cache_indices": [0, 1]}, "layer_cache_indices"),
        ({"layer_cache_types": ["full_attention"]}, "layer_cache_types"),
        ({"layer_cache_owner_indices": [0]}, "layer_cache_owner_indices"),
        ({"layer_cache_owner_indices": [0, 2]}, "cache owner"),
        ({"layer_kv_shapes": [[1, 4, 262464, 128]]}, "layer_kv_shapes"),
        ({"layer_kv_shapes": [[1, 4, 262464, 128], [1, 4, 0, 128]]}, "layer_kv_shapes"),
        (
            {"layer_kv_shapes": [[1, 4, 262464, 128], [1, 4, 1024, 128]]},
            "layer_kv_shapes",
        ),
        (
            {"layer_cache_types": ["full_attention", "full_attention"]},
            "layer_cache_types",
        ),
    ],
)
def test_contract_rejects_invalid_cache_mapping(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        parse_gemma4_cache_contract(hf_config=_hf(), hmonnx_meta=_meta(**overrides))


def test_contract_rejects_hf_layer_count_or_type_mismatch() -> None:
    with pytest.raises(ValueError, match="num_hidden_layers"):
        parse_gemma4_cache_contract(hf_config=_hf(4), hmonnx_meta=_meta())
    with pytest.raises(ValueError, match="HF layer_types"):
        parse_gemma4_cache_contract(
            hf_config=_hf(
                layer_types=["sliding_attention", "full_attention", "sliding_attention"]
            ),
            hmonnx_meta=_meta(),
        )


def _write_merak_model(tmp_path, *, architecture: str, meta: dict) -> None:
    hf_dir = tmp_path / "hf_config"
    hf_dir.mkdir()
    layer_types = meta.get("layer_types")
    if layer_types is None:
        layer_types = ["full_attention"] * 3
    (hf_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": [architecture],
                "model_type": "gemma4"
                if architecture.startswith("Gemma4")
                else "qwen3_5",
                "hidden_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_hidden_layers": 3,
                "vocab_size": 256,
                "max_position_embeddings": 262144,
                "text_config": {
                    "architectures": [architecture],
                    "model_type": "gemma4_text",
                    "hidden_size": 128,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "num_hidden_layers": 3,
                    "vocab_size": 256,
                    "max_position_embeddings": 262144,
                    "layer_types": layer_types,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "merak_config.json").write_text(
        json.dumps({"xh_model": {"model_type": "hmonnx"}}), encoding="utf-8"
    )
    (tmp_path / "golden_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.mark.parametrize(
    "architecture", ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"]
)
def test_gemma4_source_architecture_dispatch_is_exact(tmp_path, architecture) -> None:
    _write_merak_model(tmp_path, architecture=architecture, meta=_meta())

    config_dict, config = MerakConfigParser().parse(tmp_path, trust_remote_code=False)

    assert config.merak_loader_architecture == "Gemma4MerakForCausalLM"
    assert config.attention_contract_version == 2
    assert config.layer_cache_indices == [0, 1, 1]
    assert config.gemma4_cache_contract.shape_for_cache(1)[2] == 1344
    assert config_dict["architectures"] == ["Gemma4MerakForCausalLM"]
    assert json.loads(repr(config))["gemma4_cache_contract"]["version"] == 2
    assert get_registered_model_cls(architecture).__name__ == "Gemma4MerakForCausalLM"


def test_gemma4_lookalike_architecture_does_not_enter_gemma_branch(tmp_path) -> None:
    _write_merak_model(
        tmp_path, architecture="Gemma4ForSequenceClassification", meta=_meta()
    )

    _, config = MerakConfigParser().parse(tmp_path, trust_remote_code=False)

    assert config.merak_loader_architecture != "Gemma4MerakForCausalLM"
    assert not hasattr(config, "gemma4_cache_contract")


def test_gemma4_contract_and_loader_dispatch_cannot_be_overridden(tmp_path) -> None:
    _write_merak_model(
        tmp_path, architecture="Gemma4ForConditionalGeneration", meta=_meta()
    )

    _, config = MerakConfigParser().parse(
        tmp_path,
        trust_remote_code=False,
        hf_overrides={
            "attention_contract_version": 1,
            "merak_loader_architecture": "Qwen3_5MerakTextForCausalLM",
        },
    )

    assert config.attention_contract_version == 2
    assert config.merak_loader_architecture == "Gemma4MerakForCausalLM"
    assert config.gemma4_cache_contract.version == 2


def test_qwen_architecture_dispatch_does_not_enter_gemma_branch(tmp_path) -> None:
    _write_merak_model(
        tmp_path,
        architecture="Qwen3_5ForConditionalGeneration",
        meta={"model_config": {"spec_decode_mode": "none"}},
    )

    _, config = MerakConfigParser().parse(tmp_path, trust_remote_code=False)

    assert config.merak_loader_architecture.startswith("Qwen3_5Merak")
    assert not hasattr(config, "gemma4_cache_contract")


def _hmfp_spec(attention_kind: str):
    spec_cls = (
        HMFPFullAttentionSpec
        if attention_kind == "full_attention"
        else HMFPSlidingWindowSpec
    )
    kwargs = {
        "block_size": 64,
        "num_kv_heads": 4,
        "head_size": 128,
        "head_size_v": 128,
        "dtype": torch.uint8,
        "sefp_cache_dtype": "sefp_k8_v8",
    }
    if attention_kind == "sliding_attention":
        kwargs["sliding_window"] = 1024
    return spec_cls(**kwargs)


def test_merak_attention_dispatches_exact_hmfp_spec() -> None:
    cache_config = SimpleNamespace(
        merak_logical_kv_cache_dtype="sefp_k8_v8",
        cache_dtype="uint8",
        block_size=64,
    )

    def fake_attention_init(instance, *args, **kwargs):
        del args
        nn.Module.__init__(instance)
        # Match vLLM Attention.__init__: its own optional sliding-window
        # argument overwrites attributes assigned before super().__init__.
        instance.sliding_window = kwargs.get("per_layer_sliding_window")
        instance.num_kv_heads = 4
        instance.head_size = 128
        instance.head_size_v = 128
        instance.kv_cache_torch_dtype = torch.uint8

    with patch.object(Attention, "__init__", fake_attention_init):
        full = MerakAttention(
            cache_config=cache_config,
            attention_kind="full_attention",
            sliding_window=None,
        )
        local = MerakAttention(
            cache_config=cache_config,
            attention_kind="sliding_attention",
            sliding_window=1024,
        )
        shared_local = MerakAttention(
            cache_config=cache_config,
            attention_kind="sliding_attention",
            sliding_window=1024,
            kv_sharing_target_layer_name="placeholder_attn_layers.1",
        )

    vllm_config = SimpleNamespace(cache_config=cache_config)
    assert type(full.get_kv_cache_spec(vllm_config)) is HMFPFullAttentionSpec
    local_spec = local.get_kv_cache_spec(vllm_config)
    assert type(local_spec) is HMFPSlidingWindowSpec
    assert local_spec.sliding_window == 1024
    assert shared_local.get_kv_cache_spec(vllm_config) is None


def test_merak_raw_fp_sliding_spec_allows_mixed_page_padding() -> None:
    cache_config = SimpleNamespace(block_size=64)
    native_spec = SlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.float16,
        sliding_window=512,
    )
    layer = MerakAttention.__new__(MerakAttention)
    nn.Module.__init__(layer)
    layer.kv_sharing_target_layer_name = None
    layer.merak_logical_kv_cache_dtype = "float16"
    layer.attention_kind = "sliding_attention"

    with patch.object(Attention, "get_kv_cache_spec", return_value=native_spec):
        spec = layer.get_kv_cache_spec(SimpleNamespace(cache_config=cache_config))

    assert type(spec) is SlidingWindowSpec
    assert spec.indexes_kv_by_block_stride is True
    assert MerakAttentionBackend.indexes_kv_by_block_stride() is True


def test_gemma4_placeholder_chain_keeps_five_sliding_windows_per_full_layer() -> None:
    layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    shapes = [[1, 8, 1344, 256]] * 5 + [[1, 2, 262144, 512]]
    contract = parse_gemma4_cache_contract(
        hf_config=_hf(6, layer_types=layer_types),
        hmonnx_meta=_meta(
            layer_types=layer_types,
            layer_cache_types=layer_types,
            layer_cache_indices=list(range(6)),
            layer_cache_owner_indices=list(range(6)),
            layer_kv_shapes=shapes,
        ),
    )
    cache_config = SimpleNamespace(
        merak_logical_kv_cache_dtype="sefp_k8_v8",
        cache_dtype="uint8",
        block_size=64,
    )
    vllm_config = SimpleNamespace(cache_config=cache_config)
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    nn.Module.__init__(model)
    model.gemma4_cache_contract = contract
    model.vllm_config = vllm_config

    def fake_attention_init(instance, *args, **kwargs):
        del args
        nn.Module.__init__(instance)
        instance.layer_name = kwargs["prefix"]
        instance.kv_sharing_target_layer_name = kwargs.get(
            "kv_sharing_target_layer_name"
        )
        instance.sliding_window = kwargs.get("per_layer_sliding_window")
        instance.num_kv_heads = kwargs["num_kv_heads"]
        instance.head_size = kwargs["head_size"]
        instance.head_size_v = kwargs["head_size"]
        instance.kv_cache_torch_dtype = torch.uint8

    with patch.object(Attention, "__init__", fake_attention_init):
        model._register_placeholder_attention_layers(
            _hf(6, layer_types=layer_types).text_config
        )

    specs = [
        layer.get_kv_cache_spec(vllm_config) for layer in model.placeholder_attn_layers
    ]
    assert [type(spec) for spec in specs] == [
        HMFPSlidingWindowSpec,
        HMFPSlidingWindowSpec,
        HMFPSlidingWindowSpec,
        HMFPSlidingWindowSpec,
        HMFPSlidingWindowSpec,
        HMFPFullAttentionSpec,
    ]
    assert [spec.sliding_window for spec in specs[:5]] == [1024] * 5


def test_unknown_attention_kind_has_no_full_fallback() -> None:
    with pytest.raises(ValueError, match="mystery"):
        MerakAttention(attention_kind="mystery", sliding_window=None)


def test_placeholder_registration_preserves_every_layer_binding() -> None:
    contract = parse_gemma4_cache_contract(hf_config=_hf(), hmonnx_meta=_meta())
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    nn.Module.__init__(model)
    model.gemma4_cache_contract = contract
    model.vllm_config = SimpleNamespace(cache_config=SimpleNamespace())

    def fake_merak_attention_init(instance, *args, **kwargs):
        del args
        nn.Module.__init__(instance)
        instance.layer_name = kwargs["prefix"]
        instance.attention_kind = kwargs["attention_kind"]
        instance.sliding_window = kwargs["sliding_window"]

    with patch.object(MerakAttention, "__init__", fake_merak_attention_init):
        model._register_placeholder_attention_layers(_hf().text_config)

    assert [layer.layer_name for layer in model.placeholder_attn_layers] == [
        "placeholder_attn_layers.0",
        "placeholder_attn_layers.1",
        "placeholder_attn_layers.2",
    ]
    assert [layer.attention_kind for layer in model.placeholder_attn_layers] == [
        "full_attention",
        "sliding_attention",
        "sliding_attention",
    ]
    assert [binding.cache_index for binding in model.page_attention_bindings] == [
        0,
        1,
        1,
    ]


def test_mixed_placeholder_registration_uses_fp16_only_for_sliding_cache() -> None:
    contract = parse_gemma4_cache_contract(
        hf_config=_hf(),
        hmonnx_meta=_meta(
            attention_lowering="full_flash_attention",
            uses_sliding_flash_attention_v2=False,
        ),
    )
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    nn.Module.__init__(model)
    model.gemma4_cache_contract = contract
    original_cache_config = SimpleNamespace(
        merak_logical_kv_cache_dtype="sefp_k8_v8",
        cache_dtype="uint8",
        block_size=64,
    )
    model.vllm_config = SimpleNamespace(cache_config=original_cache_config)
    captured = []

    def fake_merak_attention_init(instance, *args, **kwargs):
        del args
        nn.Module.__init__(instance)
        instance.layer_name = kwargs["prefix"]
        instance.kv_sharing_target_layer_name = kwargs.get(
            "kv_sharing_target_layer_name"
        )
        captured.append(
            (
                kwargs["attention_kind"],
                kwargs["cache_config"].merak_logical_kv_cache_dtype,
                kwargs["cache_config"].cache_dtype,
            )
        )

    with patch.object(MerakAttention, "__init__", fake_merak_attention_init):
        model._register_placeholder_attention_layers(_hf().text_config)

    assert captured == [
        ("full_attention", "sefp_k8_v8", "uint8"),
        ("sliding_attention", "float16", "float16"),
        ("sliding_attention", "float16", "float16"),
    ]
    assert original_cache_config.merak_logical_kv_cache_dtype == "sefp_k8_v8"
    assert original_cache_config.cache_dtype == "uint8"


def test_placeholder_registration_uses_per_cache_geometry_and_declares_kv_sharing() -> (
    None
):
    contract = parse_gemma4_cache_contract(
        hf_config=_hf(),
        hmonnx_meta=_meta(
            layer_kv_shapes=[[1, 8, 262464, 512], [1, 4, 1344, 256]],
        ),
    )
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    nn.Module.__init__(model)
    model.gemma4_cache_contract = contract
    model.vllm_config = SimpleNamespace(cache_config=SimpleNamespace())
    calls = []

    def fake_merak_attention_init(instance, *args, **kwargs):
        del args
        nn.Module.__init__(instance)
        calls.append(kwargs)
        instance.layer_name = kwargs["prefix"]

    with patch.object(MerakAttention, "__init__", fake_merak_attention_init):
        model._register_placeholder_attention_layers(_hf().text_config)

    assert [(call["num_kv_heads"], call["head_size"]) for call in calls] == [
        (8, 512),
        (4, 256),
        (4, 256),
    ]
    assert calls[0]["kv_sharing_target_layer_name"] is None
    assert calls[1]["kv_sharing_target_layer_name"] is None
    assert calls[2]["kv_sharing_target_layer_name"] == "placeholder_attn_layers.1"


def test_bind_gemma4_cache_groups_uses_exact_layer_names_with_reversed_groups() -> None:
    contract = parse_gemma4_cache_contract(hf_config=_hf(), hmonnx_meta=_meta())
    groups = [
        SimpleNamespace(
            layer_names=["placeholder_attn_layers.1", "placeholder_attn_layers.2"],
            kv_cache_spec=_hmfp_spec("sliding_attention"),
        ),
        SimpleNamespace(
            layer_names=["placeholder_attn_layers.0"],
            kv_cache_spec=_hmfp_spec("full_attention"),
        ),
    ]

    bindings = bind_gemma4_cache_groups(contract, groups)

    assert [binding.cache_group_idx for binding in bindings] == [1, 0, 0]
    assert [binding.cache_index for binding in bindings] == [0, 1, 1]


def test_bind_mixed_gemma4_cache_groups_accepts_raw_fp_sliding_spec() -> None:
    contract = parse_gemma4_cache_contract(
        hf_config=_hf(),
        hmonnx_meta=_meta(
            attention_lowering="full_flash_attention",
            uses_sliding_flash_attention_v2=False,
        ),
    )
    full_spec = _hmfp_spec("full_attention")
    sliding_spec = SlidingWindowSpec(
        block_size=64,
        num_kv_heads=4,
        head_size=128,
        head_size_v=128,
        dtype=torch.float16,
        sliding_window=1024,
        indexes_kv_by_block_stride=True,
    )
    groups = [
        SimpleNamespace(
            layer_names=["placeholder_attn_layers.0"],
            kv_cache_spec=full_spec,
        ),
        SimpleNamespace(
            layer_names=[
                "placeholder_attn_layers.1",
                "placeholder_attn_layers.2",
            ],
            kv_cache_spec=sliding_spec,
        ),
    ]

    bindings = bind_gemma4_cache_groups(contract, groups)

    assert [binding.cache_group_idx for binding in bindings] == [0, 1, 1]


def test_bind_mixed_gemma4_cache_groups_recovers_omitted_sharing_alias() -> None:
    contract = parse_gemma4_cache_contract(
        hf_config=_hf(),
        hmonnx_meta=_meta(
            attention_lowering="full_flash_attention",
            uses_sliding_flash_attention_v2=False,
        ),
    )
    sliding_spec = SlidingWindowSpec(
        block_size=64,
        num_kv_heads=4,
        head_size=128,
        head_size_v=128,
        dtype=torch.float16,
        sliding_window=1024,
        indexes_kv_by_block_stride=True,
    )
    groups = [
        SimpleNamespace(
            layer_names=["placeholder_attn_layers.0"],
            kv_cache_spec=_hmfp_spec("full_attention"),
        ),
        SimpleNamespace(
            layer_names=["placeholder_attn_layers.1"],
            kv_cache_spec=sliding_spec,
        ),
    ]

    bindings = bind_gemma4_cache_groups(contract, groups)

    assert [binding.cache_group_idx for binding in bindings] == [0, 1, 1]


def test_bind_gemma4_cache_groups_resolves_uniform_children_and_shared_alias() -> None:
    contract = parse_gemma4_cache_contract(hf_config=_hf(), hmonnx_meta=_meta())
    full_spec = _hmfp_spec("full_attention")
    sliding_spec = _hmfp_spec("sliding_attention")
    groups = [
        SimpleNamespace(
            layer_names=["placeholder_attn_layers.0"],
            kv_cache_spec=UniformTypeKVCacheSpecs(
                block_size=64,
                kv_cache_specs={"placeholder_attn_layers.0": full_spec},
            ),
        ),
        SimpleNamespace(
            layer_names=[
                "placeholder_attn_layers.1",
                "placeholder_attn_layers.2",
            ],
            kv_cache_spec=UniformTypeKVCacheSpecs(
                block_size=64,
                kv_cache_specs={"placeholder_attn_layers.1": sliding_spec},
            ),
        ),
    ]

    bindings = bind_gemma4_cache_groups(contract, groups)

    assert [binding.cache_group_idx for binding in bindings] == [0, 1, 1]


def test_bind_gemma4_26b_uniform_children_by_exported_layer_identity() -> None:
    """The 26B 5:1 pattern must not be inferred from group/child order."""

    layer_types = [
        "full_attention" if layer_idx % 6 == 5 else "sliding_attention"
        for layer_idx in range(30)
    ]
    shapes = [
        [1, 2, 262144, 512] if kind == "full_attention" else [1, 8, 1344, 256]
        for kind in layer_types
    ]
    contract = parse_gemma4_cache_contract(
        hf_config=_hf(30, layer_types=layer_types),
        hmonnx_meta=_meta(
            layer_types=layer_types,
            layer_cache_types=layer_types,
            layer_cache_indices=list(range(30)),
            layer_cache_owner_indices=list(range(30)),
            layer_kv_shapes=shapes,
        ),
    )
    sliding_spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=8,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    full_spec = HMFPFullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=512,
        head_size_v=512,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    full_names = [
        f"placeholder_attn_layers.{idx}"
        for idx, kind in enumerate(layer_types)
        if kind == "full_attention"
    ]
    sliding_names = [
        f"placeholder_attn_layers.{idx}"
        for idx, kind in enumerate(layer_types)
        if kind == "sliding_attention"
    ]
    # Full comes first and both child dictionaries are reversed deliberately.
    groups = [
        SimpleNamespace(
            layer_names=full_names,
            kv_cache_spec=UniformTypeKVCacheSpecs(
                block_size=128,
                kv_cache_specs={name: full_spec for name in reversed(full_names)},
            ),
        ),
        SimpleNamespace(
            layer_names=sliding_names,
            kv_cache_spec=UniformTypeKVCacheSpecs(
                block_size=64,
                kv_cache_specs={name: sliding_spec for name in reversed(sliding_names)},
            ),
        ),
    ]

    bindings = bind_gemma4_cache_groups(contract, groups)

    assert [binding.cache_group_idx for binding in bindings] == [
        0 if kind == "full_attention" else 1 for kind in layer_types
    ]
    assert [binding.scheduler_block_size for binding in bindings] == [
        128 if kind == "full_attention" else 64 for kind in layer_types
    ]
    assert [binding.physical_pages_per_scheduler_block for binding in bindings] == [
        2 if kind == "full_attention" else 1 for kind in layer_types
    ]


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        (
            [
                SimpleNamespace(
                    layer_names=[
                        "placeholder_attn_layers.0",
                        "placeholder_attn_layers.0",
                    ],
                    kv_cache_spec=_hmfp_spec("full_attention"),
                )
            ],
            "more than once",
        ),
        (
            [
                SimpleNamespace(
                    layer_names=["placeholder_attn_layers.0"],
                    kv_cache_spec=_hmfp_spec("sliding_attention"),
                ),
                SimpleNamespace(
                    layer_names=[
                        "placeholder_attn_layers.1",
                        "placeholder_attn_layers.2",
                    ],
                    kv_cache_spec=_hmfp_spec("sliding_attention"),
                ),
            ],
            "type mismatch",
        ),
    ],
)
def test_bind_gemma4_cache_groups_rejects_ambiguous_mapping(groups, message) -> None:
    contract = parse_gemma4_cache_contract(hf_config=_hf(), hmonnx_meta=_meta())
    with pytest.raises(RuntimeError, match=message):
        bind_gemma4_cache_groups(contract, groups)
