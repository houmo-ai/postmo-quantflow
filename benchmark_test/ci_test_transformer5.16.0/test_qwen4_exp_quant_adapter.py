from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from xhmodel_merak.xh_llm.models.qwen4_exp.quant_adapter import (
    build_qwen4_exp_autoround_kwargs,
    quantize_with_autoround_api,
    resolve_quant_method_options,
    validate_rotation_policy,
)


def _config() -> dict:
    layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
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


def test_build_autoround_kwargs_carries_precision_and_architecture_contract(tmp_path: Path):
    model_dir = tmp_path / "Qwen3.8-Flash-Next"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_config()), encoding="utf-8")

    kwargs = build_qwen4_exp_autoround_kwargs(
        hf_model_dir=str(model_dir),
        output_dir=str(tmp_path / "quant"),
        device="cuda:0",
        quant_cfg={"method": "autoround", "group_size": 64, "rotation": False},
        export_model_cfg={},
        workflow_seed=42,
    )

    assert kwargs["bits"] == 8
    assert kwargs["expert_bits"] == 4
    assert kwargs["attention_bits"] == 8
    assert kwargs["linear_attention_bits"] == 8
    assert kwargs["shared_expert_bits"] == 8
    assert kwargs["language_quarot"] is False
    assert kwargs["artifact_format"] == "gptqmodel_hf"
    assert kwargs["auto_forward_data_parallel"] is False


def test_build_autoround_kwargs_accepts_vision_only_rotation(tmp_path: Path):
    kwargs = build_qwen4_exp_autoround_kwargs(
        hf_model_dir="unused",
        output_dir=str(tmp_path / "quant"),
        device="cuda:0",
        quant_cfg={"model_config": _config(), "rotation": "vision_only"},
        export_model_cfg={},
        workflow_seed=42,
    )

    assert kwargs["language_quarot"] is False
    assert kwargs["vision_quarot"] == "mean_preserving"


def test_language_or_global_rotation_is_rejected():
    with pytest.raises(ValueError, match="language/global QuaRot"):
        validate_rotation_policy({"rotation": "hadamard"})

    with pytest.raises(ValueError, match="language/global QuaRot"):
        validate_rotation_policy({"rotation": True})


def test_package_import_is_lazy_without_gptqmodel_or_transformers():
    import xhmodel_merak.xh_llm.models.qwen4_exp as qwen4_exp

    assert qwen4_exp.QWEN4_EXP_PROFILE.expert_bits == 4


def test_adapter_delegates_to_external_qwen4_exp_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = {}
    module = types.ModuleType("gptqmodel.recipes.qwen4_exp")

    def fake_quantize_qwen4_exp_autoround(**kwargs):
        calls.update(kwargs)
        return {"output_dir": kwargs["output_dir"]}

    module.quantize_qwen4_exp = fake_quantize_qwen4_exp_autoround
    monkeypatch.setitem(sys.modules, "gptqmodel.recipes.qwen4_exp", module)

    result = quantize_with_autoround_api(
        model_dir="weights/Qwen3.8-Flash-Next",
        output_dir=str(tmp_path / "quant"),
        device="cuda:0",
        quant_cfg={"model_config": _config(), "rotation": False},
        export_model_cfg={},
        workflow_seed=42,
    )

    assert calls["model_dir"] == "weights/Qwen3.8-Flash-Next"
    assert calls["expert_bits"] == 4
    assert calls["method"] == "rtn"
    assert calls["iters"] == 0
    assert calls["disable_opt_rtn"] is True
    assert "vision_quarot" not in calls
    assert calls["language_quarot"] is False
    assert result.quanted_model_dir == str((tmp_path / "quant").resolve())



def test_default_direct_rtn_options():
    kwargs = build_qwen4_exp_autoround_kwargs(
        hf_model_dir="unused", output_dir="quant", device="cpu",
        quant_cfg={"model_config": _config()}, export_model_cfg={}, workflow_seed=42,
    )
    assert (kwargs["method"], kwargs["iters"], kwargs["disable_opt_rtn"]) == ("rtn", 0, True)
    assert kwargs["language_quarot"] is False
    assert "vision_quarot" not in kwargs


@pytest.mark.parametrize("method", ["autoround", "auto_round"])
def test_autoround_opt_in_has_positive_iterations(method):
    assert resolve_quant_method_options({"method": method}) == ("autoround", 200, None)
    assert resolve_quant_method_options({"method": method, "iters": 3}) == ("autoround", 3, None)


@pytest.mark.parametrize("algorithm", ["autoround", "auto_round"])
def test_legacy_autoround_algorithm_preserves_explicit_opt_in(algorithm):
    assert resolve_quant_method_options({"algorithm": algorithm}) == ("autoround", 200, None)


@pytest.mark.parametrize("config", [
    {"method": "rtn", "iters": 200},
    {"method": "rtn", "iters": -1},
    {"method": "rtn", "disable_opt_rtn": False},
    {"method": "rtn", "disable_opt_rtn": None},
    {"method": "rtn", "iters": True},
    {"method": "rtn", "disable_opt_rtn": "true"},
    {"method": "autoround", "iters": 0},
    {"method": "autoround", "iters": -1},
    {"method": "autoround", "disable_opt_rtn": True},
    {"method": "gptq"},
])
def test_invalid_quant_method_options_fail_closed(config):
    with pytest.raises(ValueError):
        resolve_quant_method_options(config)


@pytest.mark.parametrize("flag", [
    "--iters", "--iters=200", "--disable-opt-rtn", "--no-disable-opt-rtn",
    "--language-quarot", "--vision-quarot=mean_preserving",
])
def test_extra_args_cannot_override_quantization_identity(flag):
    with pytest.raises(ValueError, match="cannot override"):
        resolve_quant_method_options({"runtime": {"extra_args": [flag]}})
