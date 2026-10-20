"""Qwen4 checkpoint patches are explicit, scoped and retain shared checks."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest
from safetensors.torch import load_file, save_file

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_build import checkpoint
from xhmodel_merak.model_catalog.registry import get_recipe
from xhmodel_merak.xh_llm.models.qwen4_exp.checkpoint_patch import (
    MODEL_ID,
    _terminal_validator,
    qwen4_exp_checkpoint_scope,
)


@pytest.fixture
def native_checkpoint(tmp_path):
    from transformers import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    config = Qwen4ExpTextConfig(
        max_position_embeddings=16,
        vocab_size=16, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        num_experts=2, num_experts_per_tok=1, hc_count=2, hc_lowrank=4,
        layer_types=["linear_attention", "full_attention"], ple_layer_ids=[],
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=4, indexer_compress_ratio=2,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5},
    )
    path = tmp_path / "native"
    Qwen4ExpForCausalLM(config).save_pretrained(path)
    return path


def inspect_native(path, model=MODEL_ID):
    return checkpoint._inspect_quant_weight_files(path, SimpleNamespace(
        model_id=model, model_dir=str(path), build_adapter=get_recipe(model).build_adapter,
    ))


def _patched_functions():
    return checkpoint._read_quant_metadata, checkpoint._verify_named_checkpoint_tensors, checkpoint._checkpoint_method


def test_import_does_not_install_patch():
    from xhmodel_merak.xh_llm.models.qwen4_exp import checkpoint_patch

    before = _patched_functions()
    importlib.reload(checkpoint_patch)
    assert _patched_functions() == before


@pytest.mark.parametrize("fail", [False, True])
def test_scope_restores_original_functions_and_supports_nesting(fail):
    before = _patched_functions()
    try:
        with qwen4_exp_checkpoint_scope():
            active = _patched_functions()
            assert all(new is not old for new, old in zip(active, before, strict=True))
            with qwen4_exp_checkpoint_scope():
                assert _patched_functions() == active
            if fail:
                raise RuntimeError("build failed")
    except RuntimeError as error:
        assert fail and str(error) == "build failed"
    assert _patched_functions() == before


@pytest.mark.parametrize("metadata,expected", [
    ({"auto_round_enabled": True, "auto_round_version": "v1"}, "autoround"),
    ({"meta": {"auto_round": {"auto_round_enabled": True, "auto_round_version": "v1"}}}, "autoround"),
    ({"auto_round_enabled": True, "meta": {"auto_round": {"auto_round_enabled": True}}}, "autoround"),
    ({"auto_round_enabled": False}, "gptq"),
    ({"auto_round_enabled": None}, "gptq"),
    ({"auto_round_enabled": "true"}, "gptq"),
    ({"auto_round_enabled": "false"}, "gptq"),
    ({"auto_round_enabled": 1}, "gptq"),
    ({"meta": {"auto_round": {"auto_round_enabled": "true", "auto_round_version": "v1"}}}, "gptq"),
    ({"auto_round_version": "v1"}, "gptq"),
    ({}, "gptq"),
])
@pytest.mark.parametrize("layout", ["sidecar", "embedded", "config_top"])
def test_producer_evidence_is_scoped_to_qwen4_and_does_not_edit_metadata(tmp_path, metadata, expected, layout):
    payload = {"bits": 8, "quant_method": "gptq", **metadata}
    if layout == "embedded":
        payload = {"model_type": "qwen4_exp", "quantization_config": payload}
    path = tmp_path / ("quantization_config.json" if layout == "sidecar" else "config.json")
    path.write_text(json.dumps(payload))
    content = path.read_bytes()
    assert checkpoint.detect_checkpoint_method(str(tmp_path), model=MODEL_ID) == "gptq"
    with qwen4_exp_checkpoint_scope():
        assert checkpoint.detect_checkpoint_method(str(tmp_path), model=MODEL_ID) == expected
        assert checkpoint.detect_checkpoint_method(str(tmp_path), model="qwen3.5-9b") == "gptq"
        if layout != "sidecar":
            from xhmodel_merak.model_build.runtime_assets import behavior_config_mismatches

            patched = checkpoint._read_quant_metadata(tmp_path, MODEL_ID)[0][1]
            assert behavior_config_mismatches(payload, patched) == []
    assert path.read_bytes() == content
    assert checkpoint.detect_checkpoint_method(str(tmp_path), model=MODEL_ID) == "gptq"


def test_native_layout_passes_only_inside_scope_without_fake_final_norm(native_checkpoint):
    weights = load_file(native_checkpoint / "model.safetensors")
    assert "model.norm.weight" not in weights
    assert "model.hyper_connection_mixer.hc_norm.weight" in weights
    with pytest.raises(ModelError, match="verifiably complete"):
        inspect_native(native_checkpoint)
    with qwen4_exp_checkpoint_scope():
        assert inspect_native(native_checkpoint)["weights"]
        with pytest.raises(ModelError, match="verifiably complete") as error:
            inspect_native(native_checkpoint, "qwen3.5-9b")
        assert "terminal_mismatches" not in error.value.details
        assert error.value.details["has_final_norm"] is False
    with pytest.raises(ModelError, match="verifiably complete"):
        inspect_native(native_checkpoint)


@pytest.mark.parametrize("damage", [
    "terminal_missing", "terminal_shape", "layer_missing", "embedding_missing", "truncated",
])
def test_scope_keeps_terminal_and_shared_completeness_checks(native_checkpoint, damage):
    path = native_checkpoint / "model.safetensors"
    weights = load_file(path)
    if damage == "terminal_missing":
        del weights["model.hyper_connection_mixer.hc_norm.weight"]
    elif damage == "terminal_shape":
        name = "model.hyper_connection_mixer.input_mix_weight_up.weight"
        weights[name] = weights[name][:-1].contiguous()
    elif damage == "layer_missing":
        weights = {k: v for k, v in weights.items() if not k.startswith("model.layers.1.")}
    elif damage == "embedding_missing":
        del weights["model.embed_tokens.weight"]
    else:
        with path.open("r+b") as handle:
            handle.truncate(path.stat().st_size - 1)
    if damage != "truncated":
        save_file(weights, path)
    with qwen4_exp_checkpoint_scope(), pytest.raises(ModelError) as error:
        inspect_native(native_checkpoint)
    if damage.startswith("terminal"):
        assert error.value.details["terminal_mismatches"]
        assert error.value.details["has_final_norm"] is False


def test_private_validator_drift_is_rejected_without_installing_partial_patch(monkeypatch):
    def incompatible(path, container_summaries, index_assignments, plan):
        return None

    with pytest.raises(RuntimeError, match="incompatible"):
        _terminal_validator(incompatible)
    reader = checkpoint._read_quant_metadata
    monkeypatch.setattr(checkpoint, "_verify_named_checkpoint_tensors", incompatible)
    with pytest.raises(RuntimeError, match="incompatible"), qwen4_exp_checkpoint_scope():
        raise AssertionError("Incompatible implementation must not be installed")
    assert checkpoint._read_quant_metadata is reader
    assert checkpoint._verify_named_checkpoint_tensors is incompatible


def test_non_qwen_detection_stays_unmodified_in_another_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    (tmp_path / "quantization_config.json").write_text(json.dumps({
        "bits": 8, "quant_method": "gptq", "auto_round_enabled": True,
    }))
    with qwen4_exp_checkpoint_scope(), ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(checkpoint.detect_checkpoint_method, str(tmp_path), model="qwen3.5-9b")
        assert checkpoint.detect_checkpoint_method(str(tmp_path), model=MODEL_ID) == "autoround"
        assert future.result(timeout=5) == "gptq"


@pytest.mark.parametrize("embedded", [False, True])
def test_existing_producer_evidence_is_not_overwritten_or_hidden(tmp_path, embedded):
    payload = {
        "bits": 8, "quant_method": "gptq", "auto_round_enabled": True,
        "_qwen4_producer_evidence": {"provider": "qat"},
    }
    if embedded:
        payload = {"quantization_config": payload}
    path = tmp_path / ("config.json" if embedded else "quantization_config.json")
    path.write_text(json.dumps(payload))
    with qwen4_exp_checkpoint_scope(), pytest.raises(ModelError, match="one supported checkpoint producer") as error:
        checkpoint.detect_checkpoint_method(str(tmp_path), model=MODEL_ID)
    assert "qat" in error.value.details["providers"]
    assert json.loads(path.read_text()) == payload


@pytest.mark.parametrize("marker,expected", [
    ({"auto_round_enabled": True, "iters": 0, "disable_opt_rtn": True}, "rtn"),
    ({"auto_round_enabled": True, "iters": 2, "disable_opt_rtn": False}, "autoround"),
    ({"auto_round_enabled": True, "iters": 0, "disable_opt_rtn": False}, "autoround"),
    ({"auto_round_enabled": True, "iters": 0}, "autoround"),
    ({"auto_round_enabled": True, "iters": "0", "disable_opt_rtn": True}, "autoround"),
    ({"auto_round_enabled": True, "iters": False, "disable_opt_rtn": True}, "autoround"),
    ({"auto_round_enabled": True, "iters": 0.0, "disable_opt_rtn": True}, "autoround"),
    ({"auto_round_enabled": True, "iters": 0, "disable_opt_rtn": 1}, "autoround"),
    ({"auto_round_enabled": True, "iters": 0, "disable_opt_rtn": "true"}, "autoround"),
    ({"auto_round_enabled": 1, "iters": 0, "disable_opt_rtn": True}, "gptq"),
    ({"auto_round_enabled": True, "iters": 0, "meta": {"disable_opt_rtn": True}}, "autoround"),
])
def test_direct_rtn_requires_three_strict_colocated_producer_flags(tmp_path, marker, expected):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"quantization_config": {
        "bits": 8, "quant_method": "gptq", "meta": {"auto_round": marker},
    }}))
    original = path.read_bytes()
    before = _patched_functions()
    with qwen4_exp_checkpoint_scope():
        assert checkpoint.detect_checkpoint_method(str(tmp_path), model=MODEL_ID) == expected
        assert checkpoint.detect_checkpoint_method(str(tmp_path), model="qwen3.5-9b") == "gptq"
    assert _patched_functions() == before
    assert path.read_bytes() == original


def test_rtn_saved_engine_repetition_is_not_optimizer_evidence(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"quantization_config": {
        "quant_method": "gptq", "bits": 8, "auto_round_enabled": True, "iters": 0,
        "meta": {"auto_round": {"auto_round_enabled": True, "iters": 0, "disable_opt_rtn": True}},
    }}))
    with qwen4_exp_checkpoint_scope():
        assert checkpoint.detect_checkpoint_method(str(tmp_path), model=MODEL_ID) == "rtn"


@pytest.mark.parametrize("conflict", ["qat", "auto-round", "autoround", "nested_optimizer", "optimized_rtn"])
def test_rtn_preserves_and_rejects_conflicting_optimizer_evidence(tmp_path, conflict):
    metadata = {"quant_method": "gptq", "bits": 8, "meta": {"auto_round": {
        "auto_round_enabled": True, "iters": 0, "disable_opt_rtn": True,
    }}}
    if conflict in {"nested_optimizer", "optimized_rtn"}:
        metadata["auto_round_enabled"] = True
        metadata["iters"] = 2 if conflict == "nested_optimizer" else 0
        metadata["disable_opt_rtn"] = False
    else:
        metadata["provider"] = conflict
    (tmp_path / "quantization_config.json").write_text(json.dumps(metadata))
    before = _patched_functions()
    with pytest.raises(ModelError, match="one supported checkpoint producer"), qwen4_exp_checkpoint_scope():
        checkpoint.detect_checkpoint_method(str(tmp_path), model=MODEL_ID)
    assert _patched_functions() == before


@pytest.mark.parametrize("iters,disable_opt,method", [
    (0, True, "rtn"), (2, False, "autoround"), (0, False, "autoround"),
])
def test_native_saved_checkpoint_full_inspection_preserves_rtn_and_old_autoround(
    native_checkpoint, monkeypatch, iters, disable_opt, method,
):
    from dataclasses import replace

    from xhmodel_merak.model_catalog.contracts import ResolvedPrecision

    # Use the actual native two-layer checkpoint with an appropriately small
    # recipe source constraint; all shared inspection functions execute for real.
    recipe = replace(get_recipe(MODEL_ID), source_constraints=())
    monkeypatch.setattr(
        checkpoint, "get_recipe", lambda model_id: recipe if model_id == MODEL_ID else get_recipe(model_id),
    )
    path = native_checkpoint / "config.json"
    payload = json.loads(path.read_text())
    payload["quantization_config"] = {
        "quant_method": "gptq", "bits": 8, "group_size": 64,
        "auto_round_enabled": True, "iters": iters,
        "meta": {"auto_round": {
            "auto_round_enabled": True, "iters": iters, "disable_opt_rtn": disable_opt,
        }},
    }
    path.write_text(json.dumps(payload))
    original_config = path.read_bytes()
    original_weights = (native_checkpoint / "model.safetensors").read_bytes()
    plan = SimpleNamespace(
        model_id=MODEL_ID, model_dir=str(native_checkpoint), quant_method=method,
        precision=ResolvedPrecision(weight="w8", activation="a8"), build_adapter=recipe.build_adapter,
    )
    with qwen4_exp_checkpoint_scope():
        result = checkpoint.inspect_quant_artifact(str(native_checkpoint), plan)
    assert result["method"] == method
    assert result["verification"] == "checkpoint_verified"
    assert ("rtn" in result["providers"]) is (method == "rtn")
    assert path.read_bytes() == original_config
    assert (native_checkpoint / "model.safetensors").read_bytes() == original_weights


@pytest.mark.parametrize("methods", [("rtn", "rtn"), ("rtn", "unknown")])
def test_rtn_recipe_extension_rejects_duplicate_and_unknown_methods(methods):
    from dataclasses import replace

    with pytest.raises(ValueError, match="unique supported producer names"):
        replace(get_recipe(MODEL_ID), quant_checkpoint_methods=methods)


def test_rtn_recipe_extension_keeps_public_validation_and_actual_method_list():
    from dataclasses import replace

    recipe = replace(get_recipe(MODEL_ID), quant_checkpoint_methods=("rtn", "autoround"))
    assert recipe.quant_checkpoint_methods == ("rtn", "autoround")
    with pytest.raises(ValueError, match="export_name"):
        replace(recipe, export_name="INVALID-EXPORT-NAME")
