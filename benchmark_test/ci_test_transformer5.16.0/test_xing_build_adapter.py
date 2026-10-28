# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Xing packaging and native tokenizer validation without model export."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_catalog.loading import load_adapter
from xhmodel_merak.model_integrations.autoregressive import build as text_build
from xhmodel_merak.model_integrations.xing4_0 import build as xing_build
from xhmodel_merak.model_integrations.xing4_0.checkpoint import REGION_WEIGHT_BITS, verify_quant_regions


def _source_config(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "architectures": ["Xing4_0ForCausalLM"], "model_type": "xing4_0", "hidden_size": 3584,
        "num_hidden_layers": 40, "num_attention_heads": 32, "n_routed_experts": 64,
        "num_experts_per_tok": 4, "q_lora_rank": 768, "kv_lora_rank": 512,
        "qk_rope_head_dim": 64, "hc_mult": 4, "num_nextn_predict_layers": 1,
    }))
    return source


def test_xing_fixture_binds_local_jsonl_without_changing_recipe_defaults(tmp_path, monkeypatch):
    import yaml
    from _unified_fixtures.xing import _configure_calibration
    from xhmodel_merak.model_build import api
    from xhmodel_merak.model_build.workflow import materialize_workflow_config

    source = _source_config(tmp_path)
    kwargs = {
        "model_id": "xing4.0-29b-a4b", "model_dir": str(source), "output_dir": str(tmp_path / "release"),
    }
    production_plan = api.resolve_build_plan(**kwargs)
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(json.dumps({"text": "hello world"}) + "\n")
    _configure_calibration(calibration, monkeypatch)
    plan = api.resolve_build_plan(**kwargs)
    config_path = tmp_path / "effective.yaml"
    materialize_workflow_config(plan, config_path)
    quant = yaml.safe_load(config_path.read_text())["quant"]

    assert quant["calibration"]["jsonl"] == str(calibration)
    assert quant["calibration"]["nsamples"] == 2
    assert quant["calibration"]["seqlen"] == 64
    assert quant["runtime"]["batch_size"] == 1
    assert plan.quant_method == production_plan.quant_method == "gptq"
    assert production_plan.config_replacements["quant.calibration.jsonl"] != str(calibration)


@pytest.mark.parametrize("checkpoint_alias", [False, True])
def test_xing_packages_source_config_representation_and_checkpoint_assets(tmp_path, checkpoint_alias):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = {
        "model_type": "xing4_0", "qk_rope_head_dim": 64, "rope_theta": 10000,
        "rope_scaling": {"type": "yarn", "factor": 64},
        "quantization_config": {"bits": 4, "quant_method": "gptq"},
    }
    if checkpoint_alias:
        config["head_dim"] = 64
    checkpoint_text = json.dumps(config)
    (checkpoint / "config.json").write_text(checkpoint_text)
    (checkpoint / "tokenizer.model").write_bytes(b"checkpoint-tokenizer")
    (checkpoint / "tokenizer_config.json").write_text("{}")

    stage = tmp_path / "artifact"
    hf_config = stage / "hf_config"
    hf_config.mkdir(parents=True)
    exported = {**config, "quantization_config": {"desc_act": False}}
    (hf_config / "config.json").write_text(json.dumps(exported))
    (hf_config / "tokenizer.model").write_bytes(b"source-tokenizer")

    adapter = load_adapter("build", "xing4_0.build")
    adapter.package_runtime(stage, SimpleNamespace(model_id="xing4.0-29b-a4b", quant_checkpoint=str(checkpoint)))

    expected = {**config, "quantization_config": {**config["quantization_config"], "desc_act": False}}
    assert json.loads((hf_config / "config.json").read_text()) == expected
    assert (hf_config / "tokenizer.model").read_bytes() == b"checkpoint-tokenizer"
    assert (checkpoint / "config.json").read_text() == checkpoint_text


@pytest.mark.parametrize("change", ["head_dim", "qk_rope_head_dim", "rope_theta"])
def test_xing_packaging_rejects_changed_behavior_before_copying(tmp_path, change):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = {"qk_rope_head_dim": 64, "rope_theta": 10000}
    (checkpoint / "config.json").write_text(json.dumps(config))
    stage = tmp_path / "artifact"
    hf_config = stage / "hf_config"
    hf_config.mkdir(parents=True)
    exported = {**config, "head_dim": 64, change: 128}
    exported_text = json.dumps(exported)
    (hf_config / "config.json").write_text(exported_text)
    plan = SimpleNamespace(model_id="xing4.0-29b-a4b", quant_checkpoint=str(checkpoint))

    with pytest.raises(ModelError) as error:
        xing_build.package_runtime(stage, plan)

    assert error.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert error.value.details["config_mismatches"]
    assert (hf_config / "config.json").read_text() == exported_text


@pytest.fixture
def xing_artifact(tmp_path):
    from xhmodel_merak.model_build.resolver import resolve_build_plan

    plan = resolve_build_plan(
        model_id="xing4.0-29b-a4b", model_dir=str(_source_config(tmp_path)),
        output_dir=str(tmp_path / "release"),
    )
    stage = tmp_path / "artifact"
    hf_config = stage / "hf_config"
    hf_config.mkdir(parents=True)
    config = json.loads((tmp_path / "source/config.json").read_text())
    config["quantization_config"] = {
        "bits": 4, "quant_method": "gptq",
        "dynamic": {name: {"bits": bits} for name, bits in REGION_WEIGHT_BITS.items()},
    }
    (hf_config / "config.json").write_text(json.dumps(config))
    # A real local custom tokenizer exercises Transformers' dynamic loader.
    (hf_config / "tokenization_local.py").write_text(
        "from transformers import BertTokenizer\nclass LocalXingTokenizer(BertTokenizer):\n    pass\n"
    )
    (hf_config / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nruntime\nvalidation\n")
    (hf_config / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "LocalXingTokenizer",
        "auto_map": {"AutoTokenizer": ["tokenization_local.LocalXingTokenizer", None]},
        "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
    }))
    metadata = {
        "hf_config": "hf_config", "quant_embedding": "embedding.bin",
        "prefill_hmonnx": "prefill/model.onnx", "decode_hmonnx": "decode/model.onnx",
        "model_config": {
            "context_max_length": plan.context_length, "prefill_chunk_length": plan.prefill_chunk_length,
            "max_pe_length": plan.max_pe_length,
        },
    }
    for relative in ("embedding.bin", "prefill/model.onnx", "decode/model.onnx", "merak_config.json"):
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (stage / "golden_meta_info.json").write_text(json.dumps(metadata))
    return stage, plan


def test_xing_native_tokenizer_validates_artifact_without_changing_shared_loader(xing_artifact):
    stage, plan = xing_artifact
    adapter = load_adapter("build", "xing4_0.build")
    assert adapter.normalize_export is text_build.normalize_export
    assert adapter.sanitize_artifact is text_build.sanitize_artifact
    result = adapter.validate_artifact(stage, plan)
    assert result["loadable"] is True
    assert result["requires_source"] is False
    assert result["components"] == ["prefill", "decode"]
    assert result["export_entries"]["tokenizer"] == {"path": "hf_config", "type": "tokenizer"}
    assert result["children"] == []
    assert result["golden_artifacts"] == {}
    regions = result["checkpoint_precision_regions"]
    assert {name: region["weight_bits"] for name, region in regions.items()} == REGION_WEIGHT_BITS
    # Other families still require a built-in tokenizer implementation.
    with pytest.raises(ModelError, match="local tokenizer snapshot"):
        text_build._validate_tokenizer_snapshot(stage / "hf_config", plan)


@pytest.mark.parametrize("failure", ["missing_code", "empty_tokens", "missing_graph", "runtime_length", "missing_meta"])
def test_xing_validation_rejects_incomplete_runtime(xing_artifact, failure):
    stage, plan = xing_artifact
    if failure == "missing_code":
        (stage / "hf_config/tokenization_local.py").unlink()
    elif failure == "empty_tokens":
        with (stage / "hf_config/tokenization_local.py").open("a") as stream:
            stream.write("    def __call__(self, *args, **kwargs):\n        return {'input_ids': []}\n")
    elif failure == "missing_graph":
        (stage / "prefill/model.onnx").unlink()
    elif failure == "runtime_length":
        plan = replace(plan, context_length=plan.context_length + 1)
    else:
        (stage / "merak_config.json").unlink()

    with pytest.raises(ModelError) as error:
        xing_build.validate_artifact(stage, plan)

    assert error.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED


@pytest.mark.parametrize("changed_region", [None, *REGION_WEIGHT_BITS, "missing_shared"])
def test_xing_owns_regex_region_matching_and_rejects_invalid_precision(changed_region):
    patterns = {
        "attention": r"model\.layers\.\d+\.self_attn\.(q_a_proj|q_absorbed_proj|o_proj)$",
        "dense_mlp": r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$",
        "shared_expert": r"model\.layers\.\d+\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)$",
        "routed_expert": r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
    }
    dynamic = {patterns[name]: {"bits": bits} for name, bits in REGION_WEIGHT_BITS.items()}
    if changed_region == "missing_shared":
        dynamic.pop(patterns["shared_expert"])
    elif changed_region is not None:
        dynamic[patterns[changed_region]]["bits"] = 8 if changed_region == "routed_expert" else 4
    if changed_region is not None:
        with pytest.raises(ModelError) as error:
            verify_quant_regions([{"dynamic": dynamic}], model_id="xing4.0-29b-a4b")
        assert error.value.code == ErrorCode.QUANT_ARTIFACT_MISMATCH
        assert [entry["component"] for entry in error.value.details["invalid_regions"]] == [
            "shared_expert" if changed_region == "missing_shared" else changed_region,
        ]
    else:
        result = verify_quant_regions([{"dynamic": dynamic}], model_id="xing4.0-29b-a4b")
        assert {name: entry["weight_bits"] for name, entry in result.items()} == REGION_WEIGHT_BITS


def test_xing_checks_regions_before_export_and_publication(xing_artifact):
    from xhmodel_merak.model_build.resolver import resolve_build_plan
    from xhmodel_merak.xh_llm.models.xing4_0.workflow import Xing4_0Workflow

    stage, plan = xing_artifact
    path = stage / "hf_config/config.json"
    config = json.loads(path.read_text())
    config["quantization_config"]["dynamic"]["routed_expert"]["bits"] = 8
    path.write_text(json.dumps(config))
    owner = Xing4_0Workflow(plan.model_dir, plan.config_path)
    with pytest.raises(ModelError, match="W4 routed-expert"):
        resolve_build_plan(
            model_id=plan.model_id, model_dir=plan.model_dir, quant_checkpoint=str(path.parent),
            output_dir=str(stage.parent / "rejected-release"),
        )
    with pytest.raises(ModelError, match="W4 routed-expert"):
        owner.export(SimpleNamespace(quanted_model_dir=str(path.parent)), "unused", "cpu")
    with pytest.raises(ModelError, match="W4 routed-expert"):
        xing_build.validate_artifact(stage, plan)


def test_standalone_w8_workflow_checks_explicit_w8_checkpoint_regions(tmp_path):
    from xhmodel_merak.model_integrations.xing4_0.checkpoint import inspect_quant_regions

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"quantization_config": {
        "bits": 8, "quant_method": "gptq", "dynamic": {name: {"bits": 8} for name in REGION_WEIGHT_BITS},
    }}))
    regions = inspect_quant_regions(checkpoint, routed_bits=None)
    assert {name: entry["weight_bits"] for name, entry in regions.items()} == dict.fromkeys(REGION_WEIGHT_BITS, 8)
    with pytest.raises(ModelError, match="W4 routed-expert"):
        inspect_quant_regions(checkpoint)
