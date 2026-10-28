"""Xing workflow configuration, producer API and runtime assets."""

import inspect
import json
import os
import runpy
import sys
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.model_catalog import get_recipe, resolve_workflow_config
from xhmodel_merak.xh_llm.models.xing4_0.workflow import Xing4_0Workflow


def _workflow(tmp_path):
    config = resolve_workflow_config(get_recipe("xing4.0-29b-a4b").default_variant.workflow_config)
    return Xing4_0Workflow(str(tmp_path / "source"), str(config), seed=123)


@pytest.mark.parametrize("method", ["gptq", "autoround"])
def test_workflow_passes_quant_settings_and_seed_to_the_producer(tmp_path, monkeypatch, method):
    import gptqmodel.recipes as recipes
    from xhmodel_merak.xh_llm.models.xing4_0.quant_adapter import _load_quantize_recipe

    workflow = _workflow(tmp_path)
    actual = _load_quantize_recipe()
    calls = []

    @wraps(actual)
    def producer(**kwargs):
        # Check the installed public signature without running calibration.
        inspect.signature(actual).bind(**kwargs)
        calls.append(kwargs)
        return SimpleNamespace(output_dir=kwargs["output_dir"])

    monkeypatch.setattr(recipes, "quantize_xing4_0", producer)
    result = workflow.quant(str(tmp_path / "quant"), "cpu", {"quant.method": method})
    kwargs, = calls
    assert kwargs["model_dir"] == workflow.model_dir
    assert result.raw_model_dir == workflow.model_dir
    assert result.quanted_model_dir == kwargs["output_dir"]
    assert kwargs["method"] == method
    assert kwargs["mla_weight_layout"] == "xh2_absorbed_v1"
    assert kwargs["offload_to_disk"] is False
    assert kwargs["seed"] == 123
    assert (kwargs["bits"], kwargs["group_size"], kwargs["sym"]) == (4, 64, True)
    assert [kwargs[key] for key in ("self_attn_bits", "dense_mlp_bits", "shared_expert_bits", "expert_bits")] == [8, 8, 8, 4]
    assert (kwargs["nsamples"], kwargs["seqlen"], kwargs["moe_routing_batch_size"]) == (128, 1024, 48)
    assert Path(kwargs["calibration_jsonl"]).is_file()
    assert kwargs["calibration_dataset"] is None


def test_workflow_skips_quantization_for_existing_weights(tmp_path):
    workflow = _workflow(tmp_path)
    result = workflow.quant(str(tmp_path / "quant"), "cpu", {"quant": None})
    assert result.skipped and result.raw_model_dir == workflow.model_dir
    export = workflow._build_export_config(result, workflow.workflow_config)
    assert export["model"]["hf_model"] == workflow.model_dir
    assert export["model"]["model_type"] == "Xing4_0ForCausalLM"


def test_export_writes_the_merak_runtime_loader_config(tmp_path, monkeypatch):
    from xhmodel_merak.xh_llm.workflows.base import BaseLLMWorkflow
    from xhmodel_merak.xh_llm.workflows.result import ExportResult, QuantResult

    directory = tmp_path / "export" / "hmquant_xing"
    directory.mkdir(parents=True)
    (directory / "golden_meta_info.json").write_text(json.dumps({"model_config": {}}))
    exported = ExportResult(work_dir=str(directory.parent), config_file="workflow.yaml", meta={})
    # Exercise the finalizer with real metadata; graph export is covered by the
    # unified build/relocate/inference case.
    monkeypatch.setattr(BaseLLMWorkflow, "export", lambda *args, **kwargs: exported)
    workflow = _workflow(tmp_path)
    assert workflow.export(QuantResult(raw_model_dir=workflow.model_dir, skipped=True),
                           str(directory.parent), "cpu") is exported
    config = json.loads((directory / "merak_config.json").read_text())
    assert config["model_type"] == "merak_llm"
    assert config["xh_model"] == {"model_type": "hmonnx", "meta_info": "golden_meta_info.json"}


def test_generation_facade_uses_transformers_no_init_and_meta_weights(tmp_path, monkeypatch):
    from transformers import LlamaConfig, PreTrainedModel
    from xhmodel_merak.xh_llm.models.xing4_0 import xing4_0_model

    config = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, vocab_size=32)
    monkeypatch.setattr(xing4_0_model.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)

    def fail_init(*args, **kwargs):
        raise AssertionError("A generation facade must not initialize pretrained weights")

    monkeypatch.setattr(PreTrainedModel, "_init_weights", fail_init)
    original_init = PreTrainedModel.init_weights
    model = xing4_0_model.XHXing4_0Model.get_empty_hf_model(tmp_path)
    assert model.can_generate()
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    assert PreTrainedModel.init_weights is original_init


@pytest.mark.parametrize("overrides,match", [
    ({"quant.desc_act": True}, "desc_act=false"),
    ({"quant.group_size": 32}, "group_size"),
    ({"quant.method": "unsupported"}, "method"),
    ({"quant.algorithm": "unsupported"}, "supports"),
    ({"quant.sym": "invalid"}, "boolean"),
    ({"quant.offload_to_disk": "invalid"}, "boolean"),
    ({"quant.hessian_mse": "invalid"}, "boolean"),
    ({"quant.runtime.trust_remote_code": "invalid"}, "boolean"),
    ({"quant.calibration.jsonl": "path-to-data.jsonl"}, "placeholder"),
])
def test_workflow_rejects_unsupported_quant_settings_before_training(tmp_path, monkeypatch, overrides, match):
    from xhmodel_merak.xh_llm.models.xing4_0 import quant_adapter

    def unexpected_import():
        raise AssertionError("Invalid settings must fail before loading the producer")

    monkeypatch.setattr(quant_adapter, "_load_quantize_recipe", unexpected_import)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        _workflow(tmp_path).quant(str(tmp_path / "quant"), "cpu", overrides)


@pytest.mark.parametrize("missing", [True, False])
def test_workflow_reports_the_incompatible_producer_checkout(tmp_path, monkeypatch, missing):
    import gptqmodel.recipes as recipes

    if missing:
        monkeypatch.delattr(recipes, "quantize_xing4_0", raising=False)
    else:
        def legacy_recipe(*, model_dir, output_dir):
            raise AssertionError("An unabsorbed producer must not run")

        monkeypatch.setattr(recipes, "quantize_xing4_0", legacy_recipe, raising=False)
    with pytest.raises(ImportError, match="mla_weight_layout support") as exc:
        _workflow(tmp_path).quant(str(tmp_path / "quant"), "cpu")
    assert recipes.__file__ in str(exc.value)
    assert not (tmp_path / "quant").exists()


@pytest.mark.parametrize("previous", [None, "1"])
def test_golden_restores_environment_when_model_loading_fails(tmp_path, monkeypatch, previous):
    import xhmodel_merak.xh_llm as llm

    key = "XHQUANT_MOE_LOW_M_TRITON"
    if previous is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, previous)
    root = tmp_path / "export" / "hmquant_xing"
    root.mkdir(parents=True)
    (root / "golden_meta_info.json").write_text("{}")

    def fail_loading(*args, **kwargs):
        assert os.environ[key] == "0"
        raise RuntimeError("loader failed")

    monkeypatch.setattr(llm.AutoLLMHONNXModel, "from_pretrained", fail_loading)
    with pytest.raises(RuntimeError, match="loader failed"):
        _workflow(tmp_path).dump_golden(SimpleNamespace(work_dir=str(root.parent)), "cpu", "hello")
    assert os.environ.get(key) == previous


def test_export_copies_only_declared_runtime_assets_without_rewriting_them(tmp_path):
    from xhmodel_merak.xh_llm.models.xing4_0 import XHXing4_0Model

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"auto_map": {"AutoModelForCausalLM": "modeling_xing4_0.Xing4_0ForCausalLM"}}))
    (source / "modeling_xing4_0.py").write_text("from .configuration_xing4_0 import Xing4_0Config\n# module.fn\n")
    (source / "configuration_xing4_0.py").write_text("class Xing4_0Config: pass\n")
    (source / "tokenizer.model").write_bytes(b"sentencepiece")
    (source / "debug.py").write_text("raise RuntimeError('not a runtime asset')")
    destination = tmp_path / "export"
    model = object.__new__(XHXing4_0Model)
    model.config = SimpleNamespace(hf_model=str(source))
    meta = SimpleNamespace(hf_config="hf_config")
    model._extra_export_metadata(str(destination), meta)
    assert {path.name for path in (destination / "hf_config").iterdir()} == {
        "config.json", "modeling_xing4_0.py", "configuration_xing4_0.py", "tokenizer.model",
    }
    for path in (destination / "hf_config").iterdir():
        assert path.read_bytes() == (source / path.name).read_bytes()
    (source / "tokenizer.model").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer.model"):
        model._extra_export_metadata(str(destination), meta)


@pytest.mark.parametrize("runtime", [False, True])
def test_xing_processor_uses_the_model_tokenizer(tmp_path, monkeypatch, runtime):
    from xhmodel_merak.xh_llm import base_llm_model
    from xhmodel_merak.xh_llm.hmonnx import base_llm_hmonnx_model
    from xhmodel_merak.xh_llm.models.xing4_0 import XHXing4_0HMONNXModel, XHXing4_0Model

    module = base_llm_hmonnx_model if runtime else base_llm_model
    owner = object.__new__(XHXing4_0HMONNXModel if runtime else XHXing4_0Model)
    owner.hf_model_dir = str(tmp_path)
    calls = []
    tokenizer = object()
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda path, **kwargs: calls.append(kwargs) or tokenizer)
    assert owner.get_tf_processor().tokenizer is tokenizer
    owner.get_tokenizer(trust_remote_code=False)
    assert calls == [{"trust_remote_code": True}, {"trust_remote_code": False}]


@pytest.mark.parametrize("example", ["fp_demo", "hmonnx_demo", "xing4_0_workflow", "xing4_0_xh_generate_workflow"])
def test_example_help_exits_without_loading_weights(monkeypatch, example):
    path = Path(__file__).resolve().parents[2] / "examples_merak" / "llm" / "xing4_0" / f"{example}.py"
    monkeypatch.setattr(sys, "argv", [str(path), "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(path), run_name="__main__")
    assert exc.value.code == 0
