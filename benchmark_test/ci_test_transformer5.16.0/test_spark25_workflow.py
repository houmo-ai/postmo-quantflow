"""Spark workflow routing, real AutoRound CLI contract and checkpoint handoff."""

import copy
import json
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs_merak/workflows/xh2a/llm_models/xh_token_spark/4b/xh_token_spark_4b_xh2a_w4a8.yaml"


@pytest.fixture
def adapter():
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5 import quant_adapter

    return quant_adapter


@pytest.fixture
def quant_cfg(tmp_path):
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text())["quant"]
    data = tmp_path / "calibration.jsonl"
    data.write_text('{"text":"a small calibration sample"}\n')
    cfg["calibration"]["jsonl"] = str(data)
    return cfg


def test_real_cli_parser_and_mixed_profile(adapter, quant_cfg, tmp_path, monkeypatch):
    import sys

    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.autoround_cli import parse_args

    command = adapter.build_autoround_command(
        model_dir="model", output_dir=tmp_path, device="cuda:0", quant_cfg=quant_cfg
    )
    monkeypatch.setattr(sys, "argv", ["spark-worker", *command[3:]])
    args = parse_args()
    assert args.device_map == "0"
    assert args.dataset == quant_cfg["calibration"]["jsonl"] + ":concat"
    assert (args.bits, args.group_size, args.nsamples, args.seqlen) == (4, 64, 128, 2048)
    assert args.iters == 200
    # The worker passes this unquoted string to the bundled backend parser.
    layers = dict(quant_cfg["layer_config"])
    layers["lm_head"] = {"bits": 16}
    layers["model.layers.[0-9]+.self_attn.g_proj"] = {"bits": 16}
    assert args.layer_config == adapter.layer_config_arg(
        {
            **quant_cfg["layer_config"],
            "model.embedding": {"bits": 16, "data_type": "fp"},
            "lm_head": {"bits": 16, "data_type": "fp"},
            "model.layers.[0-9]+.self_attn.g_proj": {"bits": 16, "data_type": "fp"},
        }
    )
    for index in range(36):
        for suffix in ("self_attn.q_k_v_proj", "self_attn.out_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            matches = [v["bits"] for k, v in layers.items() if re.fullmatch(k, f"model.layers.{index}.{suffix}")]
            bits = matches[0] if matches else args.bits
            assert bits == (16 if suffix == "mlp.gate_proj" else 4)
    assert layers["lm_head"]["bits"] == 16
    assert layers["model.layers.[0-9]+.self_attn.g_proj"]["bits"] == 16


def test_model_staging_does_not_mutate_source(adapter, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    text = json.dumps({"model_type": "spark2_5", "auto_map": {"AutoConfig": "remote.Config"}})
    (model / "config.json").write_text(text)
    (model / "model.safetensors").write_bytes(b"test handoff fixture")
    with adapter.staged_model(model) as staged:
        assert "auto_map" not in json.loads((staged / "config.json").read_text())
        assert (staged / "model.safetensors").resolve() == model / "model.safetensors"
    assert not staged.exists()
    assert (model / "config.json").read_text() == text


def test_routing_and_explicit_quant_skip(tmp_path):
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.workflow import Spark2_5Workflow
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(model_dir=str(tmp_path), config_path=str(CONFIG))
    assert isinstance(workflow, Spark2_5Workflow)
    result = workflow.quant(str(tmp_path / "unused"), "cuda", {"quant": None})
    assert result.skipped and result.quanted_model_dir is None
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("model_id", ["spark-x2.5-1.7b", "spark-x2.5-4b"])
@pytest.mark.parametrize("valid_metadata", [True, False])
def test_export_finalizes_merak_runtime_config(tmp_path, monkeypatch, model_id, valid_metadata):
    from xhmodel_merak.model_catalog import get_recipe
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow
    from xhmodel_merak.xh_llm.workflows.base import BaseLLMWorkflow
    from xhmodel_merak.xh_llm.workflows.result import ExportResult, QuantResult

    recipe = get_recipe(model_id)
    workflow = AutoLLMWorkflow.from_config(
        model_dir=str(tmp_path), config_path=str(ROOT / recipe.default_variant.workflow_config)
    )
    output = tmp_path / "export"
    release = output / "hmquant_spark"
    export_result = ExportResult(work_dir=str(output), config_file=str(output / "workflow.yaml"))

    def export_graph(self, quant_result, output_dir, device, config_overrides=None):
        # Substitute GPU graph export; execute the Spark hook and metadata writer.
        release.mkdir(parents=True)
        model_config = self.workflow_config.export["model"] if valid_metadata else []
        (release / "golden_meta_info.json").write_text(json.dumps({"model_config": model_config}))
        return export_result

    monkeypatch.setattr(BaseLLMWorkflow, "export", export_graph)
    quant_result = QuantResult(raw_model_dir=str(tmp_path), skipped=True)
    config_path = release / "merak_config.json"
    if not valid_metadata:
        with pytest.raises(TypeError, match="model_config"):
            workflow.export(quant_result, str(output), "cpu")
        assert not config_path.exists()
        return

    assert workflow.export(quant_result, str(output), "cpu") is export_result
    config = json.loads(config_path.read_text())
    assert config["architectures"] == ["MerakForCausalLM"]
    assert config["model_type"] == config["config_format"] == config["load_format"] == "merak_llm"
    assert config["xh_model"] == {"model_type": "hmonnx", "meta_info": "golden_meta_info.json"}
    assert config["enable_page_attention"] is False
    assert workflow.workflow_config.export["model"]["flash_attention"]["enable"] is True
    assert not config_path.with_name(".merak_config.json.tmp").exists()


@pytest.mark.parametrize(
    "change", [{"method": "gptq"}, {"bits": 5}, {"group_size": 128}, {"sym": False}, {"unknown": 1}]
)
def test_invalid_quant_configuration_rejected(adapter, quant_cfg, tmp_path, change):
    cfg = copy.deepcopy(quant_cfg)
    cfg.update(change)
    with pytest.raises(ValueError):
        adapter.build_autoround_command(model_dir="model", output_dir=tmp_path, device="cuda", quant_cfg=cfg)


def test_failed_worker_never_returns_quant_result(adapter, quant_cfg, tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"spark2_5"}')
    staged_paths = []

    def fail(command, **kwargs):
        staged_paths.append(Path(command[3]))
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(adapter.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        adapter.quantize_autoround(model_dir=model, output_dir=tmp_path / "out", device="cuda", quant_cfg=quant_cfg)
    assert not staged_paths[0].exists()


@pytest.mark.parametrize("auto_map", [None, {"AutoConfig": "configuration_spark.Spark2_5Config"}])
def test_result_uses_backend_checkpoint_subdirectory(adapter, quant_cfg, tmp_path, monkeypatch, auto_map):
    import torch
    from safetensors.torch import save_file

    model = tmp_path / "model"
    model.mkdir()
    source_config = {"model_type": "spark2_5"}
    if auto_map is not None:
        source_config["auto_map"] = auto_map
        (model / "configuration_spark.py").write_text("raise AssertionError('Must not execute checkpoint code')")
    (model / "config.json").write_text(json.dumps(source_config))
    source_bytes = (model / "config.json").read_bytes()

    def produce(command, **kwargs):
        # Only the expensive producer is substituted; path discovery and handoff are real.
        out = Path(command[command.index("--output_dir") + 1]) / "auto_gptq"
        out.mkdir(parents=True)
        (out / "config.json").write_text('{"model_type":"spark2_5","quantization_config":{"bits":4}}')
        save_file(
            {"model.layers.0.mlp.up_proj.qweight": torch.zeros((1, 8), dtype=torch.int32)}, out / "model.safetensors"
        )

    monkeypatch.setattr(adapter.subprocess, "run", produce)
    result = adapter.quantize_autoround(
        model_dir=model, output_dir=tmp_path / "out", device="cuda", quant_cfg=quant_cfg
    )
    assert result.quanted_model_dir == str(tmp_path / "out" / "auto_gptq")
    assert not result.skipped
    checkpoint = Path(result.quanted_model_dir)
    assert json.loads((checkpoint / "config.json").read_text()).get("auto_map") == auto_map
    if auto_map is not None:
        assert (checkpoint / "configuration_spark.py").read_bytes() == (model / "configuration_spark.py").read_bytes()
    assert (model / "config.json").read_bytes() == source_bytes
    with pytest.raises(FileExistsError):
        adapter.quantize_autoround(model_dir=model, output_dir=tmp_path / "out", device="cuda", quant_cfg=quant_cfg)


def test_recipes_are_discoverable():
    from xhmodel_merak.model_catalog.discovery import build_builtin_catalog

    catalog = build_builtin_catalog()
    for model_id, size, layers in (("spark-x2.5-1.7b", 2048, 28), ("spark-x2.5-4b", 2560, 36)):
        recipe = catalog.get_recipe(model_id)
        assert dict(recipe.source_constraints)["hidden_size"] == size
        assert dict(recipe.source_constraints)["num_hidden_layers"] == layers
        assert (ROOT / recipe.variants[0].workflow_config).is_file()


@pytest.mark.parametrize(
    "change,error,match",
    [
        ({"algorithm": "gptqmodel", "method": "gptq"}, NotImplementedError, "Spark GPTQ"),
        ({"algorithm": "gptq", "method": ""}, NotImplementedError, "Spark GPTQ"),
        ({"algorithm": "awq"}, NotImplementedError, "Unsupported Spark quant"),
        ({"method": "typo"}, NotImplementedError, "Unsupported Spark quant"),
        ({"algorithm": "autoround", "method": "gptq"}, ValueError, "conflicts"),
        ({"artifact_format": "invalid"}, ValueError, "artifact_format"),
        ({"output_format": "invalid"}, ValueError, "output_format"),
    ],
)
def test_workflow_rejects_before_producer(tmp_path, change, error, match):
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.workflow import Spark2_5Workflow

    workflow = Spark2_5Workflow(str(tmp_path), str(CONFIG))
    workflow.workflow_config.data["quant"].update(change)
    with pytest.raises(error, match=match):
        workflow.quant(str(tmp_path / "out"), "cuda")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("entry", ["quant", "skip", "export"])
def test_low_bit_head_rejected_even_when_skipping(tmp_path, entry):
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.workflow import Spark2_5Workflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    workflow = Spark2_5Workflow(str(tmp_path), str(CONFIG))
    workflow.workflow_config.data["quant"]["head"] = {"bits": 4}
    workflow.workflow_config.data["export"]["model"]["quant_scheme"]["nodes"]["lm_head"] = {"quant_type": "w4a8h1_sefp"}
    with pytest.raises(NotImplementedError, match="low-bit lm_head"):
        if entry == "export":
            workflow.export(QuantResult(raw_model_dir=str(tmp_path)), str(tmp_path / "out"), "cpu")
        else:
            workflow.quant(str(tmp_path / "out"), "cpu", {"quant": None} if entry == "skip" else None)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("head,error", [({"bits": 8}, None), ({"bits": 4}, ValueError), ([], TypeError)])
def test_head_contract_on_quant_skip(tmp_path, head, error):
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.workflow import Spark2_5Workflow

    workflow = Spark2_5Workflow(str(tmp_path), str(CONFIG))
    workflow.workflow_config.data["quant"]["head"] = head
    if error is None:
        result = workflow.quant(str(tmp_path / "out"), "cpu", {"quant": None})
        assert result.skipped and result.quanted_model_dir is None
    else:
        with pytest.raises(error, match="quant.head"):
            workflow.quant(str(tmp_path / "out"), "cpu", {"quant": None})
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("special_tokens", [False, True])
def test_real_local_calibration_tensor_contract(tmp_path, concat, special_tokens):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.autoround_recipe import prepare_spark_calibration

    tokenizer_impl = Tokenizer(
        WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3, "[BOS]": 4, "[EOS]": 5}, unk_token="[UNK]")
    )
    tokenizer_impl.pre_tokenizer = Whitespace()
    if special_tokens:
        tokenizer_impl.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 4), ("[EOS]", 5)]
        )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        unk_token="[UNK]",
        bos_token="[BOS]" if special_tokens else None,
        eos_token="[EOS]" if special_tokens else None,
    )
    seqlen = 8 if special_tokens else 6
    path = tmp_path / "samples.jsonl"
    # Concat must combine short rows, not silently discard them.
    text = "a b c" if concat else "a b c a b c"
    path.write_text("".join(json.dumps({"text": text}) + "\n" for _ in range(8)))
    batches = prepare_spark_calibration(tokenizer, str(path), seqlen=seqlen, seed=42, nsamples=2, concat=concat)
    assert len(batches) == 2
    for batch in batches:
        assert isinstance(batch["input_ids"], torch.Tensor)
        expected = [1, 2, 3, 1, 2, 3]
        if special_tokens and not concat:
            expected = [4, *expected, 5]
        if not (special_tokens and concat):
            assert batch["input_ids"].tolist() == [expected]
        assert batch["input_ids"].shape == (1, seqlen)
        assert batch["attention_mask"].all()
        assert batch["attention_mask"].shape == batch["input_ids"].shape


def test_worker_uses_gptqmodel_bundled_backend(adapter, quant_cfg, tmp_path):
    import os
    import sys

    command = adapter.build_autoround_command(
        model_dir="model", output_dir=tmp_path, device="cuda", quant_cfg=quant_cfg
    )
    script = """
import json, sys
from pathlib import Path
from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.autoround_recipe import autoround_root, require_bundled_autoround
backend = require_bundled_autoround()
assert Path(backend.__file__).resolve().is_relative_to(autoround_root().resolve())
from auto_round.utils.common import parse_layer_config_arg
layers = parse_layer_config_arg(sys.argv[1])
assert layers['model.layers.[0-9]+.self_attn.(q_k_v_proj|out_proj)']['bits'] == 4
assert layers['model.layers.[0-9]+.mlp.(up_proj|down_proj)']['bits'] == 4
assert layers['model.layers.[0-9]+.mlp.gate_proj']['bits'] == 16
assert layers['model.layers.[0-9]+.self_attn.g_proj']['bits'] == 16
assert layers['lm_head']['bits'] == 16
print('BUNDLED_BACKEND_OK', backend.__file__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, command[command.index("--layer_config") + 1]],
        # Source-based CI supplies SDK and GPTQModel checkouts on PYTHONPATH.
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BUNDLED_BACKEND_OK" in result.stdout


@pytest.mark.parametrize("trust", [None, False, True])
def test_spark_checkpoint_uses_local_config_without_executing_code(tmp_path, trust):
    import torch
    from _unified_small_model import write_native_checkpoint

    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5 import XHSpark2_5Model
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.spark_2_5 import Spark2_5ForCausalLM

    source = tmp_path / "source"
    write_native_checkpoint(source, "spark")
    path = source / "config.json"
    config = json.loads(path.read_text())
    config["auto_map"] = {
        "AutoConfig": "remote_config.RemoteConfig",
        "AutoModelForCausalLM": "remote_model.RemoteModel",
    }
    path.write_text(json.dumps(config))
    for name in ("remote_config.py", "remote_model.py"):
        (source / name).write_text("raise AssertionError('Checkpoint code executed')")
    before = path.read_bytes()
    kwargs = {} if trust is None else {"trust_remote_code": trust}
    model = XHSpark2_5Model.get_hf_model(str(source), dtype=torch.float32, **kwargs)
    assert isinstance(model, Spark2_5ForCausalLM)
    assert model.config.model_type == "spark2_5"
    assert model.config.architectures == ["Spark2_5ForCausalLM"]
    assert "auto_map" not in model.config.to_dict()
    assert path.read_bytes() == before
    empty = XHSpark2_5Model.get_empty_hf_model(str(source), **kwargs)
    assert isinstance(empty, Spark2_5ForCausalLM)
    assert next(empty.parameters()).device.type == "meta"


@pytest.mark.parametrize("model_type", ["spark3", "qwen3", None])
@pytest.mark.parametrize(
    "entry", ["get_hf_model", "get_empty_hf_model", "_load_gptqmodel", "_load_hf_model"]
)
def test_spark_rejects_mismatched_checkpoint_type_without_modifying_it(tmp_path, model_type, entry):
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5 import XHSpark2_5Model
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.spark_2_5 import Spark2_5Config

    path = tmp_path / "config.json"
    config = {"model_type": model_type, "num_hidden_layers": 2, "auto_map": {"AutoConfig": "remote_config.Config"}}
    if model_type is None:
        config.pop("model_type")
    path.write_text(json.dumps(config))
    (tmp_path / "remote_config.py").write_text("raise AssertionError('Checkpoint code executed')")
    before = path.read_bytes()
    # An explicit config cannot override a mismatched source checkpoint.
    kwargs = {"config": Spark2_5Config()} if entry == "_load_hf_model" else {}
    with pytest.raises(ValueError, match="Expected Spark checkpoint model_type='spark2_5'") as error:
        getattr(XHSpark2_5Model, entry)(str(tmp_path), **kwargs)
    assert f"got {model_type!r}" in str(error.value)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json", "remote_config.py"]


def test_spark_page_attention_rejected_before_quantization(tmp_path):
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.spark2_5_hmonnx_inference import XHSpark2_5HMONNXModel
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.spark2_5_model import XHSpark2_5ModelConfig
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(model_dir=str(tmp_path), config_path=str(CONFIG))
    with pytest.raises(ValueError, match="does not support PageAttention"):
        workflow.quant(str(tmp_path / "quant"), "cpu", {"export.model.enable_page_attention": True})
    assert not (tmp_path / "quant").exists()
    with pytest.raises(ValueError, match="does not support PageAttention"):
        XHSpark2_5ModelConfig(model_name="spark", enable_page_attention=True)
    assert XHSpark2_5ModelConfig(model_name="spark").enable_page_attention is False
    with pytest.raises(ValueError, match="does not support PageAttention"):
        XHSpark2_5HMONNXModel(None, enable_page_attention=True)


@pytest.mark.parametrize("past_length,input_ids", [(0, [[1, 2]]), (2, [[3]])])
def test_spark_nonpaged_sliding_window_processor(past_length, input_ids):
    import torch

    from xhmodel_merak.xh_llm.llm_data_processor import (
        SlidingWindowInputProcessorConfig,
        SlidingWindowLLMInputProcessor,
    )
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.spark2_5_model import XHSpark2_5ModelConfig
    from xhmodel_merak.xh_llm.types import CacheList

    model_config = XHSpark2_5ModelConfig(model_name="spark")
    keys = CacheList([torch.zeros(1, 1, 8, 4)])
    values = CacheList([torch.zeros(1, 1, 8, 4)])
    processor = SlidingWindowLLMInputProcessor(
        SlidingWindowInputProcessorConfig(
            embed_tokens=torch.nn.Embedding(8, 4),
            input_sequence_length=2,
            past_key_caches=keys,
            past_value_caches=values,
            sliding_window=4,
            enable_page_attention=model_config.enable_page_attention,
        )
    )
    result = processor({"input_ids": torch.tensor(input_ids), "past_seq_length": past_length})
    assert len(result) == 6
    assert result[1].item() == past_length
    assert result[2].item() == len(input_ids[0])
    assert result[4] is keys and result[5] is values
