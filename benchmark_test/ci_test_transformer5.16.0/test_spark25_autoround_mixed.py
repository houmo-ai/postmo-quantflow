"""CPU CLI and real backend configuration contracts, not AutoRound training."""

import argparse
import ast
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def example():
    # Execute the actual pure helpers without importing the unrelated GPU/model
    # registration stack. Include the real config reader and command builder.
    path = Path(__file__).resolve().parents[2] / "examples_merak/llm/xh_token_spark_2_5/spark_tokens_autoround_mixed.py"
    names = {"read_model_config", "build_mixed_layer_config_arg", "build_default_output_dir", "build_autoround_command"}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in tree.body} == names
    namespace = dict(argparse=argparse, json=json, os=os, datetime=datetime, Path=Path, re=re, sys=sys)
    exec(compile(tree, str(path), "exec"), namespace)
    return argparse.Namespace(**{name: namespace[name] for name in names})


def _parse_layer_config(value):
    """Split entries only at the outer brace level; reject malformed fields."""
    assert value.startswith("{") and value.endswith("}")
    entries, start, depth = [], 1, 0
    for index, char in enumerate(value[1:-1], start=1):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            assert depth >= 0
        elif char == "," and depth == 0:
            entries.append(value[start:index])
            start = index + 1
    assert depth == 0
    entries.append(value[start:-1])
    result = {}
    for entry in entries:
        pattern, fields = entry.split(":", 1)
        assert pattern not in result
        assert fields.startswith("{") and fields.endswith("}")
        pairs = [field.split(":") for field in fields[1:-1].split(",")]
        assert all(len(pair) == 2 for pair in pairs)
        assert len(dict(pairs)) == len(pairs)
        result[pattern] = dict(pairs)
    return result


@pytest.mark.parametrize(
    "layers,edges",
    [
        (36, {0, 1, 34, 35}),
        (32, {0, 1, 30, 31}),
        (28, {0, 1, 26, 27}),
        (5, {0, 1, 3, 4}),
        (4, {0, 1, 2, 3}),
        (3, {0, 2}),
        (2, {0, 1}),
        (1, set()),
    ],
)
def test_mixed_layer_config(example, tmp_path, layers, edges):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "spark2_5", "num_hidden_layers": layers}), encoding="utf-8"
    )
    config = _parse_layer_config(example.build_mixed_layer_config_arg(argparse.Namespace(model=str(tmp_path))))
    assert len(config) == (4 if edges else 3)
    assert list(config.values()).count({"bits": "16"}) == 2
    if layers == 32:
        assert "model.layers.(0|1|30|31).mlp.(gate_proj|up_proj|down_proj)" in config

    def overrides(name):
        return [fields for pattern, fields in config.items() if re.fullmatch(pattern, name)]

    for layer in range(layers):
        for projection in ("q_k_v_proj", "out_proj"):
            assert overrides(f"model.layers.{layer}.self_attn.{projection}") == [{"bits": "8", "data_type": "int"}]
        for projection in ("gate_proj", "up_proj", "down_proj"):
            expected = [{"bits": "8", "data_type": "int"}] if layer in edges else []
            assert overrides(f"model.layers.{layer}.mlp.{projection}") == expected
        assert overrides(f"model.layers.{layer}.self_attn.g_proj") == [{"bits": "16"}]
        for suffix in ("input_layernorm", "post_attention_layernorm", "RMSNorm"):
            assert not overrides(f"model.layers.{layer}.{suffix}")
    assert overrides("lm_head") == [{"bits": "16"}]
    for name in ("model.embedding", "model.norm"):
        assert not overrides(name)
    assert "RMSNorm" not in "".join(config)


@pytest.mark.parametrize("bits,profile", [(4, "w4w8"), (8, "w8")])
def test_command_preserves_global_bits(example, tmp_path, bits, profile):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "spark2_5", "num_hidden_layers": 32}), encoding="utf-8"
    )
    args = argparse.Namespace(
        model=str(tmp_path),
        bits=bits,
        group_size=128,
        data_type="int",
        sym=True,
        iters=1,
        nsamples=1,
        seqlen=16,
        batch_size=1,
        gradient_accumulate_steps=1,
        dataset="wikitext2",
        calibration_jsonl=None,
        concat_calibration=False,
        device_map=None,
        device="cuda:0",
        format="auto_round",
        seed=42,
        low_gpu_mem_usage=False,
        trust_remote_code=True,
    )
    args.out = example.build_default_output_dir(args)
    command = example.build_autoround_command(args)
    assert command[command.index("--scheme") + 1] == f"W{bits}A16"
    assert command[command.index("--bits") + 1] == str(bits)
    assert command[command.index("--output_dir") + 1] == args.out
    assert f"-mixed-{profile}-" in args.out
    assert command[command.index("--dataset") + 1] == "wikitext2"
    assert command[command.index("--device_map") + 1] == "0"
    assert all(isinstance(value, str) and value != "None" for value in command)
    assert command.count("--layer_config") == 1
    assert command[command.index("--layer_config") + 1] == example.build_mixed_layer_config_arg(args)


@pytest.mark.parametrize("bits,profile", [(4, "w4w8"), (8, "w8")])
def test_default_output_dir(example, bits, profile):
    args = argparse.Namespace(model="/models/spark2_5/", bits=bits, group_size=128, format="auto_round")
    output = example.build_default_output_dir(args)
    assert re.fullmatch(rf"work_dirs/spark2_5-autoround-mixed-{profile}-128g-auto_round_\d{{6}}", output)
    assert "fp16" not in output
    assert "w8w8" not in output


@pytest.mark.parametrize("script", ["spark_tokens_autoround.py", "spark_tokens_autoround_mixed.py"])
@pytest.mark.parametrize("data_type", [None, "int_sym_gptq"])
def test_cli_data_type_default_and_override(tmp_path, monkeypatch, script, data_type):
    path = Path(__file__).resolve().parents[2] / "examples_merak/llm/xh_token_spark_2_5" / script
    names = {"parse_args", "read_model_config", "build_mixed_layer_config_arg", "build_autoround_command"}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(argparse=argparse, json=json, os=os, Path=Path, re=re, sys=sys, __file__=str(path))
    exec(compile(tree, str(path), "exec"), namespace)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "spark2_5", "num_hidden_layers": 4}), encoding="utf-8"
    )
    argv = [script, "--model", str(tmp_path), "--out", str(tmp_path / "out"), "--dataset", "wikitext2"]
    if data_type is not None:
        argv.extend(["--data-type", data_type])
    monkeypatch.setattr(sys, "argv", argv)
    args = namespace["parse_args"]()
    expected = data_type or "int_sym_xh_hessian"
    assert args.data_type == expected
    command = namespace["build_autoround_command"](args)
    assert command[command.index("--data_type") + 1] == expected


@pytest.mark.parametrize("heads", [8, 32])
@pytest.mark.parametrize("bits", [4, 8])
def test_backend_precision_on_real_spark(example, tmp_path, heads, bits):
    import torch

    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.autoround_recipe import require_bundled_autoround
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.spark_2_5.configuration_spark import Spark2_5Config
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.spark_2_5.modeling_spark import Spark2_5ForCausalLM

    require_bundled_autoround()
    from auto_round.compressors.utils import set_layer_config
    from auto_round.utils.common import INNER_SUPPORTED_LAYER_TYPES, SUPPORTED_LAYER_TYPES, parse_layer_config_arg
    from auto_round.utils.model import check_to_quantized

    config = Spark2_5Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=heads,
        num_key_value_heads=8,
        head_dim=8,
        headwise_attn_output_gate=True,
        tie_word_embeddings=True,
    )
    model = Spark2_5ForCausalLM(config)
    config.save_pretrained(tmp_path)
    overrides = parse_layer_config_arg(example.build_mixed_layer_config_arg(argparse.Namespace(model=str(tmp_path))))
    resolved, _, _ = set_layer_config(
        model,
        overrides,
        f"W{bits}A16",
        torch.float32,
        SUPPORTED_LAYER_TYPES,
        INNER_SUPPORTED_LAYER_TYPES,
        quant_block_list=[[f"model.layers.{i}" for i in range(3)]],
    )
    for i in range(3):
        for suffix in ("self_attn.q_k_v_proj", "self_attn.out_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            name = f"model.layers.{i}.{suffix}"
            assert resolved[name]["bits"] == (bits if i == 1 and suffix.startswith("mlp.") else 8)
            assert check_to_quantized(resolved[name])
        name = f"model.layers.{i}.self_attn.g_proj"
        assert resolved[name]["bits"] == 16
        assert not check_to_quantized(resolved[name])
    assert model.lm_head.weight is model.model.embedding.weight
    for name, module in model.named_modules():
        if name in ("model.embedding", "lm_head") or "norm" in name:
            assert name not in resolved
            assert module.weight.is_floating_point()
