# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""DiT bundle metadata, gear bounds and package-relative resource validation."""

import copy
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("quant_type", ["w16a16_sefp", "w8a16_sefp", "w8a8_sefp"])
def test_build_calibration_disables_tf32_and_restores_caller(monkeypatch, tmp_path, fails, quant_type):
    import torch

    from xhmodel_merak.xh_other_model.models.qwen_image21 import build

    # The expensive graph exporter is outside this process-state contract.
    config = dict(
        target_device="XH2a",
        quant_type=quant_type,
        output_size=[32, 32],
        reference_size=[32, 32],
        prompt_budget=32,
        max_position=256,
    )
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

    def export(*args, **kwargs):
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        if fails:
            raise RuntimeError("conversion failed")
        return {"completed": True}

    monkeypatch.setattr(build, "_build_bundle", export)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if fails:
            with pytest.raises(RuntimeError, match="conversion failed"):
                build.build_bundle(tmp_path, tmp_path / "output", config)
        else:
            assert build.build_bundle(tmp_path, tmp_path / "output", config) == {"completed": True}
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


@pytest.fixture
def artifacts():
    path = Path(__file__).resolve().parents[2] / "xhmodel_merak/xh_other_model/models/qwen_image21/artifacts.py"
    spec = importlib.util.spec_from_file_location("qwen_image21_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "field,value", [("encoder_tokens", True), ("image_tokens", 0), ("prefix_tokens", -1), ("target_tokens", 9)]
)
def test_gear_rejects_invalid_capacity(artifacts, field, value):
    values = dict(encoder_tokens=8, image_tokens=8, prefix_tokens=16, target_tokens=4)
    values[field] = value
    with pytest.raises(ValueError):
        artifacts.DenoiseGear(**values)


def test_metadata_checks_layer_pairs_and_resource_paths(artifacts, tmp_path):
    # Resource existence is checked here; actual graph export/load is a separate SDK test.
    graphs = {name: f"{name}/model.onnx" for name in ("text", "image", "time", "head", "extract_0", "cached_0")}
    for relative in graphs.values():
        path = tmp_path / relative
        path.parent.mkdir()
        path.touch()
    meta = dict(
        schema_version=1,
        kind="qwen_image21_dit",
        input_dtype="float16",
        gear=asdict(artifacts.DenoiseGear(8, 8, 16, 4)),
        layers=1,
        heads=2,
        head_dim=64,
        context_channels=96,
        latent_channels=64,
        graphs=graphs,
    )
    metadata_file = tmp_path / "runtime_meta_info.json"
    metadata_file.write_text(json.dumps(meta))
    actual, gear = artifacts.read_dit_metadata(tmp_path)
    assert actual == meta
    assert gear.name == "p16_t4"
    for bad in ("../escape.onnx", "/absolute.onnx", "text\\model.onnx", "./text/model.onnx", "missing.onnx"):
        with pytest.raises(ValueError):
            artifacts.bundle_path(tmp_path, bad)
    link = tmp_path / "link.onnx"
    link.symlink_to(tmp_path / graphs["text"])
    with pytest.raises(ValueError, match="symlink"):
        artifacts.bundle_path(tmp_path, "link.onnx")
    del meta["graphs"]["cached_0"]
    metadata_file.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="paired layer"):
        artifacts.read_dit_metadata(tmp_path)


@pytest.mark.parametrize("quant_type", ["w16a16_sefp", "w8a16_sefp", "w8a8_sefp"])
def test_complete_bundle_is_relocatable_and_checks_cross_component_gears(tmp_path, quant_type):
    from _qwen_image21_bundle_fixture import bundle_metadata

    from xhmodel_merak.xh_other_model.models.qwen_image21.build import export_entries
    from xhmodel_merak.xh_other_model.models.qwen_image21.bundle import read_bundle_metadata

    root = tmp_path / "original"
    root.mkdir()
    meta, dit, paths = bundle_metadata(quant_type)
    (root / "dit_meta_info.json").write_text(json.dumps(dit))
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    metadata = root / "runtime_meta_info.json"
    metadata.write_text(json.dumps(meta))
    assert read_bundle_metadata(root) == meta
    entries = export_entries(root, meta)
    assert entries["embedding"]["type"] == "embedding"
    assert entries["tokenizer"] == {"path": "processor", "type": "tokenizer"}
    assert entries["vae_decode_c1536x1536"]["type"] == "hmonnx"
    assert all(set(entry) == {"path", "type"} for entry in entries.values())
    relocated = tmp_path / "relocated"
    root.rename(relocated)
    assert read_bundle_metadata(relocated) == meta
    metadata = relocated / "runtime_meta_info.json"
    bad = copy.deepcopy(meta)
    bad["feature"]["hidden_size"] = 97
    metadata.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="dimensions disagree"):
        read_bundle_metadata(relocated)
    bad = copy.deepcopy(meta)
    bad["vae_decode"]["token_capacity"] = 4608
    metadata.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="token capacity"):
        read_bundle_metadata(relocated)
    bad = copy.deepcopy(meta)
    bad["visual"]["patch_capacity"] = 4608
    metadata.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="visual patch capacity"):
        read_bundle_metadata(relocated)
    bad = copy.deepcopy(meta)
    bad["resources"]["embedding"] = "../escape"
    metadata.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="escape"):
        read_bundle_metadata(relocated)

    good = copy.deepcopy(meta)
    metadata.write_text(json.dumps(good))
    assert read_bundle_metadata(relocated) == good
    bad = copy.deepcopy(good)
    bad["quant_type"] = "w8a8_sefp" if quant_type == "w16a16_sefp" else "w16a16_sefp"
    metadata.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="deployment_config"):
        read_bundle_metadata(relocated)


@pytest.mark.parametrize(
    "name,value",
    [
        ("target_device", "M50"),
        ("quant_type", "w8a8"),
        ("output_size", [31, 32]),
        ("reference_size", [32]),
        ("feature_capacity", True),
        ("max_position", 0),
        ("prefix_capacity", 4),
    ],
)
def test_build_rejects_invalid_gears_before_loading_or_creating_output(tmp_path, name, value):
    from xhmodel_merak.xh_other_model.models.qwen_image21.build import build_bundle, validate_build_config

    config = dict(
        target_device="XH2a",
        quant_type="w16a16_sefp",
        output_size=[32, 32],
        reference_size=[32, 32],
        prompt_budget=32,
        max_position=256,
    )
    validate_build_config(config)
    config[name] = value
    with pytest.raises(ValueError):
        build_bundle(tmp_path / "missing_source", tmp_path / "output", config)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("precision", ["w16", "w8a16", "w8"])
def test_real_workflow_is_discovered_and_rejects_foreign_quant_result(tmp_path, precision):
    from xhmodel_merak.workflows import AutoWorkflow
    from xhmodel_merak.xh_other_model.workflows.result import QuantResult

    repository = Path(__file__).resolve().parents[2]
    config = repository / f"configs_merak/workflows/xh2a/other_models/qwen_image21/qwen_image21_1mp_{precision}.yaml"
    workflow = AutoWorkflow.from_config(model_dir=str(tmp_path), config_path=str(config))
    assert type(workflow).__name__ == "QwenImage21Workflow"
    result = workflow.quant(str(tmp_path / "quant"), "cuda:0")
    assert result.skipped and result.raw_model_dir == str(tmp_path)
    with pytest.raises(ValueError, match="same as self.model_dir"):
        workflow.export(
            QuantResult(raw_model_dir=str(tmp_path / "other"), skipped=True), str(tmp_path / "output"), "cuda:0"
        )
    with pytest.raises(ValueError, match="W16A16, W8A16 or W8A8"):
        workflow.quant(str(tmp_path / "quant"), "cuda:0", {"export.model.quant_scheme.quant_type": "w8a8"})
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("quant_type", ["w16a16_sefp", "w8a16_sefp", "w8a8_sefp"])
def test_nested_and_legacy_precision_fields_cannot_disagree(tmp_path, quant_type):
    from xhmodel_merak.xh_other_model.models.qwen_image21.build import (
        build_bundle,
        deployment_quant_type,
        validate_build_config,
    )

    config = dict(
        target_device="XH2a",
        output_size=[32, 32],
        reference_size=[32, 32],
        prompt_budget=32,
        max_position=256,
        model={"quant_scheme": {"target_device": "XH2a", "quant_type": quant_type}},
    )
    validate_build_config(config)
    assert deployment_quant_type(config) == quant_type
    config["quant_type"] = quant_type
    validate_build_config(config)
    config["model"]["quant_scheme"]["target_device"] = "M50"
    with pytest.raises(ValueError, match="target_device disagrees"):
        build_bundle(tmp_path / "missing", tmp_path / "output", config)
    config["model"]["quant_scheme"]["target_device"] = "XH2a"
    config["quant_type"] = "w8a8_sefp" if quant_type == "w16a16_sefp" else "w16a16_sefp"
    with pytest.raises(ValueError, match="quant_type disagrees"):
        build_bundle(tmp_path / "missing", tmp_path / "output", config)
    assert not (tmp_path / "output").exists()


def test_export_rejects_incompatible_semantics_before_creating_output(tmp_path):
    import importlib
    import types

    import torch

    package = types.ModuleType("qwen_image21_export_test")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "xhmodel_merak/xh_other_model/models/qwen_image21")]
    sys.modules[package.__name__] = package
    exporter = importlib.import_module(package.__name__ + ".export")
    artifacts_module = importlib.import_module(package.__name__ + ".artifacts")
    gear = artifacts_module.DenoiseGear(8, 8, 16, 4)
    source = types.SimpleNamespace(
        config=types.SimpleNamespace(causal_condition=False, patch_size=1), transformer_blocks=[object()]
    )
    layout = types.SimpleNamespace(prefix_capacity=16, target_tokens=4)
    arguments = dict(
        gear=gear,
        layout=layout,
        compact_rope=None,
        text_features=torch.zeros(1, 4, 96),
        image_latents=torch.zeros(1, 4, 64),
        timestep=0.9,
        target_device=None,
        quant_config=None,
    )
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="causal_condition"):
        exporter.export_dit_bundle(source, output, **arguments)
    source.config.causal_condition = True
    layout.prefix_capacity = 32
    with pytest.raises(ValueError, match="layout"):
        exporter.export_dit_bundle(source, output, **arguments)
    layout.prefix_capacity = 16
    arguments["text_features"] = arguments["text_features"].half()
    with pytest.raises(ValueError, match="FP32"):
        exporter.export_dit_bundle(source, output, **arguments)
    assert not output.exists()


def test_deployment_text_norm_preserves_large_feature_values():
    import importlib
    import types

    import torch
    from torch import nn

    from xhquant.nn import RMSNorm

    package = types.ModuleType("qwen_image21_norm_test")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "xhmodel_merak/xh_other_model/models/qwen_image21")]
    sys.modules[package.__name__] = package
    exporter = importlib.import_module(package.__name__ + ".export")
    graph = importlib.import_module(package.__name__ + ".graph")
    torch.manual_seed(37)
    source = nn.Module()
    source.txt_in = nn.Module()
    source.txt_in.text_norm = nn.RMSNorm(32, eps=1e-6)
    source.txt_in.text_norm.weight = nn.Parameter(torch.linspace(-0.9, 0.9, 32))
    source.txt_in.in_layer = nn.Linear(32, 32, bias=False)
    source.txt_in.act = nn.GELU(approximate="tanh")
    source.txt_in.out_layer = nn.Linear(32, 32, bias=False)
    weights = source.txt_in.text_norm.weight.detach().clone()
    module = exporter.make_text_projection(source)
    assert isinstance(module.norm, RMSNorm)
    features = torch.randn(1, 4, 32) * 0.1
    features[0, 0, 0] = 14000
    features[0, 1] *= 0.0001
    torch.testing.assert_close(module(features), graph.TextProjection(source)(features), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(source.txt_in.text_norm.weight, weights, atol=0, rtol=0)
    torch.testing.assert_close(module.norm.weight, 1 + weights, atol=0, rtol=0)


def test_config_rejects_reference_larger_than_visual_token_budget():
    from xhmodel_merak.xh_other_model.models.qwen_image21.build import validate_build_config

    config = dict(
        target_device="XH2a",
        quant_type="w16a16_sefp",
        output_size=[512, 768],
        reference_size=[512, 512],
        reference_capacity=4352,
        visual_patch_capacity=1024,
    )
    with pytest.raises(ValueError, match="visual patch capacity"):
        validate_build_config(config)
    config["visual_patch_capacity"] = 4352
    validate_build_config(config)


def test_complete_dit_metadata_requires_exact_graph_pair_and_cache_policy(artifacts, tmp_path):
    graphs = {"extract": "extract.onnx", "cached": "cached.onnx"}
    for path in graphs.values():
        (tmp_path / path).touch()
    meta = dict(
        schema_version=3,
        kind="qwen_image21_dit",
        input_dtype="float16",
        gear=dict(encoder_tokens=8, image_tokens=8, prefix_tokens=16, target_tokens=4),
        layers=2,
        heads=2,
        head_dim=64,
        context_channels=96,
        latent_channels=4,
        cache_policy="replace_target",
        attention="flash_attention",
        attention_metadata="query_kv_range_abs",
        target_layout="actual_then_padding",
        graphs=graphs,
    )
    path = tmp_path / "runtime_meta_info.json"
    path.write_text(json.dumps(meta))
    assert artifacts.read_dit_metadata(tmp_path)[0]["schema_version"] == 3
    meta["cache_policy"] = "append"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="replace_target"):
        artifacts.read_dit_metadata(tmp_path)
    meta["cache_policy"] = "replace_target"
    meta["graphs"]["head"] = "extract.onnx"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="graph"):
        artifacts.read_dit_metadata(tmp_path)
