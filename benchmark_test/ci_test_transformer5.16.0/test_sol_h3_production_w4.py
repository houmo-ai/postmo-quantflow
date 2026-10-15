# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Production precision and composite checkpoint identity cannot silently drift."""

import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import yaml
from test_sol_h3_unified_api import bundle_fixture, source_fixture

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_build.checkpoint import verify_quant_model_identity
from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_build.workflow import materialize_workflow_config
from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.model_integrations.sol_h3.build import validate_artifact
from xhmodel_merak.model_integrations.sol_h3.inference import adapter


@pytest.mark.parametrize("changed", [None, "hidden_size", "rope_parameters"])
def test_checkpoint_identity_reads_qwen_component_and_preserves_behavior(tmp_path, changed):
    source = tmp_path / "source"
    source.mkdir()
    source_fixture(source)
    config = {
        "model_type": "qwen3_vl",
        "text_config": {
            "model_type": "qwen3_vl_text", "hidden_size": 256, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 64,
            "rope_theta": 1000000.0,
            "rope_scaling": {"rope_type": "default", "mrope_section": [8, 12, 12]},
        },
        "vision_config": {"model_type": "qwen3_vl", "hidden_size": 128},
    }
    (source / "h3_model/text_encoder/config.json").write_text(json.dumps(config))
    (source / "config.json").write_bytes((source / "h3_model/text_encoder/config.json").read_bytes())
    plan = resolve_build_plan(model_id="sol-h3", model_dir=str(source), output_dir=str(tmp_path / "out"))
    checkpoint = copy.deepcopy(config)
    text = checkpoint["text_config"]
    text["rope_parameters"] = {**text.pop("rope_scaling"), "rope_theta": text.pop("rope_theta")}
    checkpoint["vision_config"]["model_type"] = "qwen3_vl_vision"
    if changed == "hidden_size":
        text[changed] = 512
    elif changed == "rope_parameters":
        text[changed]["rope_theta"] = 10000.0
    destination = tmp_path / "checkpoint"
    destination.mkdir()
    file = destination / "config.json"
    file.write_text(json.dumps(checkpoint))
    if changed is None:
        verify_quant_model_identity(destination, [(file, checkpoint)], plan)
    else:
        with pytest.raises(ModelError):
            verify_quant_model_identity(destination, [(file, checkpoint)], plan)
    assert json.loads((source / "config.json").read_text()) == config
    assert json.loads((source / "h3_model/text_encoder/config.json").read_text()) == config


def test_explicit_unquantized_checkpoint_build_remains_available(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    source_fixture(source)
    plan = resolve_build_plan(
        model_id="sol-h3", model_dir=str(source), output_dir=str(tmp_path / "out"), quant_method=None,
    )
    assert plan.precision.weight == "w16" and plan.quant_method is None
    destination = tmp_path / "workflow.yaml"
    materialize_workflow_config(plan, destination)
    data = yaml.safe_load(destination.read_text())
    assert data["export"]["conditioner_quant_type"] is None


@pytest.mark.parametrize("phase", ["build", "load"])
@pytest.mark.parametrize(
    "weight,graph_precision,accepted",
    [
        ("w4", "w8a8_sefp", False),
        ("w4", "w4a8h0_ssfp", True),
        ("w16", "w8a8_sefp", True),
        ("w16", "w4a8h0_ssfp", False),
    ],
)
def test_build_and_load_enforce_same_qwen_precision(tmp_path, phase, weight, graph_precision, accepted):
    # Exercise the real adapters and serialized inventory without allocating
    # full model weights or a runtime session. The fixture graphs have no GEMMs.
    meta = bundle_fixture(tmp_path)
    meta["graphs"][meta["feature_graphs"][0]]["quant_type"] = graph_precision
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    deployment = {"video": "w8a8_sefp"}
    expectation = nullcontext() if accepted else pytest.raises(ValueError, match="Qwen feature precision")
    with expectation:
        if phase == "build":
            validate_artifact(
                tmp_path,
                SimpleNamespace(
                    components=("video",), deployment_precision=deployment,
                    profile=meta["profile"], precision=SimpleNamespace(weight=weight),
                ),
            )
        else:
            manifest = {
                "parameters": {
                    "profile": meta["profile"],
                    "precision": {"checkpoint": {"weight": weight}, "deployment": deployment},
                },
                "artifact": {"files": [
                    {"path": path.relative_to(tmp_path).as_posix()}
                    for path in tmp_path.rglob("*") if path.is_file()
                ]},
            }
            adapter.validate_artifact(
                recipe=get_recipe("sol-h3"), root=tmp_path, manifest=manifest,
                meta_file=tmp_path / "runtime_meta_info.json",
            )


def test_w4_workflow_requires_checkpoint_before_loading_weights(tmp_path):
    from xhmodel_merak.xh_other_model.models.sol_h3.build import build_bundle

    source = tmp_path / "source"
    source.mkdir()
    source_fixture(source)
    plan = resolve_build_plan(model_id="sol-h3", model_dir=str(source), output_dir=str(tmp_path / "out"))
    workflow = tmp_path / "workflow.yaml"
    materialize_workflow_config(plan, workflow)
    config = yaml.safe_load(workflow.read_text())["export"]
    with pytest.raises(ValueError, match="requires matching weights"):
        build_bundle(source, tmp_path / "bundle", config, device="cpu")
