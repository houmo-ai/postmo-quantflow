"""Real component graph publication ABI; SDK media execution has separate coverage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import onnx
import pytest

from xhmodel_merak.errors import ModelError
from xhmodel_merak.xh_llm.models.qwen4_exp.vision_contract import validate_qwen4_exp_vision


@pytest.fixture(scope="module")
def exported_visual(tmp_path_factory):
    # Checkpoint construction/export stays in a child process, so this module
    # fixture cannot leak HF's initialization switches into later tests.
    root = tmp_path_factory.mktemp("qwen4_visual_artifact")
    script = """
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'benchmark_test/ci_test_transformer5.16.0'))
from test_qwen4_exp_vision import _tiny_checkpoint
from xhmodel_merak.xh_llm.models.qwen4_exp.vision import export_qwen4_exp_visual
root = Path(sys.argv[1])
_tiny_checkpoint(root / 'source')
meta = export_qwen4_exp_visual(root / 'source', root / 'export', image_token_gears=[2,4],
    quant_scheme={'quant_type':'w16a16h1_sefp'}, device='cpu', dtype='float32')
(root / 'metadata.json').write_text(json.dumps(meta))
"""
    with (root / "export.log").open("w") as log:
        subprocess.run(
            [sys.executable, "-c", script, str(root)],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "ENABLE_HMINFERENCE_V2": "1"},
            timeout=180,
        )
    return root


@pytest.fixture
def bundle(exported_visual, tmp_path):
    root = tmp_path / "bundle"
    shutil.copytree(exported_visual / "export", root)
    shutil.copytree(exported_visual / "source", root / "hf_config")
    visual = json.loads((exported_visual / "metadata.json").read_text())
    metadata = {"qwen4_exp_vision": visual, "visual_config": None, "hf_config": "hf_config"}
    (root / "golden_meta_info.json").write_text(json.dumps(metadata))
    return root, metadata


def _sync_manifest(root, visual):
    manifest = root / "visual_gears.json"
    document = json.loads(manifest.read_text())
    document["gears"] = visual["gears"]
    manifest.write_text(json.dumps(document))


def test_real_components_validate_and_publish_unique_entries(bundle):
    root, meta = bundle
    visual = meta["qwen4_exp_vision"]
    result = validate_qwen4_exp_vision(visual, root, image_token_gears=[2, 4], vision_config=visual["vision_config"])
    expected = {f"visual_m{gear}_{name}" for gear in (2, 4) for name in ("stem", "block_00", "merger")}
    assert set(result["export_entries"]) == expected
    paths = [entry["path"] for entry in result["export_entries"].values()]
    assert len(set(paths)) == 6
    for path in paths:
        onnx.checker.check_model(str(root / path))


@pytest.mark.parametrize(
    "field,value",
    [
        ("abi_version", True),
        ("dtype", "float32"),
        ("routing_policy", "largest_fit"),
        ("image_token_gears", [4, 2]),
        ("image_token_capacity", 8),
        ("patch_size", 4),
        ("hidden_size", 32),
        ("num_heads", 4),
        ("in_channels", 1),
        ("num_position_embeddings", 64),
        ("patch_token_capacity", 8),
    ],
)
def test_component_metadata_rejects_inconsistent_declarations(bundle, field, value):
    root, metadata = bundle
    visual = metadata["qwen4_exp_vision"]
    visual[field] = value
    with pytest.raises(ValueError):
        validate_qwen4_exp_vision(visual, root)


@pytest.mark.parametrize(
    "defect",
    [
        "profile",
        "source",
        "order",
        "checksum",
        "manifest",
        "absolute",
        "traversal",
        "symlink",
    ],
)
def test_component_resources_reject_wrong_source_order_hash_and_paths(bundle, defect, tmp_path):
    root, metadata = bundle
    visual = metadata["qwen4_exp_vision"]
    kwargs = {}
    part = visual["gears"][0]["components"][0]
    if defect == "profile":
        kwargs["image_token_gears"] = [2, 8]
    elif defect == "source":
        kwargs["vision_config"] = {**visual["vision_config"], "hidden_size": 32}
    elif defect == "order":
        visual["gears"][0]["components"].reverse()
    elif defect == "checksum":
        part["md5"] = "0" * 32
    elif defect == "manifest":
        (root / "visual_gears.json").write_text("{}")
    elif defect in {"absolute", "traversal"}:
        part["hmonnx"] = str((root / part["hmonnx"]).resolve()) if defect == "absolute" else "../stem.onnx"
    elif defect == "symlink":
        path = root / part["hmonnx"]
        outside = tmp_path / "outside.onnx"
        shutil.copyfile(path, outside)
        path.unlink()
        path.symlink_to(outside)
    if defect != "manifest":
        _sync_manifest(root, visual)
    with pytest.raises(ValueError):
        validate_qwen4_exp_vision(visual, root, **kwargs)


@pytest.mark.parametrize(
    "role,port,defect",
    [
        ("stem", "input", "dtype"),
        ("stem", "output", "shape"),
        ("block_00", "input", "shape"),
        ("block_00", "output", "dtype"),
        ("merger", "input", "name"),
        ("merger", "output", "shape"),
    ],
)
def test_component_graph_abi_is_checked_after_digest_refresh(bundle, role, port, defect):
    root, metadata = bundle
    visual = metadata["qwen4_exp_vision"]
    part = next(p for p in visual["gears"][0]["components"] if p["name"] == role)
    path = root / part["hmonnx"]
    graph = onnx.load(path)
    value = getattr(graph.graph, port)[0]
    if defect == "dtype":
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    elif defect == "shape":
        value.type.tensor_type.shape.dim[-1].dim_value += 1
    else:
        original_name = value.name
        value.name = "incorrect_port"
        for node in graph.graph.node:
            for index, name in enumerate(node.input):
                if name == original_name:
                    node.input[index] = value.name
    onnx.save(graph, path)
    part["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    _sync_manifest(root, visual)
    with pytest.raises(ValueError, match="tensor ABI"):
        validate_qwen4_exp_vision(visual, root)


@pytest.fixture
def plan(tmp_path):
    from test_qwen4_exp_unified_api import _config

    from xhmodel_merak.model_build.resolver import resolve_build_plan

    source = tmp_path / "plan_source"
    source.mkdir()
    config = _config()
    config["text_config"]["layer_types"] *= 12
    (source / "config.json").write_text(json.dumps(config))
    original = resolve_build_plan(
        model_id="qwen3.8-flash-next", model_dir=str(source), output_dir=str(tmp_path / "out")
    )
    return replace(original, vision_gears=(2, 4), vision_profile="tiny-components")


def test_public_build_delegates_all_text_fields_and_restores_visual_result(bundle, plan, monkeypatch):
    from xhmodel_merak.model_integrations.qwen4_exp import build

    root, metadata = bundle
    before = plan.to_dict()
    calls = []

    def shared(stage, scoped, *, speculative_validator):
        assert stage == root
        expected = replace(
            plan, vision_gears=(), vision_profile=None, components=tuple(c for c in plan.components if c != "visual")
        )
        assert scoped.to_dict() == expected.to_dict()
        assert callable(speculative_validator)
        calls.append("text")
        return {
            "components": list(scoped.components),
            "vision_profile": None,
            "export_entries": {"prefill": {"path": "prefill/model.onnx", "type": "hmonnx"}},
        }

    monkeypatch.setattr(build, "validate_autoregressive_artifact", shared)
    monkeypatch.setattr(build, "validate_serving_contract", lambda *args: calls.append("serving"))
    monkeypatch.setattr(build, "validate_external_ple_contract", lambda *args: {})
    result = build.validate_artifact(root, plan)
    assert calls == ["text", "serving"]
    assert result["components"] == list(plan.components)
    assert result["vision_profile"] == plan.vision_profile
    assert len(result["export_entries"]) == 7
    assert result["dynamic_vision"]["gears"] == metadata["qwen4_exp_vision"]["gears"]
    assert plan.to_dict() == before


def test_public_build_preserves_shared_validation_failure(bundle, plan, monkeypatch):
    reason = "shared text length mismatch"
    from xhmodel_merak.errors import ErrorCode
    from xhmodel_merak.model_integrations.qwen4_exp import build

    def reject(*args, **kwargs):
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, reason)

    monkeypatch.setattr(build, "validate_autoregressive_artifact", reject)
    with pytest.raises(ModelError, match=reason):
        build.validate_artifact(bundle[0], plan)


def test_public_load_scopes_only_legacy_vision_geometry(bundle, monkeypatch):
    from xhmodel_merak.model_catalog.registry import get_recipe
    from xhmodel_merak.model_integrations.qwen4_exp import artifact

    root, _ = bundle
    recipe = replace(get_recipe("qwen3.8-flash-next"), vision_gears=(2, 4))
    calls = []
    manifest = {"marker": "original"}

    def shared(actual_root, actual_manifest, metadata_path, scoped):
        assert actual_root == root and actual_manifest is manifest
        assert metadata_path == root / "golden_meta_info.json"
        assert scoped == replace(recipe, vision_gears=(), vision_profile=None)
        calls.append("text")

    monkeypatch.setattr(artifact, "validate_autoregressive_artifact", shared)
    monkeypatch.setattr(artifact, "validate_serving_contract", lambda *args: calls.append("serving"))
    monkeypatch.setattr(artifact, "validate_external_ple_contract", lambda *args: {})
    artifact.validate_artifact(root, manifest, root / "golden_meta_info.json", recipe)
    assert calls == ["text", "serving"]


def test_component_public_validation_rejects_dual_abi_and_source_geometry(bundle):
    from xhmodel_merak.model_integrations.qwen4_exp.artifact import validate_component_vision

    root, metadata = bundle
    metadata["visual_config"] = {"hmonnx": "legacy.onnx"}
    with pytest.raises(ModelError, match="one vision ABI"):
        validate_component_vision(root, metadata)
    metadata["visual_config"] = None
    source = root / "hf_config/config.json"
    config = json.loads(source.read_text())
    config["vision_config"]["hidden_size"] *= 2
    source.write_text(json.dumps(config))
    with pytest.raises(ModelError, match="source configuration mismatch"):
        validate_component_vision(root, metadata)


@pytest.mark.parametrize("draft_count", [1, 2])
def test_component_public_build_retains_mtp_v3_request_gate(bundle, plan, monkeypatch, draft_count):
    from xhmodel_merak.model_integrations.qwen4_exp import build

    root, metadata = bundle
    metadata["spec_decode"] = {"mode": "mtp", "runtime_contract_version": 3, "num_draft_tokens": 1}
    (root / "golden_meta_info.json").write_text(json.dumps(metadata))
    mtp_plan = replace(
        plan,
        components=(*plan.components, "mtp"),
        config_overrides={**plan.config_overrides, "export.model.num_draft_tokens": draft_count},
    )
    checks = []

    def shared(stage, scoped, *, speculative_validator):
        assert "mtp" in scoped.components
        speculative_validator(stage, metadata, scoped)
        return {"export_entries": {}}

    monkeypatch.setattr(build, "validate_autoregressive_artifact", shared)
    monkeypatch.setattr(build, "validate_serving_contract", lambda *args: checks.append("serving"))
    monkeypatch.setattr(build, "validate_external_ple_contract", lambda *args: {})
    # The closure's request gate is real; unrelated graph/cache checks are the seam.
    monkeypatch.setattr(build, "mtp_v3_graph_entries", lambda *args: {})
    if draft_count == 2:
        with pytest.raises(ModelError, match="Unexpected Qwen4 MTP candidate contract"):
            build.validate_artifact(root, mtp_plan)
        assert checks == []
    else:
        build.validate_artifact(root, mtp_plan)
        assert checks == ["serving", "serving"]


def test_public_component_checker_failure_is_a_model_error(bundle):
    from xhmodel_merak.model_integrations.qwen4_exp.artifact import validate_component_vision

    root, metadata = bundle
    visual = metadata["qwen4_exp_vision"]
    part = visual["gears"][0]["components"][-1]
    path = root / part["hmonnx"]
    graph = onnx.load(path)
    graph.graph.node[0].input[0] = "missing_producer"
    onnx.save(graph, path)
    part["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    _sync_manifest(root, visual)
    with pytest.raises(ModelError, match="Invalid Qwen4 vision component graph"):
        validate_component_vision(root, metadata)
