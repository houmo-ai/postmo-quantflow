"""MiniMax-H3 unified-api registration and artifact packaging contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_build import describe_model
from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_build.workflow import materialize_workflow_config
from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.model_integrations.minimax_h3 import artifact, build, inference
from xhmodel_merak.model_integrations.minimax_h3.artifact import COMPONENTS


MODEL_ID = "minimax-h3-fl2va"


def _write_fixture(root: Path) -> dict:
    components = {}
    for name in COMPONENTS:
        relative = f"{name}/{name}.hmonnx"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        components[name] = {
            "hmonnx_file": relative,
            "input_names": ["x"],
            "output_names": ["y"],
            "inputs": [{"name": "x", "shape": [1, 8], "dtype": "float16"}],
        }
    meta = {
        "model_type": "XHMiniMaxH3Model",
        "target_device": "XH2a",
        "components": components,
    }
    (root / "export_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def test_public_plan_materializes_minimax_workflow(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    plan = resolve_build_plan(
        model_id=MODEL_ID,
        model_dir=str(source),
        output_dir=str(tmp_path / "artifact"),
        device="cpu",
    )
    recipe = get_recipe(MODEL_ID)
    assert plan.components == COMPONENTS
    assert plan.build_adapter == "minimax_h3.build"
    assert recipe.inference_adapter == "minimax_h3.inference"
    assert plan.artifact_format == "aigc"
    assert plan.artifact_name == "hmquant_xh2_minimax_h3_fl2va_w8a8_ssfp"
    assert plan.config_replacements["export.model.model_name"] == "minimax_h3_fl2va"

    effective = tmp_path / "workflow.yaml"
    materialize_workflow_config(plan, effective)
    data = yaml.safe_load(effective.read_text(encoding="utf-8"))
    assert data["export"]["model"]["type"] == "XHMiniMaxH3Model"
    assert data["export"]["model"]["model_name"] == "minimax_h3_fl2va"
    assert data["export"]["minimax_h3"]["components"] == list(COMPONENTS)


def test_describe_model_exposes_minimax_recipe() -> None:
    description = describe_model(MODEL_ID, full=True)
    assert description["model_id"] == MODEL_ID
    assert description["task"] == "image-to-video"
    assert description["components"] == list(COMPONENTS)
    assert description["artifact_format"] == "aigc"
    assert description["input_schema"]["required"] == ["prompt"]
    assert set(description["deployment_precision"]) == set(COMPONENTS)


def test_validate_artifact_accepts_complete_component_set(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    meta = _write_fixture(root)
    plan = SimpleNamespace(model_id=MODEL_ID, components=COMPONENTS)
    result = build.validate_artifact(root, plan)
    assert result["loadable"] is True
    assert result["metadata_file"] == "export_meta_info.json"
    assert set(result["export_entries"]) == set(COMPONENTS)
    assert all(entry["type"] == "hmonnx" for entry in result["export_entries"].values())
    inference.adapter.validate_artifact(
        recipe=get_recipe(MODEL_ID),
        root=root,
        manifest={"artifact": {"files": {}}},
        meta_file=root / "export_meta_info.json",
    )
    assert artifact.load_metadata(root / "export_meta_info.json")["model_type"] == meta["model_type"]


def test_validate_artifact_rejects_incomplete_component_set(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    meta = _write_fixture(root)
    del meta["components"]["transformer"]
    (root / "export_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")
    plan = SimpleNamespace(model_id=MODEL_ID, components=COMPONENTS)
    with pytest.raises(ModelError) as error:
        build.validate_artifact(root, plan)
    assert error.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED


def test_unified_inference_is_explicitly_unsupported(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    _write_fixture(root)
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpg")
    with pytest.raises(ModelError) as error:
        inference.adapter.run(
            recipe=get_recipe(MODEL_ID),
            root=root,
            manifest={"artifact": {"files": {}}},
            meta_file=root / "export_meta_info.json",
            inputs={"prompt": "a clay fox", "image": str(image), "workflow": "fl2va"},
            device="cpu",
            components=COMPONENTS,
            selected_golden=(),
            golden_artifacts={},
        )
    assert error.value.code == ErrorCode.UNSUPPORTED_COMBINATION
