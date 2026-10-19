"""DepthSplat catalog identity and artifact trust-boundary contracts."""

import json
from types import SimpleNamespace

import pytest
import onnx
from onnx import TensorProto, helper

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_catalog.discovery import build_builtin_catalog
from xhmodel_merak.model_build.resolver import _validate_source_constraints
from xhmodel_merak.model_inference.inputs import validate_inputs
from xhmodel_merak.model_integrations.depthsplat.build import normalize_export, validate_artifact
from xhmodel_merak.model_integrations.depthsplat.inference import adapter
from xhmodel_merak.model_integrations.depthsplat.plugin import MODEL_ID


def _recipe():
    return build_builtin_catalog().get_recipe(MODEL_ID)


def _artifact(tmp_path):
    graph = tmp_path / "visual_2v_256x256" / "model.onnx"
    graph.parent.mkdir()
    value = helper.make_tensor_value_info("images", TensorProto.FLOAT16, [1, 2, 3, 256, 256])
    output_names = ["means", "covariances", "harmonics", "opacities"]
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, 2, 3, 256, 256]) for name in output_names]
    nodes = [helper.make_node("Identity", ["images"], [name]) for name in output_names]
    model = helper.make_model(helper.make_graph(nodes, "ports", [value], outputs))
    onnx.save(model, graph)
    meta = {
        "schema_version": 1,
        "model_type": "XHDepthSplatModel",
        "target_device": "XH2a",
        "quant_type": "w8a8",
        "profile": "2v_256x256",
        "input_names": ["images"],
        "output_names": output_names,
        "input_specs": {"images": {"shape": [1, 2, 3, 256, 256], "dtype": "float16"}},
        "hmonnx_file": "visual_2v_256x256/model.onnx",
        "hmonnx_external_data": [],
    }
    meta["sample_input_specs"] = {"images": {"shape": [1, 2, 3, 256, 256], "dtype": "float32"}}
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    return meta


def test_recipe_is_discovered_with_onnx_identity_and_no_checkpoint_quantizer():
    catalog = build_builtin_catalog()
    recipe = catalog.get_recipe(MODEL_ID)
    assert recipe.family == "depthsplat"
    assert recipe.components == ("visual",)
    assert recipe.quant_methods == ()
    assert recipe.deployment_precision == {"visual": "w8a8"}
    assert recipe.default_precision.weight == "w16"
    assert catalog.get_adapter("build", recipe.build_adapter).target.endswith("depthsplat.build")
    assert catalog.get_adapter("inference", recipe.inference_adapter).target.endswith("depthsplat.inference:adapter")
    assert not any("gptq" in dep or "auto-round" in dep for dep in recipe.dependencies.common)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"model_type": "DepthSplatONNX", "profile": "4v_512x960"},
        {"model_type": "depthsplat", "profile": "2v_256x256"},
    ],
)
def test_source_constraints_reject_wrong_bundle(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ModelError):
        _validate_source_constraints(_recipe(), tmp_path)


def test_source_constraints_accept_prepared_bundle_identity(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "DepthSplatONNX", "profile": "2v_256x256"}))
    _validate_source_constraints(_recipe(), tmp_path)


@pytest.mark.parametrize("inputs", [{}, {"sample": ""}, {"sample": 1}, {"sample": "sample.npz", "prompt": "ignored"}])
def test_inference_rejects_invalid_parameters(inputs):
    with pytest.raises(ModelError):
        validate_inputs(_recipe(), inputs)


def test_artifact_requires_files_and_exact_metadata(tmp_path):
    meta = _artifact(tmp_path)
    plan = SimpleNamespace(model_id=MODEL_ID, components=("visual",))
    result = validate_artifact(tmp_path, plan)
    assert result["export_entries"]["visual"] == {"path": meta["hmonnx_file"], "type": "hmonnx"}
    assert result["requires_source"] is False
    (tmp_path / meta["hmonnx_file"]).unlink()
    with pytest.raises(ModelError):
        validate_artifact(tmp_path, plan)


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile", "4v_512x960"),
        ("quant_type", "w16a16"),
        ("hmonnx_file", "../escape.onnx"),
        ("input_names", ["images", "images"]),
        ("hmonnx_external_data", ["missing.weights"]),
    ],
)
def test_artifact_rejects_contract_mismatches(tmp_path, field, value):
    meta = _artifact(tmp_path)
    meta[field] = value
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ModelError):
        validate_artifact(tmp_path, SimpleNamespace(model_id=MODEL_ID, components=("visual",)))


def test_inference_requires_manifest_inventory(tmp_path):
    meta = _artifact(tmp_path)
    meta_file = tmp_path / "runtime_meta_info.json"
    manifest = {"artifact": {"files": [{"path": meta_file.name}]}}
    with pytest.raises(ModelError, match="inventory"):
        adapter.validate_artifact(recipe=_recipe(), root=tmp_path, manifest=manifest, meta_file=meta_file)
    manifest["artifact"]["files"].append({"path": meta["hmonnx_file"]})
    adapter.validate_artifact(recipe=_recipe(), root=tmp_path, manifest=manifest, meta_file=meta_file)


def test_export_rejects_workdir_outside_stage(tmp_path):
    plan = SimpleNamespace(model_id=MODEL_ID)
    assert normalize_export(tmp_path, SimpleNamespace(work_dir=str(tmp_path)), plan) is None
    with pytest.raises(ModelError):
        normalize_export(tmp_path, SimpleNamespace(work_dir=str(tmp_path.parent)), plan)


def test_artifact_rejects_graph_metadata_port_mismatch(tmp_path):
    meta = _artifact(tmp_path)
    meta["output_names"] = ["wrong_output"]
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ModelError, match="ports"):
        validate_artifact(tmp_path, SimpleNamespace(model_id=MODEL_ID, components=("visual",)))


@pytest.mark.parametrize("spec", [{"shape": [1], "dtype": "float32"}, {"shape": [1, 2, 3, 256, 256], "dtype": "int32"}])
def test_artifact_rejects_sample_contract_mismatch(tmp_path, spec):
    meta = _artifact(tmp_path)
    meta["sample_input_specs"] = {"images": spec}
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ModelError, match="sample"):
        validate_artifact(tmp_path, SimpleNamespace(model_id=MODEL_ID, components=("visual",)))


@pytest.mark.parametrize(
    "options",
    [
        {"quant_method": "gptq"},
        {"precision": {"weight": "w4", "activation": "a16"}},
        {"components": ["decode"]},
        {"profile": "4v_512x960"},
    ],
)
def test_build_plan_rejects_unsupported_requests(tmp_path, options):
    from xhmodel_merak.model_build.resolver import resolve_build_plan

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "DepthSplatONNX", "profile": "2v_256x256"}))
    with pytest.raises(ModelError):
        resolve_build_plan(model_id=MODEL_ID, model_dir=str(tmp_path), device="cpu", **options)


def test_build_plan_resolves_workflow_and_deployment_precision(tmp_path):
    from xhmodel_merak.model_build.resolver import resolve_build_plan

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "DepthSplatONNX", "profile": "2v_256x256"}))
    plan = resolve_build_plan(model_id=MODEL_ID, model_dir=str(tmp_path), device="cpu")
    assert plan.quant_method is None
    assert plan.profile == "2v_256x256"
    assert plan.deployment_precision == {"visual": "w8a8"}
    assert plan.build_adapter == "depthsplat.build"
    assert plan.workflow_config.endswith("depthsplat_xh2a_w8a8.yaml")


def test_artifact_external_weights_must_match_graph_references(tmp_path):
    import numpy as np

    meta = _artifact(tmp_path)
    graph_path = tmp_path / meta["hmonnx_file"]
    model = onnx.load(graph_path)
    model.graph.initializer.append(onnx.numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), "weights"))
    onnx.save_model(
        model,
        graph_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    plan = SimpleNamespace(model_id=MODEL_ID, components=("visual",))
    with pytest.raises(ModelError, match="references"):
        validate_artifact(tmp_path, plan)
    meta["hmonnx_external_data"] = ["visual_2v_256x256/weights.data"]
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    assert validate_artifact(tmp_path, plan)["loadable"] is True


def test_workflow_discovery_registers_model_in_fresh_process(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    config = root / _recipe().variants[0].workflow_config
    script = """
import sys
from xhmodel_merak.xh_other_model.builder import get_model_class
from xhmodel_merak.workflows import AutoWorkflow
model_class = get_model_class({"type": "XHDepthSplatModel"})
assert model_class.__name__ == "XHDepthSplatModel"
workflow = AutoWorkflow.from_config(model_dir=sys.argv[1], config_path=sys.argv[2])
assert type(workflow).__name__ == "DepthSplatWorkflow"
result = workflow.quant(output_dir=sys.argv[1], device="cpu")
assert result.skipped and result.raw_model_dir == sys.argv[1]
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(config)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_artifact_rejects_wrong_profile_image_shape_even_when_graph_matches(tmp_path):
    meta = _artifact(tmp_path)
    graph_path = tmp_path / meta["hmonnx_file"]
    model = onnx.load(graph_path)
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 4
    onnx.save(model, graph_path)
    meta["input_specs"]["images"]["shape"][1] = 4
    meta["sample_input_specs"]["images"]["shape"][1] = 4
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ModelError, match="images"):
        validate_artifact(tmp_path, SimpleNamespace(model_id=MODEL_ID, components=("visual",)))


def test_artifact_rejects_non_gaussian_outputs_even_when_graph_matches(tmp_path):
    meta = _artifact(tmp_path)
    graph_path = tmp_path / meta["hmonnx_file"]
    model = onnx.load(graph_path)
    model.graph.output[0].name = "unrelated"
    model.graph.node[0].output[0] = "unrelated"
    onnx.save(model, graph_path)
    meta["output_names"][0] = "unrelated"
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ModelError, match="Gaussian"):
        validate_artifact(tmp_path, SimpleNamespace(model_id=MODEL_ID, components=("visual",)))


def _host_artifact(tmp_path):
    meta = _artifact(tmp_path)
    meta.update(
        schema_version=2,
        output_names=["depths", "raw_gaussians"],
        gaussian_output_names=["means", "covariances", "harmonics", "opacities"],
        postprocess={
            "type": "depthsplat_gaussian_fp32_v1",
            "sh_degree": 2,
            "gaussian_scale_min": 1e-10,
            "gaussian_scale_max": 3.0,
            "num_surfaces": 1,
        },
    )
    shapes = {"images": [1, 2, 3, 256, 256], "intrinsics": [1, 2, 3, 3], "extrinsics": [1, 2, 4, 4]}
    shapes.update({f"camera_{3+i}_sh{i}": [1, 2, 1, 1, 1, 1, 2 * i + 1, 2 * i + 1] for i in range(3)})
    meta["host_input_specs"] = {name: {"shape": shape, "dtype": "float32"} for name, shape in shapes.items()}
    import copy

    meta["sample_input_specs"] = copy.deepcopy(meta["host_input_specs"])
    graph_path = tmp_path / meta["hmonnx_file"]
    model = onnx.load(graph_path)
    del model.graph.node[:]
    del model.graph.output[:]
    for name, shape in (("depths", [1, 2, 65536, 1, 1]), ("raw_gaussians", [2, 37, 256, 256])):
        model.graph.initializer.append(helper.make_tensor(name + "_shape", TensorProto.INT64, [len(shape)], shape))
        model.graph.node.append(helper.make_node("ConstantOfShape", [name + "_shape"], [name]))
        model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
    onnx.checker.check_model(model)
    onnx.save(model, graph_path)
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    return meta


def test_schema2_accepts_host_inputs_outside_quantized_graph(tmp_path):
    meta = _host_artifact(tmp_path)
    assert set(meta["sample_input_specs"]) > set(meta["input_specs"])
    assert validate_artifact(tmp_path, SimpleNamespace(model_id=MODEL_ID, components=("visual",)))["loadable"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 4),
        ("schema_version", True),
        ("postprocess", {"type": "unknown"}),
        ("gaussian_output_names", ["depths", "raw_gaussians"]),
        ("host_input_specs", {}),
        ("output_names", ["means", "covariances", "harmonics", "opacities"]),
    ],
)
def test_schema2_rejects_unknown_or_mismatched_contract(tmp_path, field, value):
    meta = _host_artifact(tmp_path)
    meta[field] = value
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ModelError):
        validate_artifact(tmp_path, SimpleNamespace(model_id=MODEL_ID, components=("visual",)))


@pytest.mark.parametrize("field", ["postprocess", "host_input_specs", "gaussian_output_names"])
def test_legacy_schema_rejects_host_metadata_injection(tmp_path, field):
    meta = _artifact(tmp_path)
    meta[field] = {}
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ModelError, match="schema 1"):
        validate_artifact(tmp_path, SimpleNamespace(model_id=MODEL_ID, components=("visual",)))


@pytest.mark.parametrize(
    "mutation",
    ["host_dtype", "host_shape", "missing_sample", "extra_sample", "sample_dtype", "output_shape", "output_dtype"],
)
def test_schema2_rejects_malformed_camera_samples_and_raw_outputs(tmp_path, mutation):
    meta = _host_artifact(tmp_path)
    if mutation == "host_dtype":
        meta["host_input_specs"]["intrinsics"]["dtype"] = "float16"
    elif mutation == "host_shape":
        meta["host_input_specs"]["camera_5_sh2"]["shape"][-1] = 3
    elif mutation == "missing_sample":
        del meta["sample_input_specs"]["extrinsics"]
    elif mutation == "extra_sample":
        meta["sample_input_specs"]["unrelated"] = {"shape": [1], "dtype": "float32"}
    elif mutation == "sample_dtype":
        meta["sample_input_specs"]["images"]["dtype"] = "float16"
    else:
        graph_path = tmp_path / meta["hmonnx_file"]
        model = onnx.load(graph_path)
        if mutation == "output_shape":
            model.graph.output[1].type.tensor_type.shape.dim[1].dim_value = 38
        else:
            model.graph.output[0].type.tensor_type.elem_type = TensorProto.INT32
        onnx.save(model, graph_path)
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ModelError):
        validate_artifact(tmp_path, SimpleNamespace(model_id=MODEL_ID, components=("visual",)))


def _grid_artifact(tmp_path):
    import copy

    meta = _host_artifact(tmp_path)
    meta.update(
        schema_version=3,
        preprocess={
            "type": "depthsplat_matching_grid_fp32_v1",
            "num_depth_candidates": 128,
            "feature_height": 64,
            "feature_width": 64,
        },
    )
    meta["host_input_specs"].update({name: {"shape": [1, 2], "dtype": "float32"} for name in ("near", "far")})
    meta["sample_input_specs"] = copy.deepcopy(meta["host_input_specs"])
    meta["input_names"] = ["images", "near", "far", "matching_grid"]
    meta["input_specs"].update({name: {"shape": [1, 2], "dtype": "float16"} for name in ("near", "far")})
    meta["input_specs"]["matching_grid"] = {"shape": [2, 8192, 64, 2], "dtype": "float16"}
    graph_path = tmp_path / meta["hmonnx_file"]
    model = onnx.load(graph_path)
    for name in ("near", "far"):
        model.graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, 2]))
    model.graph.input.append(helper.make_tensor_value_info("matching_grid", TensorProto.FLOAT16, [2, 8192, 64, 2]))
    onnx.save(model, graph_path)
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    return meta


def test_schema3_generated_grid_is_not_a_sample_port(tmp_path):
    meta = _grid_artifact(tmp_path)
    assert "matching_grid" not in meta["sample_input_specs"]
    assert validate_artifact(tmp_path, SimpleNamespace(model_id=MODEL_ID, components=("visual",)))["loadable"]


@pytest.mark.parametrize("version", [1, 2])
def test_old_schema_cannot_inject_matching_preprocess(tmp_path, version):
    meta = _artifact(tmp_path) if version == 1 else _host_artifact(tmp_path)
    meta["preprocess"] = {"type": "depthsplat_matching_grid_fp32_v1"}
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ModelError, match="preprocess"):
        validate_artifact(tmp_path, SimpleNamespace(model_id=MODEL_ID, components=("visual",)))


@pytest.mark.parametrize(
    "mutation",
    [
        "sample_grid",
        "grid_shape",
        "grid_dtype",
        "missing_grid",
        "missing_far",
        "host_far_dtype",
        "unknown_type",
        "wrong_candidates",
    ],
)
def test_schema3_rejects_invalid_generated_grid_contract(tmp_path, mutation):
    meta = _grid_artifact(tmp_path)
    if mutation == "sample_grid":
        meta["sample_input_specs"]["matching_grid"] = meta["input_specs"]["matching_grid"]
    elif mutation == "grid_shape":
        meta["input_specs"]["matching_grid"]["shape"][1] = 4096
    elif mutation == "grid_dtype":
        meta["input_specs"]["matching_grid"]["dtype"] = "int32"
    elif mutation == "missing_grid":
        del meta["input_specs"]["matching_grid"]
        meta["input_names"].remove("matching_grid")
    elif mutation == "missing_far":
        del meta["host_input_specs"]["far"]
    elif mutation == "host_far_dtype":
        meta["host_input_specs"]["far"]["dtype"] = "float16"
    elif mutation == "unknown_type":
        meta["preprocess"]["type"] = "untrusted"
    else:
        meta["preprocess"]["num_depth_candidates"] = 64
    (tmp_path / "runtime_meta_info.json").write_text(json.dumps(meta))
    with pytest.raises(ModelError):
        validate_artifact(tmp_path, SimpleNamespace(model_id=MODEL_ID, components=("visual",)))
