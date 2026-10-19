"""Prepared-source input contracts for the real multi-view workflow."""

import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from xhmodel_merak.xh_other_model.models.depthsplat.assets import load_bundle, load_sample, onnx_contract


def _source(tmp_path):
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 2, 3, 256, 256])
    outputs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 2, 3, 256, 256])
        for n in ["means", "covariances", "harmonics", "opacities"]
    ]
    nodes = [helper.make_node("Identity", ["images"], [v.name]) for v in outputs]
    model = helper.make_model(
        helper.make_graph(nodes, "contract", [images], outputs), opset_imports=[helper.make_opsetid("", 17)]
    )
    onnx.save(model, tmp_path / "encoder.onnx")
    for name in ["calibration", "evaluation"]:
        np.savez(tmp_path / f"{name}.npz", images=np.zeros((1, 2, 3, 256, 256), np.float32))
    config = {
        "model_type": "DepthSplatONNX",
        "profile": "2v_256x256",
        "onnx_file": "encoder.onnx",
        "calibration_samples": ["calibration.npz"],
        "evaluation_samples": ["evaluation.npz"],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return config


def test_source_contract_preserves_real_port_order_and_dtype(tmp_path):
    _source(tmp_path)
    _, model, calibration, evaluation = load_bundle(tmp_path)
    specs, names = onnx_contract(model)
    assert names == ["means", "covariances", "harmonics", "opacities"]
    assert load_sample(calibration[0], specs)["images"].dtype == np.float32
    assert calibration != evaluation


def test_calibration_and_evaluation_must_be_disjoint(tmp_path):
    config = _source(tmp_path)
    config["evaluation_samples"] = config["calibration_samples"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="disjoint"):
        load_bundle(tmp_path)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((1, 2, 3, 128, 128), np.float32),
        np.zeros((1, 2, 3, 256, 256), np.float16),
        np.full((1, 2, 3, 256, 256), np.nan, np.float32),
    ],
)
def test_bad_sample_cannot_be_silently_coerced(tmp_path, value):
    _source(tmp_path)
    specs, _ = onnx_contract(tmp_path / "encoder.onnx")
    np.savez(tmp_path / "bad.npz", images=value)
    with pytest.raises(ValueError):
        load_sample(tmp_path / "bad.npz", specs)


def test_source_paths_cannot_escape_bundle(tmp_path):
    config = _source(tmp_path)
    config["onnx_file"] = "../outside.onnx"
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_bundle(tmp_path)


def test_wrong_output_contract_rejected(tmp_path):
    _source(tmp_path)
    p = tmp_path / "encoder.onnx"
    m = onnx.load(p)
    m.graph.output[0].name = "depth"
    onnx.save(m, p)
    with pytest.raises(ValueError, match="Gaussian outputs"):
        onnx_contract(p)


def _host_config():
    shapes = {
        "images": [1, 2, 3, 256, 256],
        "intrinsics": [1, 2, 3, 3],
        "extrinsics": [1, 2, 4, 4],
        **{f"camera_{3+i}_sh{i}": [1, 2, 1, 1, 1, 1, 2 * i + 1, 2 * i + 1] for i in range(3)},
    }
    return {
        "postprocess": {
            "type": "depthsplat_gaussian_fp32_v1",
            "sh_degree": 2,
            "gaussian_scale_min": 1e-10,
            "gaussian_scale_max": 3.0,
            "num_surfaces": 1,
        },
        "host_input_specs": {name: {"shape": shape, "dtype": "float32"} for name, shape in shapes.items()},
    }


def test_sample_contract_preserves_legacy_graph_inputs():
    from xhmodel_merak.xh_other_model.models.depthsplat.assets import sample_contract

    specs = {"images": {"shape": [1, 2, 3, 256, 256], "dtype": "float32"}}
    assert sample_contract({}, specs) == specs


def test_host_sample_contract_requires_exact_union_without_mutating_graph(tmp_path):
    import copy
    from xhmodel_merak.xh_other_model.models.depthsplat.assets import sample_contract

    config = _host_config()
    graph_specs = {"images": config["host_input_specs"]["images"], "near": {"shape": [1, 2], "dtype": "float32"}}
    before = copy.deepcopy(graph_specs)
    specs = sample_contract(config, graph_specs)
    assert specs == {**graph_specs, **config["host_input_specs"]}
    assert graph_specs == before
    values = {name: np.zeros(spec["shape"], np.dtype(spec["dtype"])) for name, spec in specs.items()}
    np.savez(tmp_path / "sample.npz", **values)
    assert set(load_sample(tmp_path / "sample.npz", specs)) == set(specs)
    del values["camera_5_sh2"]
    np.savez(tmp_path / "missing.npz", **values)
    with pytest.raises(ValueError, match="ports"):
        load_sample(tmp_path / "missing.npz", specs)


@pytest.mark.parametrize(
    "mutation",
    ["unknown_type", "missing_postprocess", "scale", "missing_host", "host_dtype", "host_shape", "graph_conflict"],
)
def test_host_sample_contract_rejects_inconsistent_source_metadata(mutation):
    from xhmodel_merak.xh_other_model.models.depthsplat.assets import sample_contract

    config = _host_config()
    graph_specs = {"images": {"shape": [1, 2, 3, 256, 256], "dtype": "float32"}}
    if mutation == "unknown_type":
        config["postprocess"]["type"] = "unknown"
    elif mutation == "missing_postprocess":
        del config["postprocess"]
    elif mutation == "scale":
        config["postprocess"]["gaussian_scale_max"] = 10.0
    elif mutation == "missing_host":
        del config["host_input_specs"]["extrinsics"]
    elif mutation == "host_dtype":
        config["host_input_specs"]["intrinsics"]["dtype"] = "float16"
    elif mutation == "host_shape":
        config["host_input_specs"]["camera_4_sh1"]["shape"][-1] = 5
    else:
        graph_specs["images"]["dtype"] = "float16"
    with pytest.raises(ValueError):
        sample_contract(config, graph_specs)


def _grid_config_and_specs():
    config = _host_config()
    config["preprocess"] = {
        "type": "depthsplat_matching_grid_fp32_v1",
        "num_depth_candidates": 128,
        "feature_height": 64,
        "feature_width": 64,
    }
    config["host_input_specs"].update({name: {"shape": [1, 2], "dtype": "float32"} for name in ("near", "far")})
    specs = {name: dict(config["host_input_specs"][name]) for name in ("images", "near", "far")}
    specs["matching_grid"] = {"shape": [2, 8192, 64, 2], "dtype": "float32"}
    return config, specs


def test_generated_matching_grid_cannot_be_injected_in_raw_sample(tmp_path):
    from xhmodel_merak.xh_other_model.models.depthsplat.assets import sample_contract

    config, graph_specs = _grid_config_and_specs()
    specs = sample_contract(config, graph_specs)
    assert set(specs) == set(config["host_input_specs"])
    assert "matching_grid" not in specs
    values = {name: np.zeros(spec["shape"], np.dtype(spec["dtype"])) for name, spec in specs.items()}
    np.savez(tmp_path / "raw.npz", **values)
    assert set(load_sample(tmp_path / "raw.npz", specs)) == set(specs)
    values["matching_grid"] = np.zeros((2, 8192, 64, 2), np.float32)
    np.savez(tmp_path / "injected.npz", **values)
    with pytest.raises(ValueError, match="ports"):
        load_sample(tmp_path / "injected.npz", specs)


@pytest.mark.parametrize(
    "mutation", ["grid_shape", "grid_dtype", "preprocess", "near_shape", "far_dtype", "missing_far", "extra_graph"]
)
def test_matching_grid_source_contract_rejects_malformed_metadata(mutation):
    from xhmodel_merak.xh_other_model.models.depthsplat.assets import sample_contract

    config, specs = _grid_config_and_specs()
    if mutation == "grid_shape":
        specs["matching_grid"]["shape"] = [2, 4096, 64, 2]
    elif mutation == "grid_dtype":
        specs["matching_grid"]["dtype"] = "float16"
    elif mutation == "preprocess":
        config["preprocess"]["type"] = "unknown"
    elif mutation == "near_shape":
        config["host_input_specs"]["near"]["shape"] = [2]
    elif mutation == "far_dtype":
        config["host_input_specs"]["far"]["dtype"] = "float16"
    elif mutation == "missing_far":
        del config["host_input_specs"]["far"]
    else:
        specs["camera_0_inverse"] = {"shape": [2, 1, 4, 4], "dtype": "float32"}
    with pytest.raises(ValueError):
        sample_contract(config, specs)
