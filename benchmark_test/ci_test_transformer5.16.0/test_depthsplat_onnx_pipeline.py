"""Small real ONNX/camera contracts for the standalone DepthSplat pipeline."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "xhmodel_merak/xh_other_model/models/depthsplat"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"depthsplat_ci_{name}", PACKAGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = _load("onnx_pipeline")
source = _load("source")


def _neighbor_graph(path, count=48, axis=1, base_shape=(1, 2, 1, 1), views=6):
    """Retain the broadcast spatial axes that the SDK incorrectly flattened."""
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [1, views, 4, 3])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, list(base_shape))
    target_shape = np.array([1, 2, 4, 3], dtype=np.int64)
    nodes = [helper.make_node("Expand", ["indices", "target_shape"], ["expanded"], name="broadcast")]
    outputs = []
    for i in range(count):
        name = f"selected_{i}"
        nodes.append(helper.make_node("GatherElements", ["data", "expanded"], [name], name=f"neighbor_{i}", axis=axis))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2, 4, 3]))
    graph = helper.make_graph(
        nodes, "neighbors", [data, indices], outputs, [numpy_helper.from_array(target_shape, "target_shape")]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, path)


def test_neighbor_normalization_preserves_spatial_values_and_dynamic_indices(tmp_path):
    original, normalized = tmp_path / "source.onnx", tmp_path / "normalized.onnx"
    _neighbor_graph(original)
    samples = []
    data = np.arange(72, dtype=np.float32).reshape(1, 6, 4, 3)
    for i, choices in enumerate(([1, 4], [5, 0])):
        sample = tmp_path / f"sample_{i}.npz"
        np.savez(sample, data=data, indices=np.array(choices, dtype=np.int64).reshape(1, 2, 1, 1))
        samples.append(sample)
    before = original.read_bytes()
    report = pipeline.normalize_neighbors(original, normalized, samples)
    assert original.read_bytes() == before
    assert len(report["replacements"]) == 48
    assert len(report["alignment"]) == 2
    assert all(m == {"max_abs": 0.0, "rel_l2": 0.0} for row in report["alignment"] for m in row["metrics"].values())
    assert json.loads(normalized.with_suffix(".normalization.json").read_text()) == report
    graph = onnx.load(normalized)
    assert sum(n.op_type == "GatherElements" for n in graph.graph.node) == 0
    assert sum(n.op_type == "Gather" for n in graph.graph.node) == 48
    import onnxruntime as ort

    session = ort.InferenceSession(str(normalized), providers=["CPUExecutionProvider"])
    for choices in ([1, 4], [5, 0]):
        result = session.run(None, {"data": data, "indices": np.array(choices, dtype=np.int64).reshape(1, 2, 1, 1)})
        for value in result:
            np.testing.assert_array_equal(value, data[:, choices])


@pytest.mark.parametrize(
    "change,match",
    [
        ({"count": 47}, "exactly 48"),
        ({"axis": 0}, "Unsupported GatherElements"),
        ({"views": 5}, "Unsupported shapes"),
        ({"base_shape": (1, 2, 2, 1)}, "Unsupported shapes"),
    ],
)
def test_neighbor_normalization_rejects_unverified_patterns(tmp_path, change, match):
    original, output = tmp_path / "source.onnx", tmp_path / "output.onnx"
    _neighbor_graph(original, **change)
    with pytest.raises(ValueError, match=match):
        pipeline.normalize_neighbors(original, output)
    assert not output.exists()


def test_neighbor_normalization_never_overwrites_source(tmp_path):
    path = tmp_path / "source.onnx"
    _neighbor_graph(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="must differ"):
        pipeline.normalize_neighbors(path, path)
    assert path.read_bytes() == before


def _geometry_graph(path, count=8, bad_precision=False, duplicate=False):
    nodes = []
    for index in range(count):
        node = helper.make_node(
            "MatMul",
            ["a", "b"],
            [f"y{index}"],
            name=f"geometry_{index}",
            hmfp_act_man_bit=8 if bad_precision and index == 0 else 16,
            hmfp_weight_man_bit=16,
        )
        module = pipeline.GEOMETRY_MODULES[0 if duplicate else index]
        node.metadata_props.add(key="namespace", value=f"root/{module}:MatMul/output")
        nodes.append(node)
    # An unrelated W8A8 MatMul must not be mistaken for a geometry node.
    nodes.append(helper.make_node("MatMul", ["a", "b"], ["other"], hmfp_act_man_bit=8, hmfp_weight_man_bit=8))
    onnx.save(helper.make_model(helper.make_graph(nodes, "precision", [], [])), path)


def test_geometry_validator_checks_all_eight_mantissas_and_ignores_other_matmuls(tmp_path):
    path = tmp_path / "geometry.onnx"
    _geometry_graph(path)
    rows = pipeline.validate_geometry_hmonnx(path)
    assert len(rows) == 8
    assert {row["module"] for row in rows} == set(pipeline.GEOMETRY_MODULES)
    assert all(row["lhs_man_bit"] == row["rhs_man_bit"] == 16 for row in rows)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"count": 7}, "eight geometry"),
        ({"duplicate": True}, "eight geometry"),
        ({"bad_precision": True}, "precision was not applied"),
    ],
)
def test_geometry_validator_rejects_missing_duplicate_or_w8a8_geometry(tmp_path, change, match):
    path = tmp_path / "geometry.onnx"
    _geometry_graph(path, **change)
    with pytest.raises(ValueError, match=match):
        pipeline.validate_geometry_hmonnx(path)


def _camera_sample():
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (1, 6, 1, 1))
    extrinsics[0, :, 0, 3] = [0, 1, 3, 7, 15, 31]
    return dict(
        image=np.zeros((1, 6, 3, 512, 960), np.float32),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (1, 6, 1, 1)),
        extrinsics=extrinsics,
        near=np.full((1, 6), 0.5, np.float32),
        far=np.full((1, 6), 200, np.float32),
    )


def test_cpu_host_cameras_match_analytic_translations_and_feature_intrinsics(tmp_path):
    sample = _camera_sample()
    path = tmp_path / "context.npz"
    np.savez(path, **sample)
    before = path.read_bytes()
    result = pipeline.prepare_inputs(path)
    centers = sample["extrinsics"][0, :, 0, 3]
    neighbors = np.argsort(np.abs(centers[:, None] - centers[None, :]), axis=-1)
    np.testing.assert_array_equal(result["nn_matrix"], neighbors[None])
    expected = np.tile(np.eye(4, dtype=np.float32), (6, 2, 1, 1))
    expected[:, :, 0, 3] = -centers[neighbors[:, 1:3]]
    for stage, scale in enumerate((8, 4)):
        np.testing.assert_allclose(result[f"camera_{2 * stage}_inverse"], expected, rtol=1e-6, atol=1e-6)
        k_inverse = np.diag([scale / 960, scale / 512, 1]).astype(np.float32)
        np.testing.assert_allclose(result[f"camera_{2 * stage + 1}_inverse"], np.tile(k_inverse, (12, 1, 1)), rtol=1e-6)
    np.testing.assert_array_equal(result["images"], sample["image"])
    assert all(value.dtype == np.float32 for name, value in result.items() if name != "nn_matrix")
    assert path.read_bytes() == before
    # Cameras are recomputed for this context rather than copied from a capture.
    sample["extrinsics"][0, 5, 0, 3] = 2
    np.savez(path, **sample)
    changed = pipeline.prepare_inputs(path)
    assert not np.array_equal(changed["nn_matrix"], result["nn_matrix"])
    assert not np.array_equal(changed["camera_0_inverse"], result["camera_0_inverse"])


@pytest.mark.parametrize("invalid", ["dtype", "far"])
def test_host_camera_input_contract_rejects_invalid_context(tmp_path, invalid):
    sample = _camera_sample()
    if invalid == "dtype":
        sample["intrinsics"] = sample["intrinsics"].astype(np.float64)
    else:
        sample["far"][:] = sample["near"]
    path = tmp_path / "invalid.npz"
    np.savez(path, **sample)
    with pytest.raises(ValueError, match="finite FP32|near < far"):
        pipeline.prepare_inputs(path)


def _matching_graph(consumers=1):
    shape = [2, 8192, 64, 2]
    cut = "/encoder/depth_predictor/Reshape_39_output_0"
    nodes = [helper.make_node("Identity", ["camera"], [cut])]
    nodes.extend(helper.make_node("Identity", [cut], [f"depth_{i}"]) for i in range(consumers))
    nodes.append(helper.make_node("Identity", ["images"], ["raw_gaussians"]))
    inputs = [
        helper.make_tensor_value_info("camera", TensorProto.FLOAT, shape),
        helper.make_tensor_value_info("images", TensorProto.FLOAT, [1]),
    ]
    outputs = [helper.make_tensor_value_info(f"depth_{i}", TensorProto.FLOAT, shape) for i in range(consumers)] + [
        helper.make_tensor_value_info("raw_gaussians", TensorProto.FLOAT, [1])
    ]
    graph = helper.make_graph(
        nodes, "matching", inputs, outputs, [numpy_helper.from_array(np.ones(1, np.float32), "unused_weight")]
    )
    graph.value_info.append(helper.make_tensor_value_info(cut, TensorProto.FLOAT, shape))
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


def test_matching_grid_cut_exposes_host_port_and_prunes_camera_subgraph():
    model = _matching_graph()
    source._cut_matching_grid(model)
    onnx.checker.check_model(model)
    assert [value.name for value in model.graph.input] == ["images", "matching_grid"]
    assert [dim.dim_value for dim in model.graph.input[-1].type.tensor_type.shape.dim] == [2, 8192, 64, 2]
    assert len(model.graph.node) == 2
    assert model.graph.node[0].input[0] == "matching_grid"
    assert not model.graph.initializer
    assert not model.graph.value_info


@pytest.mark.parametrize("consumers", [0, 2])
def test_matching_grid_cut_rejects_changed_upstream_boundary(consumers):
    model = _matching_graph(consumers)
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="boundary changed"):
        source._cut_matching_grid(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("fail", [False, True])
def test_export_neighbor_override_is_restored_after_normal_and_exception_paths(fail):
    original = torch.argsort
    supplied = torch.tensor([5, 2, 0])
    try:
        with source._neighbors(supplied):
            assert torch.argsort(torch.tensor([3, 1, 2])) is supplied
            if fail:
                raise RuntimeError("export interrupted")
    except RuntimeError:
        assert fail
    assert torch.argsort is original
    with source._neighbors(None):
        torch.testing.assert_close(torch.argsort(torch.tensor([3, 1, 2])), torch.tensor([1, 2, 0]))
    assert torch.argsort is original


def test_error_metrics_accumulate_across_chunks_and_enforce_output_shape():
    reference = np.ones(1000003, dtype=np.float32)
    actual = reference.copy()
    actual[0], actual[-1] = 3, 4
    result = pipeline.metrics(reference, actual)
    assert result["max_abs"] == 3
    assert result["rel_l2"] == pytest.approx(np.sqrt(13 / len(reference)))
    with pytest.raises(ValueError, match="Output shapes differ"):
        pipeline.metrics(reference.reshape(1, -1), actual)
    assert pipeline.metrics(np.zeros(1), np.zeros(1)) == {"max_abs": 0, "rel_l2": 0}
