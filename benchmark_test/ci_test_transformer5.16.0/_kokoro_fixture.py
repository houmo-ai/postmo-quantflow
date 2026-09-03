"""Small release-shaped files for packaging tests, not Kokoro numerical models."""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from xhmodel_merak.xh_other_model.models.kokoro.assets import sha256
from xhmodel_merak.xh_other_model.models.kokoro.buckets import BUCKET_ROUTES
from xhmodel_merak.xh_other_model.models.kokoro.graph import GraphArtifact, artifact_metadata
from xhmodel_merak.xh_other_model.models.kokoro.quant_export import _hmonnx_artifact_metadata
from xhmodel_merak.xh_other_model.models.kokoro.release import component_path, release_prefix


ROLES = ("text_duration", "frame_acoustic", "phase_core", "generator_istft")
PREFIX = release_prefix(BUCKET_ROUTES, "20260909")


def write_graph(path: Path, role: str, capacity: int) -> GraphArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    phase = role == "phase_core"
    inputs = [{"name": "f0" if phase else "x", "dtype": "FLOAT", "shape": [1, 2 * capacity] if phase else [1]}]
    outputs = [{"name": "sine" if phase else "y", "dtype": "FLOAT", "shape": [1, 600 * capacity, 9] if phase else [1]}]
    # Phase's real export has its own numerical test. A small constant-shape
    # expansion here retains its I/O ABI without allocating a waveform tensor.
    if phase:
        nodes = [
            helper.make_node("ReduceMean", ["f0"], ["mean"], keepdims=0),
            helper.make_node("Expand", ["mean", "shape"], ["sine"]),
        ]
        initializers = [helper.make_tensor("shape", TensorProto.INT64, [3], [1, 600 * capacity, 9])]
    else:
        nodes = [helper.make_node("Add", ["x", "weight"], ["y"])]
        initializers = [
            helper.make_tensor("weight", TensorProto.FLOAT, [1], np.ones(1, np.float32).tobytes(), raw=True)
        ]
    graph = helper.make_graph(
        nodes,
        "packaging_fixture",
        [helper.make_tensor_value_info(v["name"], TensorProto.FLOAT, v["shape"]) for v in inputs],
        [helper.make_tensor_value_info(v["name"], TensorProto.FLOAT, v["shape"]) for v in outputs],
        initializers,
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save_model(
        proto,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=path.stem.removesuffix("_with_act") + "_external_data",
        size_threshold=0,
    )
    onnx.checker.check_model(str(path))
    counts = {node.op_type: sum(n.op_type == node.op_type for n in nodes) for node in nodes}
    return GraphArtifact(
        role,
        path,
        tuple(v["name"] for v in inputs),
        tuple(v["name"] for v in outputs),
        tuple(inputs),
        tuple(outputs),
        len(nodes),
        counts,
        sha256(path),
        {},
    )


def write_release(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "hf_config/config.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({"vocab": {"a": 17}}), encoding="utf-8")
    voice_path = root / "assets/voices/zf_001.npy"
    voice_path.parent.mkdir(parents=True)
    np.save(voice_path, np.ones((1, 256), dtype=np.float32), allow_pickle=False)
    components = {role: {"buckets": {}} for role in ROLES}
    for route in BUCKET_ROUTES:
        for role in ROLES:
            text = role == "text_duration"
            capacity = route.token_max_length if text else route.frame_max_length
            key = f"t{capacity:04d}" if text else f"f{capacity:04d}"
            reference = component_path(root, PREFIX, role, key, reference=role != "phase_core")
            artifact = write_graph(reference, role, capacity)
            entry = artifact_metadata(artifact, root)
            entry["token_max_length" if text else "frame_max_length"] = capacity
            if role != "phase_core":
                published = component_path(root, PREFIX, role, key, reference=False)
                write_graph(published, role, capacity)
                entry.update(_hmonnx_artifact_metadata(published, root))
            components[role]["buckets"][key] = entry
    meta = {
        "schema_version": 2,
        "graph_mode": "bucketed_cpu_phase",
        "export_complete": True,
        "hmonnx_converted": True,
        "release_prefix": PREFIX,
        "target_device": "XH2a",
        "quant_type": "w16a16_sefp",
        "sample_rate": 24000,
        "seed": 1234,
        "phase_boundary": {"execution": "host_onnxruntime_fp32", "input": "f0", "output": "sine"},
        "component_order": list(ROLES),
        "components": components,
        "bucket_presets": {"policy": "duration_driven_paired_t_f", "routes": [r.as_dict() for r in BUCKET_ROUTES]},
        "runtime_assets": {
            "config_file": "hf_config/config.json",
            "voice_pack": {
                "name": "zf_001",
                "file": "assets/voices/zf_001.npy",
                "shape": [1, 256],
                "dtype": "float32",
                "size_bytes": voice_path.stat().st_size,
            },
        },
    }
    (root / "export_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta
