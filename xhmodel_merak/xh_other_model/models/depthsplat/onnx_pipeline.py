# Copyright (c) 2024 Haofei Xu
# Copyright 2026 HOUMO AI (ONNX pipeline and CPU camera adapter)
# SPDX-License-Identifier: MIT
"""Explicit-path 6v ONNX pipeline; camera inverses run on the CPU in FP32.

The two learned matching grids remain inside this graph. Geometry a16a16 is
SEFP with 16-bit mantissas, not FP32 camera geometry or an all-W8A8 graph.
"""

import json
from pathlib import Path
import time

import numpy as np

GEOMETRY_NODES = {"/encoder/depth_predictor/MatMul" + (f"_{i}" if i else ""): "a16a16" for i in range(8)}
GEOMETRY_MODULES = ["_encoder_depth_predictor_mat_mul" + (f"_{i}" if i else "") for i in range(8)]


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def prepare_inputs(sample):
    """Prepare fixed B=1/V=6/512x960 camera inputs without encoder weights."""
    import torch

    with np.load(sample, allow_pickle=False) as archive:
        feed = {
            name: archive["image" if name == "images" and "images" not in archive else name].copy()
            for name in ("images", "intrinsics", "extrinsics", "near", "far")
        }
    shapes = {
        "images": (1, 6, 3, 512, 960),
        "intrinsics": (1, 6, 3, 3),
        "extrinsics": (1, 6, 4, 4),
        "near": (1, 6),
        "far": (1, 6),
    }
    for name, value in feed.items():
        if value.shape != shapes[name] or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"{name}: expected finite FP32 {shapes[name]}, got {value.shape}/{value.dtype}")
    if not ((feed["near"] > 0).all() and (feed["far"] > feed["near"]).all()):
        raise ValueError("Require 0 < near < far")
    extrinsics = torch.from_numpy(feed["extrinsics"])
    xyz = extrinsics[:, :, :3, -1]
    neighbors = torch.argsort(torch.cdist(xyz, xyz, p=2), dim=-1)
    feed["nn_matrix"] = neighbors.numpy()
    inverse_targets = torch.linalg.inv(extrinsics[0, neighbors[0, :, 1:3]]).numpy()
    pixel_k = torch.from_numpy(feed["intrinsics"]).clone()
    pixel_k[:, :, 0] *= 960
    pixel_k[:, :, 1] *= 512
    for stage, downsample in enumerate((8, 4)):
        k = pixel_k.clone()
        k[:, :, :2] /= downsample
        k = k.reshape(6, 3, 3).unsqueeze(1).repeat(1, 2, 1, 1).reshape(12, 3, 3)
        feed[f"camera_{stage * 2}_inverse"] = inverse_targets.copy()
        feed[f"camera_{stage * 2 + 1}_inverse"] = torch.linalg.inv(k).numpy()
    return feed


def metrics(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError(f"Output shapes differ: {reference.shape} != {actual.shape}")
    ref, out = reference.reshape(-1), actual.reshape(-1)
    squared = denominator = maximum = 0.0
    for index in range(0, len(ref), 1000000):
        a = ref[index : index + 1000000].astype(np.float64)
        b = out[index : index + 1000000].astype(np.float64)
        delta = a - b
        squared += np.dot(delta, delta)
        denominator += np.dot(a, a)
        maximum = max(maximum, float(np.max(np.abs(delta))))
    return {"max_abs": maximum, "rel_l2": float(np.sqrt(squared / max(denominator, 1e-30)))}


def normalize_neighbors(source, output, samples=()):
    """Replace 48 broadcast GatherElements sites; keep neighbor indices dynamic.

    Works only for this fixed batch-one, six-view export. Reject unexpected
    indexing patterns instead of applying an unverified general graph rewrite.
    """
    import onnx
    from onnx import helper, numpy_helper

    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Normalization output must differ from source")
    model = onnx.load(source)
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {
        v.name: [d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]
        for v in list(inferred.graph.value_info) + list(inferred.graph.input)
    }
    producers = {v: node for node in model.graph.node for v in node.output}
    replaced, nodes = [], []
    for node in model.graph.node:
        if node.op_type != "GatherElements":
            nodes.append(node)
            continue
        axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
        expand = producers.get(node.input[1])
        if axis != 1 or expand is None or expand.op_type != "Expand":
            raise ValueError(f"Unsupported GatherElements pattern: {node.name}")
        base = expand.input[0]
        shape, data_shape = shapes.get(base, []), shapes.get(node.input[0], [])
        if (
            shape[:2] != [1, 2]
            or not all(d == 1 for d in shape[2:])
            or len(data_shape) < 2
            or data_shape[1] != 6
            or data_shape[0] not in (1, None)
        ):
            raise ValueError(f"Unsupported shapes at {node.name}: {shape}, {data_shape}")
        stem = node.name + "_dynamic_view_indices"
        flat, target = stem + "_flat", stem + "_shape"
        model.graph.initializer.append(numpy_helper.from_array(np.array([-1], np.int64), target))
        nodes.append(helper.make_node("Reshape", [base, target], [flat], name=stem))
        nodes.append(
            helper.make_node("Gather", [node.input[0], flat], list(node.output), name=node.name + "_axis1", axis=1)
        )
        replaced.append({"node": node.name, "index_base": base, "base_shape": shape, "data_shape": data_shape})
    if len(replaced) != 48:
        raise ValueError(f"Expected exactly 48 neighbor gathers, found {len(replaced)}")
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    del model, inferred
    report = {"scope": "fixed B=1, V=6; dynamic neighbor indices", "replacements": replaced, "alignment": []}
    if samples:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        options.inter_op_num_threads = 1
        # Separate sessions avoid retaining both large graphs while executing.
        references = []
        original = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
        for sample in samples:
            with np.load(sample, allow_pickle=False) as archive:
                feed = {v.name: archive[v.name] for v in original.get_inputs()}
            references.append(original.run(None, feed))
        del original
        normalized = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
        for sample, reference in zip(samples, references):
            with np.load(sample, allow_pickle=False) as archive:
                feed = {v.name: archive[v.name] for v in normalized.get_inputs()}
            actual = normalized.run(None, feed)
            values = {v.name: metrics(a, b) for v, a, b in zip(normalized.get_outputs(), reference, actual)}
            if any(not np.isfinite(m["rel_l2"]) or m["rel_l2"] >= 1e-5 for m in values.values()):
                raise ValueError(f"Gather normalization changed outputs: {values}")
            report["alignment"].append({"sample": str(sample), "metrics": values})
    write_report(output.with_suffix(".normalization.json"), report)
    return report


def validate_geometry_hmonnx(path):
    import onnx

    rows = []
    model = onnx.load(path, load_external_data=False)
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        namespace = next((p.value for p in node.metadata_props if p.key == "namespace"), "")
        parts = namespace.split("/")
        scope = parts[1].split(":")[0] if len(parts) > 1 else ""
        if scope not in GEOMETRY_MODULES:
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get("hmfp_act_man_bit") != 16 or attrs.get("hmfp_weight_man_bit") != 16:
            raise ValueError(f"Geometry precision was not applied to {scope}")
        rows.append({"module": scope, "node": node.name, "lhs_man_bit": 16, "rhs_man_bit": 16})
    if len(rows) != 8 or {r["module"] for r in rows} != set(GEOMETRY_MODULES):
        raise ValueError("Expected eight geometry MatMul nodes in HMONNX")
    return rows


def quantize(source, calibration_paths, output_dir, geometry16=False, device="cuda:0", simplify=True):
    import onnx
    import torch
    from xhquant.api import DeviceType, QuantScheme, create_quant_config, ptq_quantize, xhquant_init
    from xhquant.api.ptq_export_hmonnx import _convert_model_to_quanted_model, convert_quanted_model_to_hmonnx
    from xhquant.common.types import FrontendType, PrecisionMode

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))
    model = onnx.load(source, load_external_data=False)
    if any(n.op_type == "GatherElements" for n in model.graph.node):
        raise ValueError("Run normalize before quantization: SDK broadcast GatherElements is not supported")
    names = [v.name for v in model.graph.input]
    output_names = [v.name for v in model.graph.output]
    if output_names != ["depths", "raw_gaussians"]:
        raise ValueError(f"Expected raw DepthSplat export, got {output_names}")
    calibration = []
    for path in calibration_paths:
        with np.load(path, allow_pickle=False) as archive:
            calibration.append([torch.from_numpy(archive[name].copy()) for name in names])
    if not calibration:
        raise ValueError("At least one calibration sample is required")
    report = {
        "stage": "frontend",
        "source": str(source),
        "calibration": list(map(str, calibration_paths)),
        "target": "XH2a",
        "quant_type": "w8a8",
        "geometry16": geometry16,
        "profile": "6v512960_two_scales",
        "device": device,
        "simplify": simplify,
    }
    start = time.perf_counter()
    previous_grad = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(False)
        xhquant_init(str(output_dir / "sdk.log"), False)
        config = create_quant_config(
            QuantScheme(target_device=DeviceType.XH2a, quant_type="w8a8", nodes=GEOMETRY_NODES if geometry16 else {})
        )
        graph = _convert_model_to_quanted_model(
            str(source),
            FrontendType.ONNX,
            calibration[0],
            DeviceType.XH2a,
            config,
            use_ptq=False,
            input_names=names,
            simplify=simplify,
        )
        if geometry16:
            for name in GEOMETRY_MODULES:
                module = graph.get_submodule(name)
                if (int(module.i_cfg_1.qspec.man_bit), int(module.i_cfg_2.qspec.man_bit)) != (16, 16):
                    raise ValueError(f"Frontend did not apply a16a16 to {name}")
        report["stage"] = "ptq"
        write_report(output_dir / "report.json", report)
        ptq_quantize(graph, calibration, PrecisionMode.ALIGNED, torch.device(device))
        report["stage"] = "hmonnx_export"
        convert_quanted_model_to_hmonnx(graph, calibration[0], str(output_dir / "model.onnx"), names, output_names)
        if geometry16:
            report["geometry_nodes"] = validate_geometry_hmonnx(output_dir / "model.onnx")
        report["stage"] = "exported"
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        torch.set_grad_enabled(previous_grad)
        report["elapsed_s"] = time.perf_counter() - start
        write_report(output_dir / "report.json", report)
    return report


def infer(hmonnx, sample, output, device="cuda:0", reference=None):
    import onnx
    import torch
    from xhquant.xhonnxruntime.hmonnx_inference import HMONNXGoldenInference

    if torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))
    model = onnx.load(hmonnx, load_external_data=False)
    with np.load(sample, allow_pickle=False) as archive:
        args = [
            torch.from_numpy(
                archive[v.name].astype(onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type))
            ).to(device)
            for v in model.graph.input
        ]
    session = HMONNXGoldenInference(str(hmonnx))
    session.to(device)
    start = time.perf_counter()
    with torch.inference_mode():
        result = session(*args)
    if not isinstance(result, (list, tuple)):
        result = (result,)
    if len(result) != len(model.graph.output):
        raise ValueError("HMONNX output count differs from graph contract")
    arrays = {v.name: tensor.float().cpu().numpy() for v, tensor in zip(model.graph.output, result)}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("HMONNX produced nonfinite outputs")
    elapsed = time.perf_counter() - start
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    report = {
        "hmonnx": str(hmonnx),
        "sample": str(sample),
        "device": device,
        "backend": "HMONNX GoldenInference simulation",
        "seconds": elapsed,
        "outputs": {name: list(value.shape) for name, value in arrays.items()},
    }
    if reference:
        with np.load(reference, allow_pickle=False) as archive:
            report["metrics"] = {name: metrics(archive[name], value) for name, value in arrays.items()}
    write_report(output.with_suffix(".json"), report)
    return report
