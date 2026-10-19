# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Prepared ONNX and real sample contracts, independent of upstream imports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xhmodel_merak.model_artifact.files import confined_file

OUTPUT_NAMES = ("means", "covariances", "harmonics", "opacities")
PROFILE = "2v_256x256"


def load_bundle(root):
    root = Path(root).resolve()
    config = json.loads(confined_file(root, "config.json", "source config").read_text())
    if config.get("model_type") != "DepthSplatONNX" or config.get("profile") != PROFILE:
        raise ValueError("Expected DepthSplatONNX source with profile 2v_256x256")
    model = confined_file(root, config["onnx_file"], "onnx_file")
    samples = config.get("calibration_samples", [])
    if not samples:
        raise ValueError("calibration_samples must contain real multi-view samples")
    if len(set(samples)) != len(samples):
        raise ValueError("duplicate calibration samples")
    evaluation = config.get("evaluation_samples", [])
    calibration_paths = [confined_file(root, p, "calibration sample") for p in samples]
    evaluation_paths = [confined_file(root, p, "evaluation sample") for p in evaluation]
    if set(calibration_paths) & set(evaluation_paths):
        raise ValueError("calibration and evaluation samples must be disjoint")
    return config, model, calibration_paths, evaluation_paths


POSTPROCESS = {
    "type": "depthsplat_gaussian_fp32_v1",
    "sh_degree": 2,
    "gaussian_scale_min": 1e-10,
    "gaussian_scale_max": 3.0,
    "num_surfaces": 1,
}
RAW_OUTPUT_SHAPES = {"depths": [1, 2, 65536, 1, 1], "raw_gaussians": [2, 37, 256, 256]}
HOST_INPUT_SPECS = {
    name: {"shape": shape, "dtype": "float32"}
    for name, shape in {
        "images": [1, 2, 3, 256, 256],
        "intrinsics": [1, 2, 3, 3],
        "extrinsics": [1, 2, 4, 4],
        **{f"camera_{3+i}_sh{i}": [1, 2, 1, 1, 1, 1, 2 * i + 1, 2 * i + 1] for i in range(3)},
    }.items()
}


PREPROCESS = {
    "type": "depthsplat_matching_grid_fp32_v1",
    "num_depth_candidates": 128,
    "feature_height": 64,
    "feature_width": 64,
}
GRID_SPEC = {"shape": [2, 8192, 64, 2], "dtype": "float32"}
PREPOST_HOST_INPUT_SPECS = {
    **HOST_INPUT_SPECS,
    "near": {"shape": [1, 2], "dtype": "float32"},
    "far": {"shape": [1, 2], "dtype": "float32"},
}


def sample_contract(config, graph_specs):
    if "postprocess" not in config:
        if "host_input_specs" in config or "preprocess" in config:
            raise ValueError("host inputs and preprocess require postprocess")
        return dict(graph_specs)
    host_specs = HOST_INPUT_SPECS
    original_specs = dict(graph_specs)
    if "preprocess" in config:
        if config["preprocess"] != PREPROCESS:
            raise ValueError("Unsupported FP32 matching grid preprocess")
        if set(graph_specs) != {"images", "near", "far", "matching_grid"}:
            raise ValueError("Preprocessed source requires images, near, far and matching_grid only")
        if graph_specs.get("matching_grid") != GRID_SPEC:
            raise ValueError("Source matching_grid must have the fixed FP32 shape")
        host_specs = PREPOST_HOST_INPUT_SPECS
        del original_specs["matching_grid"]
    if config["postprocess"] != POSTPROCESS or config.get("host_input_specs") != host_specs:
        raise ValueError("Unsupported FP32 Gaussian postprocess or host input contract")
    for name in original_specs.keys() & host_specs.keys():
        if original_specs[name] != host_specs[name]:
            raise ValueError(f"Source graph and host input disagree: {name}")
    return {**original_specs, **host_specs}


def graph_inputs(feed, input_names, preprocess=None):
    """Generate graph-only inputs while preserving original FP32 Host arrays."""
    generated = {}
    if preprocess is not None:
        if preprocess != PREPROCESS:
            raise ValueError("Unsupported FP32 matching grid preprocess")
        from .preprocess import prepare_matching_grid

        generated["matching_grid"] = prepare_matching_grid(feed, preprocess)
    return {name: generated[name] if name in generated else feed[name] for name in input_names}


def onnx_contract(path, postprocess=None):
    import onnx
    from onnx import helper

    model = onnx.load(str(path), load_external_data=False)
    initializer_names = {value.name for value in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializer_names]
    specs = {}
    for value in inputs:
        shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(f"DepthSplat requires static input shapes: {value.name}: {shape}")
        specs[value.name] = {
            "shape": shape,
            "dtype": str(np.dtype(helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type))),
        }
    names = [value.name for value in model.graph.output]
    expected = OUTPUT_NAMES if postprocess is None else tuple(RAW_OUTPUT_SHAPES)
    if postprocess is not None and postprocess != POSTPROCESS:
        raise ValueError("Unsupported Gaussian postprocess")
    if names != list(expected):
        raise ValueError(f"Expected Gaussian outputs {expected}, got {names}")
    if postprocess is not None:
        for value in model.graph.output:
            tensor = value.type.tensor_type
            if [dim.dim_value for dim in tensor.shape.dim] != RAW_OUTPUT_SHAPES[value.name] or tensor.elem_type != 1:
                raise ValueError(f"Invalid raw FP32 output: {value.name}")
    if specs.get("images", {}).get("shape") != [1, 2, 3, 256, 256]:
        raise ValueError("images must have shape [1,2,3,256,256]")
    return specs, names


def load_sample(path, specs):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(specs):
            raise ValueError(f"Sample ports differ: expected {list(specs)}, got {archive.files}")
        values = {}
        for name, spec in specs.items():
            value = np.asarray(archive[name])
            if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
                raise ValueError(f"Invalid {name}: shape={value.shape}, dtype={value.dtype}; expected {spec}")
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite input: {name}")
            values[name] = np.ascontiguousarray(value)
    return values
