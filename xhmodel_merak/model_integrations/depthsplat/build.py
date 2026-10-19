"""Validate and publish workflow-owned DepthSplat artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import confined_file
from xhmodel_merak.model_build.contracts import BuildPlan


def load_runtime_metadata(meta_file: Path, resolve_file: Callable, components: tuple[str, ...]) -> dict[str, Any]:
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    expected = {
        "model_type": "XHDepthSplatModel",
        "target_device": "XH2a",
        "quant_type": "w8a8",
        "profile": "2v_256x256",
    }
    if not isinstance(meta, dict) or components != ("visual",):
        raise ValueError("DepthSplat requires object metadata and only the visual component")
    version = meta.get("schema_version")
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("DepthSplat schema_version must be 1, 2 or 3")
    if version < 3 and "preprocess" in meta:
        raise ValueError("DepthSplat schema 1/2 cannot contain preprocess metadata")
    if version == 3 and meta.get("preprocess") != {
        "type": "depthsplat_matching_grid_fp32_v1",
        "num_depth_candidates": 128,
        "feature_height": 64,
        "feature_width": 64,
    }:
        raise ValueError("DepthSplat schema 3 requires the fixed FP32 matching-grid preprocess")
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"DepthSplat metadata {key} must be {value!r}")
    for key in ("input_names", "output_names"):
        names = meta.get(key)
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
            raise ValueError(f"DepthSplat metadata {key} must contain port names")
        if len(set(names)) != len(names):
            raise ValueError(f"DepthSplat metadata {key} contains duplicate ports")
    gaussian_names = ["means", "covariances", "harmonics", "opacities"]
    host_specs = {}
    if version == 1:
        if any(key in meta for key in ("postprocess", "host_input_specs", "gaussian_output_names")):
            raise ValueError("DepthSplat schema 1 cannot contain host postprocess metadata")
        if meta["output_names"] != gaussian_names:
            raise ValueError("DepthSplat requires ordered Gaussian output ports")
    else:
        if meta["output_names"] != ["depths", "raw_gaussians"] or meta.get("gaussian_output_names") != gaussian_names:
            raise ValueError("DepthSplat schema 2 requires ordered raw and Gaussian output ports")
        postprocess = {
            "type": "depthsplat_gaussian_fp32_v1",
            "sh_degree": 2,
            "gaussian_scale_min": 1e-10,
            "gaussian_scale_max": 3.0,
            "num_surfaces": 1,
        }
        if meta.get("postprocess") != postprocess:
            raise ValueError("DepthSplat schema 2 requires the fixed FP32 Gaussian postprocess")
        host_shapes = {
            "images": [1, 2, 3, 256, 256],
            "intrinsics": [1, 2, 3, 3],
            "extrinsics": [1, 2, 4, 4],
            **{f"camera_{3+i}_sh{i}": [1, 2, 1, 1, 1, 1, 2 * i + 1, 2 * i + 1] for i in range(3)},
        }
        if version == 3:
            host_shapes.update(near=[1, 2], far=[1, 2])
        host_specs = {name: {"shape": shape, "dtype": "float32"} for name, shape in host_shapes.items()}
        if meta.get("host_input_specs") != host_specs:
            raise ValueError("DepthSplat host_input_specs disagree with the fixed FP32 camera contract")
    specs = meta.get("input_specs")
    if not isinstance(specs, dict) or set(specs) != set(meta["input_names"]):
        raise ValueError("DepthSplat input_specs must match input_names")
    for name, spec in specs.items():
        if not isinstance(spec, dict):
            raise ValueError(f"DepthSplat input spec {name} must be an object")
        shape = spec.get("shape")
        if not isinstance(shape, list) or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError(f"DepthSplat input spec {name} requires a static shape")
        if spec.get("dtype") not in {"float16", "float32", "int32", "int64", "bool"}:
            raise ValueError(f"DepthSplat input spec {name} has unsupported dtype")
    images = specs.get("images", {})
    if images.get("shape") != [1, 2, 3, 256, 256] or images.get("dtype") not in {"float16", "float32"}:
        raise ValueError("DepthSplat images require floating [1,2,3,256,256] inputs")
    generated_inputs = {"matching_grid"} if version == 3 else set()
    if version == 3:
        if meta["input_names"] != ["images", "near", "far", "matching_grid"]:
            raise ValueError("DepthSplat schema 3 requires ordered images, near, far, matching_grid ports")
        for name in ("near", "far"):
            if specs[name]["shape"] != [1, 2] or specs[name]["dtype"] not in {"float16", "float32"}:
                raise ValueError(f"DepthSplat graph {name} requires floating [1,2] inputs")
        grid = specs.get("matching_grid", {})
        if grid.get("shape") != [2, 8192, 64, 2] or grid.get("dtype") not in {"float16", "float32"}:
            raise ValueError("DepthSplat matching_grid requires floating [2,8192,64,2] inputs")
    sample_specs = meta.get("sample_input_specs")
    if not isinstance(sample_specs, dict) or set(sample_specs) != (set(specs) - generated_inputs) | set(host_specs):
        raise ValueError("DepthSplat sample_input_specs must match graph and host inputs")
    for name, spec in host_specs.items():
        if sample_specs[name] != spec:
            raise ValueError(f"DepthSplat sample host input {name} must preserve FP32 shape and dtype")
    for name in set(specs) - generated_inputs:
        spec = sample_specs[name]
        if not isinstance(spec, dict) or spec.get("shape") != specs[name]["shape"]:
            raise ValueError(f"DepthSplat sample shape for {name} disagrees with graph")
        pair = (spec.get("dtype"), specs[name]["dtype"])
        if pair[0] != pair[1] and pair != ("float32", "float16"):
            raise ValueError(f"DepthSplat sample dtype for {name} cannot convert to graph dtype")
    graph_path = resolve_file(meta.get("hmonnx_file"), "hmonnx_file")
    external = meta.get("hmonnx_external_data")
    if not isinstance(external, list) or any(not isinstance(path, str) for path in external):
        raise ValueError("DepthSplat hmonnx_external_data must be a list of paths")
    for path in external:
        resolve_file(path, "hmonnx_external_data")
    import onnx

    from google.protobuf.message import DecodeError

    try:
        graph = onnx.load(str(graph_path), load_external_data=False).graph
    except DecodeError as exc:
        raise ValueError("DepthSplat HMONNX is not a valid protobuf") from exc
    initializer_names = {value.name for value in graph.initializer}
    inputs = [value for value in graph.input if value.name not in initializer_names]
    if [value.name for value in inputs] != meta["input_names"] or [value.name for value in graph.output] != meta[
        "output_names"
    ]:
        raise ValueError("DepthSplat graph ports disagree with runtime metadata")
    dtype_names = {1: "float32", 10: "float16", 6: "int32", 7: "int64", 9: "bool"}
    for value in inputs:
        tensor = value.type.tensor_type
        spec = specs[value.name]
        if [dim.dim_value for dim in tensor.shape.dim] != spec["shape"] or dtype_names.get(tensor.elem_type) != spec[
            "dtype"
        ]:
            raise ValueError(f"DepthSplat graph input {value.name} disagrees with runtime metadata")
    if version >= 2:
        output_shapes = {"depths": [1, 2, 65536, 1, 1], "raw_gaussians": [2, 37, 256, 256]}
        for value in graph.output:
            tensor = value.type.tensor_type
            if [dim.dim_value for dim in tensor.shape.dim] != output_shapes[value.name] or dtype_names.get(
                tensor.elem_type
            ) not in {"float16", "float32"}:
                raise ValueError(f"DepthSplat graph output {value.name} violates the fixed raw output ABI")
    declared = {resolve_file(path, "hmonnx_external_data") for path in external}
    referenced = set()
    for tensor in graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            location = dict((entry.key, entry.value) for entry in tensor.external_data).get("location")
            if not location:
                raise ValueError("DepthSplat external tensor is missing its location")
            # Validate the ONNX-relative reference before resolving it.
            confined_file(graph_path.parent, location, "onnx.external_data")
            referenced.add(graph_path.parent / location)
    if {path.resolve() for path in referenced} != declared:
        raise ValueError("DepthSplat external weight references disagree with runtime metadata")
    return meta


def normalize_export(stage_path: Path, export_result: Any, plan: BuildPlan) -> None:
    if Path(export_result.work_dir).resolve() != stage_path.resolve():
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED, "DepthSplat export must use the build stage", model=plan.model_id
        )
    return None


def package_runtime(stage_path: Path, plan: BuildPlan) -> None:
    # The workflow owns all relative graph and runtime assets.
    del stage_path, plan


def sanitize_artifact(stage_path: Path, plan: BuildPlan) -> None:
    # Shared publication sanitizes text resources; graph paths stay unchanged.
    del stage_path, plan


def validate_artifact(stage_path: Path, plan: BuildPlan) -> dict[str, Any]:
    root = stage_path.resolve()
    try:
        meta_file = confined_file(root, "runtime_meta_info.json", "metadata_file")
        meta = load_runtime_metadata(meta_file, lambda path, field: confined_file(root, path, field), plan.components)
    except (ValueError, OSError) as exc:
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, str(exc), model=plan.model_id) from exc
    return {
        "metadata_file": meta_file.name,
        "export_entries": {"visual": {"path": meta["hmonnx_file"], "type": "hmonnx"}},
        "runtime_kind": "depthsplat_gaussian_encoder",
        "components": list(plan.components),
        "requires_source": False,
        "loadable": True,
    }
