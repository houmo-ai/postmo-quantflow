# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Synthetic/real connected Golden for the independently reusable prompt component."""

import json
from pathlib import Path


def validate_component_golden(root, *, steps=(0, 1)):
    import numpy as np
    import onnx

    from .bundle import bundle_path
    from .upsampler_build import read_component

    root = Path(root).resolve()
    component = read_component(root)
    inventory = []
    for role, relative in component["graphs"].items():
        path = bundle_path(root, relative)
        graph = onnx.load(str(path), load_external_data=False).graph
        for step in steps:
            directory = path.parent / f"step_{step}"
            for kind, ports in (("inputs", graph.input), ("outputs", graph.output)):
                for port in ports:
                    name = port.name if kind == "inputs" else port.name.replace("/", "_")
                    values = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                    shape = tuple(d.dim_value for d in port.type.tensor_type.shape.dim)
                    if values.shape != shape:
                        raise ValueError(f"{role} Golden {port.name} shape disagrees with the graph")
                    expected_dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(port.type.tensor_type.elem_type))
                    if values.dtype != expected_dtype:
                        raise ValueError(f"{role} Golden {port.name} dtype disagrees with the graph")
                    if kind == "outputs" and not np.isfinite(values).all():
                        raise FloatingPointError(f"{role} Golden {port.name} contains nonfinite outputs")
            inventory.append(
                dict(
                    role=role,
                    step=step,
                    inputs=len(graph.input),
                    outputs=len(graph.output),
                    output_finite=True,
                    shape_checked=True,
                )
            )
    return inventory


def complete_case_golden(runtime, root, step, reference):
    """Cover unvisited gears with explicit calibration views through the real processor."""
    import math

    from PIL import Image, ImageOps

    from .bundle import bundle_path
    from .upsampler_build import visual_graphs

    root = Path(root)
    component = runtime.component
    meta = json.loads(bundle_path(root, component["metadata"]).read_text())
    visual = visual_graphs(component, meta)
    runtime.cancel_golden()
    cases = []
    try:
        for role, gear in visual.items():
            if (bundle_path(root, component["graphs"][role]).parent / f"step_{step}").exists():
                continue
            # A near-square whole merge-grid fills the requested smallest-fit gear.
            capacity = gear["image_token_capacity"]
            cols = math.isqrt(capacity)
            rows = capacity // cols
            factor = meta["visual_config"]["patch_size"] * meta["visual_config"]["spatial_merge_size"]
            size = (cols * factor, rows * factor)
            source = reference if reference is not None else Image.new("RGB", size, (140, 100, 60))
            image = ImageOps.fit(source.convert("RGB"), size, method=Image.Resampling.LANCZOS)
            missing = {
                key
                for key, path in component["graphs"].items()
                if key in ("prefill", "decode", role) and not (bundle_path(root, path).parent / f"step_{step}").exists()
            }
            runtime.capture_golden(step, roles=missing)
            caption = runtime("Describe this image as a complete scene caption.", [image], (1024, 1024))
            cases.append(
                dict(
                    role=role,
                    calibration_hw=[image.height, image.width],
                    caption=caption,
                    stats=dict(runtime.last_stats),
                )
            )
            runtime.cancel_golden()
        for role, path in component["graphs"].items():
            directory = bundle_path(root, path).parent / f"step_{step}"
            if not directory.is_dir() or not any(directory.iterdir()):
                raise RuntimeError(f"missing connected prompt Golden: {role}: {directory}")
    finally:
        runtime.cancel_golden()
    return cases


def dump_component_golden(root, reference, *, device="cuda:0"):
    from PIL import Image

    from .bundle import bundle_path
    from .upsampler import PromptUpsampler
    from .upsampler_build import read_component

    root = Path(root).resolve()
    component = read_component(root)
    for path in component["graphs"].values():
        if any(bundle_path(root, path).parent.glob("step_*")):
            raise FileExistsError("prompt component Golden already exists")
    references = [reference] if isinstance(reference, (str, Path)) else list(reference)
    if not references or len(references) > 10:
        raise ValueError("prompt Golden requires one to ten references")
    real = []
    for reference in references:
        with Image.open(reference) as image:
            real.append(image.convert("RGB"))
    runtime = PromptUpsampler(root, component, device=device)
    cases = []
    try:
        synthetic = [Image.new("RGB", (1024, 1024), (140, 100, 60)) for _ in real]
        for step, (name, images) in enumerate([("synthetic", synthetic), ("real", real)]):
            runtime.capture_golden(step)
            caption = runtime("Combine the reference objects in one coherent scene.", images, (1024, 1024))
            stats = dict(runtime.last_stats)
            captures = complete_case_golden(runtime, root, step, images[0])
            cases.append(dict(name=name, step=step, caption=caption, stats=stats, captures=captures))
    finally:
        runtime.close()
    report = dict(
        complete=True,
        cases=cases,
        graphs=component["graphs"],
        feed_source="connected_hmonnx_outputs",
        io_validation=validate_component_golden(root),
    )
    destination = root / "upsampler_golden_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return str(destination)
