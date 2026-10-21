# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Collect synthetic and optional real Golden steps from the connected runtime."""

import json
import os
from pathlib import Path
from uuid import uuid4


def materialize_graph_links(directory, root):
    """Keep SDK Golden graphs without introducing symlinks into a safe bundle."""
    root = Path(root).resolve()
    for path in Path(directory).rglob("*"):
        if not path.is_symlink():
            continue
        source = path.resolve(strict=True)
        source.relative_to(root)
        if not source.is_file():
            raise ValueError("Golden links must refer to regular in-bundle graph resources")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}")
        try:
            os.link(source, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def dump_golden(root, *, device="cuda:0", inputs=None):
    from PIL import Image

    from xhquant.api import HMONNXGoldenInference

    from .bundle import ROLES, bundle_path, load_runtime, read_metadata

    root = Path(root).resolve()
    meta = read_metadata(root)
    for path in meta["graphs"].values():
        if any(bundle_path(root, path).parent.glob("step_*")):
            raise FileExistsError("Golden already exists; use a fresh destination")
    for path in meta.get("prompt_upsampler", {}).get("graphs", {}).values():
        if any(bundle_path(root, path).parent.glob("step_*")):
            raise FileExistsError("prompt upsampler Golden already exists; use a fresh destination")
    seen, captured = {}, set()
    golden_step = 0

    class Runner:
        def __init__(self, role, path):
            self.role = role
            self.directory = path.parent
            self.session = HMONNXGoldenInference(str(path))
            self.session.to(device)
            self.session.legacy_mode = False
            self.session.golden_dir = str(path.parent)

        def __call__(self, *values):
            self.session.step = golden_step
            self.session.save_golden = self.role not in captured
            result = self.session(*values)
            if self.session.save_golden:
                materialize_graph_links(self.directory / f"step_{golden_step}", root)
            captured.add(self.role)
            seen[self.role] = seen.get(self.role, 0) + 1
            return result

    values = dict(inputs or {})
    prompt = values.pop(
        "prompt",
        (
            "A red ceramic teapot rests on a wooden table by a window. Soft daylight illuminates its glossy "
            "curved body and casts a gentle shadow across the wood grain."
        ),
    )
    image_paths = values.pop("images", [])
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.copy())
    height, width = meta["capacity"]["reference_size"]
    synthetic = [Image.new("RGB", (width, height), (140, 100, 60))]
    cases = [("synthetic", synthetic)] + ([("real", images)] if images else [])
    runtime = load_runtime(root, device=device, session_factory=Runner)
    case_reports = []
    for golden_step, (name, references) in enumerate(cases):
        captured.clear()
        if values.get("prompt_upsampling"):
            if runtime.upsampler is None:
                raise ValueError("Golden prompt upsampling requires a packaged component")
            runtime.upsampler.capture_golden(golden_step)
        image, _ = runtime.generate(prompt, images=references, **values)
        if captured != set(ROLES):
            raise RuntimeError("Golden did not execute every deployed component")
        image.save(root / f"golden_{name}.png")
        prompt_captures = None
        if values.get("prompt_upsampling"):
            from .upsampler_golden import complete_case_golden

            prompt_captures = complete_case_golden(runtime.upsampler, root, golden_step, references[0])
        case_reports.append(
            dict(
                name=name,
                step=golden_step,
                geometry=runtime.last_geometry,
                caption=runtime.last_caption,
                upsampler=runtime.last_upsampler_stats,
                additional_visual_captures=prompt_captures,
            )
        )
    report = dict(
        complete=True,
        synthetic_reference=True,
        real_reference=bool(images),
        cases=case_reports,
        calls=seen,
        graphs=meta["graphs"],
        geometry=runtime.last_geometry,
        feed_source="connected_hmonnx_outputs",
        prompt_upsampler=meta.get("prompt_upsampler"),
    )
    if runtime.upsampler is not None:
        runtime.upsampler.close()
    path = root / "golden_meta_info.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return str(path)
