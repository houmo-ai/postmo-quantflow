# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Connected synthetic and representative-image Golden through the deployment runtime."""

import errno
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .runtime import WallRuntime


def dump_golden(root, *, device, inputs, output_dir=None):
    if not inputs or len(inputs.get("images", [])) != 2:
        raise ValueError("Golden requires two representative camera images and state/instruction")
    source = Path(root).resolve()
    destination = Path(output_dir).resolve() if output_dir else source.parent / (source.name + "_golden")
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Golden diagnostics must be outside the published runtime package")

    def link_or_copy(src, dst):
        try:
            return os.link(src, dst)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            return shutil.copy2(src, dst)

    shutil.copytree(source, destination, copy_function=link_or_copy)
    root = destination
    records = []
    for step, kind in enumerate(("synthetic", "representative")):
        runtime = WallRuntime(root, device=device, golden_step=step)
        values = dict(inputs)
        values["images"] = [Image.open(p).convert("RGB") if isinstance(p, (str, Path)) else p for p in values["images"]]
        if kind == "synthetic":
            rng = np.random.default_rng(11)
            values["images"] = [
                Image.fromarray(rng.integers(0, 256, (336, 448, 3), dtype=np.uint8)),
                Image.fromarray(rng.integers(0, 256, (448, 252, 3), dtype=np.uint8)),
            ]
        runtime.predict(**values)
        if runtime.captured != {"visual", "prefill", "denoise"}:
            raise ValueError("incomplete connected Golden")
        records.append(dict(step=step, kind=kind, graphs=sorted(runtime.captured)))
        del runtime
    report = dict(complete=True, backend="hmonnx_v2", output_dir=str(destination), records=records)
    (Path(root) / "golden_meta_info.json").write_text(json.dumps(report, indent=2))
    return report
