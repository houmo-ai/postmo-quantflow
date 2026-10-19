# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Runtime for prepared static Gaussian encoder graphs."""

from pathlib import Path

import numpy as np

from xhmodel_merak.model_artifact.files import confined_file
from .assets import graph_inputs, load_sample


def run_artifact(root, meta, sample_path, device, *, golden_dir=None):
    import torch
    from xhmodel_merak.exec_backends import HMONNXGoldenInference

    path = confined_file(Path(root).resolve(), meta["hmonnx_file"], "hmonnx_file")
    feed = load_sample(sample_path, meta["sample_input_specs"])
    prepared = graph_inputs(feed, meta["input_names"], meta.get("preprocess"))
    session = HMONNXGoldenInference(str(path))
    try:
        session.to(device)
        if golden_dir is not None:
            Path(golden_dir).mkdir(parents=True, exist_ok=False)
            session.save_golden = True
            session.golden_dir = str(golden_dir)
            session.step = 0
        args = [
            torch.from_numpy(prepared[name].astype(meta["input_specs"][name]["dtype"])).to(device)
            for name in meta["input_names"]
        ]
        with torch.inference_mode():
            outputs = session(*args)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        if len(outputs) != len(meta["output_names"]):
            raise ValueError("HMONNX output count differs from metadata")
        result = {name: value.detach().float().cpu().numpy() for name, value in zip(meta["output_names"], outputs)}
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("HMONNX produced non-finite outputs")
        if meta["schema_version"] in (2, 3):
            from .postprocess import postprocess_gaussians

            result = postprocess_gaussians(result, feed, meta["postprocess"])
            if golden_dir is not None:
                np.savez_compressed(Path(golden_dir) / "host_gaussians.npz", **result)
        return result
    finally:
        del session
