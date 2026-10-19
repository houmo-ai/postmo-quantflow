# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Compare reloaded HMONNX to float ONNX on held-out real scene inputs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--outputs-dir", help="Save Gaussian arrays for downstream rendering")
    args = parser.parse_args()
    import onnxruntime as ort
    from xhmodel_merak.xh_other_model.models.depthsplat.assets import (
        graph_inputs,
        load_bundle,
        load_sample,
        onnx_contract,
        sample_contract,
    )
    from xhmodel_merak.xh_other_model.models.depthsplat.evaluation import compare_outputs
    from xhmodel_merak.xh_other_model.models.depthsplat.runtime import run_artifact

    config, source, _, samples = load_bundle(args.model_dir)
    if not samples:
        raise ValueError("No held-out evaluation_samples in source config")
    specs, names = onnx_contract(source, config.get("postprocess"))
    sample_specs = sample_contract(config, specs)
    from xhmodel_merak.model_artifact.files import confined_file
    from xhmodel_merak.model_integrations.depthsplat.build import load_runtime_metadata

    root = Path(args.artifact_dir).resolve()
    meta = load_runtime_metadata(
        confined_file(root, "runtime_meta_info.json", "metadata_file"),
        lambda path, field: confined_file(root, path, field),
        ("visual",),
    )
    if (
        sample_specs != meta["sample_input_specs"]
        or config.get("postprocess") != meta.get("postprocess")
        or config.get("preprocess") != meta.get("preprocess")
    ):
        raise ValueError("Source ONNX ports differ from artifact sample contract")
    session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    report = {"comparison": "float_onnx_vs_reloaded_hmonnx", "business_accuracy_verified": False, "samples": []}
    for index, sample in enumerate(samples):
        feed = load_sample(sample, sample_specs)
        reference = dict(zip(names, session.run(names, graph_inputs(feed, specs, config.get("preprocess")))))
        if "postprocess" in config:
            from xhmodel_merak.xh_other_model.models.depthsplat.postprocess import postprocess_gaussians

            reference = postprocess_gaussians(reference, feed, config["postprocess"])
        actual = run_artifact(args.artifact_dir, meta, sample, args.device)
        record = {"sample": str(sample), "outputs": compare_outputs(reference, actual)}
        if args.outputs_dir:
            import numpy as np

            output_dir = Path(args.outputs_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for label, values in (("float", reference), ("quantized", actual)):
                output = output_dir / f"{index:04d}-{sample.stem}-{label}.npz"
                np.savez_compressed(output, **values)
                record[f"{label}_gaussians"] = str(output)
        report["samples"].append(record)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
