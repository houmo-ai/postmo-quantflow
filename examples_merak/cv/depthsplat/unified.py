# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Build and run DepthSplat with the public Merak API; save Gaussian arrays."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--model-dir", help="Build this prepared ONNX bundle first")
    parser.add_argument("--sample", required=True, help="Raw input NPZ matching the artifact sample contract")
    parser.add_argument("--output", required=True, help="Output Gaussian NPZ")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import numpy as np
    from xhmodel_merak.model_build import build_model
    from xhmodel_merak.model_inference import inference_model

    model_id = "depthsplat-gs-small-re10k-256x256-view2"
    artifact_dir = args.artifact_dir
    if args.model_dir:
        build = build_model(
            model_id=model_id,
            model_dir=args.model_dir,
            output_dir=artifact_dir,
            device=args.device,
        )
        artifact_dir = build.output_dir
    result = inference_model(
        artifact_dir=artifact_dir, expected_model_id=model_id, inputs={"sample": args.sample}, device=args.device
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result.stats["outputs"])
    print(f"Saved Gaussian outputs to {output}; inference {result.latency_s:.3f}s")


if __name__ == "__main__":
    main()
