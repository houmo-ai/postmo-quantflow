# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""6v camera preparation, ONNX normalization, quantization and HMONNX inference."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-inputs")
    prepare.add_argument("--sample", required=True, help="Context NPZ with original FP32 cameras and images")
    prepare.add_argument("--output", required=True)
    normalize = commands.add_parser("normalize")
    normalize.add_argument("--onnx", required=True)
    normalize.add_argument("--output", required=True)
    normalize.add_argument("--sample", action="append", default=[], help="Prepared NPZ; check ORT equivalence")
    quantize = commands.add_parser("quantize")
    quantize.add_argument("--onnx", required=True, help="Normalized 6v encoder ONNX")
    quantize.add_argument("--calibration", action="append", required=True, help="Prepared calibration NPZ")
    quantize.add_argument("--output-dir", required=True)
    quantize.add_argument("--geometry16", action="store_true", help="Use SEFP a16a16 on eight geometry MatMuls")
    quantize.add_argument("--no-simplify", action="store_true")
    quantize.add_argument("--device", default="cuda:0")
    infer = commands.add_parser("infer")
    infer.add_argument("--hmonnx", required=True)
    infer.add_argument("--sample", required=True)
    infer.add_argument("--output", required=True, help="Raw depths/raw_gaussians NPZ for host postprocessing")
    infer.add_argument("--reference", help="Optional float raw NPZ for error metrics")
    infer.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(8)
    from xhmodel_merak.xh_other_model.models.depthsplat import onnx_pipeline as pipeline

    if args.command == "prepare-inputs":
        import numpy as np

        values = pipeline.prepare_inputs(args.sample)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **values)
        report = {"output": args.output, "shapes": {n: list(v.shape) for n, v in values.items()}}
    elif args.command == "normalize":
        report = pipeline.normalize_neighbors(args.onnx, args.output, args.sample)
    elif args.command == "quantize":
        report = pipeline.quantize(
            args.onnx, args.calibration, args.output_dir, args.geometry16, args.device, not args.no_simplify
        )
    else:
        report = pipeline.infer(args.hmonnx, args.sample, args.output, args.device, args.reference)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
