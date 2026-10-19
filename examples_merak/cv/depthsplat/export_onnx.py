# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Export real 2v/6v DepthSplat contexts from local upstream source and weights."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--dino-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", choices=("2v_256x256", "6v_512x960"), required=True)
    parser.add_argument(
        "--sample", action="append", required=True, help="Prepared context NPZ; repeat for multiple scenes"
    )
    parser.add_argument(
        "--calibration-sample",
        action="append",
        default=[],
        help="Prepared calibration context NPZ; exported separately from evaluation samples",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--verify", action="store_true", help="Compare every real sample against CPU ONNX Runtime")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    from xhmodel_merak.xh_other_model.models.depthsplat.source import export_bundle

    report = export_bundle(
        args.source_dir,
        args.dino_dir,
        args.checkpoint,
        args.profile,
        args.sample,
        args.output_dir,
        args.device,
        args.calibration_sample,
        args.verify,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
