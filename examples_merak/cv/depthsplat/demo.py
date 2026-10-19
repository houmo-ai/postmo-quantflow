# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Prepare real 2v/6v inputs or render a trajectory from inference outputs."""

import argparse
import json


def indices(value):
    return [int(index) for index in value.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare original FP32 context and trajectory indices")
    render = commands.add_parser("render", help="Render MP4 and evaluate PSNR/SSIM against real frames")
    for command in (prepare, render):
        command.add_argument("--source-dir", required=True, help="External pinned DepthSplat source checkout")
        command.add_argument(
            "--dataset-dir", required=True, help="RealEstate10K .torch chunk, split or dataset directory"
        )
        command.add_argument("--scene", required=True)
        command.add_argument("--profile", choices=("2v", "6v"), required=True)
    prepare.add_argument(
        "--context-indices", type=indices, help="Comma-separated frames; defaults to official scene index"
    )
    prepare.add_argument("--target-indices", type=indices, help="Comma-separated frames; defaults to all scene frames")
    prepare.add_argument("--output", required=True)
    render.add_argument("--context", required=True, dest="context_path")
    inputs = render.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--gaussians", dest="gaussians_path")
    inputs.add_argument("--raw", dest="raw_path")
    render.add_argument("--output-dir", required=True)
    render.add_argument("--device", default="cuda:0")
    render.add_argument("--max-frames", type=int)
    render.add_argument("--chunk-size", type=int, default=10)
    render.add_argument("--fps", type=int, default=30)
    render.add_argument("--baseline-metrics", help="Optional metrics.json from the same scene/profile float render")
    args = vars(parser.parse_args())
    command = args.pop("command")
    from xhmodel_merak.xh_other_model.models.depthsplat.demo import prepare_sample, render_video

    report = prepare_sample(**args) if command == "prepare" else render_video(**args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
