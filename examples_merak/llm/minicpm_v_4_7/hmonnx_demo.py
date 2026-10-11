"""Run MiniCPM-V-4.7 HMONNX inference from root export metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from xhmodel_merak.xh_llm.models.minicpm_v_4_7.inference import MiniCPMV47HMONNXRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MiniCPM-V-4.7 HMONNX inference.")
    parser.add_argument("--export-meta", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--video", type=Path, action="append", default=[])
    parser.add_argument("--prompt", default="请描述这张图片。")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--downsample-mode", choices=("4x", "16x"), default="4x")
    parser.add_argument("--max-slice-nums", type=int, default=36)
    parser.add_argument("--video-max-slice-nums", type=int, default=1)
    parser.add_argument("--video-max-num-frames", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.export_meta.is_file():
        raise FileNotFoundError(f"Export metadata not found: {args.export_meta}")
    media = [*args.image, *args.video]
    if not media:
        raise ValueError("Pass at least one --image or --video")
    missing = [str(path) for path in media if not path.is_file()]
    if missing:
        raise FileNotFoundError("Media does not exist: " + ", ".join(missing))

    messages = {
        "text": args.prompt,
        "images": [str(path.resolve()) for path in args.image] or None,
        "videos": [str(path.resolve()) for path in args.video] or None,
    }
    messages = {key: value for key, value in messages.items() if value is not None}
    runtime = MiniCPMV47HMONNXRuntime(args.export_meta, device=args.device)
    output = runtime.generate(
        messages,
        downsample_mode=args.downsample_mode,
        max_slice_nums=args.max_slice_nums,
        video_max_slice_nums=args.video_max_slice_nums,
        video_max_num_frames=args.video_max_num_frames,
        max_new_tokens=args.max_new_tokens,
    )
    print(output)


if __name__ == "__main__":
    main()
