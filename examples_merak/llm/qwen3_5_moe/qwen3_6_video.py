# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Video generation through the public model inference API."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--video", required=True, help="Local video file or sorted RGB frame directory")
    parser.add_argument("--prompt", default="描述这段视频。")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    from xhmodel_merak.model_inference import inference_model
    result = inference_model(artifact_dir=args.artifact_dir, device=args.device, inputs={
        "prompt": args.prompt, "video": args.video, "max_new_tokens": args.max_new_tokens,
    })
    print(result.output_text)


if __name__ == "__main__":
    main()
