# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Run a relocated Sol-H3 release without its original checkpoint/source folders."""

import argparse
import json

from xhmodel_merak.model_inference import inference_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ltx-audio-mode", choices=("joint", "frozen"),
        help="LTX audio policy (Ref2VA defaults to joint; other modes use the bundle default)",
    )
    parser.add_argument(
        "--ltx-guide-policy", choices=("midguide", "none"),
        help="LTX guide policy (Ref2VA defaults to midguide; none runs without a midguide)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image", help="First keyframe")
    parser.add_argument("--last-image", help="Last keyframe")
    parser.add_argument("--references", help='JSON array of ordered {"kind", "path"} references')
    args = parser.parse_args()
    inputs = {"prompt": args.prompt, "seed": args.seed, "output_path": args.output}
    for name in ("ltx_audio_mode", "ltx_guide_policy"):
        if getattr(args, name) is not None:
            inputs[name] = getattr(args, name)
    for name in ("image", "last_image"):
        if getattr(args, name) is not None:
            inputs[name] = getattr(args, name)
    if args.references is not None:
        inputs["references"] = json.loads(args.references)
    result = inference_model(
        artifact_dir=args.artifact_dir,
        expected_model_id="sol-h3",
        device=args.device,
        inputs=inputs,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
