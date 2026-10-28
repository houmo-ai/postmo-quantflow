# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Build or run MiniCPM-OCR through the public API."""

import argparse
import json


def main():
    from xhmodel_merak.model_build import build_model
    from xhmodel_merak.model_inference import inference_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir")
    parser.add_argument("--deployment-precision", choices=["w8a8h1_sefp", "w16a16h0_sefp"], default="w8a8h1_sefp")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--image")
    parser.add_argument("--prompt", default="请识别图片中的文字。")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.model_dir:
        result = build_model(
            model_id="minicpm-ocr-1.0",
            model_dir=args.model_dir,
            output_dir=args.artifact_dir,
            deployment_precision=dict.fromkeys(("visual", "prefill", "decode"), args.deployment_precision),
            device=args.device,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if args.image:
        result = inference_model(
            artifact_dir=args.artifact_dir,
            inputs={"image": args.image, "prompt": args.prompt, "max_new_tokens": args.max_new_tokens},
            device=args.device,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
