# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Build or score with the public OCR RM API."""

import argparse
import json
from pathlib import Path


def main():
    from xhmodel_merak.model_build import build_model
    from xhmodel_merak.model_inference import inference_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir")
    parser.add_argument("--deployment-precision", choices=["w8a8h1_sefp", "w16a16h1_sefp"], default="w8a8h1_sefp")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--orig-image")
    parser.add_argument("--render-image")
    parser.add_argument("--ocr-text-file")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.model_dir:
        result = build_model(
            model_id="ocr-rm-2b-value",
            model_dir=args.model_dir,
            output_dir=args.artifact_dir,
            deployment_precision=dict.fromkeys(("visual", "prefill", "decode"), args.deployment_precision),
            device=args.device,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if args.orig_image:
        if not args.render_image or not args.ocr_text_file:
            parser.error("--orig-image requires --render-image and --ocr-text-file")
        result = inference_model(
            artifact_dir=args.artifact_dir,
            device=args.device,
            inputs={
                "orig_image": args.orig_image,
                "render_image": args.render_image,
                "ocr_text": Path(args.ocr_text_file).read_text().rstrip("\r\n"),
            },
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
