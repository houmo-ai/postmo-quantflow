# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Generate from text, images and videos using the exported GLM-5.3 components."""

import argparse
import json

from xhmodel_merak.xh_llm.models.glm5_3_flash.demo import generate_glm53


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--prompt", default="请用一句话介绍你自己。")
    parser.add_argument("--image", action="append", default=[], help="image file; repeat for multiple images")
    parser.add_argument("--video", action="append", default=[], help="video file; repeat for multiple videos")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--json", action="store_true", help="include token IDs and timing")
    args = parser.parse_args()
    result = generate_glm53(
        args.meta,
        prompt=args.prompt,
        image=args.image,
        video=args.video,
        max_new_tokens=args.max_new_tokens,
        devices=args.devices,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["output_text"])


if __name__ == "__main__":
    main()
