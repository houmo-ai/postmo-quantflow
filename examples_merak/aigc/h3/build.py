# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Publish native H3 with optional independently selectable LoRA adapters."""

import argparse
import json

from xhmodel_merak.model_build import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quant-checkpoint")
    parser.add_argument("--lora", action="append", help="Adapter directory containing h3_adapter.json; repeatable")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--profile", choices=("all-native-768p-5s",),
        help="Unified five-mode package (default: all-native-768p-5s); use workflow YAML for other gears",
    )
    args = parser.parse_args()
    result = build_model(
        model_id="minimax-h3",
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        quant_checkpoint=args.quant_checkpoint,
        device=args.device,
        profile=args.profile,
        lora={"paths": args.lora} if args.lora else None,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
