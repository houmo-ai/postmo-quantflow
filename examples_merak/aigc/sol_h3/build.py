# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Build the complete independent Sol-H3 release through the public API."""

import argparse
import json

from xhmodel_merak.model_build import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/data01/datasets/Sol-H3-merak-source")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quant-checkpoint", help="Reuse a complete Qwen3-VL GPTQ/AutoRound W4 checkpoint")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--profile", help="Registered unified API profile")
    selection.add_argument("--workflow", help="Custom workflow YAML with export capacities and calibration media")
    args = parser.parse_args()
    if args.workflow:
        if args.quant_checkpoint:
            parser.error("--quant-checkpoint requires a registered --profile for checkpoint identity validation")
        from xhmodel_merak.workflows import AutoWorkflow

        workflow = AutoWorkflow.from_config(model_dir=args.model_dir, config_path=args.workflow)
        quant = workflow.quant(output_dir=args.output_dir + "_quant", device=args.device)
        result = workflow.export(quant, output_dir=args.output_dir, device=args.device)
        print(json.dumps({"work_dir": result.work_dir, "config_file": result.config_file}, indent=2))
    else:
        result = build_model(
            model_id="sol-h3",
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            device=args.device,
            profile=args.profile,
            quant_checkpoint=args.quant_checkpoint,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
