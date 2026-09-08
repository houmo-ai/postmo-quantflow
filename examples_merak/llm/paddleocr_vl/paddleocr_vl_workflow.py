# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Run PaddleOCR-VL quantization, HMONNX export and golden through AutoWorkflow."""

import argparse
from pathlib import Path

from xhmodel_merak.workflows import AutoWorkflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config", default="configs_merak/workflows/xh2a/llm_models/paddleocr_vl/paddleocr_vl_1_6.yaml"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--golden-image", required=True)
    parser.add_argument("--golden-text", default="OCR:")
    parser.add_argument("--calibration-jsonl", help="Image/prompt/answer records for the quantized workflow config.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    workflow = AutoWorkflow.from_config(args.model_dir, args.config, seed=args.seed)
    if args.calibration_jsonl:
        if workflow.workflow_config.quant is None:
            parser.error("--calibration-jsonl requires --config with an enabled quant section")
        workflow.workflow_config = workflow.workflow_config.with_overrides(
            {"quant.calibration_jsonl": str(Path(args.calibration_jsonl).resolve())}
        )
    output = Path(args.output_dir)
    quant = workflow.quant(str(output / "checkpoint"), args.device)
    exported = workflow.export(quant, str(output / "export"), args.device)
    meta = workflow.dump_golden(exported, args.device, {"image": args.golden_image, "text": args.golden_text})
    print(f"HMONNX and golden: {meta}")


if __name__ == "__main__":
    main()
