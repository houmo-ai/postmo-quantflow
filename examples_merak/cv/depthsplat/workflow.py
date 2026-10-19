# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Build a prepared DepthSplat ONNX bundle using Merak AutoWorkflow."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config", default="configs_merak/workflows/xh2a/other_models/depthsplat/depthsplat_xh2a_w8a8.yaml"
    )
    args = parser.parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    workflow = AutoWorkflow.from_config(model_dir=args.model_dir, config_path=args.config)
    quant = workflow.quant(output_dir=args.output_dir, device=args.device)
    result = workflow.export(quant, output_dir=args.output_dir, device=args.device)
    print(json.dumps({"work_dir": result.work_dir, "meta": result.meta}, indent=2))


if __name__ == "__main__":
    main()
