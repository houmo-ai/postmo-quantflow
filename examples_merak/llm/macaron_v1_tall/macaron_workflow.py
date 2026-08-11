"""Run Macaron with the standard AutoLLM workflow entry point.

The YAML reuses Qwen quantization and base-export settings. Its Macaron
``model_type`` selects ``MacaronV1TallWorkflow`` through AutoLLM's existing
``WORKFLOW_CLS`` mechanism; that class calls the Qwen parent export, then
builds the Macaron runtime-LoRA export.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG = (
    "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/"
    "macaron_v1_tall_w8a8.yaml"
)


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize and export Macaron-V1-Tall with runtime LoRA.",
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--quant-output-dir", default="work_dirs/macaron_v1_tall_quant"
    )
    parser.add_argument(
        "--export-output-dir", default="work_dirs/macaron_v1_tall_export"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--export-from-quanted-model",
        action="store_true",
        help="if --model-dir is a quanted model, set this param to True",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--context-max-length", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow
    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
    )
    # Keep direct export aligned with the Qwen3.5/Qwen3.6 workflow example.
    if args.export_from_quanted_model:
        quant_result = workflow.quant(
            output_dir=args.quant_output_dir,
            device=args.device,
            config_overrides={"quant": None},
        )
    else:
        quant_result = workflow.quant(
            output_dir=args.quant_output_dir,
            device=args.device,
        )

    _remove_output_dir_if_needed(args.export_output_dir, args.overwrite)
    config_overrides: dict[str, object] = {}
    if args.context_max_length is not None:
        config_overrides["export.model.context_max_length"] = (
            args.context_max_length
        )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=args.export_output_dir,
        device=args.device,
        config_overrides=config_overrides,
    )
    print(f"export_result: {export_result}")
    if args.dump_golden:
        workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages={"text": "Write a Python function to sort a list."},
        )


if __name__ == "__main__":
    main()
