import argparse
import shutil
from pathlib import Path


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MiniCPM5 xhquant workflow.")
    parser.add_argument("--model-dir", required=True, help="HF model directory.")
    parser.add_argument(
        "--config-path",
        required=True,
        help="Workflow YAML path under configs_merak/workflows/xh2a/llm_models/minicpm5.",
    )
    parser.add_argument(
        "--quant-output-dir",
        default="work_dirs/minicpm5_quant",
        help="Quantization output directory.",
    )
    parser.add_argument(
        "--export-output-dir",
        default="work_dirs/minicpm5_export",
        help="Export output directory.",
    )
    parser.add_argument("--device", default="cuda", help="Device for quant/export/golden.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument(
        "--export-from-quanted-model",
        action="store_true",
        help="Skip quantization when model-dir already contains a quantized HF model.",
    )
    parser.add_argument("--bits", type=int, default=None, help="Override quant.bits in the workflow config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
    )

    if args.export_from_quanted_model:
        config_overrides = {"quant": None}
    else:
        config_overrides = {}
        if args.bits is not None:
            config_overrides["quant.bits"] = args.bits

    quant_result = workflow.quant(
        output_dir=args.quant_output_dir,
        device=args.device,
        config_overrides=config_overrides,
    )
    print(f"quant_result: {quant_result}")

    _remove_output_dir_if_needed(args.export_output_dir, args.overwrite)
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=args.export_output_dir,
        device=args.device,
    )
    print(f"export_result: {export_result}")

    if args.dump_golden:
        workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages={"text": "用中文简单介绍 MiniCPM5。"},
        )


if __name__ == "__main__":
    main()
