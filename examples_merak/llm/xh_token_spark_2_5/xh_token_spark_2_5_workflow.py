import argparse
import shutil
import sys
from pathlib import Path


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the xh_token_spark_2_5 Merak quant/export workflow.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="HF model directory.",
    )
    parser.add_argument(
        "--config-path", "--config",
        required=True,
        help="Workflow YAML path, choose one in configs_merak/workflows/xh2a/llm_models/xh_token_spark",
    )
    parser.add_argument(
        "--quant-output-dir",
        default="work_dirs/xh_token_spark_2_5_quant",
        help="Quantization output directory. Default: work_dirs/xh_token_spark_2_5_quant",
    )
    parser.add_argument(
        "--export-output-dir",
        default="work_dirs/xh_token_spark_2_5_export",
        help="Export output directory. Default: work_dirs/xh_token_spark_2_5_export",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for quant/export/golden/quick test. Default: cuda",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing export output directories before running.",
    )
    parser.add_argument(
        "--dump-golden",
        action="store_true",
        help="Dump golden data after export.",
    )
    parser.add_argument(
        "--export-from-quanted-model",
        action="store_true",
        help="if --model-dir is a quanted model, set this param to True",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="quantization bits, set this param to override config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # A direct script invocation must use this checkout, not another editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    # 初始化工作流
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
    )

    # quant
    if args.export_from_quanted_model:
        quant_result = workflow.quant(
            output_dir=args.quant_output_dir,
            device=args.device,
            # 从已量化的 HF 模型导出，需要跳过量化阶段
            config_overrides={"quant": None},
        )
    else:
        # 可覆盖config.yaml中已有字段
        config_overrides = {}
        if args.bits:
            config_overrides["quant.bits"] = args.bits
        quant_result = workflow.quant(
            output_dir=args.quant_output_dir,
            device=args.device,
            config_overrides=config_overrides,
        )
    print(f"quant_result: {quant_result}")

    # export
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
            input_messages={"text": "用中文简单介绍自己。"},
        )


if __name__ == "__main__":
    main()
