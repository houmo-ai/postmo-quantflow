"""Run the standard Qwen4-Exp quantization and HMONNX export workflow."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG = (
    "configs_merak/workflows/xh2a/llm_models/qwen4_exp/flash_next/"
    "qwen3_8_flash_next_unified.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantize and export Qwen3.8-Flash-Next through the Qwen4-Exp workflow.",
    )
    parser.add_argument("--model-dir", required=True, help="Raw or quantized Hugging Face model directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG, help="Merak workflow YAML.")
    parser.add_argument("--quant-output-dir", default="work_dirs/qwen4_exp_quant")
    parser.add_argument("--export-output-dir", default="work_dirs/qwen4_exp_export")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--export-from-quanted-model",
        action="store_true",
        help="Treat --model-dir as an existing GPTQModel-HF artifact and skip quantization.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing export output directory.")
    parser.add_argument("--context-max-length", type=int, default=None)
    parser.add_argument("--prefill-chunk-length", type=int, default=None)
    parser.add_argument("--max-quant-layers", type=int, default=None, help="Debug-only partial quantization limit.")
    parser.add_argument("--debug", action="store_true")
    return parser


def _positive_override(overrides: dict[str, object], key: str, value: int | None) -> None:
    if value is None:
        return
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}")
    overrides[key] = value


def main() -> None:
    args = build_parser().parse_args()

    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        seed=args.seed,
        debug=args.debug,
    )
    if args.export_from_quanted_model and args.max_quant_layers is not None:
        raise ValueError("--max-quant-layers cannot be used when quantization is skipped")
    quant_overrides: dict[str, object] = {"quant": None} if args.export_from_quanted_model else {}
    _positive_override(quant_overrides, "quant.max_quant_layers", args.max_quant_layers)
    quant_result = workflow.quant(
        output_dir=args.quant_output_dir,
        device=args.device,
        config_overrides=quant_overrides,
    )
    print(f"quant_result: {quant_result}")

    export_dir = Path(args.export_output_dir)
    if args.overwrite and export_dir.exists():
        shutil.rmtree(export_dir)
    export_overrides: dict[str, object] = {}
    _positive_override(
        export_overrides,
        "export.model.context_max_length",
        args.context_max_length,
    )
    _positive_override(
        export_overrides,
        "export.model.prefill_chunk_length",
        args.prefill_chunk_length,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=str(export_dir),
        device=args.device,
        config_overrides=export_overrides,
    )
    print(f"export_result: {export_result}")
    print(
        "conversation_validation: "
        f"{workflow.dump_golden(export_result, args.device, '你好，请简要介绍一下你自己。')}"
    )


if __name__ == "__main__":
    main()
