"""Run the MiniCPM-V-4.7 Merak export workflow."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG = "configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/1b/minicpm_v_4_7_xh2a_w8a8.yaml"


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MiniCPM-V-4.7 Merak workflow.")
    parser.add_argument("--model-dir", required=True, help="MiniCPM-V-4.7 HF model directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG)
    parser.add_argument("--quant-output-dir", default="work_dirs/minicpm_v_4_7_merak_quant")
    parser.add_argument("--quanted-model-dir", default=None,
                        help="Reuse an existing MiniCPM causal QDQ or calibrated-head checkpoint")
    parser.add_argument("--export-output-dir", default="work_dirs/minicpm_v_4_7_merak_export")
    parser.add_argument("--dspark-model-dir", "--dflash-model-dir", dest="dspark_model_dir",
                        default=None, help="Paired DSpark checkpoint directory.")
    parser.add_argument("--confidence-threshold", type=float, default=None,
                        help="Enable DSpark cumulative-confidence truncation with this threshold.")
    parser.add_argument("--num-draft-tokens", type=int, default=None)
    parser.add_argument("--les-prediction-k", type=int, default=None,
                        help="MoE single-token decode: predicted next-layer experts; 0 disables LES.")
    parser.add_argument("--device", default="cuda:0")
    capacity = parser.add_mutually_exclusive_group()
    capacity.add_argument("--patch-token-gears", type=int, nargs="+", help="Shared input-patch capacities for 4x and 16x.")
    capacity.add_argument("--token-capacity", type=int, default=None, help="Legacy single input-patch capacity.")
    capacity.add_argument("--image-token-gears", type=int, nargs="+", help="Post-merge visual output capacities.")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--prefill-length", type=int, default=None)
    parser.add_argument("--quant-type", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--golden-image", default=None)
    parser.add_argument("--golden-prompt", default="请描述这张图片。")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {}
    if args.token_capacity is not None:
        result["export.vision.patch_token_gears"] = None
        result["export.vision.image_token_gears"] = None
        result["export.vision.token_capacity"] = args.token_capacity
    if getattr(args, "image_token_gears", None) is not None:
        result["export.vision.patch_token_gears"] = None
        result["export.vision.token_capacity"] = None
        result["export.vision.image_token_gears"] = args.image_token_gears
    if getattr(args, "patch_token_gears", None) is not None:
        result["export.vision.image_token_gears"] = None
        result["export.vision.token_capacity"] = None
        result["export.vision.patch_token_gears"] = args.patch_token_gears
    if args.context_length is not None:
        result["export.model.context_max_length"] = args.context_length
    if args.prefill_length is not None:
        result["export.model.prefill_chunk_length"] = args.prefill_length
    if args.quant_type is not None:
        for component in ("vision_4x", "vision_16x", "llm"):
            result[f"export.components.{component}.quant_type"] = args.quant_type
    if getattr(args, "dspark_model_dir", None):
        result["export.model.spec_decode_mode"] = "dspark"
        result["export.model.dspark_config.hf_model"] = args.dspark_model_dir
    if getattr(args, "confidence_threshold", None) is not None:
        result["export.model.dspark_config.enable_confidence_truncation"] = True
        result["export.model.dspark_config.confidence_threshold"] = args.confidence_threshold
    if getattr(args, "num_draft_tokens", None) is not None:
        result["export.model.num_draft_tokens"] = args.num_draft_tokens
    if getattr(args, "les_prediction_k", None) is not None:
        result["export.model.les_prediction_k"] = args.les_prediction_k
    return result


def main() -> None:
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        debug=args.debug,
    )
    overrides = _overrides(args)
    if args.quanted_model_dir:
        from xhmodel_merak.xh_llm.workflows.result import QuantResult

        quant_result = QuantResult(raw_model_dir=args.model_dir,
                                   quanted_model_dir=args.quanted_model_dir)
    else:
        quant_result = workflow.quant(
            output_dir=args.quant_output_dir,
            device=args.device,
            config_overrides=overrides,
        )
    export_dir = Path(args.export_output_dir)
    _remove_output_dir_if_needed(str(export_dir), args.overwrite)
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=str(export_dir),
        device=args.device,
        config_overrides=overrides,
    )
    print(f"Exported MiniCPM-V-4.7 artifacts to {export_result.work_dir}")
    if args.dump_golden:
        input_messages = {"text": args.golden_prompt}
        if args.golden_image:
            input_messages["image"] = args.golden_image
        golden_dir = workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages=input_messages,
        )
        print(f"Golden data written under {golden_dir}")


if __name__ == "__main__":
    main()
