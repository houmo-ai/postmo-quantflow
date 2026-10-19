"""Run the Qwen3.5-MoE dynamic-prune workflow (SAFE by default).

The stage order intentionally matches ``qwen3_5_workflow.py``:
``quant -> export -> dump_golden -> quick_test_hmonnx``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from xhmodel_merak.xh_llm.hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import Qwen3_5HMONNXKVCacheMixin
from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_hmonnx_inference import (
    XHQwen3_5MoeHMONNXModel,
)
from xhmodel_merak.xh_llm.models.qwen3_5_moe_dynamic_prune.qwen3_5_moe_dynamic_prune_model import (
    XHQwen3_5MoeDynamicPruneModel,
)
from xhmodel_merak.xh_llm.types import LLMModelMeta
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow


class DynamicPruneTextOnlyHMONNXModel(XHQwen3_5MoeHMONNXModel):
    """Reuse the MoE runtime without constructing an absent visual graph."""

    def __init__(self, meta_info: LLMModelMeta, **kwargs):
        VisonLLMHMONNXModel.__init__(self, meta_info, **kwargs)
        self.visual_meta = getattr(meta_info.model_config, "visual_config", None)
        if self.visual_meta is None:
            raise RuntimeError(
                "The LLM-only Qwen3.5-MoE artifact is missing "
                "model_config.visual_config preprocessing geometry."
            )
        self.visual = None
        self._kvcache_mixin = Qwen3_5HMONNXKVCacheMixin(self.kvcache_config)
        self._kvcache_mixin.split_conv_cache = bool(
            getattr(meta_info.model_config, "split_conv_cache", False)
        )
        self._sync_page_attention_mode_to_kvcache()

    def _set_device(self, device):
        return VisonLLMHMONNXModel._set_device(self, device)

    def _set_dtype(self, dtype):
        return VisonLLMHMONNXModel._set_dtype(self, dtype)

    def _set_enable_golden(self, enable: bool) -> None:
        VisonLLMHMONNXModel._set_enable_golden(self, enable)

    def get_tf_processor(self):
        raise RuntimeError(
            "Visual inputs require an exported visual HMONNX model, "
            "but this dynamic-prune artifact is LLM-only."
        )


def _is_llm_only_meta(meta_file: str | Path) -> bool:
    with Path(meta_file).open(encoding="utf-8") as file:
        meta = json.load(file)
    return meta.get("visual_config") is None


@contextmanager
def use_dynamic_prune_text_only_hmonnx_runtime() -> Iterator[None]:
    original_runtime = XHQwen3_5MoeDynamicPruneModel.HMONNXINFERENCE_CLS
    XHQwen3_5MoeDynamicPruneModel.HMONNXINFERENCE_CLS = DynamicPruneTextOnlyHMONNXModel
    try:
        yield
    finally:
        XHQwen3_5MoeDynamicPruneModel.HMONNXINFERENCE_CLS = original_runtime


def dump_dynamic_prune_golden(
    workflow: Any,
    export_result: Any,
    device: str,
    input_messages: Any,
) -> str:
    """Use the standard golden workflow with an LLM-only runtime when needed."""

    root_meta_file = workflow._find_golden_meta_file(export_result)
    if not _is_llm_only_meta(root_meta_file):
        return workflow.dump_golden(export_result, device, input_messages)

    with use_dynamic_prune_text_only_hmonnx_runtime():
        return workflow.dump_golden(export_result, device, input_messages)


def _acep_coverage(value: str) -> float:
    coverage = float(value)
    if not 0.0 < coverage <= 1.0:
        raise argparse.ArgumentTypeError(f"ACEP coverage must be in (0, 1], got {coverage}")
    return coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen3.5-MoE dynamic-prune workflow.")
    parser.add_argument("--model-dir", required=True, help="HF model directory.")
    parser.add_argument(
        "--config-path",
        default=(
            "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/"
            "qwen3_6_35b_a3b_dynamic_prune.yaml"
        ),
        help="Dynamic-prune workflow YAML path (defaults to decode-only SAFE).",
    )
    parser.add_argument("--quant-output-dir", default="work_dirs/qwen3_5_safe_quant")
    parser.add_argument("--export-output-dir", default="work_dirs/qwen3_5_safe_export")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument(
        "--coverage",
        "--threshold",
        dest="threshold",
        type=_acep_coverage,
        default=None,
        help=(
            "Override ACEP/SAFE cumulative proxy-probability coverage. "
            "The legacy --threshold spelling is retained as an alias."
        ),
    )
    parser.add_argument("--s-scalar-path", default=None, help="Optional .pt/.json expert score file.")
    parser.add_argument(
        "--prune-algorithm", choices=("acep", "safe"), default=None,
        help="Override the YAML algorithm (default workflow uses SAFE).",
    )
    parser.add_argument("--ace-priors-path", default=None, help="SAFE AMP+AIMER .pt artifact.")
    parser.add_argument("--exfold-artifact-path", default=None, help="SAFE pairwise coeff/loss/norms .pt artifact.")
    parser.add_argument("--max-projection-loss", type=float, choices=(1.0,), default=None, help="SAFE fixed at 1.0.")
    parser.add_argument(
        "--even-fold-policy", choices=("nearest_mass",), default=None, help="SAFE fixed at nearest_mass."
    )
    parser.add_argument(
        "--auto-s-scalar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Automatically compute ACEP scores or calibrate SAFE AMP/AIMER + ExFold when no files are supplied.",
    )
    parser.add_argument("--calibration-text-path", default=None, help="Optional local raw WikiText training text.")
    parser.add_argument("--calibration-nsamples", type=int, default=None)
    parser.add_argument("--calibration-sequence-length", type=int, default=None)
    parser.add_argument("--calibration-batch-size", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--only-first-block",
        action="store_true",
        help="Export the standard Qwen3.5/Qwen3.6 first hybrid block for graph inspection.",
    )
    parser.add_argument("--context-max-length", "--context-length", type=int, default=None)
    parser.add_argument(
        "--enable-fuse-gdr-ops",
        dest="fuse_gdr_ops",
        action="store_true",
        default=None,
        help="Enable GDRChunkScan fusion for this export run.",
    )
    parser.add_argument(
        "--disable-fuse-gdr-ops",
        dest="fuse_gdr_ops",
        action="store_false",
        default=None,
        help="Disable GDRChunkScan fusion for this export run.",
    )
    parser.add_argument(
        "--enable-fuse-gdr-block-recurrent-ops",
        action="store_true",
        default=None,
        help="Enable GDR block recurrent fusion for this export run.",
    )
    parser.add_argument(
        "--disable-fuse-gdr-block-recurrent-ops",
        dest="enable_fuse_gdr_block_recurrent_ops",
        action="store_false",
        default=None,
        help="Disable GDR block recurrent fusion for this export run.",
    )
    return parser.parse_args()


def _validate_safe_inputs(prune: dict[str, Any]) -> None:
    """Check required files before loading weights or removing an old export."""
    if prune.get("mode") == "off" or prune.get("algorithm") != "safe":
        return
    cache = prune.get("s_scalar_path")
    priors, exfold = prune.get("ace_priors_path"), prune.get("exfold_artifact_path")
    if cache and (priors or exfold):
        raise ValueError("SAFE accepts either --s-scalar-path or raw prior/ExFold files, not both")
    if cache:
        paths = [cache]
    elif priors and exfold:
        paths = [priors, exfold]
    elif not priors and not exfold and prune.get("auto_s_scalar", True):
        return
    else:
        raise ValueError(
            "SAFE requires --ace-priors-path and --exfold-artifact-path, "
            "or a packed v4 SAFE cache via --s-scalar-path, or auto_s_scalar=true. "
            "Set these paths in the CLI or workflow YAML."
        )
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"SAFE calibration file does not exist: {path}")


def main() -> None:
    args = parse_args()
    workflow = AutoLLMWorkflow.from_config(args.model_dir, args.config_path)
    overrides = {}
    for argument, option in (
        ("prune_algorithm", "algorithm"),
        ("ace_priors_path", "ace_priors_path"),
        ("exfold_artifact_path", "exfold_artifact_path"),
        ("max_projection_loss", "max_projection_loss"),
        ("even_fold_policy", "even_fold_policy"),
    ):
        value = getattr(args, argument)
        if value is not None:
            overrides[f"export.model.dynamic_prune.{option}"] = value
    for name in ("text_path", "nsamples", "sequence_length", "batch_size"):
        value = getattr(args, f"calibration_{name}")
        if value is not None:
            overrides[f"export.model.dynamic_prune.calibration.{name}"] = value
    if args.threshold is not None:
        overrides["export.model.dynamic_prune.threshold"] = args.threshold
    if args.s_scalar_path is not None:
        overrides["export.model.dynamic_prune.s_scalar_path"] = args.s_scalar_path
    if args.auto_s_scalar is not None:
        overrides["export.model.dynamic_prune.auto_s_scalar"] = args.auto_s_scalar
    if args.model_name:
        overrides["export.model.model_name"] = args.model_name.strip().lower().replace(".", "_").replace("-", "_")
    if args.only_first_block:
        overrides["export.model.only_first_block"] = True
    if args.context_max_length is not None:
        if args.context_max_length <= 0:
            raise ValueError("--context-max-length must be positive")
        overrides["export.model.context_max_length"] = args.context_max_length
    if args.fuse_gdr_ops is not None:
        overrides["export.model.fuse_gdr_ops"] = args.fuse_gdr_ops
    if args.enable_fuse_gdr_block_recurrent_ops is not None:
        overrides["export.model.fuse_gdr_block_recurrent_ops"] = args.enable_fuse_gdr_block_recurrent_ops

    effective_config = workflow.workflow_config.with_overrides(overrides)
    _validate_safe_inputs(effective_config.export["model"].get("dynamic_prune") or {})
    if args.overwrite and Path(args.export_output_dir).exists():
        shutil.rmtree(args.export_output_dir)

    quant_result = workflow.quant(args.quant_output_dir, args.device)
    print(f"quant_result: {quant_result}")
    export_result = workflow.export(quant_result, args.export_output_dir, args.device, overrides)
    print(f"export_result: {export_result}")

    if args.dump_golden:
        print(
            dump_dynamic_prune_golden(
                workflow,
                export_result,
                args.device,
                "用中文简单介绍 Qwen3.5。",
            )
        )
    if args.quick_test:
        from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import print_quick_test_result, quick_test_hmonnx

        print_quick_test_result(
            quick_test_hmonnx(
                export_result,
                prompt="用中文简单介绍 Qwen3.5。",
                device=args.device,
                max_new_tokens=64,
                do_sample=False,
            )
        )


if __name__ == "__main__":
    main()
