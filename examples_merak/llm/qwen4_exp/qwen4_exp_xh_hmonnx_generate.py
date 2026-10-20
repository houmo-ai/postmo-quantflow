"""Generate text directly from a Qwen4-Exp serving HMONNX package."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _meta_file(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if path.is_file():
        return path
    meta = path / "golden_meta_info.json"
    if meta.is_file():
        return meta
    raise FileNotFoundError(f"No golden_meta_info.json found at {path}")


def _devices(value: str) -> tuple[str, list[int] | None]:
    tokens = [token.strip().removeprefix("cuda:") for token in value.split(",") if token.strip()]
    if tokens == ["cpu"]:
        return "cpu", None
    if not tokens or any(not token.isdigit() for token in tokens):
        raise ValueError("--device must be 'cpu' or comma-separated GPU ids such as '0' or '0,1'")
    gpu_ids = list(dict.fromkeys(int(token) for token in tokens))
    return f"cuda:{gpu_ids[0]}", gpu_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen4-Exp HMONNX text generation.")
    parser.add_argument("--config", required=True, help="Serving package directory or golden_meta_info.json.")
    parser.add_argument("--prompt", default="用中文简单介绍一下你自己。")
    parser.add_argument("--device", default="0", help="'cpu' or comma-separated GPU ids.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-output-tokens", type=int, default=0)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--auto-offload", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--use-v2", action="store_true")
    parser.add_argument("--golden-output-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.use_v2:
        os.environ["ENABLE_HMINFERENCE_V2"] = "1"
    runtime_device, device_map = _devices(args.device)

    from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import print_quick_test_result
    from xhmodel_merak.xh_llm.models.qwen4_exp.hmonnx_validation import hmonnx_generate

    result = hmonnx_generate(
        meta_file=_meta_file(args.config),
        prompt=args.prompt,
        device=runtime_device,
        device_map=device_map,
        max_new_tokens=args.max_new_tokens,
        min_output_tokens=args.min_output_tokens,
        do_sample=args.do_sample,
        think=args.think,
        fast=args.fast,
        debug=args.debug,
        auto_offload=args.auto_offload,
        cuda_graph=args.cuda_graph,
        page_attention=True,
        golden_output_dir=args.golden_output_dir,
    )
    print_quick_test_result(result)


if __name__ == "__main__":
    main()
