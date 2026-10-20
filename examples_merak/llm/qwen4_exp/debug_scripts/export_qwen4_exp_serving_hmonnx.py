#!/usr/bin/env python3
"""Export Qwen3.8-Flash-Next weights as a reusable serving package."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from xhmodel_merak.xh_llm.models.qwen4_exp.serving_workflow import export_qwen4_exp_serving_package


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--context-length", type=int, default=262144)
    parser.add_argument("--prefill-chunk-length", type=int, default=256)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-name", default="Qwen3.8-Flash-Next-HMONNX")
    parser.add_argument("--phase", choices=("prefill", "decode", "both"), default="both")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse validated phase artifacts in an existing output directory",
    )
    return parser.parse_args()


def _load_export_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    model_config = dict(data["export"]["model"])
    return model_config


def main() -> None:
    args = _parse_args()
    config = _load_export_config(args.config)
    config.update(context_max_length=args.context_length, prefill_chunk_length=args.prefill_chunk_length)
    manifest = export_qwen4_exp_serving_package(
        model_dir=args.model_dir, output_dir=args.output_dir, model_config=config,
        device=args.device, device_map=args.device_map, model_name=args.model_name,
        phase=args.phase, resume=args.resume,
    )
    shutil.copy2(args.config, Path(args.output_dir) / "export_source_config.yaml")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
