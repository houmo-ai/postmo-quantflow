# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Inspect custom size YAMLs without loading weights, exporting graphs or using a GPU."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from xhmodel_merak.xh_other_model.models.qwen_image21.artifacts import DenoiseGear
from xhmodel_merak.xh_other_model.models.qwen_image21.build import validate_build_config
from xhmodel_merak.xh_other_model.models.qwen_image21.capacity import (
    DEFAULT_CAPACITY,
    derive_text_capacities,
    fit_reference_size,
)


BASE_CONFIG = Path(__file__).resolve().parents[3] / (
    "configs_merak/workflows/xh2a/other_models/qwen_image21/qwen_image21_turbo_1mp_w8.yaml"
)


def load_config(path):
    workflow = yaml.safe_load(Path(path).read_text())
    if not isinstance(workflow, dict) or workflow.get("quant") is not None:
        raise ValueError("expected a workflow mapping with quant: null")
    if not isinstance(workflow.get("export"), dict):
        raise ValueError("workflow requires an export mapping")
    return validate_build_config(workflow["export"])


def describe_capacity(config, processor, *, baseline=None, reference_input=None, transformer_config=None):
    config = validate_build_config(config)
    budget = derive_text_capacities(processor, config)
    if budget["feature_capacity"] > config["max_position"]:
        raise ValueError("max_position must cover the derived feature capacity")
    gear = DenoiseGear(
        budget["feature_capacity"],
        max(config["target_capacity"], config["reference_capacity"]),
        budget["prefix_capacity"],
        config["target_capacity"],
        budget["dit_text_capacity"],
    )
    # Shape/constant contracts only. The configured exporter currently rebuilds
    # all six graphs and validates the resulting cross-component metadata.
    boundaries = {
        "text_encode": [budget["feature_capacity"], config["max_position"]],
        "visual": [config["visual_patch_capacity"]],
        "vae_encode": list(config["canvas_size"]),
        "vae_decode": list(config["canvas_size"]),
        "denoise_text": [gear.text_tokens, gear.cache_tokens],
        "denoise_image": [gear.image_tokens, gear.cache_tokens],
    }
    plan = {
        "status": "configuration_checked_not_exported",
        "model_id": config.get("model_id", "qwen-image-2.1"),
        "deployment_config": {key: config[key] for key in DEFAULT_CAPACITY},
        "text_budget": budget,
        "dit_gear": {
            "image_tokens": gear.image_tokens,
            "cache_tokens": gear.cache_tokens,
            "target_tokens": gear.target_tokens,
            "text_tokens": gear.text_tokens,
        },
        "graph_boundaries": boundaries,
        "required_target_tokens": config["output_size"][0] * config["output_size"][1] // 256,
        "required_reference_tokens": config["reference_size"][0] * config["reference_size"][1] // 256,
    }
    if baseline is not None:
        previous = describe_capacity(baseline, processor)["graph_boundaries"]
        plan["changed_graph_boundaries"] = [role for role in boundaries if boundaries[role] != previous[role]]
    if reference_input is not None:
        fitted = fit_reference_size(*reference_input, config["canvas_size"], config["reference_capacity"])
        plan["reference_input"] = {
            "original_hw": list(reference_input),
            "fitted_hw": list(fitted),
            "resized": tuple(reference_input) != fitted,
        }
    geometry = transformer_config or {}
    keys = ("num_layers", "num_attention_heads", "attention_head_dim")
    if all(type(geometry.get(key)) is int and geometry[key] > 0 for key in keys):
        elements = 2 * geometry["num_layers"] * geometry["num_attention_heads"] * geometry["attention_head_dim"]
        plan["logical_fp16_kv_bytes_per_cfg_branch"] = elements * gear.cache_tokens * 2
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="Local checkpoint; only tokenizer/config files read"
    )
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--reference-input", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer

    processor = SimpleNamespace(
        tokenizer=AutoTokenizer.from_pretrained(
            args.model_dir / "processor",
            local_files_only=True,
            trust_remote_code=False,
        )
    )
    architecture_path = args.model_dir / "transformer/config.json"
    architecture = json.loads(architecture_path.read_text()) if architecture_path.is_file() else {}
    baseline = load_config(args.baseline_config)
    report = {"model_dir": str(args.model_dir.resolve()), "baseline_config": str(args.baseline_config), "plans": []}
    for path in args.config:
        plan = describe_capacity(
            load_config(path),
            processor,
            baseline=baseline,
            reference_input=args.reference_input,
            transformer_config=architecture,
        )
        report["plans"].append({"config": str(path), **plan})
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
