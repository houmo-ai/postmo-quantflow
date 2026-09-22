"""Qwen recipe defaults and read-only checkpoint reuse (no runtime imports)."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path


def resolve_quantization(recipe, variant, workflow, checkpoint, precision):
    # Macaron/Ornith have independently qualified precision policies.
    if not recipe.model_id.startswith(("qwen3.5-", "qwen3.6-", "qwen3.8-")):
        return {}
    quant = copy.deepcopy(workflow.get("quant") or {})
    scheme = copy.deepcopy(workflow["export"]["model"]["quant_scheme"])
    weight = precision.weight if precision is not None else recipe.default_precision.weight
    bits = 5 if weight != "w16" else 8
    if checkpoint and not (Path(checkpoint) / "config.json").is_file():
        # The common artifact inspector owns the missing-model-config error.
        return {}
    if checkpoint:
        config = json.loads((Path(checkpoint) / "config.json").read_text())
        qconfig = config.get("quantization_config", {})
        if not qconfig:
            for name in ("quantize_config.json", "quant_config.json", "quantization_config.json"):
                path = Path(checkpoint) / name
                if path.is_file():
                    qconfig = json.loads(path.read_text())
                    break
        head = qconfig.get("meta", {}).get("lm_head_quantization")
        # Legacy floating heads retain the established W8 export policy.
        bits = head["bits"] if head else 8
        if qconfig.get("lm_head") and not head:
            raise ValueError("Reused packed head needs lm_head_quantization metadata; no implicit recalibration")
        if bits not in (4, 5, 8):
            raise ValueError("Unsupported checkpoint head precision")
        if variant.speculative_mode == "mtp" and not config.get("mtp_quantization"):
            raise ValueError("MTP checkpoint is incomplete; calibrate MTP before unified checkpoint reuse")
    if variant.requires_lora and variant.speculative_mode and not checkpoint:
        raise ValueError("LoRA + MTP requires quant_checkpoint; build the reusable base first")
    node = scheme.setdefault("nodes", {}).setdefault("lm_head", {})
    node["quant_type"] = re.sub(r"^w\d+", f"w{bits}", node.get("quant_type", scheme["quant_type"]))
    quant["head"] = {"bits": bits}
    quant.pop("auto_mixed_precision", None)
    if weight == "wmix":
        quant["auto_mixed_precision"] = {
            "candidates": [4, 5, 8], "initial_bits": 5, "max_kl": 0.03,
            "max_ppl_increase": 0.02, "nsamples": 32, "seqlen": 512,
            "positions_per_sample": 64, "max_trials": 24, "seed": 42,
            "image_dataset": "xh2modelzoo://data/calib_data/realworldqa_search_64.jsonl",
            "image_nsamples": 64, "image_seqlen": 2048, "image_max_pixels": 393216,
        }
    quant.pop("mtp", None)
    if variant.speculative_mode == "mtp" and weight != "w16":
        quant["mtp"] = ({"expert_bits": 4} if recipe.family == "qwen3.5-moe" else {"block_bits": 4})
        quant["mtp"].update(algorithm="inherit", group_size=64)
    result = {"quant": quant, "export.model.quant_scheme": scheme}
    if variant.requires_lora:
        result["export.model.lora"] = {
            "path": [], "w_schema": {"bits": 8, "fp_mode": "sefp"},
            "a_schema": {"bits": 16, "fp_mode": "sefp"},
        }
    return result
