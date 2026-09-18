"""Map producer-owned static Qwen MoE precision plans to Merak node schemes."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path


def apply_checkpoint_precision(export_cfg, model_dir, *, required=False):
    """The checkpoint, including logical W5, is authoritative for searched nodes."""
    model_cfg = export_cfg["model"]
    if str(model_cfg.get("model_type", "")).endswith("_visual"):
        return export_cfg
    path = Path(model_dir) / "config.json"
    if not path.is_file() and not required:
        return export_cfg
    config = json.loads(path.read_text())
    quant = config.get("quantization_config", {})
    plan = quant.get("meta", {}).get("auto_mixed_precision")
    if plan is None:
        if required:
            raise ValueError("auto_mixed_precision was requested but the checkpoint has no precision plan; "
                             "run quantization before exporting this model")
        return export_cfg
    # Validation of producer metadata and packed tensors belongs to GPTQModel.
    from gptqmodel.quantization.auto_mixed_precision import read_precision_plan
    plan = read_precision_plan(model_dir)
    result = copy.deepcopy(export_cfg)
    scheme = result["model"].setdefault("quant_scheme", {})
    quant_type = scheme.get("quant_type", "")
    if not re.fullmatch(r"w[458]a\d+h\d+_sefp", quant_type):
        raise ValueError("Qwen automatic mixed precision export requires an integer SEFP quant_type")
    nodes = scheme.setdefault("nodes", {})
    for name, bits in plan["module_bits"].items():
        export_names = [name]
        if model_cfg.get("split_conv_cache", True) and name.endswith(".linear_attn.in_proj_qkv"):
            prefix = name.removesuffix("in_proj_qkv")
            export_names.extend(prefix + f"in_proj_{part}" for part in ("q", "k", "v"))
        # Eager paths use dotted module names; FX paths use normalized names.
        for node in (alias for exported in export_names for alias in (exported, exported.replace(".", "_"))):
            existing = nodes.get(node, {})
            if isinstance(existing, str):
                existing = {"quant_type": existing}
            expected = re.sub(r"^w\d+", f"w{bits}", existing.get("quant_type", quant_type))
            if (("quant_type" in existing and existing["quant_type"] != expected)
                    or any(existing.get(key) for key in ("w_scheme", "w_schema", "w_cfg"))):
                raise ValueError(f"Export precision conflicts with checkpoint plan at {node}: W{bits}")
            nodes[node] = {**existing, "quant_type": expected}
    return result
