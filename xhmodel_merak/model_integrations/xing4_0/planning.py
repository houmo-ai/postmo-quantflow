"""Validate Xing's redundant native head dimension before creating a workflow."""

import json

from xhmodel_merak.errors import ErrorCode, ModelError


def resolve_config(*, recipe, model_dir, quant_checkpoint=None, **kwargs):
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if "head_dim" in config and (
        "qk_rope_head_dim" not in config or config["head_dim"] != config["qk_rope_head_dim"]
    ):
        raise ModelError(
            ErrorCode.INVALID_PARAMETER,
            "Xing head_dim must equal the explicit qk_rope_head_dim",
            model=recipe.model_id,
            mismatches=[{
                "field": "head_dim", "expected": config.get("qk_rope_head_dim"), "actual": config["head_dim"],
            }],
        )
    if quant_checkpoint is not None:
        from .checkpoint import inspect_quant_regions

        inspect_quant_regions(quant_checkpoint, model_id=recipe.model_id)
    return {}
