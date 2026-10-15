# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Validate local build inputs during planning, without loading weights."""

import json

from xhmodel_merak.errors import invalid_parameter


def resolve_config(*, recipe, model_dir, precision=None, **_):
    from xhmodel_merak.xh_other_model.models.sol_h3.assets import SourceAssets

    try:
        assets = SourceAssets.read(model_dir)
        if precision is not None and precision.weight != "w16":
            validate_checkpoint_source_config(assets)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise invalid_parameter(recipe.model_id, "model_dir", str(model_dir), str(error)) from error
    # Explicit W16 checkpoint builds remain the W8/W16 graph reference path.
    if precision is not None and precision.weight == "w16":
        return {"export.conditioner_quant_type": None}
    return {}


def validate_checkpoint_source_config(assets):
    """The inventory exposes its quantized component through the standard config view."""
    path = assets.root / "config.json"
    component = assets["h3_model"] / "text_encoder/config.json"
    if not path.is_file():
        raise ValueError(f"source inventory requires a config.json copied from {component}")
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or config != json.loads(component.read_text()):
        raise ValueError("source inventory config.json must match the H3 Qwen3-VL component config")


def normalize_checkpoint_config(config, *, source=False):
    # Transformers 5 moves the same explicit M-RoPE values into rope_parameters.
    text = config["text_config"]
    if "rope_parameters" not in text and isinstance(text.get("rope_scaling"), dict):
        rope = text.pop("rope_scaling")
        if "rope_theta" in text:
            rope["rope_theta"] = text.pop("rope_theta")
        text["rope_parameters"] = rope
    vision = config.get("vision_config", {})
    if vision.get("model_type") == "qwen3_vl":
        vision["model_type"] = "qwen3_vl_vision"
    return config
