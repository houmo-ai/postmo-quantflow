"""Lightweight source binding for Gemma4's separate, shared-KV assistant."""

import json
from pathlib import Path

from xhmodel_merak.errors import invalid_parameter


def resolve_config(*, recipe, variant, model_dir, sources, workflow, quant_checkpoint=None, precision=None):
    if variant.speculative_mode != "mtp":
        return {}
    assistant = sources["mtp"].path
    target = json.loads((model_dir / "config.json").read_text())
    draft = json.loads((Path(assistant) / "config.json").read_text())
    text, draft_text = target["text_config"], draft["text_config"]
    expected = {
        "model_type": target["model_type"] + "_assistant",
        "backbone_hidden_size": text["hidden_size"],
    }
    mismatch = {key: (draft.get(key), value) for key, value in expected.items() if draft.get(key) != value}
    for key in ("vocab_size", "num_key_value_heads", "head_dim"):
        if draft_text.get(key) != text.get(key):
            mismatch[key] = (draft_text.get(key), text.get(key))
    for key, fallback in (("num_global_key_value_heads", "num_key_value_heads"), ("global_head_dim", "head_dim")):
        if (draft_text.get(key) or draft_text.get(fallback)) != (text.get(key) or text.get(fallback)):
            mismatch[key] = (draft_text.get(key), text.get(key))
    layers = draft_text.get("layer_types", [])
    n = draft_text.get("num_hidden_layers", 0)
    if not n or len(layers) != n or draft_text.get("num_kv_shared_layers") != n:
        mismatch["layers"] = "assistant must share KV for every layer"
    if not set(layers) <= set(text.get("layer_types", [])):
        mismatch["layer_types"] = "assistant layer types are absent from target"
    if mismatch:
        raise invalid_parameter(recipe.model_id, "sources.mtp", assistant, f"incompatible assistant: {mismatch}")
    return {
        "export.model.mtp_config.assistant_hf_model": assistant,
        "export.model.mtp_config.target_hf_model": str(model_dir),
    }
