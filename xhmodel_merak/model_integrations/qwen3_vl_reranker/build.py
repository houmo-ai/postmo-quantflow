"""Package the existing LLM export while retaining only its actual components."""

import json

from xhmodel_merak.model_build.artifacts import flatten_single_export_root
from xhmodel_merak.model_integrations.autoregressive.build import package_runtime, sanitize_artifact

from .artifact import validate_score_contract


def normalize_export(stage_path, export_result, plan):
    del export_result, plan
    previous_name = flatten_single_export_root(stage_path)
    metadata = stage_path / "golden_meta_info.json"
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    visual = meta["visual_config"]
    if visual.get("visual_input_mode") == "patches":
        return previous_name
    config = meta["model_config"]["visual_config"]
    name = f"visual_{config['max_size_w']}x{config['max_size_h']}x{config['max_size_t']}"
    old = stage_path / "visual"
    old.rename(stage_path / name)
    visual["hmonnx"] = name + "/" + visual["hmonnx"].split("/", 1)[1]
    metadata.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return previous_name


def validate_artifact(stage_path, plan):
    metadata = stage_path / "golden_meta_info.json"
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    validate_score_contract(stage_path, meta)
    for key, expected in [
        ("context_max_length", plan.context_length),
        ("prefill_chunk_length", plan.prefill_chunk_length),
        ("max_pe_length", plan.max_pe_length),
    ]:
        if int(meta["model_config"][key]) != int(expected):
            raise ValueError(f"Reranker metadata does not match planned {key}")
    entries = {
        "prefill": {"path": meta["prefill_hmonnx"], "type": "prefill"},
        "visual": {"path": meta["visual_config"]["hmonnx"], "type": "hmonnx"},
        "tokenizer": {"path": meta["hf_config"], "type": "tokenizer"},
        "quant_embedding": {"path": meta["quant_embedding"], "type": "embedding"},
    }
    if plan.vision_gears:
        from xhmodel_merak.model_integrations.autoregressive.vision import validate_dynamic_vision

        vision = validate_dynamic_vision(
            stage_path,
            meta["visual_config"],
            model_id=plan.model_id,
            vision_profile=plan.vision_profile,
            vision_gears=plan.vision_gears,
        )
        entries.pop("visual")
        entries["visual_gears"] = {"path": vision["gear_manifest"], "type": "json"}
        for gear in vision["gears"]:
            entries[f"visual_m{gear['image_token_capacity']}"] = {"path": gear["hmonnx"], "type": "hmonnx"}
    return {
        "metadata_file": metadata.name,
        "golden_meta_info": metadata.name,
        "components": list(plan.components),
        "golden_components": [],
        "golden_artifacts": {},
        "requires_source": False,
        "loadable": True,
        "export_entries": entries,
    }


__all__ = ["normalize_export", "package_runtime", "sanitize_artifact", "validate_artifact"]
