"""Root-metadata contract shared by MiniCPM-V-4.7 build and load."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

FileResolver = Callable[[Any, str], Path]

_DENSE_MODEL_TYPE = "MiniCPMV47ForConditionalGeneration"
_MOE_MODEL_TYPE = "MiniCPMV47MoeForConditionalGeneration"
_SUPPORTED_MERAK_TYPES = {_DENSE_MODEL_TYPE, _MOE_MODEL_TYPE}
_VISION_MODES = ("4x", "16x")
_LLM_FILE_FIELDS = (
    "metadata",
    "prefill_hmonnx",
    "decode_hmonnx",
    "quant_embedding",
)


def load_metadata(path: Path) -> dict[str, Any]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"MiniCPM metadata must contain a JSON object: {path.name}")
    return data


def validate_runtime_contract(
    meta: Mapping[str, Any],
    *,
    components: Sequence[str],
    resolve_file: FileResolver,
) -> None:
    """Shared build/load checks for MiniCPM root export metadata."""

    from xhmodel_merak.xh_llm.minicpm_v_series.vision_gears import vision_profile_entries

    if meta.get("model_type") != "MiniCPM-V-4.7":
        raise ValueError("MiniCPM artifact model_type must be MiniCPM-V-4.7")
    merak_model_type = meta.get("merak_model_type")
    if merak_model_type not in _SUPPORTED_MERAK_TYPES:
        raise ValueError(
            "MiniCPM artifact merak_model_type must be "
            f"{_DENSE_MODEL_TYPE!r} or {_MOE_MODEL_TYPE!r}"
        )
    declared = meta.get("components")
    if not isinstance(declared, list) or not declared:
        raise ValueError("MiniCPM artifact must declare a non-empty components list")
    if any(not isinstance(name, str) or not name for name in declared):
        raise ValueError("MiniCPM components must be non-empty strings")
    def normalize(names):
        return tuple("dspark" if name == "dflash" else name for name in names)

    if normalize(components) != normalize(declared):
        raise ValueError(
            "MiniCPM recipe components do not match export metadata: "
            f"recipe={list(components)!r} export={declared!r}"
        )

    vision = meta.get("vision")
    if not isinstance(vision, Mapping):
        raise ValueError("MiniCPM artifact requires a vision metadata object")
    for mode in _VISION_MODES:
        component_name = f"vision_{mode}"
        if component_name not in declared:
            continue
        profile = vision.get(mode)
        if not isinstance(profile, Mapping):
            raise ValueError(f"MiniCPM vision profile {mode!r} is missing")
        if profile.get("downsample_mode") != mode:
            raise ValueError(f"MiniCPM vision profile {mode!r} has mismatched downsample_mode")
        for index, entry in enumerate(vision_profile_entries(profile, mode)):
            resolve_file(entry["hmonnx"], f"vision.{mode}.gears.{index}.hmonnx")
        if "gears" in profile:
            manifest_path = resolve_file(profile.get("gear_manifest"), f"vision.{mode}.gear_manifest")
            if load_metadata(manifest_path) != profile:
                raise ValueError(f"MiniCPM vision {mode} gear manifest differs from runtime metadata")

    if "llm" in declared:
        llm = meta.get("llm")
        if not isinstance(llm, Mapping):
            raise ValueError("MiniCPM artifact requires llm metadata when llm is enabled")
        for field in _LLM_FILE_FIELDS:
            resolve_file(llm.get(field), f"llm.{field}")
        if "les" in llm:
            from xhmodel_merak.xh_llm.models.minicpm_v_4_7.les_graph import validate_les_contract

            validate_les_contract(llm["les"])
            if merak_model_type != _MOE_MODEL_TYPE or "dspark" in normalize(declared):
                raise ValueError("LES metadata requires MiniCPM MoE ordinary decode")
            child = load_metadata(resolve_file(llm["metadata"], "llm.metadata"))
            if child.get("les") != llm["les"]:
                raise ValueError("MiniCPM root and child LES metadata differ")
        hf_config = llm.get("hf_config")
        if not isinstance(hf_config, str) or not hf_config:
            raise ValueError("MiniCPM llm.hf_config must be a relative directory path")
        # Packaged HF config is a directory; validate through its config.json file.
        resolve_file(f"{hf_config.rstrip('/')}/config.json", "llm.hf_config/config.json")
        if "dspark" in normalize(declared):
            spec = llm.get("spec_decode")
            if not isinstance(spec, Mapping) or spec.get("mode") not in {"dspark", "dflash"}:
                raise ValueError("MiniCPM DSpark component requires a speculative decoding contract")
            draft = spec.get("draft")
            if not isinstance(draft, Mapping):
                raise ValueError("MiniCPM DSpark contract requires draft graphs")
            for field in ("context_hmonnx", "context_decode_hmonnx", "decode_hmonnx"):
                resolve_file(draft.get(field), f"llm.spec_decode.draft.{field}")
            if type(spec.get("num_draft_tokens")) is not int or spec["num_draft_tokens"] <= 0:
                raise ValueError("MiniCPM DSpark requires positive num_draft_tokens")
            if spec.get("verify_length") != spec["num_draft_tokens"] + 1:
                raise ValueError("MiniCPM DSpark verify_length must include the anchor token")
            if type(draft.get("noise_token_id")) is not int or draft["noise_token_id"] < 0:
                raise ValueError("MiniCPM DSpark requires noise_token_id")
            if spec.get("enable_confidence_truncation", False):
                graphs = spec.get("verify_graphs") or {}
                if set(graphs) != {str(i) for i in range(1, spec["verify_length"] + 1)}:
                    raise ValueError("DSpark confidence truncation requires every exact verify length")
                for length, path in graphs.items():
                    resolve_file(path, f"llm.spec_decode.verify_graphs.{length}")
        for field in ("context_max_length", "prefill_chunk_length"):
            value = llm.get(field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"MiniCPM llm.{field} must be a positive integer")


def describe_export(meta: Mapping[str, Any]) -> dict[str, Any]:
    from xhmodel_merak.xh_llm.minicpm_v_series.vision_gears import vision_profile_entries

    entries: dict[str, Any] = {}
    vision = meta.get("vision") or {}
    if isinstance(vision, Mapping):
        for mode, profile in vision.items():
            if not isinstance(profile, Mapping):
                continue
            for entry in vision_profile_entries(profile, mode):
                name = f"visual_{mode}_m{entry['image_token_capacity']}" if "gears" in profile else f"vision_{mode}"
                entries[name] = {"path": entry["hmonnx"], "type": "hmonnx"}
            if "gears" in profile:
                entries[f"visual_{mode}_gears"] = {"path": profile["gear_manifest"], "type": "json"}
    llm = meta.get("llm")
    if isinstance(llm, Mapping):
        if isinstance(llm.get("prefill_hmonnx"), str):
            entries["prefill"] = {"path": llm["prefill_hmonnx"], "type": "prefill"}
        if isinstance(llm.get("decode_hmonnx"), str):
            entries["decode"] = {"path": llm["decode_hmonnx"], "type": "decode"}
        if isinstance(llm.get("hf_config"), str):
            entries["tokenizer"] = {"path": llm["hf_config"], "type": "tokenizer"}
    for name in ("context", "context_decode", "decode"):
        draft = (llm or {}).get("spec_decode", {}).get("draft", {})
        if isinstance(draft.get(f"{name}_hmonnx"), str):
            entries[f"dspark_{name}"] = {"path": draft[f"{name}_hmonnx"], "type": "hmonnx"}
    for length, path in (llm or {}).get("spec_decode", {}).get("verify_graphs", {}).items():
        entries[f"verify_{length}"] = {"path": path, "type": "decode"}
    if meta.get("hf_model"):
        entries["processor"] = {"path": meta["hf_model"], "type": "tokenizer"}
    return entries


__all__ = ["describe_export", "load_metadata", "validate_runtime_contract"]
