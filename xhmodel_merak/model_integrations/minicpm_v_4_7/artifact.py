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
    if tuple(components) != tuple(declared):
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
        if type(profile.get("token_capacity")) is not int or profile["token_capacity"] <= 0:
            raise ValueError(f"MiniCPM vision profile {mode!r} requires a positive token_capacity")
        resolve_file(profile.get("hmonnx"), f"vision.{mode}.hmonnx")

    if "llm" in declared:
        llm = meta.get("llm")
        if not isinstance(llm, Mapping):
            raise ValueError("MiniCPM artifact requires llm metadata when llm is enabled")
        for field in _LLM_FILE_FIELDS:
            resolve_file(llm.get(field), f"llm.{field}")
        hf_config = llm.get("hf_config")
        if not isinstance(hf_config, str) or not hf_config:
            raise ValueError("MiniCPM llm.hf_config must be a relative directory path")
        # Packaged HF config is a directory; validate through its config.json file.
        resolve_file(f"{hf_config.rstrip('/')}/config.json", "llm.hf_config/config.json")
        for field in ("context_max_length", "prefill_chunk_length"):
            value = llm.get(field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"MiniCPM llm.{field} must be a positive integer")


def describe_export(meta: Mapping[str, Any]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    vision = meta.get("vision") or {}
    if isinstance(vision, Mapping):
        for mode, profile in vision.items():
            if isinstance(profile, Mapping) and isinstance(profile.get("hmonnx"), str):
                entries[f"vision_{mode}"] = {"path": profile["hmonnx"], "type": "hmonnx"}
    llm = meta.get("llm")
    if isinstance(llm, Mapping):
        if isinstance(llm.get("prefill_hmonnx"), str):
            entries["prefill"] = {"path": llm["prefill_hmonnx"], "type": "prefill"}
        if isinstance(llm.get("decode_hmonnx"), str):
            entries["decode"] = {"path": llm["decode_hmonnx"], "type": "decode"}
        if isinstance(llm.get("hf_config"), str):
            entries["tokenizer"] = {"path": llm["hf_config"], "type": "tokenizer"}
    return entries


__all__ = ["describe_export", "load_metadata", "validate_runtime_contract"]
