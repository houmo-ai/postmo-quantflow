"""Shared MiniMax-H3 export metadata checks for build and load."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


COMPONENTS = (
    "vae_encoder",
    "vae_decoder",
    "audio_vae_encoder",
    "audio_vae_decoder",
    "transformer",
    "vision_encoder",
    "text_encoder",
)
FileResolver = Callable[[Any, str], Path]


def load_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"MiniMax-H3 metadata must contain a JSON object: {path.name}")
    return data


def validate_runtime_contract(
    meta: Mapping[str, Any],
    *,
    components: Sequence[str],
    resolve_file: FileResolver,
) -> None:
    """Validate the exported component inventory used by build and inference."""
    if meta.get("model_type") != "XHMiniMaxH3Model":
        raise ValueError("MiniMax-H3 artifacts must declare model_type=XHMiniMaxH3Model")
    if meta.get("target_device") != "XH2a":
        raise ValueError("MiniMax-H3 artifacts must target XH2a")

    declared = meta.get("components")
    expected = tuple(components)
    if expected != COMPONENTS:
        raise ValueError("MiniMax-H3 public recipe requires the complete registered component set")
    if not isinstance(declared, Mapping) or set(declared) != set(expected):
        raise ValueError(
            "MiniMax-H3 Artifact component set does not match the Registry: "
            f"expected={list(expected)}, actual={sorted(declared) if isinstance(declared, Mapping) else declared}"
        )

    for name in expected:
        component = declared[name]
        if not isinstance(component, Mapping):
            raise ValueError(f"MiniMax-H3 component {name!r} must contain a metadata object")
        hmonnx = component.get("hmonnx_file")
        if not isinstance(hmonnx, str) or not hmonnx:
            raise ValueError(f"MiniMax-H3 component {name!r} is missing hmonnx_file")
        resolve_file(hmonnx, "hmonnx_file")
        input_names = component.get("input_names")
        inputs = component.get("inputs")
        if (
            not isinstance(input_names, list)
            or not input_names
            or any(not isinstance(item, str) or not item for item in input_names)
        ):
            raise ValueError(f"MiniMax-H3 component {name!r} must declare non-empty input_names")
        if not isinstance(inputs, list) or len(inputs) != len(input_names):
            raise ValueError(f"MiniMax-H3 component {name!r} inputs must align with input_names")
        for item in inputs:
            if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                raise ValueError(f"MiniMax-H3 component {name!r} inputs must contain named tensor specs")
            shape = item.get("shape")
            if not isinstance(shape, list) or any(type(dim) is not int or dim <= 0 for dim in shape):
                raise ValueError(f"MiniMax-H3 component {name!r} input shapes must be positive integers")


__all__ = ["COMPONENTS", "load_metadata", "validate_runtime_contract"]
