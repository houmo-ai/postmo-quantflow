from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


_NON_BEHAVIORAL_CONFIG_FIELDS = {
    "_commit_hash",
    "_name_or_path",
    "_transformers_version",
    "compression_config",
    "dtype",
    "gptqmodel_version",
    "quantization_config",
    "quantization_status",
    "quantize_config",
    "quantizer_config",
    "torch_dtype",
    "transformers_version",
}
_CRITICAL_CHECKPOINT_ONLY_FIELDS = {
    "head_dim",
    "layer_types",
    "max_position_embeddings",
    "partial_rotary_factor",
    "rope_scaling",
    "rope_theta",
    "rope_type",
    "sliding_window",
}


def behavior_config_mismatches(
    expected: Any,
    actual: Any,
    field: str = "",
) -> list[dict[str, Any]]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        mismatches: list[dict[str, Any]] = []
        for key, expected_value in expected.items():
            key_name = str(key)
            if key_name in _NON_BEHAVIORAL_CONFIG_FIELDS:
                continue
            nested_field = f"{field}.{key_name}" if field else key_name
            if key not in actual:
                mismatches.append({"field": nested_field, "expected": expected_value, "actual": None})
                continue
            mismatches.extend(behavior_config_mismatches(expected_value, actual[key], nested_field))
        for key, actual_value in actual.items():
            key_name = str(key)
            if (
                key in expected
                or key_name in _NON_BEHAVIORAL_CONFIG_FIELDS
                or key_name not in _CRITICAL_CHECKPOINT_ONLY_FIELDS
            ):
                continue
            nested_field = f"{field}.{key_name}" if field else key_name
            mismatches.append({"field": nested_field, "expected": None, "actual": actual_value})
        return mismatches
    if expected != actual:
        return [{"field": field, "expected": expected, "actual": actual}]
    return []


def quant_runtime_asset_paths(path: Path) -> list[Path]:
    excluded_names = {
        "all_results.json",
        "trainer_state.json",
        "training_args.json",
    }
    runtime_suffixes = {
        ".jinja",
        ".json",
        ".merges",
        ".model",
        ".npy",
        ".npz",
        ".py",
        ".tiktoken",
        ".txt",
        ".vocab",
    }
    runtime_markers = ("chat_template", "feature_extractor", "preprocessor", "processor", "tokenizer")
    assets: list[Path] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        name = candidate.name.lower()
        relative = candidate.relative_to(path)
        if (
            name in excluded_names
            or "provenance" in name
            or name.endswith((".safetensors.index.json", ".bin.index.json"))
            or candidate.suffix.lower() in {".bin", ".gguf", ".log", ".safetensors", ".yaml", ".yml"}
        ):
            continue
        runtime_named = any(marker in candidate.name.lower() for marker in runtime_markers)
        runtime_directory = any(
            any(marker in part.lower() for marker in runtime_markers) for part in relative.parts[:-1]
        )
        if candidate.suffix.lower() in runtime_suffixes or runtime_named or runtime_directory:
            assets.append(candidate)
    return assets


__all__ = ["behavior_config_mismatches", "quant_runtime_asset_paths"]
