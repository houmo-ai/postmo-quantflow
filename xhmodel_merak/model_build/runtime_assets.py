from __future__ import annotations

import ast
import json
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
    "use_cache",
}


def explicit_rope_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a fully declared, unscaled legacy RoPE; never infer theta.

    Older HF configs use rope_scaling=null plus a scalar rope_theta. Newer
    configs serialize those same values under rope_parameters.
    """
    result = dict(config)
    theta = config.get("rope_theta")
    if (
        "rope_parameters" not in config
        and "rope_scaling" in config
        and config["rope_scaling"] is None
        and isinstance(theta, (int, float))
        and not isinstance(theta, bool)
        and theta > 0
    ):
        result.pop("rope_scaling")
        result.pop("rope_theta")
        rope = {"rope_type": "default", "rope_theta": theta}
        if "partial_rotary_factor" in config:
            rope["partial_rotary_factor"] = config["partial_rotary_factor"]
        result["rope_parameters"] = rope
    return result


def _same_explicit_derived_field(expected: Mapping, actual: Mapping, key: str, value: Any) -> bool:
    if key == "use_cache" and isinstance(value, bool):
        # A duplicate at the composite root is harmless only when both text
        # configs explicitly declare the same value. Text-level changes fail.
        return all(
            isinstance(config.get("text_config"), Mapping) and config["text_config"].get(key) is value
            for config in (expected, actual)
        )
    if key == "layer_types" and isinstance(value, list):
        # HF expands an explicitly supplied hybrid-attention interval into a
        # layer list. Require the complete pattern and both source parameters.
        for config in (expected, actual):
            count, interval = config.get("num_hidden_layers"), config.get("full_attention_interval")
            if type(count) is not int or type(interval) is not int or count <= 0 or interval <= 0:
                return False
            if len(value) != count or value != [
                "full_attention" if (index + 1) % interval == 0 else "linear_attention" for index in range(count)
            ]:
                return False
        return True
    return _same_explicit_rope_factor(expected, actual, key, value)


def _same_explicit_rope_factor(expected: Mapping, actual: Mapping, key: str, value: Any) -> bool:
    """Accept only a redundant scalar, not a missing or inferred RoPE default.

    HF may serialize partial_rotary_factor both at the text-config root and in
    rope_parameters. Both configs must explicitly agree on the nested value;
    a conflicting duplicate is still a behavioral mismatch.
    """
    if key != "partial_rotary_factor" or isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not 0 < value <= 1:
        return False
    for config in (expected, actual):
        rope = config.get("rope_parameters")
        if not isinstance(rope, Mapping) or isinstance(rope.get(key), bool) or rope.get(key) != value:
            return False
    return True


def behavior_config_mismatches(
    expected: Any,
    actual: Any,
    field: str = "",
    *,
    type_aliases: tuple[tuple[str, str, str], ...] = (),
) -> list[dict[str, Any]]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected, actual = explicit_rope_config(expected), explicit_rope_config(actual)
        mismatches: list[dict[str, Any]] = []
        for key, expected_value in expected.items():
            key_name = str(key)
            if key_name in _NON_BEHAVIORAL_CONFIG_FIELDS:
                continue
            nested_field = f"{field}.{key_name}" if field else key_name
            if key not in actual:
                if _same_explicit_derived_field(expected, actual, key_name, expected_value):
                    continue
                mismatches.append({"field": nested_field, "expected": expected_value, "actual": None})
                continue
            mismatches.extend(
                behavior_config_mismatches(expected_value, actual[key], nested_field, type_aliases=type_aliases)
            )
        for key, actual_value in actual.items():
            key_name = str(key)
            if (
                key in expected
                or key_name in _NON_BEHAVIORAL_CONFIG_FIELDS
                or key_name not in _CRITICAL_CHECKPOINT_ONLY_FIELDS
            ):
                continue
            nested_field = f"{field}.{key_name}" if field else key_name
            if _same_explicit_derived_field(expected, actual, key_name, actual_value):
                continue
            mismatches.append({"field": nested_field, "expected": None, "actual": actual_value})
        return mismatches
    if isinstance(expected, str) and isinstance(actual, str):
        for path, old, new in type_aliases:
            if field == path and {expected, actual}.issubset({old, new}):
                return []
    if expected != actual:
        return [{"field": field, "expected": expected, "actual": actual}]
    return []


def quant_runtime_asset_paths(path: Path) -> list[Path]:
    """Select HF configuration/tokenizer assets and their explicit local dependencies.

    A JSON or Python suffix alone does not make an evaluation report or export
    script a runtime dependency. Never import checkpoint code to discover files.
    """
    names = {
        "config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "tokenizer.tiktoken",
        "chat_template.json",
        "chat_template.jinja",
    }
    assets = {
        candidate
        for candidate in path.iterdir()
        if candidate.is_file()
        and (
            candidate.name in names
            or (candidate.name.endswith("_config.json") and not candidate.name.startswith(("train", "eval")))
        )
    }
    templates = path / "chat_templates"
    if templates.is_dir():
        assets.update(candidate for candidate in templates.rglob("*.jinja") if candidate.is_file())

    def add_relative(value: str, *, required: bool = False) -> None:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or "\\" in value:
            raise ValueError(f"Runtime asset reference must stay inside its HF directory: {value}")
        candidate = path / relative
        if candidate.is_file():
            assets.add(candidate)
        elif required:
            raise ValueError(f"Referenced runtime asset is missing: {value}")

    def visit_config(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit_config(item)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                if key == "auto_map" and isinstance(item, Mapping):
                    for targets in item.values():
                        for target in targets if isinstance(targets, list) else [targets]:
                            if isinstance(target, str):
                                module = target.split("--")[-1].rsplit(".", 1)[0]
                                add_relative(module.replace(".", "/") + ".py")
                elif key.endswith("_file") and isinstance(item, str) and item:
                    # Historical source-machine absolute paths are not copied.
                    if not Path(item).is_absolute():
                        add_relative(item, required=True)
                else:
                    visit_config(item)

    for candidate in sorted(assets):
        if candidate.name == "config.json" or candidate.name.endswith("_config.json"):
            visit_config(json.loads(candidate.read_text(encoding="utf-8")))
    visited: set[Path] = set()
    while pending := {candidate for candidate in assets - visited if candidate.suffix == ".py"}:
        for candidate in pending:
            visited.add(candidate)
            for node in ast.walk(ast.parse(candidate.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.level:
                    parent = candidate.parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    modules = [node.module] if node.module else [alias.name for alias in node.names]
                    for module in modules:
                        dependency = parent / (module.replace(".", "/") + ".py")
                        add_relative(dependency.relative_to(path).as_posix())
    return sorted(assets)


__all__ = ["behavior_config_mismatches", "explicit_rope_config", "quant_runtime_asset_paths"]
