"""Qwen Recipe planning: validate family-owned source and visual contracts.

This module must stay lightweight: listing recipes never imports it, and
resolving a build only reads local JSON/YAML, without loading model runtimes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError, invalid_parameter
from xhmodel_merak.model_build.contracts import SourceRef
from xhmodel_merak.model_catalog.contracts import ModelRecipe, VariantRecipe


def resolve_config(
    *,
    recipe: ModelRecipe,
    variant: VariantRecipe,
    model_dir: Path,
    sources: Mapping[str, SourceRef],
    workflow: Mapping[str, Any],
    quant_checkpoint: str | None = None,
    precision=None,
) -> dict[str, Any]:
    """Bind the assistant checkpoint and reject drift in the Workflow visual ABI."""
    if recipe.vision_gears:
        _validate_visual_workflow(recipe, workflow)
    from .quantization import resolve_quantization
    replacements = resolve_quantization(recipe, variant, workflow, quant_checkpoint, precision)
    if variant.speculative_mode != "dflash":
        return replacements
    source = sources["dflash"]
    return {
        **replacements,
        "export.model.dflash_config.hf_model": source.path,
        "export.model.dflash_config.noise_token_id": _validate_dflash_source(recipe, source, model_dir),
    }


def _validate_visual_workflow(recipe: ModelRecipe, workflow: Mapping[str, Any]) -> None:
    export = workflow.get("export")
    model = export.get("model") if isinstance(export, Mapping) else None
    visual = model.get("visual_config") if isinstance(model, Mapping) else None
    expected = {
        "visual_input_mode": "patches",
        "image_token_gears": list(recipe.vision_gears),
        "image_token_capacity": max(recipe.vision_gears),
        "spatial_merge_size": 2,
    }
    mismatches = {}
    for field, value in expected.items():
        actual = visual.get(field) if isinstance(visual, Mapping) else None
        if actual != value or type(actual) is not type(value):
            mismatches[field] = {"expected": value, "actual": actual}
    if isinstance(visual, Mapping):
        for field in ("max_size_w", "max_size_h"):
            if field in visual:
                mismatches[field] = {"expected": "absent in patches mode", "actual": visual[field]}
    if mismatches:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Qwen Workflow visual contract does not match the registered Recipe",
            model=recipe.model_id,
            mismatches=mismatches,
        )


def _validate_dflash_source(recipe: ModelRecipe, source: SourceRef, target_dir: Path) -> int:
    config_path = Path(source.path) / "config.json"
    try:
        assistant_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise invalid_parameter(
            recipe.model_id,
            "sources.dflash",
            source.path,
            f"requires a readable assistant config.json: {exc}",
        ) from exc
    if not isinstance(assistant_config, Mapping):
        raise invalid_parameter(
            recipe.model_id,
            "sources.dflash",
            source.path,
            "assistant config.json must contain a mapping",
        )
    dflash_config = assistant_config.get("dflash_config")
    if not isinstance(dflash_config, Mapping):
        raise invalid_parameter(
            recipe.model_id,
            "sources.dflash",
            source.path,
            "assistant config.json must declare a dflash_config mapping",
        )

    mask_token_id = dflash_config.get("mask_token_id")
    noise_token_id = dflash_config.get("noise_token_id")
    if mask_token_id is not None and noise_token_id is not None and mask_token_id != noise_token_id:
        raise invalid_parameter(
            recipe.model_id,
            "sources.dflash.dflash_config",
            dict(dflash_config),
            "mask_token_id and noise_token_id must agree when both are present",
        )
    token_id = mask_token_id if mask_token_id is not None else noise_token_id
    if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
        raise invalid_parameter(
            recipe.model_id,
            "sources.dflash.dflash_config.mask_token_id",
            token_id,
            "must provide a non-negative integer mask_token_id or noise_token_id",
        )

    target_config_path = target_dir / "config.json"
    try:
        target_config = json.loads(target_config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise invalid_parameter(
            recipe.model_id,
            "model_dir",
            str(target_dir),
            f"requires a readable target config.json: {exc}",
        ) from exc
    if not isinstance(target_config, Mapping):
        raise invalid_parameter(
            recipe.model_id,
            "model_dir",
            str(target_dir),
            "target config.json must contain a mapping",
        )

    target_text_config = target_config.get("text_config")
    target_text = target_text_config if isinstance(target_text_config, Mapping) else target_config
    target_hidden_size = target_text.get("hidden_size", target_config.get("hidden_size"))
    target_num_layers = target_text.get("num_hidden_layers", target_config.get("num_hidden_layers"))
    target_vocab_size = target_text.get("vocab_size", target_config.get("vocab_size"))

    assistant_hidden_size = assistant_config.get("hidden_size")
    if (
        assistant_hidden_size is not None
        and target_hidden_size is not None
        and assistant_hidden_size != target_hidden_size
    ):
        raise invalid_parameter(
            recipe.model_id,
            "sources.dflash.hidden_size",
            assistant_hidden_size,
            f"must match target hidden_size={target_hidden_size!r}",
        )
    assistant_vocab_size = assistant_config.get("vocab_size")
    if assistant_vocab_size is not None and target_vocab_size is not None and assistant_vocab_size != target_vocab_size:
        raise invalid_parameter(
            recipe.model_id,
            "sources.dflash.vocab_size",
            assistant_vocab_size,
            f"must match target vocab_size={target_vocab_size!r}",
        )

    num_target_layers = assistant_config.get("num_target_layers")
    if num_target_layers is not None:
        if isinstance(num_target_layers, bool) or not isinstance(num_target_layers, int) or num_target_layers <= 0:
            raise invalid_parameter(
                recipe.model_id,
                "sources.dflash.num_target_layers",
                num_target_layers,
                "must be a positive integer",
            )
        if isinstance(target_num_layers, int) and num_target_layers != target_num_layers:
            raise invalid_parameter(
                recipe.model_id,
                "sources.dflash.num_target_layers",
                num_target_layers,
                f"must match target num_hidden_layers={target_num_layers!r}",
            )

    target_layer_ids = dflash_config.get("target_layer_ids")
    if target_layer_ids is not None:
        valid_target_layers = (
            isinstance(target_layer_ids, Sequence)
            and not isinstance(target_layer_ids, (str, bytes))
            and bool(target_layer_ids)
            and all(isinstance(layer, int) and not isinstance(layer, bool) and layer >= 0 for layer in target_layer_ids)
            and len(set(target_layer_ids)) == len(target_layer_ids)
        )
        layer_limit = num_target_layers if isinstance(num_target_layers, int) else target_num_layers
        if valid_target_layers and isinstance(layer_limit, int):
            valid_target_layers = all(layer < layer_limit for layer in target_layer_ids)
        if not valid_target_layers:
            raise invalid_parameter(
                recipe.model_id,
                "sources.dflash.dflash_config.target_layer_ids",
                target_layer_ids,
                "must be unique non-negative target layer IDs inside the target model",
            )

    target_identity = {
        "target_model_id": recipe.model_id,
        "target_model_name": recipe.export_name,
        "target_model_type": target_config.get("model_type"),
        "target_hidden_size": target_hidden_size,
        "target_num_hidden_layers": target_num_layers,
    }
    for field, expected in target_identity.items():
        value = dflash_config.get(field, assistant_config.get(field))
        if value is not None and expected is not None and value != expected:
            raise invalid_parameter(
                recipe.model_id,
                f"sources.dflash.{field}",
                value,
                f"must match target identity {expected!r}",
            )
    return token_id
