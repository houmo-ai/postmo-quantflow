"""Lightweight validation for the local MOSS model bundle."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_catalog.contracts import ModelRecipe, VariantRecipe
from xhmodel_merak.model_build.contracts import SourceRef


_REQUIRED_FILES = (
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "configuration_moss_transcribe_diarize.py",
    "modeling_moss_transcribe_diarize.py",
    "processing_moss_transcribe_diarize.py",
)


def resolve_config(
    *,
    recipe: ModelRecipe,
    variant: VariantRecipe,
    model_dir: Path,
    sources: Mapping[str, SourceRef],
    workflow: Mapping[str, Any],
) -> dict[str, Any]:
    del variant, sources
    root = model_dir.resolve()
    try:
        config = _read_json(root / "config.json")
        processor = _read_json(root / "processor_config.json")
        preprocessor = _read_json(root / "preprocessor_config.json")
        generation = _read_json(root / "generation_config.json")
        index = _read_json(root / "model.safetensors.index.json")
        for name in _REQUIRED_FILES[3:]:
            _require_file(root / name, name)
        _validate_config(config)
        _validate_processor(processor)
        _validate_preprocessor(preprocessor)
        _validate_generation(generation)
        _validate_shards(root, index)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "MOSS model_dir is not a valid local model bundle",
            model=recipe.model_id,
            path=str(root),
            reason=str(error),
        ) from error
    return _workflow_length_replacements(workflow)


def _workflow_length_replacements(workflow: Mapping[str, Any]) -> dict[str, Any]:
    export = workflow.get("export")
    model = export.get("model") if isinstance(export, Mapping) else None
    if not isinstance(model, Mapping):
        return {}
    replacements: dict[str, Any] = {}
    value = model.get("max_pe_length")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        replacements["export.model.max_pe_length"] = value
    return replacements


def _read_json(path: Path) -> dict[str, Any]:
    _require_file(path, path.name)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing required model resource: {name}")


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("model_type") != "moss_transcribe_diarize":
        raise ValueError("config.json has an unexpected model_type")
    text = config.get("text_config")
    audio = config.get("audio_config")
    if not isinstance(text, Mapping) or text.get("model_type") != "qwen3":
        raise ValueError("config.json must contain a qwen3 text_config")
    if not isinstance(audio, Mapping) or audio.get("model_type") != "whisper":
        raise ValueError("config.json must contain a whisper audio_config")
    if config.get("audio_merge_size") != 4 or config.get("adaptor_input_dim") != 4096:
        raise ValueError("config.json has unsupported audio adaptor geometry")


def _validate_processor(processor: Mapping[str, Any]) -> None:
    expected = {
        "audio_tokens_per_second": 12.5,
        "audio_merge_size": 4,
        "time_marker_every_seconds": 5,
        "enable_time_marker": True,
    }
    for key, value in expected.items():
        if processor.get(key) != value:
            raise ValueError(f"processor_config.json has unsupported {key}")


def _validate_preprocessor(preprocessor: Mapping[str, Any]) -> None:
    expected = {
        "sampling_rate": 16000,
        "feature_size": 80,
        "hop_length": 160,
        "n_fft": 400,
        "n_samples": 480000,
        "nb_max_frames": 3000,
    }
    for key, value in expected.items():
        if preprocessor.get(key) != value:
            raise ValueError(f"preprocessor_config.json has unsupported {key}")


def _validate_generation(generation: Mapping[str, Any]) -> None:
    max_new_tokens = generation.get("max_new_tokens")
    eos_token_id = generation.get("eos_token_id")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ValueError("generation_config.json must define a positive integer max_new_tokens")
    if not isinstance(eos_token_id, int) or isinstance(eos_token_id, bool) or eos_token_id < 0:
        raise ValueError("generation_config.json must define a non-negative integer eos_token_id")


def _validate_shards(root: Path, index: Mapping[str, Any]) -> None:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")
    for shard in set(weight_map.values()):
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError("weight_map contains an invalid shard path")
        _require_file(root / shard, shard)


__all__ = ["resolve_config"]
