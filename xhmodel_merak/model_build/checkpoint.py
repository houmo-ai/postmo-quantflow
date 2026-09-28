"""Verify externally produced quantized checkpoints without executing pickle code."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import file_sha256
from xhmodel_merak.model_catalog.identity import source_constraint_mismatches
from xhmodel_merak.model_catalog.registry import get_recipe

from . import workspace
from .contracts import BuildPlan
from .runtime_assets import (
    behavior_config_mismatches as _behavior_config_mismatches,
)
from .runtime_assets import explicit_rope_config
from .runtime_assets import (
    quant_runtime_asset_paths as _quant_runtime_asset_paths,
)


_QUANT_METADATA_NAMES = (
    "quantization_config.json",
    "quantize_config.json",
    "quant_config.json",
    "config.json",
    "qwen35_autoround_recipe_provenance.json",
    "gemma4_quant_recipe_provenance.json",
    "ling3_flash_quant_recipe_provenance.json",
    "paddleocr_vl_recipe_result.json",
)


def _read_quant_metadata(path: Path, model: str) -> list[tuple[Path, Mapping[str, Any]]]:
    if not path.is_dir():
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            f"Quant Artifact is not an existing directory: {path}",
            model=model,
            path=str(path),
        )
    metadata_files = [candidate for name in _QUANT_METADATA_NAMES if (candidate := path / name).is_file()]
    if not metadata_files:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Quant Artifact contains no supported quantization metadata",
            model=model,
            path=str(path),
        )

    result = []
    for metadata_path in metadata_files:
        if metadata_path.is_symlink():
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                f"Quant Artifact metadata must not be a symlink: {metadata_path}",
                model=model,
                path=str(metadata_path),
            )
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                f"Cannot parse Quant Artifact metadata {metadata_path}: {exc}",
                model=model,
                path=str(metadata_path),
            ) from exc
        if not isinstance(payload, Mapping):
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                f"Quant Artifact metadata must contain an object: {metadata_path}",
                model=model,
                path=str(metadata_path),
            )
        result.append((metadata_path, payload))
    return result


def detect_checkpoint_method(path_value: str, *, model: str) -> str:
    """Read provider evidence only; tensor/identity verification still runs before export."""
    providers = {
        provider
        for _, payload in _read_quant_metadata(Path(path_value).expanduser().resolve(), model)
        for provider in _find_quant_providers(payload)
    }
    return _checkpoint_method(providers, model=model, path=path_value)


def checkpoint_has_mixed_precision(path_value: str, *, model: str) -> bool:
    """Read the search declaration from embedded or sidecar quant metadata."""
    for _, payload in _read_quant_metadata(Path(path_value).expanduser().resolve(), model):
        config = payload.get("quantization_config", payload)
        if isinstance(config, Mapping):
            meta = config.get("meta")
            if isinstance(meta, Mapping) and meta.get("auto_mixed_precision"):
                return True
    return False


def _checkpoint_method(providers: set[str], *, model: str, path: str) -> str:
    providers = {_normalize_quant_provider(provider) for provider in providers}
    methods = set()
    if providers & {"autoround", "auto-round"}:
        # AutoRound commonly saves GPTQ-compatible weights and declares both.
        methods.add("autoround")
    elif providers & {"gptq", "gptqmodel"}:
        methods.add("gptq")
    if "qat" in providers:
        methods.add("qat")
    if len(methods) != 1:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Quant Artifact must declare one supported checkpoint producer",
            model=model,
            path=path,
            providers=sorted(providers),
        )
    return methods.pop()


def inspect_quant_artifact(path_value: str | None, plan: BuildPlan) -> dict[str, Any] | None:
    if path_value is None or plan.quant_method is None:
        return None
    path = Path(path_value).resolve()
    named_payloads = _read_quant_metadata(path, plan.model_id)
    payloads = [payload for _, payload in named_payloads]
    metadata: list[dict[str, Any]] = []
    base_bit_values: set[int] = set()
    discovered_bit_values: set[int] = set()
    providers: set[str] = set()
    for metadata_path, payload in named_payloads:
        metadata.append(
            {
                "file": metadata_path.name,
                "size": metadata_path.stat().st_size,
                "sha256": file_sha256(metadata_path),
            }
        )
        providers.update(_find_quant_providers(payload))
        discovered_bit_values.update(_find_all_quant_bits(payload))
        base_bits = _find_base_quant_bits(payload)
        if base_bits is not None:
            base_bit_values.add(base_bits)
        command_base_bits, command_regions = _command_quant_evidence(payload)
        base_bit_values.update(command_base_bits)
        discovered_bit_values.update(command_base_bits)
        for values in command_regions.values():
            discovered_bit_values.update(values)

    if len(base_bit_values) > 1:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Quant Artifact declares conflicting default weight precisions",
            model=plan.model_id,
            path=str(path),
            actual_bits=sorted(base_bit_values),
        )
    discovered_bits = next(iter(base_bit_values), None)
    if not discovered_bit_values:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Quant Artifact metadata does not declare checkpoint weight precision",
            model=plan.model_id,
            path=str(path),
        )

    expected_bits = (
        int(plan.precision.weight.removeprefix("w")) if re.fullmatch(r"w\d+", plan.precision.weight) else None
    )
    if expected_bits is not None and discovered_bits != expected_bits:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            f"Quant Artifact weight precision {discovered_bits!r} does not match requested W{expected_bits}",
            model=plan.model_id,
            path=str(path),
            expected_bits=expected_bits,
            actual_bits=discovered_bits,
        )
    region_bits = _verify_quant_regions(payloads, plan, discovered_bits)
    method = _checkpoint_method(providers, model=plan.model_id, path=str(path))
    if method != plan.quant_method:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Quant Artifact checkpoint producer does not match requested quant_method",
            model=plan.model_id,
            expected=plan.quant_method,
            actual=method,
            providers=sorted(providers),
        )

    verify_quant_model_identity(path, named_payloads, plan)
    weight_artifacts = _inspect_quant_weight_files(path, plan)
    runtime_assets = [
        {
            "file": candidate.relative_to(path).as_posix(),
            "size": candidate.stat().st_size,
            "sha256": file_sha256(candidate),
        }
        for candidate in _quant_runtime_asset_paths(path)
    ]
    return {
        "method": plan.quant_method,
        "weight_bits": discovered_bits,
        "weight_bits_present": sorted(discovered_bit_values),
        "region_weight_bits": region_bits,
        "providers": sorted(providers),
        "metadata": metadata,
        "runtime_assets": runtime_assets,
        **weight_artifacts,
        "verification": "checkpoint_verified",
    }


def verify_quant_model_identity(
    artifact_path: Path,
    payloads: list[tuple[Path, Mapping[str, Any]]],
    plan: BuildPlan,
) -> None:
    checkpoint_configs = [payload for path, payload in payloads if path.name == "config.json"]
    source_config_path = Path(plan.model_dir) / "config.json"
    if len(checkpoint_configs) != 1:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Quant Artifact must contain one loadable model config.json",
            model=plan.model_id,
            path=str(artifact_path),
        )
    try:
        source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            f"Cannot read source model identity for Quant Artifact validation: {exc}",
            model=plan.model_id,
            path=str(source_config_path),
        ) from exc
    if not isinstance(source_config, Mapping):
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Source model config.json must contain an object",
            model=plan.model_id,
            path=str(source_config_path),
        )

    comparison_source = (
        _canonicalize_legacy_paddleocr_vl_config(source_config)
        if plan.model_id == "paddleocr-vl-1.6"
        else source_config
    )
    expected = _model_config_identity(comparison_source)
    if not expected:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Source model config.json exposes no comparable model identity",
            model=plan.model_id,
            path=str(source_config_path),
        )
    for checkpoint_config in checkpoint_configs:
        actual = _model_config_identity(checkpoint_config)
        mismatches = {
            field: {"expected": expected_value, "actual": actual.get(field)}
            for field, expected_value in expected.items()
            if actual.get(field) != expected_value
        }
        if mismatches:
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                "Quant Artifact model config does not match the source model config",
                model=plan.model_id,
                path=str(artifact_path),
                identity_mismatches=mismatches,
            )
        constraint_mismatches = source_constraint_mismatches(get_recipe(plan.model_id), checkpoint_config)
        if constraint_mismatches:
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                "Quant Artifact model config does not satisfy the installed Registry source constraints",
                model=plan.model_id,
                path=str(artifact_path),
                source_constraint_mismatches=constraint_mismatches,
            )
        config_mismatches = _behavior_config_mismatches(
            comparison_source, checkpoint_config, type_aliases=get_recipe(plan.model_id).config_type_aliases
        )
        if _has_separately_quantized_lm_head(artifact_path, checkpoint_config):
            # GPTQModel unties a shared LM head when it saves a separately
            # quantized head. This is an intentional checkpoint transform,
            # evidenced by its weight index and quantization declaration.
            config_mismatches = [
                mismatch for mismatch in config_mismatches
                if not (
                    mismatch["field"] in {"tie_word_embeddings", "text_config.tie_word_embeddings"}
                    and mismatch["expected"] is True
                    and mismatch["actual"] is False
                )
            ]
        if config_mismatches:
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                "Quant Artifact model config does not match the source model config",
                model=plan.model_id,
                path=str(artifact_path),
                config_mismatches=config_mismatches,
            )


def _canonicalize_legacy_paddleocr_vl_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Compare the official flat config with Transformers 5.16's nested form.

    The checkpoint must still prove every explicitly supplied source value.
    Only the documented PaddleOCR-VL text and vision type split is accepted.
    """
    if config.get("model_type") != "paddleocr_vl" or "text_config" in config:
        return config
    normalized = dict(config)
    text_fields = {
        "head_dim", "hidden_act", "hidden_size", "intermediate_size",
        "max_position_embeddings", "num_attention_heads", "num_hidden_layers",
        "num_key_value_heads", "pad_token_id", "rms_norm_eps", "use_bias",
        "use_cache", "vocab_size",
    }
    text_config = {field: normalized.pop(field) for field in text_fields if field in normalized}
    text_config["model_type"] = "paddleocr_vl_text"
    if "tie_word_embeddings" in normalized:
        text_config["tie_word_embeddings"] = normalized["tie_word_embeddings"]
    rope_scaling = normalized.get("rope_scaling")
    rope_theta = normalized.get("rope_theta")
    if isinstance(rope_scaling, Mapping) and isinstance(rope_theta, (int, float)) and not isinstance(rope_theta, bool):
        text_config["rope_parameters"] = {**rope_scaling, "rope_theta": rope_theta}
        normalized.pop("rope_scaling")
        normalized.pop("rope_theta")
    normalized["text_config"] = text_config
    vision_config = normalized.get("vision_config")
    if isinstance(vision_config, Mapping) and vision_config.get("model_type") == "paddleocr_vl":
        normalized["vision_config"] = {**vision_config, "model_type": "paddleocr_vl_vision"}
    return normalized


def _has_separately_quantized_lm_head(path: Path, config: Mapping[str, Any]) -> bool:
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, Mapping) or quant_config.get("lm_head") is not True:
        return False
    metadata = quant_config.get("meta")
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("lm_head_quantization"), Mapping):
        return False
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    weights = index.get("weight_map") if isinstance(index, Mapping) else None
    return isinstance(weights, Mapping) and any(
        name.startswith("lm_head.") and name.endswith(".qweight") for name in weights
    )


def _model_config_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    text_configs: list[Mapping[str, Any]] = []

    def collect(item: Any) -> None:
        if not isinstance(item, Mapping):
            return
        for key, child in item.items():
            if key == "text_config" and isinstance(child, Mapping):
                text_configs.append(child)
            collect(child)

    collect(value)
    text_config = text_configs[0] if len(text_configs) == 1 else None
    identity: dict[str, Any] = {}
    for field in (
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "rope_theta",
    ):
        candidate = text_config.get(field) if text_config is not None and field in text_config else value.get(field)
        if field == "rope_theta" and candidate is None:
            rope = explicit_rope_config(text_config if text_config is not None else value).get("rope_parameters")
            if isinstance(rope, Mapping):
                candidate = rope.get("rope_theta")
        if isinstance(candidate, (str, int, float)) and not isinstance(candidate, bool):
            identity[field] = candidate
    architectures = value.get("architectures")
    if isinstance(architectures, list) and architectures and all(isinstance(item, str) for item in architectures):
        identity["architectures"] = tuple(architectures)
    return identity


def _normalize_quant_provider(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _find_base_quant_bits(value: Mapping[str, Any]) -> int | None:
    quantization_config = value.get("quantization_config")
    if isinstance(quantization_config, Mapping):
        bits = _direct_weight_bits(quantization_config)
        if bits is not None:
            return bits
    return _direct_weight_bits(value)


def _direct_weight_bits(value: Mapping[str, Any]) -> int | None:
    for key in ("bits", "w_bits", "weight_bits"):
        bits = value.get(key)
        if isinstance(bits, int) and not isinstance(bits, bool):
            return bits
    weight = value.get("weight")
    if isinstance(weight, str) and re.fullmatch(r"w\d+", weight.lower()):
        return int(weight[1:])
    return None


def _find_all_quant_bits(value: Any) -> set[int]:
    bits: set[int] = set()
    if isinstance(value, Mapping):
        if _is_float_quant_entry(value):
            return bits
        for key, child in value.items():
            if key in {"bits", "w_bits", "weight_bits"} and isinstance(child, int) and not isinstance(child, bool):
                bits.add(child)
            elif key == "weight" and isinstance(child, str) and re.fullmatch(r"w\d+", child.lower()):
                bits.add(int(child[1:]))
            bits.update(_find_all_quant_bits(child))
    elif isinstance(value, list):
        for child in value:
            bits.update(_find_all_quant_bits(child))
    return bits


def _is_float_quant_entry(value: Mapping[str, Any]) -> bool:
    data_type = value.get("data_type")
    return isinstance(data_type, str) and data_type.lower() in {
        "bf16",
        "bfloat16",
        "float",
        "float16",
        "fp",
        "fp16",
    }


def _verify_quant_regions(
    payloads: list[Any],
    plan: BuildPlan,
    base_bits: int | None,
) -> dict[str, dict[str, Any]]:
    expected: dict[str, int] = {}
    for component, precision in plan.precision.components.items():
        weight = str(precision.get("weight", ""))
        if re.fullmatch(r"w\d+", weight):
            expected[component] = int(weight[1:])
    if not expected:
        return {}

    evidence: dict[str, set[int]] = {component: set() for component in expected}
    for payload in payloads:
        _collect_region_bit_evidence(payload, evidence)
        _, command_regions = _command_quant_evidence(payload)
        for component, values in command_regions.items():
            if component in evidence:
                evidence[component].update(values)

    has_explicit_evidence = any(evidence.values())
    verified: dict[str, dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    for component, expected_bits in expected.items():
        actual = evidence[component]
        source = "explicit"
        if not actual and base_bits is not None and has_explicit_evidence:
            actual = {base_bits}
            source = "default_with_explicit_overrides"
        if actual != {expected_bits}:
            invalid.append(
                {
                    "component": component,
                    "expected_bits": expected_bits,
                    "actual_bits": sorted(actual),
                    "evidence": source if actual else "missing",
                }
            )
            continue
        verified[component] = {"weight_bits": expected_bits, "evidence": source}
    if invalid:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Quant Artifact does not prove every registered mixed-precision region",
            model=plan.model_id,
            invalid_regions=invalid,
        )
    return verified


def _collect_region_bit_evidence(value: Any, evidence: dict[str, set[int]]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(child, Mapping) and not _is_float_quant_entry(child):
                bits = _direct_weight_bits(child)
                if bits is not None:
                    for component in evidence:
                        if _quant_key_matches_region(str(key), component):
                            evidence[component].add(bits)
            _collect_region_bit_evidence(child, evidence)
    elif isinstance(value, list):
        for child in value:
            _collect_region_bit_evidence(child, evidence)


def _quant_key_matches_region(key: str, component: str) -> bool:
    normalized = key.lower().replace("\\", "")
    aliases = {
        "attention": ("attention", "attn", "self_attn", "linear_attn"),
        "dense_mlp": ("dense_mlp",),
        "shared_expert": ("shared_expert",),
        "routed_expert": ("routed_expert", ".experts", "experts."),
        "experts": ("routed_expert", ".experts", "experts.", "experts"),
    }.get(component, (component.lower(),))
    return any(alias in normalized for alias in aliases)


def _command_quant_evidence(value: Any) -> tuple[set[int], dict[str, set[int]]]:
    base_bits: set[int] = set()
    regions: dict[str, set[int]] = {}
    if isinstance(value, Mapping):
        command = value.get("command")
        if isinstance(command, list) and all(isinstance(item, str) for item in command):
            flags = _parse_command_flags(command)
            for name in ("bits", "llm_bits"):
                if name in flags:
                    base_bits.add(flags[name])
            flag_regions = {
                "attn_bits": "attention",
                "self_attn_bits": "attention",
                "shared_expert_bits": "shared_expert",
                "expert_bits": "routed_expert",
                "expert_down_bits": "routed_expert",
                "routed_expert_bits": "routed_expert",
            }
            for flag, component in flag_regions.items():
                if flag in flags:
                    regions.setdefault(component, set()).add(flags[flag])
        for child in value.values():
            child_base, child_regions = _command_quant_evidence(child)
            base_bits.update(child_base)
            for component, bits in child_regions.items():
                regions.setdefault(component, set()).update(bits)
    elif isinstance(value, list):
        for child in value:
            child_base, child_regions = _command_quant_evidence(child)
            base_bits.update(child_base)
            for component, bits in child_regions.items():
                regions.setdefault(component, set()).update(bits)
    return base_bits, regions


def _parse_command_flags(command: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    index = 0
    while index < len(command):
        item = command[index]
        if not item.startswith("--"):
            index += 1
            continue
        name, separator, inline_value = item[2:].partition("=")
        name = name.replace("-", "_")
        has_separate_value = not separator and index + 1 < len(command) and not command[index + 1].startswith("--")
        raw_value = inline_value if separator else (command[index + 1] if has_separate_value else "")
        if raw_value.isdigit():
            result[name] = int(raw_value)
        index += 2 if has_separate_value else 1
    return result


def _find_quant_providers(value: Any) -> set[str]:
    providers: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"provider", "quant_method", "method", "algorithm"} and isinstance(child, str):
                providers.add(child)
            if key in {"backend", "packing_format", "recipe"} and isinstance(child, str):
                normalized = _normalize_quant_provider(child)
                if normalized in {"auto-round", "autoround", "auto-round-gptqmodel"}:
                    providers.add("auto-round")
                if normalized in {"gptq", "gptqmodel"} or normalized.startswith("gptqmodel-"):
                    providers.add("gptq")
                if normalized == "qat":
                    providers.add("qat")
            providers.update(_find_quant_providers(child))
    elif isinstance(value, list):
        for child in value:
            providers.update(_find_quant_providers(child))
    return providers


def _inspect_quant_weight_files(path: Path, plan: BuildPlan) -> dict[str, Any]:
    symlinks = [candidate.relative_to(path).as_posix() for candidate in path.rglob("*") if candidate.is_symlink()]
    if symlinks:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Quant Artifact must not contain symlinks",
            model=plan.model_id,
            path=str(path),
            symlinks=sorted(symlinks),
        )

    indexes = sorted(
        candidate
        for candidate in path.rglob("*.index.json")
        if candidate.is_file() and candidate.name.endswith((".safetensors.index.json", ".bin.index.json"))
    )
    if len(indexes) > 1:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Quant Artifact must contain at most one authoritative weight index",
            model=plan.model_id,
            path=str(path),
            indexes=sorted(candidate.relative_to(path).as_posix() for candidate in indexes),
        )
    indexed_weight_paths: set[Path] = set()
    index_assignments: dict[str, Path] = {}
    index_metadata: list[dict[str, Any]] = []
    for index_path in indexes:
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                f"Cannot parse Quant Artifact weight index {index_path}: {exc}",
                model=plan.model_id,
                path=str(index_path),
            ) from exc
        weight_map = index_payload.get("weight_map") if isinstance(index_payload, Mapping) else None
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                f"Quant Artifact weight index has no weight_map: {index_path}",
                model=plan.model_id,
                path=str(index_path),
            )
        expected_suffix = ".safetensors" if index_path.name.endswith(".safetensors.index.json") else ".bin"
        for tensor_name, shard in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ModelError(
                    ErrorCode.QUANT_ARTIFACT_MISMATCH,
                    f"Quant Artifact weight index contains an invalid tensor name: {index_path}",
                    model=plan.model_id,
                    path=str(index_path),
                    tensor_name=tensor_name,
                )
            shard_path = _confined_quant_file(
                path,
                index_path.parent,
                shard,
                plan,
                field="weight_map",
            )
            if shard_path.suffix.lower() != expected_suffix:
                raise ModelError(
                    ErrorCode.QUANT_ARTIFACT_MISMATCH,
                    f"Quant Artifact weight index references an incompatible shard: {index_path}",
                    model=plan.model_id,
                    path=str(index_path),
                    shard=shard_path.relative_to(path).as_posix(),
                    expected_suffix=expected_suffix,
                )
            indexed_weight_paths.add(shard_path)
            index_assignments[tensor_name] = shard_path
        index_metadata.append(
            {
                "file": index_path.relative_to(path).as_posix(),
                "size": index_path.stat().st_size,
                "sha256": file_sha256(index_path),
            }
        )

    discovered_weight_paths = {
        candidate.resolve()
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.name != "model_extra_tensors.safetensors"
        and (
            candidate.suffix == ".safetensors"
            or candidate.suffix == ".gguf"
            or (candidate.suffix == ".bin" and candidate.name.startswith(("model", "pytorch_model")))
        )
    }
    if indexes:
        unreferenced = discovered_weight_paths - indexed_weight_paths
        if unreferenced:
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                "Quant Artifact contains weight containers not referenced by its index",
                model=plan.model_id,
                path=str(path),
                unreferenced=sorted(candidate.relative_to(path).as_posix() for candidate in unreferenced),
            )
        weight_paths = indexed_weight_paths
    else:
        weight_paths = discovered_weight_paths
        if len(weight_paths) > 1:
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                "Quant Artifact with multiple weight containers requires one authoritative index",
                model=plan.model_id,
                path=str(path),
                weights=sorted(candidate.relative_to(path).as_posix() for candidate in weight_paths),
            )
    if not weight_paths:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Quant Artifact contains no checkpoint weight files",
            model=plan.model_id,
            path=str(path),
        )

    weights: list[dict[str, Any]] = []
    container_summaries: list[dict[str, Any]] = []
    for weight_path in sorted(weight_paths):
        size = weight_path.stat().st_size
        if size <= 0:
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                f"Quant Artifact weight file is empty: {weight_path}",
                model=plan.model_id,
                path=str(weight_path),
            )
        summary = _validate_quant_weight_container(weight_path, plan)
        summary["path"] = weight_path
        container_summaries.append(summary)
        weights.append(
            {
                "file": weight_path.relative_to(path).as_posix(),
                "size": size,
                "sha256": file_sha256(weight_path),
            }
        )
    formats = {summary["format"] for summary in container_summaries}
    if len(formats) != 1:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Quant Artifact must use one checkpoint weight format",
            model=plan.model_id,
            path=str(path),
            formats=sorted(formats),
        )
    if formats <= {"safetensors", "torch_zip"}:
        _verify_named_checkpoint_tensors(path, container_summaries, index_assignments, plan)
    return {"weight_indexes": index_metadata, "weights": weights}


def _verify_named_checkpoint_tensors(
    path: Path,
    container_summaries: list[dict[str, Any]],
    index_assignments: Mapping[str, Path],
    plan: BuildPlan,
) -> None:
    tensor_locations: dict[str, list[Path]] = {}
    tensor_shapes: dict[str, tuple[int, ...]] = {}
    for summary in container_summaries:
        container_path = summary["path"]
        for name, shape in summary.get("tensor_shapes", {}).items():
            tensor_locations.setdefault(name, []).append(container_path)
            tensor_shapes[name] = tuple(shape)

    duplicate_tensors = {
        name: sorted(candidate.relative_to(path).as_posix() for candidate in locations)
        for name, locations in tensor_locations.items()
        if len(locations) > 1
    }
    if duplicate_tensors:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Quant Artifact defines the same tensor in multiple weight shards",
            model=plan.model_id,
            path=str(path),
            duplicate_tensors=duplicate_tensors,
        )

    if index_assignments:
        missing_or_misplaced = [
            {
                "tensor": name,
                "expected_shard": expected_path.relative_to(path).as_posix(),
                "actual_shards": [
                    candidate.relative_to(path).as_posix() for candidate in tensor_locations.get(name, ())
                ],
            }
            for name, expected_path in index_assignments.items()
            if tensor_locations.get(name) != [expected_path]
        ]
        unindexed = sorted(set(tensor_locations) - set(index_assignments))
        if missing_or_misplaced or unindexed:
            raise ModelError(
                ErrorCode.QUANT_ARTIFACT_MISMATCH,
                "Quant Artifact weight index does not match the tensors stored in its shards",
                model=plan.model_id,
                path=str(path),
                missing_or_misplaced=missing_or_misplaced,
                unindexed_tensors=unindexed,
            )

    tensor_count = len(tensor_locations)
    element_count = sum(summary["element_count"] for summary in container_summaries)
    max_rank = max((summary["max_rank"] for summary in container_summaries), default=0)
    expected_layers = _source_num_hidden_layers(plan)
    layer_pattern = re.compile(r"(?:^|\.)(?:model(?:\.language_model)?|language_model)\.layers\.(\d+)(?:\.|$)")
    layer_tensors: dict[int, list[str]] = {}
    for name in tensor_locations:
        match = layer_pattern.search(name)
        if match is not None:
            layer_tensors.setdefault(int(match.group(1)), []).append(name)
    actual_layers = set(layer_tensors)
    required_layers = set(range(expected_layers))
    recipe = get_recipe(plan.model_id)
    allowed_layers = required_layers | _auxiliary_checkpoint_layers(plan, expected_layers)
    has_embedding = any(name.endswith(recipe.checkpoint_embedding_suffixes) for name in tensor_locations)
    has_final_norm = any(
        re.search(r"(?:^|\.)(?:model(?:\.language_model)?|language_model)\.norm\.weight$", name)
        for name in tensor_locations
    )
    incomplete_layer_groups: dict[int, list[str]] = {}
    for layer, names in layer_tensors.items():
        lowered = [name.lower() for name in names]
        missing_groups = []
        if not any("attn" in name or "attention" in name or "mixer" in name for name in lowered):
            missing_groups.append("attention_or_mixer")
        if not any("mlp" in name or "expert" in name or "feed_forward" in name for name in lowered):
            missing_groups.append("mlp_or_expert")
        if not any("norm" in name for name in lowered):
            missing_groups.append("norm")
        if missing_groups:
            incomplete_layer_groups[layer] = missing_groups

    shape_mismatches = _checkpoint_shape_mismatches(tensor_shapes, layer_pattern, plan)
    if (
        tensor_count < 2
        or element_count <= 1
        or max_rank < 2
        or not required_layers.issubset(actual_layers)
        or not actual_layers.issubset(allowed_layers)
        or not has_embedding
        or not has_final_norm
        or incomplete_layer_groups
        or shape_mismatches
    ):
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Quant Artifact weights do not constitute a verifiably complete model checkpoint",
            model=plan.model_id,
            path=str(path),
            tensor_count=tensor_count,
            element_count=element_count,
            max_rank=max_rank,
            expected_num_hidden_layers=expected_layers,
            missing_layers=sorted(required_layers - actual_layers),
            unexpected_layers=sorted(actual_layers - allowed_layers),
            incomplete_layer_groups=incomplete_layer_groups,
            has_embedding=has_embedding,
            has_final_norm=has_final_norm,
            shape_mismatches=shape_mismatches,
        )


def _auxiliary_checkpoint_layers(plan: BuildPlan, main_layers: int) -> set[int]:
    """Allow configured trailing draft layers without requiring them for a base export."""
    field = get_recipe(plan.model_id).checkpoint_auxiliary_layers_field
    if field is None:
        return set()
    config = json.loads((Path(plan.model_dir) / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    count = text.get(field, 0)
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= main_layers:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Checkpoint auxiliary layer count must be a non-negative integer no larger than the main layer count",
            model=plan.model_id,
            field=field,
            value=count,
        )
    return set(range(main_layers, main_layers + count))


def _checkpoint_shape_mismatches(
    tensor_shapes: Mapping[str, tuple[int, ...]],
    layer_pattern: re.Pattern[str],
    plan: BuildPlan,
) -> list[dict[str, Any]]:
    config_path = Path(plan.model_dir) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    identity = _model_config_identity(config) if isinstance(config, Mapping) else {}
    hidden_size = identity.get("hidden_size")
    vocab_size = identity.get("vocab_size")
    embedding_suffixes = get_recipe(plan.model_id).checkpoint_embedding_suffixes
    mismatches: list[dict[str, Any]] = []
    for name, shape in tensor_shapes.items():
        expected_shape: tuple[int, ...] | None = None
        if name.endswith(embedding_suffixes) and isinstance(hidden_size, int):
            expected_shape = (
                vocab_size if isinstance(vocab_size, int) else shape[0] if len(shape) == 2 else -1,
                hidden_size,
            )
        elif (
            isinstance(hidden_size, int)
            and name.endswith("norm.weight")
            and (
                # Only residual-stream norms directly on a decoder layer have
                # hidden_size channels. Attention Q/K norms use head_dim;
                # recurrent and MLA norms have other model-specific widths.
                (
                    (layer_match := layer_pattern.search(name)) is not None
                    and "." not in name[layer_match.end() :].removesuffix(".weight")
                )
                or re.search(
                    r"(?:^|\.)(?:model(?:\.language_model)?|language_model)\.norm\.weight$",
                    name,
                )
            )
        ):
            expected_shape = (hidden_size,)
        if expected_shape is not None and shape != expected_shape:
            mismatches.append({"tensor": name, "expected": list(expected_shape), "actual": list(shape)})
    return mismatches


def _validate_quant_weight_container(path: Path, plan: BuildPlan) -> dict[str, Any]:
    try:
        suffix = path.suffix.lower()
        if suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(path, framework="np") as handle:
                keys = list(handle.keys())
                shapes = [tuple(int(value) for value in handle.get_slice(key).get_shape()) for key in keys]
                if not keys or any(any(dimension <= 0 for dimension in shape) for shape in shapes):
                    raise ValueError("contains no non-empty tensors")
            element_count = 0
            for shape in shapes:
                elements = 1
                for dimension in shape:
                    elements *= dimension
                element_count += elements
            return {
                "format": "safetensors",
                "tensor_count": len(keys),
                "element_count": element_count,
                "max_rank": max((len(shape) for shape in shapes), default=0),
                "tensor_names": keys,
                "tensor_shapes": dict(zip(keys, shapes, strict=True)),
            }
        if suffix == ".gguf":
            from gguf import GGUFReader

            reader = GGUFReader(str(path), "r")
            tensors = list(reader.tensors)
            if not tensors:
                raise ValueError("contains no tensors")
            file_size = path.stat().st_size
            for tensor in tensors:
                shape = tuple(int(dimension) for dimension in tensor.shape)
                if not shape or any(dimension <= 0 for dimension in shape):
                    raise ValueError(f"tensor {tensor.name!r} has an invalid shape")
                if tensor.n_bytes <= 0 or tensor.data_offset < reader.data_offset:
                    raise ValueError(f"tensor {tensor.name!r} has an invalid data range")
                if tensor.data_offset + tensor.n_bytes > file_size:
                    raise ValueError(f"tensor {tensor.name!r} extends beyond the GGUF file")
            if not any(tensor.n_elements > 1 for tensor in tensors):
                raise ValueError("does not contain a non-scalar model tensor")
            return {
                "format": "gguf",
                "tensor_count": len(tensors),
                "element_count": sum(int(tensor.n_elements) for tensor in tensors),
                "max_rank": max(len(tensor.shape) for tensor in tensors),
            }
        if suffix == ".bin":
            return _validate_torch_zip_checkpoint(path)
        raise ValueError(f"unsupported checkpoint suffix {path.suffix!r}")
    except Exception as exc:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            f"Quant Artifact weight container is invalid: {path}: {exc}",
            model=plan.model_id,
            path=str(path),
        ) from exc


def _source_num_hidden_layers(plan: BuildPlan) -> int:
    config_path = Path(plan.model_dir) / "config.json"
    try:
        source_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            f"Cannot read source num_hidden_layers for safetensors validation: {exc}",
            model=plan.model_id,
            path=str(config_path),
        ) from exc
    identity = _model_config_identity(source_config) if isinstance(source_config, Mapping) else {}
    num_hidden_layers = identity.get("num_hidden_layers")
    if isinstance(num_hidden_layers, bool) or not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            "Source model config does not declare a positive num_hidden_layers",
            model=plan.model_id,
            path=str(config_path),
            num_hidden_layers=num_hidden_layers,
        )
    return num_hidden_layers


def _validate_torch_zip_checkpoint(path: Path) -> dict[str, Any]:
    import pickletools
    import struct
    import zipfile

    if not zipfile.is_zipfile(path):
        raise ValueError("legacy pickle checkpoints are not safely structurally verifiable")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if not infos or len(names) != len(set(names)):
            raise ValueError("has no entries or contains duplicate ZIP members")
        if any(
            not name
            or Path(name).is_absolute()
            or PureWindowsPath(name).is_absolute()
            or ".." in Path(name).parts
            or "\\" in name
            or info.flag_bits & 0x1
            for name, info in zip(names, infos, strict=True)
        ):
            raise ValueError("contains unsafe or encrypted ZIP members")

        data_pickles = [name for name in names if name == "data.pkl" or name.endswith("/data.pkl")]
        if len(data_pickles) != 1:
            raise ValueError("does not contain exactly one Torch data.pkl")
        prefix = data_pickles[0][: -len("data.pkl")]
        version_name = f"{prefix}version"
        storage_names = [name for name in names if name.startswith(f"{prefix}data/") and name != f"{prefix}data/"]
        if version_name not in names or not storage_names:
            raise ValueError("is missing Torch version or tensor storage entries")

        file_size = path.stat().st_size
        central_directory = int(getattr(archive, "start_dir", file_size))
        with path.open("rb") as stream:
            for info in infos:
                stream.seek(info.header_offset)
                header = stream.read(30)
                if len(header) != 30:
                    raise ValueError(f"ZIP member {info.filename!r} has a truncated local header")
                signature, *_, name_size, extra_size = struct.unpack("<IHHHHHIIIHH", header)
                if signature != 0x04034B50:
                    raise ValueError(f"ZIP member {info.filename!r} has an invalid local header")
                data_start = info.header_offset + 30 + name_size + extra_size
                if data_start + info.compress_size > central_directory:
                    raise ValueError(f"ZIP member {info.filename!r} extends beyond the archive data")

        data_info = archive.getinfo(data_pickles[0])
        if data_info.file_size > 64 * 1024 * 1024:
            raise ValueError("Torch data.pkl is unreasonably large")
        pickle_payload = archive.read(data_info)
        operations = list(pickletools.genops(pickle_payload))
        if not operations or operations[-1][0].name != "STOP" or operations[-1][2] + 1 != len(pickle_payload):
            raise ValueError("Torch data.pkl is truncated or has trailing payload")
        if not any("_rebuild_tensor" in str(argument) for _, argument, _ in operations):
            raise ValueError("Torch data.pkl does not describe tensor weights")

        version_info = archive.getinfo(version_name)
        if version_info.file_size > 64:
            raise ValueError("Torch archive version is invalid")
        version = archive.read(version_info)
        if not version.strip().isdigit():
            raise ValueError("Torch archive version is invalid")
        if any(archive.getinfo(name).file_size <= 0 for name in storage_names):
            raise ValueError("Torch archive contains an empty tensor storage")

    import torch

    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("Torch checkpoint does not contain a non-empty state dict")
    tensor_shapes: dict[str, tuple[int, ...]] = {}
    for name, value in payload.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Torch checkpoint contains an invalid tensor name")
        if not isinstance(value, torch.Tensor):
            continue
        shape = tuple(int(dimension) for dimension in value.shape)
        if any(dimension <= 0 for dimension in shape):
            raise ValueError(f"tensor {name!r} has an invalid shape")
        tensor_shapes[name] = shape
    if not tensor_shapes:
        raise ValueError("Torch checkpoint state dict contains no tensors")
    element_count = 0
    for shape in tensor_shapes.values():
        elements = 1
        for dimension in shape:
            elements *= dimension
        element_count += elements
    return {
        "format": "torch_zip",
        "tensor_count": len(tensor_shapes),
        "element_count": element_count,
        "max_rank": max((len(shape) for shape in tensor_shapes.values()), default=0),
        "tensor_names": list(tensor_shapes),
        "tensor_shapes": tensor_shapes,
    }


def _confined_quant_file(
    path: Path,
    base: Path,
    value: Any,
    plan: BuildPlan,
    *,
    field: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            f"Quant Artifact {field} contains an invalid file path",
            model=plan.model_id,
            value=value,
        )
    relative = Path(value)
    if relative.is_absolute() or PureWindowsPath(value).is_absolute() or ".." in relative.parts or "\\" in value:
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            f"Quant Artifact {field} contains an unsafe file path: {value}",
            model=plan.model_id,
            value=value,
        )
    candidate = base / relative
    resolved = candidate.resolve()
    if not resolved.is_relative_to(path) or workspace.path_has_symlink(path, candidate) or not resolved.is_file():
        raise ModelError(
            ErrorCode.QUANT_ARTIFACT_MISMATCH,
            f"Quant Artifact {field} references a missing or unsafe file: {value}",
            model=plan.model_id,
            value=value,
        )
    return resolved
