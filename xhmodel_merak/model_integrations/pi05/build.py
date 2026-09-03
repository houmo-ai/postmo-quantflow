from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_build.contracts import BuildPlan
from xhmodel_merak.model_build.fingerprints import resource_sha256


_POLICY_FILES = (
    "config.json",
    "configuration.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_2_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def normalize_export(stage_path: Path, export_result: Any, plan: BuildPlan) -> str | None:
    del stage_path, export_result, plan
    return None


def package_runtime(stage_path: Path, plan: BuildPlan) -> None:
    runtime_root = stage_path / "runtime"
    policy_root = runtime_root / "policy"
    tokenizer_root = runtime_root / "tokenizer"

    source_root = _source_directory(Path(plan.model_dir), plan, "PI0.5 policy source")
    missing_policy = []
    policy_sources = {}
    for name in _POLICY_FILES:
        source = _confined_source_file(source_root, name, plan, "PI0.5 policy source")
        if source is None:
            missing_policy.append(name)
        else:
            policy_sources[name] = source
    if missing_policy:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"PI0.5 source is missing runtime processor assets: {missing_policy}",
            model=plan.model_id,
            missing=missing_policy,
        )
    _validate_policy_snapshot(source_root, plan)

    tokenizer = _source_tree(
        Path(plan.resources["tokenizer"].path),
        plan,
        "PI0.5 tokenizer source",
    )
    if tokenizer_root.exists() or tokenizer_root.is_symlink():
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "PI0.5 export unexpectedly contains a runtime/tokenizer path",
            model=plan.model_id,
        )
    policy_root.mkdir(parents=True, exist_ok=True)
    for name, source in policy_sources.items():
        shutil.copy2(source, policy_root / name)
    shutil.copytree(tokenizer, tokenizer_root)
    copied_sha256 = resource_sha256(tokenizer_root)
    if copied_sha256 != plan.resources["tokenizer"].sha256:
        raise _invalid(
            plan,
            "PI0.5 packaged tokenizer does not match the resolved resource fingerprint",
            expected=plan.resources["tokenizer"].sha256,
            actual=copied_sha256,
        )

    meta_path = stage_path / "export_meta_info.json"
    meta = _load_json_mapping(meta_path, plan, "PI0.5 export metadata")
    contract = meta.get("compact_prefix")
    max_length = contract.get("text_max_length") if isinstance(contract, Mapping) else None
    _validate_tokenizer_snapshot(tokenizer_root, max_length, plan)
    meta["hf_model"] = "runtime/policy"
    meta["runtime"] = {
        "kind": "pi05_compact_hmonnx",
        "policy": "runtime/policy",
        "tokenizer": "runtime/tokenizer",
        "token_embedding": "Gemma2B/token_embedding.pt",
    }
    for component_name in ("gemma", "expert"):
        component = meta.get("components", {}).get(component_name)
        if not isinstance(component, Mapping):
            continue
        component_meta_path = _confined_file(
            stage_path,
            f"{component.get('component_dir', '')}/{component.get('meta_file', '')}",
            plan,
        )
        component_meta = json.loads(component_meta_path.read_text(encoding="utf-8"))
        component_meta["hf_model"] = "../../runtime/policy"
        component_meta_path.write_text(
            json.dumps(component_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitize_artifact(stage_path: Path, plan: BuildPlan) -> None:
    del plan
    meta_path = stage_path / "export_meta_info.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("source_paths", None)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_artifact(stage_path: Path, plan: BuildPlan) -> dict[str, Any]:
    meta_path = stage_path / "export_meta_info.json"
    if not meta_path.is_file():
        raise _invalid(plan, "PI0.5 Artifact is missing export_meta_info.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    components = meta.get("components")
    if not isinstance(components, Mapping) or set(components) != set(plan.components):
        raise _invalid(
            plan,
            "PI0.5 Artifact component set does not match the Registry",
            expected=list(plan.components),
            actual=sorted(components) if isinstance(components, Mapping) else components,
        )
    if any(not isinstance(component, Mapping) for component in components.values()):
        raise _invalid(plan, "PI0.5 Artifact components must contain metadata objects")

    other = components["other"]
    graphs = other.get("graphs") if isinstance(other, Mapping) else None
    graph_names = (
        [graph.get("name") if isinstance(graph, Mapping) else None for graph in graphs]
        if isinstance(graphs, list)
        else []
    )
    expected_graph_names = {"action_in_proj", "action_out_proj", "time_mlp"}
    if (
        len(graph_names) != len(expected_graph_names)
        or any(not isinstance(name, str) for name in graph_names)
        or set(graph_names) != expected_graph_names
        or any(
            not isinstance(graph, Mapping) or not _positive_shape(graph.get("input_shape")) for graph in graphs or ()
        )
    ):
        raise _invalid(
            plan,
            "PI0.5 compact runtime graph set is incomplete or ambiguous",
            expected=sorted(expected_graph_names),
            actual=graph_names,
        )

    required = [
        "runtime/policy/config.json",
        "runtime/policy/policy_preprocessor.json",
        "runtime/policy/policy_postprocessor.json",
        "runtime/policy/policy_preprocessor_step_2_normalizer_processor.safetensors",
        "runtime/policy/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "runtime/tokenizer/tokenizer.model",
        "runtime/tokenizer/tokenizer_config.json",
        "runtime/tokenizer/special_tokens_map.json",
    ]
    vision = components["vision"]
    required.append(_component_path(vision, "hmonnx_file"))
    for graph in graphs:
        required.append(_component_path(other, "hmonnx_file", graph=graph))
    nested_components: dict[str, Mapping[str, Any]] = {}
    for name in ("gemma", "expert"):
        component = components[name]
        nested_meta_path = _confined_file(
            stage_path,
            f"{component.get('component_dir', '')}/{component.get('meta_file', '')}",
            plan,
        )
        nested = _load_json_mapping(nested_meta_path, plan, f"PI0.5 {name} component metadata")
        nested_components[name] = nested
        required.extend(
            [
                _component_path(component, "prefill_onnx_file", graph=nested),
                _component_path(component, "decode_onnx_file", graph=nested),
            ]
        )
        if name == "gemma":
            required.append(_component_path(component, "token_embedding_file", graph=nested))
    _require_relative_files(stage_path, required, plan)
    policy_root = stage_path / "runtime/policy"
    _validate_policy_snapshot(policy_root, plan)
    policy_config = _load_json_mapping(policy_root / "config.json", plan, "PI0.5 policy config.json")
    _validate_compact_runtime(meta, components, nested_components, policy_config, plan)
    contract = meta["compact_prefix"]
    max_length = contract["text_max_length"]
    _validate_tokenizer_snapshot(stage_path / "runtime/tokenizer", max_length, plan)
    return {
        "metadata_file": meta_path.name,
        "runtime_kind": "pi05_compact_hmonnx",
        "components": list(plan.components),
        "requires_source": False,
        "loadable": True,
    }


def _component_path(component: Mapping[str, Any], key: str, *, graph: Mapping[str, Any] | None = None) -> str:
    component_dir = component.get("component_dir")
    value = (graph or component).get(key)
    if not isinstance(component_dir, str) or not component_dir or not isinstance(value, str) or not value:
        return ""
    return f"{component_dir}/{value}"


def _positive_shape(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0 for dimension in value)
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_compact_runtime(
    meta: Mapping[str, Any],
    components: Mapping[str, Mapping[str, Any]],
    nested_components: Mapping[str, Mapping[str, Any]],
    policy_config: Mapping[str, Any],
    plan: BuildPlan,
) -> None:
    contract = meta.get("compact_prefix")
    if not isinstance(contract, Mapping):
        raise _invalid(plan, "PI0.5 Artifact has no compact_prefix runtime contract")

    selected_image_indices = contract.get("selected_image_indices")
    action_horizon = contract.get("action_horizon")
    text_max_length = contract.get("text_max_length")
    prefix_sequence_length = contract.get("prefix_sequence_length")
    cache_length = contract.get("cache_length")
    if selected_image_indices != [0, 1]:
        raise _invalid(plan, "PI0.5 LIBERO runtime requires selected_image_indices [0, 1]")
    if action_horizon != 50 or isinstance(action_horizon, bool):
        raise _invalid(plan, "PI0.5 LIBERO runtime requires action_horizon 50")
    lengths = {
        "text_max_length": text_max_length,
        "prefix_sequence_length": prefix_sequence_length,
        "cache_length": cache_length,
    }
    invalid_lengths = {name: value for name, value in lengths.items() if not _positive_int(value)}
    if invalid_lengths:
        raise _invalid(plan, "PI0.5 compact runtime lengths must be positive integers", invalid=invalid_lengths)
    expected_prefix_length = len(selected_image_indices) * 256 + text_max_length
    if prefix_sequence_length != expected_prefix_length or cache_length < prefix_sequence_length + action_horizon:
        raise _invalid(
            plan,
            "PI0.5 compact prefix and action horizon exceed or disagree with runtime capacity",
            expected_prefix_length=expected_prefix_length,
            prefix_sequence_length=prefix_sequence_length,
            action_horizon=action_horizon,
            cache_length=cache_length,
        )

    state_dim = _feature_dimension(policy_config, "input_features", "observation.state", plan)
    action_dim = _feature_dimension(policy_config, "output_features", "action", plan)
    if state_dim != 8 or action_dim != 7:
        raise _invalid(
            plan,
            "PI0.5 LIBERO policy dimensions do not match the public contract",
            state_dim=state_dim,
            action_dim=action_dim,
        )
    max_action_dim = policy_config.get("max_action_dim")
    num_inference_steps = policy_config.get("num_inference_steps", 10)
    image_resolution = policy_config.get("image_resolution")
    if not _positive_int(max_action_dim) or max_action_dim < action_dim:
        raise _invalid(plan, "PI0.5 policy max_action_dim must be at least 7", value=max_action_dim)
    if not _positive_int(num_inference_steps):
        raise _invalid(plan, "PI0.5 policy num_inference_steps must be a positive integer")
    if not isinstance(image_resolution, list) or len(image_resolution) != 2 or not _positive_shape(image_resolution):
        raise _invalid(plan, "PI0.5 policy image_resolution must contain two positive dimensions")

    runtime = meta.get("runtime")
    gemma = components["gemma"]
    gemma_meta = nested_components["gemma"]
    expected_runtime = {
        "kind": "pi05_compact_hmonnx",
        "policy": "runtime/policy",
        "tokenizer": "runtime/tokenizer",
        "token_embedding": _component_path(gemma, "token_embedding_file", graph=gemma_meta),
    }
    if (
        not isinstance(runtime, Mapping)
        or any(runtime.get(field) != value for field, value in expected_runtime.items())
        or meta.get("hf_model") != "runtime/policy"
    ):
        raise _invalid(plan, "PI0.5 runtime metadata does not match the packaged compact runtime")

    vision_shape = components["vision"].get("input_shape")
    expected_vision_shape = [1, 3, *image_resolution]
    if vision_shape != expected_vision_shape:
        raise _invalid(
            plan,
            "PI0.5 vision graph input does not match policy image_resolution",
            expected=expected_vision_shape,
            actual=vision_shape,
        )
    graphs = {graph["name"]: graph for graph in components["other"]["graphs"]}
    action_in_shape = graphs["action_in_proj"]["input_shape"]
    action_out_shape = graphs["action_out_proj"]["input_shape"]
    time_mlp_shape = graphs["time_mlp"]["input_shape"]
    if action_in_shape != [1, action_horizon, max_action_dim]:
        raise _invalid(plan, "PI0.5 action_in_proj input does not match horizon/max_action_dim")
    if (
        len(action_out_shape) != 3
        or action_out_shape[:2] != [1, action_horizon]
        or len(time_mlp_shape) != 2
        or time_mlp_shape[0] != 1
        or action_out_shape[2] != time_mlp_shape[1]
        or time_mlp_shape[1] % 2 != 0
    ):
        raise _invalid(plan, "PI0.5 action_out_proj and time_mlp graph dimensions are incompatible")

    common_nested = {
        "selected_image_indices": selected_image_indices,
        "text_max_length": text_max_length,
        "prefix_sequence_length": prefix_sequence_length,
        "action_horizon": action_horizon,
        "cache_length": cache_length,
    }
    for name, nested in nested_components.items():
        mismatches = {
            field: {"expected": expected, "actual": nested.get(field)}
            for field, expected in common_nested.items()
            if nested.get(field) != expected
        }
        if mismatches:
            raise _invalid(plan, f"PI0.5 {name} metadata disagrees with compact_prefix", mismatches=mismatches)
        expected_attention_shape = [1, 1, 1, cache_length]
        if (
            nested.get("prefill_attention_mask_shape") != expected_attention_shape
            or nested.get("decode_attention_mask_shape") != expected_attention_shape
        ):
            raise _invalid(plan, f"PI0.5 {name} attention metadata disagrees with cache_length")

    expert_meta = nested_components["expert"]
    gemma_cache_shape = gemma_meta.get("kv_cache_shape")
    expert_cache_shape = expert_meta.get("kv_cache_shape")
    if (
        not _positive_shape(gemma_cache_shape)
        or len(gemma_cache_shape) < 3
        or gemma_cache_shape[2] != cache_length
        or expert_cache_shape != gemma_cache_shape
    ):
        raise _invalid(plan, "PI0.5 Gemma and expert KV cache metadata is incompatible")
    gemma_layers = gemma_meta.get("num_hidden_layers")
    if not _positive_int(gemma_layers) or expert_meta.get("num_hidden_layers") != gemma_layers:
        raise _invalid(plan, "PI0.5 Gemma and expert layer-count metadata is incompatible")
    if (
        gemma_meta.get("prefill_input_sequence_length") != prefix_sequence_length
        or expert_meta.get("decode_input_sequence_length") != action_horizon
        or expert_meta.get("hidden_size") != action_out_shape[2]
    ):
        raise _invalid(plan, "PI0.5 graph metadata does not match compact runtime dimensions")


def _validate_policy_snapshot(policy_root: Path, plan: BuildPlan) -> None:
    documents = {
        name: _load_json_mapping(policy_root / name, plan, f"PI0.5 policy {name}")
        for name in _POLICY_FILES
        if name.endswith(".json")
    }
    config = documents["config.json"]
    specs = (
        (
            "policy_preprocessor_step_2_normalizer_processor.safetensors",
            "observation.state",
            "STATE",
            _feature_dimension(config, "input_features", "observation.state", plan),
        ),
        (
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            "action",
            "ACTION",
            _feature_dimension(config, "output_features", "action", plan),
        ),
    )
    for name, prefix, feature_type, dimension in specs:
        _validate_stats_file(
            policy_root / name,
            prefix,
            _normalization_mode(config, feature_type, plan),
            dimension,
            plan,
        )


def _load_json_mapping(path: Path, plan: BuildPlan, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _invalid(plan, f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise _invalid(plan, f"{label} must contain a JSON object")
    return value


def _feature_dimension(
    config: Mapping[str, Any],
    section: str,
    name: str,
    plan: BuildPlan,
) -> int:
    features = config.get(section)
    feature = features.get(name) if isinstance(features, Mapping) else None
    shape = feature.get("shape") if isinstance(feature, Mapping) else None
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] <= 0
    ):
        raise _invalid(plan, f"PI0.5 policy feature {name!r} must have one positive dimension")
    return shape[0]


def _normalization_mode(config: Mapping[str, Any], feature_type: str, plan: BuildPlan) -> str:
    mapping = config.get("normalization_mapping", {})
    if not isinstance(mapping, Mapping):
        raise _invalid(plan, "PI0.5 policy normalization_mapping must be a JSON object")
    value = mapping.get(feature_type)
    mode = str(value or "QUANTILES").rsplit(".", 1)[-1].lower()
    if mode not in {"mean_std", "min_max", "quantiles", "quantile10"}:
        raise _invalid(plan, f"PI0.5 policy has unsupported {feature_type} normalization mode: {value!r}")
    return mode


def _validate_stats_file(path: Path, prefix: str, mode: str, dimension: int, plan: BuildPlan) -> None:
    try:
        from safetensors.numpy import load_file

        values = load_file(path)
    except Exception as exc:
        raise _invalid(plan, f"PI0.5 policy asset {path.name} is not valid safetensors: {exc}") from exc
    if not values:
        raise _invalid(plan, f"PI0.5 policy asset {path.name} contains no tensors")
    keys = {
        "mean_std": ("mean", "std"),
        "min_max": ("min", "max"),
        "quantiles": ("q01", "q99"),
        "quantile10": ("q10", "q90"),
    }[mode]
    names = tuple(f"{prefix}.{key}" for key in keys)
    try:
        low, high = (np.asarray(values[name]) for name in names)
    except KeyError as exc:
        raise _invalid(plan, f"PI0.5 policy asset {path.name} is missing normalization tensors: {names}") from exc
    for name, value in zip(names, (low, high), strict=True):
        if not np.issubdtype(value.dtype, np.number) or value.size < dimension:
            raise _invalid(
                plan,
                f"PI0.5 normalization tensor {name!r} is not structurally usable",
                expected_values=dimension,
                actual_values=int(value.size),
            )
        if not np.isfinite(value.reshape(-1)[:dimension]).all():
            raise _invalid(plan, f"PI0.5 normalization tensor {name!r} contains non-finite values")
    selected_low = low.reshape(-1)[:dimension]
    selected_high = high.reshape(-1)[:dimension]
    if mode == "mean_std":
        invalid_bounds = (selected_high < 0).any()
    else:
        invalid_bounds = (selected_high < selected_low).any()
    if invalid_bounds:
        raise _invalid(plan, f"PI0.5 policy asset {path.name} contains invalid normalization bounds")


def _validate_tokenizer_snapshot(tokenizer_root: Path, max_length: Any, plan: BuildPlan) -> None:
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise _invalid(plan, "PI0.5 compact metadata has no positive text_max_length")
    required = ("tokenizer.model", "tokenizer_config.json", "special_tokens_map.json")
    missing = [name for name in required if not (tokenizer_root / name).is_file()]
    if missing:
        raise _invalid(plan, "PI0.5 tokenizer is not a complete HF snapshot", missing=missing)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_root,
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            raise ValueError("pad_token_id is not configured")
        encoded = tokenizer(
            ["Task: validation;\nAction: "],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if tuple(encoded["input_ids"].shape) != (1, max_length):
            raise ValueError(f"unexpected input_ids shape {tuple(encoded['input_ids'].shape)}")
        if tuple(encoded["attention_mask"].shape) != (1, max_length):
            raise ValueError(f"unexpected attention_mask shape {tuple(encoded['attention_mask'].shape)}")
    except Exception as exc:
        raise _invalid(plan, f"PI0.5 tokenizer snapshot failed the runtime smoke contract: {exc}") from exc


def _source_directory(value: Path, plan: BuildPlan, label: str) -> Path:
    path = value.expanduser()
    if path.is_symlink():
        raise _invalid(plan, f"{label} root must not be a symlink", path=str(path))
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _invalid(plan, f"{label} directory is unavailable: {exc}", path=str(path)) from exc
    if not resolved.is_dir():
        raise _invalid(plan, f"{label} is not a directory", path=str(path))
    return resolved


def _confined_source_file(source_root: Path, value: str, plan: BuildPlan, label: str) -> Path | None:
    relative = Path(value)
    candidate = source_root / relative
    cursor = source_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise _invalid(plan, f"{label} must not contain symlinks", path=value)
    if relative.is_absolute() or ".." in relative.parts or not candidate.exists():
        return None
    try:
        path = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _invalid(plan, f"{label} file is unavailable: {exc}", path=value) from exc
    if not path.is_relative_to(source_root):
        raise _invalid(plan, f"{label} file escapes its resolved source root", path=value)
    return path if path.is_file() else None


def _source_tree(value: Path, plan: BuildPlan, label: str) -> Path:
    root = _source_directory(value, plan, label)
    for child in root.rglob("*"):
        if child.is_symlink():
            raise _invalid(plan, f"{label} must not contain symlinks", path=str(child.relative_to(root)))
        try:
            resolved = child.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise _invalid(plan, f"{label} child is unavailable: {exc}", path=str(child)) from exc
        if not resolved.is_relative_to(root):
            raise _invalid(plan, f"{label} child escapes its resolved source root", path=str(child))
    return root


def _require_relative_files(stage_path: Path, values: list[str], plan: BuildPlan) -> None:
    invalid = []
    root = stage_path.resolve()
    for value in values:
        relative = Path(value)
        candidate = root / relative
        path = candidate.resolve()
        if (
            not value
            or relative.is_absolute()
            or ".." in relative.parts
            or candidate.is_symlink()
            or not path.is_relative_to(root)
            or not path.is_file()
        ):
            invalid.append(value)
    if invalid:
        raise _invalid(plan, "PI0.5 Artifact is not self-contained", invalid_runtime_assets=invalid)


def _confined_file(stage_path: Path, value: str, plan: BuildPlan) -> Path:
    relative = Path(value)
    root = stage_path.resolve()
    candidate = root / relative
    path = candidate.resolve()
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or candidate.is_symlink()
        or not path.is_relative_to(root)
        or not path.is_file()
    ):
        raise _invalid(plan, "PI0.5 component metadata escapes the Artifact", path=value)
    return path


def _invalid(plan: BuildPlan, message: str, **details: Any) -> ModelError:
    return ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, message, model=plan.model_id, **details)


__all__ = ["normalize_export", "package_runtime", "sanitize_artifact", "validate_artifact"]
