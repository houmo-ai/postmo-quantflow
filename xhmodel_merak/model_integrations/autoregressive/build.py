"""Build adapter for the shared autoregressive Artifact contract."""

from __future__ import annotations

import json
import os
import shutil
import stat
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_build.artifacts import flatten_single_export_root
from xhmodel_merak.model_build.contracts import BuildPlan
from xhmodel_merak.model_build.runtime_assets import behavior_config_mismatches, quant_runtime_asset_paths

from .vision import validate_dynamic_vision


def normalize_export(stage_path: Path, export_result: Any, plan: BuildPlan) -> str | None:
    del export_result, plan
    return flatten_single_export_root(stage_path)


def package_runtime(stage_path: Path, plan: BuildPlan) -> None:
    if not plan.quant_checkpoint:
        return

    checkpoint = Path(plan.quant_checkpoint).resolve()
    destination = stage_path / "hf_config"
    exported_config_path = destination / "config.json"
    exported_config = _load_json_object(exported_config_path) if exported_config_path.is_file() else None
    checkpoint_config_path = checkpoint / "config.json"
    checkpoint_config = _load_json_object(checkpoint_config_path)
    if exported_config is not None:
        mismatches = behavior_config_mismatches(checkpoint_config, exported_config)
        if mismatches:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                "Exported runtime config does not match the reused Quant Artifact config",
                model=plan.model_id,
                config_mismatches=mismatches,
            )

    for source in quant_runtime_asset_paths(checkpoint):
        relative = source.relative_to(checkpoint)
        _copy_runtime_asset(source, destination / relative)

    if exported_config is not None:
        for field in ("compression_config", "quantization_config"):
            checkpoint_value = checkpoint_config.get(field)
            exported_value = exported_config.get(field)
            if isinstance(checkpoint_value, Mapping) and isinstance(exported_value, Mapping):
                _add_missing_values(checkpoint_value, exported_value)
        exported_config_path.write_text(
            json.dumps(checkpoint_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Cannot read reused checkpoint runtime config {path}: {exc}",
            path=str(path),
        ) from exc
    if not isinstance(value, dict):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Reused checkpoint runtime config must contain an object: {path}",
            path=str(path),
        )
    return value


def _add_missing_values(target: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    mutable_target = target if isinstance(target, dict) else None
    if mutable_target is None:
        return
    for key, value in source.items():
        if key not in mutable_target:
            mutable_target[key] = value
        elif isinstance(mutable_target[key], Mapping) and isinstance(value, Mapping):
            _add_missing_values(mutable_target[key], value)


def _copy_runtime_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.xhcopy-{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise OSError(f"runtime asset is not a regular file: {source}")
        with os.fdopen(os.dup(descriptor), "rb") as source_stream, temporary.open("xb") as destination_stream:
            shutil.copyfileobj(source_stream, destination_stream)
        os.chmod(temporary, source_stat.st_mode & 0o777)
        temporary.replace(destination)
    except OSError as exc:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Cannot package reused checkpoint runtime asset {source}: {exc}",
            source=str(source),
            destination=str(destination),
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def sanitize_artifact(stage_path: Path, plan: BuildPlan) -> None:
    meta_path = stage_path / "golden_meta_info.json"
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    local_hf_config = "hf_config"
    meta["hf_config"] = local_hf_config
    _replace_source_model_paths(meta, local_hf_config)
    meta = _scrub_absolute_paths(meta)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hf_config = stage_path / local_hf_config
    if hf_config.is_dir():
        for path in hf_config.rglob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                path.name == "tokenizer_config.json"
                and payload.get("tokenizer_class") == "TokenizersBackend"
                and (hf_config / "tokenizer.json").is_file()
            ):
                payload["tokenizer_class"] = "PreTrainedTokenizerFast"
            path.write_text(
                json.dumps(_scrub_known_path_fields(payload), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


def validate_artifact(stage_path: Path, plan: BuildPlan) -> dict[str, Any]:
    meta_path = stage_path / "golden_meta_info.json"
    merak_path = stage_path / "merak_config.json"
    missing = [path.name for path in (meta_path, merak_path) if not path.is_file()]
    if missing:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Exported autoregressive Artifact is missing required runtime files: {missing}",
            model=plan.model_id,
            missing=missing,
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"golden_meta_info.json must contain an object: {meta_path}",
            model=plan.model_id,
        )

    visual = meta.get("visual_config")
    dynamic_vision = None
    if plan.vision_gears:
        dynamic_vision = validate_dynamic_vision(
            stage_path,
            visual,
            model_id=plan.model_id,
            vision_profile=plan.vision_profile,
            vision_gears=plan.vision_gears,
        )
    _validate_runtime_lengths(meta, plan)
    _validate_speculative_assets(stage_path, meta, plan)
    _validate_local_runtime_assets(stage_path, meta, plan)
    _validate_tokenizer_snapshot(stage_path / str(meta["hf_config"]), plan)
    golden_artifacts = _collect_golden_artifacts(stage_path, plan)
    return {
        "metadata_file": meta_path.name,
        "golden_meta_info": meta_path.name,
        "merak_config": merak_path.name,
        "components": list(plan.components),
        "golden_components": list(plan.golden_components),
        "golden_artifacts": golden_artifacts,
        "vision_profile": plan.vision_profile,
        "dynamic_vision": dynamic_vision,
        "requires_source": False,
        "loadable": True,
    }


def _collect_golden_artifacts(stage_path: Path, plan: BuildPlan) -> dict[str, list[str]]:
    requested = set(plan.golden_components)
    if not requested:
        return {}
    result: dict[str, list[str]] = {}
    for path in stage_path.rglob("*"):
        if not path.is_file() or "golden" not in {part.lower() for part in path.parts}:
            continue
        relative = path.relative_to(stage_path).as_posix()
        lowered = relative.lower()
        for component in requested:
            aliases = {component.lower()}
            if component == "visual":
                aliases.add("vision")
            if any(alias in lowered for alias in aliases):
                result.setdefault(component, []).append(relative)
                break
    return {component: sorted(paths) for component, paths in sorted(result.items())}


def _replace_source_model_paths(value: Any, local_hf_config: str) -> None:
    if isinstance(value, list):
        for item in value:
            _replace_source_model_paths(item, local_hf_config)
        return
    if not isinstance(value, dict):
        return
    for key, item in list(value.items()):
        if key in {"hf_model", "fallback_hf_model"} and isinstance(item, str):
            value[key] = local_hf_config
        else:
            _replace_source_model_paths(item, local_hf_config)


def _scrub_absolute_paths(value: Any) -> Any:
    if isinstance(value, str) and Path(value).is_absolute():
        return f"source://external/{Path(value).name}"
    if isinstance(value, list):
        return [_scrub_absolute_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _scrub_absolute_paths(item) for key, item in value.items()}
    return value


def _scrub_known_path_fields(value: Any) -> Any:
    path_fields = {"_name_or_path", "name_or_path", "hf_model", "fallback_hf_model"}
    if isinstance(value, list):
        return [_scrub_known_path_fields(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in path_fields and isinstance(item, str) and Path(item).is_absolute():
            result[key] = f"source://external/{Path(item).name}"
        else:
            result[key] = _scrub_known_path_fields(item)
    return result


def _validate_local_runtime_assets(stage_path: Path, meta: Mapping[str, Any], plan: BuildPlan) -> None:
    required: dict[str, tuple[Any, bool]] = {
        "hf_config": (meta.get("hf_config"), True),
        "quant_embedding": (meta.get("quant_embedding"), False),
        "prefill_hmonnx": (meta.get("prefill_hmonnx"), False),
        "decode_hmonnx": (meta.get("decode_hmonnx"), False),
    }
    visual = meta.get("visual_config")
    if "visual" in plan.components:
        if plan.vision_gears and isinstance(visual, Mapping):
            gears = visual.get("gears")
            if isinstance(gears, list):
                for index, gear in enumerate(gears):
                    required[f"visual_config.gears[{index}].hmonnx"] = (
                        gear.get("hmonnx") if isinstance(gear, Mapping) else None,
                        False,
                    )
            else:
                required["visual_config.hmonnx"] = (visual.get("hmonnx"), False)
        else:
            required["visual_config.hmonnx"] = (
                visual.get("hmonnx") if isinstance(visual, Mapping) else None,
                False,
            )
    if "video_visual" in plan.components:
        video = meta.get("video_visual_config")
        required["video_visual_config.hmonnx"] = (
            video.get("hmonnx") if isinstance(video, Mapping) else None,
            False,
        )
    if "audio" in plan.components:
        audio = meta.get("audio_config")
        required["audio_config.hmonnx"] = (
            audio.get("hmonnx") if isinstance(audio, Mapping) else None,
            False,
        )
    for field in getattr(plan, "required_runtime_files", ()):
        required[field] = (meta.get(field), False)

    invalid: list[dict[str, Any]] = []
    root = stage_path.resolve()
    for field, (value, expect_directory) in required.items():
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            invalid.append({"field": field, "value": value, "reason": "must be a relative Artifact path"})
            continue
        path = (root / value).resolve()
        if not path.is_relative_to(root):
            invalid.append({"field": field, "value": value, "reason": "escapes the Artifact directory"})
            continue
        exists = path.is_dir() if expect_directory else path.is_file()
        if not exists:
            invalid.append({"field": field, "value": value, "reason": "runtime asset does not exist"})
    if invalid:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Exported autoregressive Artifact is not self-contained for runtime loading",
            model=plan.model_id,
            invalid_runtime_assets=invalid,
        )


def _validate_speculative_assets(stage_path: Path, meta: Mapping[str, Any], plan: BuildPlan) -> None:
    modes = [component for component in plan.components if component in {"mtp", "dflash"}]
    if not modes:
        return
    expected_mode = modes[0]
    spec_decode = meta.get("spec_decode")
    if not isinstance(spec_decode, Mapping) or spec_decode.get("mode") != expected_mode:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Speculative Artifact is missing its registered draft runtime contract",
            model=plan.model_id,
            expected_mode=expected_mode,
        )
    draft = spec_decode.get("draft")
    num_draft_tokens = spec_decode.get("num_draft_tokens")
    verify_length = spec_decode.get("verify_length")
    invalid = []
    if spec_decode.get("runtime_contract_version") != 2:
        invalid.append({"field": "runtime_contract_version", "value": spec_decode.get("runtime_contract_version")})
    if (
        isinstance(num_draft_tokens, bool)
        or not isinstance(num_draft_tokens, int)
        or num_draft_tokens <= 0
        or verify_length != num_draft_tokens + 1
    ):
        invalid.append({"field": "num_draft_tokens/verify_length", "value": [num_draft_tokens, verify_length]})
    overrides = getattr(plan, "config_overrides", None)
    override_key = "export.model.num_draft_tokens"
    if isinstance(overrides, Mapping) and override_key in overrides and num_draft_tokens != overrides[override_key]:
        invalid.append(
            {
                "field": "num_draft_tokens",
                "value": num_draft_tokens,
                "expected": overrides[override_key],
            }
        )

    if expected_mode == "mtp":
        graph_pairs = [
            (
                "prefill_hmonnx",
                spec_decode.get("draft_prefill_onnx") or spec_decode.get("draft_onnx"),
                "draft_prefill_onnx/draft_onnx",
            ),
            (
                "decode_hmonnx",
                spec_decode.get("draft_decode_onnx") or spec_decode.get("draft_onnx"),
                "draft_decode_onnx/draft_onnx",
            ),
        ]
    else:
        noise_token_id = draft.get("noise_token_id") if isinstance(draft, Mapping) else None
        if isinstance(noise_token_id, bool) or not isinstance(noise_token_id, int) or noise_token_id < 0:
            invalid.append({"field": "draft.noise_token_id", "value": noise_token_id})
        noise_override_key = "export.model.dflash_config.noise_token_id"
        expected_noise_token_id = overrides.get(noise_override_key) if isinstance(overrides, Mapping) else None
        if (
            not isinstance(overrides, Mapping)
            or noise_override_key not in overrides
            or noise_token_id != expected_noise_token_id
        ):
            invalid.append(
                {
                    "field": "draft.noise_token_id",
                    "value": noise_token_id,
                    "expected": expected_noise_token_id,
                }
            )
        graph_pairs = [
            ("context_hmonnx", spec_decode.get("draft_context_onnx"), "draft_context_onnx"),
            (
                "decode_hmonnx",
                spec_decode.get("draft_decode_onnx") or spec_decode.get("draft_onnx"),
                "draft_decode_onnx/draft_onnx",
            ),
        ]
        nested_context_decode = draft.get("context_decode_hmonnx") if isinstance(draft, Mapping) else None
        top_context_decode = spec_decode.get("draft_context_decode_onnx")
        if nested_context_decode is not None or top_context_decode is not None:
            graph_pairs.append(("context_decode_hmonnx", top_context_decode, "draft_context_decode_onnx"))

    root = stage_path.resolve()
    for nested_field, top_value, top_field in graph_pairs:
        nested_value = draft.get(nested_field) if isinstance(draft, Mapping) else None
        nested_path = _confined_runtime_path(root, root, nested_value)
        top_path = _confined_runtime_path(root, root, top_value)
        if nested_path is None:
            invalid.append({"field": f"draft.{nested_field}", "value": nested_value})
        if top_path is None:
            invalid.append({"field": top_field, "value": top_value})
        if nested_path is not None and top_path is not None and nested_path != top_path:
            invalid.append(
                {
                    "field": f"{top_field}/draft.{nested_field}",
                    "value": [top_value, nested_value],
                    "reason": "top-level runtime alias and contract-v2 path disagree",
                }
            )
    if invalid:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Speculative Artifact has an incomplete draft graph set",
            model=plan.model_id,
            mode=expected_mode,
            invalid_runtime_assets=invalid,
        )


def _validate_runtime_lengths(meta: Mapping[str, Any], plan: BuildPlan) -> None:
    model_config = meta.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Autoregressive runtime metadata is missing model_config",
            model=plan.model_id,
        )
    expected = {
        "context_max_length": plan.context_length,
        "prefill_chunk_length": plan.prefill_chunk_length,
        "max_pe_length": plan.max_pe_length,
    }
    mismatches = {
        field: {"expected": value, "actual": model_config.get(field)}
        for field, value in expected.items()
        if value is not None and model_config.get(field) != value
    }
    if mismatches:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "Autoregressive runtime lengths do not match the resolved BuildPlan",
            model=plan.model_id,
            mismatches=mismatches,
        )


def _validate_tokenizer_snapshot(path: Path, plan: BuildPlan) -> None:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        messages = [{"role": "user", "content": "runtime validation"}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(text, return_token_type_ids=False)
        input_ids = encoded.get("input_ids")
        if not isinstance(input_ids, list) or not input_ids:
            raise ValueError("tokenizer produced no input_ids")
    except Exception as exc:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Autoregressive Artifact has no usable local tokenizer snapshot: {exc}",
            model=plan.model_id,
            path=str(path),
        ) from exc


def _confined_runtime_path(root: Path, base: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = base / relative
    path = candidate.resolve()
    if candidate.is_symlink() or not path.is_relative_to(root) or not path.is_file():
        return None
    return path


__all__ = ["normalize_export", "package_runtime", "sanitize_artifact", "validate_artifact"]
