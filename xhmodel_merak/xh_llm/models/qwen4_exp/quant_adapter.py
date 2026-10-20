"""Thin GPTQModel RTN/AutoRound adapter for Qwen4-Exp."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...workflows.result import QuantResult
from .precision import (
    QWEN4_EXP_PROFILE,
    validate_qwen4_exp_config,
)


DEFAULT_CALIBRATION_JSONL = "data/calib_data/NeelNanda-pile-10k.jsonl"
_REPO_RESOURCE_PREFIXES = ("xh2modelzoo://", "repo://")


def quantize_with_qwen4_exp_api(
    *,
    model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
    export_model_cfg: Mapping[str, Any],
    workflow_seed: int,
) -> QuantResult:
    kwargs = build_qwen4_exp_quant_kwargs(
        hf_model_dir=model_dir,
        output_dir=output_dir,
        device=device,
        quant_cfg=quant_cfg,
        export_model_cfg=export_model_cfg,
        workflow_seed=workflow_seed,
    )
    recipe = _load_callable(
        "gptqmodel.recipes.qwen4_exp",
        "quantize_qwen4_exp",
        "Qwen4-Exp quant requires GPTQModel with the RTN/AutoRound qwen4_exp recipe API.",
    )
    result = recipe(**kwargs)
    quanted_model_dir = _result_output_dir(result, kwargs["output_dir"])
    return QuantResult(
        raw_model_dir=model_dir,
        quanted_model_dir=_normalize_path(quanted_model_dir),
        meta=result,
    )


def build_qwen4_exp_quant_kwargs(
    *,
    hf_model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
    export_model_cfg: Mapping[str, Any],
    workflow_seed: int,
) -> dict[str, Any]:
    method, iters, disable_opt_rtn = resolve_quant_method_options(quant_cfg)
    _validate_artifact_format(quant_cfg)
    group_size = _group_size(quant_cfg)
    model_config = _resolve_model_config(hf_model_dir, quant_cfg, export_model_cfg)
    architecture = validate_qwen4_exp_config(model_config)
    rotation = validate_rotation_policy(quant_cfg)
    runtime_cfg = _section(quant_cfg, "runtime")
    calibration_cfg = _section(quant_cfg, "calibration")
    moe_cfg = _section(quant_cfg, "moe")
    vision_quarot = quant_cfg.get("vision_quarot")
    if rotation == "vision_only" and vision_quarot is None:
        vision_quarot = "mean_preserving"

    kwargs: dict[str, Any] = {
        "model_dir": hf_model_dir,
        "output_dir": _quant_output_path(output_dir, quant_cfg),
        "method": method,
        "artifact_format": str(quant_cfg.get("artifact_format") or quant_cfg.get("output_format") or "gptqmodel_hf"),
        "max_quant_layers": quant_cfg.get("max_quant_layers"),
        "language_quarot": False,
        "vision_quarot": vision_quarot,
        "bits": int(quant_cfg.get("bits", QWEN4_EXP_PROFILE.base_bits)),
        "attention_bits": int(moe_cfg.get("attention_bits", moe_cfg.get("attn_bits", 8))),
        "linear_attention_bits": int(moe_cfg.get("linear_attention_bits", 8)),
        "shared_expert_bits": int(moe_cfg.get("shared_expert_bits", 8)),
        "expert_bits": int(moe_cfg.get("expert_bits", QWEN4_EXP_PROFILE.expert_bits)),
        "group_size": group_size,
        "iters": iters,
        "disable_opt_rtn": disable_opt_rtn,
        "nsamples": int(calibration_cfg.get("nsamples", quant_cfg.get("nsamples", 128))),
        "seqlen": int(calibration_cfg.get("seqlen", quant_cfg.get("seqlen", 2048))),
        "batch_size": int(runtime_cfg.get("batch_size", quant_cfg.get("batch_size", 1))),
        "calibration_jsonl": _resolve_dataset_value(
            calibration_cfg.get(
                "dataset",
                calibration_cfg.get("jsonl", quant_cfg.get("dataset", DEFAULT_CALIBRATION_JSONL)),
            )
        ),
        "calibration_text_key": str(calibration_cfg.get("text_key", "text")),
        "device": str(runtime_cfg.get("device", device)),
        "device_map": runtime_cfg.get("device_map"),
        "infer_device_map": runtime_cfg.get("infer_device_map"),
        "auto_forward_data_parallel": _as_bool(runtime_cfg.get("auto_forward_data_parallel", False)),
        "moe_routing": str(moe_cfg.get("routing", "bypass")),
        "moe_routing_batch_size": runtime_cfg.get("moe_routing_batch_size"),
        "wait_for_submodule_finalizers": _as_bool(runtime_cfg.get("wait_for_submodule_finalizers", True)),
        "gc_mode": str(runtime_cfg.get("gc_mode", "on_stage_end")),
        "trust_remote_code": _as_bool(runtime_cfg.get("trust_remote_code", False)),
        "preserve_mtp": _as_bool(quant_cfg.get("preserve_mtp", True)),
        "skip_eval": _as_bool(runtime_cfg.get("skip_eval", True)),
        "dry_run": _as_bool(runtime_cfg.get("dry_run", False)),
        "seed": int(runtime_cfg.get("seed", quant_cfg.get("seed", workflow_seed))),
        "extra_args": runtime_cfg.get("extra_args"),
    }
    # Keep validation evidence local; the external recipe accepts only
    # executable CLI parameters and records its own provenance.
    del architecture
    return {key: value for key, value in kwargs.items() if value is not None}


def resolve_quant_method_options(quant_cfg: Mapping[str, Any]) -> tuple[str, int, bool | None]:
    """Require direct RTN or positive-iteration AutoRound without CLI overrides."""
    algorithm = str(quant_cfg.get("algorithm") or "gptqmodel").lower().replace("-", "_")
    default_method = "autoround" if algorithm in {"autoround", "auto_round"} else "rtn"
    method = str(quant_cfg.get("method") or default_method).lower().replace("-", "_")
    if method == "auto_round":
        method = "autoround"
    if method not in {"rtn", "autoround"}:
        raise ValueError(f"Unsupported Qwen4-Exp quant method: {method!r}")
    iters = quant_cfg.get("iters")
    if iters is None:
        iters = 0 if method == "rtn" else 200
    if isinstance(iters, bool) or not isinstance(iters, int):
        raise ValueError("Qwen4-Exp quant.iters must be an integer")
    disable_opt_rtn = quant_cfg.get("disable_opt_rtn", True if method == "rtn" else None)
    if disable_opt_rtn is not None and not isinstance(disable_opt_rtn, bool):
        raise ValueError("Qwen4-Exp quant.disable_opt_rtn must be boolean")
    if method == "rtn":
        if iters != 0 or disable_opt_rtn is not True:
            raise ValueError("Direct RTN requires iters=0 and disable_opt_rtn=true")
    elif iters <= 0 or disable_opt_rtn is not None:
        raise ValueError("AutoRound requires positive iters and no disable_opt_rtn setting")
    runtime = _section(quant_cfg, "runtime")
    extra_args = runtime.get("extra_args") or []
    if not isinstance(extra_args, (list, tuple)) or not all(isinstance(arg, str) for arg in extra_args):
        raise ValueError("Qwen4-Exp runtime.extra_args must be a list of strings")
    controlled = {"--iters", "--disable-opt-rtn", "--no-disable-opt-rtn", "--language-quarot", "--vision-quarot"}
    if any(arg.split("=", 1)[0] in controlled for arg in extra_args):
        raise ValueError("Qwen4-Exp extra_args cannot override quant method or rotation options")
    return method, iters, disable_opt_rtn


def validate_rotation_policy(quant_cfg: Mapping[str, Any]) -> str | bool:
    value = quant_cfg.get("rotation", False)
    if value in (None, False):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"", "none", "null", "false", "off", "disabled"}:
            return False
        if normalized in {"vision", "vision_only", "vision_quarot", "vision_only_quarot"}:
            return "vision_only"
    scope = str(quant_cfg.get("rotation_scope", "")).strip().lower().replace("-", "_")
    if value is True and scope in {"vision", "vision_only"}:
        return "vision_only"
    raise ValueError(
        "Qwen4-Exp language/global QuaRot is not supported: hyper-connection sigmoid "
        "mixing and PLE/ngram streams are not invariant to hidden-space rotation. "
        "Use rotation=false or rotation='vision_only'."
    )


def _resolve_model_config(
    hf_model_dir: str,
    quant_cfg: Mapping[str, Any],
    export_model_cfg: Mapping[str, Any],
) -> Mapping[str, Any]:
    for source in (
        quant_cfg.get("model_config"),
        export_model_cfg.get("model_config"),
        quant_cfg.get("hf_config"),
        export_model_cfg.get("hf_config"),
    ):
        if isinstance(source, Mapping):
            return source
    config_path = Path(hf_model_dir) / "config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = json.load(stream)
        if isinstance(loaded, Mapping):
            return loaded
    raise ValueError("Qwen4-Exp quant requires a model_config mapping or hf_model_dir/config.json")


def _load_callable(module_name: str, callable_name: str, message: str):
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(message) from exc
    try:
        return getattr(module, callable_name)
    except AttributeError as exc:
        raise ImportError(message) from exc


def _result_output_dir(result: Any, fallback: str) -> str:
    if result is None:
        return fallback
    if isinstance(result, Mapping):
        return str(result.get("output_dir") or result.get("quanted_model_dir") or fallback)
    return str(getattr(result, "output_dir", None) or getattr(result, "quanted_model_dir", None) or fallback)


def _validate_artifact_format(quant_cfg: Mapping[str, Any]) -> None:
    artifact_format = quant_cfg.get("artifact_format")
    output_format = quant_cfg.get("output_format")
    if artifact_format is not None and output_format is not None and artifact_format != output_format:
        raise ValueError("quant.artifact_format and quant.output_format must match when both are provided")
    resolved = artifact_format or output_format or "gptqmodel_hf"
    if resolved != "gptqmodel_hf":
        raise ValueError(f"Qwen4-Exp quant requires artifact_format='gptqmodel_hf', got {resolved!r}")


def _group_size(quant_cfg: Mapping[str, Any]) -> int:
    group_size = int(quant_cfg.get("group_size", QWEN4_EXP_PROFILE.group_size))
    if group_size != QWEN4_EXP_PROFILE.group_size:
        raise ValueError(f"Qwen4-Exp quant group_size must be {QWEN4_EXP_PROFILE.group_size}, got {group_size!r}")
    return group_size


def _section(quant_cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = quant_cfg.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"quant.{name} must be a mapping when provided")
    return value


def _quant_output_path(output_dir: str, quant_cfg: Mapping[str, Any]) -> str:
    return _normalize_path(quant_cfg.get("save_path") or quant_cfg.get("output_dir") or output_dir)


def _resolve_dataset_value(value: Any) -> str:
    text = os.path.expanduser(os.path.expandvars(str(value)))
    if any(text.startswith(prefix) for prefix in _REPO_RESOURCE_PREFIXES):
        relative_path = text
        for prefix in _REPO_RESOURCE_PREFIXES:
            if relative_path.startswith(prefix):
                relative_path = relative_path.removeprefix(prefix)
                break
        relative_path = relative_path.lstrip("/")
        return str((_repo_root() / relative_path).resolve())
    if Path(text).expanduser().is_file():
        return str(Path(text).expanduser().resolve())
    return text


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs_merak").is_dir() and (parent / "xhmodel_merak").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.normpath(str(path)))


# Preserve the previous Python import names while using the neutral producer API.
build_qwen4_exp_autoround_kwargs = build_qwen4_exp_quant_kwargs
quantize_with_autoround_api = quantize_with_qwen4_exp_api


__all__ = [
    "build_qwen4_exp_quant_kwargs",
    "quantize_with_qwen4_exp_api",
    "resolve_quant_method_options",
    "build_qwen4_exp_autoround_kwargs",
    "quantize_with_autoround_api",
    "validate_rotation_policy",
]
