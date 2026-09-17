"""Compare Qwen3 QUANTED_ALIGNED and HMONNX at every exported layer Tag.

The two paths consume the same prompt, then independently feed back their own
greedy tokens. Reports identify the first divergence; later layer comparisons
include differences in token history. The default ``exported`` prefill shape matches
the fixed HMONNX prefill graph.  Use ``--quanted-prefill-shape dynamic`` as a
diagnostic experiment for the legacy forward script's prompt-length prefill.
"""

from __future__ import annotations

import argparse
import csv
import errno
import gc
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MODEL_DIR = "data/models/Qwen3-8B-GPTQ-Int4"
DEFAULT_CONFIG_PATH = "configs_merak/xh2a/llm_models/qwen3/8b/qwen3_8b_xh2a_w4a8_hf_gptq_2k.py"
DEFAULT_OUTPUT_DIR = "work_dirs/qwen3_8b_error_analysis/compare_exported"
_LAYER_TAG_PATTERN = re.compile(r"^layer_(\d+)$")
_PREFILL_STAGE_PATTERN = re.compile(r"^prefill(?:\.\d+)?$")
_CAPTURE_STAGE_PATTERN = re.compile(r"^(?:prefill(?:\.\d+)?|decode(?:\.\d+)?)$")
_ROTARY_PATTERN = re.compile(
    r"^(?P<stage>prefill(?:\.\d+)?|decode(?:\.\d+)?)\.rotary\.(?P<component>cos|sin)$"
)
_POSITION_EMBEDDINGS_PATTERN = re.compile(
    r"^(?P<stage>prefill(?:\.\d+)?|decode(?:\.\d+)?)\.layer\.\d+\.position_embeddings\."
    r"(?P<component>cos|sin)$"
)
_ACTIVE_QUANTED_STORE: CaptureStore | None = None
_HASH_CHUNK_ELEMENTS = 16 * 1024 * 1024
_QWEN3_LINEAR_PROJECTIONS_PER_LAYER = 7


@dataclass
class CapturedTensor:
    tensor: torch.Tensor
    dtype: str
    shape: tuple[int, ...]


class CaptureStore:
    """CPU activation store shared by layer hooks and cache snapshots."""

    def __init__(self) -> None:
        self.tensors: dict[str, CapturedTensor] = {}
        self.stage = "prefill"
        self.valid_sequence_length: int | None = None

    def set_stage(self, stage: str, *, valid_sequence_length: int | None = None) -> None:
        if _CAPTURE_STAGE_PATTERN.fullmatch(stage) is None:
            raise ValueError(f"unsupported capture stage: {stage}")
        self.stage = stage
        self.valid_sequence_length = valid_sequence_length

    def record(self, name: str, value: Any) -> None:
        tensor = _as_tensor(value)
        if tensor is None:
            return
        detached = tensor.detach()
        self.tensors[f"{self.stage}.{name}"] = CapturedTensor(
            tensor=detached.float().cpu().contiguous(),
            dtype=str(detached.dtype).removeprefix("torch."),
            shape=tuple(detached.shape),
        )


def _load_merak_config_local(config_path: str, model_dir: str, seed: int) -> tuple[Any, Any]:
    from xhmodel_merak.xh_llm import AutoLLMConfig
    from xhquant.api import Config

    cfg = Config.fromfile(config_path)
    if hasattr(cfg, "hf_model_dir"):
        cfg.hf_model_dir = model_dir
    cfg.model.hf_model = model_dir
    cfg.seed = seed
    return cfg, AutoLLMConfig.from_pretrained(cfg.model)


def _load_xh_model(*, config_path: str, model_dir: str, seed: int, work_dir: Path) -> Any:
    from xhmodel_merak.xh_llm import AutoLLMModel
    from xhquant.api import get_xhquant_logger

    cfg, model_cfg = _load_merak_config_local(config_path, model_dir, seed)
    logger = get_xhquant_logger()
    logger.info(f"Config:\n{cfg.pretty_text}")
    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.work_dir = str(work_dir)
    logger.info(f"Resolved model type: {type(xh_model).__name__}")
    return xh_model


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    try:
        return mapping[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported dtype {name!r}; choose one of {sorted(mapping)}") from error


def _select_cuda_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)


def _enable_kvcache_fast_mode() -> None:
    from xhquant.core.context import get_current_context

    get_current_context().kvcache.fast_mode = True


def _set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _release_cuda_memory(device: torch.device | None = None) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    if device is not None and device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    else:
        torch.cuda.empty_cache()


def _input_ids_sha256(input_ids: torch.Tensor) -> str:
    values = input_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def _decode_stage(step: int, max_decode_steps: int) -> str:
    if step < 0 or step >= max_decode_steps:
        raise ValueError(f"decode step {step} is outside [0, {max_decode_steps})")
    return "decode" if max_decode_steps == 1 else f"decode.{step:04d}"


def _prefill_chunks(prompt_length: int, chunk_length: int) -> list[tuple[str, int, int]]:
    """Return stable report stage names and token ranges for chunked prefill."""
    if prompt_length <= 0:
        raise ValueError("input_ids must contain at least one token")
    if chunk_length <= 0:
        raise ValueError(f"prefill chunk length must be positive, got {chunk_length}")
    ranges = [(start, min(start + chunk_length, prompt_length)) for start in range(0, prompt_length, chunk_length)]
    if len(ranges) == 1:
        start, end = ranges[0]
        return [("prefill", start, end)]
    return [(f"prefill.{index:04d}", start, end) for index, (start, end) in enumerate(ranges)]


def _assert_same_input_ids(reference: torch.Tensor, candidate: torch.Tensor) -> None:
    reference = reference.detach().to(device="cpu", dtype=torch.int64)
    candidate = candidate.detach().to(device="cpu", dtype=torch.int64)
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"input_ids shape mismatch: reference={tuple(reference.shape)}, candidate={tuple(candidate.shape)}"
        )
    if torch.equal(reference, candidate):
        return
    first = tuple(int(index) for index in torch.nonzero(reference != candidate, as_tuple=False)[0].tolist())
    raise RuntimeError(
        f"input_ids differ at {first}: reference={int(reference[first])}, candidate={int(candidate[first])}"
    )


def _remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def _slice_cache(tensor: torch.Tensor, *, axis: int, valid_length: int, start: int = 0) -> torch.Tensor:
    axis = axis if axis >= 0 else tensor.ndim + axis
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"cache axis {axis} is invalid for shape {tuple(tensor.shape)}")
    slices = [slice(None)] * tensor.ndim
    slices[axis] = slice(start, valid_length)
    return tensor[tuple(slices)]


def _record_wrap_cache(store: CaptureStore, xh_model: Any, *, valid_length: int, start: int = 0) -> None:
    cache_axis = int(getattr(xh_model.kvcache_config, "cache_axis", 2))
    for layer_index, (key_cache, value_cache) in enumerate(
        zip(xh_model.past_key_caches, xh_model.past_value_caches, strict=True)
    ):
        store.record(
            f"cache.{layer_index:03d}.key",
            _slice_cache(key_cache.detach(), axis=cache_axis, start=start, valid_length=valid_length),
        )
        store.record(
            f"cache.{layer_index:03d}.value",
            _slice_cache(value_cache.detach(), axis=cache_axis, start=start, valid_length=valid_length),
        )


def _safe_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _tensor_metrics(reference: CapturedTensor | None, candidate: CapturedTensor | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "reference_shape": list(reference.shape) if reference is not None else None,
        "candidate_shape": list(candidate.shape) if candidate is not None else None,
        "comparison_shape": None,
        "broadcast_compared": False,
        "reference_dtype": reference.dtype if reference is not None else None,
        "candidate_dtype": candidate.dtype if candidate is not None else None,
        "exact": False,
        "max_abs": None,
        "mean_abs": None,
        "rmse": None,
        "max_rel": None,
        "cosine": None,
        "reference_nonfinite": None,
        "candidate_nonfinite": None,
        "different_elements": None,
        "total_elements": None,
        "different_ratio": None,
    }
    if reference is None or candidate is None:
        row["status"] = "missing"
        return row
    if reference.shape != candidate.shape:
        row["status"] = "shape_mismatch"
        return row

    reference_tensor = reference.tensor
    candidate_tensor = candidate.tensor
    row["comparison_shape"] = list(reference_tensor.shape)
    row["reference_nonfinite"] = int((~torch.isfinite(reference_tensor)).sum().item())
    row["candidate_nonfinite"] = int((~torch.isfinite(candidate_tensor)).sum().item())
    row["exact"] = bool(torch.equal(reference_tensor, candidate_tensor))
    row["status"] = "exact" if row["exact"] else "different"
    total_elements = reference_tensor.numel()
    different_elements = int(torch.count_nonzero(reference_tensor != candidate_tensor).item())
    row["total_elements"] = total_elements
    row["different_elements"] = different_elements
    row["different_ratio"] = different_elements / total_elements if total_elements else 0.0
    if total_elements == 0:
        row.update(max_abs=0.0, mean_abs=0.0, rmse=0.0, max_rel=0.0, cosine=1.0)
        return row

    difference = (reference_tensor - candidate_tensor).abs()
    relative = difference / reference_tensor.abs().clamp_min(1e-12)
    row["max_abs"] = _safe_number(float(difference.max().item()))
    row["mean_abs"] = _safe_number(float(difference.mean().item()))
    row["rmse"] = _safe_number(float(torch.sqrt(torch.mean((reference_tensor - candidate_tensor) ** 2)).item()))
    row["max_rel"] = _safe_number(float(relative.max().item()))
    reference_flat = reference_tensor.reshape(-1)
    candidate_flat = candidate_tensor.reshape(-1)
    if torch.count_nonzero(reference_flat) == 0 and torch.count_nonzero(candidate_flat) == 0:
        cosine = 1.0
    else:
        cosine = float(F.cosine_similarity(reference_flat, candidate_flat, dim=0, eps=1e-12).item())
    row["cosine"] = _safe_number(cosine)
    return row


def _first_output_token_mismatch(
    reference_tokens: list[int],
    candidate_tokens: list[int],
    *,
    input_length: int,
    max_decode_steps: int,
) -> dict[str, Any] | None:
    for generation_index in range(max(len(reference_tokens), len(candidate_tokens))):
        reference_token = reference_tokens[generation_index] if generation_index < len(reference_tokens) else None
        candidate_token = candidate_tokens[generation_index] if generation_index < len(candidate_tokens) else None
        if reference_token == candidate_token:
            continue
        prediction_stage = "prefill" if generation_index == 0 else _decode_stage(generation_index - 1, max_decode_steps)
        return {
            "output_token_number": generation_index + 1,
            "generation_index": generation_index,
            "position_id": input_length + generation_index,
            "prediction_stage": prediction_stage,
            "native_token_id": reference_token,
            "wrap_token_id": candidate_token,
        }
    return None


def _summarize_components(
    rows: list[dict[str, Any]],
    pattern: re.Pattern[str],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for component in ("cos", "sin"):
        component_rows = [
            row
            for row in rows
            if (match := pattern.fullmatch(row["name"])) is not None and match.group("component") == component
        ]
        total_elements = sum(int(row["total_elements"] or 0) for row in component_rows)
        different_elements = sum(int(row["different_elements"] or 0) for row in component_rows)
        comparable_rows = [row for row in component_rows if row["total_elements"] is not None]
        summary[component] = {
            "exact": bool(component_rows) and all(bool(row["exact"]) for row in component_rows),
            "compared_tensors": len(component_rows),
            "different_tensors": sum(not bool(row["exact"]) for row in component_rows),
            "total_elements": total_elements,
            "different_elements": different_elements,
            "different_ratio": different_elements / total_elements if total_elements else 0.0,
            "max_abs": max(
                (float(row["max_abs"]) for row in comparable_rows if row["max_abs"] is not None),
                default=None,
            ),
        }
    return summary


def _summarize_position_embeddings_by_stage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stages = {
        match.group("stage")
        for row in rows
        if (match := _POSITION_EMBEDDINGS_PATTERN.fullmatch(row["name"])) is not None
    }

    def stage_sort_key(stage: str) -> tuple[int, int]:
        if stage.startswith("prefill"):
            return (0, 0 if stage == "prefill" else int(stage.rsplit(".", maxsplit=1)[1]))
        return (1, 0 if stage == "decode" else int(stage.rsplit(".", maxsplit=1)[1]))

    result: dict[str, dict[str, Any]] = {}
    for stage in sorted(stages, key=stage_sort_key):
        stage_rows = [
            row
            for row in rows
            if (match := _POSITION_EMBEDDINGS_PATTERN.fullmatch(row["name"])) is not None
            and match.group("stage") == stage
        ]
        components = _summarize_components(stage_rows, _POSITION_EMBEDDINGS_PATTERN)
        for component, component_summary in components.items():
            component_summary["mismatched_position_ids"] = sorted(
                {
                    position_id
                    for row in stage_rows
                    if row["name"].endswith(f".{component}")
                    for position_id in row.get("mismatched_position_ids", [])
                }
            )
        result[stage] = components
    return result


def _build_report(
    reference: CaptureStore,
    candidate: CaptureStore,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    layer_output_pattern = re.compile(r"^(prefill(?:\.\d+)?|decode(?:\.\d+)?)\.layer\.(\d+)\.output$")
    rows: list[dict[str, Any]] = []
    first_bad: dict[str, int | None] = {"prefill": None, "decode": None}
    for name in sorted(set(reference.tensors) | set(candidate.tensors)):
        row = {"name": name, **_tensor_metrics(reference.tensors.get(name), candidate.tensors.get(name))}
        if name.endswith(".logits") and row["status"] not in {"missing", "shape_mismatch"}:
            reference_argmax = int(reference.tensors[name].tensor.reshape(-1).argmax().item())
            candidate_argmax = int(candidate.tensors[name].tensor.reshape(-1).argmax().item())
            row.update(
                reference_argmax=reference_argmax,
                candidate_argmax=candidate_argmax,
                argmax_equal=reference_argmax == candidate_argmax,
            )
        rows.append(row)
        match = layer_output_pattern.fullmatch(name)
        if match is not None and not row["exact"]:
            capture_stage, layer_index_text = match.groups()
            stage = "decode" if capture_stage.startswith("decode") else "prefill"
            if first_bad[stage] is None:
                first_bad[stage] = int(layer_index_text)

    reference_tokens = [int(token) for token in metadata.get("native_generated_token_ids", [])]
    candidate_tokens = [int(token) for token in metadata.get("wrap_generated_token_ids", [])]
    max_decode_steps = max(int(metadata.get("max_decode_steps", 0)), 1)
    mismatch = _first_output_token_mismatch(
        reference_tokens,
        candidate_tokens,
        input_length=int(metadata.get("input_shape", [0, 0])[-1]),
        max_decode_steps=max_decode_steps,
    )
    if mismatch is not None:
        index = int(mismatch["generation_index"])
        reference_texts = metadata.get("native_generated_token_texts", [])
        candidate_texts = metadata.get("wrap_generated_token_texts", [])
        mismatch["native_token_text"] = reference_texts[index] if index < len(reference_texts) else None
        mismatch["wrap_token_text"] = candidate_texts[index] if index < len(candidate_texts) else None
    return {
        "metadata": metadata,
        "summary": {
            "exact": all(bool(row["exact"]) for row in rows) and mismatch is None,
            "compared_tensors": len(rows),
            "different_tensors": sum(not bool(row["exact"]) for row in rows),
            "first_bad_layer": first_bad,
            "rotary": _summarize_components(rows, _ROTARY_PATTERN),
            "position_embeddings": _summarize_components(rows, _POSITION_EMBEDDINGS_PATTERN),
            "position_embeddings_by_stage": _summarize_position_embeddings_by_stage(rows),
            "output_tokens_exact": mismatch is None,
            "first_output_token_mismatch": mismatch,
        },
        "comparisons": rows,
    }


def _write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "layerwise_diff.json"
    csv_path = output_dir / "layerwise_diff.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = [
        "name",
        "status",
        "exact",
        "same_token_history",
        "reference_shape",
        "candidate_shape",
        "comparison_shape",
        "broadcast_compared",
        "reference_dtype",
        "candidate_dtype",
        "max_abs",
        "mean_abs",
        "rmse",
        "max_rel",
        "cosine",
        "reference_nonfinite",
        "candidate_nonfinite",
        "different_elements",
        "total_elements",
        "different_ratio",
        "reference_argmax",
        "candidate_argmax",
        "argmax_equal",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["comparisons"]:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return json_path, csv_path


def _extract_query(data: Any, *, source: str) -> str:
    if not isinstance(data, dict) or not isinstance(data.get("query"), str):
        raise TypeError(f"{source} must contain an object with a string 'query' field")
    return data["query"]


def _read_prompt(value: str) -> str:
    path = Path(value)
    try:
        is_file = path.is_file()
    except OSError as error:
        if error.errno != errno.ENAMETOOLONG:
            raise
        is_file = False
    if not is_file:
        return value
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return _extract_query(json.loads(text), source=str(path))
    if path.suffix.lower() == ".jsonl":
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"prompt file is empty: {path}")
        return _extract_query(json.loads(lines[0]), source=str(path))
    return text


def _as_tensor(value: Any) -> torch.Tensor | None:
    """Find a tensor, including values wrapped by xhquant.QTensor."""
    if isinstance(value, torch.Tensor):
        return value
    dequantize = getattr(value, "dequantize", None)
    if callable(dequantize):
        candidate = dequantize()
        if isinstance(candidate, torch.Tensor):
            return candidate
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _as_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _as_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _last_valid_token_tensor(store: CaptureStore, value: Any, *, label: str) -> torch.Tensor:
    """Select the prompt tail from full prefill output or the only gathered output."""
    tensor = _as_tensor(value)
    if tensor is None:
        raise TypeError(f"{label} returned no tensor: {type(value).__name__}")
    if tensor.ndim < 3:
        return tensor
    if _PREFILL_STAGE_PATTERN.fullmatch(store.stage) is not None and tensor.shape[1] != 1:
        valid_length = store.valid_sequence_length
        if valid_length is None or not 0 < valid_length <= tensor.shape[1]:
            raise RuntimeError(
                f"{label} cannot select valid prefill tail: valid_length={valid_length}, shape={tuple(tensor.shape)}"
            )
        return tensor[:, valid_length - 1 : valid_length, ...]
    return tensor[:, -1:, ...]


def _tensor_fingerprint(value: Any) -> dict[str, Any] | None:
    """Return a stable full-tensor hash without making one giant CPU copy."""
    tensor = _as_tensor(value)
    if tensor is None:
        return None
    flat = tensor.detach().reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, flat.numel(), _HASH_CHUNK_ELEMENTS):
        chunk = flat[start : start + _HASH_CHUNK_ELEMENTS].contiguous().cpu()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": digest.hexdigest(),
    }


def _optional_attr(value: Any, name: str, default: Any) -> Any:
    if value is None:
        return default
    try:
        return getattr(value, name)
    except (AttributeError, KeyError, TypeError):
        return default


def _linear_quant_spec(module: nn.Module) -> dict[str, Any]:
    """Read the effective W/A bits and weight FP mode from eager or HMONNX Linear."""
    w_qspec = _optional_attr(_optional_attr(module, "w_cfg", None), "qspec", None)
    i_qspec = _optional_attr(_optional_attr(module, "i_cfg", None), "qspec", None)

    weight_mode = _optional_attr(w_qspec, "fp_mode", None)
    if weight_mode is None:
        weight_mode = _optional_attr(module, "weight_mode", None)
    if weight_mode is None:
        weight_mode = _optional_attr(module, "fp_mode", "")

    weight_bits = _optional_attr(w_qspec, "man_bit", None)
    if weight_bits is None:
        weight_bits = _optional_attr(module, "weight_bits", None)
    if weight_bits is None:
        weight_bits = _optional_attr(module, "hmfp_weight_man_bit", -1)

    activation_bits = _optional_attr(i_qspec, "man_bit", None)
    if activation_bits is None:
        activation_bits = _optional_attr(module, "activation_bits", None)
    if activation_bits is None:
        activation_bits = _optional_attr(module, "hmfp_act_man_bit", -1)

    mode = str(weight_mode).lower()
    return {
        "weight_bits": int(weight_bits),
        "activation_bits": int(activation_bits),
        "weight_mode": mode,
        "scheme": f"w{int(weight_bits)}a{int(activation_bits)}_{mode or 'unknown'}",
    }


def _quantized_linear_modules(roots: list[nn.Module]) -> list[tuple[str, nn.Module]]:
    candidates: list[tuple[str, nn.Module]] = []
    seen: set[int] = set()
    for root_index, root in enumerate(roots):
        for name, module in root.named_modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if not isinstance(getattr(module, "in_features", None), int):
                continue
            if not isinstance(getattr(module, "out_features", None), int):
                continue
            if not hasattr(module, "qweight") or not hasattr(module, "scale_or_exp"):
                continue
            display_name = name or f"<root_{root_index}:{type(module).__name__}>"
            candidates.append((display_name, module))
    return candidates


def _select_lm_head(candidates: list[tuple[str, nn.Module]], *, label: str) -> tuple[str, nn.Module]:
    if not candidates:
        raise RuntimeError(f"cannot locate {label} quantized Linear modules")
    max_out_features = max(int(module.out_features) for _, module in candidates)
    widest = [(name, module) for name, module in candidates if int(module.out_features) == max_out_features]
    if len(widest) != 1:
        names = [name for name, _ in widest]
        raise RuntimeError(f"cannot uniquely locate {label} lm_head: widest modules={names}")
    return widest[0]


def _quantization_contract_audit(
    roots: list[nn.Module],
    *,
    label: str,
    expected_other_count: int,
) -> dict[str, Any]:
    """Check lm_head=W8A8 SEFP and every other quantized Linear=W4A8 SSFP."""
    candidates = _quantized_linear_modules(roots)
    lm_head_name, lm_head = _select_lm_head(candidates, label=label)
    lm_head_spec = _linear_quant_spec(lm_head)
    lm_head_exact = (
        lm_head_spec["weight_bits"] == 8
        and lm_head_spec["activation_bits"] == 8
        and lm_head_spec["weight_mode"] == "sefp"
    )

    other_specs: Counter[str] = Counter()
    violations: list[dict[str, Any]] = []
    for name, module in candidates:
        if module is lm_head:
            continue
        spec = _linear_quant_spec(module)
        other_specs[spec["scheme"]] += 1
        if not (spec["weight_bits"] == 4 and spec["activation_bits"] == 8 and spec["weight_mode"] == "ssfp"):
            violations.append({"module_name": name, **spec})

    other_count = len(candidates) - 1
    expected_total_count = expected_other_count + 1
    inventory_exact = len(candidates) == expected_total_count and other_count == expected_other_count
    other_exact = inventory_exact and not violations
    return {
        "label": label,
        "exact": lm_head_exact and other_exact,
        "total_linear_modules": len(candidates),
        "expected_total_linear_modules": expected_total_count,
        "inventory_exact": inventory_exact,
        "lm_head": {
            "module_name": lm_head_name,
            "module_type": type(lm_head).__name__,
            "expected_scheme": "w8a8_sefp",
            "exact": lm_head_exact,
            **lm_head_spec,
        },
        "other_linears": {
            "count": other_count,
            "expected_count": expected_other_count,
            "expected_scheme": "w4a8_ssfp",
            "exact": other_exact,
            "scheme_counts": dict(sorted(other_specs.items())),
            "violations": violations,
        },
    }


def _lm_head_fingerprint(roots: list[nn.Module], *, label: str) -> dict[str, Any]:
    """Fingerprint the Linear with the largest output width (Qwen3 lm_head)."""
    name, module = _select_lm_head(_quantized_linear_modules(roots), label=label)
    out_features = int(module.out_features)
    qweight = _tensor_fingerprint(getattr(module, "qweight", None))
    scale_or_exp = _tensor_fingerprint(getattr(module, "scale_or_exp", None))
    if qweight is None or scale_or_exp is None:
        raise RuntimeError(f"{label} lm_head has no fixed qweight/scale_or_exp after execution")
    quant_spec = _linear_quant_spec(module)
    w_cfg = _optional_attr(module, "w_cfg", None)
    qspec = _optional_attr(w_cfg, "qspec", None)
    weight_rounding = _optional_attr(module, "hmfp_weight_rounding", None)
    if weight_rounding is None:
        weight_rounding = _optional_attr(qspec, "rounding_mode", "")
    return {
        "module_name": name,
        "module_type": type(module).__name__,
        "in_features": int(module.in_features),
        "out_features": out_features,
        "weight_mode": quant_spec["weight_mode"],
        "weight_bits": quant_spec["weight_bits"],
        "activation_bits": quant_spec["activation_bits"],
        "weight_rounding": str(weight_rounding),
        "qweight": qweight,
        "scale_or_exp": scale_or_exp,
    }


def _record_layer_boundary(store: CaptureStore, layer_index: int, output: Any) -> None:
    tensor = _as_tensor(output)
    if tensor is None:
        raise TypeError(f"layer_{layer_index} Tag returned no tensor: {type(output).__name__}")
    valid_length = store.valid_sequence_length
    if _PREFILL_STAGE_PATTERN.fullmatch(store.stage) is not None and valid_length is not None and tensor.ndim >= 3:
        if tensor.shape[1] < valid_length:
            raise RuntimeError(
                f"layer_{layer_index} sequence length {tensor.shape[1]} is shorter than "
                f"the valid prefill length {valid_length}"
            )
        tensor = tensor[:, :valid_length, ...]
    store.record(f"layer.{layer_index:03d}.output", tensor)


def _tag_hook(store: CaptureStore, layer_index: int):
    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        _record_layer_boundary(store, layer_index, output)

    return hook


def _layer_tag_modules(root: nn.Module) -> dict[int, nn.Module]:
    result: dict[int, nn.Module] = {}
    for _, module in root.named_modules():
        match = _LAYER_TAG_PATTERN.fullmatch(str(getattr(module, "tag_name", "")))
        if match is None:
            continue
        layer_index = int(match.group(1))
        previous = result.setdefault(layer_index, module)
        if previous is not module:
            raise RuntimeError(f"multiple Tag modules expose layer_{layer_index}")
    return result


def _validate_tag_set(tags: dict[int, nn.Module], *, expected_layers: int, label: str) -> None:
    expected = set(range(expected_layers))
    actual = set(tags)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(f"{label} layer Tag mismatch: missing={missing}, unexpected={unexpected}")


def _register_quanted_tag_hooks(
    root: nn.Module,
    store: CaptureStore,
    *,
    expected_layers: int,
) -> list[Any]:
    tags = _layer_tag_modules(root)
    # The current QUANTED_ALIGNED graph retains xh::_pipe_tag as an FX
    # call_function node.  Older graph conversions materialized XHTag modules,
    # so keep module hooks as a compatibility path and use dispatch capture for
    # the current representation.
    if tags:
        _validate_tag_set(tags, expected_layers=expected_layers, label="QUANTED_ALIGNED modules")
        handles = [tags[index].register_forward_hook(_tag_hook(store, index)) for index in sorted(tags)]
        rmsnorm_modules = [module for _, module in root.named_modules() if "rmsnorm" in type(module).__name__.lower()]
        if not rmsnorm_modules:
            raise RuntimeError("cannot locate final RMSNorm in QUANTED_ALIGNED module graph")

        def final_norm_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _last_valid_token_tensor(store, output, label="QUANTED_ALIGNED final_norm")
            store.record("final_norm", tensor)

        handles.append(rmsnorm_modules[-1].register_forward_hook(final_norm_hook))
        return handles

    graph = getattr(root, "graph", None)
    if graph is None:
        raise RuntimeError("QUANTED_ALIGNED exposes neither Tag modules nor an FX graph")
    graph_tags: set[int] = set()
    for node in graph.nodes:
        if node.op != "call_function" or str(node.target) != "xh._pipe_tag.default":
            continue
        if len(node.args) < 2:
            continue
        match = _LAYER_TAG_PATTERN.fullmatch(str(node.args[1]))
        if match is None:
            continue
        graph_tags.add(int(match.group(1)))
        # The custom op accepts Tensor but not xhquant.QTensor in aligned eager
        # execution.  Replacing this identity marker with an identity Python
        # callback both makes Tag-enabled QUANTED_ALIGNED runnable and exposes
        # its dequantized boundary value without changing model semantics.
        node.target = _capture_quanted_pipe_tag
    _validate_tag_set(
        {index: root for index in graph_tags},
        expected_layers=expected_layers,
        label="QUANTED_ALIGNED FX graph",
    )
    rmsnorm_nodes = []
    for node in graph.nodes:
        if node.op != "call_module":
            continue
        module = root.get_submodule(str(node.target))
        if "rmsnorm" in type(module).__name__.lower():
            rmsnorm_nodes.append(node)
    if not rmsnorm_nodes:
        raise RuntimeError("cannot locate final RMSNorm in QUANTED_ALIGNED FX graph")
    final_norm_node = rmsnorm_nodes[-1]
    with graph.inserting_after(final_norm_node):
        capture_node = graph.call_function(
            _capture_quanted_named_value,
            args=(final_norm_node, "final_norm"),
        )
    for user in list(final_norm_node.users):
        if user is not capture_node:
            user.replace_input_with(final_norm_node, capture_node)
    root.recompile()
    return []


def _capture_quanted_pipe_tag(value: Any, tag_name: str, _tag_type: str, _content: str) -> Any:
    match = _LAYER_TAG_PATTERN.fullmatch(str(tag_name))
    if match is not None and _ACTIVE_QUANTED_STORE is not None:
        _record_layer_boundary(_ACTIVE_QUANTED_STORE, int(match.group(1)), value)
    return value


def _capture_quanted_named_value(value: Any, name: str) -> Any:
    if _ACTIVE_QUANTED_STORE is not None:
        tensor = _last_valid_token_tensor(_ACTIVE_QUANTED_STORE, value, label=f"QUANTED_ALIGNED {name}")
        _ACTIVE_QUANTED_STORE.record(name, tensor)
    return value


def _validate_captured_layers(
    store: CaptureStore,
    *,
    stages: list[str],
    expected_layers: int,
    label: str,
) -> None:
    expected = set(range(expected_layers))
    for stage in stages:
        prefix = f"{stage}.layer."
        actual = {
            int(name[len(prefix) :].split(".", maxsplit=1)[0])
            for name in store.tensors
            if name.startswith(prefix) and name.endswith(".output")
        }
        if actual != expected:
            raise RuntimeError(
                f"{label} captured layer mismatch at {stage}: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )


def _hmonnx_module_roots(hmonnx_component: Any) -> list[nn.Module]:
    session = hmonnx_component.hmonnx_session
    initialize = getattr(session, "initialize", None)
    if callable(initialize):
        initialize()

    candidates = [session, getattr(session, "_session", None)]
    roots: list[nn.Module] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        for value in (candidate, getattr(candidate, "graph_module", None)):
            if isinstance(value, nn.Module) and id(value) not in seen:
                roots.append(value)
                seen.add(id(value))
        node_modules = getattr(candidate, "node_modules", None)
        if node_modules is not None:
            for value in node_modules:
                if isinstance(value, nn.Module) and id(value) not in seen:
                    roots.append(value)
                    seen.add(id(value))
    if not roots:
        raise RuntimeError(f"cannot locate modules below HMONNX session {type(session).__name__}")
    return roots


def _register_hmonnx_tag_hooks(
    hmonnx_model: Any,
    store: CaptureStore,
    *,
    expected_layers: int,
) -> list[Any]:
    handles: list[Any] = []
    for phase, component in (
        ("prefill", hmonnx_model.prefill_model),
        ("decode", hmonnx_model.decode_model),
    ):
        tags: dict[int, nn.Module] = {}
        roots = _hmonnx_module_roots(component)
        graph_final_norm_modules: list[nn.Module] = []
        ordered_rmsnorm_modules: list[nn.Module] = []
        saw_fx_graph = False
        for root in roots:
            if "rmsnorm" in type(root).__name__.lower():
                ordered_rmsnorm_modules.append(root)
            for layer_index, module in _layer_tag_modules(root).items():
                previous = tags.setdefault(layer_index, module)
                if previous is not module:
                    raise RuntimeError(f"multiple HMONNX {phase} Tag modules expose layer_{layer_index}")
            graph = getattr(root, "graph", None)
            if graph is not None:
                saw_fx_graph = True
                rmsnorm_modules = [
                    root.get_submodule(str(node.target))
                    for node in graph.nodes
                    if node.op == "call_module"
                    and "rmsnorm" in type(root.get_submodule(str(node.target))).__name__.lower()
                ]
                if rmsnorm_modules:
                    graph_final_norm_modules.append(rmsnorm_modules[-1])
        _validate_tag_set(tags, expected_layers=expected_layers, label=f"HMONNX {phase}")
        handles.extend(tags[index].register_forward_hook(_tag_hook(store, index)) for index in sorted(tags))
        candidates = graph_final_norm_modules if saw_fx_graph else ordered_rmsnorm_modules
        candidates = list({id(module): module for module in candidates}.values())
        if not candidates:
            continue
        final_norm = candidates[-1]

        def final_norm_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _last_valid_token_tensor(store, output, label="HMONNX final_norm")
            store.record("final_norm", tensor)

        handles.append(final_norm.register_forward_hook(final_norm_hook))
    return handles


def _tokenize(tokenizer: Any, prompt: str, *, think: bool) -> tuple[list[dict[str, str]], torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=think,
    )
    inputs = tokenizer([text], return_tensors="pt", truncation=True)
    return messages, inputs.input_ids.to(dtype=torch.long, device="cpu")


def _token_texts(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]


def _add_generation_comparison(report: dict[str, Any], tokenizer: Any, hmonnx_tokenizer: Any) -> None:
    """Compare independent continuations, decoding prefixes to preserve byte-token semantics."""
    meta = report["metadata"]
    reference = meta["quanted_generated_token_ids"]
    candidate = meta["hmonnx_generated_token_ids"]
    report["generated_outputs"] = {
        "quanted_aligned": {
            "token_ids": list(reference),
            "text": tokenizer.decode(reference, skip_special_tokens=False),
        },
        "hmonnx": {
            "token_ids": list(candidate),
            "text": hmonnx_tokenizer.decode(candidate, skip_special_tokens=False),
        },
    }
    first_token = None
    first_text = None
    steps = []
    for index in range(max(len(reference), len(candidate))):
        left = reference[index] if index < len(reference) else None
        right = candidate[index] if index < len(candidate) else None
        left_text = tokenizer.decode(reference[:index + 1], skip_special_tokens=False)
        right_text = hmonnx_tokenizer.decode(candidate[:index + 1], skip_special_tokens=False)
        stage = (meta["prefill_chunks"][-1]["stage"] if index == 0
                 else _decode_stage(index - 1, meta["max_decode_steps"]))
        step = {
            "generation_index": index,
            "output_token_number": index + 1,
            "position_id": meta["input_shape"][-1] + index,
            "prediction_stage": stage,
            "quanted_token_id": left,
            "hmonnx_token_id": right,
            "quanted_text": left_text,
            "hmonnx_text": right_text,
            "same_token_history": reference[:index] == candidate[:index],
            "tokens_equal": left == right,
            "texts_equal": left_text == right_text,
        }
        steps.append(step)
        if first_token is None and left != right:
            first_token = step.copy()
        if first_text is None and left_text != right_text:
            first_text = step.copy()
            common = 0
            for a, b in zip(left_text, right_text, strict=False):
                if a != b:
                    break
                common += 1
            first_text["character_index"] = common
    stage_histories = {step["prediction_stage"]: step["same_token_history"] for step in steps}
    for row in report["comparisons"]:
        # Tensor names have a suffix; extract the stage independently of that suffix.
        stage_name = re.match(r"^(prefill(?:\.\d+)?|decode(?:\.\d+)?)\.", row["name"]).group(1)
        row["same_token_history"] = stage_histories.get(stage_name, True)
    report["generation_comparison"] = steps
    report["summary"].update(
        first_output_token_mismatch=first_token,
        first_output_text_mismatch=first_text,
        output_tokens_exact=reference == candidate,
        output_texts_exact=all(step["texts_equal"] for step in steps),
    )
    report["summary"]["exact"] = report["summary"]["exact"] and first_token is None and first_text is None


def _add_logits_diagnosis(report: dict[str, Any]) -> None:
    """Locate the first captured logits mismatch in execution order, not name order."""
    meta = report["metadata"]
    stages = [chunk["stage"] for chunk in meta["prefill_chunks"]]
    stages += [_decode_stage(i, meta["max_decode_steps"]) for i in range(meta["max_decode_steps"])]
    rows = {row["name"]: row for row in report["comparisons"]}
    generation = {step["prediction_stage"]: step for step in report["generation_comparison"]}
    first = None
    issues = []
    prior_cache_differences = []
    for stage in stages:
        logits = rows.get(f"{stage}.logits")
        if logits is None or logits.get("status") in {"missing", "shape_mismatch"}:
            issues.append({"stage": stage, "status": "missing" if logits is None else logits["status"]})
        elif not logits["exact"] and first is None:
            layers = sorted(
                (row for name, row in rows.items()
                 if name.startswith(f"{stage}.layer.") and name.endswith(".output") and not row["exact"]),
                key=lambda row: int(row["name"].split(".layer.")[1].split(".")[0]),
            )
            norm = rows.get(f"{stage}.final_norm")
            same_history = logits.get("same_token_history")
            phase = "prefill" if stage.startswith("prefill") else "decode"
            head = meta.get("quanted_lm_head")
            exported_head = meta.get("hmonnx_lm_heads", {}).get(phase)
            parameters_exact = (
                all(head.get(key) is not None and head[key] == exported_head.get(key)
                    for key in ("qweight", "scale_or_exp"))
                if head is not None and exported_head is not None else None
            )
            evidence = []
            if same_history is False:
                evidence.append("输入 token 历史已分叉，当前误差包含不同输入的影响。")
            if prior_cache_differences:
                evidence.append("此前阶段已观察到 KV cache 差异，可能传播到当前阶段。")
            if layers:
                evidence.append(f"本阶段最早观察到差异的层边界：{layers[0]['name']}；不能据此认定该层是根因。")
            if norm is not None and not norm["exact"]:
                evidence.append("final_norm 已不同，lm_head 的浮点输入已存在差异。")
            if parameters_exact is False:
                evidence.append("当前路径的 lm_head 权重或指数指纹不同，需核对权重来源及导出转换。")
            if norm is not None and norm["exact"] and parameters_exact is True:
                evidence.append("final_norm 与 lm_head 参数一致，需检查输入量化、Linear 执行和输出处理。")
            step = generation.get(stage, {})
            chunk = next((item for item in meta["prefill_chunks"] if item["stage"] == stage), None)
            prediction_position = step.get("position_id") if chunk is None else chunk.get("end")
            first = {
                "stage": stage,
                "execution_index": stages.index(stage),
                "output_token_number": step.get("output_token_number"),
                "generation_index": step.get("generation_index"),
                "prediction_position_id": prediction_position,
                "input_position_id": prediction_position - 1 if prediction_position is not None else None,
                "used_for_generation": stage in generation,
                "logits": logits,
                "first_different_layer": layers[0]["name"] if layers else None,
                "final_norm": norm,
                "lm_head_parameters_exact": parameters_exact,
                "prior_cache_differences": list(prior_cache_differences),
                "evidence": evidence,
                "root_cause": "尚未确定；以上是边界观测，需相同输入下的算子级比对确认。",
            }
        prior_cache_differences.extend(
            name for name, row in rows.items() if name.startswith(f"{stage}.cache.") and not row["exact"]
        )
    report["summary"]["first_logits_mismatch"] = first
    report["summary"]["logits_comparison_issues"] = issues


def _run_quanted_aligned(
    *,
    model_dir: str,
    config_path: str,
    input_ids: torch.Tensor,
    messages: list[dict[str, str]],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    think: bool,
    work_dir: Path,
    max_new_tokens: int,
    prefill_shape: str,
) -> tuple[CaptureStore, list[int], dict[str, Any]]:
    from xhmodel_merak.xh_llm import LLMInferenceContextManager, LLMModelState
    from xhquant.api import set_random_seed

    _select_cuda_device(device)
    _enable_kvcache_fast_mode()
    _set_determinism(seed)
    set_random_seed(seed)
    os.environ["LAYER_TAG_ENABLE"] = "1"

    xh_model = _load_xh_model(
        config_path=config_path,
        model_dir=model_dir,
        seed=seed,
        work_dir=work_dir / "quanted_aligned",
    )
    xh_model.config.enable_auto_offload = False
    xh_model.config.enable_prefill_chunk = False
    xh_model.set_state(LLMModelState.QUANTED_ALIGNED)
    xh_model.to(device=device, dtype=dtype)
    xh_model.eval()

    processor_inputs = xh_model.get_tf_processor().apply_chat_template(messages, enable_think=think)
    _assert_same_input_ids(input_ids, processor_inputs.input_ids)
    inference_model = xh_model.get_inference_model()
    if not isinstance(inference_model, nn.Module):
        raise RuntimeError(f"QUANTED_ALIGNED inference model is not nn.Module: {type(inference_model).__name__}")

    expected_layers = int(xh_model.kvcache_config.num_layers)
    chunk_length = int(xh_model.config.prefill_chunk_length)
    prompt_length = int(input_ids.shape[1])
    prefill_chunks = _prefill_chunks(prompt_length, chunk_length)

    store = CaptureStore()
    handles = _register_quanted_tag_hooks(inference_model, store, expected_layers=expected_layers)
    generated: list[int] = []
    lm_head_fingerprint: dict[str, Any] | None = None
    quantization_contract: dict[str, Any] | None = None
    quanted_prefill_lengths: list[int] = []
    global _ACTIVE_QUANTED_STORE
    _ACTIVE_QUANTED_STORE = store
    try:
        with LLMInferenceContextManager(xh_model, devices=[device]), torch.inference_mode():
            xh_model.set_prefill()
            logits_tensor: torch.Tensor | None = None
            for stage, start, end in prefill_chunks:
                valid_chunk_length = end - start
                quanted_prefill_length = chunk_length if prefill_shape == "exported" else valid_chunk_length
                quanted_prefill_lengths.append(quanted_prefill_length)
                xh_model.set_input_sequence_length(quanted_prefill_length)
                data_processor = xh_model.get_data_preprocessor()
                store.set_stage(stage, valid_sequence_length=valid_chunk_length)
                inputs = data_processor(
                    {
                        "input_ids": input_ids[:, start:end],
                        "past_seq_length": start,
                    }
                )
                logits = xh_model.forward(*inputs)
                logits_tensor = _last_valid_token_tensor(store, logits, label=f"QUANTED_ALIGNED {stage} logits")
                store.record("logits", logits_tensor)
                _record_wrap_cache(store, xh_model, start=start, valid_length=end)
            if logits_tensor is None:
                raise RuntimeError("QUANTED_ALIGNED prefill did not execute")
            if max_new_tokens:
                generated.append(int(logits_tensor[:, -1, :].argmax(dim=-1).item()))

            if max_new_tokens > 1:
                xh_model.set_decode()
            data_processor = xh_model.get_data_preprocessor()
            for step in range(max_new_tokens - 1):
                teacher_token = generated[-1]
                past_length = prompt_length + step
                store.set_stage(
                    _decode_stage(step, max_new_tokens - 1),
                    valid_sequence_length=past_length + 1,
                )
                decode_ids = torch.tensor([[teacher_token]], dtype=torch.long)
                inputs = data_processor({"input_ids": decode_ids, "past_seq_length": past_length})
                logits = xh_model.forward(*inputs)
                logits_tensor = _last_valid_token_tensor(store, logits, label="QUANTED_ALIGNED logits")
                store.record("logits", logits_tensor)
                generated.append(int(logits_tensor[:, -1, :].argmax(dim=-1).item()))
                _record_wrap_cache(
                    store,
                    xh_model,
                    start=past_length,
                    valid_length=past_length + 1,
                )
        captured_stages = [stage for stage, _, _ in prefill_chunks] + [
            _decode_stage(step, max_new_tokens - 1) for step in range(max_new_tokens - 1)
        ]
        _validate_captured_layers(
            store,
            stages=captured_stages,
            expected_layers=expected_layers,
            label="QUANTED_ALIGNED",
        )
        lm_head_fingerprint = _lm_head_fingerprint([inference_model], label="QUANTED_ALIGNED")
        quantization_contract = _quantization_contract_audit(
            [inference_model],
            label="QUANTED_ALIGNED",
            expected_other_count=expected_layers * _QWEN3_LINEAR_PROJECTIONS_PER_LAYER,
        )
    finally:
        _ACTIVE_QUANTED_STORE = None
        _remove_hooks(handles)
        del xh_model, inference_model
        _release_cuda_memory(device)

    return (
        store,
        generated,
        {
            "expected_layers": expected_layers,
            "exported_prefill_chunk_length": chunk_length,
            "prefill_chunks": [
                {"stage": stage, "start": start, "end": end, "valid_length": end - start}
                for stage, start, end in prefill_chunks
            ],
            "quanted_prefill_input_sequence_length": quanted_prefill_lengths[-1],
            "quanted_prefill_input_sequence_lengths": quanted_prefill_lengths,
            "quanted_lm_head": lm_head_fingerprint,
            "quanted_quantization_contract": quantization_contract,
        },
    )


def _run_hmonnx(
    *,
    meta_path: str,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    expected_layers: int,
) -> tuple[CaptureStore, list[int], dict[str, Any], Any]:
    from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager

    _select_cuda_device(device)
    hmonnx_model = AutoLLMHONNXModel.from_pretrained(meta_path, device_map=[device])
    if dtype != torch.float16:
        raise ValueError("exported Qwen3 HMONNX metadata uses float16; --dtype must be fp16")
    hmonnx_model.to(device=device)
    prompt_length = int(input_ids.shape[1])
    chunk_length = int(hmonnx_model.meta_info.model_config.prefill_chunk_length)
    prefill_chunks = _prefill_chunks(prompt_length, chunk_length)

    store = CaptureStore()
    handles = _register_hmonnx_tag_hooks(hmonnx_model, store, expected_layers=expected_layers)
    predictions: list[int] = []
    try:
        with LLMInferenceContextManager(hmonnx_model), torch.inference_mode():
            hmonnx_model.set_prefill()
            data_processor = hmonnx_model.get_data_preprocessor()
            logits_tensor: torch.Tensor | None = None
            for stage, start, end in prefill_chunks:
                store.set_stage(stage, valid_sequence_length=end - start)
                inputs = data_processor(
                    {
                        "input_ids": input_ids[:, start:end],
                        "past_seq_length": start,
                    }
                )
                logits = hmonnx_model.forward(*inputs)
                logits_tensor = _last_valid_token_tensor(store, logits, label=f"HMONNX {stage} logits")
                store.record("logits", logits_tensor)
                _record_wrap_cache(store, hmonnx_model, start=start, valid_length=end)
            if logits_tensor is None:
                raise RuntimeError("HMONNX prefill did not execute")
            if max_new_tokens:
                predictions.append(int(logits_tensor[:, -1, :].argmax(dim=-1).item()))

            if max_new_tokens > 1:
                hmonnx_model.set_decode()
            data_processor = hmonnx_model.get_data_preprocessor()
            for step in range(max_new_tokens - 1):
                past_length = prompt_length + step
                store.set_stage(
                    _decode_stage(step, max_new_tokens - 1),
                    valid_sequence_length=past_length + 1,
                )
                decode_ids = torch.tensor([[predictions[-1]]], dtype=torch.long)
                inputs = data_processor({"input_ids": decode_ids, "past_seq_length": past_length})
                logits = hmonnx_model.forward(*inputs)
                logits_tensor = _last_valid_token_tensor(store, logits, label="HMONNX logits")
                store.record("logits", logits_tensor)
                predictions.append(int(logits_tensor[:, -1, :].argmax(dim=-1).item()))
                _record_wrap_cache(
                    store,
                    hmonnx_model,
                    start=past_length,
                    valid_length=past_length + 1,
                )
        captured_stages = [stage for stage, _, _ in prefill_chunks] + [
            _decode_stage(step, max_new_tokens - 1) for step in range(max_new_tokens - 1)
        ]
        _validate_captured_layers(
            store,
            stages=captured_stages,
            expected_layers=expected_layers,
            label="HMONNX",
        )
    finally:
        _remove_hooks(handles)

    hmonnx_lm_heads: dict[str, dict[str, Any]] = {}
    hmonnx_quantization_contracts: dict[str, dict[str, Any]] = {}
    for phase, component in (
        ("prefill", hmonnx_model.prefill_model),
        ("decode", hmonnx_model.decode_model),
    ):
        roots = _hmonnx_module_roots(component)
        hmonnx_lm_heads[phase] = _lm_head_fingerprint(roots, label=f"HMONNX {phase}")
        hmonnx_quantization_contracts[phase] = _quantization_contract_audit(
            roots,
            label=f"HMONNX {phase}",
            expected_other_count=expected_layers * _QWEN3_LINEAR_PROJECTIONS_PER_LAYER,
        )
    return (
        store,
        predictions,
        {
            "hmonnx_prefill_chunk_length": chunk_length,
            "hmonnx_prefill_chunks": [
                {"stage": stage, "start": start, "end": end, "valid_length": end - start}
                for stage, start, end in prefill_chunks
            ],
            "hmonnx_lm_heads": hmonnx_lm_heads,
            "hmonnx_quantization_contracts": hmonnx_quantization_contracts,
        },
        hmonnx_model.get_tokenizer(),
    )


def _write_summary(report: dict[str, Any], output_dir: Path) -> Path:
    summary = report["summary"]
    mismatch = summary["first_output_token_mismatch"]
    contract = report["quantization_contract"]
    prefill_chunks = report["metadata"]["prefill_chunks"]
    chunk_ranges = ", ".join(f"{chunk['stage']}=[{chunk['start']}:{chunk['end']})" for chunk in prefill_chunks)
    lines = [
        "# QUANTED_ALIGNED 与 HMONNX 逐层比对",
        "",
        f"- 总体完全一致：`{summary['exact']}`",
        f"- 量化契约完全符合：`{contract['exact']}`",
        f"- lm_head 参数指纹完全一致：`{summary['lm_head_parameters_exact']}`",
        f"- Prefill chunks：`{len(prefill_chunks)}`（{chunk_ranges}）",
        f"- 差异 tensor：`{summary['different_tensors']}/{summary['compared_tensors']}`",
        f"- Prefill 首个差异层：`{summary['first_bad_layer']['prefill']}`",
        f"- Decode 首个差异层：`{summary['first_bad_layer']['decode']}`",
        f"- 首个输出 token 差异：`{mismatch}`",
        f"- 首个生成文本差异：`{summary['first_output_text_mismatch']}`",
        "- 两端独立 greedy 生成，固定执行 max_new_tokens（遇到 EOS 也继续）。",
        "- 首次 token 分叉的下一步起，逐层差异包含输入 token 历史不同的影响。",
        "",
        "## 最终生成结果",
        "",
        "仅包含新增 token，不含 prompt；文本由完整生成序列解码，保留特殊 token。",
        "以下为 JSON 表示，文本中的换行显示为转义字符。",
        "",
        *["    " + line for line in json.dumps(
            report["generated_outputs"], ensure_ascii=False, indent=2
        ).splitlines()],
        "",
        "## 首次 logits 不一致",
        "",
        "按 prefill chunk、decode 的实际执行顺序统计，每次只比较最后有效位置的 logits。",
        "若存在缺失或 shape 不匹配，该结果仅代表可比较阶段中的首次数值差异。",
        "中间 prefill chunk 的 logits 不用于生成，生成序号为 null。索引从 0 开始，输出序号从 1 开始。",
        "",
        *["    " + line for line in json.dumps({
            "first_logits_mismatch": summary["first_logits_mismatch"],
            "logits_comparison_issues": summary["logits_comparison_issues"],
        }, ensure_ascii=False, indent=2).splitlines()],
        "",
        "## Linear 量化契约",
        "",
        "预期：`lm_head = W8A8 SEFP`；其余量化 Linear 全部为 `W4A8 SSFP`。",
        "",
        "| 路径 | Linear 总数 | lm_head 实际配置 | 其他 Linear 配置及数量 | 状态 |",
        "|---|---:|---|---|---:|",
    ]
    for name, audit in contract["paths"].items():
        other_schemes = ", ".join(
            f"{scheme}: {count}" for scheme, count in audit["other_linears"]["scheme_counts"].items()
        )
        lines.append(
            f"| {name} | {audit['total_linear_modules']} | {audit['lm_head']['scheme']} | "
            f"{other_schemes or '-'} | {'exact' if audit['exact'] else 'VIOLATION'} |"
        )
    violations = [
        (path_name, violation)
        for path_name, audit in contract["paths"].items()
        for violation in audit["other_linears"]["violations"]
    ]
    bad_heads = [
        (path_name, audit["lm_head"]) for path_name, audit in contract["paths"].items() if not audit["lm_head"]["exact"]
    ]
    bad_inventories = [
        (path_name, audit) for path_name, audit in contract["paths"].items() if not audit["inventory_exact"]
    ]
    if bad_heads or bad_inventories or violations:
        lines.extend(["", "### 量化契约违规节点", ""])
        for path_name, head in bad_heads:
            lines.append(
                f"- `{path_name}:{head['module_name']}`：期望 `{head['expected_scheme']}`，实际 `{head['scheme']}`"
            )
        for path_name, audit in bad_inventories:
            lines.append(
                f"- `{path_name}`：期望 {audit['expected_total_linear_modules']} 个量化 Linear，"
                f"实际发现 {audit['total_linear_modules']} 个"
            )
        for path_name, violation in violations:
            lines.append(f"- `{path_name}:{violation['module_name']}`：期望 `w4a8_ssfp`，实际 `{violation['scheme']}`")
    lines.extend(
        [
            "",
            "## 逐层数值误差",
            "",
            "| 名称 | 状态 | max_abs | mean_abs | cosine |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["comparisons"]:
        if row["name"].endswith((".output", ".final_norm", ".logits")):
            lines.append(
                f"| {row['name']} | {row['status']} | {row['max_abs']} | {row['mean_abs']} | {row['cosine']} |"
            )
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _apply_report_integrity(
    report: dict[str, Any],
    *,
    quantization_contract_exact: bool,
    lm_head_parameters_exact: bool,
) -> None:
    summary = report["summary"]
    summary["quantization_contract_exact"] = quantization_contract_exact
    summary["lm_head_parameters_exact"] = lm_head_parameters_exact
    summary["exact"] = bool(summary["exact"] and quantization_contract_exact and lm_head_parameters_exact)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmonnx-meta", required=True, help="Fresh export's golden_meta_info.json.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--prompt", default="1+1等于几", help="Prompt text or UTF-8 text/JSON/JSONL file.")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--quanted-prefill-shape",
        choices=("exported", "dynamic"),
        default="exported",
        help="Use the exported 256-token shape or the forward script's prompt-length shape.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fail-on-diff", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    for label, value in (
        ("model directory", args.model_dir),
        ("Merak config", args.config_path),
        ("HMONNX metadata", args.hmonnx_meta),
    ):
        if not Path(value).exists():
            raise FileNotFoundError(f"{label} does not exist: {value}")

    from transformers import AutoTokenizer

    from xhquant.api import xhquant_init

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(output_dir / "compare.log"), False)
    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)
    prompt = _read_prompt(args.prompt)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    messages, input_ids = _tokenize(tokenizer, prompt, think=args.think)

    quanted_store, quanted_tokens, quanted_meta = _run_quanted_aligned(
        model_dir=args.model_dir,
        config_path=args.config_path,
        input_ids=input_ids,
        messages=messages,
        device=device,
        dtype=dtype,
        seed=args.seed,
        think=args.think,
        work_dir=output_dir,
        max_new_tokens=args.max_new_tokens,
        prefill_shape=args.quanted_prefill_shape,
    )
    hmonnx_store, hmonnx_tokens, hmonnx_meta, hmonnx_tokenizer = _run_hmonnx(
        meta_path=args.hmonnx_meta,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        device=device,
        dtype=dtype,
        expected_layers=quanted_meta["expected_layers"],
    )

    hmonnx_messages, hmonnx_input_ids = _tokenize(hmonnx_tokenizer, prompt, think=args.think)
    if hmonnx_messages != messages:
        raise RuntimeError("HMONNX and checkpoint chat messages differ")
    _assert_same_input_ids(input_ids, hmonnx_input_ids)
    if quanted_meta["prefill_chunks"] != hmonnx_meta["hmonnx_prefill_chunks"]:
        raise RuntimeError(
            "QUANTED_ALIGNED and HMONNX prefill chunk boundaries differ: "
            f"quanted={quanted_meta['prefill_chunks']}, hmonnx={hmonnx_meta['hmonnx_prefill_chunks']}"
        )
    quanted_lm_head = quanted_meta["quanted_lm_head"]
    hmonnx_lm_heads = hmonnx_meta["hmonnx_lm_heads"]
    lm_head_parameters_exact = all(
        quanted_lm_head[key]["sha256"] == phase_fingerprint[key]["sha256"]
        for phase_fingerprint in hmonnx_lm_heads.values()
        for key in ("qweight", "scale_or_exp")
    )
    quantization_contract = {
        "expected": {
            "lm_head": "w8a8_sefp",
            "other_linears": "w4a8_ssfp",
        },
        "paths": {
            "quanted_aligned": quanted_meta["quanted_quantization_contract"],
            "hmonnx_prefill": hmonnx_meta["hmonnx_quantization_contracts"]["prefill"],
            "hmonnx_decode": hmonnx_meta["hmonnx_quantization_contracts"]["decode"],
        },
    }
    quantization_contract["exact"] = all(audit["exact"] for audit in quantization_contract["paths"].values())
    metadata = {
        "reference": "quanted_aligned",
        "candidate": "hmonnx",
        "teacher_forcing": False,
        "generation_mode": "independent_greedy_fixed_length",
        "prompt": prompt,
        "think": args.think,
        "input_shape": list(input_ids.shape),
        "input_ids": input_ids.tolist(),
        "input_ids_sha256": _input_ids_sha256(input_ids),
        "max_decode_steps": max(args.max_new_tokens - 1, 0),
        "native_generated_token_ids": quanted_tokens,
        "wrap_generated_token_ids": hmonnx_tokens,
        "native_generated_token_texts": _token_texts(tokenizer, quanted_tokens),
        "wrap_generated_token_texts": _token_texts(tokenizer, hmonnx_tokens),
        "quanted_generated_token_ids": quanted_tokens,
        "hmonnx_generated_token_ids": hmonnx_tokens,
        "quanted_prefill_shape": args.quanted_prefill_shape,
        "lm_head_parameters_exact": lm_head_parameters_exact,
        **quanted_meta,
        **hmonnx_meta,
    }
    report = _build_report(quanted_store, hmonnx_store, metadata=metadata)
    _add_generation_comparison(report, tokenizer, hmonnx_tokenizer)
    _add_logits_diagnosis(report)
    report["quantization_contract"] = quantization_contract
    _apply_report_integrity(
        report,
        quantization_contract_exact=quantization_contract["exact"],
        lm_head_parameters_exact=lm_head_parameters_exact,
    )
    json_path, csv_path = _write_report(report, output_dir)
    summary_path = _write_summary(report, output_dir)
    print(f"report: {json_path}")
    print(f"table: {csv_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 1 if args.fail_on_diff and not report["summary"]["exact"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
