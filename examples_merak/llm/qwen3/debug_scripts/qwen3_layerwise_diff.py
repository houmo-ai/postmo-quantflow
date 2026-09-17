"""Deterministic layerwise comparison between native and Merak-wrapped Qwen3.

The two models are loaded sequentially so a large checkpoint does not need to
be resident twice. Both sides consume the same token IDs. The script compares a
real cache-backed prefill and configurable teacher-forced decode steps.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from examples_merak.llm.qwen3.debug_scripts.qwen3_xh2a_forward import _read_prompt


DEFAULT_MODEL_DIR = "data/models/Qwen3-8B-GPTQ-Int4"
DEFAULT_CONFIG_PATH = "configs_merak/workflows/xh2a/llm_models/qwen3/8b/qwen3_8b_xh2a_w4a8.yaml"
DEFAULT_WORK_DIR = "work_dirs/qwen3_8b_layerwise_diff"
_LAYER_OUTPUT_PATTERN = re.compile(r"^(prefill|decode(?:\.\d+)?)\.layer\.(\d+)\.output$")
_DECODE_STAGE_PATTERN = re.compile(r"^decode(?:\.\d+)?$")


@dataclass
class CapturedTensor:
    tensor: torch.Tensor
    dtype: str
    shape: tuple[int, ...]


class CaptureStore:
    """CPU activation store shared by module hooks and cache snapshots."""

    def __init__(self) -> None:
        self.tensors: dict[str, CapturedTensor] = {}
        self.stage = "prefill"
        self.valid_sequence_length: int | None = None
        self.active_layer: int | None = None
        self._rope_calls: dict[tuple[str, int], int] = {}

    def set_stage(self, stage: str, *, valid_sequence_length: int | None = None) -> None:
        if stage != "prefill" and _DECODE_STAGE_PATTERN.fullmatch(stage) is None:
            raise ValueError(f"unsupported capture stage: {stage}")
        self.stage = stage
        self.valid_sequence_length = valid_sequence_length
        self.active_layer = None

    def record(self, name: str, value: Any, *, last_token: bool = False) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            return
        if (
            name.endswith(".softmax")
            and self.valid_sequence_length is not None
            and tensor.shape[-1] > self.valid_sequence_length
        ):
            tensor = tensor[..., : self.valid_sequence_length]
        if last_token and tensor.ndim >= 3:
            tensor = tensor[:, -1:, ...]
        detached = tensor.detach()
        self.tensors[f"{self.stage}.{name}"] = CapturedTensor(
            tensor=detached.float().cpu().contiguous(),
            dtype=str(detached.dtype).removeprefix("torch."),
            shape=tuple(detached.shape),
        )

    def record_pair(self, name: str, value: Any) -> None:
        if not isinstance(value, (tuple, list)) or len(value) < 2:
            return
        self.record(f"{name}.cos", value[0])
        self.record(f"{name}.sin", value[1])

    def record_rope(self, value: Any) -> None:
        if self.active_layer is None:
            return
        call_key = (self.stage, self.active_layer)
        call_index = self._rope_calls.get(call_key, 0)
        rope_name = "rope_q" if call_index == 0 else "rope_k" if call_index == 1 else f"rope_{call_index}"
        self._rope_calls[call_key] = call_index + 1
        self.record(f"layer.{self.active_layer:03d}.{rope_name}", value)


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _valid_attention_scores(store: CaptureStore, scores: torch.Tensor) -> torch.Tensor:
    valid_length = store.valid_sequence_length or scores.shape[-1]
    scores = scores[..., :valid_length]
    query_length = scores.shape[-2]
    past_length = max(valid_length - query_length, 0)
    query_positions = torch.arange(query_length, device=scores.device) + past_length
    key_positions = torch.arange(valid_length, device=scores.device)
    future_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    return scores.masked_fill(future_mask, 0)


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported dtype {name!r}; choose one of {sorted(mapping)}") from error


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


def _input_ids_sha256(input_ids: torch.Tensor) -> str:
    values = input_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def _decode_stage(step: int, decode_steps: int) -> str:
    """Keep the historical one-step key while making multi-step keys unique."""
    if step < 0 or step >= decode_steps:
        raise ValueError(f"decode step {step} is outside [0, {decode_steps})")
    return "decode" if decode_steps == 1 else f"decode.{step:03d}"


def _assert_same_input_ids(reference: torch.Tensor, candidate: torch.Tensor) -> None:
    reference = reference.detach().to(device="cpu", dtype=torch.int64)
    candidate = candidate.detach().to(device="cpu", dtype=torch.int64)
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"native/wrap input_ids shape mismatch: native={tuple(reference.shape)}, wrap={tuple(candidate.shape)}"
        )
    if torch.equal(reference, candidate):
        return
    mismatch = torch.nonzero(reference != candidate, as_tuple=False)
    first = tuple(int(index) for index in mismatch[0].tolist())
    raise RuntimeError(
        f"native/wrap input_ids differ at {first}: native={int(reference[first])}, wrap={int(candidate[first])}"
    )


def _find_text_model(root: nn.Module) -> nn.Module:
    pending = [root]
    visited: set[int] = set()
    while pending:
        module = pending.pop(0)
        if id(module) in visited:
            continue
        visited.add(id(module))
        if hasattr(module, "layers") and hasattr(module, "norm"):
            return module
        for attribute in ("model", "language_model", "text_model"):
            child = getattr(module, attribute, None)
            if isinstance(child, nn.Module):
                pending.append(child)
    raise RuntimeError(f"cannot locate Qwen3 text decoder below {type(root).__name__}")


def _module_output_hook(store: CaptureStore, name: str, *, last_token: bool = False):
    def hook(_module, _args, _kwargs, output):
        store.record(name, output, last_token=last_token)

    return hook


def _module_input_hook(store: CaptureStore, name: str):
    def hook(_module, args, kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        store.record(name, hidden_states)

    return hook


def _register_capture_hooks(
    text_model: nn.Module,
    embedding: nn.Module,
    store: CaptureStore,
    *,
    capture_details: bool,
) -> list[Any]:
    handles: list[Any] = []
    handles.append(embedding.register_forward_hook(_module_output_hook(store, "embedding"), with_kwargs=True))
    handles.append(
        text_model.norm.register_forward_hook(
            _module_output_hook(store, "final_norm", last_token=True),
            with_kwargs=True,
        )
    )

    rotary_emb = getattr(text_model, "rotary_emb", None)
    if isinstance(rotary_emb, nn.Module):

        def rotary_hook(_module, _args, _kwargs, output):
            store.record_pair("rotary", output)

        handles.append(rotary_emb.register_forward_hook(rotary_hook, with_kwargs=True))

    for layer_index, layer in enumerate(text_model.layers):
        layer_prefix = f"layer.{layer_index:03d}"
        handles.append(
            layer.register_forward_pre_hook(
                _module_input_hook(store, f"{layer_prefix}.input"),
                with_kwargs=True,
            )
        )
        handles.append(
            layer.register_forward_hook(
                _module_output_hook(store, f"{layer_prefix}.output"),
                with_kwargs=True,
            )
        )

        attention = getattr(layer, "self_attn", None)
        if not isinstance(attention, nn.Module):
            continue

        def attention_pre_hook(_module, _args, _kwargs, *, index=layer_index):
            store.active_layer = index

        def attention_post_hook(_module, _args, _kwargs, output, *, index=layer_index):
            store.record(f"layer.{index:03d}.self_attn", output)
            store.active_layer = None

        handles.append(attention.register_forward_pre_hook(attention_pre_hook, with_kwargs=True))

        if capture_details:
            detail_modules = {
                "input_layernorm": getattr(layer, "input_layernorm", None),
                "q_proj": getattr(attention, "q_proj", None),
                "k_proj": getattr(attention, "k_proj", None),
                "v_proj": getattr(attention, "v_proj", None),
                "q_norm": getattr(attention, "q_norm", None),
                "k_norm": getattr(attention, "k_norm", None),
                "o_proj": getattr(attention, "o_proj", None),
                "post_attention_layernorm": getattr(layer, "post_attention_layernorm", None),
                "mlp": getattr(layer, "mlp", None),
            }
            for detail_name, detail_module in detail_modules.items():
                if isinstance(detail_module, nn.Module):
                    handles.append(
                        detail_module.register_forward_hook(
                            _module_output_hook(store, f"{layer_prefix}.{detail_name}"),
                            with_kwargs=True,
                        )
                    )

            rope = getattr(attention, "rope", None)
            if isinstance(rope, nn.Module):

                def rope_hook(_module, _args, _kwargs, output):
                    store.record_rope(output)

                handles.append(rope.register_forward_hook(rope_hook, with_kwargs=True))

            masked_softmax = getattr(attention, "masked_softmax", None)
            if isinstance(masked_softmax, nn.Module):

                def softmax_input_hook(_module, args, kwargs, *, index=layer_index):
                    scores = args[0] if args else kwargs.get("x")
                    if isinstance(scores, torch.Tensor):
                        store.record(
                            f"layer.{index:03d}.attention_scores",
                            _valid_attention_scores(store, scores),
                        )

                handles.append(
                    masked_softmax.register_forward_pre_hook(
                        softmax_input_hook,
                        with_kwargs=True,
                    )
                )
                handles.append(
                    masked_softmax.register_forward_hook(
                        _module_output_hook(store, f"{layer_prefix}.softmax"),
                        with_kwargs=True,
                    )
                )

        handles.append(attention.register_forward_hook(attention_post_hook, with_kwargs=True))
    return handles


@contextmanager
def _native_attention_capture(store: CaptureStore) -> Iterator[None]:
    """Capture functional RoPE and eager softmax outputs used by native Qwen3."""

    from transformers.models.qwen3 import modeling_qwen3

    original_rope = modeling_qwen3.apply_rotary_pos_emb
    original_softmax = F.softmax

    def apply_rotary_pos_emb(*args, **kwargs):
        query_states, key_states = original_rope(*args, **kwargs)
        if store.active_layer is not None:
            layer_prefix = f"layer.{store.active_layer:03d}"
            store.record(f"{layer_prefix}.rope_q", query_states)
            store.record(f"{layer_prefix}.rope_k", key_states)
        return query_states, key_states

    def softmax(input_tensor, *args, **kwargs):
        if store.active_layer is not None and input_tensor.ndim >= 4:
            store.record(
                f"layer.{store.active_layer:03d}.attention_scores",
                _valid_attention_scores(store, input_tensor),
            )
        output = original_softmax(input_tensor, *args, **kwargs)
        if store.active_layer is not None and input_tensor.ndim >= 4:
            store.record(f"layer.{store.active_layer:03d}.softmax", output.to(input_tensor.dtype))
        return output

    modeling_qwen3.apply_rotary_pos_emb = apply_rotary_pos_emb
    F.softmax = softmax
    try:
        yield
    finally:
        modeling_qwen3.apply_rotary_pos_emb = original_rope
        F.softmax = original_softmax


def _remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def _slice_cache(tensor: torch.Tensor, *, axis: int, valid_length: int) -> torch.Tensor:
    axis = axis if axis >= 0 else tensor.ndim + axis
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"cache axis {axis} is invalid for shape {tuple(tensor.shape)}")
    slices = [slice(None)] * tensor.ndim
    slices[axis] = slice(0, valid_length)
    return tensor[tuple(slices)]


def _record_native_cache(store: CaptureStore, cache: Any, *, valid_length: int) -> None:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        pairs = [(layer.keys, layer.values) for layer in layers]
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        pairs = list(zip(cache.key_cache, cache.value_cache, strict=True))
    else:
        pairs = [(items[0], items[1]) for items in cache]
    for layer_index, (key_cache, value_cache) in enumerate(pairs):
        store.record(f"cache.{layer_index:03d}.key", _slice_cache(key_cache, axis=2, valid_length=valid_length))
        store.record(f"cache.{layer_index:03d}.value", _slice_cache(value_cache, axis=2, valid_length=valid_length))


def _record_wrap_cache(store: CaptureStore, xh_model: Any, *, valid_length: int) -> None:
    cache_axis = int(getattr(xh_model.kvcache_config, "cache_axis", 2))
    for layer_index, (key_cache, value_cache) in enumerate(
        zip(xh_model.past_key_caches, xh_model.past_value_caches, strict=True)
    ):
        store.record(
            f"cache.{layer_index:03d}.key",
            _slice_cache(key_cache.detach(), axis=cache_axis, valid_length=valid_length),
        )
        store.record(
            f"cache.{layer_index:03d}.value",
            _slice_cache(value_cache.detach(), axis=cache_axis, valid_length=valid_length),
        )


def _record_wrap_rotary(store: CaptureStore, text_model: nn.Module, *, start: int, length: int) -> None:
    rotary = getattr(text_model, "rotary_emb", None)
    if not isinstance(rotary, nn.Module):
        return
    for name in ("cos", "sin"):
        cached = getattr(rotary, f"{name}_cached", None)
        if not isinstance(cached, torch.Tensor):
            continue
        values = cached[:, :, start : start + length, :]
        if values.shape[1] == 1:
            values = values.squeeze(1)
        store.record(f"rotary.{name}", values)


def _safe_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _tensor_metrics(reference: CapturedTensor | None, candidate: CapturedTensor | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "reference_shape": list(reference.shape) if reference is not None else None,
        "candidate_shape": list(candidate.shape) if candidate is not None else None,
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
    }
    if reference is None or candidate is None:
        row["status"] = "missing"
        return row
    if reference.shape != candidate.shape:
        row["status"] = "shape_mismatch"
        return row

    reference_tensor = reference.tensor
    candidate_tensor = candidate.tensor
    row["reference_nonfinite"] = int((~torch.isfinite(reference_tensor)).sum().item())
    row["candidate_nonfinite"] = int((~torch.isfinite(candidate_tensor)).sum().item())
    row["exact"] = bool(torch.equal(reference_tensor, candidate_tensor))
    row["status"] = "exact" if row["exact"] else "different"
    if reference_tensor.numel() == 0:
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


def _build_report(
    reference: CaptureStore,
    candidate: CaptureStore,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        match = _LAYER_OUTPUT_PATTERN.match(name)
        if match is not None and not row["exact"]:
            capture_stage, layer_index_text = match.groups()
            stage = "decode" if capture_stage.startswith("decode") else "prefill"
            layer_index = int(layer_index_text)
            if first_bad[stage] is None:
                first_bad[stage] = layer_index
    return {
        "metadata": metadata or {},
        "summary": {
            "exact": all(bool(row["exact"]) for row in rows),
            "compared_tensors": len(rows),
            "different_tensors": sum(not bool(row["exact"]) for row in rows),
            "first_bad_layer": first_bad,
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
        "reference_shape",
        "candidate_shape",
        "reference_dtype",
        "candidate_dtype",
        "max_abs",
        "mean_abs",
        "rmse",
        "max_rel",
        "cosine",
        "reference_nonfinite",
        "candidate_nonfinite",
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


def _print_report(report: dict[str, Any]) -> None:
    metadata = report["metadata"]
    print(
        f"input_ids: shape={metadata.get('input_shape')} sha256={metadata.get('input_ids_sha256')} "
        f"decode_steps={metadata.get('decode_steps', int(metadata.get('teacher_token') is not None))} "
        f"teacher_token={metadata.get('teacher_token')}"
    )
    multi_step_decode = int(metadata.get("decode_steps", 1)) > 1
    for row in report["comparisons"]:
        if multi_step_decode and row["name"].startswith("decode.") and not row["name"].endswith(".logits"):
            continue
        if _LAYER_OUTPUT_PATTERN.match(row["name"]) or row["name"].endswith(("embedding", "final_norm", "logits")):
            argmax_text = (
                f" argmax={row.get('reference_argmax')}/{row.get('candidate_argmax')}"
                if row["name"].endswith(".logits")
                else ""
            )
            print(
                f"{row['name']}: status={row['status']} max_abs={row['max_abs']} "
                f"mean_abs={row['mean_abs']} cosine={row['cosine']}{argmax_text}"
            )
    for stage, layer_index in report["summary"]["first_bad_layer"].items():
        print(f"{stage} first bad layer: {layer_index if layer_index is not None else 'none'}")
        if layer_index is None:
            continue
        prefix = f"{stage}.layer.{layer_index:03d}."
        if stage == "decode" and multi_step_decode:
            first_bad_row = next(
                (
                    row
                    for row in report["comparisons"]
                    if re.match(rf"^decode\.\d+\.layer\.{layer_index:03d}\.", row["name"]) and not row["exact"]
                ),
                None,
            )
            if first_bad_row is not None:
                prefix = first_bad_row["name"].split(".layer.", maxsplit=1)[0] + f".layer.{layer_index:03d}."
        for row in report["comparisons"]:
            if row["name"].startswith(prefix) and not row["name"].endswith((".input", ".output")):
                print(
                    f"  {row['name'][len(prefix) :]}: status={row['status']} "
                    f"max_abs={row['max_abs']} mean_abs={row['mean_abs']} cosine={row['cosine']}"
                )


def _model_structure(config: Any) -> dict[str, Any]:
    return {
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "vocab_size": int(config.vocab_size),
    }


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _selected_device_max_memory(device: torch.device) -> dict[int | str, int]:
    """Reserve execution headroom and exclude every non-selected CUDA device."""
    free_bytes, _ = torch.cuda.mem_get_info(device)
    reserve_bytes = 5 * 1024**3
    usable_bytes = free_bytes - reserve_bytes
    if usable_bytes < 1024**3:
        raise RuntimeError(
            f"{device} has only {free_bytes / 1024**3:.2f} GiB free; "
            "at least 6 GiB is required for auto-offloaded execution"
        )
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    # The wrap model is fully materialized on CPU before Accelerate computes its
    # device map.  Using currently available RAM here double-counts that model
    # allocation and can incorrectly spill otherwise resident layers to disk.
    cpu_bytes = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") * 0.8)
    return {device_index: usable_bytes, "cpu": cpu_bytes}


def _load_dequantized_native_model(
    *,
    model_dir: str,
    config_path: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    work_dir: Path,
    auto_offload: bool,
    debug: bool,
) -> tuple[nn.Module, Any, Any]:
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        model_dir=model_dir,
        config_path=config_path,
        seed=seed,
        debug=debug,
    )
    xh_model = workflow.get_model(work_dir=str(work_dir))
    # Loading with device_map="auto" would place packed weights before their
    # FP16 expansion.  Always dequantize on CPU, then dispatch the dense model.
    xh_model.config.enable_auto_offload = False
    model = xh_model.get_native_model()
    model.config._attn_implementation = "eager"
    model.eval()

    packed_modules = [name for name, module in model.named_modules() if hasattr(module, "qweight")]
    if packed_modules:
        raise RuntimeError(
            "native model still contains packed GPTQ modules after get_native_model(): "
            + ", ".join(packed_modules[:20])
        )

    if auto_offload:
        from xhmodel_merak.xh_llm.utils import hf_auto_offload

        hf_auto_offload(
            model,
            max_memory=_selected_device_max_memory(device),
            dtype=dtype,
            offload_buffers=True,
        )
    else:
        model.to(device=device, dtype=dtype)
    return model, xh_model, workflow


def _run_native(
    *,
    model_dir: str,
    config_path: str,
    messages: list[dict[str, str]],
    input_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    think: bool,
    work_dir: Path,
    decode_steps: int,
    capture_details: bool,
    auto_offload: bool,
    debug: bool,
) -> tuple[CaptureStore, list[int], dict[str, Any]]:
    from transformers import DynamicCache

    from xhquant.api import set_random_seed

    # Match qwen3_native_generate_workflow.py: Merak's native loader first
    # materializes GPTQ TorchQuantLinear modules on CPU, then converts them to
    # dense FP16 Linear modules through _dequantize_gptqmodel_hf_model().
    _set_determinism(seed)
    set_random_seed(seed)
    model, xh_model, workflow = _load_dequantized_native_model(
        model_dir=model_dir,
        config_path=config_path,
        device=device,
        dtype=dtype,
        seed=seed,
        work_dir=work_dir,
        auto_offload=auto_offload,
        debug=debug,
    )

    native_model_inputs = xh_model.get_tf_processor().apply_chat_template(messages, enable_think=think)
    _assert_same_input_ids(input_ids, native_model_inputs.input_ids)

    text_model = _find_text_model(model)
    store = CaptureStore()
    handles = _register_capture_hooks(
        text_model,
        model.get_input_embeddings(),
        store,
        capture_details=capture_details,
    )
    native_ids = input_ids.to(device)
    attention_mask = torch.ones_like(native_ids)
    cache = DynamicCache(config=model.config)

    try:
        with torch.no_grad(), _native_attention_capture(store):
            store.set_stage("prefill", valid_sequence_length=native_ids.shape[1])
            prefill_output = model(
                input_ids=native_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            store.record("logits", prefill_output.logits, last_token=True)
            next_token = int(prefill_output.logits[:, -1, :].argmax(dim=-1).item())
            cache = prefill_output.past_key_values
            _record_native_cache(store, cache, valid_length=native_ids.shape[1])

            teacher_tokens: list[int] = []
            for step in range(decode_steps):
                teacher_tokens.append(next_token)
                valid_length = native_ids.shape[1] + step + 1
                store.set_stage(_decode_stage(step, decode_steps), valid_sequence_length=valid_length)
                decode_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
                decode_mask = torch.ones((1, valid_length), dtype=torch.long, device=device)
                decode_output = model(
                    input_ids=decode_ids,
                    attention_mask=decode_mask,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                store.record("logits", decode_output.logits, last_token=True)
                next_token = int(decode_output.logits[:, -1, :].argmax(dim=-1).item())
                cache = decode_output.past_key_values
                _record_native_cache(store, cache, valid_length=valid_length)
        metadata = {
            "attention_implementation": str(model.config._attn_implementation),
            "weight_implementation": "dequantized_fp16_linear",
            "cache_implementation": "dynamic",
            "structure": _model_structure(text_model.config),
            "chip_arch": str(workflow.workflow_config.export["model"]["chip_arch"]),
        }
    finally:
        _remove_hooks(handles)
        del model, xh_model, workflow, text_model, cache
        _release_cuda_memory()
    return store, teacher_tokens, metadata


def _run_wrap(
    *,
    model_dir: str,
    config_path: str,
    messages: list[dict[str, str]],
    input_ids: torch.Tensor,
    teacher_tokens: list[int],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    think: bool,
    work_dir: Path,
    decode_steps: int,
    capture_details: bool,
    auto_offload: bool,
    debug: bool,
) -> tuple[CaptureStore, dict[str, Any]]:
    from xhmodel_merak.xh_llm import LLMInferenceContextManager, LLMModelState
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow
    from xhquant.api import set_random_seed

    _enable_kvcache_fast_mode()
    _set_determinism(seed)
    set_random_seed(seed)
    workflow = AutoLLMWorkflow.from_config(
        model_dir=model_dir,
        config_path=config_path,
        seed=seed,
        debug=debug,
    )
    xh_model = workflow.get_model(work_dir=str(work_dir))
    if not bool(xh_model.config.use_cache):
        raise RuntimeError("layerwise prefill/decode comparison requires export.model.use_cache=true")
    xh_model.config.enable_prefill_chunk = False
    # Build and dequantize on CPU first.  Enabling Merak auto-offload before
    # wrapping makes GPTQModel's ``device_map="auto"`` distribute the load over
    # every visible GPU; it also sizes the packed checkpoint before its FP16
    # expansion.  Delaying the flag keeps model construction deterministic and
    # lets the inference context dispatch the finished wrap model afterwards.
    xh_model.config.enable_auto_offload = False
    xh_model.set_state(LLMModelState.WRAP)
    xh_model.config.enable_auto_offload = auto_offload
    xh_model.to(device=device, dtype=dtype)
    xh_model.eval()

    processor = xh_model.get_tf_processor()
    wrap_model_inputs = processor.apply_chat_template(messages, enable_think=think)
    _assert_same_input_ids(input_ids, wrap_model_inputs.input_ids)

    wrap_root = xh_model.get_inference_model()
    if not isinstance(wrap_root, nn.Module):
        raise RuntimeError(f"wrap inference model must be nn.Module, got {type(wrap_root).__name__}")
    if auto_offload:
        from xhmodel_merak.xh_llm.utils import hf_auto_offload

        hf_auto_offload(
            wrap_root,
            max_memory=_selected_device_max_memory(device),
            dtype=dtype,
            offload_buffers=True,
        )
    text_model = _find_text_model(wrap_root)
    embedding = xh_model.get_input_embeddings()
    if not isinstance(embedding, nn.Module):
        raise RuntimeError("wrap token embedding is unavailable")

    store = CaptureStore()
    handles = _register_capture_hooks(
        text_model,
        embedding,
        store,
        capture_details=capture_details,
    )
    try:
        with LLMInferenceContextManager(xh_model, devices=[device]), torch.no_grad():
            store.set_stage("prefill", valid_sequence_length=input_ids.shape[1])
            _record_wrap_rotary(store, text_model, start=0, length=input_ids.shape[1])
            xh_model.set_prefill()
            xh_model.set_input_sequence_length(input_ids.shape[1])
            data_processor = xh_model.get_data_preprocessor()
            prefill_inputs = data_processor({"input_ids": input_ids, "past_seq_length": 0})
            prefill_logits = xh_model.forward(*prefill_inputs)
            store.record("logits", prefill_logits, last_token=True)
            _record_wrap_cache(store, xh_model, valid_length=input_ids.shape[1])

            for step, teacher_token in enumerate(teacher_tokens):
                past_length = input_ids.shape[1] + step
                valid_length = past_length + 1
                store.set_stage(_decode_stage(step, decode_steps), valid_sequence_length=valid_length)
                _record_wrap_rotary(store, text_model, start=past_length, length=1)
                xh_model.set_decode()
                decode_ids = torch.tensor([[teacher_token]], dtype=torch.long)
                decode_inputs = data_processor({"input_ids": decode_ids, "past_seq_length": past_length})
                decode_logits = xh_model.forward(*decode_inputs)
                store.record("logits", decode_logits, last_token=True)
                _record_wrap_cache(store, xh_model, valid_length=valid_length)
        metadata = {
            "attention_implementation": "merak_graph_forward",
            "structure": _model_structure(text_model.config),
            "chip_arch": str(workflow.workflow_config.export["model"]["chip_arch"]),
        }
    finally:
        _remove_hooks(handles)
        del xh_model, wrap_root, text_model
        _release_cuda_memory()
    return store, metadata


def _build_input_ids(model_dir: str, prompt: str, think: bool) -> tuple[Any, list[dict[str, str]], torch.Tensor]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=think,
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    return tokenizer, messages, model_inputs["input_ids"].to(dtype=torch.long, device="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Native/HF checkpoint directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH, help="Matching Merak workflow YAML.")
    parser.add_argument("--prompt", default="1+1等于几", help="Prompt text or UTF-8 text file.")
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR, help="Directory for logs and reports.")
    parser.add_argument("--device", default="cuda:0", help="Execution device. Default: cuda:0.")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode on both sides.")
    parser.add_argument("--skip-decode", action="store_true", help="Compare prefill only.")
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=1,
        help="Number of greedy teacher-forced decode tokens to compare. Default: 1.",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Capture only layer boundaries, norm, logits and cache.",
    )
    parser.add_argument(
        "--auto-offload",
        action="store_true",
        help="Offload native and wrap weights between CPU and the selected GPU.",
    )
    parser.add_argument("--fail-on-diff", action="store_true", help="Exit non-zero when any compared tensor differs.")
    parser.add_argument("--debug", action="store_true", help="Enable xhquant debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = str(Path(args.model_dir))
    config_path = str(Path(args.config_path))
    if not Path(model_dir).is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"workflow config does not exist: {config_path}")
    if args.decode_steps < 0:
        raise ValueError("--decode-steps must be non-negative")
    decode_steps = 0 if args.skip_decode else args.decode_steps

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    dtype = _torch_dtype(args.dtype)
    _set_determinism(args.seed)
    from xhquant.api import xhquant_init

    xhquant_init(str(work_dir / "layerwise_diff.log"), args.debug)

    prompt = _read_prompt(args.prompt)
    tokenizer, messages, input_ids = _build_input_ids(model_dir, prompt, args.think)
    input_metadata = {
        "model_dir": str(Path(model_dir).resolve()),
        "config_path": str(Path(config_path).resolve()),
        "prompt": prompt,
        "think": bool(args.think),
        "seed": int(args.seed),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": str(dtype).removeprefix("torch."),
        "input_shape": list(input_ids.shape),
        "input_ids_sha256": _input_ids_sha256(input_ids),
        "tokens": input_ids.tolist(),
        "transformers_version": __import__("transformers").__version__,
        "decode_steps": decode_steps,
    }

    native_store, teacher_tokens, native_metadata = _run_native(
        model_dir=model_dir,
        config_path=config_path,
        messages=messages,
        input_ids=input_ids,
        device=device,
        dtype=dtype,
        seed=args.seed,
        think=args.think,
        work_dir=work_dir,
        decode_steps=decode_steps,
        capture_details=not args.no_details,
        auto_offload=args.auto_offload,
        debug=args.debug,
    )
    wrap_store, wrap_metadata = _run_wrap(
        model_dir=model_dir,
        config_path=config_path,
        messages=messages,
        input_ids=input_ids,
        teacher_tokens=teacher_tokens,
        device=device,
        dtype=dtype,
        seed=args.seed,
        think=args.think,
        work_dir=work_dir,
        decode_steps=decode_steps,
        capture_details=not args.no_details,
        auto_offload=args.auto_offload,
        debug=args.debug,
    )
    del tokenizer

    if native_metadata["structure"] != wrap_metadata["structure"]:
        raise RuntimeError(
            "native/wrap model structure mismatch: "
            f"native={native_metadata['structure']}, wrap={wrap_metadata['structure']}"
        )
    metadata = {
        **input_metadata,
        "teacher_token": teacher_tokens[0] if teacher_tokens else None,
        "teacher_tokens": teacher_tokens,
        "native": native_metadata,
        "wrap": wrap_metadata,
    }
    report = _build_report(native_store, wrap_store, metadata=metadata)
    json_path, csv_path = _write_report(report, work_dir)
    _print_report(report)
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    return 1 if args.fail_on_diff and not report["summary"]["exact"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
