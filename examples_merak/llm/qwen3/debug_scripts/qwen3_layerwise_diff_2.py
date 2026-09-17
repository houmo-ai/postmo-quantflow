"""Deterministic layerwise comparison between native and Merak-wrapped Qwen3.

Native greedy generation runs until EOS or the 2K total-token limit. The wrapped
model consumes the same generated context so layerwise differences remain
attributable to the implementation, while its own predicted tokens identify the
first output-token mismatch. Two distinct CUDA devices can run both paths as a
token-streaming pipeline; single-device execution remains sequential.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import queue
import random
import re
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from tqdm import tqdm


DEFAULT_MODEL_DIR = None
DEFAULT_CONFIG_PATH = "configs_merak/xh2a/llm_models/qwen3/8b/qwen3_8b_xh2a_w4a8_hf_gptq_2k.py"
DEFAULT_WORK_DIR = "work_dirs/qwen3_8b_layerwise_diff"
DEFAULT_MAX_TOTAL_TOKENS = 2048
_LAYER_OUTPUT_PATTERN = re.compile(r"^(prefill|decode(?:\.\d+)?)\.layer\.(\d+)\.output$")
_DECODE_STAGE_PATTERN = re.compile(r"^decode(?:\.\d+)?$")
_ROTARY_PATTERN = re.compile(r"^(?P<stage>prefill|decode(?:\.\d+)?)\.rotary\.(?P<component>cos|sin)$")
_POSITION_EMBEDDINGS_PATTERN = re.compile(
    r"^(?P<stage>prefill|decode(?:\.\d+)?)\.layer\.(?P<layer>\d+)\."
    r"position_embeddings\.(?P<component>cos|sin)$"
)


def _extract_query_from_prompt_object(prompt_data: Any, *, source: str) -> str:
    if not isinstance(prompt_data, dict):
        raise TypeError(f"{source} must contain an object, got {type(prompt_data).__name__}")
    query = prompt_data.get("query")
    if not isinstance(query, str):
        raise TypeError(f"{source} field 'query' must be a string")
    return query


def _read_prompt(value: str) -> str:
    prompt_path = Path(value)
    if not prompt_path.is_file():
        return value
    prompt_text = prompt_path.read_text(encoding="utf-8")
    if prompt_path.suffix.lower() == ".jsonl":
        lines = [line.strip() for line in prompt_text.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"JSONL prompt file is empty: {prompt_path}")
        return _extract_query_from_prompt_object(json.loads(lines[0]), source="JSONL prompt")
    if prompt_path.suffix.lower() == ".json":
        return _extract_query_from_prompt_object(json.loads(prompt_text), source="JSON prompt")
    return prompt_text


@dataclass
class CapturedTensor:
    tensor: torch.Tensor
    dtype: str
    shape: tuple[int, ...]


@dataclass
class _TokenStreamError:
    error: BaseException


_TOKEN_STREAM_END = object()


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


def _load_merak_config(config_path: str, model_dir: str | None, seed: int) -> tuple[Any, Any]:
    """Load the legacy Python config and resolve its registered LLM config."""
    from xhmodel_merak.xh_llm import AutoLLMConfig
    from xhquant.api import Config

    cfg = Config.fromfile(config_path)
    if model_dir:
        if hasattr(cfg, "hf_model_dir"):
            cfg.hf_model_dir = model_dir
        cfg.model.hf_model = model_dir
    cfg.seed = seed
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    return cfg, model_cfg


def _load_xh_model(*, config_path: str, model_dir: str, seed: int, work_dir: Path) -> Any:
    """Construct an XH LLM directly, without a workflow intermediary."""
    from xhmodel_merak.xh_llm import AutoLLMModel
    from xhquant.api import get_xhquant_logger

    cfg, model_cfg = _load_merak_config(config_path, model_dir, seed)
    logger = get_xhquant_logger()
    logger.info(f"Config:\n{cfg.pretty_text}")
    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.work_dir = str(work_dir)
    logger.info(f"Resolved model type: {type(xh_model).__name__}")
    return xh_model


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


def _decode_stage(step: int, max_decode_steps: int) -> str:
    """Return a sortable stage key for an autoregressive decode step."""
    if step < 0 or step >= max_decode_steps:
        raise ValueError(f"decode step {step} is outside [0, {max_decode_steps})")
    return "decode" if max_decode_steps == 1 else f"decode.{step:04d}"


def _normalize_token_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, (tuple, list, set)):
        return {int(item) for item in value}
    raise TypeError(f"token IDs must be an int or a sequence of ints, got {type(value).__name__}")


def _generation_budget(*, input_length: int, max_total_tokens: int) -> int:
    if max_total_tokens <= 0:
        raise ValueError("max_total_tokens must be positive")
    if input_length > max_total_tokens:
        raise ValueError(f"input length {input_length} exceeds the total comparison limit {max_total_tokens}")
    return max_total_tokens - input_length


def _token_progress(*, total: int, description: str, enabled: bool, position: int = 0):
    return tqdm(
        total=total,
        desc=description,
        unit="token",
        dynamic_ncols=True,
        disable=not enabled,
        position=position,
    )


def _all_but_last(tokens: Iterable[int]) -> Iterator[int]:
    iterator = iter(tokens)
    try:
        pending = next(iterator)
    except StopIteration:
        return
    for token in iterator:
        yield pending
        pending = token


def _iter_token_queue(token_queue: queue.Queue[Any]) -> Iterator[int]:
    while True:
        item = token_queue.get()
        if item is _TOKEN_STREAM_END:
            return
        if isinstance(item, _TokenStreamError):
            raise RuntimeError("native token stream failed") from item.error
        yield int(item)


def _first_output_token_mismatch(
    native_tokens: list[int],
    wrap_tokens: list[int],
    *,
    input_length: int,
    max_decode_steps: int,
) -> dict[str, Any] | None:
    compared_length = max(len(native_tokens), len(wrap_tokens))
    for generation_index in range(compared_length):
        native_token = native_tokens[generation_index] if generation_index < len(native_tokens) else None
        wrap_token = wrap_tokens[generation_index] if generation_index < len(wrap_tokens) else None
        if native_token == wrap_token:
            continue
        prediction_stage = "prefill" if generation_index == 0 else _decode_stage(generation_index - 1, max_decode_steps)
        return {
            "output_token_number": generation_index + 1,
            "generation_index": generation_index,
            "position_id": input_length + generation_index,
            "prediction_stage": prediction_stage,
            "native_token_id": native_token,
            "wrap_token_id": wrap_token,
        }
    return None


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


def _decoder_layer_input_hook(store: CaptureStore, layer_prefix: str):
    def hook(_module, args, kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        store.record(f"{layer_prefix}.input", hidden_states)
        store.record_pair(f"{layer_prefix}.position_embeddings", kwargs.get("position_embeddings"))

    return hook


def _register_capture_hooks(
    text_model: nn.Module,
    embedding: nn.Module,
    store: CaptureStore,
    *,
    capture_details: bool,
    handles: list[Any] | None = None,
) -> list[Any]:
    if handles is None:
        handles = []
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
                _decoder_layer_input_hook(store, layer_prefix),
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


class _NativeSoftmaxCaptureMode(TorchFunctionMode):
    """Thread-local capture of the eager attention softmax call."""

    def __init__(self, store: CaptureStore) -> None:
        super().__init__()
        self.store = store

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        input_tensor = args[0] if args else kwargs.get("input")
        if (
            func is F.softmax
            and self.store.active_layer is not None
            and isinstance(input_tensor, torch.Tensor)
            and input_tensor.ndim >= 4
        ):
            layer_prefix = f"layer.{self.store.active_layer:03d}"
            self.store.record(
                f"{layer_prefix}.attention_scores",
                _valid_attention_scores(self.store, input_tensor),
            )
            self.store.record(f"{layer_prefix}.softmax", output.to(input_tensor.dtype))
        return output


def _register_native_rope_hooks(text_model: nn.Module, store: CaptureStore) -> list[Any]:
    """Recompute the pure RoPE transform from model-local hook inputs."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    handles: list[Any] = []
    states: dict[int, dict[str, Any]] = {}
    for layer_index, layer in enumerate(text_model.layers):
        attention = getattr(layer, "self_attn", None)
        q_norm = getattr(attention, "q_norm", None)
        k_norm = getattr(attention, "k_norm", None)
        if not all(isinstance(module, nn.Module) for module in (attention, q_norm, k_norm)):
            continue

        def attention_pre_hook(_module, _args, kwargs, *, index=layer_index):
            states[index] = {"position_embeddings": kwargs.get("position_embeddings")}

        def q_norm_hook(_module, _args, _kwargs, output, *, index=layer_index):
            states.setdefault(index, {})["query_states"] = output

        def k_norm_hook(_module, _args, _kwargs, output, *, index=layer_index):
            state = states.get(index, {})
            query_states = state.get("query_states")
            position_embeddings = state.get("position_embeddings")
            if not isinstance(query_states, torch.Tensor) or not isinstance(output, torch.Tensor):
                return
            if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) < 2:
                return
            cos, sin = position_embeddings[:2]
            rope_q, rope_k = apply_rotary_pos_emb(
                query_states.transpose(1, 2),
                output.transpose(1, 2),
                cos,
                sin,
            )
            store.record(f"layer.{index:03d}.rope_q", rope_q)
            store.record(f"layer.{index:03d}.rope_k", rope_k)
            states.pop(index, None)

        handles.append(attention.register_forward_pre_hook(attention_pre_hook, with_kwargs=True))
        handles.append(q_norm.register_forward_hook(q_norm_hook, with_kwargs=True))
        handles.append(k_norm.register_forward_hook(k_norm_hook, with_kwargs=True))
    return handles


@contextmanager
def _native_attention_capture(store: CaptureStore, text_model: nn.Module | None = None) -> Iterator[None]:
    """Capture native attention details without mutating process-global callables."""
    handles = _register_native_rope_hooks(text_model, store) if text_model is not None else []
    try:
        with _NativeSoftmaxCaptureMode(store):
            yield
    finally:
        _remove_hooks(handles)


def _native_detail_capture_context(
    store: CaptureStore,
    text_model: nn.Module,
    *,
    capture_details: bool,
):
    return _native_attention_capture(store, text_model) if capture_details else nullcontext()


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


def _record_native_cache(store: CaptureStore, cache: Any, *, valid_length: int, start: int = 0) -> None:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        pairs = [(layer.keys, layer.values) for layer in layers]
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        pairs = list(zip(cache.key_cache, cache.value_cache, strict=True))
    else:
        pairs = [(items[0], items[1]) for items in cache]
    for layer_index, (key_cache, value_cache) in enumerate(pairs):
        store.record(
            f"cache.{layer_index:03d}.key",
            _slice_cache(key_cache, axis=2, start=start, valid_length=valid_length),
        )
        store.record(
            f"cache.{layer_index:03d}.value",
            _slice_cache(value_cache, axis=2, start=start, valid_length=valid_length),
        )


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


def _align_position_embedding_tensors(
    reference: CapturedTensor,
    candidate: CapturedTensor,
) -> tuple[torch.Tensor, torch.Tensor, bool] | None:
    reference_tensor = reference.tensor
    candidate_tensor = candidate.tensor
    if reference.shape == candidate.shape:
        return reference_tensor, candidate_tensor, False
    if (
        reference_tensor.ndim + 1 == candidate_tensor.ndim
        and reference_tensor.shape[0] == candidate_tensor.shape[0]
        and candidate_tensor.shape[1] == 1
        and reference_tensor.shape[1:] == candidate_tensor.shape[2:]
    ):
        return reference_tensor.unsqueeze(1), candidate_tensor, True
    if (
        candidate_tensor.ndim + 1 == reference_tensor.ndim
        and candidate_tensor.shape[0] == reference_tensor.shape[0]
        and reference_tensor.shape[1] == 1
        and candidate_tensor.shape[1:] == reference_tensor.shape[2:]
    ):
        return reference_tensor, candidate_tensor.unsqueeze(1), True
    return None


def _tensor_metrics(
    reference: CapturedTensor | None,
    candidate: CapturedTensor | None,
    *,
    allow_broadcast: bool = False,
) -> dict[str, Any]:
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
        "different_elements": None,
        "total_elements": None,
        "different_ratio": None,
        "broadcast_compared": False,
        "comparison_shape": None,
    }
    if reference is None or candidate is None:
        row["status"] = "missing"
        return row
    if allow_broadcast:
        aligned = _align_position_embedding_tensors(reference, candidate)
        if aligned is None:
            row["status"] = "shape_mismatch"
            return row
        reference_tensor, candidate_tensor, row["broadcast_compared"] = aligned
    else:
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


def _summarize_rotary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return _summarize_components(rows, _ROTARY_PATTERN)


def _summarize_position_embeddings(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize cos/sin values at the input of every decoder layer."""
    return _summarize_components(rows, _POSITION_EMBEDDINGS_PATTERN)


def _position_ids_for_stage(stage: str, sequence_length: int, metadata: dict[str, Any]) -> list[int]:
    if stage == "prefill":
        start = 0
    else:
        input_shape = metadata.get("input_shape")
        if not input_shape:
            return []
        step = 0 if stage == "decode" else int(stage.rsplit(".", maxsplit=1)[1])
        start = int(input_shape[-1]) + step
    return list(range(start, start + sequence_length))


def _mismatched_position_ids(
    stage: str,
    reference: CapturedTensor | None,
    candidate: CapturedTensor | None,
    metadata: dict[str, Any],
) -> list[int]:
    available = reference if reference is not None else candidate
    if available is None or available.tensor.ndim < 2:
        return []
    position_ids = _position_ids_for_stage(stage, int(available.tensor.shape[-2]), metadata)
    if not position_ids:
        return []
    if reference is None or candidate is None:
        return position_ids
    aligned = _align_position_embedding_tensors(reference, candidate)
    if aligned is None:
        return position_ids
    reference_tensor, candidate_tensor, _ = aligned
    different = reference_tensor != candidate_tensor
    sequence_axis = different.ndim - 2
    reduction_dims = tuple(axis for axis in range(different.ndim) if axis != sequence_axis)
    different_by_position = different.any(dim=reduction_dims)
    different_indices = torch.nonzero(different_by_position, as_tuple=False).flatten().tolist()
    return [position_ids[index] for index in different_indices]


def _summarize_position_embeddings_by_stage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stages = {
        match.group("stage")
        for row in rows
        if (match := _POSITION_EMBEDDINGS_PATTERN.fullmatch(row["name"])) is not None
    }

    def stage_sort_key(stage: str) -> tuple[int, int]:
        if stage == "prefill":
            return (0, 0)
        return (1, 0 if stage == "decode" else int(stage.rsplit(".", maxsplit=1)[1]))

    summary: dict[str, dict[str, Any]] = {}
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
        summary[stage] = components
    return summary


def _build_report(
    reference: CaptureStore,
    candidate: CaptureStore,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    first_bad: dict[str, int | None] = {"prefill": None, "decode": None}
    for name in sorted(set(reference.tensors) | set(candidate.tensors)):
        position_match = _POSITION_EMBEDDINGS_PATTERN.fullmatch(name)
        reference_capture = reference.tensors.get(name)
        candidate_capture = candidate.tensors.get(name)
        row = {
            "name": name,
            **_tensor_metrics(
                reference_capture,
                candidate_capture,
                allow_broadcast=position_match is not None,
            ),
        }
        if position_match is not None:
            row["mismatched_position_ids"] = _mismatched_position_ids(
                position_match.group("stage"),
                reference_capture,
                candidate_capture,
                metadata or {},
            )
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
    metadata = metadata or {}
    native_tokens = [int(token) for token in metadata.get("native_generated_token_ids", [])]
    wrap_tokens = [int(token) for token in metadata.get("wrap_generated_token_ids", [])]
    max_decode_steps = max(int(metadata.get("max_decode_steps", 0)), 1)
    first_token_mismatch = _first_output_token_mismatch(
        native_tokens,
        wrap_tokens,
        input_length=int(metadata.get("input_shape", [0, 0])[-1]),
        max_decode_steps=max_decode_steps,
    )
    if first_token_mismatch is not None:
        generation_index = int(first_token_mismatch["generation_index"])
        native_token_texts = metadata.get("native_generated_token_texts", [])
        wrap_token_texts = metadata.get("wrap_generated_token_texts", [])
        first_token_mismatch.update(
            native_token_text=(
                native_token_texts[generation_index] if generation_index < len(native_token_texts) else None
            ),
            wrap_token_text=(wrap_token_texts[generation_index] if generation_index < len(wrap_token_texts) else None),
        )
    return {
        "metadata": metadata,
        "summary": {
            "exact": all(bool(row["exact"]) for row in rows) and first_token_mismatch is None,
            "compared_tensors": len(rows),
            "different_tensors": sum(not bool(row["exact"]) for row in rows),
            "first_bad_layer": first_bad,
            "rotary": _summarize_rotary(rows),
            "position_embeddings": _summarize_position_embeddings(rows),
            "position_embeddings_by_stage": _summarize_position_embeddings_by_stage(rows),
            "output_tokens_exact": first_token_mismatch is None,
            "first_output_token_mismatch": first_token_mismatch,
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
        "mismatched_position_ids",
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
        f"generated_tokens={metadata.get('generated_tokens', 0)} "
        f"decode_steps={metadata.get('decode_steps', 0)} "
        f"stop_reason={metadata.get('generation_stop_reason')} "
        f"max_total_tokens={metadata.get('max_total_tokens')}"
    )
    token_mismatch = report["summary"]["first_output_token_mismatch"]
    if token_mismatch is None:
        print("first output token mismatch: none")
    else:
        print(
            "first output token mismatch: "
            f"output_token_number={token_mismatch['output_token_number']} "
            f"generation_index={token_mismatch['generation_index']} "
            f"position_id={token_mismatch['position_id']} "
            f"stage={token_mismatch['prediction_stage']} "
            f"native={token_mismatch['native_token_id']}:{token_mismatch['native_token_text']!r} "
            f"wrap={token_mismatch['wrap_token_id']}:{token_mismatch['wrap_token_text']!r}"
        )
    print(f"native generated text: {metadata.get('native_generated_text', '')!r}")
    print(f"wrap generated text: {metadata.get('wrap_generated_text', '')!r}")
    for component, rotary in report["summary"]["rotary"].items():
        status = "exact" if rotary["exact"] else "different"
        print(
            f"rotary.{component}: status={status} tensors={rotary['compared_tensors']} "
            f"different_tensors={rotary['different_tensors']} "
            f"different_elements={rotary['different_elements']}/{rotary['total_elements']} "
            f"different_ratio={rotary['different_ratio']:.8f} max_abs={rotary['max_abs']}"
        )
    for stage, components in report["summary"]["position_embeddings_by_stage"].items():
        for component, position_embedding in components.items():
            status = "exact" if position_embedding["exact"] else "different"
            mismatch_text = (
                f" position_ids={position_embedding['mismatched_position_ids']}"
                if not position_embedding["exact"]
                else ""
            )
            print(
                f"position_embeddings.{stage}.{component}: status={status} "
                f"tensors={position_embedding['compared_tensors']} "
                f"different_tensors={position_embedding['different_tensors']} "
                f"different_elements={position_embedding['different_elements']}/"
                f"{position_embedding['total_elements']} "
                f"different_ratio={position_embedding['different_ratio']:.8f} "
                f"max_abs={position_embedding['max_abs']}{mismatch_text}"
            )
    multi_step_decode = any(re.match(r"^decode\.\d+\.", row["name"]) for row in report["comparisons"])
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


def _report_text(report: dict[str, Any]) -> str:
    output = io.StringIO()
    with redirect_stdout(output):
        _print_report(report)
    return output.getvalue()


def _model_structure(config: Any) -> dict[str, Any]:
    return {
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "vocab_size": int(config.vocab_size),
    }


def _select_cuda_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)


def _release_cuda_memory(device: torch.device | None = None) -> None:
    gc.collect()
    if torch.cuda.is_available():
        if device is not None and device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        else:
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
) -> tuple[nn.Module, Any]:
    del debug  # Logging is configured by main(); retained for call-site compatibility.
    xh_model = _load_xh_model(
        config_path=config_path,
        model_dir=model_dir,
        seed=seed,
        work_dir=work_dir,
    )
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

    # Normalize parameters and floating-point buffers before dispatching.  The
    # dtype passed to hf_auto_offload() is used to build the device map, but
    # accelerate.dispatch_model() does not cast the model.  Without this step,
    # auto-offload leaves native RoPE inv_freq in FP32 while the non-offload and
    # wrap paths cast it to the requested dtype.
    model.to(dtype=dtype)
    if auto_offload:
        from xhmodel_merak.xh_llm.utils import hf_auto_offload

        hf_auto_offload(
            model,
            max_memory=_selected_device_max_memory(device),
            dtype=dtype,
            offload_buffers=True,
        )
    else:
        model.to(device=device)
    return model, xh_model


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
    max_total_tokens: int,
    capture_details: bool,
    show_progress: bool,
    progress_position: int,
    token_callback: Callable[[int], None] | None,
    auto_offload: bool,
    debug: bool,
) -> tuple[CaptureStore, list[int], dict[str, Any]]:
    from transformers import DynamicCache

    from xhquant.api import get_xhquant_logger, set_random_seed

    # Match qwen3_native_generate_workflow.py: Merak's native loader first
    # materializes GPTQ TorchQuantLinear modules on CPU, then converts them to
    # dense FP16 Linear modules through _dequantize_gptqmodel_hf_model().
    _select_cuda_device(device)
    _set_determinism(seed)
    set_random_seed(seed)
    logger = get_xhquant_logger()
    run_started = time.perf_counter()
    logger.info(f"[native] loading dequantized model: device={device}, dtype={dtype}, auto_offload={auto_offload}")
    load_started = time.perf_counter()
    model, xh_model = _load_dequantized_native_model(
        model_dir=model_dir,
        config_path=config_path,
        device=device,
        dtype=dtype,
        seed=seed,
        work_dir=work_dir,
        auto_offload=auto_offload,
        debug=debug,
    )
    logger.info(f"[native] model ready in {time.perf_counter() - load_started:.2f}s")

    native_model_inputs = xh_model.get_tf_processor().apply_chat_template(messages, enable_think=think)
    _assert_same_input_ids(input_ids, native_model_inputs.input_ids)

    text_model = _find_text_model(model)
    store = CaptureStore()
    handles: list[Any] = []
    cache: Any = None
    try:
        _register_capture_hooks(
            text_model,
            model.get_input_embeddings(),
            store,
            capture_details=capture_details,
            handles=handles,
        )
        native_ids = input_ids.to(device)
        attention_mask = torch.ones_like(native_ids)
        cache = DynamicCache(config=model.config)
        generation_budget = _generation_budget(
            input_length=int(native_ids.shape[1]),
            max_total_tokens=max_total_tokens,
        )
        max_decode_steps = max(generation_budget - 1, 0)
        eos_token_ids = _normalize_token_ids(getattr(model.generation_config, "eos_token_id", None))
        if not eos_token_ids:
            eos_token_ids = _normalize_token_ids(getattr(model.config, "eos_token_id", None))
        generated_tokens: list[int] = []
        generation_stop_reason = "max_total_tokens"
        detail_context = _native_detail_capture_context(
            store,
            text_model,
            capture_details=capture_details,
        )
        logger.info(
            f"[native] prefill start: input_tokens={native_ids.shape[1]}, "
            f"generation_budget={generation_budget}, capture_details={capture_details}"
        )

        with (
            torch.no_grad(),
            detail_context,
            _token_progress(
                total=generation_budget,
                description=f"Native {device}",
                enabled=show_progress,
                position=progress_position,
            ) as progress,
        ):
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

            if generation_budget > 0:
                generated_tokens.append(next_token)
                if token_callback is not None:
                    token_callback(next_token)
                progress.update(1)
                progress.set_postfix(token_id=next_token)
            step = 0
            while len(generated_tokens) < generation_budget and next_token not in eos_token_ids:
                valid_length = native_ids.shape[1] + step + 1
                store.set_stage(_decode_stage(step, max_decode_steps), valid_sequence_length=valid_length)
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
                generated_tokens.append(next_token)
                if token_callback is not None:
                    token_callback(next_token)
                progress.update(1)
                progress.set_postfix(token_id=next_token)
                cache = decode_output.past_key_values
                # Retain only the newly appended cache token during decode. Full
                # snapshots at every step would make a 2K comparison use O(T^2)
                # host memory; attention scores still validate all cached keys.
                _record_native_cache(store, cache, start=valid_length - 1, valid_length=valid_length)
                step += 1
            if generated_tokens and generated_tokens[-1] in eos_token_ids:
                generation_stop_reason = "eos"
        logger.info(
            f"[native] generation complete: generated_tokens={len(generated_tokens)}, "
            f"decode_steps={step}, stop_reason={generation_stop_reason}, "
            f"elapsed={time.perf_counter() - run_started:.2f}s"
        )
        metadata = {
            "attention_implementation": str(model.config._attn_implementation),
            "weight_implementation": "dequantized_fp16_linear",
            "cache_implementation": "dynamic",
            "structure": _model_structure(text_model.config),
            "chip_arch": str(xh_model.config.chip_arch),
            "eos_token_ids": sorted(eos_token_ids),
            "generation_stop_reason": generation_stop_reason,
            "generated_tokens": len(generated_tokens),
            "decode_steps": step,
            "max_decode_steps": max_decode_steps,
        }
    finally:
        _remove_hooks(handles)
        del model, xh_model, text_model, cache
        _release_cuda_memory(device)
    return store, generated_tokens, metadata


def _run_wrap(
    *,
    model_dir: str,
    config_path: str,
    messages: list[dict[str, str]],
    input_ids: torch.Tensor,
    native_generated_tokens: list[int] | None,
    native_token_stream: Iterable[int] | None,
    expected_generated_tokens: int | None,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    think: bool,
    work_dir: Path,
    max_total_tokens: int,
    capture_details: bool,
    show_progress: bool,
    progress_position: int,
    auto_offload: bool,
    debug: bool,
) -> tuple[CaptureStore, list[int], dict[str, Any]]:
    from xhmodel_merak.xh_llm import LLMInferenceContextManager, LLMModelState
    from xhquant.api import get_xhquant_logger, set_random_seed

    del debug  # Logging is configured by main(); retained for call-site compatibility.
    if (native_generated_tokens is None) == (native_token_stream is None):
        raise ValueError("provide exactly one of native_generated_tokens or native_token_stream")
    _select_cuda_device(device)
    _enable_kvcache_fast_mode()
    _set_determinism(seed)
    set_random_seed(seed)
    logger = get_xhquant_logger()
    run_started = time.perf_counter()
    logger.info(f"[wrap] loading wrapped model: device={device}, dtype={dtype}, auto_offload={auto_offload}")
    load_started = time.perf_counter()
    xh_model = _load_xh_model(
        config_path=config_path,
        model_dir=model_dir,
        seed=seed,
        work_dir=work_dir,
    )
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
    logger.info(f"[wrap] model ready in {time.perf_counter() - load_started:.2f}s")

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
    generation_budget = _generation_budget(
        input_length=int(input_ids.shape[1]),
        max_total_tokens=max_total_tokens,
    )
    max_decode_steps = max(generation_budget - 1, 0)
    wrap_generated_tokens: list[int] = []
    token_source: Iterable[int] = (
        native_generated_tokens if native_generated_tokens is not None else native_token_stream
    )
    replay_total = (
        len(native_generated_tokens)
        if native_generated_tokens is not None
        else expected_generated_tokens
        if expected_generated_tokens is not None
        else generation_budget
    )

    store = CaptureStore()
    handles: list[Any] = []
    try:
        _register_capture_hooks(
            text_model,
            embedding,
            store,
            capture_details=capture_details,
            handles=handles,
        )
        logger.info(
            f"[wrap] prefill start: input_tokens={input_ids.shape[1]}, "
            f"expected_replay_tokens={replay_total}, capture_details={capture_details}"
        )
        with (
            LLMInferenceContextManager(xh_model, devices=[device]),
            torch.no_grad(),
            _token_progress(
                total=replay_total,
                description=f"Wrap {device}",
                enabled=show_progress,
                position=progress_position,
            ) as progress,
        ):
            store.set_stage("prefill", valid_sequence_length=input_ids.shape[1])
            _record_wrap_rotary(store, text_model, start=0, length=input_ids.shape[1])
            xh_model.set_prefill()
            xh_model.set_input_sequence_length(input_ids.shape[1])
            data_processor = xh_model.get_data_preprocessor()
            prefill_inputs = data_processor({"input_ids": input_ids, "past_seq_length": 0})
            prefill_logits = xh_model.forward(*prefill_inputs)
            store.record("logits", prefill_logits, last_token=True)
            _record_wrap_cache(store, xh_model, valid_length=input_ids.shape[1])
            if generation_budget > 0:
                wrap_token = int(prefill_logits[:, -1, :].argmax(dim=-1).item())
                wrap_generated_tokens.append(wrap_token)
                progress.update(1)
                progress.set_postfix(token_id=wrap_token)

            # Feed Native's dynamically generated context to both models. If an
            # output token differs, continuing from separate contexts would make
            # subsequent layer errors incomparable.
            for step, generated_token in enumerate(_all_but_last(token_source)):
                past_length = input_ids.shape[1] + step
                valid_length = past_length + 1
                store.set_stage(_decode_stage(step, max_decode_steps), valid_sequence_length=valid_length)
                _record_wrap_rotary(store, text_model, start=past_length, length=1)
                xh_model.set_decode()
                decode_ids = torch.tensor([[generated_token]], dtype=torch.long)
                decode_inputs = data_processor({"input_ids": decode_ids, "past_seq_length": past_length})
                decode_logits = xh_model.forward(*decode_inputs)
                store.record("logits", decode_logits, last_token=True)
                wrap_token = int(decode_logits[:, -1, :].argmax(dim=-1).item())
                wrap_generated_tokens.append(wrap_token)
                progress.update(1)
                progress.set_postfix(token_id=wrap_token)
                _record_wrap_cache(store, xh_model, start=valid_length - 1, valid_length=valid_length)
            if native_generated_tokens is None:
                # The streaming total starts as an upper bound because Native
                # may stop early on EOS. Finalize it to the observed token count.
                progress.total = len(wrap_generated_tokens)
                progress.refresh()
        logger.info(
            f"[wrap] replay complete: predicted_tokens={len(wrap_generated_tokens)}, "
            f"decode_steps={max(len(wrap_generated_tokens) - 1, 0)}, "
            f"elapsed={time.perf_counter() - run_started:.2f}s"
        )
        metadata = {
            "attention_implementation": "merak_graph_forward",
            "structure": _model_structure(text_model.config),
            "chip_arch": str(xh_model.config.chip_arch),
            "generated_tokens": len(wrap_generated_tokens),
            "decode_steps": max(len(wrap_generated_tokens) - 1, 0),
        }
    finally:
        _remove_hooks(handles)
        del xh_model, wrap_root, text_model
        _release_cuda_memory(device)
    return store, wrap_generated_tokens, metadata


def _run_models_parallel(
    *,
    native_kwargs: dict[str, Any],
    wrap_kwargs: dict[str, Any],
    expected_generated_tokens: int,
) -> tuple[
    tuple[CaptureStore, list[int], dict[str, Any]],
    tuple[CaptureStore, list[int], dict[str, Any]],
]:
    token_queue: queue.Queue[Any] = queue.Queue()

    def run_native_worker():
        try:
            result = _run_native(**native_kwargs, token_callback=token_queue.put)
        except BaseException as error:
            token_queue.put(_TokenStreamError(error))
            raise
        token_queue.put(_TOKEN_STREAM_END)
        return result

    streaming_wrap_kwargs = {
        **wrap_kwargs,
        "native_generated_tokens": None,
        "native_token_stream": _iter_token_queue(token_queue),
        "expected_generated_tokens": expected_generated_tokens,
    }
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="qwen3-layerwise") as executor:
        native_future = executor.submit(run_native_worker)
        wrap_future = executor.submit(_run_wrap, **streaming_wrap_kwargs)
        native_result = native_future.result()
        wrap_result = wrap_future.result()
    return native_result, wrap_result


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


def _resolve_execution_devices(args: argparse.Namespace) -> tuple[torch.device, torch.device]:
    def normalize(label: str, value: str) -> torch.device:
        device = torch.device(value)
        if device.type != "cuda":
            return device
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested for {label}, but CUDA is unavailable: {device}")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"invalid {label} CUDA device index {index}")
        return torch.device("cuda", index)

    native_device = normalize("native", args.native_device or args.device)
    wrap_device = normalize("wrap", args.wrap_device or args.device)
    for label, device in (("native", native_device), ("wrap", wrap_device)):
        if args.auto_offload and device.type != "cuda":
            raise ValueError(f"--auto-offload requires a CUDA {label} device")
    if args.parallel:
        if native_device.type != "cuda" or wrap_device.type != "cuda":
            raise ValueError("--parallel requires two CUDA devices")
        if native_device == wrap_device:
            raise ValueError("--parallel requires distinct native and wrap CUDA devices")
    return native_device, wrap_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Optional override for model.hf_model in the Python config.",
    )
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH, help="Merak Python model config.")
    parser.add_argument("--prompt", default="1+1等于几", help="Prompt text or UTF-8 text file.")
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR, help="Directory for logs and reports.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Fallback execution device for both paths. Default: cuda:0.",
    )
    parser.add_argument("--native-device", help="Native execution device; defaults to --device.")
    parser.add_argument("--wrap-device", help="Wrap execution device; defaults to --device.")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Pipeline Native and Wrap concurrently; requires distinct CUDA devices.",
    )
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode on both sides.")
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=DEFAULT_MAX_TOTAL_TOKENS,
        help="Stop after prompt plus generated output reaches this length. Default: 2048.",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Capture only layer boundaries, norm, logits and cache.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm token progress bars.")
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
    config_path = str(Path(args.config_path))
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"model config does not exist: {config_path}")
    _, resolved_model_cfg = _load_merak_config(config_path, args.model_dir, args.seed)
    model_dir = str(Path(resolved_model_cfg.hf_model))
    if not Path(model_dir).is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    model_context_limit = int(resolved_model_cfg.context_max_length)
    comparison_limit = min(DEFAULT_MAX_TOTAL_TOKENS, model_context_limit)
    if args.max_total_tokens > comparison_limit:
        raise ValueError(
            f"--max-total-tokens cannot exceed {comparison_limit} "
            f"(tool limit={DEFAULT_MAX_TOTAL_TOKENS}, model context={model_context_limit})"
        )
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    native_device, wrap_device = _resolve_execution_devices(args)
    dtype = _torch_dtype(args.dtype)
    _set_determinism(args.seed)
    from xhquant.api import get_xhquant_logger, xhquant_init

    xhquant_init(str(work_dir / "layerwise_diff.log"), args.debug)
    logger = get_xhquant_logger()
    comparison_started = time.perf_counter()
    logger.info(
        f"[compare] start: model={model_dir}, native_device={native_device}, "
        f"wrap_device={wrap_device}, parallel={args.parallel}, dtype={dtype}, "
        f"max_total_tokens={args.max_total_tokens}, capture_details={not args.no_details}"
    )

    prompt = _read_prompt(args.prompt)
    logger.info(f"[compare] tokenizing prompt: source={args.prompt!r}, characters={len(prompt)}")
    tokenizer, messages, input_ids = _build_input_ids(model_dir, prompt, args.think)
    generation_budget = _generation_budget(
        input_length=int(input_ids.shape[1]),
        max_total_tokens=args.max_total_tokens,
    )
    max_decode_steps = max(generation_budget - 1, 0)
    logger.info(
        f"[compare] input ready: input_tokens={input_ids.shape[1]}, "
        f"generation_budget={generation_budget}, max_decode_steps={max_decode_steps}"
    )
    input_metadata = {
        "model_dir": str(Path(model_dir).resolve()),
        "config_path": str(Path(config_path).resolve()),
        "prompt": prompt,
        "think": bool(args.think),
        "seed": int(args.seed),
        "device": str(args.device),
        "native_device": str(native_device),
        "wrap_device": str(wrap_device),
        "parallel": bool(args.parallel),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": str(dtype).removeprefix("torch."),
        "input_shape": list(input_ids.shape),
        "input_ids_sha256": _input_ids_sha256(input_ids),
        "tokens": input_ids.tolist(),
        "transformers_version": __import__("transformers").__version__,
        "max_total_tokens": int(args.max_total_tokens),
        "generation_budget": generation_budget,
        "max_decode_steps": max_decode_steps,
        "cache_capture": "full_prefill_and_appended_decode_token",
    }

    native_kwargs = dict(
        model_dir=model_dir,
        config_path=config_path,
        messages=messages,
        input_ids=input_ids,
        device=native_device,
        dtype=dtype,
        seed=args.seed,
        think=args.think,
        work_dir=work_dir,
        max_total_tokens=args.max_total_tokens,
        capture_details=not args.no_details,
        show_progress=not args.no_progress,
        progress_position=0,
        auto_offload=args.auto_offload,
        debug=args.debug,
    )
    wrap_kwargs = dict(
        model_dir=model_dir,
        config_path=config_path,
        messages=messages,
        input_ids=input_ids,
        device=wrap_device,
        dtype=dtype,
        seed=args.seed,
        think=args.think,
        work_dir=work_dir,
        max_total_tokens=args.max_total_tokens,
        capture_details=not args.no_details,
        show_progress=not args.no_progress,
        progress_position=1 if args.parallel else 0,
        auto_offload=args.auto_offload,
        debug=args.debug,
    )
    if args.parallel:
        logger.info("[compare] starting two-GPU token-streaming pipeline")
        native_result, wrap_result = _run_models_parallel(
            native_kwargs=native_kwargs,
            wrap_kwargs=wrap_kwargs,
            expected_generated_tokens=generation_budget,
        )
        native_store, native_generated_tokens, native_metadata = native_result
        wrap_store, wrap_generated_tokens, wrap_metadata = wrap_result
    else:
        logger.info("[compare] starting sequential Native then Wrap execution")
        native_store, native_generated_tokens, native_metadata = _run_native(
            **native_kwargs,
            token_callback=None,
        )
        wrap_store, wrap_generated_tokens, wrap_metadata = _run_wrap(
            **wrap_kwargs,
            native_generated_tokens=native_generated_tokens,
            native_token_stream=None,
            expected_generated_tokens=len(native_generated_tokens),
        )
    if native_metadata["structure"] != wrap_metadata["structure"]:
        raise RuntimeError(
            "native/wrap model structure mismatch: "
            f"native={native_metadata['structure']}, wrap={wrap_metadata['structure']}"
        )
    metadata = {
        **input_metadata,
        "generated_tokens": len(native_generated_tokens),
        "decode_steps": int(native_metadata["decode_steps"]),
        "generation_stop_reason": native_metadata["generation_stop_reason"],
        "eos_token_ids": native_metadata["eos_token_ids"],
        "native_generated_token_ids": native_generated_tokens,
        "wrap_generated_token_ids": wrap_generated_tokens,
        "native_generated_token_texts": [
            tokenizer.decode([token], skip_special_tokens=False) for token in native_generated_tokens
        ],
        "wrap_generated_token_texts": [
            tokenizer.decode([token], skip_special_tokens=False) for token in wrap_generated_tokens
        ],
        "native_generated_text": tokenizer.decode(native_generated_tokens, skip_special_tokens=False),
        "wrap_generated_text": tokenizer.decode(wrap_generated_tokens, skip_special_tokens=False),
        "native": native_metadata,
        "wrap": wrap_metadata,
    }
    del tokenizer
    logger.info(
        f"[compare] building tensor report: native_captures={len(native_store.tensors)}, "
        f"wrap_captures={len(wrap_store.tensors)}"
    )
    report = _build_report(native_store, wrap_store, metadata=metadata)
    json_path, csv_path = _write_report(report, work_dir)
    logger.info(
        f"[compare] report complete: exact={report['summary']['exact']}, "
        f"different_tensors={report['summary']['different_tensors']}, "
        f"output_tokens_exact={report['summary']['output_tokens_exact']}, "
        f"elapsed={time.perf_counter() - comparison_started:.2f}s"
    )
    logger.info(f"[compare] JSON report: {json_path}")
    logger.info(f"[compare] CSV report: {csv_path}")
    report_text = _report_text(report)
    print(report_text, end="")
    log_path = work_dir / "layerwise_diff.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n=== Human-readable layerwise diff report ===\n")
        log_file.write(report_text)
        log_file.write(f"JSON report: {json_path}\n")
        log_file.write(f"CSV report: {csv_path}\n")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    return 1 if args.fail_on_diff and not report["summary"]["exact"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
