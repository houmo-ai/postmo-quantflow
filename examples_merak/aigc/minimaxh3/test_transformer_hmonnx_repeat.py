#!/usr/bin/env python3
"""Isolate MiniMax-H3 transformer HMONNX multi-call behavior.

Demo ``pipe()`` denoise calls the transformer once per inference step
(default ``num_inference_steps=4``). This script skips the full pipeline and
only answers:

1. Can the same ``HMONNXGoldenInference`` session be invoked N times?
2. Optional: does ``_HMONNXTransformerModule`` also survive N calls after FP
   fallback release?

Example:
  CUDA_VISIBLE_DEVICES=6 python examples_merak/aigc/minimaxh3/test_transformer_hmonnx_repeat.py \\
    --export-dir work_dirs/minimax_h3_w8a8ssfp --calls 4 --device cuda:0

Known failure (verified 2026-09-20):
  First HMONNX forward dies at full-sequence Softmax ``node_softmax_2`` with
  shape ``(1, 56, 8868, 8868)`` (numel=4_403_919_744 > int32 max). Smaller
  Softmax ``(1, 56, 512, 512)`` succeeds. Both ``softmax_xh2a_default``
  (``lut_align_triton``) and ``softmax_xh2a_fast`` (``accu``/``hsum``) IMA once
  numel exceeds ``2**31-1``; plain ``torch.softmax`` on the same shape is fine.
  Use ``--probe-softmax`` to reproduce the boundary without loading the ONNX.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from xhmodel_merak.xh_other_model.models.minimax_h3.minimax_h3_converter import MiniMaxH3Converter


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("work_dirs/minimax_h3_w8a8ssfp"),
        help="Directory containing export_meta_info.json and transformer HMONNX",
    )
    parser.add_argument("--calls", type=int, default=4, help="How many times to invoke the same session")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 under CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("session", "adapter", "both"),
        default="session",
        help="session=HMONNXGoldenInference only; adapter wraps install-like host precompute",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Retain every call's outputs (stresses memory). Default frees outputs each call.",
    )
    parser.add_argument(
        "--vary-indices",
        action="store_true",
        help="Rebuild in-range timestep/adaln indices each call while keeping shapes fixed",
    )
    parser.add_argument(
        "--probe-softmax",
        action="store_true",
        help="Skip ONNX; probe Softmax backends around int32 numel boundary and real attn shape",
    )
    return parser.parse_args()


def _load_transformer_meta(export_dir: Path) -> tuple[Path, dict[str, Any]]:
    export_dir = export_dir.resolve()
    meta_path = export_dir / "export_meta_info.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing export metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    component = meta.get("components", {}).get("transformer")
    if not isinstance(component, dict) or not component.get("hmonnx_file"):
        raise KeyError(f"Export has no transformer component: {meta_path}")
    path = (export_dir / str(component["hmonnx_file"])).resolve()
    if export_dir not in path.parents or not path.is_file():
        raise FileNotFoundError(f"Invalid transformer HMONNX path: {path}")
    return path, component


def _mem_snapshot(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cpu"
    try:
        free, total = torch.cuda.mem_get_info(device)
        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
    except Exception as exc:  # noqa: BLE001 - CUDA may already be poisoned
        return f"<cuda mem unavailable after error: {type(exc).__name__}>"
    return (
        f"free={free / (1024**3):.2f}GiB/{total / (1024**3):.2f}GiB "
        f"alloc={alloc / (1024**3):.2f}GiB reserved={reserved / (1024**3):.2f}GiB"
    )


def _probe_softmax(device: torch.device) -> None:
    """Reproduce Softmax IMA around int32 numel overflow without loading ONNX."""
    from xhquant.backend.xh2a.functions.softmax import (
        generate_softmax_xh2a_lut,
        softmax_xh2a_default,
        softmax_xh2a_fast,
    )

    int32_max = 2**31 - 1
    table = generate_softmax_xh2a_lut().to(device)
    cut_t, vals_t, scales_t = table.cut_points, table.table, table.mul_scale
    cut, vals, scales = cut_t.tolist(), vals_t.tolist(), scales_t.tolist()
    cases = [
        ("torch", (1, 56, 8868, 8868), "torch"),
        ("xh2a_default_below", (1, 24, 8868, 8868), "default"),
        ("xh2a_default_above", (1, 28, 8868, 8868), "default"),
        ("xh2a_fast_below", (1, 24, 8868, 8868), "fast"),
        ("xh2a_fast_above", (1, 28, 8868, 8868), "fast"),
        ("xh2a_fast_real", (1, 56, 8868, 8868), "fast"),
        ("xh2a_default_real", (1, 56, 8868, 8868), "default"),
    ]
    for label, shape, kind in cases:
        numel = 1
        for dim in shape:
            numel *= int(dim)
        print(
            f"[softmax-probe] {label} shape={shape} numel={numel} "
            f"overflow={numel > int32_max} input_GiB={numel * 2 / (1024**3):.2f}",
            flush=True,
        )
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            x = torch.randn(*shape, device=device, dtype=torch.float16)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if kind == "torch":
                y = torch.softmax(x, dim=-1)
            elif kind == "fast":
                y = softmax_xh2a_fast(x, cut, vals, scales, dim=-1)
            else:
                y = softmax_xh2a_default(x, cut_t, vals_t, scales_t, dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device) / (1024**3)
            else:
                peak = 0.0
            print(f"[softmax-probe] {label} OK peak={peak:.2f}GiB out={tuple(y.shape)}", flush=True)
            del y, x
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - probe must continue across poisoned CUDA
            print(f"[softmax-probe] {label} FAIL {type(exc).__name__}: {exc}", flush=True)
            print(f"[softmax-probe] mem={_mem_snapshot(device)}", flush=True)
            # CUDA context is usually dead after IMA; stop further probes in-process.
            return


INDEX_INPUT_NAMES = {
    "timestep_indices",
    "adaln_indices",
    "video_indices",
    "audio_indices",
    "text_indices",
    "pack_gather_indices",
}


def _session_input_dtypes(session: Any) -> dict[str, torch.dtype]:
    """Prefer live HMONNX session dtypes over export_meta_info (meta may say int64)."""
    for attr in ("input_dtypes", "inputs_dtype", "input_type_map"):
        value = getattr(session, attr, None)
        if isinstance(value, dict) and value:
            return {str(k): v for k, v in value.items()}

    nested = getattr(session, "_session", None)
    if nested is not None:
        for attr in ("input_dtypes", "inputs_dtype", "input_type_map"):
            value = getattr(nested, attr, None)
            if isinstance(value, dict) and value:
                return {str(k): v for k, v in value.items()}
        names = getattr(nested, "input_names", None)
        dtypes = getattr(nested, "input_dtypes", None)
        if names is not None and dtypes is not None and len(names) == len(dtypes):
            return {str(name): dtype for name, dtype in zip(names, dtypes)}

    # Adapter path already casts indices to int32; mirror that when session metadata is opaque.
    return {name: torch.int32 for name in INDEX_INPUT_NAMES}


def _build_named_inputs(
    component: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
    dtype_overrides: dict[str, torch.dtype] | None = None,
) -> dict[str, torch.Tensor]:
    specs = list(component.get("inputs") or [])
    if not specs:
        raise ValueError("transformer meta has empty inputs")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    index_tensors = MiniMaxH3Converter._build_transformer_index_tensors(
        specs,
        device=torch.device("cpu"),
        generator=generator,
    )
    overrides = dtype_overrides or {}
    named: dict[str, torch.Tensor] = {}
    for spec in specs:
        name = str(spec["name"])
        shape = tuple(int(v) for v in spec["shape"])
        dtype = overrides.get(name, DTYPE_MAP[str(spec.get("dtype", "float16"))])
        if name in INDEX_INPUT_NAMES and name not in overrides:
            # Graph ABI used by adapter/runtime is int32 even when export meta says int64.
            dtype = torch.int32
        if name in index_tensors:
            tensor = index_tensors[name].to(dtype=dtype)
        elif name == "encoder_attention_mask":
            valid_length = int(spec.get("valid_length", shape[-1]))
            tensor = torch.zeros(shape, dtype=dtype)
            tensor[..., valid_length:] = torch.finfo(dtype).min
        elif name == "packed_attention_mask":
            # Filled after the loop from encoder_attention_mask + text_indices.
            tensor = torch.zeros(shape, dtype=dtype)
        elif dtype.is_floating_point:
            tensor = torch.randn(shape, dtype=dtype, generator=generator)
        elif dtype == torch.bool:
            tensor = torch.zeros(shape, dtype=dtype)
        else:
            tensor = torch.zeros(shape, dtype=dtype)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != meta {shape}")
        named[name] = tensor.to(device=device, non_blocking=device.type == "cuda")
    # text_indices are host-only (not in export input_names); keep them for packing builders.
    if "text_indices" not in named and "text_indices" in index_tensors:
        text_dtype = overrides.get("text_indices", torch.int32)
        named["text_indices"] = index_tensors["text_indices"].to(
            device=device,
            dtype=text_dtype,
            non_blocking=device.type == "cuda",
        )
    if "packed_attention_mask" in named:
        from xhmodel_merak.xh_other_model.models.minimax_h3.transformer_wrapper import (
            build_packed_attention_mask,
        )

        if "encoder_attention_mask" not in named or "text_indices" not in named:
            raise KeyError("packed_attention_mask requires encoder_attention_mask and text_indices")
        named["packed_attention_mask"] = build_packed_attention_mask(
            named["encoder_attention_mask"],
            named["text_indices"],
            sequence_length=int(named["packed_attention_mask"].shape[-1]),
        ).to(device=device, dtype=named["packed_attention_mask"].dtype)
    if "pack_gather_indices" in named:
        from xhmodel_merak.xh_other_model.models.minimax_h3.transformer_wrapper import (
            build_pack_gather_indices,
        )

        missing = [name for name in ("text_indices", "video_indices", "audio_indices") if name not in named]
        if missing:
            raise KeyError(f"pack_gather_indices requires modality indices, missing: {missing}")
        named["pack_gather_indices"] = build_pack_gather_indices(
            named["text_indices"],
            named["video_indices"],
            named["audio_indices"],
            sequence_length=int(named["pack_gather_indices"].shape[0]),
        ).to(device=device, dtype=named["pack_gather_indices"].dtype)
    return named


def _ordered_inputs(component: dict[str, Any], named: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    names = tuple(component.get("input_names") or [spec["name"] for spec in component["inputs"]])
    missing = [name for name in names if name not in named]
    if missing:
        raise KeyError(f"Missing transformer inputs: {missing}")
    return tuple(named[name] for name in names)


def _align_named_to_session(named: dict[str, torch.Tensor], session: Any) -> dict[str, torch.Tensor]:
    dtypes = _session_input_dtypes(session)
    aligned = {}
    for name, tensor in named.items():
        expected = dtypes.get(name)
        if expected is not None and tensor.dtype != expected:
            tensor = tensor.to(dtype=expected)
        aligned[name] = tensor
    return aligned


def _refresh_indices(
    named: dict[str, torch.Tensor],
    component: dict[str, Any],
    *,
    seed: int,
    call_idx: int,
) -> None:
    specs = list(component["inputs"])
    generator = torch.Generator(device="cpu").manual_seed(seed + call_idx + 1)
    index_tensors = MiniMaxH3Converter._build_transformer_index_tensors(
        specs,
        device=torch.device("cpu"),
        generator=generator,
    )
    device = next(iter(named.values())).device
    for name, tensor in index_tensors.items():
        named[name] = tensor.to(device=device, dtype=named[name].dtype, non_blocking=device.type == "cuda")


def _run_session_calls(
    *,
    hmonnx_path: Path,
    component: dict[str, Any],
    device: torch.device,
    calls: int,
    seed: int,
    keep_outputs: bool,
    vary_indices: bool,
) -> None:
    from xhquant.api import HMONNXGoldenInference

    print(f"[session] loading {hmonnx_path}", flush=True)
    print(f"[session] before load: {_mem_snapshot(device)}", flush=True)
    session = HMONNXGoldenInference(str(hmonnx_path), exec_device=device)
    session.exec_device = device
    named = _build_named_inputs(component, device=device, seed=seed)
    named = _align_named_to_session(named, session)
    print(
        "[session] input dtypes=" + ", ".join(f"{name}:{tensor.dtype}" for name, tensor in named.items()),
        flush=True,
    )
    kept = []
    print(f"[session] after inputs: {_mem_snapshot(device)}", flush=True)

    for call_idx in range(calls):
        if vary_indices and call_idx:
            _refresh_indices(named, component, seed=seed, call_idx=call_idx)
            named = _align_named_to_session(named, session)
        inputs = _ordered_inputs(component, named)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        print(f"[session] call {call_idx + 1}/{calls} begin: {_mem_snapshot(device)}", flush=True)
        try:
            outputs = session(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = time.perf_counter() - t0
            if isinstance(outputs, (tuple, list)):
                shapes = [tuple(t.shape) if torch.is_tensor(t) else type(t).__name__ for t in outputs]
            else:
                shapes = [tuple(outputs.shape) if torch.is_tensor(outputs) else type(outputs).__name__]
            print(
                f"[session] call {call_idx + 1}/{calls} ok in {dt:.2f}s shapes={shapes} mem={_mem_snapshot(device)}",
                flush=True,
            )
            if keep_outputs:
                kept.append(outputs)
            else:
                del outputs
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception:
            print(f"[session] call {call_idx + 1}/{calls} FAILED after {time.perf_counter() - t0:.2f}s", flush=True)
            print(f"[session] mem at failure: {_mem_snapshot(device)}", flush=True)
            traceback.print_exc()
            raise

    print(f"[session] all {calls} calls succeeded; final mem={_mem_snapshot(device)}", flush=True)
    del session
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


class _HostOnlyTransformer(torch.nn.Module):
    """Minimal host modules so adapter can precompute temb/adaln/rotary."""

    def __init__(self, temb_dim: int, rope_dim: int, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
        self.config = type("Cfg", (), {"patch_size": (1, 2, 2)})()
        self.time_proj = torch.nn.Identity()
        self.time_embedder = _Temb(temb_dim, dtype)
        self.rope = _Rope(rope_dim, dtype)


class _Temb(torch.nn.Module):
    def __init__(self, temb_dim: int, dtype: torch.dtype):
        super().__init__()
        self.linear_1 = torch.nn.Linear(1, temb_dim, bias=False).to(dtype=dtype)
        self.temb_dim = temb_dim
        self._dtype = dtype

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        # Match export ABI leading dim from caller-provided unique timesteps.
        if timestep.ndim == 0:
            timestep = timestep.reshape(1)
        flat = timestep.reshape(-1, 1).to(dtype=self._dtype)
        return self.linear_1(flat)


class _Rope(torch.nn.Module):
    def __init__(self, rope_dim: int, dtype: torch.dtype):
        super().__init__()
        self.rope_dim = rope_dim
        self._dtype = dtype

    def forward(self, position_ids: torch.Tensor):
        seq = position_ids.shape[0]
        zeros = torch.zeros(seq, self.rope_dim, device=position_ids.device, dtype=self._dtype)
        return zeros, zeros


def _run_adapter_calls(
    *,
    hmonnx_path: Path,
    component: dict[str, Any],
    device: torch.device,
    calls: int,
    seed: int,
    keep_outputs: bool,
    vary_indices: bool,
) -> None:
    from xhmodel_merak.xh_other_model.models.minimax_h3.components_hmonnx import (
        _HMONNXTransformerModule,
        _LazyHMONNXRuntime,
    )

    named = _build_named_inputs(component, device=device, seed=seed)
    temb_dim = int(named["temb"].shape[-1])
    rope_dim = int(named["rotary_cos"].shape[-1])
    host = _HostOnlyTransformer(temb_dim, rope_dim, torch.float16).to(device)
    shapes = tuple(tuple(named[name].shape) for name in component["input_names"])
    runtime = _LazyHMONNXRuntime(hmonnx_path, device, shapes)
    adapter = _HMONNXTransformerModule(runtime, host, tuple(component["input_names"]))
    adapter._release_fallback()
    print(f"[adapter] installed; fallback released; mem={_mem_snapshot(device)}", flush=True)

    seq = int(named["adaln_indices"].shape[0])
    text_len = int(named["text_indices"].shape[0])
    video_len = int(named["video_indices"].shape[0])
    audio_len = int(named["audio_indices"].shape[0])
    num_timesteps = int(named["temb"].shape[0])
    kept = []

    for call_idx in range(calls):
        if vary_indices and call_idx:
            _refresh_indices(named, component, seed=seed, call_idx=call_idx)
        # Rebuild host-side kwargs from packed tensors so adapter precompute path is used.
        timestep = torch.linspace(0.1, 0.9, steps=num_timesteps, device=device, dtype=torch.float32)
        timestep_indices = named["timestep_indices"].to(dtype=torch.long)
        # Derive tags from adaln formula: adaln = ti * 3 + tag
        token_tags = named["adaln_indices"].to(dtype=torch.long) - timestep_indices * 3
        position_ids = torch.zeros(seq, 3, device=device, dtype=torch.float32)
        position_ids[:, 0] = torch.arange(seq, device=device, dtype=torch.float32)
        kwargs = {
            "hidden_states": named["hidden_states"],
            "audio_hidden_states": named["audio_hidden_states"],
            "encoder_hidden_states": named["encoder_hidden_states"],
            "timestep": timestep,
            "timestep_indices": timestep_indices,
            "token_tags": token_tags,
            "position_ids": position_ids,
            "video_indices": named["video_indices"].to(dtype=torch.long),
            "audio_indices": named["audio_indices"].to(dtype=torch.long),
            "text_indices": named["text_indices"].to(dtype=torch.long),
            "encoder_attention_mask": named["encoder_attention_mask"],
        }
        # Sanity: packing lengths must match export geometry.
        assert text_len + video_len + audio_len == seq, (text_len, video_len, audio_len, seq)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        print(f"[adapter] call {call_idx + 1}/{calls} begin: {_mem_snapshot(device)}", flush=True)
        try:
            outputs = adapter(**kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = time.perf_counter() - t0
            if isinstance(outputs, (tuple, list)):
                shapes_out = [tuple(t.shape) for t in outputs if torch.is_tensor(t)]
            else:
                shapes_out = [tuple(outputs.shape)]
            print(
                f"[adapter] call {call_idx + 1}/{calls} ok in {dt:.2f}s shapes={shapes_out} mem={_mem_snapshot(device)}",
                flush=True,
            )
            if keep_outputs:
                kept.append(outputs)
            else:
                del outputs
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception:
            print(f"[adapter] call {call_idx + 1}/{calls} FAILED after {time.perf_counter() - t0:.2f}s", flush=True)
            print(f"[adapter] mem at failure: {_mem_snapshot(device)}", flush=True)
            traceback.print_exc()
            raise

    print(f"[adapter] all {calls} calls succeeded; runtime still set={adapter.runtime is not None}", flush=True)


def main() -> None:
    args = _parse_args()
    if args.calls <= 0:
        raise ValueError("--calls must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    if args.probe_softmax:
        print(f"mode=probe-softmax device={device}", flush=True)
        print(f"start mem: {_mem_snapshot(device)}", flush=True)
        _probe_softmax(device)
        return

    hmonnx_path, component = _load_transformer_meta(args.export_dir)
    print(
        f"transformer={hmonnx_path}\n"
        f"calls={args.calls} mode={args.mode} device={device} "
        f"vary_indices={args.vary_indices} keep_outputs={args.keep_outputs}",
        flush=True,
    )
    print(f"input_names={component.get('input_names')}", flush=True)
    for idx, spec in enumerate(component.get("inputs") or []):
        print(f"  [{idx}] {spec.get('name')} {spec.get('shape')} {spec.get('dtype')}", flush=True)
    print(f"start mem: {_mem_snapshot(device)}", flush=True)

    try:
        if args.mode in ("session", "both"):
            _run_session_calls(
                hmonnx_path=hmonnx_path,
                component=component,
                device=device,
                calls=args.calls,
                seed=args.seed,
                keep_outputs=args.keep_outputs,
                vary_indices=args.vary_indices,
            )
        if args.mode in ("adapter", "both"):
            _run_adapter_calls(
                hmonnx_path=hmonnx_path,
                component=component,
                device=device,
                calls=args.calls,
                seed=args.seed,
                keep_outputs=args.keep_outputs,
                vary_indices=args.vary_indices,
            )
    finally:
        if device.type == "cuda":
            print(
                f"peak alloc={torch.cuda.max_memory_allocated(device) / (1024**3):.2f}GiB "
                f"peak reserved={torch.cuda.max_memory_reserved(device) / (1024**3):.2f}GiB",
                flush=True,
            )


if __name__ == "__main__":
    main()
