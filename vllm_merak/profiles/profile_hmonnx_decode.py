#!/usr/bin/env python3
"""Profile steady-state decode latency of a PageAttention HMONNX graph.

The session always uses HMONNXInferenceV2 with auto-offload and CUDA Graph.
PageAttention SEFP caches and their graph-stable metadata buffers are inferred
from the parsed graph. --past-seq-len controls the active context, while
--max-seq-len controls the independently allocated cache capacity.

Example:
  CUDA_VISIBLE_DEVICES=4,5 conda run -n xhquant_vllm_attn_cpu \
    python profiles/profile_hmonnx_decode.py --past-seq-len 4096
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from xhquant.core.hmfp_kv_cache import HMFPPagedKVCache, allocate_hmfp_paged_kv_cache
from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2
from xhquant.xhonnxruntime.parsers import PageAttention, PageAttentionContext

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / (
    "data/models/hmquant_xh2_qwen3_6_35b_a3b_w4a8_256_256k_mpe256k_448x448_20260723/"
    "decode/hmquant_xh2_qwen3_6_35b_a3b_w4a8_256_256k_mpe256k_448x448_20260723_"
    "decode_with_act_linear_fused_page_attention_f1d4b4a3bbc6b6c6.onnx"
)


@dataclass(frozen=True)
class PageAttentionSpec:
    node_name: str
    input_names: tuple[str, ...]
    device: str
    num_heads: int
    num_kv_heads: int
    head_size: int
    nshare: int
    block_size: int
    active_blocks: int
    block_table_blocks: int
    num_blocks: int
    k_bits: int
    v_bits: int
    cache_bytes: int


def _parse_devices(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("--devices must contain at least one logical CUDA device index")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("--devices contains duplicate indices")
    return devices


def _as_cuda_device(value: Any) -> torch.device:
    if isinstance(value, int):
        return torch.device(f"cuda:{value}")
    device = torch.device(value)
    if device.type != "cuda":
        raise RuntimeError(f"expected a CUDA execution device, got {device}")
    return device


def _all_execution_devices(session: HMONNXInferenceV2) -> list[torch.device]:
    node_device_map = getattr(session.interpreter, "node_device_map", {})
    devices = {
        _as_cuda_device(info.device) for info in node_device_map.values() if torch.device(info.device).type == "cuda"
    }
    if not devices:
        devices = {_as_cuda_device(session.device)}
    return sorted(devices, key=lambda device: device.index if device.index is not None else -1)


def _synchronize(devices: Iterable[torch.device]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _cache_bytes(cache: HMFPPagedKVCache) -> int:
    return sum(_tensor_bytes(tensor) for tensor in cache.tensors())


def _qdata_dtype(bits: int) -> torch.dtype:
    if bits <= 8:
        return torch.int8
    if bits <= 16:
        return torch.int16
    raise ValueError(f"unsupported SEFP qdata width: {bits} bits")


def _block_table_capacity(active_blocks: int, allocated_blocks: int) -> int:
    if active_blocks <= 0:
        raise ValueError(f"active_blocks must be positive, got {active_blocks}")
    capacity = 1 << (active_blocks - 1).bit_length()
    return min(capacity, allocated_blocks)


def _page_attention_nodes(session: HMONNXInferenceV2) -> list[tuple[Any, PageAttention]]:
    result: list[tuple[Any, PageAttention]] = []
    graph_module = session.graph_module
    if graph_module is None:
        raise RuntimeError("HMONNXInferenceV2 did not create an FX GraphModule")
    for node in graph_module.graph.nodes:
        if node.op != "call_module":
            continue
        module = graph_module.get_submodule(str(node.target))
        if isinstance(module, PageAttention):
            result.append((node, module))
    if not result:
        raise RuntimeError("the HMONNX graph has no parsed PageAttention modules")
    return result


def _runtime_page_attention_modules(session: HMONNXInferenceV2) -> list[PageAttention]:
    """Return unique PageAttention instances from the source and split stage graphs."""
    roots: list[torch.nn.Module] = []
    if session.graph_module is not None:
        roots.append(session.graph_module)

    interpreter = session.interpreter
    pipe = getattr(interpreter, "_pipe", None)
    split_gm = getattr(pipe, "split_gm", None)
    if isinstance(split_gm, torch.nn.Module):
        roots.append(split_gm)
    single = getattr(interpreter, "_single_cuda_graph_module", None)
    if isinstance(single, torch.nn.Module):
        roots.append(single)

    unique: dict[int, PageAttention] = {}
    for root in roots:
        for module in root.modules():
            if isinstance(module, PageAttention):
                unique[id(module)] = module
    return list(unique.values())


def _infer_head_size(session: HMONNXInferenceV2, module: PageAttention) -> int:
    if len(module.input_names) < 3:
        raise RuntimeError(f"PageAttention has incomplete Q/K/V inputs: {module.input_names}")
    head_sizes = []
    for name in module.input_names[:3]:
        info = session.tensors.get(name)
        if info is None or len(info.shape) == 0:
            raise RuntimeError(f"missing tensor metadata for PageAttention input {name!r}")
        head_sizes.append(int(info.shape[-1]))
    if len(set(head_sizes)) != 1:
        raise RuntimeError(f"PageAttention Q/K/V head sizes disagree: {head_sizes}")
    return head_sizes[0]


def _bind_page_attention_caches(
    session: HMONNXInferenceV2,
    *,
    past_seq_len: int,
    current_seq_len: int,
    max_seq_len: int,
) -> tuple[list[HMFPPagedKVCache], list[PageAttentionSpec]]:
    """Allocate one inferred SEFP cache per PageAttention and bind stable metadata."""
    node_device_map = getattr(session.interpreter, "node_device_map", None)
    if not isinstance(node_device_map, dict):
        raise RuntimeError("auto-offload interpreter does not expose node_device_map")

    page_nodes = _page_attention_nodes(session)
    contexts_by_inputs: dict[tuple[str, ...], PageAttentionContext] = {}
    caches: list[HMFPPagedKVCache] = []
    specs: list[PageAttentionSpec] = []
    metadata_by_device: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    total_seq_len = past_seq_len + current_seq_len
    for node, module in page_nodes:
        node_info = node_device_map.get(node.name)
        if node_info is None:
            raise RuntimeError(f"auto-offload has no device assignment for PageAttention node {node.name!r}")
        device = _as_cuda_device(node_info.device)
        nshare = int(module.nshare)
        # The Merak PageAttention physical page is one SEFP sharing group.  This
        # is the runtime-supported block size and avoids a separate CLI setting.
        block_size = nshare
        active_blocks = max(1, math.ceil(total_seq_len / block_size))
        num_blocks = max(1, math.ceil(max_seq_len / block_size))
        block_table_blocks = _block_table_capacity(active_blocks, num_blocks)
        head_size = _infer_head_size(session, module)

        cache = allocate_hmfp_paged_kv_cache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=int(module.num_kv_heads),
            head_size=head_size,
            device=device,
            nshare=nshare,
            k_qdata_dtype=_qdata_dtype(int(module.k_bits)),
            v_qdata_dtype=_qdata_dtype(int(module.v_bits)),
        )
        caches.append(cache)

        device_key = str(device)
        metadata = metadata_by_device.get(device_key)
        if metadata is None:
            # Identity logical->physical mapping.  slot_mapping points the one
            # decode token at its physical position immediately after the past.
            block_ids = torch.arange(active_blocks, dtype=torch.int64, device=device)
            if block_table_blocks > active_blocks:
                padding = block_ids.new_full((block_table_blocks - active_blocks,), active_blocks - 1)
                block_ids = torch.cat((block_ids, padding))
            slot_mapping = torch.arange(
                past_seq_len,
                past_seq_len + current_seq_len,
                dtype=torch.int64,
                device=device,
            )
            metadata = (block_ids, slot_mapping)
            metadata_by_device[device_key] = metadata
        else:
            block_ids, slot_mapping = metadata
            if block_ids.numel() != block_table_blocks or slot_mapping.numel() != current_seq_len:
                raise RuntimeError("PageAttention modules on one device require incompatible metadata capacities")

        context = PageAttentionContext(
            paged_kv_cache=cache,
            block_ids=metadata[0],
            slot_mapping=metadata[1],
            block_size=block_size,
        )
        input_key = tuple(module.input_names)
        if input_key in contexts_by_inputs:
            raise RuntimeError(f"duplicate PageAttention input signature: {input_key}")
        contexts_by_inputs[input_key] = context
        module.set_context(context)

        specs.append(
            PageAttentionSpec(
                node_name=node.name,
                input_names=input_key,
                device=device_key,
                num_heads=int(module.num_heads),
                num_kv_heads=int(module.num_kv_heads),
                head_size=head_size,
                nshare=nshare,
                block_size=block_size,
                active_blocks=active_blocks,
                block_table_blocks=block_table_blocks,
                num_blocks=num_blocks,
                k_bits=int(module.k_bits),
                v_bits=int(module.v_bits),
                cache_bytes=_cache_bytes(cache),
            )
        )

    # Pipe construction can retain or clone stage modules depending on the
    # installed torch version.  Bind both forms so capture never observes an
    # uninitialized PageAttention context.
    bound_keys: set[tuple[str, ...]] = set()
    for module in _runtime_page_attention_modules(session):
        key = tuple(module.input_names)
        context = contexts_by_inputs.get(key)
        if context is not None:
            module.set_context(context)
            bound_keys.add(key)
    missing = set(contexts_by_inputs) - bound_keys
    if missing:
        raise RuntimeError(f"failed to bind PageAttention runtime module(s): {sorted(missing)}")
    return caches, specs


def _infer_current_seq_len(session: HMONNXInferenceV2) -> int:
    if not session.inputs:
        raise RuntimeError("HMONNX graph has no inputs")
    primary = session.inputs[0]
    if len(primary.shape) < 2:
        raise RuntimeError(f"cannot infer decode sequence length from {primary.name}: {primary.shape}")
    current_seq_len = int(primary.shape[1])
    if current_seq_len <= 0:
        raise RuntimeError(f"invalid decode sequence length inferred from {primary.name}: {current_seq_len}")
    return current_seq_len


def _construct_inputs(
    session: HMONNXInferenceV2,
    *,
    past_seq_len: int,
    current_seq_len: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Construct every runtime input from HMONNX TensorInfo metadata."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    primary_name = session.inputs[0].name
    feed: dict[str, torch.Tensor] = {}

    for info in session.inputs:
        name = info.name
        shape = tuple(int(dim) for dim in info.shape)
        dtype = info.dtype
        lower = name.lower()

        if name == primary_name:
            if dtype.is_floating_point:
                value = torch.empty(shape, dtype=dtype, device=device).normal_(mean=0.0, std=0.02)
            else:
                value = torch.zeros(shape, dtype=dtype, device=device)
        elif lower in {"valid_length", "past_seq_length"}:
            value = torch.full(shape, past_seq_len, dtype=dtype, device=device)
        elif lower in {"current_length", "current_input_length"}:
            value = torch.full(shape, current_seq_len, dtype=dtype, device=device)
        elif lower in {"time_position_ids", "height_position_ids", "hight_position_ids", "width_position_ids"}:
            positions = torch.arange(
                past_seq_len,
                past_seq_len + math.prod(shape),
                dtype=dtype,
                device=device,
            )
            value = positions.reshape(shape)
        elif "mask" in lower:
            value = torch.zeros(shape, dtype=dtype, device=device)
            if value.numel() > 0:
                slices = [slice(None)] * value.dim()
                slices[-1] = slice(0, min(current_seq_len, shape[-1]))
                value[tuple(slices)] = 1
        else:
            # Linear-attention conv/recurrent states and any future graph inputs
            # are allocated exactly as declared by the HMONNX input metadata.
            value = torch.zeros(shape, dtype=dtype, device=device)
        feed[name] = value
    return feed


def _cuda_graph_wrappers(session: HMONNXInferenceV2) -> list[Any]:
    interpreter = session.interpreter
    wrappers: dict[int, Any] = {}
    single = getattr(interpreter, "_single_cuda_graph_module", None)
    if single is not None and hasattr(single, "has_captured_graph"):
        wrappers[id(single)] = single
    pipe = getattr(interpreter, "_pipe", None)
    split_gm = getattr(pipe, "split_gm", None)
    if isinstance(split_gm, torch.nn.Module):
        for module in split_gm.modules():
            candidate = getattr(module, "_interpreter", module)
            if hasattr(candidate, "has_captured_graph"):
                wrappers[id(candidate)] = candidate
    if hasattr(interpreter, "has_captured_graph"):
        wrappers[id(interpreter)] = interpreter
    return list(wrappers.values())


def _assert_cuda_graph_active(session: HMONNXInferenceV2) -> list[dict[str, Any]]:
    wrappers = _cuda_graph_wrappers(session)
    if not wrappers:
        raise RuntimeError("no CUDA Graph execution wrapper was found")
    status = []
    failed = []
    for wrapper in wrappers:
        active = bool(wrapper.has_captured_graph)
        reason = getattr(wrapper, "capture_disabled_reason", None)
        item = {
            "label": str(getattr(wrapper, "capture_label", type(wrapper).__name__)),
            "device": str(getattr(wrapper, "device", "unknown")),
            "active": active,
            "disabled_reason": reason,
        }
        status.append(item)
        if not active:
            failed.append(item)
    if failed:
        details = "; ".join(f"{item['label']}: {item['disabled_reason']}" for item in failed)
        raise RuntimeError(f"CUDA Graph capture is not active for every stage: {details}")
    return status


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _latency_summary(latencies_ms: list[float], tokens_per_step: int) -> dict[str, float | int]:
    mean_ms = statistics.fmean(latencies_ms)
    return {
        "iterations": len(latencies_ms),
        "mean_ms": mean_ms,
        "median_ms": statistics.median(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "p90_ms": _percentile(latencies_ms, 90),
        "p95_ms": _percentile(latencies_ms, 95),
        "p99_ms": _percentile(latencies_ms, 99),
        "tokens_per_second": tokens_per_step * 1000.0 / mean_ms,
    }


def _write_trace(
    trace_path: Path,
    session: HMONNXInferenceV2,
    feed: dict[str, torch.Tensor],
    devices: list[torch.device],
) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        with torch.profiler.record_function("hmonnx_decode_cuda_graph_replay"):
            session.run(feed)
            _synchronize(devices)
    profiler.export_chrome_trace(str(trace_path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="PageAttention decode HMONNX path")
    parser.add_argument("--past-seq-len", type=int, required=True, help="past sequence length before this decode token")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256 * 1024,
        help="preallocated SEFP cache capacity in tokens (default: 262144)",
    )
    parser.add_argument("--devices", type=_parse_devices, default=None, help="logical CUDA indices, e.g. 0,1")
    parser.add_argument("--warmup", type=int, default=5, help="steady CUDA Graph replay warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="measured steady decode iterations")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--json", type=Path, default=None, help="optional JSON result path")
    parser.add_argument("--trace", type=Path, default=None, help="optional Chrome trace for one steady replay")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate arguments and print capacity without loading HMONNX"
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"HMONNX file does not exist: {model_path}")
    if args.past_seq_len < 0:
        raise ValueError("--past-seq-len must be non-negative")
    if args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive")
    if args.past_seq_len + 1 > args.max_seq_len:
        raise ValueError("--past-seq-len plus the decode token exceeds --max-seq-len")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.dry_run:
        num_blocks = math.ceil(args.max_seq_len / 64)
        active_blocks = math.ceil((args.past_seq_len + 1) / 64)
        block_table_blocks = _block_table_capacity(active_blocks, num_blocks)
        print(f"model: {model_path}")
        print(
            "SEFP capacity: "
            f"max_seq_len={args.max_seq_len}, block_size=64, "
            f"num_blocks={num_blocks}, active_blocks={active_blocks}, "
            f"block_table_blocks={block_table_blocks}"
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("auto_offload + cuda_graph profiling requires CUDA")

    exec_devices = args.devices
    if exec_devices is None:
        exec_devices = list(range(torch.cuda.device_count()))
    if not exec_devices:
        raise RuntimeError("no visible CUDA devices")
    for index in exec_devices:
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is outside [0, {torch.cuda.device_count()})")

    config = HMONNXInferenceConfig(
        enable_cuda_graph=True,
        enable_auto_offload=True,
        enable_golden=False,
        exec_devices=exec_devices,
    )
    init_start = time.perf_counter()
    session = HMONNXInferenceV2(str(model_path), config)
    init_seconds = time.perf_counter() - init_start

    current_seq_len = _infer_current_seq_len(session)
    caches, page_specs = _bind_page_attention_caches(
        session,
        past_seq_len=args.past_seq_len,
        current_seq_len=current_seq_len,
        max_seq_len=args.max_seq_len,
    )
    execution_devices = _all_execution_devices(session)
    input_device = _as_cuda_device(session.device)
    feed = _construct_inputs(
        session,
        past_seq_len=args.past_seq_len,
        current_seq_len=current_seq_len,
        device=input_device,
        seed=args.seed,
    )

    print(f"model: {model_path}")
    print(f"HMONNX inputs/outputs: {len(session.inputs)}/{len(session.outputs)}")
    print(f"session init: {init_seconds:.3f} s")
    print(f"auto-offload devices: {', '.join(map(str, execution_devices))}")
    placement = Counter(spec.device for spec in page_specs)
    print(
        "PageAttention: "
        f"layers={len(page_specs)}, placement={dict(placement)}, "
        f"past_seq_len={args.past_seq_len}, current_seq_len={current_seq_len}, "
        f"block_size={page_specs[0].block_size}, active_blocks={page_specs[0].active_blocks}, "
        f"block_table_blocks={page_specs[0].block_table_blocks}, "
        f"allocated_blocks={page_specs[0].num_blocks}, "
        f"SEFP cache={sum(spec.cache_bytes for spec in page_specs) / (1024**2):.2f} MiB"
    )

    # First call performs runtime warmup and CUDA Graph capture.  It is never
    # included in the steady-state latency samples.
    capture_start = time.perf_counter()
    outputs = session.run(feed)
    _synchronize(execution_devices)
    capture_seconds = time.perf_counter() - capture_start
    graph_status = _assert_cuda_graph_active(session)
    print(f"CUDA Graph capture call: {capture_seconds:.3f} s")
    for item in graph_status:
        print(f"  replay active: {item['label']} ({item['device']})")

    for _ in range(args.warmup):
        outputs = session.run(feed)
    _synchronize(execution_devices)

    latencies_ms: list[float] = []
    for _ in range(args.iters):
        _synchronize(execution_devices)
        start = time.perf_counter()
        outputs = session.run(feed)
        _synchronize(execution_devices)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    # Keep graph outputs and caches alive through the final synchronization.
    _ = outputs, caches
    summary = _latency_summary(latencies_ms, current_seq_len)
    print(
        "steady decode: "
        f"mean={summary['mean_ms']:.3f} ms/token, "
        f"median={summary['median_ms']:.3f}, p90={summary['p90_ms']:.3f}, "
        f"p95={summary['p95_ms']:.3f}, p99={summary['p99_ms']:.3f}, "
        f"min={summary['min_ms']:.3f}, max={summary['max_ms']:.3f}, "
        f"throughput={summary['tokens_per_second']:.3f} tokens/s"
    )

    if args.trace is not None:
        trace_path = args.trace.expanduser().resolve()
        _write_trace(trace_path, session, feed, execution_devices)
        print(f"trace: {trace_path}")

    result = {
        "model": str(model_path),
        "past_seq_len": args.past_seq_len,
        "max_seq_len": args.max_seq_len,
        "current_seq_len": current_seq_len,
        "warmup_iterations": args.warmup,
        "session_init_seconds": init_seconds,
        "capture_call_seconds": capture_seconds,
        "execution_devices": [str(device) for device in execution_devices],
        "cuda_graph": graph_status,
        "page_attention": [asdict(spec) for spec in page_specs],
        "total_sefp_cache_bytes": sum(spec.cache_bytes for spec in page_specs),
        "latency": summary,
        "samples_ms": latencies_ms,
    }
    if args.json is not None:
        json_path = args.json.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"json: {json_path}")


if __name__ == "__main__":
    main()
