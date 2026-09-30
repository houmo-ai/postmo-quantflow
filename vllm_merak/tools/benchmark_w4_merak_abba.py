#!/usr/bin/env python3
"""Fail-closed packed-W4 A/B benchmark through the real vLLM Merak runtime.

The controller starts four fresh engines in ABBA order on the same visible
GPU set.  Every worker runs prompt-only (one generated token) and multi-step
decode requests, records vLLM engine timestamps, exact token IDs, model
fingerprints, and per-device CUDA memory.  The final report is accepted only
when packed W4 is bit-exact, faster in both TTFT and decode, and uses less
peak allocated memory than the existing expanded-W8 runtime representation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ABBA = ("packed", "expanded", "expanded", "packed")
_DERIVED_PAGE_ATTENTION_GRAPH = re.compile(
    r"_page_attention_[0-9a-f]{16,64}\.onnx$"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Optional torch-profiler output directory (diagnostic runs).",
    )
    parser.add_argument(
        "--min-prefill-speedup",
        type=float,
        default=1.02,
        help="Minimum speedup required in each ABBA process pair.",
    )
    parser.add_argument(
        "--min-decode-speedup",
        type=float,
        default=1.02,
        help="Minimum speedup required in each ABBA process pair.",
    )
    parser.add_argument(
        "--kv-cache-format",
        default="hmbf16",
        help="Merak KV format; SEFP PageAttention exports require hmbf16.",
    )
    parser.add_argument(
        "--allow-regression",
        action="store_true",
        help="Write diagnostic evidence without failing on speed/memory.",
    )
    parser.add_argument("--_worker-mode", choices=("packed", "expanded"))
    parser.add_argument("--_worker-output", type=Path)
    parser.add_argument("--_artifact-manifest", type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_onnx_graphs(graph: Any):
    import onnx

    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _iter_onnx_graphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from _iter_onnx_graphs(subgraph)


def _external_data_paths(graph_path: Path) -> set[Path]:
    import onnx

    model = onnx.load(str(graph_path), load_external_data=False)
    paths: set[Path] = set()
    for graph in _iter_onnx_graphs(model.graph):
        tensors = list(graph.initializer)
        for sparse in graph.sparse_initializer:
            tensors.extend((sparse.values, sparse.indices))
        for tensor in tensors:
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                continue
            external = {item.key: item.value for item in tensor.external_data}
            location = external.get("location")
            if not location:
                raise RuntimeError(
                    f"{graph_path}: external initializer {tensor.name!r} "
                    "has no location"
                )
            path = (graph_path.parent / location).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.add(path)
    return paths


def _manifest_label(path: Path, model_dir: Path) -> str:
    try:
        return str(path.relative_to(model_dir))
    except ValueError:
        return f"external::{path}"


def _canonical_graph_paths(model_dir: Path) -> list[Path]:
    graph_paths = sorted(model_dir.rglob("*.onnx"))
    canonical = [
        path
        for path in graph_paths
        if not _DERIVED_PAGE_ATTENTION_GRAPH.search(path.name)
    ]
    return canonical or graph_paths


def _artifact_manifest(model_dir: Path) -> dict[str, Any]:
    required = (
        model_dir / "golden_meta_info.json",
        model_dir / "merak_config.json",
        model_dir / "hf_config" / "config.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    graph_paths = _canonical_graph_paths(model_dir)
    if not graph_paths:
        raise RuntimeError(f"no HMONNX graphs found under {model_dir}")
    external_paths = set()
    for graph_path in graph_paths:
        external_paths.update(_external_data_paths(graph_path))
    files = {
        path.resolve()
        for path in (*required, *graph_paths, *sorted(external_paths))
    }
    rows = []
    for path in sorted(files, key=str):
        before = path.stat()
        sha256 = _sha256(path)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError(f"artifact changed while hashing: {path}")
        rows.append(
            {
                "path": _manifest_label(path, model_dir),
                "absolute_path": str(path),
                "size": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "sha256": sha256,
                "kind": (
                    "onnx_external_data"
                    if path in external_paths
                    else "onnx_graph"
                    if path.suffix == ".onnx"
                    else "metadata"
                ),
            }
        )
    fingerprint_rows = [
        {
            "path": row["path"],
            "size": row["size"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]
    encoded = json.dumps(
        fingerprint_rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "files": rows,
        "external_data_files": len(external_paths),
        "total_bytes": sum(row["size"] for row in rows),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _verify_artifact_manifest(manifest: dict[str, Any]) -> None:
    for row in manifest["files"]:
        path = Path(row["absolute_path"])
        stat = path.stat()
        if stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise RuntimeError(f"artifact changed after manifest creation: {path}")


def _w4_artifact_facts(model_dir: Path) -> dict[str, Any]:
    import onnx

    node_count = 0
    initializer_refs = set()
    graph_counts = {}
    for graph_path in _canonical_graph_paths(model_dir):
        model = onnx.load(str(graph_path), load_external_data=False)
        graph_node_count = 0
        for graph in _iter_onnx_graphs(model.graph):
            for node in graph.node:
                attributes = {
                    attribute.name: onnx.helper.get_attribute_value(attribute)
                    for attribute in node.attribute
                }
                if attributes.get("hmfp_weight_man_bit") != 4:
                    continue
                graph_node_count += 1
                node_count += 1
                if node.op_type == "Linear" and len(node.input) > 1:
                    initializer_refs.add(
                        (str(graph_path.relative_to(model_dir)), node.input[1])
                    )
                elif node.op_type == "MoeBlock" and len(node.input) in (14, 15):
                    weight_start = 3 if len(node.input) == 15 else 2
                    for offset in (0, 3, 6):
                        initializer_refs.add(
                            (
                                str(graph_path.relative_to(model_dir)),
                                node.input[weight_start + offset],
                            )
                        )
        if graph_node_count:
            graph_counts[str(graph_path.relative_to(model_dir))] = (
                graph_node_count
            )
    if node_count == 0 or not initializer_refs:
        raise RuntimeError(f"{model_dir}: no W4 HMONNX weights found")
    return {
        "w4_nodes": node_count,
        "w4_initializer_references": len(initializer_refs),
        "graphs": graph_counts,
    }


def _git_state(path: Path) -> dict[str, Any] | None:
    root = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if root.returncode:
        return None
    repo = Path(root.stdout.strip())
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            text=True,
        ).strip()
    )
    return {"root": str(repo), "revision": revision, "dirty": dirty}


def _runtime_metadata(torch: Any, model_dir: Path) -> dict[str, Any]:
    packages = {}
    for name in (
        "torch",
        "transformers",
        "vllm",
        "vllm-merak",
        "xhquant-llm",
        "xhquanttool",
        "xhmodel_zoo",
        "triton",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            "-i",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    repo_candidates = {
        "vllm_merak": Path(__file__).resolve().parents[1],
        "model_artifact": model_dir,
    }
    source_modules = {
        "xhquant": "xhquant.xhonnxruntime.llm_hmonnx_loader",
        "xhmodel_merak": (
            "xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model"
        ),
    }
    for name, module_name in source_modules.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file:
            repo_candidates[name] = Path(module_file).resolve()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": (
            driver.stdout.strip().splitlines()[0]
            if driver.returncode == 0 and driver.stdout.strip()
            else None
        ),
        "repositories": {
            name: _git_state(path)
            for name, path in repo_candidates.items()
        },
        "runtime_modes": {
            "vllm_enforce_eager": True,
            "hmonnx_cuda_graph_requested": (
                os.environ.get("VLLM_MERAK_CUDA_GRAPH") == "1"
            ),
            "hmonnx_v2_requested": (
                os.environ.get("ENABLE_HMINFERENCE_V2") == "1"
            ),
            "forbid_cuda_w4_unpack": (
                os.environ.get("XHQUANT_FORBID_CUDA_W4_UNPACK") == "1"
            ),
        },
    }


def _selected_logprobs(completion: Any, token_ids: list[int]) -> list[list[dict[str, Any]]]:
    positions = getattr(completion, "logprobs", None)
    if positions is None or len(positions) != len(token_ids):
        raise RuntimeError("vLLM did not return per-token logprobs")
    result = []
    for position in positions:
        entries = []
        for token_id, value in sorted(position.items()):
            entries.append(
                {
                    "token_id": int(token_id),
                    "logprob": float(value.logprob),
                    "rank": int(value.rank) if value.rank is not None else None,
                    "decoded_token": value.decoded_token,
                }
            )
        result.append(entries)
    return result


def _metric_row(output: Any, elapsed_s: float, max_tokens: int) -> dict[str, Any]:
    completion = output.outputs[0]
    token_ids = list(completion.token_ids)
    if len(token_ids) != max_tokens:
        raise RuntimeError(
            f"expected {max_tokens} generated tokens, got {len(token_ids)}"
        )
    metrics = output.metrics
    first_latency = None
    decode_span = None
    decode_per_token = None
    if metrics is not None:
        first = float(metrics.first_token_ts)
        last = float(metrics.last_token_ts)
        first_latency = float(metrics.first_token_latency)
        if first <= 0 or last < first or first_latency <= 0:
            raise RuntimeError(f"invalid vLLM engine timestamps: {metrics!r}")
        decode_span = last - first
        decode_per_token = decode_span / (max_tokens - 1) if max_tokens > 1 else None
    return {
        "elapsed_s": elapsed_s,
        "max_tokens": max_tokens,
        "token_ids": token_ids,
        "top_logprobs": _selected_logprobs(completion, token_ids),
        "finish_reason": completion.finish_reason,
        "first_token_latency_s": first_latency,
        "decode_span_s": decode_span,
        "decode_per_token_s": decode_per_token,
    }


def _memory_rows(torch: Any) -> list[dict[str, Any]]:
    return [
        {
            "logical_device": index,
            "name": torch.cuda.get_device_name(index),
            "uuid": str(torch.cuda.get_device_properties(index).uuid),
            "allocated_mib": torch.cuda.memory_allocated(index) / 2**20,
            "reserved_mib": torch.cuda.memory_reserved(index) / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(index) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(index) / 2**20,
        }
        for index in range(torch.cuda.device_count())
    ]


def _runtime_initializer_summary(llm: Any) -> dict[str, int]:
    try:
        executor = llm.llm_engine.model_executor
        wrapper = executor.driver_worker
        worker = getattr(wrapper, "worker", wrapper)
        root = worker.model_runner.model
    except AttributeError as exc:
        raise RuntimeError(
            "cannot locate the in-process vLLM driver model for W4 audit"
        ) from exc

    queue = [root]
    visited = set()
    child_names = (
        "inference_model",
        "model",
        "language_model",
        "_llm_model",
        "llm_model",
    )
    while queue and len(visited) < 32:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        summary = getattr(candidate, "runtime_initializer_summary", None)
        if isinstance(summary, dict):
            return {str(key): int(value) for key, value in summary.items()}
        for name in child_names:
            child = getattr(candidate, name, None)
            if child is not None:
                queue.append(child)
    raise RuntimeError(
        "loaded Merak model does not expose runtime_initializer_summary"
    )


def _validate_runtime_w4_summary(
    summary: dict[str, int],
    *,
    expected_packed: bool,
) -> None:
    planned = summary.get("runtime_w4_planned_references", 0)
    packed = summary.get("runtime_w4_packed_references", 0)
    unique = summary.get("runtime_w4_packed_initializers", 0)
    shared = summary.get("runtime_w4_shared_references", 0)
    saved = summary.get("runtime_w4_packed_saved_nbytes", 0)
    if expected_packed:
        if planned <= 0 or packed != planned:
            raise RuntimeError(
                "not every planned W4 initializer reference was packed: "
                f"planned={planned}, packed={packed}"
            )
        if unique <= 0 or shared <= 0 or saved <= 0:
            raise RuntimeError(
                "packed W4 did not preserve shared half-size storage: "
                f"unique={unique}, shared={shared}, saved_nbytes={saved}"
            )
    elif any((planned, packed, unique, shared, saved)):
        raise RuntimeError(
            "expanded baseline unexpectedly contains packed W4 state: "
            f"{summary}"
        )


def _run_worker(args: argparse.Namespace) -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import torch
    from vllm import LLM, SamplingParams

    model_dir = args.model_dir.expanduser().resolve()
    if args.decode_tokens < 2:
        raise ValueError("--decode-tokens must be at least 2")
    if min(args.prompt_tokens, args.warmup, args.repeat) <= 0:
        raise ValueError("prompt-tokens, warmup, and repeat must be positive")
    if args._worker_output is None:
        raise ValueError("worker output path is required")
    if args._artifact_manifest is None:
        raise ValueError("controller artifact manifest path is required")
    artifact_manifest = json.loads(
        args._artifact_manifest.read_text(encoding="utf-8")
    )
    _verify_artifact_manifest(artifact_manifest)
    expected_packed = args._worker_mode == "packed"
    actual_packed = os.environ.get("XHQUANT_PACKED_W4", "1") != "0"
    if actual_packed != expected_packed:
        raise RuntimeError(
            "worker mode and XHQUANT_PACKED_W4 disagree: "
            f"{args._worker_mode!r} vs {actual_packed}"
        )

    torch.cuda.init()
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(torch.device(f"cuda:{device}"))
    load_started = time.perf_counter()
    llm_kwargs = {}
    if args.profile_dir is not None:
        profile_dir = args.profile_dir.expanduser().resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)
        llm_kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir),
            "torch_profiler_with_stack": False,
            "torch_profiler_use_gzip": False,
            "torch_profiler_record_shapes": True,
        }
    llm = LLM(
        model=str(model_dir),
        trust_remote_code=True,
        config_format="merak_llm",
        load_format="merak_llm",
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        disable_chunked_mm_input=True,
        **llm_kwargs,
    )
    torch.cuda.synchronize()
    load_elapsed_s = time.perf_counter() - load_started
    memory_after_load = _memory_rows(torch)
    runtime_initializer_summary = _runtime_initializer_summary(llm)
    _validate_runtime_w4_summary(
        runtime_initializer_summary,
        expected_packed=expected_packed,
    )

    tokenizer = llm.get_tokenizer()
    unit = tokenizer.encode("测试", add_special_tokens=False) or [1]
    prompt_ids = (unit * ((args.prompt_tokens + len(unit) - 1) // len(unit)))[
        : args.prompt_tokens
    ]
    if len(prompt_ids) != args.prompt_tokens:
        raise RuntimeError("failed to construct the exact prompt token count")
    params = {
        count: SamplingParams(
            temperature=0,
            max_tokens=count,
            seed=args.seed,
            ignore_eos=True,
            logprobs=5,
        )
        for count in (1, args.decode_tokens)
    }

    def generate(count: int) -> dict[str, Any]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.profiler.record_function(
            f"merak_request_max_tokens_{count}"
        ):
            output = llm.generate(
                {"prompt_token_ids": prompt_ids},
                params[count],
                use_tqdm=False,
            )[0]
        torch.cuda.synchronize()
        return _metric_row(output, time.perf_counter() - started, count)

    for _ in range(args.warmup):
        generate(1)
        generate(args.decode_tokens)
    if args.profile_dir is not None:
        llm.start_profile(f"w4-{args._worker_mode}")
    rows: list[dict[str, Any]] = []
    for iteration in range(args.repeat):
        # Alternate the within-process order to cancel monotonic temperature
        # and allocator drift without changing the logical requests.
        counts = (
            (1, args.decode_tokens) if iteration % 2 == 0 else (args.decode_tokens, 1)
        )
        for count in counts:
            row = generate(count)
            row["iteration"] = iteration
            rows.append(row)
    if args.profile_dir is not None:
        llm.stop_profile()
    for iteration in range(args.repeat):
        pair = [row for row in rows if row["iteration"] == iteration]
        one = next(row for row in pair if row["max_tokens"] == 1)
        multi = next(row for row in pair if row["max_tokens"] == args.decode_tokens)
        derived = (multi["elapsed_s"] - one["elapsed_s"]) / (args.decode_tokens - 1)
        if derived <= 0:
            raise RuntimeError(
                "paired wall-clock decode estimate is non-positive: "
                f"iteration={iteration}, one={one['elapsed_s']}, "
                f"multi={multi['elapsed_s']}"
            )
        multi["derived_decode_per_token_s"] = derived

    payload = {
        "mode": args._worker_mode,
        "packed_w4": actual_packed,
        "model_dir": str(model_dir),
        "artifact_manifest": artifact_manifest,
        "runtime_metadata": _runtime_metadata(torch, model_dir),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": [
            {
                "logical_device": index,
                "name": torch.cuda.get_device_name(index),
                "uuid": str(torch.cuda.get_device_properties(index).uuid),
            }
            for index in range(torch.cuda.device_count())
        ],
        "prompt_tokens": len(prompt_ids),
        "decode_tokens": args.decode_tokens,
        "load_elapsed_s": load_elapsed_s,
        "runtime_initializer_summary": runtime_initializer_summary,
        "memory_after_load": memory_after_load,
        "memory_final": _memory_rows(torch),
        "rows": rows,
        "profile_dir": (
            str(args.profile_dir.expanduser().resolve())
            if args.profile_dir is not None
            else None
        ),
    }
    args._worker_output.parent.mkdir(parents=True, exist_ok=True)
    args._worker_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _median(run_rows: list[dict[str, Any]], key: str, count: int) -> float:
    return statistics.median(_metric_values(run_rows, key, count))


def _metric_values(
    run_rows: list[dict[str, Any]],
    key: str,
    count: int,
) -> list[float]:
    values = [
        float(row[key])
        for run in run_rows
        for row in run["rows"]
        if row["max_tokens"] == count and row[key] is not None
    ]
    if not values:
        raise RuntimeError(f"no values for metric {key!r}, max_tokens={count}")
    return values


def _describe(values: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "median": statistics.median(values),
        "stdev": stdev,
        "cv": stdev / mean if mean else 0.0,
    }


def _run_metric_median(run: dict[str, Any], key: str, count: int) -> float:
    return statistics.median(
        float(row[key])
        for row in run["rows"]
        if row["max_tokens"] == count and row[key] is not None
    )


def _abba_pair_speedups(
    runs: list[dict[str, Any]],
    key: str,
    count: int,
) -> list[float]:
    if tuple(run["mode"] for run in runs) != ABBA:
        raise RuntimeError("ABBA run order was not preserved")
    return [
        _run_metric_median(runs[1], key, count)
        / _run_metric_median(runs[0], key, count),
        _run_metric_median(runs[2], key, count)
        / _run_metric_median(runs[3], key, count),
    ]


def _assert_logprobs_equal(
    actual: list[list[dict[str, Any]]],
    expected: list[list[dict[str, Any]]],
    *,
    atol: float = 1e-6,
) -> None:
    if len(actual) != len(expected):
        raise RuntimeError("top-logprob position count mismatch")
    for position, (actual_entries, expected_entries) in enumerate(
        zip(actual, expected, strict=True)
    ):
        actual_facts = [
            (entry["token_id"], entry["rank"], entry["decoded_token"])
            for entry in actual_entries
        ]
        expected_facts = [
            (entry["token_id"], entry["rank"], entry["decoded_token"])
            for entry in expected_entries
        ]
        if actual_facts != expected_facts:
            raise RuntimeError(
                f"top-logprob token/rank mismatch at position {position}: "
                f"{actual_facts} != {expected_facts}"
            )
        for actual_entry, expected_entry in zip(
            actual_entries,
            expected_entries,
            strict=True,
        ):
            if abs(actual_entry["logprob"] - expected_entry["logprob"]) > atol:
                raise RuntimeError(
                    f"top-logprob value mismatch at position {position}, "
                    f"token={actual_entry['token_id']}: "
                    f"{actual_entry['logprob']} != {expected_entry['logprob']}"
                )


def _sum_peak_mib(run: dict[str, Any]) -> float:
    return sum(float(row["peak_allocated_mib"]) for row in run["memory_final"])


def _run_controller(args: argparse.Namespace) -> None:
    from check_hmonnx_attention_abi import check_model

    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.min_prefill_speedup <= 1 or args.min_decode_speedup <= 1:
        raise ValueError("minimum speedup thresholds must be greater than 1")
    attention_preflight = check_model(model_dir)
    (output_dir / "attention-abi-preflight.json").write_text(
        json.dumps(attention_preflight, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = _artifact_manifest(model_dir)
    w4_artifact_facts = _w4_artifact_facts(model_dir)
    manifest_path = output_dir / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    runs: list[dict[str, Any]] = []
    for index, mode in enumerate(ABBA):
        output = output_dir / f"{index:02d}-{mode}.json"
        log = output_dir / f"{index:02d}-{mode}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model-dir",
            str(model_dir),
            "--output-dir",
            str(output_dir),
            "--cuda-visible-devices",
            args.cuda_visible_devices,
            "--prompt-tokens",
            str(args.prompt_tokens),
            "--decode-tokens",
            str(args.decode_tokens),
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--seed",
            str(args.seed),
            "--kv-cache-format",
            args.kv_cache_format,
            "--_worker-mode",
            mode,
            "--_worker-output",
            str(output),
            "--_artifact-manifest",
            str(manifest_path),
        ]
        if args.profile_dir is not None:
            command.extend(
                [
                    "--profile-dir",
                    str(
                        args.profile_dir.expanduser().resolve()
                        / f"{index:02d}-{mode}"
                    ),
                ]
            )
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
                "VLLM_PLUGINS": "merak",
                "ENABLE_HMINFERENCE_V2": "1",
                "VLLM_MERAK_USE_V2": "1",
                "VLLM_MERAK_AUTO_OFFLOAD": "1",
                "VLLM_MERAK_CUDA_GRAPH": "1",
                "VLLM_MERAK_KV_CACHE_FORMAT": args.kv_cache_format,
                "VLLM_CPU_KVCACHE_SPACE": env.get("VLLM_CPU_KVCACHE_SPACE", "8"),
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "XHQUANT_PACKED_W4": "1" if mode == "packed" else "0",
                "XHQUANT_FORBID_CUDA_W4_UNPACK": "1",
            }
        )
        with log.open("w", encoding="utf-8") as sink:
            completed = subprocess.run(
                command,
                env=env,
                stdout=sink,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"{mode} worker {index} failed with code "
                f"{completed.returncode}; see {log}"
            )
        run = json.loads(output.read_text(encoding="utf-8"))
        if (
            run["mode"] != mode
            or run["artifact_manifest"]["sha256"] != manifest["sha256"]
        ):
            raise RuntimeError(f"worker {index} changed mode or artifact")
        _verify_artifact_manifest(manifest)
        runs.append(run)

    device_sets = {
        tuple((row["name"], row["uuid"]) for row in run["devices"]) for run in runs
    }
    if len(device_sets) != 1:
        raise RuntimeError(f"ABBA runs did not use the same GPUs: {device_sets}")
    reference_tokens: dict[int, list[int]] = {}
    reference_logprobs: dict[int, list[list[dict[str, Any]]]] = {}
    for run in runs:
        for row in run["rows"]:
            count = int(row["max_tokens"])
            ids = list(row["token_ids"])
            expected = reference_tokens.setdefault(count, ids)
            if ids != expected:
                raise RuntimeError(
                    f"token mismatch for max_tokens={count}: {ids} != {expected}"
                )
            expected_logprobs = reference_logprobs.setdefault(
                count,
                row["top_logprobs"],
            )
            _assert_logprobs_equal(row["top_logprobs"], expected_logprobs)
    if reference_tokens[args.decode_tokens][:1] != reference_tokens[1]:
        raise RuntimeError(
            "one-token and multi-token requests disagree on the first token"
        )

    packed = [run for run in runs if run["mode"] == "packed"]
    expanded = [run for run in runs if run["mode"] == "expanded"]
    engine_timestamps = all(
        row["first_token_latency_s"] is not None for run in runs for row in run["rows"]
    )
    prefill_key = "first_token_latency_s" if engine_timestamps else "elapsed_s"
    decode_key = (
        "decode_per_token_s" if engine_timestamps else "derived_decode_per_token_s"
    )
    packed_ttft = _median(packed, prefill_key, 1)
    expanded_ttft = _median(expanded, prefill_key, 1)
    packed_decode = _median(packed, decode_key, args.decode_tokens)
    expanded_decode = _median(expanded, decode_key, args.decode_tokens)
    prefill_pair_speedups = _abba_pair_speedups(runs, prefill_key, 1)
    decode_pair_speedups = _abba_pair_speedups(
        runs,
        decode_key,
        args.decode_tokens,
    )
    packed_peak = statistics.median(_sum_peak_mib(run) for run in packed)
    expanded_peak = statistics.median(_sum_peak_mib(run) for run in expanded)
    memory_pair_savings = [
        _sum_peak_mib(runs[1]) - _sum_peak_mib(runs[0]),
        _sum_peak_mib(runs[2]) - _sum_peak_mib(runs[3]),
    ]
    runtime_metadata_fingerprints = {
        json.dumps(run["runtime_metadata"], sort_keys=True) for run in runs
    }
    if len(runtime_metadata_fingerprints) != 1:
        raise RuntimeError("ABBA workers used different runtime metadata")
    summary = {
        "accepted": (
            min(prefill_pair_speedups) >= args.min_prefill_speedup
            and min(decode_pair_speedups) >= args.min_decode_speedup
            and min(memory_pair_savings) > 0
        ),
        "sequence": ABBA,
        "attention_abi_preflight": attention_preflight,
        "w4_artifact_facts": w4_artifact_facts,
        "artifact_manifest_sha256": manifest["sha256"],
        "artifact_external_data_files": manifest["external_data_files"],
        "artifact_total_bytes": manifest["total_bytes"],
        "devices": list(next(iter(device_sets))),
        "exact_tokens": reference_tokens,
        "exact_top_logprobs": reference_logprobs,
        "runtime_metadata": runs[0]["runtime_metadata"],
        "timing_source": (
            "vllm_engine_timestamps"
            if engine_timestamps
            else "paired_synchronized_wall_clock"
        ),
        "packed_ttft_median_s": packed_ttft,
        "expanded_ttft_median_s": expanded_ttft,
        "prefill_speedup": expanded_ttft / packed_ttft,
        "prefill_pair_speedups": prefill_pair_speedups,
        "min_prefill_speedup_required": args.min_prefill_speedup,
        "prefill_samples": {
            "packed": _describe(_metric_values(packed, prefill_key, 1)),
            "expanded": _describe(_metric_values(expanded, prefill_key, 1)),
        },
        "packed_decode_per_token_median_s": packed_decode,
        "expanded_decode_per_token_median_s": expanded_decode,
        "decode_speedup": expanded_decode / packed_decode,
        "decode_pair_speedups": decode_pair_speedups,
        "min_decode_speedup_required": args.min_decode_speedup,
        "decode_samples": {
            "packed": _describe(
                _metric_values(packed, decode_key, args.decode_tokens)
            ),
            "expanded": _describe(
                _metric_values(expanded, decode_key, args.decode_tokens)
            ),
        },
        "packed_peak_allocated_mib": packed_peak,
        "expanded_peak_allocated_mib": expanded_peak,
        "peak_memory_saved_mib": expanded_peak - packed_peak,
        "memory_pair_savings_mib": memory_pair_savings,
        "runs": [
            {
                "mode": run["mode"],
                "load_elapsed_s": run["load_elapsed_s"],
                "peak_allocated_mib": _sum_peak_mib(run),
                "runtime_initializer_summary": run[
                    "runtime_initializer_summary"
                ],
            }
            for run in runs
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["accepted"] and not args.allow_regression:
        raise SystemExit(2)


def main() -> None:
    args = _parser().parse_args()
    if args._worker_mode:
        _run_worker(args)
    else:
        _run_controller(args)


if __name__ == "__main__":
    main()
