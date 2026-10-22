"""Run the pinned official non-audio MMEB-v3 suite with vLLM or vLLM Merak.

Each GPU evaluates a small group of official datasets with one persistent
engine. Group-level isolation keeps retries and resume deterministic while
avoiding unsupported multi-rank sharing of one Merak engine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


OFFICIAL_COMMIT = "5f539a2f5196b08bbcf29ac71fc90a95ccad7f11"
V3_DATA_REVISION = "4a5560b2b64384204b6fea8a82ea986eba51f5aa"
V2_DATA_REVISION = "e7bbfeb69a70dfe32ff36da3d6d8dbe31fc36af1"
DEFAULT_MODALITIES = ("image", "video", "visdoc", "text", "tool", "memory", "gui")
ISOLATED_DATASETS = {"text/LongEmbed/narrativeqa"}


@dataclass(frozen=True)
class Job:
    modality: str
    group_name: str
    dataset_names: tuple[str, ...]
    config: Path
    output: Path
    log: Path
    estimated_seconds: float
    shard_index: int | None = None
    shard_count: int = 1
    final_output: Path | None = None


@dataclass(frozen=True)
class Task:
    modality: str
    dataset_name: str
    config: Path
    output: Path
    log: Path
    task_config: dict[str, object]
    estimated_seconds: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def _score_path(output: Path, dataset_name: str) -> Path:
    return output / f"{dataset_name}_score.json"


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def _write_pickle_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _shard_bounds(length: int, index: int, count: int) -> tuple[int, int]:
    if length < 0 or count <= 0 or not 0 <= index < count:
        raise ValueError(f"Invalid shard range inputs: length={length}, index={index}, count={count}")
    return length * index // count, length * (index + 1) // count


def _read_jsonl(path: Path) -> list[object]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _historical_timings(run_root: Path) -> dict[tuple[str, str], float]:
    timings: dict[tuple[str, str], float] = {}
    # Timings remain useful for scheduling even when an older run's embeddings
    # are not reusable (for example, after a prompt-tokenization fix).
    for path in run_root.glob("wemm_*/results/**/timing.csv"):
        try:
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("status") != "success":
                        continue
                    key = (str(row["modality"]), str(row["dataset_name"]))
                    timings[key] = max(timings.get(key, 0.0), float(row["duration_seconds"]))
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return timings


def _estimate_seconds(
    args: argparse.Namespace,
    modality: str,
    dataset_name: str,
    task: dict[str, object],
    historical: dict[tuple[str, str], float],
) -> float:
    if (modality, dataset_name) in historical:
        return historical[(modality, dataset_name)]
    input_bytes = 0
    for key in ("data_path", "query_file", "candidate_file", "qrels_file"):
        value = task.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = args.data / path
        try:
            if path.is_file():
                input_bytes += path.stat().st_size
        except OSError:
            pass
    modality_floor = {
        "image": 120.0,
        "video": 900.0,
        "visdoc": 300.0,
        "text": 900.0,
        "tool": 240.0,
        "memory": 300.0,
        "gui": 600.0,
    }[modality]
    parser_multiplier = {
        "mcmr": 16.0,
        "video_retrieval": 4.0,
        "long_context": 8.0,
    }.get(str(task.get("dataset_parser")), 1.0)
    return modality_floor * parser_multiplier + input_bytes / (4 * 1024 * 1024)


def _protocol_matches(
    args: argparse.Namespace,
    protocol: dict[str, object],
    config: Path,
) -> bool:
    current_adapter_sha256 = _sha256(args.repo / "examples_merak/llm/wemm_embedding/mmeb/mmeb_v3_vllm_merak.py")
    current_eval_adapter_sha256 = _sha256(args.repo / "examples_merak/llm/wemm_embedding/mmeb/mmeb_v3_eval.py")
    protocol_adapter_sha256 = protocol.get("wemm_vllm_adapter_sha256")
    vllm = protocol.get("vllm") or {}
    expected_num_seqs = int(args.vllm_max_num_seqs)
    expected_model = args.model if args.backend == "vllm" else args.source_model
    expected_hf_config = expected_model / "config.json"
    upstream_args = protocol.get("upstream_args") or []

    def upstream_value(flag: str) -> object:
        try:
            return upstream_args[upstream_args.index(flag) + 1]
        except (AttributeError, IndexError, ValueError):
            return None

    checks = [
        protocol.get("backend") == args.backend,
        protocol.get("official_repo_commit") == OFFICIAL_COMMIT,
        protocol.get("mmeb_v3_data_revision") == V3_DATA_REVISION,
        protocol.get("mmeb_v2_media_revision") == V2_DATA_REVISION,
        protocol_adapter_sha256 == current_adapter_sha256,
        protocol.get("mmeb_v3_eval_adapter_sha256") == current_eval_adapter_sha256,
        protocol.get("dataset_config_sha256") == _sha256(config),
        protocol.get("vllm_version") == args.vllm_version,
        protocol.get("transformers_version") == args.transformers_version,
        protocol.get("hf_config_sha256") == _sha256(expected_hf_config),
        Path(str(protocol.get("model_name", ""))).resolve() == expected_model.resolve(),
        upstream_value("--model_backbone") == "wemm_embedding",
        upstream_value("--pooling") == "last",
        upstream_value("--normalize") == "true",
        vllm.get("max_model_len") == 262144,
        vllm.get("max_images") == 16,
        vllm.get("max_num_seqs") == expected_num_seqs,
        (protocol.get("image_preprocessing") or {}).get("mode") == "dynamic",
        (protocol.get("image_preprocessing") or {}).get("min_pixels") == 65536,
        (protocol.get("image_preprocessing") or {}).get("max_pixels") == 1572864,
        (protocol.get("image_preprocessing") or {}).get("preserve_aspect_ratio") is True,
        (protocol.get("video_preprocessing") or {}).get("num_frames") == 64,
        (protocol.get("video_preprocessing") or {}).get("frame_size") == 224,
        (protocol.get("video_preprocessing") or {}).get("min_pixels") == 4 * 32 * 32,
        (protocol.get("video_preprocessing") or {}).get("max_pixels") == 256 * 32 * 32,
        (protocol.get("video_preprocessing") or {}).get("total_pixels") == 8192 * 32 * 32,
        (protocol.get("video_preprocessing") or {}).get("sample_fps") == 1,
    ]
    if args.backend == "vllm":
        checks.append(protocol.get("vllm_wemm_model_sha256") == args.vllm_wemm_model_sha256)
    if args.backend == "vllm_merak":
        checks.extend(
            [
                protocol.get("vllm_version") == args.vllm_version,
                protocol.get("vllm_merak_source_sha256") == args.vllm_merak_source_sha256,
                protocol.get("hmonnx_meta_sha256") == _sha256(args.model / "golden_meta_info.json"),
            ]
        )
    return all(checks)


def _task_result_valid(
    args: argparse.Namespace,
    task: Task,
) -> bool:
    score_path = _score_path(task.output, task.dataset_name)
    protocol_path = task.output / "protocol.json"
    if not score_path.is_file() or not protocol_path.is_file():
        return False
    try:
        protocol = json.loads(protocol_path.read_text())
        score = json.loads(score_path.read_text())
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    metric = "ndcg_linear@5" if task.modality in {"text", "visdoc"} else "hit@1"
    return metric in score and _protocol_matches(args, protocol, task.config)


def _protocol_file_matches(args: argparse.Namespace, path: Path, config: Path) -> bool:
    try:
        return _protocol_matches(args, json.loads(path.read_text()), config)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _balanced_groups(tasks: list[Task], group_size: int) -> list[list[Task]]:
    if not tasks:
        return []
    isolated = [[task] for task in tasks if task.dataset_name in ISOLATED_DATASETS]
    tasks = [task for task in tasks if task.dataset_name not in ISOLATED_DATASETS]
    if not tasks:
        return isolated
    group_count = math.ceil(len(tasks) / group_size)
    groups: list[list[Task]] = [[] for _ in range(group_count)]
    weights = [0.0] * group_count
    for task in sorted(tasks, key=lambda item: (-item.estimated_seconds, item.dataset_name)):
        candidates = [index for index, group in enumerate(groups) if len(group) < group_size]
        index = min(candidates, key=lambda item: (weights[item], len(groups[item]), item))
        groups[index].append(task)
        weights[index] += task.estimated_seconds
    return isolated + groups


def _build_jobs(args: argparse.Namespace) -> tuple[list[Job], list[dict[str, object]], list[dict[str, object]]]:
    tasks_by_modality: dict[str, list[Task]] = {modality: [] for modality in args.modalities}
    dataset_selectors = set(getattr(args, "datasets", ()) or ())
    matched_dataset_selectors: set[str] = set()
    historical = _historical_timings(args.output.parent)
    previous_jobs: dict[tuple[str, str], dict[str, object]] = {}
    previous_manifest = args.output / "run_manifest.json"
    if previous_manifest.is_file():
        try:
            payload = json.loads(previous_manifest.read_text())
            previous_jobs = {(str(job["modality"]), str(job["dataset_name"])): job for job in payload.get("jobs", [])}
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            previous_jobs = {}
    config_root = args.output / "configs"
    for modality in args.modalities:
        official_config = args.official / "experiments" / "public" / "eval" / f"{modality}.yaml"
        tasks = yaml.safe_load(official_config.read_text())
        for dataset_name, task in tasks.items():
            selectors = (dataset_name, f"{modality}:{dataset_name}")
            if dataset_selectors:
                matched = next((item for item in selectors if item in dataset_selectors), None)
                if matched is None:
                    continue
                matched_dataset_selectors.add(matched)
            safe = _safe_name(dataset_name)
            config_path = config_root / modality / f"{safe}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump({dataset_name: task}, sort_keys=False, allow_unicode=True))
            output = args.output / "results" / modality / safe
            # The upstream evaluator embeds slash-containing task names into
            # filenames; create those parents before it opens the checkpoints.
            (output / dataset_name).parent.mkdir(parents=True, exist_ok=True)
            tasks_by_modality[modality].append(
                Task(
                    modality=modality,
                    dataset_name=dataset_name,
                    config=config_path,
                    output=output,
                    log=args.output / "logs" / modality / f"{safe}.log",
                    task_config=task,
                    estimated_seconds=_estimate_seconds(args, modality, dataset_name, task, historical),
                )
            )

    unmatched_datasets = dataset_selectors - matched_dataset_selectors
    if unmatched_datasets:
        raise ValueError(f"Unknown --dataset selectors: {sorted(unmatched_datasets)}")

    jobs: list[Job] = []
    manifest_jobs: list[dict[str, object]] = []
    reused: list[dict[str, object]] = []
    shard_selectors = set(getattr(args, "shard_datasets", ()) or ())
    matched_shard_selectors: set[str] = set()
    job_mode = getattr(args, "job_mode", "group")
    shards_per_dataset = int(getattr(args, "shards_per_dataset", 1))

    def shard_selector(task: Task) -> str | None:
        candidates = (task.dataset_name, f"{task.modality}:{task.dataset_name}")
        for candidate in candidates:
            if candidate in shard_selectors:
                return candidate
        return None

    for modality in args.modalities:
        pending: list[Task] = []
        for task in tasks_by_modality[modality]:
            selected_shard = shard_selector(task)
            if selected_shard is not None:
                matched_shard_selectors.add(selected_shard)
            reusable_task = task
            if not _task_result_valid(args, reusable_task):
                previous = previous_jobs.get((task.modality, task.dataset_name))
                if previous:
                    reusable_task = Task(
                        modality=task.modality,
                        dataset_name=task.dataset_name,
                        config=Path(str(previous["config"])),
                        output=Path(str(previous["output"])),
                        log=Path(str(previous["log"])),
                        task_config=task.task_config,
                        estimated_seconds=task.estimated_seconds,
                    )
            if _task_result_valid(args, reusable_task):
                manifest_jobs.append(
                    {
                        "modality": reusable_task.modality,
                        "dataset_name": reusable_task.dataset_name,
                        "config": str(reusable_task.config),
                        "output": str(reusable_task.output),
                        "log": str(reusable_task.log),
                    }
                )
                reused.append(
                    {
                        "status": "reused",
                        "gpu": None,
                        "job": {
                            "modality": reusable_task.modality,
                            "dataset_names": [reusable_task.dataset_name],
                            "output": str(reusable_task.output),
                        },
                        "scores": [str(_score_path(reusable_task.output, reusable_task.dataset_name))],
                    }
                )
            else:
                pending.append(task)

        regular_pending: list[Task] = []
        for task in pending:
            selector = shard_selector(task)
            if selector is None:
                regular_pending.append(task)
                continue
            safe = _safe_name(task.dataset_name)
            manifest_jobs.append(
                {
                    "modality": task.modality,
                    "dataset_name": task.dataset_name,
                    "config": str(task.config),
                    "output": str(task.output),
                    "log": str(task.log),
                    "shard_count": shards_per_dataset,
                }
            )
            for shard_index in range(shards_per_dataset):
                shard_output = args.output / "embedding_shards" / modality / safe / f"shard_{shard_index:04d}"
                jobs.append(
                    Job(
                        modality=modality,
                        group_name=f"dataset_{safe}_shard_{shard_index:04d}",
                        dataset_names=(task.dataset_name,),
                        config=task.config,
                        output=shard_output,
                        log=(args.output / "embedding_shard_logs" / modality / safe / f"shard_{shard_index:04d}.log"),
                        estimated_seconds=task.estimated_seconds / shards_per_dataset,
                        shard_index=shard_index,
                        shard_count=shards_per_dataset,
                        final_output=task.output,
                    )
                )

        group_size = 1 if job_mode == "dataset" else args.datasets_per_engine
        for group_index, group in enumerate(_balanced_groups(regular_pending, group_size)):
            identity = "\0".join(sorted(task.dataset_name for task in group)).encode()
            if job_mode == "dataset":
                task = group[0]
                group_name = f"dataset_{_safe_name(task.dataset_name)}"
                group_config = task.config
                group_output = task.output
                group_log = task.log
            else:
                group_name = f"group_{group_index:02d}_{hashlib.sha256(identity).hexdigest()[:8]}"
                group_config = args.output / "group_configs" / modality / f"{group_name}.yaml"
                group_config.parent.mkdir(parents=True, exist_ok=True)
                group_config.write_text(
                    yaml.safe_dump(
                        {task.dataset_name: task.task_config for task in group},
                        sort_keys=False,
                        allow_unicode=True,
                    )
                )
                group_output = args.output / "group_results" / modality / group_name
                group_log = args.output / "group_logs" / modality / f"{group_name}.log"
            for task in group:
                (_score_path(group_output, task.dataset_name)).parent.mkdir(parents=True, exist_ok=True)
                manifest_jobs.append(
                    {
                        "modality": task.modality,
                        "dataset_name": task.dataset_name,
                        "config": str(group_config),
                        "output": str(group_output),
                        "log": str(group_log),
                        "shard_count": 1,
                    }
                )
            jobs.append(
                Job(
                    modality=modality,
                    group_name=group_name,
                    dataset_names=tuple(task.dataset_name for task in group),
                    config=group_config,
                    output=group_output,
                    log=group_log,
                    estimated_seconds=sum(task.estimated_seconds for task in group),
                )
            )
    unmatched = shard_selectors - matched_shard_selectors
    if unmatched:
        raise ValueError(f"Unknown --shard-dataset selectors: {sorted(unmatched)}")
    jobs.sort(key=lambda item: (-item.estimated_seconds, item.modality, item.group_name))
    manifest_jobs.sort(key=lambda item: (args.modalities.index(str(item["modality"])), str(item["dataset_name"])))
    return jobs, manifest_jobs, reused


def _command(args: argparse.Namespace, job: Job, *, score_only: bool = False) -> list[str]:
    batch_size = 1 if job.modality in {"text", "gui"} else args.batch_size
    command = [
        str(args.python),
        "-u",
        str(args.repo / "examples_merak/llm/wemm_embedding/mmeb/mmeb_v3_eval.py"),
        "--backend",
        args.backend,
        "--protocol-file",
        str(job.output / "protocol.json"),
        "--vllm-max-model-len",
        "262144",
        "--vllm-max-num-seqs",
        str(args.vllm_max_num_seqs),
        "--vllm-max-images",
        "16",
        "--vllm-gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--image-min-pixels",
        "65536",
        "--image-max-pixels",
        "1572864",
        "--video-min-pixels",
        "4096",
        "--video-max-pixels",
        "262144",
        "--model_name",
        str(args.model if args.backend == "vllm" else args.source_model),
        "--model_backbone",
        "wemm_embedding",
        "--pooling",
        "last",
        "--normalize",
        "true",
        "--per_device_eval_batch_size",
        str(batch_size),
        "--dataloader_num_workers",
        str(args.dataloader_workers),
        "--video_max_frames",
        "64",
        "--video_frame_size",
        "224",
        "--dataset_config",
        str(job.config),
        "--encode_output_path",
        str(job.output),
        "--data_basedir",
        str(args.data),
    ]
    if args.backend == "vllm_merak":
        command[5:5] = [
            "--hmonnx-meta",
            str(args.model),
            "--vllm-merak-source-sha256",
            args.vllm_merak_source_sha256,
        ]
    if job.shard_index is not None:
        command[5:5] = [
            "--embedding-shard-only",
            "--embedding-shard-index",
            str(job.shard_index),
            "--embedding-shard-count",
            str(job.shard_count),
        ]
    if score_only:
        command[5:5] = ["--score-only"]
    if job.modality in {"text", "gui"}:
        command.extend(("--max_len", "262144"))
    return command


def _shard_result_valid(args: argparse.Namespace, job: Job) -> bool:
    if job.shard_index is None or len(job.dataset_names) != 1:
        return False
    protocol_path = job.output / "protocol.json"
    shard_root = job.output / "embedding_shard"
    manifest_path = shard_root / "manifest.json"
    required = [
        protocol_path,
        manifest_path,
        shard_root / "qry.npy",
        shard_root / "qry_info.jsonl",
        shard_root / "cand.npy",
        shard_root / "cand_info.jsonl",
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        protocol = json.loads(protocol_path.read_text())
        shard_protocol = protocol.get("embedding_shard") or {}
        manifest = json.loads(manifest_path.read_text())
        if not _protocol_matches(args, protocol, job.config):
            return False
        if shard_protocol != {
            "index": job.shard_index,
            "count": job.shard_count,
            "shard_only": True,
        }:
            return False
        if (
            manifest.get("schema_version") != 1
            or manifest.get("dataset_name") != job.dataset_names[0]
            or manifest.get("dataset_config_sha256") != _sha256(job.config)
            or manifest.get("shard_index") != job.shard_index
            or manifest.get("shard_count") != job.shard_count
        ):
            return False
        for side in ("query", "candidate"):
            details = manifest[side]
            expected = _shard_bounds(int(details["total"]), job.shard_index, job.shard_count)
            if (int(details["start"]), int(details["end"])) != expected:
                return False
            stem = "qry" if side == "query" else "cand"
            embeddings = np.load(shard_root / f"{stem}.npy", mmap_mode="r", allow_pickle=False)
            infos = _read_jsonl(shard_root / f"{stem}_info.jsonl")
            if len(embeddings) != expected[1] - expected[0] or len(infos) != len(embeddings):
                return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _run_job(args: argparse.Namespace, gpu: str, job: Job) -> dict[str, object]:
    if job.shard_index is not None and _shard_result_valid(args, job):
        return {
            "status": "reused",
            "kind": "embedding_shard",
            "gpu": gpu,
            "job": asdict(job),
            "artifacts": [str(job.output / "embedding_shard")],
        }
    score_paths = [_score_path(job.output, name) for name in job.dataset_names]
    protocol_path = job.output / "protocol.json"
    reusable = False
    if all(path.is_file() and path.stat().st_size > 0 for path in score_paths) and protocol_path.is_file():
        try:
            protocol = json.loads(protocol_path.read_text())
            metric = "ndcg_linear@5" if job.modality in {"text", "visdoc"} else "hit@1"
            reusable = _protocol_matches(args, protocol, job.config) and all(
                metric in json.loads(path.read_text()) for path in score_paths
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            reusable = False
    if job.shard_index is None and reusable:
        return {
            "status": "reused",
            "gpu": gpu,
            "job": asdict(job),
            "scores": [str(path) for path in score_paths],
        }
    if protocol_path.exists() and not _protocol_file_matches(args, protocol_path, job.config):
        stale_output = job.output.with_name(f"{job.output.name}.stale.{time.time_ns()}")
        shutil.move(str(job.output), stale_output)

    job.output.mkdir(parents=True, exist_ok=True)
    job.log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "MMEB_V3_DATA_DIR": str(args.data),
            "MMEB_DEVICE": "cuda:0",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "EVAL_MODALITY": job.modality,
            "EVAL_DATASET_TIMING_LOG": str(job.output / "timing.csv"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    if args.backend == "vllm_merak":
        env.update({"ENABLE_HMINFERENCE_V2": "1", "VLLM_PLUGINS": "merak"})
    else:
        # The Merak distribution is installed as a vLLM plugin in this env.
        # Disable it explicitly so the float baseline uses vLLM's CUDA platform.
        env["VLLM_PLUGINS"] = ""
    python_path = [str(args.vllm_merak), str(args.repo), str(args.official)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    env["PATH"] = os.pathsep.join((str(args.python.parent), env.get("PATH", "")))

    started = time.time()
    command = _command(args, job)
    result = None
    attempts = []
    for attempt in range(1, args.retries + 2):
        with job.log.open("a", encoding="utf-8") as log:
            log.write(f"\nMMEB_JOB_START attempt={attempt} gpu={gpu} time={datetime.now(timezone.utc).isoformat()}\n")
            log.write("COMMAND=" + json.dumps(command) + "\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=args.official,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log.write(
                f"MMEB_JOB_END attempt={attempt} returncode={result.returncode} "
                f"time={datetime.now(timezone.utc).isoformat()}\n"
            )
        attempts.append({"attempt": attempt, "returncode": result.returncode})
        if result.returncode == 0:
            if job.shard_index is not None and _shard_result_valid(args, job):
                break
            if job.shard_index is None and all(path.is_file() and path.stat().st_size > 0 for path in score_paths):
                break

    success = bool(result is not None and result.returncode == 0)
    if job.shard_index is not None:
        success = success and _shard_result_valid(args, job)
    else:
        success = success and all(path.is_file() and path.stat().st_size > 0 for path in score_paths)
    return {
        "status": "success" if success else "failed",
        "kind": "embedding_shard" if job.shard_index is not None else "dataset",
        "gpu": gpu,
        "job": asdict(job),
        "scores": [str(path) for path in score_paths],
        "duration_seconds": round(time.time() - started, 3),
        "attempts": attempts,
    }


def _merge_embedding_shards(args: argparse.Namespace, shard_jobs: list[Job]) -> Job:
    if not shard_jobs:
        raise ValueError("No embedding shards were provided")
    shard_jobs = sorted(shard_jobs, key=lambda job: int(job.shard_index))
    first = shard_jobs[0]
    dataset_name = first.dataset_names[0]
    if len(shard_jobs) != first.shard_count:
        raise ValueError(f"{dataset_name} has {len(shard_jobs)} shard jobs, expected {first.shard_count}")
    if [job.shard_index for job in shard_jobs] != list(range(first.shard_count)):
        raise ValueError(f"{dataset_name} shard indexes are incomplete or duplicated")
    if any(
        job.dataset_names != first.dataset_names
        or job.config != first.config
        or job.final_output != first.final_output
        or job.shard_count != first.shard_count
        for job in shard_jobs
    ):
        raise ValueError(f"{dataset_name} shard jobs do not share one dataset contract")
    if any(not _shard_result_valid(args, job) for job in shard_jobs):
        raise ValueError(f"{dataset_name} has an invalid or incomplete embedding shard")

    query_parts: list[np.ndarray] = []
    query_infos: list[object] = []
    candidate_shards: list[tuple[np.ndarray, list[object]]] = []
    merge_manifest = {
        "schema_version": 1,
        "dataset_name": dataset_name,
        "dataset_config_sha256": _sha256(first.config),
        "shard_count": first.shard_count,
        "shards": [],
    }
    expected_starts = {"query": 0, "candidate": 0}
    expected_totals: dict[str, int | None] = {"query": None, "candidate": None}
    for job in shard_jobs:
        shard_root = job.output / "embedding_shard"
        manifest_path = shard_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for side, stem in (("query", "qry"), ("candidate", "cand")):
            details = manifest[side]
            start, end, total = (int(details[key]) for key in ("start", "end", "total"))
            if start != expected_starts[side]:
                raise ValueError(
                    f"{dataset_name} {side} shard {job.shard_index} starts at {start}; expected {expected_starts[side]}"
                )
            if expected_totals[side] is None:
                expected_totals[side] = total
            elif expected_totals[side] != total:
                raise ValueError(f"{dataset_name} {side} shards disagree on total size")
            embeddings = np.load(shard_root / f"{stem}.npy", allow_pickle=False)
            side_infos = _read_jsonl(shard_root / f"{stem}_info.jsonl")
            if len(embeddings) != end - start or len(side_infos) != end - start:
                raise ValueError(f"{dataset_name} {side} shard {job.shard_index} has invalid row count")
            if side == "query":
                query_parts.append(embeddings)
                query_infos.extend(side_infos)
            else:
                # Keep the independently loaded arrays as the backing storage
                # for candidate row views. Concatenating all candidate shards
                # would temporarily double host memory on very large corpora.
                candidate_shards.append((embeddings, side_infos))
            expected_starts[side] = end
        merge_manifest["shards"].append(
            {
                "index": job.shard_index,
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
            }
        )

    for side in ("query", "candidate"):
        if expected_starts[side] != expected_totals[side]:
            raise ValueError(
                f"{dataset_name} {side} shards cover {expected_starts[side]} rows, expected {expected_totals[side]}"
            )
    dimension = int(
        json.loads((shard_jobs[0].output / "embedding_shard/manifest.json").read_text())["embedding_dimension"]
    )
    query_embeddings = (
        np.concatenate(query_parts, axis=0) if query_parts else np.empty((0, dimension), dtype=np.float32)
    )
    if len(query_embeddings) != len(query_infos):
        raise ValueError(f"{dataset_name} merged query embeddings and metadata differ in length")
    candidate_rows = sum(len(embeddings) for embeddings, _ids in candidate_shards)
    candidate_id_rows = sum(len(ids) for _embeddings, ids in candidate_shards)
    if candidate_rows != candidate_id_rows:
        raise ValueError(f"{dataset_name} merged candidate embeddings and IDs differ in length")
    candidate_embed_dict = {
        candidate_id: embedding
        for embeddings, ids in candidate_shards
        for candidate_id, embedding in zip(ids, embeddings, strict=True)
    }

    final_output = first.final_output
    if final_output is None:
        raise ValueError(f"{dataset_name} shard jobs do not define final_output")
    final_output.mkdir(parents=True, exist_ok=True)
    query_path = final_output / f"{dataset_name}_qry"
    candidate_path = final_output / f"{dataset_name}_tgt"
    info_path = final_output / f"{dataset_name}_info.jsonl"
    score_path = _score_path(final_output, dataset_name)
    score_path.unlink(missing_ok=True)
    _write_pickle_atomic(query_path, query_embeddings)
    _write_pickle_atomic(candidate_path, candidate_embed_dict)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_tmp = info_path.with_suffix(info_path.suffix + ".tmp")
    with info_tmp.open("w", encoding="utf-8") as handle:
        for info in query_infos:
            handle.write(json.dumps(info, ensure_ascii=False) + "\n")
    info_tmp.replace(info_path)
    merge_manifest.update(
        {
            "query_rows": len(query_embeddings),
            "candidate_rows": candidate_rows,
            "embedding_dimension": dimension,
            "query_path": str(query_path),
            "candidate_path": str(candidate_path),
            "info_path": str(info_path),
        }
    )
    _write_json_atomic(final_output / "embedding_shard_merge.json", merge_manifest)
    return Job(
        modality=first.modality,
        group_name=f"dataset_{_safe_name(dataset_name)}_score",
        dataset_names=(dataset_name,),
        config=first.config,
        output=final_output,
        log=args.output / "score_logs" / first.modality / f"{_safe_name(dataset_name)}.log",
        estimated_seconds=0.0,
    )


def _run_score_only(args: argparse.Namespace, job: Job) -> dict[str, object]:
    score_path = _score_path(job.output, job.dataset_names[0])
    job.log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "MMEB_V3_DATA_DIR": str(args.data),
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "EVAL_MODALITY": job.modality,
            "EVAL_DATASET_TIMING_LOG": str(job.output / "timing.csv"),
            "PYTHONUNBUFFERED": "1",
            "VLLM_PLUGINS": "merak" if args.backend == "vllm_merak" else "",
        }
    )
    if args.backend == "vllm_merak":
        env["ENABLE_HMINFERENCE_V2"] = "1"
    python_path = [str(args.vllm_merak), str(args.repo), str(args.official)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    env["PATH"] = os.pathsep.join((str(args.python.parent), env.get("PATH", "")))
    command = _command(args, job, score_only=True)
    started = time.time()
    with job.log.open("a", encoding="utf-8") as log:
        log.write(f"\nMMEB_SCORE_START time={datetime.now(timezone.utc).isoformat()}\n")
        log.write("COMMAND=" + json.dumps(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=args.official,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"MMEB_SCORE_END returncode={result.returncode} time={datetime.now(timezone.utc).isoformat()}\n")
    success = result.returncode == 0 and score_path.is_file() and score_path.stat().st_size > 0
    return {
        "status": "success" if success else "failed",
        "kind": "shard_merge_score",
        "gpu": None,
        "job": asdict(job),
        "scores": [str(score_path)],
        "duration_seconds": round(time.time() - started, 3),
        "returncode": result.returncode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backend", choices=("vllm", "vllm_merak"), required=True)
    parser.add_argument("--vllm-merak", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help="Run only an exact DATASET or MODALITY:DATASET selector; repeatable.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument(
        "--job-mode",
        choices=("group", "dataset"),
        default="group",
        help="Use one engine process per balanced group or per individual dataset.",
    )
    parser.add_argument("--datasets-per-engine", type=int, default=8)
    parser.add_argument(
        "--shard-dataset",
        dest="shard_datasets",
        action="append",
        default=[],
        help="Exact DATASET or MODALITY:DATASET selector to encode in independent shards; repeatable.",
    )
    parser.add_argument("--shards-per-dataset", type=int, default=4)
    parser.add_argument("--vllm-max-num-seqs", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.vllm_merak = args.vllm_merak.resolve()
    args.official = args.official.resolve()
    args.data = args.data.resolve()
    args.model = args.model.resolve()
    args.source_model = args.source_model.resolve()
    args.output = args.output.resolve()
    existing_manifest = args.output / "run_manifest.json"
    if existing_manifest.is_file() and json.loads(existing_manifest.read_text()).get("result_archive"):
        parser.error("This is a completed score archive; choose a new --output directory for inference")
    args.python = args.python.resolve()
    args.vllm_version = importlib.metadata.version("vllm")
    args.transformers_version = importlib.metadata.version("transformers")
    if args.backend == "vllm":
        import inspect

        from vllm_merak.model_executor.model_loader import wemm_embedding_native

        args.vllm_wemm_model_sha256 = _sha256(Path(inspect.getfile(wemm_embedding_native)).resolve())
    else:
        args.vllm_wemm_model_sha256 = None
    args.vllm_merak_source_sha256 = _source_tree_sha256(args.vllm_merak / "vllm_merak")
    unknown = set(args.modalities) - set(DEFAULT_MODALITIES)
    if unknown:
        parser.error(f"Unsupported modalities: {sorted(unknown)}")
    if args.datasets_per_engine <= 0:
        parser.error("--datasets-per-engine must be positive")
    if args.shard_datasets and args.shards_per_dataset < 2:
        parser.error("--shards-per-dataset must be at least 2 when --shard-dataset is used")
    if args.vllm_max_num_seqs is None:
        args.vllm_max_num_seqs = 1
    if args.vllm_max_num_seqs <= 0:
        parser.error("--vllm-max-num-seqs must be positive")
    if args.backend == "vllm_merak" and args.vllm_max_num_seqs != 1:
        parser.error("vLLM Merak hybrid linear state requires --vllm-max-num-seqs 1")

    revision = subprocess.check_output(["git", "-C", str(args.official), "rev-parse", "HEAD"], text=True).strip()
    if revision != OFFICIAL_COMMIT:
        raise RuntimeError(f"Expected official evaluator {OFFICIAL_COMMIT}, got {revision}")

    args.output.mkdir(parents=True, exist_ok=True)
    jobs, manifest_jobs, initial_results = _build_jobs(args)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_commit": revision,
        "mmeb_v3_data_revision": V3_DATA_REVISION,
        "mmeb_v2_data_revision": V2_DATA_REVISION,
        "model": str(args.model),
        "source_model": str(args.source_model),
        "backend": args.backend,
        "vllm_version": args.vllm_version,
        "transformers_version": args.transformers_version,
        "vllm_wemm_model_sha256": args.vllm_wemm_model_sha256,
        "vllm_merak_source_sha256": (args.vllm_merak_source_sha256 if args.backend == "vllm_merak" else None),
        "source_hf_config_sha256": _sha256(args.source_model / "config.json"),
        "model_files": {
            name: _sha256(args.model / name)
            for name in (
                ("golden_meta_info.json", "merak_config.json", "visual_gears.json")
                if args.backend == "vllm_merak"
                else ("config.json", "embedding_chat_template.jinja")
            )
        },
        "modalities": args.modalities,
        "datasets": args.datasets,
        "num_jobs": len(manifest_jobs),
        "num_engine_jobs": len(jobs),
        "gpus": args.gpus.split(","),
        "audio_policy": "unsupported; 11 official audio tasks receive score 0",
        "video_num_frames": 64,
        "max_model_len": 262144,
        "max_num_seqs": args.vllm_max_num_seqs,
        "datasets_per_engine": args.datasets_per_engine,
        "job_mode": args.job_mode,
        "shard_datasets": args.shard_datasets,
        "shards_per_dataset": args.shards_per_dataset,
        "dataloader_workers": args.dataloader_workers,
        "enable_prefix_caching": False,
        "jobs": manifest_jobs,
    }
    _write_json_atomic(args.output / "run_manifest.json", manifest)

    pending: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        pending.put(job)
    results: list[dict[str, object]] = list(initial_results)
    results_lock = threading.Lock()
    _write_json_atomic(args.output / "run_status.json", results)

    def worker(gpu: str) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                result = _run_job(args, gpu, job)
            except Exception as exc:  # keep other GPUs progressing
                result = {
                    "status": "failed",
                    "gpu": gpu,
                    "job": asdict(job),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            with results_lock:
                results.append(result)
                _write_json_atomic(args.output / "run_status.json", results)
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in args.gpus.split(",")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    shard_groups: dict[tuple[str, str], list[Job]] = {}
    for job in jobs:
        if job.shard_index is not None:
            shard_groups.setdefault((job.modality, job.dataset_names[0]), []).append(job)
    for (_modality, dataset_name), shard_jobs in shard_groups.items():
        try:
            score_job = _merge_embedding_shards(args, shard_jobs)
            result = _run_score_only(args, score_job)
        except Exception as exc:
            result = {
                "status": "failed",
                "kind": "shard_merge_score",
                "gpu": None,
                "job": {
                    "modality": shard_jobs[0].modality,
                    "dataset_names": [dataset_name],
                    "output": str(shard_jobs[0].final_output),
                },
                "error": f"{type(exc).__name__}: {exc}",
            }
        with results_lock:
            results.append(result)
            _write_json_atomic(args.output / "run_status.json", results)

    failed = [result for result in results if result["status"] == "failed"]
    completed_tasks = 0
    for item in manifest_jobs:
        task = Task(
            modality=str(item["modality"]),
            dataset_name=str(item["dataset_name"]),
            config=Path(str(item["config"])),
            output=Path(str(item["output"])),
            log=Path(str(item["log"])),
            task_config={},
            estimated_seconds=0.0,
        )
        completed_tasks += int(_task_result_valid(args, task))
    reused_tasks = len(initial_results)
    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total": len(manifest_jobs),
        "completed": completed_tasks,
        "success": completed_tasks - reused_tasks,
        "reused": reused_tasks,
        "failed_groups": len(failed),
        "failed_jobs": [result["job"] for result in failed],
    }
    _write_json_atomic(args.output / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
