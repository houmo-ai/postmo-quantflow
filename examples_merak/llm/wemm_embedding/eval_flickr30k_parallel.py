"""Run Flickr30K HMONNX embedding evaluation with data-parallel workers."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import torch


try:
    from .flickr30k_utils import retrieval_metrics
    from .retrieval_dataset import load_retrieval_dataset
    from .runtime_utils import load_hmonnx_embedder, resolve_hmonnx_meta
except ImportError:
    from flickr30k_utils import retrieval_metrics
    from retrieval_dataset import load_retrieval_dataset
    from runtime_utils import load_hmonnx_embedder, resolve_hmonnx_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-type", default="flickr30k")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-min-pixels", type=int, default=64 * 32 * 32)
    parser.add_argument("--image-max-pixels", type=int, default=1536 * 4 * 16 * 16)
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=7200.0,
        help="Seconds to wait for each worker (default: 7200)",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def _bounds(length: int, index: int, count: int) -> tuple[int, int]:
    return length * index // count, length * (index + 1) // count


def _worker(
    index: int,
    workers: int,
    export_dir: str,
    captions: list[str],
    images: list[str],
    output_dir: str,
    device: str,
    image_min_pixels: int,
    image_max_pixels: int,
) -> None:
    os.environ["ENABLE_HMINFERENCE_V2"] = "1"
    model, meta_file = _load_model(export_dir, device)
    text_start, text_end = _bounds(len(captions), index, workers)
    image_start, image_end = _bounds(len(images), index, workers)
    text_items = [{"text": item} for item in captions[text_start:text_end]]
    image_items = [
        {"image": item, "min_pixels": image_min_pixels, "max_pixels": image_max_pixels}
        for item in images[image_start:image_end]
    ]
    text_embeddings = []
    image_embeddings = []
    started = time.time()
    for offset, item in enumerate(text_items):
        text_embeddings.append(model.embed_items([item]).cpu())
        if (offset + 1) % 100 == 0 or offset + 1 == len(text_items):
            print(
                f"[worker {index}] text {offset + 1}/{len(text_items)} "
                f"({(offset + 1) / max(time.time() - started, 1e-6):.2f}/s)",
                flush=True,
            )
    for offset, item in enumerate(image_items):
        image_embeddings.append(model.embed_items([item]).cpu())
        if (offset + 1) % 100 == 0 or offset + 1 == len(image_items):
            print(
                f"[worker {index}] image {offset + 1}/{len(image_items)} "
                f"({(offset + 1) / max(time.time() - started, 1e-6):.2f}/s)",
                flush=True,
            )
    output_path = Path(output_dir) / f"worker_{index:02d}.pt"
    torch.save(
        {
            "text_start": text_start,
            "text_end": text_end,
            "image_start": image_start,
            "image_end": image_end,
            "text_embeddings": _concat_worker_embeddings(text_embeddings),
            "image_embeddings": _concat_worker_embeddings(image_embeddings),
            "meta_file": str(meta_file),
        },
        output_path,
    )


def _load_model(export_dir: str, device: str):
    return load_hmonnx_embedder(export_dir, device)


def _advance_half_open_range(start: int, end: int, cursor: int, name: str) -> int:
    """Accept adjacent half-open ranges [start, end) and reject overlap or gaps."""

    if end < start:
        raise ValueError(f"{name} shard has inverted range: {start}-{end}")
    if start < cursor:
        raise ValueError(
            f"{name} shard range overlaps previous coverage: previous ended at {cursor}, got {start}-{end}"
        )
    if start > cursor:
        raise ValueError(
            f"{name} shard range has a gap after previous coverage: previous ended at {cursor}, got {start}-{end}"
        )
    return end


def validate_shard_ranges(
    ranges: list[tuple[int, int]],
    *,
    expected_count: int,
    name: str,
) -> int:
    """Fail closed if half-open ``[start, end)`` shards overlap, gap, or miss coverage.

    Adjacent ranges are valid when the next ``start`` equals the previous ``end``.
    Overlap (``start < cursor``) and gaps (``start > cursor``) raise ``ValueError``
    before any embedding concatenation. The final cursor must equal the
    dataset size ``expected_count`` so no rows are dropped or invented.
    """

    cursor = 0
    for start, end in ranges:
        cursor = _advance_half_open_range(start, end, cursor, name)
    if cursor != expected_count:
        raise ValueError(f"{name} shard coverage {cursor} does not match dataset size {expected_count}")
    return cursor


def _concat_worker_embeddings(pieces: list[torch.Tensor]) -> torch.Tensor:
    if not pieces:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(pieces, dim=0)


def _load_worker_shard(output_dir: Path, index: int) -> dict[str, Any]:
    path = output_dir / f"worker_{index:02d}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"parallel eval shard is missing: {path}. A worker likely failed before writing results."
        )
    shard = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(shard, dict):
        raise TypeError(f"parallel eval shard must be a mapping: {path}")
    return shard


def _merge_modality(
    shards: list[dict[str, Any]],
    *,
    start_key: str,
    end_key: str,
    embedding_key: str,
    expected_count: int,
    name: str,
) -> torch.Tensor:
    ordered = sorted(shards, key=lambda shard: (int(shard[start_key]), int(shard[end_key])))
    ranges = [(int(shard[start_key]), int(shard[end_key])) for shard in ordered]
    validate_shard_ranges(ranges, expected_count=expected_count, name=name)
    pieces: list[torch.Tensor] = []
    dim: int | None = None
    for shard, (start, end) in zip(ordered, ranges, strict=True):
        embeddings = shard.get(embedding_key)
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(f"{name} embeddings must be a tensor, got {type(embeddings)!r}")
        expected_rows = end - start
        shape = tuple(embeddings.shape)
        if len(shape) < 2:
            got = "0D scalar" if len(shape) == 0 else "1D vector"
            raise ValueError(f"{name} embeddings must be rank-2, got {got} {list(shape)} for range {start}-{end}")
        if len(shape) != 2:
            raise ValueError(f"{name} embeddings must be rank-2, got shape {list(shape)} for range {start}-{end}")
        rows, current_dim = shape[0], shape[1]
        if rows != expected_rows:
            raise ValueError(f"{name} shard rows {rows} do not match range {start}-{end}")
        if expected_rows:
            if dim is None:
                dim = int(current_dim)
            elif int(current_dim) != dim:
                raise ValueError(f"{name} embedding dimension {current_dim} does not match {dim}")
            pieces.append(embeddings)
    if not pieces:
        if expected_count != 0:
            raise ValueError(f"{name} embeddings are missing for {expected_count} rows")
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(pieces, dim=0)


def _merge(
    output_dir: Path,
    workers: int,
    num_captions: int,
    num_images: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    shards = [_load_worker_shard(output_dir, index) for index in range(workers)]
    text = _merge_modality(
        shards,
        start_key="text_start",
        end_key="text_end",
        embedding_key="text_embeddings",
        expected_count=num_captions,
        name="text",
    )
    image = _merge_modality(
        shards,
        start_key="image_start",
        end_key="image_end",
        embedding_key="image_embeddings",
        expected_count=num_images,
        name="image",
    )
    return text, image


def _terminate_workers(processes: list[Any]) -> None:
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except Exception:
            continue
    for process in processes:
        try:
            process.join(1)
        except Exception:
            continue
        try:
            if process.is_alive():
                process.kill()
                process.join(1)
        except Exception:
            continue


def wait_for_worker_processes(
    processes: list[Any],
    *,
    timeout: float | None = None,
    name: str = "HMONNX",
) -> None:
    """Join workers and fail closed on errors, timeouts, or join exceptions."""

    for index, process in enumerate(processes):
        try:
            process.join(timeout)
        except Exception as exc:
            _terminate_workers(processes)
            raise RuntimeError(
                f"{name} workers failed (worker {index} join raised {type(exc).__name__}: {exc}); "
                "partial shards were not merged"
            ) from exc
        if process.is_alive():
            _terminate_workers(processes)
            raise TimeoutError(
                f"{name} workers still running at indices [{index}]; remaining workers were terminated "
                "and partial shards were not merged"
            )
        if process.exitcode != 0:
            _terminate_workers(processes)
            raise RuntimeError(
                f"{name} workers failed at index {index} with exit code {process.exitcode}; "
                "partial shards were not merged"
            )


def _remove_worker_shards(output_dir: Path, workers: int) -> None:
    for index in range(workers):
        (output_dir / f"worker_{index:02d}.pt").unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.image_min_pixels <= 0 or args.image_max_pixels < args.image_min_pixels:
        raise ValueError("Expected 0 < --image-min-pixels <= --image-max-pixels")
    dataset = load_retrieval_dataset(args.dataset_type, args.dataset_dir).limited(args.max_images)
    images, captions, caption_owners, image_to_captions = dataset.flatten()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_worker_shards(output_dir, args.workers)
    context = mp.get_context("spawn")
    processes = []
    try:
        for index in range(args.workers):
            process = context.Process(
                target=_worker,
                args=(
                    index,
                    args.workers,
                    args.export_dir,
                    captions,
                    images,
                    str(output_dir),
                    args.device,
                    args.image_min_pixels,
                    args.image_max_pixels,
                ),
            )
            process.start()
            processes.append(process)
        wait_for_worker_processes(processes, timeout=args.worker_timeout, name="HMONNX")
        text_embeddings, image_embeddings = _merge(
            output_dir,
            args.workers,
            num_captions=len(captions),
            num_images=len(images),
        )
    finally:
        _terminate_workers(processes)
        _remove_worker_shards(output_dir, args.workers)
    torch.save(text_embeddings, output_dir / "text_emb.pt")
    torch.save(image_embeddings, output_dir / "image_emb.pt")
    items = dataset.ordering_metadata()
    (output_dir / "items.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "backend": "hmonnx",
        "runtime": "HMONNXInferenceV2",
        "dataset": dataset.name,
        "dataset_split": dataset.split,
        "dataset_metadata": dataset.metadata,
        "dataset_dir": args.dataset_dir,
        "num_images": len(images),
        "num_captions": len(captions),
        "workers": args.workers,
        "embedding_dir": str(output_dir),
        "export_dir": args.export_dir,
        "quant_meta": str(resolve_hmonnx_meta(args.export_dir)),
        "image_preprocessing": {
            "mode": "dynamic",
            "resized_height": None,
            "resized_width": None,
            "min_pixels": args.image_min_pixels,
            "max_pixels": args.image_max_pixels,
            "preserve_aspect_ratio": True,
        },
        "text_to_image": retrieval_metrics(text_embeddings, image_embeddings, [[owner] for owner in caption_owners]),
        "image_to_text": retrieval_metrics(image_embeddings, text_embeddings, image_to_captions),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


__all__ = [
    "validate_shard_ranges",
    "wait_for_worker_processes",
]


if __name__ == "__main__":
    main()
