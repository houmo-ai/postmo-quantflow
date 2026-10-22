"""Evaluate WeMM-Embedding on Flickr30K retrieval, matching the Qwen3-VL-Embedding protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


try:
    from ..native_utils import embed_native_items, load_native_embedder
    from ..runtime_utils import load_hmonnx_embedder
    from .flickr30k_utils import retrieval_metrics
    from .retrieval_dataset import load_retrieval_dataset
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flickr30k_utils import retrieval_metrics
    from native_utils import embed_native_items, load_native_embedder
    from retrieval_dataset import load_retrieval_dataset
    from runtime_utils import load_hmonnx_embedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["native", "hmonnx"], required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-type", default="flickr30k")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument(
        "--native-image-size",
        type=int,
        default=None,
        help="Optional fixed square size; omit for the dynamic-resolution protocol",
    )
    parser.add_argument("--image-min-pixels", type=int, default=64 * 32 * 32)
    parser.add_argument("--image-max-pixels", type=int, default=1536 * 4 * 16 * 16)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--report",
        default="work_dirs/wemm_embedding/eval_flickr30k_1000/report.json",
    )
    parser.add_argument(
        "--save-embeddings",
        default=None,
        help="Directory for text_emb.pt, image_emb.pt, and Flickr ordering metadata",
    )
    return parser.parse_args()


def _load_hmonnx_model(export_dir: str, device: str):
    return load_hmonnx_embedder(export_dir, device)


def _embed_hmonnx(
    model: Any,
    items: list[dict[str, Any]],
    batch_size: int,
    label: str,
) -> torch.Tensor:
    embeddings = []
    start = time.time()
    for offset in range(0, len(items), batch_size):
        batch = items[offset : offset + batch_size]
        embeddings.append(model.embed_items(batch).cpu())
        completed = offset + len(batch)
        if completed == len(items) or completed % 10 == 0:
            rate = completed / max(time.time() - start, 1e-6)
            print(f"[{label}] {completed}/{len(items)} ({rate:.2f} samples/s)", flush=True)
    return torch.cat(embeddings, dim=0)


def _embed_native(
    model: Any,
    processor: Any,
    items: list[dict[str, Any]],
    device: str,
    batch_size: int,
    image_size: int | None,
    image_min_pixels: int,
    image_max_pixels: int,
    label: str,
) -> torch.Tensor:
    embeddings = []
    start = time.time()
    for offset in range(0, len(items), batch_size):
        batch = items[offset : offset + batch_size]
        embeddings.append(
            embed_native_items(
                model,
                processor,
                batch,
                device=device,
                batch_size=len(batch),
                image_size=image_size,
                image_min_pixels=image_min_pixels,
                image_max_pixels=image_max_pixels,
            )
        )
        completed = offset + len(batch)
        if completed == len(items) or completed % 10 == 0:
            rate = completed / max(time.time() - start, 1e-6)
            print(f"[{label}] {completed}/{len(items)} ({rate:.2f} samples/s)", flush=True)
    return torch.cat(embeddings, dim=0)


def main() -> None:
    args = parse_args()
    if args.image_min_pixels <= 0 or args.image_max_pixels < args.image_min_pixels:
        raise ValueError("Expected 0 < --image-min-pixels <= --image-max-pixels")
    dataset = load_retrieval_dataset(args.dataset_type, args.dataset_dir).limited(args.max_images)
    images, captions, caption_owners, image_to_captions = dataset.flatten()

    started = time.time()
    if args.backend == "native":
        if not args.model_dir:
            raise ValueError("--model-dir is required for the native backend")
        model, processor = load_native_embedder(args.model_dir, device=args.device)
        text_embeddings = _embed_native(
            model,
            processor,
            [{"text": caption} for caption in captions],
            device=args.device,
            batch_size=args.text_batch_size,
            image_size=None,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
            label="text",
        )
        image_embeddings = _embed_native(
            model,
            processor,
            [{"image": image} for image in images],
            device=args.device,
            batch_size=args.image_batch_size,
            image_size=args.native_image_size,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
            label="image",
        )
        extra = {
            "model_dir": args.model_dir,
            "native_image_size": args.native_image_size,
            "image_preprocessing": {
                "mode": "fixed_square" if args.native_image_size is not None else "dynamic",
                "resized_height": args.native_image_size,
                "resized_width": args.native_image_size,
                "min_pixels": args.image_min_pixels,
                "max_pixels": args.image_max_pixels,
                "preserve_aspect_ratio": args.native_image_size is None,
            },
        }
    else:
        if not args.export_dir:
            raise ValueError("--export-dir is required for the hmonnx backend")
        model, meta_file = _load_hmonnx_model(args.export_dir, args.device)
        text_embeddings = _embed_hmonnx(
            model,
            [{"text": caption} for caption in captions],
            args.text_batch_size,
            "text",
        )
        image_embeddings = _embed_hmonnx(
            model,
            [
                {
                    "image": image,
                    "min_pixels": args.image_min_pixels,
                    "max_pixels": args.image_max_pixels,
                }
                for image in images
            ],
            args.image_batch_size,
            "image",
        )
        extra = {
            "export_dir": args.export_dir,
            "quant_meta": str(meta_file),
            "image_preprocessing": {
                "mode": "dynamic",
                "resized_height": None,
                "resized_width": None,
                "min_pixels": args.image_min_pixels,
                "max_pixels": args.image_max_pixels,
                "preserve_aspect_ratio": True,
            },
        }

    text_to_image = retrieval_metrics(
        text_embeddings,
        image_embeddings,
        [[owner] for owner in caption_owners],
    )
    image_to_text = retrieval_metrics(
        image_embeddings,
        text_embeddings,
        image_to_captions,
    )
    if args.save_embeddings:
        embedding_dir = Path(args.save_embeddings)
        embedding_dir.mkdir(parents=True, exist_ok=True)
        torch.save(text_embeddings, embedding_dir / "text_emb.pt")
        torch.save(image_embeddings, embedding_dir / "image_emb.pt")
        (embedding_dir / "items.json").write_text(
            json.dumps(dataset.ordering_metadata(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    report = {
        "backend": args.backend,
        "dataset": dataset.name,
        "dataset_split": dataset.split,
        "dataset_metadata": dataset.metadata,
        "dataset_dir": args.dataset_dir,
        "num_images": len(images),
        "num_captions": len(captions),
        "elapsed_sec": round(time.time() - started, 3),
        "text_to_image": text_to_image,
        "image_to_text": image_to_text,
        "embedding_dir": args.save_embeddings,
        **extra,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
