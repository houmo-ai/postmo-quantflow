"""Compare native BF16 WeMM embeddings with the exported HMONNX runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


try:
    from .native_utils import embed_native_items, load_native_embedder
    from .runtime_utils import load_hmonnx_embedder
except ImportError:
    from native_utils import embed_native_items, load_native_embedder
    from runtime_utils import load_hmonnx_embedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument(
        "--text",
        nargs="+",
        default=[
            "A dog playing in the park",
            "A cat sitting on a chair",
            "麻婆豆腐是怎样制作的？",
            "Quantization can reduce model memory use.",
        ],
    )
    parser.add_argument("--image", nargs="*", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--native-image-size", type=int, default=None)
    parser.add_argument("--image-min-pixels", type=int, default=64 * 32 * 32)
    parser.add_argument("--image-max-pixels", type=int, default=1536 * 4 * 16 * 16)
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON path for the cosine-parity report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    items: list[dict] = [{"text": text} for text in args.text]
    labels = [f"text:{text}" for text in args.text]
    if args.image:
        for image_path in args.image:
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            items.append({"image": Image.open(path).convert("RGB")})
            labels.append(f"image:{path}")

    native_model, processor = load_native_embedder(args.model_dir, device=args.device)
    native_embeddings = embed_native_items(
        native_model,
        processor,
        items,
        device=args.device,
        batch_size=1,
        image_size=args.native_image_size if any("image" in item for item in items) else None,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
    )
    del native_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hmonnx_model, meta_file = load_hmonnx_embedder(args.export_dir, args.device)
    hmonnx_items = [
        {
            **item,
            **({"min_pixels": args.image_min_pixels, "max_pixels": args.image_max_pixels} if "image" in item else {}),
        }
        for item in items
    ]
    hmonnx_embeddings = hmonnx_model.embed_items(hmonnx_items).cpu()

    per_item_cosine = F.cosine_similarity(native_embeddings, hmonnx_embeddings, dim=-1)
    report = {
        "model_dir": args.model_dir,
        "export_dir": args.export_dir,
        "quant_meta": str(meta_file),
        "embedding_shape": list(hmonnx_embeddings.shape),
        "labels": labels,
        "native_norms": native_embeddings.norm(dim=-1).tolist(),
        "hmonnx_norms": hmonnx_embeddings.norm(dim=-1).tolist(),
        "per_item_cosine": per_item_cosine.tolist(),
        "mean_cosine": float(per_item_cosine.mean()),
        "min_cosine": float(per_item_cosine.min()),
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
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
