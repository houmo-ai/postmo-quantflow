"""Download the Flickr30K test split into the layout expected by flickr30k_utils."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="work_dirs/wemm_embedding/datasets/flickr30k",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--dataset",
        default="Tristan/flickr30k_test",
        help="Hugging Face dataset id with Flickr30K captions and images",
    )
    return parser.parse_args()


def _captions_from_row(row: dict) -> list[str]:
    for key in ("caption", "captions", "raw", "sentences"):
        value = row.get(key)
        if isinstance(value, list):
            captions = [str(item).strip() for item in value if str(item).strip()]
            if captions:
                return captions
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                captions = [str(item).strip() for item in parsed if str(item).strip()]
                if captions:
                    return captions
            return [value.strip()]
    return []


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "flickr30k-images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split="test")
    annotation_path = output_dir / "flickr_annotations_30k.csv"
    kept = 0
    with annotation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "split", "raw"],
        )
        writer.writeheader()
        for index, row in enumerate(dataset):
            split = str(row.get("split", "test")).strip() or "test"
            if split not in {"test", "testset"}:
                continue
            captions = _captions_from_row(row)
            image = row.get("image")
            if not captions or image is None:
                continue
            filename = str(row.get("filename") or row.get("image_id") or f"{index:05d}.jpg")
            if Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                filename = f"{Path(filename).stem}.jpg"
            image_path = image_dir / filename
            if isinstance(image, Image.Image):
                image.convert("RGB").save(image_path)
            else:
                raise TypeError(f"Unsupported image type: {type(image)!r}")
            writer.writerow(
                {
                    "filename": filename,
                    "split": "test",
                    "raw": json.dumps(captions, ensure_ascii=False),
                }
            )
            kept += 1
            if args.max_images and kept >= args.max_images:
                break
    if kept == 0:
        raise RuntimeError(f"No Flickr30K test images were written from {args.dataset}")
    print(json.dumps({"output_dir": str(output_dir), "num_images": kept}, indent=2))


if __name__ == "__main__":
    main()
