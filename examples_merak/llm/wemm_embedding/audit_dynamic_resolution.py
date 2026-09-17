"""Audit the dynamic image sizes and HMONNX visual gears used by WeMM evals."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from qwen_vl_utils import smart_resize

from examples_merak.llm.wemm_embedding.flickr30k_utils import load_flickr30k


GEARS = (96, 196, 384, 704, 1536)


def audit_images(paths: list[Path], *, min_pixels: int, max_pixels: int) -> dict:
    original_sizes: Counter[str] = Counter()
    resized_sizes: Counter[str] = Counter()
    gear_counts: Counter[str] = Counter()
    token_counts = []
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=32,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        visual_tokens = (resized_height // 32) * (resized_width // 32)
        gear = next((capacity for capacity in GEARS if visual_tokens <= capacity), None)
        if gear is None:
            raise ValueError(f"{path}: {visual_tokens} visual tokens exceed the m{GEARS[-1]} graph")
        original_sizes[f"{width}x{height}"] += 1
        resized_sizes[f"{resized_width}x{resized_height}"] += 1
        gear_counts[f"m{gear}"] += 1
        token_counts.append(visual_tokens)
    return {
        "images": len(paths),
        "unique_original_sizes": len(original_sizes),
        "unique_resized_sizes": len(resized_sizes),
        "visual_token_min": min(token_counts),
        "visual_token_max": max(token_counts),
        "fixed_448x448_images": resized_sizes.get("448x448", 0),
        "gear_counts": {f"m{gear}": gear_counts[f"m{gear}"] for gear in GEARS},
        "resized_size_counts": dict(sorted(resized_sizes.items())),
    }


def mmeb_query_images(data_root: Path, task: str) -> list[Path]:
    table_path = data_root / "image-query" / task / "test-00000-of-00001.parquet"
    rows = pq.read_table(table_path, columns=["qry_img_path"]).to_pylist()
    media_root = data_root / "image-tasks" / "MMEB"
    paths = [media_root / row["qry_img_path"] for row in rows]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{task}: missing query image {missing[0]}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flickr-dir", type=Path, required=True)
    parser.add_argument("--mmeb-data-root", type=Path, required=True)
    parser.add_argument("--min-pixels", type=int, default=64 * 32 * 32)
    parser.add_argument("--max-pixels", type=int, default=1536 * 4 * 16 * 16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        parser.error("Expected 0 < --min-pixels <= --max-pixels")

    flickr_paths = [Path(path) for path, _captions in load_flickr30k(args.flickr_dir)]
    datasets = {
        "Flickr30K-1000": flickr_paths,
        "Visual7W": mmeb_query_images(args.mmeb_data_root, "Visual7W"),
        "ChartQA": mmeb_query_images(args.mmeb_data_root, "ChartQA"),
    }
    result = {
        "image_preprocessing": {
            "mode": "dynamic",
            "resized_height": None,
            "resized_width": None,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "preserve_aspect_ratio": True,
        },
        "spatial_merge_size": 2,
        "patch_size": 16,
        "visual_gears": list(GEARS),
        "datasets": {
            name: audit_images(paths, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
            for name, paths in datasets.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
