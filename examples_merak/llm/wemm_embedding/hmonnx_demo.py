"""Run the unified WeMM-Embedding HMONNX interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


try:
    from .runtime_utils import load_hmonnx_embedder
except ImportError:
    from runtime_utils import load_hmonnx_embedder


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument(
        "--text",
        nargs="+",
        default=[
            "A dog playing in the park.",
            "A cat sitting on a chair.",
        ],
    )
    parser.add_argument("--image", nargs="*", default=[])
    parser.add_argument("--video", nargs="*", default=[])
    parser.add_argument(
        "--mixed-text",
        default="Represent the content and semantics of this image.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def _validate_media_files(paths: list[str], kind: str) -> list[str]:
    media_files = []
    for path_text in paths:
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"{kind} file not found: {path}")
        media_files.append(str(path))
    return media_files


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from xhquant.api import xhquant_init

    xhquant_init(None, args.debug)
    model, meta_file = load_hmonnx_embedder(args.export_dir, args.device)
    if args.fast:
        model.to_fast()

    images = _validate_media_files(args.image, "Image")
    item_groups: list[tuple[str, torch.Tensor]] = []
    if args.text:
        item_groups.append(("text", model.embed_texts(args.text)))
    if images:
        item_groups.append(("image", model.embed_images(images)))
        mixed = model.embed_items([{"image": image, "text": args.mixed_text} for image in images])
        item_groups.append(("mixed", mixed))
    if args.video:
        videos = _validate_media_files(args.video, "Video")
        item_groups.append(("video", model.embed_items([{"video": video} for video in videos])))

    embeddings = torch.cat([group for _, group in item_groups], dim=0)
    report = {
        "meta_file": str(meta_file),
        "groups": {
            name: {
                "shape": list(group.shape),
                "dtype": str(group.dtype),
                "norms": group.norm(dim=-1).tolist(),
            }
            for name, group in item_groups
        },
        "similarity": (embeddings.float() @ embeddings.float().T).tolist(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
