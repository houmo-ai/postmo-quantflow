"""Compare all Flickr30K native and HMONNX embedding rows by cosine similarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", required=True)
    parser.add_argument("--hmonnx-dir", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def _load_tensor(directory: Path, name: str) -> torch.Tensor:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(path)
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(f"Expected a rank-2 tensor in {path}")
    return tensor.float()


def _load_items(directory: Path) -> dict:
    path = directory / "items.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: torch.Tensor) -> dict[str, float | int]:
    percentiles = torch.quantile(values, torch.tensor([0.05, 0.5, 0.95]))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "p5": float(percentiles[0]),
        "p50": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "max": float(values.max()),
    }


def _compare(
    native_dir: Path,
    hmonnx_dir: Path,
    tensor_name: str,
    items: dict,
) -> dict[str, float | int]:
    native = _load_tensor(native_dir, tensor_name)
    hmonnx = _load_tensor(hmonnx_dir, tensor_name)
    if native.shape != hmonnx.shape:
        raise ValueError(f"{tensor_name} shape mismatch: native={tuple(native.shape)} hmonnx={tuple(hmonnx.shape)}")
    cosine = F.cosine_similarity(native, hmonnx, dim=-1)
    return _stats(cosine)


def main() -> None:
    args = parse_args()
    native_dir = Path(args.native_dir)
    hmonnx_dir = Path(args.hmonnx_dir)
    native_items = _load_items(native_dir)
    hmonnx_items = _load_items(hmonnx_dir)
    if native_items != hmonnx_items:
        raise ValueError("Native and HMONNX embedding ordering metadata differ")
    text_stats = _compare(native_dir, hmonnx_dir, "text_emb.pt", native_items)
    image_stats = _compare(native_dir, hmonnx_dir, "image_emb.pt", native_items)
    report = {
        "native_dir": str(native_dir),
        "hmonnx_dir": str(hmonnx_dir),
        "num_images": len(native_items["images"]),
        "num_captions": len(native_items["captions"]),
        "text_cosine": text_stats,
        "image_cosine": image_stats,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
