"""Summarize WeMM Flickr30K accuracy reports and pick the best quantized model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_KEYS = [
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr@10",
    "ndcg@10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        default="work_dirs/wemm_embedding/eval_flickr30k_1000",
    )
    parser.add_argument(
        "--output",
        default="work_dirs/wemm_embedding/eval_flickr30k_1000/summary_ranked.json",
    )
    return parser.parse_args()


def _score(report: dict) -> float:
    text_to_image = report["text_to_image"]
    image_to_text = report["image_to_text"]
    return (
        text_to_image["recall@1"] + text_to_image["mrr@10"] + image_to_text["recall@1"] + image_to_text["mrr@10"]
    ) / 4.0


def load_accuracy_reports(report_dir: str | Path) -> list[dict]:
    report_dir = Path(report_dir)
    if not report_dir.is_dir():
        raise FileNotFoundError(f"Flickr30K report directory not found: {report_dir}")
    reports = []
    for path in sorted(report_dir.glob("*.json")):
        if path.name in {"summary.json", "summary_ranked.json"} or "smoke" in path.name:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "text_to_image" not in data or "image_to_text" not in data:
            continue
        reports.append({"path": str(path), **data, "score": _score(data)})
    if not reports:
        raise FileNotFoundError(f"No Flickr30K accuracy reports found under {report_dir}")
    return reports


def main() -> None:
    args = parse_args()
    reports = load_accuracy_reports(args.report_dir)
    quantized = [item for item in reports if item.get("backend") == "hmonnx"]
    best = max(quantized, key=lambda item: item["score"]) if quantized else None
    summary = {
        "reports": reports,
        "best_quantized": (
            None
            if best is None
            else {
                "path": best["path"],
                "export_dir": best.get("export_dir"),
                "quant_meta": best.get("quant_meta"),
                "score": best["score"],
                "text_to_image": best["text_to_image"],
                "image_to_text": best["image_to_text"],
                "num_images": best["num_images"],
                "num_captions": best["num_captions"],
            }
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["best_quantized"], ensure_ascii=False, indent=2))
    print()
    for item in reports:
        print(
            f"{Path(item['path']).name:40s} "
            f"t2i_r1={item['text_to_image']['recall@1']:.4f} "
            f"i2t_r1={item['image_to_text']['recall@1']:.4f} "
            f"score={item['score']:.4f}"
        )


if __name__ == "__main__":
    main()
