#!/usr/bin/env python3
"""Build Torch vs HMONNX accuracy table from bench summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ORDER = ("OCRBench", "ChartQA_TEST", "AI2D_TEST（Direct）")


def _load_summaries(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return {item["dataset"]: item for item in data}


def _fmt(dataset: str, score: float) -> str:
    if dataset == "OCRBench":
        return str(int(round(score)))
    return f"{score:.2f}%"


def _delta(dataset: str, torch_score: float, hmonnx_score: float) -> str:
    diff = hmonnx_score - torch_score
    if dataset == "OCRBench":
        return f"{diff:+.0f}"
    return f"{diff:+.2f} pp"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-summary", type=Path, required=True)
    parser.add_argument("--hmonnx-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    torch_map = _load_summaries(args.torch_summary)
    hmonnx_map = _load_summaries(args.hmonnx_summary)

    lines = [
        "| 数据集 | 样本数 | Torch | HMONNX | 差值 |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = []
    for name in ORDER:
        if name not in torch_map or name not in hmonnx_map:
            continue
        t = torch_map[name]
        h = hmonnx_map[name]
        n = t.get("num_samples") or h.get("num_samples")
        row = {
            "dataset": name,
            "num_samples": n,
            "torch": t["score"],
            "hmonnx": h["score"],
            "delta": h["score"] - t["score"],
        }
        rows.append(row)
        lines.append(
            f"| {name} | {n} | {_fmt(name, t['score'])} | {_fmt(name, h['score'])} | {_delta(name, t['score'], h['score'])} |"
        )

    table = "\n".join(lines) + "\n"
    print(table)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"table_markdown": table, "rows": rows}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
