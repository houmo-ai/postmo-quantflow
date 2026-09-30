"""Compare completed float-vLLM and quantized-vLLM-Merak MMEB-v3 runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _fmt(value: float | None) -> str:
    return "pending" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("float_run", type=Path)
    parser.add_argument("quant_run", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    float_summary = json.loads((args.float_run / "aggregate/summary.json").read_text())
    quant_summary = json.loads((args.quant_run / "aggregate/summary.json").read_text())
    float_tasks = {row["dataset"]: row for row in float_summary["tasks"]}
    quant_tasks = {row["dataset"]: row for row in quant_summary["tasks"]}
    if set(float_tasks) != set(quant_tasks):
        raise ValueError("Float and quantized task sets differ")

    task_rows = []
    for name in sorted(float_tasks):
        left = float_tasks[name]
        right = quant_tasks[name]
        if left["metric"] != right["metric"] or left["modality"] != right["modality"]:
            raise ValueError(f"Protocol mismatch for {name}")
        float_score = left["score_percent"]
        quant_score = right["score_percent"]
        task_rows.append(
            {
                "modality": left["modality"],
                "dataset": name,
                "metric": left["metric"],
                "float_vllm": float_score,
                "quant_vllm_merak": quant_score,
                "delta_quant_minus_float": (
                    None if float_score is None or quant_score is None else quant_score - float_score
                ),
            }
        )

    order = [
        "V3-All",
        "Supported-Only",
        "MMEB-v2",
        "Image",
        "Video",
        "VisDoc",
        "Text",
        "Reasoning Retrieval",
        "Instruction Following",
        "Long Context",
        "Multi Condition",
        "General Retrieval",
        "Agent",
        "Tool",
        "GUI",
        "Memory",
        "MCMR",
        "Audio",
    ]
    aggregates = {}
    for name in order:
        left = float_summary["aggregates"][name]
        right = quant_summary["aggregates"][name]
        float_score = left["score"]
        quant_score = right["score"]
        aggregates[name] = {
            "num_tasks": left["num_tasks"],
            "float_vllm": float_score,
            "quant_vllm_merak": quant_score,
            "delta_quant_minus_float": (
                None if float_score is None or quant_score is None else quant_score - float_score
            ),
        }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": args.title,
        "float_run": str(args.float_run.resolve()),
        "quant_run": str(args.quant_run.resolve()),
        "protocol": float_summary["protocol"],
        "aggregates": aggregates,
        "tasks": task_rows,
    }
    (output / "comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with (output / "per_task_delta.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)

    lines = [
        f"# {args.title}",
        "",
        "| Group | Tasks | Float vLLM | Quantized vLLM Merak | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in order:
        row = aggregates[name]
        lines.append(
            f"| {name} | {row['num_tasks']} | {_fmt(row['float_vllm'])} | "
            f"{_fmt(row['quant_vllm_merak'])} | {_fmt(row['delta_quant_minus_float'])} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
