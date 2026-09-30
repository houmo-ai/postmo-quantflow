#!/usr/bin/env python3
"""Compare the first two invocations in an HMONNX per-node JSONL trace."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_trace(path: Path) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    summaries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            payload = row.get("payload", {})
            event = row.get("event")
            if event == "hmonnx_node_output":
                groups[(payload["stage"], int(payload["invocation"]))].append(payload)
            elif event == "hmonnx_node_trace_summary":
                summaries.append(payload)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["node_index"]))
    return dict(groups), summaries


def compare(path: Path) -> dict[str, Any]:
    groups, summaries = _read_trace(path)
    stages = sorted({stage for stage, _ in groups})
    report: dict[str, Any] = {
        "trace": str(path.resolve()),
        "summaries": summaries,
        "stages": {},
        "all_equal": True,
    }
    for stage in stages:
        left = groups.get((stage, 0), [])
        right = groups.get((stage, 1), [])
        first_difference = None
        equal_nodes = 0
        for position, (lhs, rhs) in enumerate(zip(left, right, strict=False)):
            identity_keys = ("node_index", "node_name", "node_op", "target", "backend_type")
            identity_equal = all(lhs.get(key) == rhs.get(key) for key in identity_keys)
            output_equal = lhs.get("output") == rhs.get("output")
            if identity_equal and output_equal:
                equal_nodes += 1
                continue
            first_difference = {
                "position": position,
                "left": {key: lhs.get(key) for key in identity_keys},
                "right": {key: rhs.get(key) for key in identity_keys},
                "identity_equal": identity_equal,
                "output_equal": output_equal,
                "left_output": lhs.get("output"),
                "right_output": rhs.get("output"),
            }
            break

        count_equal = len(left) == len(right)
        stage_equal = count_equal and first_difference is None
        if not stage_equal:
            report["all_equal"] = False
        report["stages"][stage] = {
            "invocation_0_nodes": len(left),
            "invocation_1_nodes": len(right),
            "equal_prefix_nodes": equal_nodes,
            "all_equal": stage_equal,
            "first_difference": first_difference,
            "backend_types": dict(
                sorted(
                    Counter(
                        row["backend_type"] or row["node_op"] for row in left
                    ).items()
                )
            ),
        }

    missing = sorted(
        stage
        for stage in {"visual", "prefill", "decode"}
        if (stage, 0) not in groups or (stage, 1) not in groups
    )
    report["missing_stages"] = missing
    if missing:
        report["all_equal"] = False
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.trace)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["all_equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
