"""Score saved predictions using a local copy of Google's IFEval package.

This entry point never imports or loads the model and does not download code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Expected nonempty JSONL objects: {path}")
    return records


def align_predictions(dataset, predictions, *, allow_partial=False, include_failed=False):
    """Match IDs, keep the last retry, and explicitly handle incomplete output."""
    expected = {}
    prompts = set()
    for row in dataset:
        record_id = str(row.get("id", row.get("key", "")))
        if not record_id or record_id in expected:
            raise ValueError(f"Missing or duplicate dataset ID: {record_id!r}")
        for key in ("key", "prompt", "instruction_id_list", "kwargs"):
            if key not in row:
                raise ValueError(f"Dataset {record_id} is missing {key}")
        if row["prompt"] in prompts:
            raise ValueError("Official IFEval scorer requires unique prompts")
        prompts.add(row["prompt"])
        expected[record_id] = row
    latest = {}
    for row in predictions:
        if "id" not in row:
            raise ValueError("Every prediction needs an id")
        latest[str(row["id"])] = row
    extra = set(latest) - set(expected)
    missing = set(expected) - set(latest)
    if extra:
        raise ValueError(f"Prediction IDs absent from dataset: {sorted(extra)}")
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} predictions; use --allow-partial for an explicitly partial report")

    inputs, responses = [], []
    failures = truncated = 0
    selected = []
    for record_id, row in expected.items():
        if record_id not in latest:
            continue
        prediction = latest[record_id]
        is_truncated = prediction.get("finish_reason") == "length"
        failed = prediction.get("status") != "complete" or is_truncated
        failures += int(failed)
        truncated += int(is_truncated)
        response = prediction.get("response", "")
        if not isinstance(response, str):
            raise ValueError(f"Prediction {record_id} has a non-string response")
        if failed and not include_failed:
            response = ""
        inputs.append({key: row[key] for key in ("key", "prompt", "instruction_id_list", "kwargs")})
        responses.append({"prompt": row["prompt"], "response": response})
        selected.append(prediction)
    if not inputs:
        raise ValueError("No predictions match the dataset")
    info = {
        "dataset_samples": len(expected),
        "scored_samples": len(inputs),
        "missing_samples": len(missing),
        "raw_records": len(predictions),
        "unique_predictions": len(latest),
        "duplicate_policy": "last record per id",
        "failed_samples": failures,
        "truncated_samples": truncated,
        "response_policy": "saved response including failed/truncated outputs"
        if include_failed
        else "empty response for failed/truncated outputs",
        "status_counts": dict(Counter(str(row.get("status", "missing")) for row in selected)),
    }
    return inputs, responses, info


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="IFEval JSONL with original instruction IDs and kwargs"
    )
    parser.add_argument(
        "--scorer-dir", type=Path, required=True, help="Directory containing instruction_following_eval/"
    )
    parser.add_argument("--output-dir", type=Path, help="Default: <predictions directory>/official_scores")
    parser.add_argument(
        "--allow-partial", action="store_true", help="Score available IDs only and mark the report partial"
    )
    parser.add_argument(
        "--include-failed-responses",
        action="store_true",
        help="Check saved failed/truncated text instead of empty responses; matches historical FP16 reports",
    )
    args = parser.parse_args(argv)

    inputs, responses, info = align_predictions(
        read_jsonl(args.dataset),
        read_jsonl(args.predictions),
        allow_partial=args.allow_partial,
        include_failed=args.include_failed_responses,
    )
    package = args.scorer_dir.resolve() / "instruction_following_eval"
    if not (package / "evaluation_lib.py").is_file():
        raise FileNotFoundError(f"Missing official scorer: {package / 'evaluation_lib.py'}")
    sys.path.insert(0, str(package.parent))
    evaluation_lib = importlib.import_module("instruction_following_eval.evaluation_lib")
    if Path(evaluation_lib.__file__).resolve().parent != package:
        raise RuntimeError("Imported scorer differs from --scorer-dir")
    from langdetect import DetectorFactory

    random.seed(0)
    DetectorFactory.seed = 0
    output = args.output_dir or args.predictions.parent / "official_scores"
    output.mkdir(parents=True, exist_ok=True)
    inputs_path = output / "official_inputs.jsonl"
    responses_path = output / "official_responses.jsonl"
    write_jsonl(inputs_path, inputs)
    write_jsonl(responses_path, responses)
    official_inputs = evaluation_lib.read_prompt_list(str(inputs_path))
    official_responses = evaluation_lib.read_prompt_to_response_dict(str(responses_path))
    summary = {
        "benchmark": "IFEval",
        "status": "partial" if info["missing_samples"] or info["failed_samples"] else "complete",
        "score_scale": "percent",
        "prediction_file": str(args.predictions.resolve()),
        "prediction_sha256": sha256(args.predictions),
        "dataset_file": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "scorer_directory": str(package),
        "scorer_sources": {path.name: sha256(path) for path in sorted(package.glob("*.py"))},
        "seed": 0,
        **info,
        "metrics": {},
    }
    for mode in ("strict", "loose"):
        check = getattr(evaluation_lib, f"test_instruction_following_{mode}")
        checked = [check(inp, official_responses) for inp in official_inputs]
        evaluation_lib.write_outputs(str(output / f"eval_results_{mode}.jsonl"), checked)
        instruction_total = sum(len(row.follow_instruction_list) for row in checked)
        counts = (
            ("prompt", sum(row.follow_all_instructions for row in checked), len(checked)),
            ("instruction", sum(sum(row.follow_instruction_list) for row in checked), instruction_total),
        )
        for level, passed, total in counts:
            if not total:
                raise ValueError("IFEval samples must have instructions to score")
            name = f"{level}_{mode}"
            summary["metrics"][name] = {"passed": passed, "total": total, "score": 100 * passed / total}
            print(f"{name}: {passed}/{total} = {100 * passed / total:.4f}%")
    (output / "ifeval_scores.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# IFEval",
        "",
        f"Status: {summary['status']}; scored {info['scored_samples']}/{info['dataset_samples']} samples.",
        "",
        f"Response policy: {info['response_policy']}.",
        "",
        "| Metric | Passed / Total | Score (%) |",
        "|---|---:|---:|",
    ]
    for name, metric in summary["metrics"].items():
        lines.append(f"| {name} | {metric['passed']} / {metric['total']} | {metric['score']:.4f} |")
    (output / "ifeval_scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Scores: {output / 'ifeval_scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
