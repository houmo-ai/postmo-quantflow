"""Verify paired MMEB-v3 query/candidate coverage and compare official scores."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def cosine_summary(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(f"Embedding shapes differ: {left.shape} vs {right.shape}")
    left = left.astype(np.float32, copy=False)
    right = right.astype(np.float32, copy=False)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    if not np.all(np.isfinite(denominator)) or np.any(denominator == 0):
        raise ValueError("Non-finite or zero-norm embeddings")
    cosine = np.sum(left * right, axis=1) / denominator
    return {
        "mean": float(np.mean(cosine)),
        "median": float(np.median(cosine)),
        "p01": float(np.quantile(cosine, 0.01)),
        "min": float(np.min(cosine)),
    }


def compare(hf_dir: Path, hmonnx_dir: Path, task: str) -> dict:
    hf_protocol = json.loads((hf_dir / "protocol.json").read_text())
    hmonnx_protocol = json.loads((hmonnx_dir / "protocol.json").read_text())
    hf_image_preprocessing = hf_protocol.get("image_preprocessing")
    hmonnx_image_preprocessing = hmonnx_protocol.get("image_preprocessing")
    if hf_image_preprocessing != hmonnx_image_preprocessing:
        raise ValueError(
            f"{task}: image preprocessing differs between backends: "
            f"hf={hf_image_preprocessing}, hmonnx={hmonnx_image_preprocessing}"
        )
    if not isinstance(hf_image_preprocessing, dict) or hf_image_preprocessing.get("mode") != "dynamic":
        raise ValueError(f"{task}: expected a recorded dynamic image preprocessing protocol")
    info_name = f"{task}_info.jsonl"
    hf_info = [json.loads(line) for line in (hf_dir / info_name).read_text().splitlines()]
    hmonnx_info = [json.loads(line) for line in (hmonnx_dir / info_name).read_text().splitlines()]
    if hf_info != hmonnx_info:
        raise ValueError(f"{task}: query order or ground truth differs between backends")
    with (hf_dir / f"{task}_qry").open("rb") as f:
        hf_queries = pickle.load(f)
    with (hmonnx_dir / f"{task}_qry").open("rb") as f:
        hmonnx_queries = pickle.load(f)
    if len(hf_queries) != len(hf_info) or len(hmonnx_queries) != len(hmonnx_info):
        raise ValueError(f"{task}: query vector count differs from metadata count")
    candidate_name = f"{task}_tgt"
    with (hf_dir / candidate_name).open("rb") as f:
        hf_candidates = pickle.load(f)
    with (hmonnx_dir / candidate_name).open("rb") as f:
        hmonnx_candidates = pickle.load(f)
    if list(hf_candidates) != list(hmonnx_candidates):
        raise ValueError(f"{task}: candidate IDs or order differs between backends")
    hf_cand_vectors = np.stack(list(hf_candidates.values()))
    hmonnx_cand_vectors = np.stack(list(hmonnx_candidates.values()))
    with (hf_dir / f"{task}_score.json").open() as f:
        hf_scores = json.load(f)
    with (hmonnx_dir / f"{task}_score.json").open() as f:
        hmonnx_scores = json.load(f)
    hf_predictions = [json.loads(line) for line in (hf_dir / f"{task}_pred.jsonl").read_text().splitlines()]
    hmonnx_predictions = [json.loads(line) for line in (hmonnx_dir / f"{task}_pred.jsonl").read_text().splitlines()]
    if len(hf_predictions) != len(hf_info) or len(hmonnx_predictions) != len(hmonnx_info):
        raise ValueError(f"{task}: prediction count differs from query count")
    paired_top1 = {"both_correct": 0, "hf_only": 0, "hmonnx_only": 0, "both_wrong": 0}
    for hf_row, hmonnx_row in zip(hf_predictions, hmonnx_predictions, strict=True):
        if hf_row["label"] != hmonnx_row["label"]:
            raise ValueError(f"{task}: prediction labels differ between backends")
        hf_correct = bool(hf_row["prediction"] and hf_row["prediction"][0] in hf_row["label"])
        hmonnx_correct = bool(hmonnx_row["prediction"] and hmonnx_row["prediction"][0] in hmonnx_row["label"])
        key = (
            "both_correct"
            if hf_correct and hmonnx_correct
            else "hf_only"
            if hf_correct
            else "hmonnx_only"
            if hmonnx_correct
            else "both_wrong"
        )
        paired_top1[key] += 1
    if abs((paired_top1["both_correct"] + paired_top1["hf_only"]) / len(hf_info) - hf_scores["hit@1"]) > 1e-6:
        raise ValueError(f"{task}: HF per-query top-1 does not reproduce the official score")
    if (
        abs((paired_top1["both_correct"] + paired_top1["hmonnx_only"]) / len(hmonnx_info) - hmonnx_scores["hit@1"])
        > 1e-6
    ):
        raise ValueError(f"{task}: HMONNX per-query top-1 does not reproduce the official score")
    return {
        "task": task,
        "queries": len(hf_info),
        "candidates": len(hf_candidates),
        "image_preprocessing": hf_image_preprocessing,
        "query_cosine": cosine_summary(hf_queries, hmonnx_queries),
        "candidate_cosine": cosine_summary(hf_cand_vectors, hmonnx_cand_vectors),
        "paired_top1": paired_top1,
        "hf": hf_scores,
        "hmonnx": hmonnx_scores,
        "delta_hmonnx_minus_hf": {
            key: hmonnx_scores[key] - value
            for key, value in hf_scores.items()
            if isinstance(value, (int, float)) and isinstance(hmonnx_scores.get(key), (int, float))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_dir", type=Path)
    parser.add_argument("hmonnx_dir", type=Path)
    parser.add_argument("tasks", nargs="+")
    args = parser.parse_args()
    print(
        json.dumps([compare(args.hf_dir, args.hmonnx_dir, task) for task in args.tasks], indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
