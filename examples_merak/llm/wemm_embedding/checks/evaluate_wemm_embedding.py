"""Evaluate WeMM embedding models against a reproducible text retrieval set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


try:
    from ..native_utils import embed_native_items, load_native_embedder
    from ..runtime_utils import load_hmonnx_embedder
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from native_utils import embed_native_items, load_native_embedder
    from runtime_utils import load_hmonnx_embedder


DEFAULT_CASES = [
    {
        "id": "food_zh",
        "query": "麻婆豆腐是怎样制作的？",
        "candidates": [
            "麻婆豆腐以嫩豆腐、豆瓣酱、花椒和辣椒烹制而成。",
            "巴黎是法国的首都，也是重要的旅游城市。",
            "CUDA 是用于 GPU 加速的并行计算平台。",
            "春天通常气温回升，植物开始生长。",
        ],
        "positive": 0,
    },
    {
        "id": "weather_en",
        "query": "What causes a rainbow to appear?",
        "candidates": [
            "A rainbow forms when sunlight is refracted, reflected, and dispersed in raindrops.",
            "The Pacific Ocean is the largest ocean on Earth.",
            "A compiler translates source code into another form of code.",
            "Regular exercise can improve cardiovascular health.",
        ],
        "positive": 0,
    },
    {
        "id": "python_zh",
        "query": "Python 中如何读取 JSON 文件？",
        "candidates": [
            "可以使用 json.load 打开文件并解析 JSON 数据。",
            "光合作用把光能转化为化学能。",
            "TCP 是一种面向连接的网络传输协议。",
            "向量数据库通常用于相似度检索。",
        ],
        "positive": 0,
    },
    {
        "id": "retrieval_en",
        "query": "Which data structure is commonly used for first-in-first-out processing?",
        "candidates": [
            "A queue processes items in first-in-first-out order.",
            "A stack processes items in last-in-first-out order.",
            "A hash table maps keys to values using a hash function.",
            "A tree organizes nodes in a hierarchical structure.",
        ],
        "positive": 0,
    },
    {
        "id": "science_zh",
        "query": "水在标准大气压下的沸点是多少？",
        "candidates": [
            "水在标准大气压下的沸点是摄氏一百度。",
            "铁的熔点约为摄氏一千五百三十八度。",
            "地球绕太阳公转一周约需要一年。",
            "声音需要介质才能传播。",
        ],
        "positive": 0,
    },
    {
        "id": "systems_en",
        "query": "Why is quantization useful for neural network inference?",
        "candidates": [
            "Quantization can reduce model memory use and improve inference efficiency.",
            "Backpropagation computes gradients for optimizing model parameters.",
            "A filesystem organizes persistent data into files and directories.",
            "A database index accelerates lookup of stored records.",
        ],
        "positive": 0,
    },
]


def load_cases(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_CASES
    cases = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            _validate_case(case, f"{path}:{line_number}")
            cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def _validate_case(case: dict[str, Any], location: str) -> None:
    required = {"query", "candidates", "positive"}
    missing = required - set(case)
    if missing:
        raise ValueError(f"{location} is missing fields: {sorted(missing)}")
    if not isinstance(case["query"], str) or not case["query"]:
        raise ValueError(f"{location} query must be a non-empty string")
    if not isinstance(case["candidates"], list) or not case["candidates"]:
        raise ValueError(f"{location} candidates must be a non-empty list")
    if not all(isinstance(candidate, str) and candidate for candidate in case["candidates"]):
        raise ValueError(f"{location} candidates must contain non-empty strings")
    if not isinstance(case["positive"], int) or not 0 <= case["positive"] < len(case["candidates"]):
        raise ValueError(f"{location} positive must be a valid candidate index")


def retrieval_metrics(scores: list[list[float]], positives: list[int]) -> dict[str, float]:
    if len(scores) != len(positives) or not scores:
        raise ValueError("scores and positives must have the same non-empty length")
    reciprocal_ranks = []
    ndcg_scores = []
    hits = []
    for row, positive in zip(scores, positives, strict=True):
        if not row or not 0 <= positive < len(row):
            raise ValueError("each score row must contain the positive index")
        ranking = sorted(range(len(row)), key=row.__getitem__, reverse=True)
        rank = ranking.index(positive) + 1
        hits.append(float(rank == 1))
        reciprocal_ranks.append(1.0 / rank)
        ndcg_scores.append(1.0 / math.log2(rank + 1) if rank <= 5 else 0.0)
    return {
        "recall@1": sum(hits) / len(hits),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "ndcg@5": sum(ndcg_scores) / len(ndcg_scores),
    }


def _mean_cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.shape != candidate.shape:
        raise ValueError(f"embedding shapes differ: {reference.shape} vs {candidate.shape}")
    return float(F.cosine_similarity(reference.float(), candidate.float(), dim=-1).mean())


class EmbeddingRunner:
    def __init__(self, model_dir: str, device: str, kind: str) -> None:
        self.kind = kind
        self.device = device
        if kind == "baseline":
            self.model, self.processor = load_native_embedder(model_dir, device=device)
        elif kind == "hmonnx":
            self.model, _ = load_hmonnx_embedder(model_dir, device)
        else:
            raise ValueError(f"Unsupported runner kind: {kind}")

    def embed(self, text: str) -> torch.Tensor:
        if self.kind == "hmonnx":
            return self.model.embed({"text": text}).detach().float().cpu()
        return embed_native_items(
            self.model,
            self.processor,
            [{"text": text}],
            device=self.device,
            batch_size=1,
        )


def evaluate_runner(runner: EmbeddingRunner, cases: list[dict[str, Any]]) -> dict[str, Any]:
    cache: dict[str, torch.Tensor] = {}

    def get_embedding(text: str) -> torch.Tensor:
        if text not in cache:
            cache[text] = runner.embed(text)
        return cache[text]

    score_rows = []
    details = []
    for case in cases:
        query_embedding = get_embedding(case["query"])
        candidate_embeddings = torch.cat([get_embedding(candidate) for candidate in case["candidates"]], dim=0)
        scores = torch.matmul(candidate_embeddings, query_embedding.squeeze(0))
        row = [float(score) for score in scores]
        score_rows.append(row)
        ranking = sorted(range(len(row)), key=row.__getitem__, reverse=True)
        details.append(
            {
                "id": case.get("id", str(len(details))),
                "positive": case["positive"],
                "ranking": ranking,
                "scores": row,
            }
        )
    return {
        "metrics": retrieval_metrics(score_rows, [case["positive"] for case in cases]),
        "details": details,
        "embedding_count": len(cache),
        "embeddings": cache,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model-dir", required=True)
    parser.add_argument("--quant-meta", required=True)
    parser.add_argument("--cases", help="JSONL with query, candidates, and positive fields")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    baseline = evaluate_runner(EmbeddingRunner(args.baseline_model_dir, args.device, "baseline"), cases)
    quant = evaluate_runner(EmbeddingRunner(args.quant_meta, args.device, "hmonnx"), cases)
    shared_texts = sorted(set(baseline["embeddings"]) & set(quant["embeddings"]))
    agreement = torch.tensor(
        [
            float(
                F.cosine_similarity(
                    baseline["embeddings"][text].float(),
                    quant["embeddings"][text].float(),
                    dim=-1,
                ).mean()
            )
            for text in shared_texts
        ]
    )

    output = {
        "baseline_model_dir": args.baseline_model_dir,
        "quant_meta": args.quant_meta,
        "case_count": len(cases),
        "metrics": {
            "baseline": baseline["metrics"],
            "quantized": quant["metrics"],
            "quantized_vs_baseline": {
                "mean_cosine": float(agreement.mean()),
                "min_cosine": float(agreement.min()),
                "shared_text_count": len(shared_texts),
            },
        },
        "details": {"quantized": quant["details"]},
        "note": "This local suite is a reproducible smoke benchmark, not the official MMEB-v2/v3 score.",
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
