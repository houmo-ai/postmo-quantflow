"""Dataset, prompt, scoring and report helpers shared by the vLLM client.

Extracted from spark_tokens_xh2_eval.py; inference and model imports are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data/evaluation/spark_x2_5"


CATEGORY_BY_BENCHMARK = {
    "BFCL-V4": "Agent",
    "SciCode": "Code",
    "AIME-2026": "Math",
    "IFEval": "General & Knowledge",
    "AA-LCR": "General & Knowledge",
    "GPQA-Diamond": "General & Knowledge",
}


CATEGORY_ORDER = ("Agent", "Code", "Math", "General & Knowledge")


OFFICIAL_SCORER_BENCHMARKS = {"IFEval", "BFCL-V4", "SciCode"}


REFERENCE_FIELDS = ("answer", "target", "reference", "label", "gold", "expected")


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def load_suite(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation suite not found: {path}")
    with path.open(encoding="utf-8") as handle:
        suite = yaml.safe_load(handle)
    if not isinstance(suite, dict) or not isinstance(suite.get("benchmarks"), list):
        raise ValueError(f"{path} must contain a benchmarks list")
    for index, benchmark in enumerate(suite["benchmarks"], start=1):
        if not isinstance(benchmark, dict) or not benchmark.get("name"):
            raise ValueError(f"Benchmark entry {index} in {path} needs a name")
    return suite


def parse_assignments(values: Iterable[str], value_type: type = str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected NAME=VALUE, got: {item}")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name or not value.strip():
            raise ValueError(f"Expected NAME=VALUE, got: {item}")
        parsed[name.casefold()] = value_type(value.strip())
    return parsed


def selected_benchmark_names(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    names = {name.strip().casefold() for value in values for name in value.split(",") if name.strip()}
    return names or None


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")


def resolve_dataset_path(
    benchmark: dict[str, Any], suite_path: Path, data_dir: Path, overrides: dict[str, str]
) -> Path | None:
    name = str(benchmark["name"])
    override = overrides.get(name.casefold())
    if override:
        return Path(override).expanduser().resolve()

    dataset = benchmark.get("dataset") or {}
    if isinstance(dataset, dict):
        configured_path = dataset.get("path") or dataset.get("file")
        if configured_path:
            path = Path(str(configured_path)).expanduser()
            if not path.is_absolute():
                path = suite_path.parent / path
            return path.resolve()

    stems = (name, slugify(name), name.casefold().replace("-", "_"))
    for stem in dict.fromkeys(stems):
        for suffix in (".jsonl", ".json"):
            candidate = data_dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate.resolve()
    return None


def read_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.casefold() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("records", "data", "examples", "items"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        records = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path} must contain JSON objects")
    if limit is not None:
        records = records[:limit]
    result = []
    for index, record in enumerate(records, start=1):
        record = dict(record)
        record.setdefault("id", str(index))
        result.append(record)
    return result


def get_prompt_source(record: dict[str, Any]) -> dict[str, Any]:
    raw_data = record.get("raw_data")
    if isinstance(raw_data, dict):
        return {**raw_data, **record}
    return record


def stable_shuffle(values: list[str], seed: int, record_id: str) -> list[str]:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    values = list(values)
    rng.shuffle(values)
    return values


def prepare_record(record: dict[str, Any], benchmark_name: str, seed: int) -> dict[str, Any]:
    prepared = dict(record)
    source = get_prompt_source(prepared)

    # Common GPQA release schema. Choices are shuffled reproducibly to avoid a
    # fixed position for the correct answer.
    if benchmark_name.casefold() == "gpqa-diamond".casefold() and "Question" in source:
        correct = source.get("Correct Answer")
        choices = [correct]
        choices.extend(source.get(f"Incorrect Answer {index}") for index in range(1, 4))
        choices = [str(choice) for choice in choices if choice is not None]
        choices = stable_shuffle(choices, seed, str(prepared["id"]))
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        prepared["prompt"] = (
            str(source["Question"])
            + "\n\n"
            + "\n".join(f"{letters[index]}. {choice}" for index, choice in enumerate(choices))
        )
        prepared["prompt"] += "\n\nAnswer with only the letter of the correct option."
        prepared["choices"] = choices
        prepared["expected"] = letters[choices.index(str(correct))]
        return prepared

    choices = source.get("choices") or source.get("options")
    question = source.get("question") or source.get("problem") or source.get("problem_statement")
    if question is not None and choices:
        if isinstance(choices, dict):
            choice_pairs = [(str(key), str(value)) for key, value in choices.items()]
        else:
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            choice_pairs = [(letters[index], str(value)) for index, value in enumerate(choices)]
        prepared["prompt"] = str(question) + "\n\n" + "\n".join(f"{key}. {value}" for key, value in choice_pairs)
        prepared["prompt"] += "\n\nAnswer with only the letter of the correct option."
        prepared["choices"] = [value for _, value in choice_pairs]
    elif question is not None and not any(source.get(field) is not None for field in ("prompt", "query", "text")):
        prepared["prompt"] = str(question)
    return prepared


def render_prompt(record: dict[str, Any], tokenizer: Any, enable_thinking: bool) -> str:
    source = get_prompt_source(record)
    messages = source.get("messages")
    metadata = source.get("metadata")
    metadata_prompt = metadata.get("prompt") if isinstance(metadata, dict) else None
    if metadata_prompt is not None:
        messages = [{"role": "user", "content": str(metadata_prompt)}]
    if messages is None:
        for field in ("prompt", "query", "text", "question", "problem", "problem_statement"):
            if source.get(field) is not None:
                messages = [{"role": "user", "content": str(source[field])}]
                break
    if messages is None:
        raise ValueError("Record needs messages, prompt, query, text, question, problem, or problem_statement")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat_template")

    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    tools = source.get("tools") or source.get("functions")
    if tools:
        template_kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **template_kwargs)


def find_reference(record: dict[str, Any]) -> Any:
    source = get_prompt_source(record)
    for field in REFERENCE_FIELDS:
        if source.get(field) is not None:
            return source[field]
    return None


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return text.strip()


def normalize_answer(value: Any) -> str:
    text = strip_thinking(str(value)).strip()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]
    text = re.sub(r"^(?:final\s+answer|answer|答案)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("`$ ").rstrip(".。")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def extract_choice(text: str, choices: list[str] | None = None) -> str | None:
    content = strip_thinking(text)
    patterns = (
        r"(?:final\s+answer|answer|答案)\s*(?:is|为)?\s*[:：]?\s*\(?([A-Z])\)?",
        r"\b\(?([A-Z])\)?\s*$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, content, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    if choices:
        normalized = normalize_answer(content)
        for index, choice in enumerate(choices):
            if normalize_answer(choice) == normalized:
                return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index]
    return None


def score_prediction(benchmark_name: str, record: dict[str, Any], response: str) -> tuple[Any, bool | None]:
    reference = find_reference(record)
    if reference is None or benchmark_name in OFFICIAL_SCORER_BENCHMARKS:
        return None, None

    choices = record.get("choices")
    normalized_reference = str(reference).strip()
    if choices or re.fullmatch(r"[A-Za-z]", normalized_reference):
        predicted = extract_choice(response, choices if isinstance(choices, list) else None)
        expected = normalized_reference.upper()
        if expected.isdigit() and choices:
            expected = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[int(expected)]
        return predicted, predicted == expected

    predicted = normalize_answer(response)
    expected = normalize_answer(reference)
    return predicted, predicted == expected


def resolve_generation_config(
    args: argparse.Namespace, suite: dict[str, Any], benchmark: dict[str, Any]
) -> dict[str, Any]:
    global_inference = suite.get("global_inference") or {}
    official = suite.get("official_compatible_run") or {}
    official_enabled = bool(official.get("enabled"))
    official_decode = official.get("decode") or {}
    benchmark_generation = benchmark.get("generation") or {}

    do_sample = args.do_sample if args.do_sample is not None else official_enabled
    max_new_tokens = args.max_new_tokens or int(benchmark_generation.get("max_new_tokens", 512))
    generation: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation.update(
            temperature=(
                args.temperature
                if args.temperature is not None
                else official_decode.get("temperature", global_inference.get("temperature", 1.0))
            ),
            top_p=(
                args.top_p
                if args.top_p is not None
                else official_decode.get("top_p", global_inference.get("top_p", 0.95))
            ),
            top_k=(
                args.top_k
                if args.top_k is not None
                else official_decode.get("top_k", global_inference.get("top_k", -1))
            ),
            repetition_penalty=args.repetition_penalty,
        )
    # Both backends use Transformers generation semantics.
    if generation.get("top_k", 0) < 0:
        generation["top_k"] = 0
    generation["repetition_penalty"] = args.repetition_penalty
    generation["presence_penalty"] = args.presence_penalty
    return generation


def read_completed_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    # Iterate physical lines so Unicode separators inside JSON strings survive.
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "complete" and record.get("finish_reason") != "length":
                completed[str(record["id"])] = record
    return completed


def format_score(result: dict[str, Any]) -> str:
    score = result.get("score")
    if score is None:
        return "—"
    if isinstance(score, float) and math.isfinite(score):
        return f"{score:.1f}"
    return str(score)


def write_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    results = summary["results"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["category"]].append(result)

    model_name = summary["model_name"]
    lines = [
        "# Benchmarks",
        "",
        f"Model: `{summary['model']}`  ",
        f"Precision: `{summary['precision']}`",
        "",
        f"| Benchmark | {model_name} | Status | Samples |",
        "|---|---:|---|---:|",
    ]
    ordered_categories = list(CATEGORY_ORDER)
    ordered_categories.extend(category for category in grouped if category not in CATEGORY_ORDER)
    for category in ordered_categories:
        if not grouped.get(category):
            continue
        lines.append(f"| **{category}** |  |  |  |")
        for result in grouped[category]:
            lines.append(
                f"| {result['benchmark']} | {format_score(result)} | {result['status']} | {result['num_samples']} |"
            )
    lines.extend(
        [
            "",
            "Scores are percentages. `—` means that no valid score is available.",
            (
                "Tasks marked `generated_only` require their official benchmark scorer; "
                "use `--score NAME=VALUE` to merge it."
            ),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
