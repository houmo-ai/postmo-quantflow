"""Evaluate a benchmark suite through an existing vLLM chat service (no local weights)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import eval_common as common  # noqa: E402


class Client:
    def __init__(self, base_url, timeout=3600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if os.environ.get("VLLM_API_KEY"):
            self.headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]

    def open(self, path, body=None):
        url = self.base_url + path
        if path == "/tokenize":
            url = self.base_url.removesuffix("/v1") + path
        request = urllib.request.Request(
            url, data=None if body is None else json.dumps(body).encode(), headers=self.headers
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc

    def json(self, path, body=None):
        with self.open(path, body) as response:
            return json.load(response)

    def generate(self, body, label):
        content, reasoning, calls = [], [], {}
        finish, usage, done = None, {}, False
        started = last = time.monotonic()
        chunks = 0
        with self.open(
            "/chat/completions",
            {
                **body,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as stream:
            for line in stream:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning") or delta.get("reasoning_content") or "")
                    for call in delta.get("tool_calls") or []:
                        item = calls.setdefault(
                            call["index"], {"type": "function", "function": {"name": "", "arguments": ""}}
                        )
                        if call.get("id"):
                            item["id"] = call["id"]
                        for key in ("name", "arguments"):
                            item["function"][key] += (call.get("function") or {}).get(key) or ""
                    finish = choice.get("finish_reason") or finish
                chunks += 1
                if time.monotonic() - last >= 30:
                    print(f"{label}: streaming chunks={chunks} elapsed={time.monotonic() - started:.0f}s", flush=True)
                    last = time.monotonic()
        if not done or finish is None:
            raise RuntimeError("Incomplete SSE response: missing finish_reason or [DONE]")
        text = "".join(content)
        thought = "".join(reasoning)
        if "</think>" in text:
            prefix, text = text.rsplit("</think>", 1)
            thought += prefix.removeprefix("<think>")
        return dict(
            content=text.strip(),
            reasoning=thought.strip(),
            tool_calls=[calls[k] for k in sorted(calls)],
            finish_reason=finish,
            usage=usage,
        )


class ChatRequest:
    """Reuse the common prompt selection without loading a second tokenizer/template."""

    chat_template = True

    def apply_chat_template(self, messages, **kwargs):
        result = {"messages": messages, "chat_template_kwargs": {"enable_thinking": kwargs["enable_thinking"]}}
        if kwargs.get("tools"):
            # BFCL stores bare function definitions; only add the API envelope.
            # Keep the benchmark's original names, types and parameter schema.
            result["tools"] = [
                tool if "function" in tool else {"type": "function", "function": tool} for tool in kwargs["tools"]
            ]
        return result


def request_body(record, name, args, generation, thinking):
    prepared = common.prepare_record(record, name, args.seed)
    body = common.render_prompt(prepared, ChatRequest(), thinking)
    seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{name}:{record['id']}".encode()).digest()[:4], "big")
    body.update(
        model=args.model,
        max_tokens=generation["max_new_tokens"],
        seed=seed,
        temperature=generation.get("temperature", 0.0),
        top_p=generation.get("top_p", 1.0),
        top_k=generation.get("top_k", 0) or -1,
        repetition_penalty=generation["repetition_penalty"],
        presence_penalty=generation["presence_penalty"],
    )
    return prepared, body


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    p.add_argument("--model", default="spark2_5", help="Served model ID, not a filesystem path.")
    p.add_argument("--suite", type=Path, default=SCRIPT_DIR / "mini_eval_vllm_128k.yaml")
    p.add_argument("--data-dir", type=Path, default=common.DEFAULT_DATA_DIR)
    p.add_argument("--dataset", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--benchmarks", nargs="+")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--limit", type=common.positive_int)
    p.add_argument("--max-new-tokens", type=common.positive_int)
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-p", type=float)
    p.add_argument("--top-k", type=int)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--timeout", type=common.positive_int, default=3600, help="Socket idle timeout in seconds.")
    p.add_argument("--score", action="append", default=[], metavar="NAME=PERCENT")
    p.add_argument("--ifeval-scorer-dir", type=Path, help="Local Google instruction_following_eval package parent.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Check service, datasets, and budgets without generation.")
    return p


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def normalize_resume_metadata(metadata):
    """Compare effective budgets across legacy CLI overrides and YAML settings."""
    normalized = json.loads(json.dumps(metadata))
    normalized.pop("precision", None)  # Legacy report label, not an inference setting.
    normalized.setdefault("generation_budget_policy", "cap_to_remaining_context")
    normalized.setdefault("context_length", normalized.get("served_model", {}).get("max_model_len"))
    suite = normalized.get("suite", {})
    suite.pop("suite", None)  # Suite title is a report label.
    if "benchmarks" in suite:
        suite["benchmarks"] = [
            b for b in suite["benchmarks"] if b["name"] in normalized.get("generation", {})
        ]
    for benchmark in normalized.get("suite", {}).get("benchmarks", []):
        generation = normalized.get("generation", {}).get(benchmark["name"])
        if generation is not None:
            benchmark.setdefault("generation", {})["max_new_tokens"] = generation["max_new_tokens"]
    return normalized


def summarize(benchmark, predictions, total, output, generation):
    scored = [p for p in predictions if p.get("correct") is not None]
    failed = sum(p["status"] == "failed" for p in predictions)
    correct = sum(p["correct"] is True for p in scored)
    status = (
        "running"
        if len(predictions) < total
        else "failed"
        if failed == total
        else "partial"
        if failed
        else "complete"
        if scored
        else "generated_only"
    )
    return dict(
        benchmark=benchmark["name"],
        category=common.CATEGORY_BY_BENCHMARK.get(benchmark["name"], "Other"),
        status=status,
        score=round(100 * correct / len(scored), 4) if scored else None,
        num_samples=total,
        processed_samples=len(predictions),
        failed=failed,
        correct=correct if scored else None,
        scored_samples=len(scored),
        metrics=benchmark.get("metrics", []),
        generation=generation,
        scoring="external_official_required"
        if benchmark["name"] in common.OFFICIAL_SCORER_BENCHMARKS
        else "choice_accuracy"
        if benchmark["name"] == "GPQA-Diamond"
        else "normalized_exact_match",
        raw_output_path=str(output),
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.suite = args.suite.expanduser().resolve()
    args.output_dir = (
        args.output_dir or common.REPO_ROOT / "work_dirs" / f"eval_{common.slugify(args.model)}_vllm"
    ).resolve()
    suite = common.load_suite(args.suite)
    requested = common.selected_benchmark_names(args.benchmarks)
    known = {b["name"].casefold() for b in suite["benchmarks"]}
    overrides = common.parse_assignments(args.dataset)
    scores = common.parse_assignments(args.score, float)
    for names in (requested or set(), overrides, scores):
        if set(names) - known:
            raise ValueError(f"Unknown benchmarks: {set(names) - known}")
    if any(not 0 <= n <= 100 for n in scores.values()):
        raise ValueError("Official scores must be between 0 and 100")
    benchmarks = [b for b in suite["benchmarks"] if not requested or b["name"].casefold() in requested]
    if not benchmarks:
        raise ValueError("No benchmarks selected")
    client = Client(args.base_url, args.timeout)
    models = client.json("/models")["data"]
    model = next((m for m in models if m["id"] == args.model), None)
    if model is None:
        raise ValueError(f"Model {args.model} not served; available: {[m['id'] for m in models]}")
    context = model.get("max_model_len")
    if not context:
        context = client.json("/tokenize", {"model": args.model, "prompt": "Hi"})["max_model_len"]
    paths, datasets, generations = {}, {}, {}
    for b in benchmarks:
        name = b["name"]
        path = common.resolve_dataset_path(b, args.suite, args.data_dir, overrides)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing dataset: {name}")
        rows = common.read_records(path, args.limit)
        ids = [str(r["id"]) for r in rows]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{name}: dataset must be nonempty with unique IDs")
        generation = common.resolve_generation_config(args, suite, b)
        if generation["max_new_tokens"] <= 0:
            raise ValueError(f"{name}: max_new_tokens must be positive")
        paths[name], datasets[name], generations[name] = path, rows, generation
    thinking = (
        args.thinking
        if args.thinking is not None
        else suite.get("global_inference", {}).get("mode", "thinking") == "thinking"
    )
    plan = dict(
        base_url=client.base_url,
        model=args.model,
        context_length=context,
        output_dir=str(args.output_dir),
        benchmarks=[
            dict(name=b["name"], num_samples=len(datasets[b["name"]]), generation=generations[b["name"]])
            for b in benchmarks
        ],
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    if (
        args.ifeval_scorer_dir
        and not (args.ifeval_scorer_dir / "instruction_following_eval/evaluation_lib.py").is_file()
    ):
        raise FileNotFoundError("Missing IFEval official scorer")
    metadata = dict(
        backend="vllm",
        generation_budget_policy="cap_to_remaining_context",
        context_length=context,
        base_url=client.base_url,
        model=args.model,
        served_model={k: model.get(k) for k in ("id", "root", "max_model_len")},
        seed=args.seed,
        thinking=thinking,
        limit=args.limit,
        suite=suite,
        generation=generations,
        datasets={n: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n, p in paths.items()},
    )
    config = args.output_dir / "run_config.json"
    if args.resume:
        if config.exists():
            saved_metadata = json.loads(config.read_text())
            if normalize_resume_metadata(saved_metadata) != normalize_resume_metadata(metadata):
                raise ValueError("Resume configuration changed; use a new --output-dir")
        elif list(args.output_dir.glob("*/predictions.jsonl")):
            raise ValueError("Cannot resume predictions without run_config.json")
    elif config.exists() or list(args.output_dir.glob("*/predictions.jsonl")):
        raise ValueError("Output already contains a run; use --resume or a new --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(config, metadata)
    summary = dict(
        suite=suite.get("suite", {}).get("name", args.suite.stem),
        model=args.model,
        model_name=args.model,
        precision="unknown",
        backend="vllm",
        results=[],
        status="running",
        notes=[
            "AA-LCR uses diagnostic exact match, not its official judge.",
            "BFCL/SciCode produce single-response predictions; official tool execution/code harness is external.",
            "Quantization comparisons require separately evaluated reference models.",
        ],
    )

    def save():
        atomic_json(args.output_dir / "results.json", summary)
        common.write_markdown_report(summary, args.output_dir / "benchmark.md")

    save()
    for benchmark in benchmarks:
        name = benchmark["name"]
        generation = generations[name]
        output = args.output_dir / common.slugify(name) / "predictions.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        completed = common.read_completed_predictions(output) if args.resume else {}
        predictions = []
        summary["results"].append(summarize(benchmark, predictions, len(datasets[name]), output, generation))
        with output.open("a" if args.resume else "w", encoding="utf-8") as handle:
            for index, raw in enumerate(datasets[name], 1):
                rid = str(raw["id"])
                if rid in completed:
                    prediction = completed[rid]
                else:
                    started = time.monotonic()
                    prediction = dict(id=rid, benchmark=name, status="failed", raw_data=raw, generation=generation)
                    print(f"{name} {index}/{len(datasets[name])} id={rid} starting", flush=True)
                    try:
                        prepared, body = request_body(raw, name, args, generation, thinking)
                        token_body = {
                            k: body[k] for k in ("model", "messages", "tools", "chat_template_kwargs") if k in body
                        }
                        tokens = client.json("/tokenize", {**token_body, "add_generation_prompt": True})["count"]
                        prediction.update(
                            messages=body["messages"],
                            tools=body.get("tools"),
                            input_tokens=tokens,
                            reference=common.find_reference(prepared),
                            request=body,
                        )
                        remaining = context - tokens
                        if remaining <= 0:
                            raise ValueError(
                                f"Input ({tokens}) leaves no room for output in context {context}; "
                                "input not truncated"
                            )
                        body["max_tokens"] = min(generation["max_new_tokens"], remaining)
                        prediction["generation"] = {**generation, "max_new_tokens": body["max_tokens"]}
                        response = client.generate(body, f"{name}/{rid}")
                        prediction.update(
                            response=response["content"],
                            thinking=response["reasoning"],
                            raw_response=response,
                            tool_calls=response["tool_calls"],
                            output_tokens=response["usage"].get("completion_tokens"),
                            finish_reason=response["finish_reason"],
                        )
                        if response["finish_reason"] not in ("stop", "tool_calls"):
                            raise ValueError(
                                f"Generation incomplete: finish_reason={response['finish_reason']}; not scored"
                            )
                        predicted, correct = common.score_prediction(name, prepared, response["content"])
                        prediction.update(status="complete", predicted=predicted, correct=correct)
                    except Exception as exc:
                        prediction["error"] = f"{type(exc).__name__}: {exc}"
                        print(f"{name}/{rid}: {prediction['error']}", flush=True)
                    prediction["latency_seconds"] = round(time.monotonic() - started, 6)
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    handle.flush()
                predictions.append(prediction)
                result = summarize(benchmark, predictions, len(datasets[name]), output, generation)
                result.update(model=args.model, precision="unknown")
                if name.casefold() in scores and len(predictions) == len(datasets[name]):
                    result.update(
                        score=scores[name.casefold()],
                        scoring="external_official",
                        status="partial" if result["failed"] else "complete_official",
                    )
                summary["results"][-1] = result
                save()
                print(f"{name} {index}/{len(datasets[name])}: {prediction['status']}", flush=True)
        if name == "IFEval" and args.ifeval_scorer_dir:
            import score_spark_tokens_ifeval as scorer

            try:
                scorer_args = [
                    "--predictions",
                    str(output),
                    "--dataset",
                    str(paths[name]),
                    "--scorer-dir",
                    str(args.ifeval_scorer_dir),
                ]
                if args.limit:
                    scorer_args.append("--allow-partial")
                scorer.main(scorer_args)
                official = json.loads((output.parent / "official_scores/ifeval_scores.json").read_text())
                summary["results"][-1].update(
                    score=official["metrics"]["prompt_strict"]["score"],
                    metric_scores={k + "_accuracy": v["score"] for k, v in official["metrics"].items()},
                    scoring="google_ifeval",
                    status="partial" if official["status"] == "partial" else "complete_official",
                )
            except Exception as exc:
                summary["results"][-1]["scorer_error"] = f"{type(exc).__name__}: {exc}"
                print(f"IFEval scorer failed: {exc}", flush=True)
            save()
    summary["status"] = "finished_with_failures" if any(r["failed"] for r in summary["results"]) else "finished"
    save()
    return int(any(r["failed"] or r.get("scorer_error") for r in summary["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
