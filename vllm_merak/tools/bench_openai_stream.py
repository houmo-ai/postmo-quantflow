#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible streaming completions endpoint as JSONL."""

import argparse
from collections.abc import Callable
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, TextIO
from urllib.request import Request, urlopen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18036/v1")
    parser.add_argument("--model", default="qwen3_6_35b_a3b_merak")
    tokenizer_group = parser.add_mutually_exclusive_group(required=True)
    tokenizer_group.add_argument("--tokenizer")
    tokenizer_group.add_argument("--model-dir", type=Path)
    parser.add_argument("--target-prompt-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--unit", default="测试 ")
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser


def resolve_tokenizer_source(tokenizer: str | None, model_dir: Path | None) -> str:
    if tokenizer:
        return tokenizer
    if model_dir is not None:
        return str(model_dir / "hf_config")
    raise ValueError("either --tokenizer or --model-dir is required")


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_prompt(tokenizer: Any, target_tokens: int, unit: str) -> tuple[str, int]:
    """Build the closest repeated prompt to the requested local token count."""
    suffix = "\n请只输出一段简短总结。"

    def candidate(repeats: int) -> tuple[str, int]:
        prompt = unit * repeats + suffix
        return prompt, _token_count(tokenizer, prompt)

    low = 0
    high = max(1, target_tokens // max(1, _token_count(tokenizer, unit)))
    _, high_count = candidate(high)
    while high_count < target_tokens:
        low = high
        high *= 2
        _, high_count = candidate(high)

    while low + 1 < high:
        middle = (low + high) // 2
        _, middle_count = candidate(middle)
        if middle_count < target_tokens:
            low = middle
        else:
            high = middle

    choices = (candidate(low), candidate(high))
    prompt, actual_tokens = min(choices, key=lambda item: abs(item[1] - target_tokens))
    if actual_tokens != target_tokens:
        raise ValueError(
            f"cannot build exactly {target_tokens} prompt tokens; "
            f"nearest candidate has {actual_tokens}"
        )
    return prompt, actual_tokens


def calculate_metrics(
    *,
    started_at: float,
    first_token_at: float,
    last_token_at: float,
    finished_at: float,
    completion_tokens: int,
) -> dict[str, float | int]:
    decode_tokens = max(0, completion_tokens - 1)
    decode_time = max(0.0, last_token_at - first_token_at)
    decode_rate = decode_tokens / decode_time if decode_time > 0 else 0.0
    return {
        "ttft_s": first_token_at - started_at,
        "total_s": finished_at - started_at,
        "decode_tokens": decode_tokens,
        "decode_time_s": decode_time,
        "decode_tok_s": decode_rate,
    }


def build_request_body(*, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one measured run is required")

    def stats(name: str) -> dict[str, float]:
        values = [float(run[name]) for run in runs]
        return {"median": statistics.median(values), "p95": _percentile(values, 0.95)}

    def consistent_int(name: str) -> int:
        values = {int(run[name]) for run in runs}
        if len(values) != 1:
            raise ValueError(f"measured runs disagree on {name}: {sorted(values)}")
        return values.pop()

    return {
        "type": "summary",
        "repeat": len(runs),
        "target_prompt_tokens": consistent_int("target_prompt_tokens"),
        "prompt_tokens_local": consistent_int("prompt_tokens_local"),
        "prompt_tokens_usage": consistent_int("prompt_tokens_usage"),
        "ttft_s": stats("ttft_s"),
        "decode_tok_s": stats("decode_tok_s"),
    }


def benchmark_once(
    *,
    base_url: str,
    model: str,
    tokenizer: Any,
    prompt: str,
    prompt_tokens_local: int,
    target_prompt_tokens: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    body = build_request_body(model=model, prompt=prompt, max_tokens=max_tokens)
    request = Request(
        base_url.rstrip("/") + "/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_at = time.perf_counter()
    first_token_at = None
    last_token_at = None
    finish_reason = None
    saw_done = False
    chunks = 0
    text_parts: list[str] = []
    usage = None
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - caller selects endpoint
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                saw_done = True
                break
            event = json.loads(data)
            if event.get("error") is not None:
                error = event["error"]
                detail = (
                    error.get("message", error) if isinstance(error, dict) else error
                )
                raise RuntimeError(f"server error in SSE stream: {detail}")
            if event.get("usage") is not None:
                usage = event["usage"]
            choices = event.get("choices") or []
            event_finish_reason = choices[0].get("finish_reason") if choices else None
            if event_finish_reason is not None:
                if finish_reason is not None and event_finish_reason != finish_reason:
                    raise RuntimeError(
                        "conflicting finish reasons in SSE stream: "
                        f"{finish_reason!r} then {event_finish_reason!r}"
                    )
                finish_reason = event_finish_reason
            delta = choices[0].get("text") if choices else None
            if delta:
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                last_token_at = now
                chunks += 1
                text_parts.append(delta)
    finished_at = time.perf_counter()

    if not saw_done:
        raise RuntimeError("incomplete SSE stream: missing [DONE] marker")
    if usage is None:
        raise RuntimeError("incomplete SSE stream: missing usage event")
    if first_token_at is None or last_token_at is None:
        raise RuntimeError("stream completed without a text token")
    if finish_reason != "length":
        raise RuntimeError(
            "unexpected finish_reason for ignore_eos benchmark: "
            f"expected 'length', got {finish_reason!r}"
        )
    text = "".join(text_parts)
    if usage.get("completion_tokens") is None:
        raise RuntimeError("usage event is missing completion_tokens")
    if usage.get("prompt_tokens") is None:
        raise RuntimeError("usage event is missing prompt_tokens")
    completion_tokens = int(usage["completion_tokens"])
    prompt_tokens_usage = int(usage["prompt_tokens"])
    if completion_tokens != max_tokens:
        raise RuntimeError(
            "ignore_eos benchmark returned "
            f"completion_tokens={completion_tokens}, expected max_tokens={max_tokens}"
        )
    if not (prompt_tokens_usage == prompt_tokens_local == target_prompt_tokens):
        raise RuntimeError(
            "prompt token mismatch: "
            f"usage={prompt_tokens_usage}, local={prompt_tokens_local}, "
            f"target={target_prompt_tokens}"
        )
    result = calculate_metrics(
        started_at=started_at,
        first_token_at=first_token_at,
        last_token_at=last_token_at,
        finished_at=finished_at,
        completion_tokens=completion_tokens,
    )
    result.update(
        {
            "target_prompt_tokens": target_prompt_tokens,
            "prompt_tokens_local": prompt_tokens_local,
            "prompt_tokens_usage": prompt_tokens_usage,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "chunks": chunks,
            "sample": text[:120],
            "usage": usage,
        }
    )
    return result


def run_benchmark(
    args: argparse.Namespace,
    tokenizer: Any,
    *,
    benchmark_fn: Callable[..., dict[str, Any]] = benchmark_once,
    output: TextIO | None = None,
) -> None:
    output = output or sys.stdout
    prompt, prompt_tokens_local = build_prompt(
        tokenizer, args.target_prompt_tokens, args.unit
    )

    measured: list[dict[str, Any]] = []
    total_runs = args.warmup + args.repeat
    for index in range(total_runs):
        is_warmup = index < args.warmup
        result = benchmark_fn(
            base_url=args.base_url,
            model=args.model,
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_tokens_local=prompt_tokens_local,
            target_prompt_tokens=args.target_prompt_tokens,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        result.update(
            {
                "type": "run",
                "phase": "warmup" if is_warmup else "measured",
                "round": index + 1 if is_warmup else index - args.warmup + 1,
            }
        )
        print(json.dumps(result, ensure_ascii=False), file=output, flush=True)
        if not is_warmup:
            measured.append(result)

    print(json.dumps(summarize(measured), ensure_ascii=False), file=output, flush=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for name in ("target_prompt_tokens", "max_tokens", "repeat"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    from transformers import AutoTokenizer

    tokenizer_source = resolve_tokenizer_source(args.tokenizer, args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    run_benchmark(args, tokenizer)


if __name__ == "__main__":
    main()
