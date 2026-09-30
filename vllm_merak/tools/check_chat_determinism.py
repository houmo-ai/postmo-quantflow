#!/usr/bin/env python3
"""Repeat one streaming chat request and compare generated delta prefixes.

The request intentionally omits ``max_tokens``.  ``--chunks`` is only a client
side observation limit: the HTTP connection is closed after that many
reasoning/content deltas have been received.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _make_body(image_path: Path, prompt: str, model: str) -> bytes:
    image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()


def _read_prefix(
    url: str,
    body: bytes,
    *,
    chunks: int,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    deltas: list[dict[str, str]] = []
    response_id = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            event = json.loads(data)
            response_id = response_id or event.get("id")
            choices = event.get("choices") or ()
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            for key in ("reasoning", "content"):
                value = delta.get(key)
                if value:
                    deltas.append({"kind": key, "text": value})
                    break
            if len(deltas) >= chunks:
                break

    encoded = json.dumps(
        deltas,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return {
        "response_id": response_id,
        "generated_deltas": len(deltas),
        "delta_sequence_sha256": _sha256(encoded),
        "text_sha256": _sha256(
            "".join(delta["text"] for delta in deltas).encode()
        ),
        "preview": "".join(delta["text"] for delta in deltas)[:80],
        "deltas": deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("prompt")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:18408/v1/chat/completions",
    )
    parser.add_argument("--model", default="qwen3_5_9b_w4a8")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--chunks", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")
    if args.chunks < 1:
        parser.error("--chunks must be positive")

    body = _make_body(args.image, args.prompt, args.model)
    results = []
    baseline: list[dict[str, str]] | None = None
    all_equal = True
    for index in range(args.runs):
        result = _read_prefix(
            args.url,
            body,
            chunks=args.chunks,
            timeout=args.timeout,
        )
        sequence = result.pop("deltas")
        if baseline is None:
            baseline = sequence
        equal_to_baseline = sequence == baseline
        all_equal = all_equal and equal_to_baseline
        result["run"] = index + 1
        result["equal_to_baseline"] = equal_to_baseline
        results.append(result)
        print(
            f"run={index + 1} id={result['response_id']} "
            f"deltas={result['generated_deltas']} "
            f"sha256={result['delta_sequence_sha256']} "
            f"equal={equal_to_baseline}",
            flush=True,
        )

    report = {
        "url": args.url,
        "runs": args.runs,
        "client_observation_chunks": args.chunks,
        "request_has_max_tokens": False,
        "request_body_sha256": _sha256(body),
        "image_sha256": _sha256(args.image.read_bytes()),
        "all_equal": all_equal,
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if all_equal else 1


if __name__ == "__main__":
    sys.exit(main())
