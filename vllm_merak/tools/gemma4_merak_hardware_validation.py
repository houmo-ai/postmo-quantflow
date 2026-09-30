#!/usr/bin/env python3
"""Run a fail-closed Gemma4 Merak hardware matrix and emit JSONL evidence.

vLLM remains eager in both modes. ``hmonnx-cuda-graph`` enables only the Merak
HMONNX V2 CUDA Graph runtime. This harness deliberately has no reasoning or
tool parser, so it cannot silently apply Qwen response semantics to Gemma4.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple, TextIO, cast


class ValidationCase(NamedTuple):
    target_prompt_tokens: int
    mode: str
    input_kind: str
    image: Path | None
    repeat: int
    request_count: int


class Gemma4ArtifactContract(NamedTuple):
    meta_path: Path
    meta_sha256: str
    attention_contract_version: int
    prefill_chunk_size: int
    visual_token_count: int


class TraceRun(NamedTuple):
    run_id: str
    model_trace: Path
    kv_trace: Path


def load_gemma4_artifact_contract(model_dir: Path) -> Gemma4ArtifactContract:
    """Load immutable visual/chunk facts from the exported HMONNX metadata."""

    model_dir = model_dir.expanduser().resolve()
    merak_config_path = model_dir / "merak_config.json"
    if not merak_config_path.is_file():
        raise FileNotFoundError(merak_config_path)
    merak_config = json.loads(merak_config_path.read_text(encoding="utf-8"))
    xh_model = merak_config.get("xh_model")
    if not isinstance(xh_model, dict):
        raise ValueError("Gemma4 merak_config.json is missing xh_model metadata")
    meta_path = Path(str(xh_model.get("meta_info") or "golden_meta_info.json"))
    if not meta_path.is_absolute():
        meta_path = model_dir / meta_path
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    encoded = meta_path.read_bytes()
    meta = json.loads(encoded)
    contract_version = meta.get("attention_contract_version")
    if contract_version != 2:
        raise ValueError(
            "Gemma4 hardware validation requires attention_contract_version=2"
        )
    model_config = meta.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Gemma4 artifact is missing model_config")
    prefill_chunk_size = model_config.get("prefill_chunk_length")
    if prefill_chunk_size != 320:
        raise ValueError(
            "Gemma4 atomic visual chunk must be exactly 320 tokens, "
            f"got {prefill_chunk_size!r}"
        )
    visual_config = meta.get("visual_config")
    if not isinstance(visual_config, dict):
        raise ValueError("Gemma4 image harness requires exported visual_config")
    visual_token_count = visual_config.get("num_image_tokens")
    if type(visual_token_count) is not int or not 0 < visual_token_count <= 280:
        raise ValueError(
            "Gemma4 exported visual span must be positive with maximum 280 tokens, "
            f"got {visual_token_count!r}"
        )
    return Gemma4ArtifactContract(
        meta_path=meta_path,
        meta_sha256=hashlib.sha256(encoded).hexdigest(),
        attention_contract_version=contract_version,
        prefill_chunk_size=prefill_chunk_size,
        visual_token_count=visual_token_count,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "hmonnx-cuda-graph"), required=True)
    parser.add_argument("--prompt-token-lengths", nargs="+", type=int)
    parser.add_argument("--boundaries", action="store_true")
    parser.add_argument("--window", type=int)
    parser.add_argument("--prompt-text", default="请简要说明输入内容。")
    parser.add_argument("--unit", default="测试 ")
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--visual-token-count",
        type=int,
        help="Optional assertion; the exported artifact remains authoritative.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument(
        "--three-requests",
        action="store_const",
        const=3,
        dest="request_count",
        help=(
            "Run three sequential request IDs in one unchanged engine; the middle "
            "request is the release-to-reallocation scheduler barrier."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--allowed-local-media-path", type=Path)
    parser.add_argument("--seed", type=int, default=1024)
    # Kept as an observable invariant for argument-level regression tests. It
    # is intentionally not exposed as a CLI option.
    parser.set_defaults(reasoning_parser=None)
    return parser


def _boundary_lengths(window: int) -> tuple[int, ...]:
    return window - 1, window, window + 1, window + 320, 4 * window


def build_validation_cases(args: argparse.Namespace) -> list[ValidationCase]:
    if bool(args.boundaries) == bool(args.prompt_token_lengths):
        raise ValueError(
            "select either --boundaries or --prompt-token-lengths, exactly once"
        )
    if args.boundaries:
        if args.window is None or args.window <= 0:
            raise ValueError("--boundaries requires a positive --window")
        lengths = _boundary_lengths(args.window)
    else:
        lengths = tuple(args.prompt_token_lengths)
    if any(length <= 0 for length in lengths):
        raise ValueError("prompt token lengths must be positive")
    if args.repeat <= 0 or args.request_count <= 0:
        raise ValueError("--repeat and --request-count must be positive")
    if args.mode == "hmonnx-cuda-graph" and args.repeat < 2:
        raise ValueError("hmonnx-cuda-graph validation requires --repeat >= 2")
    if args.request_count < 3:
        raise ValueError(
            "Gemma4 hardware validation requires --request-count >= 3: the "
            "middle request is required as a release-to-reallocation barrier "
            "for KV recycle/zero proof"
        )
    if args.max_tokens <= 0 or args.max_model_len <= 0:
        raise ValueError("--max-tokens and --max-model-len must be positive")
    if args.image is not None and not args.image.is_file():
        raise FileNotFoundError(args.image)

    input_kind = "image" if args.image is not None else "text"
    return [
        ValidationCase(
            target_prompt_tokens=length,
            mode=args.mode,
            input_kind=input_kind,
            image=args.image,
            repeat=args.repeat,
            request_count=args.request_count,
        )
        for length in lengths
    ]


def configure_runtime_environment(mode: str) -> None:
    if mode not in {"eager", "hmonnx-cuda-graph"}:
        raise ValueError(f"unsupported validation mode {mode!r}")
    os.environ["VLLM_MERAK_USE_V2"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if mode == "hmonnx-cuda-graph":
        os.environ["VLLM_MERAK_CUDA_GRAPH"] = "1"
    else:
        os.environ.pop("VLLM_MERAK_CUDA_GRAPH", None)


def _configure_trace_sinks(args: argparse.Namespace) -> TraceRun:
    trace_dir = args.trace_dir or args.output.parent / f"{args.output.stem}-trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    model_trace = trace_dir / f"model-{run_id}.jsonl"
    kv_trace = trace_dir / f"kv-{run_id}.jsonl"
    for sink in (model_trace, kv_trace):
        sink.open("x", encoding="utf-8").close()
    os.environ["VLLM_MERAK_TRACE_RUN_ID"] = run_id
    os.environ["VLLM_MERAK_GEMMA4_TRACE_JSONL"] = str(model_trace)
    os.environ["VLLM_MERAK_KV_TRACE_JSONL"] = str(kv_trace)
    return TraceRun(run_id, model_trace, kv_trace)


def load_trace_rows(trace_path: Path, run_id: str) -> list[dict[str, Any]]:
    """Load one trace while rejecting malformed or cross-run evidence."""

    rows: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"malformed trace row {trace_path}:{line_number}"
                ) from exc
            if row.get("run_id") != run_id:
                raise RuntimeError(
                    f"stale trace row {trace_path}:{line_number}: "
                    f"expected run_id={run_id!r}, got {row.get('run_id')!r}"
                )
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"trace row must be an object: {trace_path}:{line_number}"
                )
            rows.append(row)
    return rows


def validate_trace_run(trace_path: Path, run_id: str) -> int:
    """Fail closed if a trace contains malformed or cross-run evidence."""

    return len(load_trace_rows(trace_path, run_id))


def _positive_pointer(value: Any) -> bool:
    return type(value) is int and value > 0


def _graph_replay_observed(graph_runtime: Any) -> bool:
    if not isinstance(graph_runtime, dict):
        return False
    for key, value in graph_runtime.items():
        normalized = str(key).lower()
        if "replay" not in normalized:
            continue
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, (int, float)) and value > 0:
            return True
        if isinstance(value, str) and "replay" in value.lower():
            return True
    state = str(graph_runtime.get("cuda_graph_state", "")).lower()
    return "replay" in state


def _normalize_trace_ranges(value: Any) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []

    def visit(candidate: Any) -> None:
        if (
            isinstance(candidate, (list, tuple))
            and len(candidate) == 2
            and all(type(item) is int for item in candidate)
        ):
            pair = (int(candidate[0]), int(candidate[1]))
            if pair != (0, 0):
                if pair[0] < 0 or pair[1] < pair[0]:
                    raise RuntimeError(
                        f"Gemma4 visual trace contains invalid range {pair}"
                    )
                ranges.append(pair)
            return
        if isinstance(candidate, (list, tuple)):
            for child in candidate:
                visit(child)
            return
        if candidate is not None:
            raise RuntimeError("Gemma4 visual trace range payload is malformed")

    visit(value)
    return tuple(ranges)


def validate_hardware_evidence(
    *,
    model_rows: list[dict[str, Any]],
    kv_rows: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    """Validate semantic Gemma4 PageAttention and KV lifecycle evidence."""

    if mode not in {"eager", "hmonnx-cuda-graph"}:
        raise RuntimeError(f"unsupported evidence mode {mode!r}")
    request_plans = [row for row in model_rows if row.get("event") == "request_plan"]
    chunk_rows = [row for row in model_rows if row.get("event") == "chunk_context"]
    if not request_plans or not chunk_rows:
        raise RuntimeError(
            "Gemma4 hardware evidence requires request_plan and chunk_context rows"
        )

    plans_by_request_phase: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for plan in request_plans:
        plan_p = plan.get("P")
        plan_q = plan.get("Q")
        chunk_plan = plan.get("chunk_plan")
        if (
            type(plan_p) is not int
            or plan_p < 0
            or type(plan_q) is not int
            or plan_q <= 0
            or not isinstance(chunk_plan, list)
            or not chunk_plan
        ):
            raise RuntimeError("Gemma4 request_plan evidence is malformed")
        cursor = 0
        for chunk in chunk_plan:
            if (
                not isinstance(chunk, list)
                or len(chunk) != 2
                or type(chunk[0]) is not int
                or type(chunk[1]) is not int
                or chunk[0] != cursor
                or chunk[1] <= chunk[0]
            ):
                raise RuntimeError(
                    "Gemma4 request_plan chunk sequence is not contiguous"
                )
            cursor = chunk[1]
        if cursor != plan_q:
            raise RuntimeError("Gemma4 request_plan chunks do not cover Q exactly")
        _normalize_trace_ranges(plan.get("inclusive_mm_ranges"))
        plans_by_request_phase.setdefault(
            (plan.get("request_id"), plan.get("phase")), []
        ).append(plan)

    groups_by_step: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    graph_replay_observed = False
    for row in chunk_rows:
        p = row.get("P")
        q = row.get("Q")
        if type(p) is not int or p < 0 or type(q) is not int or q <= 0:
            raise RuntimeError(f"invalid PageAttention P/Q evidence: P={p!r}, Q={q!r}")
        if row.get("seq_lens") != [p + q]:
            raise RuntimeError(
                "Gemma4 seq_lens evidence violates absolute P+Q: "
                f"P={p}, Q={q}, seq_lens={row.get('seq_lens')!r}"
            )
        attention_kind = row.get("attention_kind")
        if attention_kind == "full_attention":
            if row.get("kv_window_start_abs") is not None:
                raise RuntimeError(
                    "Gemma4 full attention must not carry kv_window_start_abs"
                )
        elif attention_kind == "sliding_attention":
            window = row.get("sliding_window")
            if type(window) is not int or window <= 0:
                raise RuntimeError("Gemma4 sliding evidence has invalid window")
            expected_start = [max(0, p - window + 1)]
            if row.get("kv_window_start_abs") != expected_start:
                raise RuntimeError(
                    "Gemma4 kv_window_start_abs violates max(0, P-W+1): "
                    f"expected={expected_start}, got={row.get('kv_window_start_abs')!r}"
                )
        else:
            raise RuntimeError(
                f"Gemma4 chunk evidence has unsupported attention kind {attention_kind!r}"
            )

        metadata_ptrs = row.get("metadata_ptrs")
        if not isinstance(metadata_ptrs, dict):
            raise RuntimeError("Gemma4 chunk evidence is missing metadata pointers")
        for field in ("block_ids", "slot_mapping", "seq_lens"):
            if not _positive_pointer(metadata_ptrs.get(field)):
                raise RuntimeError(
                    f"Gemma4 chunk evidence has invalid {field} metadata pointer"
                )
        if attention_kind == "sliding_attention" and not _positive_pointer(
            metadata_ptrs.get("kv_window_start_abs")
        ):
            raise RuntimeError(
                "Gemma4 sliding evidence has no kv_window_start_abs pointer"
            )
        mm_ranges = row.get("mm_prefix_ranges")
        actual_visual_ranges = _normalize_trace_ranges(mm_ranges)
        candidate_plans = [
            plan
            for plan in plans_by_request_phase.get(
                (row.get("request_id"), row.get("phase")), []
            )
            if int(plan["P"]) <= p and p + q <= int(plan["P"]) + int(plan["Q"])
        ]
        if len(candidate_plans) != 1:
            raise RuntimeError(
                "Gemma4 chunk evidence does not resolve to one request visual plan"
            )
        planned_visual_ranges = _normalize_trace_ranges(
            candidate_plans[0].get("inclusive_mm_ranges")
        )
        expected_visual_ranges = []
        for start, end in planned_visual_ranges:
            intersects = end >= p and start < p + q
            contained = start >= p and end < p + q
            if intersects and not contained:
                raise RuntimeError(
                    "Gemma4 visual plan was split across an executed chunk"
                )
            if contained:
                expected_visual_ranges.append((start, end))
        if actual_visual_ranges != tuple(expected_visual_ranges):
            raise RuntimeError(
                "Gemma4 chunk visual metadata does not match the atomic request plan: "
                f"expected={expected_visual_ranges}, got={actual_visual_ranges}"
            )
        if mm_ranges is not None and not _positive_pointer(
            metadata_ptrs.get("mm_prefix_ranges")
        ):
            raise RuntimeError("Gemma4 visual evidence has no mm_prefix_ranges pointer")

        backing_ptrs = row.get("backing_ptrs")
        backing_bytes = row.get("backing_bytes")
        if not isinstance(backing_ptrs, dict) or not backing_ptrs:
            raise RuntimeError("Gemma4 chunk evidence has no KV backing pointers")
        if not isinstance(backing_bytes, dict) or set(backing_bytes) != set(
            backing_ptrs
        ):
            raise RuntimeError("Gemma4 KV backing byte evidence is incomplete")
        if not all(_positive_pointer(value) for value in backing_ptrs.values()):
            raise RuntimeError("Gemma4 KV backing pointer evidence is invalid")
        if not all(
            type(value) is int and value > 0 for value in backing_bytes.values()
        ):
            raise RuntimeError("Gemma4 KV backing byte evidence is invalid")

        expected_execution_mode = (
            "cuda_graph" if mode == "hmonnx-cuda-graph" else "eager"
        )
        if row.get("execution_mode") != expected_execution_mode:
            raise RuntimeError(
                "Gemma4 PageAttention execution mode does not match visual metadata: "
                f"expected={expected_execution_mode!r}, "
                f"got={row.get('execution_mode')!r}"
            )
        graph_replay_observed |= _graph_replay_observed(row.get("graph_runtime"))
        step_key = (row.get("request_id"), row.get("phase"), p, q)
        groups_by_step.setdefault(step_key, []).append(row)

    for step_key, group_rows in groups_by_step.items():
        kinds = {row.get("attention_kind") for row in group_rows}
        if kinds != {"full_attention", "sliding_attention"}:
            raise RuntimeError(
                f"Gemma4 hardware evidence lacks full/sliding groups for step {step_key}"
            )
        mm_values = {
            json.dumps(row.get("mm_prefix_ranges"), sort_keys=True)
            for row in group_rows
        }
        if len(mm_values) != 1:
            raise RuntimeError(
                f"Gemma4 full/sliding visual ranges differ for step {step_key}"
            )

    ownership_rows = [row for row in kv_rows if row.get("event") == "ownership"]
    release_rows = [row for row in kv_rows if row.get("event") == "release"]
    zero_rows = [row for row in kv_rows if row.get("event") == "zero_complete"]
    if not ownership_rows:
        raise RuntimeError("Gemma4 KV evidence has no ownership row")
    if not release_rows:
        raise RuntimeError("Gemma4 KV evidence has no release row for recycle proof")
    if not zero_rows:
        raise RuntimeError("Gemma4 KV evidence has no zero_complete row")
    released_proofs: set[tuple[int, int, int]] = set()
    for row in release_rows:
        release_records = row.get("release_records")
        if not isinstance(release_records, list) or not release_records:
            raise RuntimeError("Gemma4 KV release evidence has no physical records")
        for record in release_records:
            if not isinstance(record, dict):
                raise RuntimeError("Gemma4 KV release record is malformed")
            group_idx = record.get("group_idx")
            backing_ptr = record.get("backing_ptr")
            block_ids = record.get("block_ids")
            if (
                type(group_idx) is not int
                or not _positive_pointer(backing_ptr)
                or not isinstance(block_ids, list)
            ):
                raise RuntimeError("Gemma4 KV release record is malformed")
            released_proofs.update(
                (group_idx, backing_ptr, int(block_id)) for block_id in block_ids
            )

    recycled_zero_proofs: set[tuple[int, int, int]] = set()
    for row in zero_rows:
        if row.get("ordering") != "before_model_execute":
            raise RuntimeError(
                "Gemma4 KV zeroing was not ordered before model execution"
            )
        if not isinstance(row.get("zeroed_block_ids"), list):
            raise RuntimeError("Gemma4 KV zero evidence is missing block IDs")
        unique_backings = row.get("unique_backings")
        if not isinstance(unique_backings, list) or not unique_backings:
            raise RuntimeError("Gemma4 KV zero evidence has no physical backings")
        for backing in unique_backings:
            if isinstance(backing, dict):
                if not _positive_pointer(backing.get("ptr")) or not (
                    type(backing.get("bytes")) is int and backing["bytes"] > 0
                ):
                    raise RuntimeError("Gemma4 KV physical backing evidence is invalid")
            elif not _positive_pointer(backing):
                raise RuntimeError("Gemma4 KV physical backing evidence is invalid")
        if not isinstance(row.get("zero_records"), list):
            raise RuntimeError("Gemma4 KV zero evidence is missing zero records")
        recycled_ids = {int(block_id) for block_id in row.get("recycled_block_ids", ())}
        if not recycled_ids:
            continue
        zeroed_ids = {int(block_id) for block_id in row["zeroed_block_ids"]}
        if not recycled_ids <= zeroed_ids:
            raise RuntimeError("Gemma4 recycled KV blocks were not zeroed")
        for record in row["zero_records"]:
            if not isinstance(record, dict):
                raise RuntimeError("Gemma4 KV zero record is malformed")
            group_idx = record.get("group_idx")
            backing_ptr = record.get("backing_ptr")
            block_ids = record.get("block_ids")
            if (
                type(group_idx) is not int
                or not _positive_pointer(backing_ptr)
                or not isinstance(block_ids, list)
            ):
                raise RuntimeError("Gemma4 KV zero record is malformed")
            recycled_zero_proofs.update(
                (group_idx, backing_ptr, int(block_id))
                for block_id in block_ids
                if int(block_id) in recycled_ids
            )

    recycled_proofs = released_proofs & recycled_zero_proofs
    if not recycled_proofs:
        raise RuntimeError(
            "Gemma4 hardware claim requires released KV blocks to be recycled "
            "and zeroed on the same group/backing before model execution"
        )

    if mode == "hmonnx-cuda-graph" and not graph_replay_observed:
        raise RuntimeError(
            "HMONNX CUDA Graph hardware claim requires observed graph replay evidence"
        )
    return {
        "hardware_claim": True,
        "request_plan_rows": len(request_plans),
        "chunk_context_rows": len(chunk_rows),
        "ownership_rows": len(ownership_rows),
        "release_rows": len(release_rows),
        "zero_complete_rows": len(zero_rows),
        "recycled_block_proofs": len(recycled_proofs),
        "graph_replay_observed": graph_replay_observed,
    }


def promote_hardware_claim(output_path: Path, evidence_summary: dict[str, Any]) -> None:
    """Promote pending rows only after all semantic evidence passed."""

    if evidence_summary.get("hardware_claim") is not True:
        raise RuntimeError("refusing to promote an unvalidated hardware claim")
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or rows[0].get("event") != "manifest":
        raise RuntimeError("hardware output is missing its manifest row")
    for row in rows:
        row["hardware_claim"] = True
    rows[0]["evidence_summary"] = evidence_summary
    promoted = output_path.with_suffix(output_path.suffix + ".promoted")
    with promoted.open("x", encoding="utf-8") as sink:
        for row in rows:
            _write_jsonl(sink, row)
    os.replace(promoted, output_path)


def _exact_token_ids(
    tokenizer: Any, target: int, prompt_text: str, unit: str
) -> list[int]:
    prefix_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    unit_ids = list(tokenizer.encode(unit, add_special_tokens=False))
    if not unit_ids:
        raise ValueError("--unit encodes to zero tokens")
    repeated = (unit_ids * ((target + len(unit_ids) - 1) // len(unit_ids)))[:target]
    if prefix_ids and len(prefix_ids) < target:
        repeated[-len(prefix_ids) :] = prefix_ids
    if len(repeated) != target:
        raise RuntimeError(f"failed to construct exactly {target} token IDs")
    return repeated


def _text_input(tokenizer: Any, case: ValidationCase, args: argparse.Namespace) -> Any:
    from vllm.inputs import TokensPrompt

    return TokensPrompt(
        prompt_token_ids=_exact_token_ids(
            tokenizer, case.target_prompt_tokens, args.prompt_text, args.unit
        )
    )


def _image_messages(
    case: ValidationCase, args: argparse.Namespace, repeats: int
) -> list[dict[str, Any]]:
    assert case.image is not None
    prefix = args.unit * repeats
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prefix + args.prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": case.image.expanduser().resolve().as_uri()},
                },
            ],
        }
    ]


def build_exact_image_messages(
    tokenizer: Any,
    case: ValidationCase,
    args: argparse.Namespace,
    artifact: Gemma4ArtifactContract,
    actual_visual_token_count: int | None = None,
) -> list[dict[str, Any]]:
    """Build a chat whose one placeholder expands to the exact target length."""

    def candidate(repeats: int) -> tuple[list[dict[str, Any]], int]:
        messages = _image_messages(case, args, repeats)
        rendered_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        # transformers>=5 returns a BatchEncoding from apply_chat_template()
        # by default.  ``len(BatchEncoding)`` is the number of fields (usually
        # input_ids and attention_mask), not the rendered token count.  Using
        # it here made the doubling search effectively unbounded until the
        # tokenizer's regex engine failed on a gigantic repeated string.
        if isinstance(rendered_ids, Mapping):
            rendered_ids = rendered_ids["input_ids"]
        # The rendered prompt contains one image placeholder token. Gemma4
        # expands it to the image-specific span measured by the real processor
        # and wraps that span in <start_of_image>/<end_of_image>.  The two
        # boundary tokens are introduced by Gemma4Processor rather than the
        # tokenizer-only chat template.
        visual_token_count = (
            artifact.visual_token_count
            if actual_visual_token_count is None
            else actual_visual_token_count
        )
        expanded_length = len(rendered_ids) - 1 + visual_token_count + 2
        return messages, expanded_length

    low, high = 0, 1
    _, high_count = candidate(high)
    while high_count < case.target_prompt_tokens:
        low, high = high, high * 2
        _, high_count = candidate(high)
    while low + 1 < high:
        middle = (low + high) // 2
        _, count = candidate(middle)
        if count < case.target_prompt_tokens:
            low = middle
        else:
            high = middle
    for repeats in sorted({max(0, low + offset) for offset in range(-4, 6)}):
        messages, count = candidate(repeats)
        if count == case.target_prompt_tokens:
            return messages
    raise ValueError(
        "cannot construct exact expanded image prompt: "
        f"target={case.target_prompt_tokens}, nearest repetitions={low}/{high}"
    )


def measure_image_visual_token_count(
    model_dir: Path,
    image_path: Path,
    artifact: Gemma4ArtifactContract,
) -> int:
    """Run the real HF Gemma4 processor once to measure this image's span."""

    from PIL import Image
    from transformers.models.gemma4 import Gemma4Processor

    assets = model_dir / "hf_config"
    if not assets.is_dir():
        assets = model_dir
    processor = Gemma4Processor.from_pretrained(str(assets), trust_remote_code=True)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        outputs = processor(
            text=processor.image_token,
            images=image,
            return_tensors="pt",
            max_soft_tokens=artifact.visual_token_count,
        )
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    count = int((outputs["input_ids"] == image_token_id).sum().item())
    if not 0 < count <= artifact.visual_token_count:
        raise ValueError(
            "Gemma4 processor image span is outside the exported capacity: "
            f"actual={count}, capacity={artifact.visual_token_count}"
        )
    return count


def validate_processed_visual_span(
    prompt_token_ids: list[int] | tuple[int, ...],
    image_token_id: int,
    artifact: Gemma4ArtifactContract,
    expected_visual_token_count: int | None = None,
) -> tuple[int, int]:
    """Require one exact contiguous visual span in the processor output."""

    positions = [
        index
        for index, token_id in enumerate(prompt_token_ids)
        if int(token_id) == image_token_id
    ]
    if not positions:
        raise ValueError("Gemma4 processor output has no visual token span")
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError("Gemma4 processor visual tokens are not contiguous")
    expected = expected_visual_token_count or artifact.visual_token_count
    if len(positions) != expected:
        raise ValueError(
            "Gemma4 processor visual span does not match artifact visual span "
            f"{expected}: got {len(positions)}"
        )
    return positions[0], positions[-1]


def _result_row(
    *,
    case: ValidationCase,
    output: Any,
    elapsed_s: float,
    repeat_index: int,
    request_index: int,
    visual_span: tuple[int, int] | None = None,
) -> dict[str, Any]:
    prompt_token_ids = list(getattr(output, "prompt_token_ids", ()) or ())
    actual = len(prompt_token_ids)
    if actual != case.target_prompt_tokens:
        raise RuntimeError(
            "exact prompt token validation failed: "
            f"target={case.target_prompt_tokens}, actual={actual}, "
            f"input_kind={case.input_kind}"
        )
    candidates = list(getattr(output, "outputs", ()) or ())
    if not candidates:
        raise RuntimeError("vLLM returned no completion candidates")
    first = candidates[0]
    text = str(getattr(first, "text", ""))
    token_ids = list(getattr(first, "token_ids", ()) or ())
    if not text.strip() and not token_ids:
        raise RuntimeError("vLLM returned an empty completion")
    row = {
        "schema": "vllm-merak.gemma4.hardware-validation/v1",
        "event": "request_result",
        "mode": case.mode,
        "input_kind": case.input_kind,
        "target_prompt_tokens": case.target_prompt_tokens,
        "actual_prompt_tokens": actual,
        "repeat_index": repeat_index,
        "request_index": request_index,
        "request_id": getattr(output, "request_id", None),
        "elapsed_s": elapsed_s,
        "completion_token_ids": token_ids,
        "completion_text": text,
        "finish_reason": getattr(first, "finish_reason", None),
        "hardware_claim": False,
    }
    if visual_span is not None:
        row["visual_span"] = list(visual_span)
        row["visual_token_count"] = visual_span[1] - visual_span[0] + 1
    return row


def _write_jsonl(output: TextIO, row: dict[str, Any]) -> None:
    print(json.dumps(row, ensure_ascii=False, sort_keys=True), file=output, flush=True)


def _referenced_graph_fingerprints(
    artifact: Gemma4ArtifactContract,
) -> list[dict[str, Any]]:
    meta = json.loads(artifact.meta_path.read_text(encoding="utf-8"))
    candidates: set[Path] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and value.lower().endswith((".hmonnx", ".onnx")):
            path = Path(value)
            if not path.is_absolute():
                path = artifact.meta_path.parent / path
            if path.is_file():
                candidates.add(path.resolve())

    visit(meta)
    fingerprints = []
    for path in sorted(candidates):
        encoded = path.read_bytes()
        fingerprints.append(
            {
                "path": str(path),
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    return fingerprints


def _runtime_versions(torch_module: Any) -> dict[str, Any]:
    packages = {}
    for package in ("vllm", "vllm-merak", "xhquant", "torch"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    device = None
    if torch_module.cuda.is_available():
        device = torch_module.cuda.get_device_name(torch_module.cuda.current_device())
    return {
        "packages": packages,
        "torch_cuda": getattr(torch_module.version, "cuda", None),
        "device": device,
    }


def _is_unified_artifact(model_dir: Path) -> bool:
    config_path = model_dir.expanduser().resolve() / "hf_config" / "config.json"
    if not config_path.is_file():
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config.get("model_type") == "gemma4_unified"


def build_llm_kwargs(
    args: argparse.Namespace,
    artifact: Gemma4ArtifactContract,
    compilation_config: Any,
    allowed_media: Path | None,
) -> dict[str, Any]:
    """Build a deterministic engine contract for lifecycle validation."""

    llm_kwargs: dict[str, Any] = {
        "model": str(args.model_dir.expanduser().resolve()),
        "enforce_eager": True,
        "compilation_config": compilation_config,
        "trust_remote_code": True,
        "config_format": "merak_llm",
        "load_format": "merak_llm",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        # Contract-v2 currently has one fixed sliding-cache owner.  The
        # request_count cases below are deliberately sequential and must not
        # be turned into concurrent scheduler capacity.
        "max_num_seqs": 1,
        # Identical prompts must return their blocks to the allocator instead
        # of remaining referenced by the prefix cache.  Request 3 can then
        # deterministically allocate after request 1's release was observed.
        "enable_prefix_caching": False,
        "disable_chunked_mm_input": True,
        # The Unified artifact advertises image, video, and audio processors.
        # Restrict profiling to the modality exercised by this validation run;
        # otherwise vLLM reserves the largest unused modality item (for
        # example, a 2496-token video) and rejects the documented 2K engine
        # token budget before loading a text-only request.
        "limit_mm_per_prompt": {
            "image": 1 if getattr(args, "image", None) is not None else 0,
            "video": 0,
            "audio": 0,
        },
    }
    if not _is_unified_artifact(args.model_dir):
        llm_kwargs["mm_processor_kwargs"] = {
            "max_soft_tokens": artifact.visual_token_count,
        }
    if allowed_media is not None:
        llm_kwargs["allowed_local_media_path"] = str(allowed_media)
    return llm_kwargs


def run_validation(args: argparse.Namespace) -> None:
    cases = build_validation_cases(args)
    artifact = load_gemma4_artifact_contract(args.model_dir)
    if (
        args.visual_token_count is not None
        and args.visual_token_count != artifact.visual_token_count
    ):
        raise ValueError(
            "--visual-token-count is only an assertion and does not match the "
            f"artifact: cli={args.visual_token_count}, "
            f"artifact={artifact.visual_token_count}"
        )
    configure_runtime_environment(args.mode)
    trace_run = _configure_trace_sinks(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(
        f".{args.output.name}.{trace_run.run_id}.tmp"
    )

    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig
    from vllm.config.compilation import CUDAGraphMode

    compilation_config = CompilationConfig()
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    allowed_media = args.allowed_local_media_path
    if allowed_media is None and args.image is not None:
        allowed_media = args.image.expanduser().resolve().parent
    llm_kwargs = build_llm_kwargs(
        args,
        artifact,
        compilation_config,
        allowed_media,
    )

    llm = None
    requests_completed = False
    actual_visual_token_count = (
        measure_image_visual_token_count(args.model_dir, args.image, artifact)
        if args.image is not None
        else None
    )
    try:
        llm = LLM(**cast(Any, llm_kwargs))
        tokenizer = llm.get_tokenizer()
        sampling = SamplingParams(
            temperature=0,
            max_tokens=args.max_tokens,
            seed=args.seed,
            ignore_eos=True,
        )
        with temporary_output.open("x", encoding="utf-8") as output_file:
            _write_jsonl(
                output_file,
                {
                    "schema": "vllm-merak.gemma4.hardware-validation/v1",
                    "event": "manifest",
                    "model_dir": str(args.model_dir.expanduser().resolve()),
                    "run_id": trace_run.run_id,
                    "mode": args.mode,
                    "reasoning_parser": None,
                    "model_trace": str(trace_run.model_trace),
                    "kv_trace": str(trace_run.kv_trace),
                    "artifact_meta_path": str(artifact.meta_path),
                    "artifact_meta_sha256": artifact.meta_sha256,
                    "referenced_graph_fingerprints": (
                        _referenced_graph_fingerprints(artifact)
                    ),
                    "runtime_versions": _runtime_versions(torch),
                    "attention_contract_version": (artifact.attention_contract_version),
                    "prefill_chunk_size": artifact.prefill_chunk_size,
                    "visual_token_count": artifact.visual_token_count,
                    "actual_visual_token_count": actual_visual_token_count,
                    "cuda_available": torch.cuda.is_available(),
                    "hardware_claim": False,
                },
            )
            for case in cases:
                for repeat_index in range(1, case.repeat + 1):
                    for request_index in range(1, case.request_count + 1):
                        started = time.perf_counter()
                        if case.input_kind == "text":
                            outputs = llm.generate(
                                [_text_input(tokenizer, case, args)],
                                sampling_params=sampling,
                                use_tqdm=False,
                            )
                        else:
                            outputs = llm.chat(
                                [
                                    build_exact_image_messages(
                                        tokenizer,
                                        case,
                                        args,
                                        artifact,
                                        actual_visual_token_count,
                                    )
                                ],
                                sampling_params=sampling,
                                use_tqdm=False,
                            )
                        elapsed_s = time.perf_counter() - started
                        if len(outputs) != 1:
                            raise RuntimeError(
                                f"expected one vLLM output, got {len(outputs)}"
                            )
                        visual_span = None
                        if case.input_kind == "image":
                            image_token_id = tokenizer.convert_tokens_to_ids(
                                "<|image|>"
                            )
                            if type(image_token_id) is not int or image_token_id < 0:
                                raise RuntimeError(
                                    "Gemma4 tokenizer has no <|image|> token ID"
                                )
                            visual_span = validate_processed_visual_span(
                                list(outputs[0].prompt_token_ids),
                                image_token_id,
                                artifact,
                                actual_visual_token_count,
                            )
                        _write_jsonl(
                            output_file,
                            _result_row(
                                case=case,
                                output=outputs[0],
                                elapsed_s=elapsed_s,
                                repeat_index=repeat_index,
                                request_index=request_index,
                                visual_span=visual_span,
                            ),
                        )
        requests_completed = True
    finally:
        engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
        if "torch" in locals() and torch.distributed.is_available():
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        from vllm_merak.gemma4_trace import flush_traces

        flush_traces()
        if requests_completed and temporary_output.exists():
            model_rows = load_trace_rows(trace_run.model_trace, trace_run.run_id)
            kv_rows = load_trace_rows(trace_run.kv_trace, trace_run.run_id)
            if not model_rows or not kv_rows:
                temporary_output.unlink(missing_ok=True)
                raise RuntimeError(
                    "Gemma4 validation produced incomplete trace evidence: "
                    f"model_rows={len(model_rows)}, kv_rows={len(kv_rows)}"
                )
            evidence_summary = validate_hardware_evidence(
                model_rows=model_rows,
                kv_rows=kv_rows,
                mode=args.mode,
            )
            promote_hardware_claim(temporary_output, evidence_summary)
            os.replace(temporary_output, args.output)
        else:
            temporary_output.unlink(missing_ok=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_validation(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
