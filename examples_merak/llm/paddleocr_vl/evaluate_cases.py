# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Replay the issue images with identical preprocessing on HF or HMONNX."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "work_dirs/paddleocr_vl/cases/manifest.json"

OTSL_SOURCE = (
    "https://github.com/PaddlePaddle/PaddleX/blob/"
    "ffb64904d23708863ff5b8da312a5cbd52a7f462/"
    "paddlex/inference/pipelines/paddleocr_vl/uilts.py#L707-L854"
)
OTSL_TOKEN = re.compile(r"<(fcel|ecel|lcel|ucel|xcel|nl)>")
MERGE_MARKERS = {"lcel", "ucel", "xcel"}


def parse_otsl(text):
    """Keep the official six-tag grid before any padding/span repair.

    PaddleX's converter pads/truncates ragged rows before building HTML. Those
    repairs can obscure generated defects, so this diagnostic keeps every raw
    slot and declines merged-cell positional scoring instead.
    """
    tokens = list(OTSL_TOKEN.finditer(text))
    rows, row, issues = [], [], []
    for index, token in enumerate(tokens):
        end = tokens[index + 1].start() if index + 1 < len(tokens) else len(text)
        payload = text[token.end() : end]
        kind = token[1]
        if kind == "nl":
            rows.append(row)
            row = []
            if payload.strip():
                issues.append({"kind": "text_after_row_separator", "text": payload})
        else:
            row.append({"marker": kind, "text": payload})
            if kind != "fcel" and payload.strip():
                issues.append({"kind": "text_after_non_filled_cell", "marker": kind, "text": payload})
    if row:
        rows.append(row)
    if tokens and text[: tokens[0].start()].strip():
        issues.append({"kind": "text_before_grid", "text": text[: tokens[0].start()]})
    markers = Counter(token[1] for token in tokens)
    return {
        "format": "otsl",
        "source_rules": OTSL_SOURCE,
        "rows": rows,
        "row_count": len(rows),
        "row_lengths": [len(row) for row in rows],
        "marker_counts": dict(markers),
        "has_merge_markers": bool(MERGE_MARKERS & markers.keys()),
        "ends_with_row_separator": bool(tokens and tokens[-1][1] == "nl"),
        "issues": issues,
        "repairs_applied": False,
        "span_interpretation": "Raw continuation slots retained; no merged-cell expansion or semantic scoring.",
    }


def phrase_repetition(text, max_unit=64):
    """Find exact tandem repeats of 2..64 characters without cleaning text.

    Offset equality runs make this O(text length * maximum unit width). This
    bounded diagnostic catches short phrase loops; it is not an exhaustive
    detector of all semantic or nonconsecutive repetition.
    """
    best = None
    for width in range(2, min(max_unit, len(text) // 3) + 1):
        run = 0
        for index in range(width, len(text)):
            if text[index] == text[index - width]:
                run += 1
            else:
                run = 0
            count = (run + width) // width
            if count < 3:
                continue
            start = index - run + 1 - width
            unit = text[start : start + width]
            if not unit.strip() or len(set(unit)) < 2:
                continue
            covered = count * width
            if best is None or (covered, -width) > (best["covered_characters"], -len(best["unit"])):
                best = {
                    "unit": unit,
                    "consecutive_complete_repetitions": count,
                    "covered_characters": covered,
                    "start_char_offset_0based": start,
                    "end_char_offset_0based_exclusive": start + covered,
                }
    if best:
        best["total_exact_occurrences"] = text.count(best["unit"])
    return best


class TableRows(HTMLParser):
    """Read emitted rows, retaining empty rows and cell boundaries for diagnostics."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self.row = None
        self.cell = None
        self.closed_tables = 0
        self.has_spans = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self.row = []
            self.tables[-1].append(self.row)
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []
            self.row.append(self.cell)
            attributes = dict(attrs)
            self.has_spans |= any(attributes.get(key, "1") != "1" for key in ("rowspan", "colspan"))
        elif tag == "br" and self.cell is not None:
            self.cell.append("\n")

    def handle_endtag(self, tag):
        if tag in {"td", "th"}:
            self.cell = None
        elif tag == "tr":
            self.row = None
            self.cell = None
        elif tag == "table":
            self.closed_tables += 1
            self.row = None
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def rows(self):
        return [[["".join(cell).strip() for cell in row] for row in table] for table in self.tables]


def _compact(text):
    return re.sub(r"\s+", "", text)


def analyze_output(text, assertions=None):
    """Report degeneration and partial visual labels, without claiming full OCR GT."""
    parsed = TableRows()
    parsed.feed(text)
    tables = parsed.rows()
    otsl = parse_otsl(text) if OTSL_TOKEN.search(text) else None
    table_format = "otsl" if otsl else "html" if tables else "plain_text"
    if otsl:
        # Keep continuation markers visible in the legacy string-cell surface:
        # an all-merge row is not an empty generated row.
        tables = [
            [
                [f"<{cell['marker']}>" if cell["marker"] in MERGE_MARKERS else cell["text"].strip() for cell in row]
                for row in otsl["rows"]
            ]
        ]
    rows = [row for table in tables for row in table]
    nonempty = [index for index, row in enumerate(rows) if any(row)]
    runs = list(re.finditer(r"(\S)\1*", text))
    longest = max(runs, key=lambda match: len(match[0])) if runs else None
    anchors = []
    assertions = assertions or {}
    for assertion in assertions.get("text_assertions", []):
        anchors.append((assertion["id"], assertion["expected"]))
    for assertion in assertions.get("cell_assertions", []):
        for column, expected in assertion["expected_cells"].items():
            anchors.append((f"row_{assertion['row_1based_excluding_header']}:{column}", expected))
    for key in ("weekday_headers", "meal_labels", "table_headers"):
        anchors.extend((key, value) for value in assertions.get(key, []))
    for assertion in assertions.get("semantic_cell_assertions", []):
        anchors.append((f"{assertion['weekday']}:{assertion['meal']}", assertion["dish"]))
    anchor_results = [
        {"id": key, "expected": expected, "present": _compact(expected) in _compact(text)} for key, expected in anchors
    ]
    # The bank's explicit column labels permit positional checks in a generated
    # table. Presence-only anchors above cannot detect account/name swaps.
    bank_checks = []
    headers = assertions.get("table_headers")
    bank_table = next((table for table in tables if table and table[0] == headers), None) if headers else None
    bank_reason = None
    if (otsl and otsl["has_merge_markers"]) or parsed.has_spans:
        bank_reason = "merged_cells_not_scored"
    elif otsl and otsl["issues"]:
        bank_reason = "malformed_otsl_not_scored"
    elif bank_table and any(len(row) != len(headers) for row in bank_table):
        bank_reason = "ragged_rows_not_scored"
    for assertion in assertions.get("cell_assertions", []) if bank_reason is None else []:
        row_index = assertion["row_1based_excluding_header"]
        for column, expected in assertion["expected_cells"].items():
            index = headers.index(column)
            actual = None
            if bank_table and row_index < len(bank_table) and index < len(bank_table[row_index]):
                actual = bank_table[row_index][index]
            bank_checks.append(
                {
                    "row_1based_excluding_header": row_index,
                    "column": column,
                    "expected": expected,
                    "actual": actual,
                    "matches": _compact(actual) == _compact(expected) if actual is not None else False,
                }
            )
    return {
        "characters": len(text),
        "longest_identical_nonspace_character": longest[1] if longest else None,
        "longest_identical_nonspace_run": len(longest[0]) if longest else 0,
        "phrase_repetition": phrase_repetition(text),
        "phrase_repetition_scope": (
            "Exact tandem repeats of 2..64 characters; legitimate headers can repeat. "
            "Diagnostic only, without generation stopping or output repair."
        ),
        "table_format": table_format,
        "otsl": otsl,
        "table_count": len(tables),
        "closed_table_count": parsed.closed_tables,
        "table_row_count": len(rows),
        "empty_table_rows": sum(not any(row) for row in rows),
        "trailing_empty_table_rows": len(rows) - 1 - nonempty[-1] if nonempty else len(rows),
        "tables": tables,
        "partial_visual_anchors": {
            "scope": (
                "Partial, visually reviewed anchors; substring presence does not prove "
                "row/column accuracy or full OCR accuracy."
            ),
            "hit": [item for item in anchor_results if item["present"]],
            "missing": [item for item in anchor_results if not item["present"]],
        },
        "bank_cells": {
            "table_header_matched": bank_table is not None,
            "status": "evaluated_partial_only" if bank_table and bank_reason is None else "not_evaluable",
            "reason": bank_reason or (None if bank_table else "no_matching_table_header"),
            "checks": bank_checks,
        },
    }


def select_samples(manifest, case_ids=None, include_regions=False):
    requested = set(case_ids or [])
    available = {case["id"] for case in manifest["cases"]}
    if requested - available:
        raise ValueError(f"Unknown cases: {sorted(requested - available)}")
    samples = []
    for case in manifest["cases"]:
        if requested and case["id"] not in requested:
            continue
        full = case["full_image"]
        samples.append({"id": full["id"], "case": case, "region": None, "settings": full["replay_defaults"]})
        if include_regions:
            for region in case["regions"]:
                if not region.get("primary_problem_region"):
                    continue
                resize = region["crop_recovery"]["static_resize"]
                settings = {
                    "prompt": region["replay_prompt"],
                    "task": region["task"],
                    "resize_mode": resize["resize_mode"],
                    "target_size": resize["target_size"],
                    "content_size": resize.get("effective_content_size"),
                    "max_new_tokens": region["max_new_tokens"],
                    "context_max_length": region["context_max_length"],
                }
                samples.append({"id": region["id"], "case": case, "region": region, "settings": settings})
    return samples


def generation_summary(input_ids, sequence, eos_token_ids, max_new_tokens):
    if sequence[: len(input_ids)] != input_ids:
        raise ValueError("generate() must return the original input IDs followed by generated IDs")
    generated = sequence[len(input_ids) :]
    eos = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids or [])
    ended_on_eos = bool(generated and generated[-1] in eos)
    return {
        "input_ids": input_ids,
        "output_ids": generated,
        "sequence_ids": sequence,
        "prompt_tokens": len(input_ids),
        "generated_tokens": len(generated),
        "ended_on_eos": ended_on_eos,
        "reached_token_limit": len(generated) >= max_new_tokens,
        "finish_reason": "eos" if ended_on_eos else ("max_new_tokens" if len(generated) >= max_new_tokens else "other"),
    }


def generation_cache_options(model, backend, hf_cache=None):
    """Keep checkpoint defaults unless explicitly overridden, and record their source."""
    if backend == "hmonnx":
        if hf_cache is not None:
            raise ValueError("--hf-cache/--no-hf-cache only apply to the HF backend")
        return {}, {"use_cache": True, "use_cache_source": "fixed_hmonnx_kv_cache_abi"}
    if hf_cache is None:
        return {}, {
            "use_cache": model.generation_config.use_cache,
            "use_cache_source": "checkpoint_generation_config",
        }
    return {"use_cache": hf_cache}, {"use_cache": hf_cache, "use_cache_source": "explicit_hf_cache_override"}


def _resolve(path, manifest_path):
    path = Path(path)
    if path.is_absolute():
        return path
    for candidate in (REPO_ROOT / path, manifest_path.parent / path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def evaluate(args):
    import torch

    from xhmodel_merak.xh_llm.models.paddleocr_vl.hf import (
        load_paddleocr_config,
        load_paddleocr_model,
        load_paddleocr_processor,
    )
    from xhmodel_merak.xh_llm.models.paddleocr_vl.processing import (
        load_rgb_image,
        prepare_image,
        prepare_inputs,
        prepare_native_inputs,
        recover_region_image,
    )

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    samples = select_samples(manifest, args.case, args.include_regions)
    labels = json.loads(_resolve(manifest["visual_assertions"], manifest_path).read_text())["cases"]
    output_dir = Path(args.output_dir)
    dtype = getattr(torch, args.dtype)
    processor = load_paddleocr_processor(args.model_dir)
    started = time.monotonic()
    if args.backend == "hf":
        config = load_paddleocr_config(args.model_dir)
        if getattr(config, "quantization_config", None):
            from xhmodel_merak.xh_llm.models.paddleocr_vl.model import XHPaddleOCRVLModel

            model = XHPaddleOCRVLModel.get_hf_model(
                args.model_dir, dtype=dtype, device_map="cpu", gptq_weight_mode="dequantize"
            )
            model.set_attn_implementation("eager")
            model = model.to(args.device).eval()
        else:
            model = load_paddleocr_model(args.model_dir, dtype=dtype).to(args.device)
    else:
        from xhmodel_merak.xh_llm import AutoLLMHONNXModel
        from xhquant.api import xhquant_init

        xhquant_init(str(output_dir / "hmonnx.log"), False)
        model = AutoLLMHONNXModel.from_pretrained(args.meta_file, device_map=[args.device])
        model.to(args.device)
    load_seconds = time.monotonic() - started
    cache_kwargs, cache_evidence = generation_cache_options(model, args.backend, args.hf_cache)
    results = []
    failures = 0
    for sample in samples:
        sample_dir = output_dir / sample["id"]
        sample_dir.mkdir()
        timestamp = datetime.now(timezone.utc).isoformat()
        began = time.monotonic()
        result = {
            "id": sample["id"],
            "case_id": sample["case"]["id"],
            "backend": args.backend,
            "started_at": timestamp,
            "status": "running",
            "settings": sample["settings"],
            "is_layout_region": sample["region"] is not None,
            **cache_evidence,
        }
        _write_json(sample_dir / "result.json", result)
        try:
            case, settings, region = sample["case"], sample["settings"], sample["region"]
            image_path = _resolve(case["input_image"], manifest_path)
            original_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if original_hash != case["sha256"]:
                raise ValueError(f"Input image checksum differs from the case manifest: {image_path}")
            image = load_rgb_image(image_path)
            if region:
                source = json.loads(_resolve(region["historical_source_json"], manifest_path).read_text())
                image = recover_region_image(image, region, layout_source=source)
            if args.native_resize:
                # prepared.png records the actual processor input. Native
                # resize happens inside the processor; tensor hashes below
                # capture its resulting resized/normalized patch values.
                prepared = image
                inputs = prepare_native_inputs(processor, image, settings["prompt"])
                _, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
                width = grid_w * processor.image_processor.patch_size
                height = grid_h * processor.image_processor.patch_size
            else:
                width, height = settings["target_size"]
                prepared = prepare_image(
                    image, settings["resize_mode"], width, height, content_size=settings.get("content_size")
                )
                inputs = prepare_inputs(processor, prepared, settings["prompt"], width=width, height=height)
            prepared.save(sample_dir / "prepared.png")
            input_ids = inputs["input_ids"][0].tolist()
            budget = args.max_new_tokens or settings["max_new_tokens"]
            context_limit = settings["context_max_length"]
            if len(input_ids) + budget > context_limit:
                raise ValueError(
                    f"Prompt ({len(input_ids)}) + output budget ({budget}) exceeds context ({context_limit})"
                )
            result.update(
                {
                    "image": str(image_path),
                    "original_sha256": original_hash,
                    "prepared_pixels_sha256": hashlib.sha256(prepared.tobytes()).hexdigest(),
                    "preprocessing": {
                        "source_size": case["image_size"],
                        "crop_size": list(image.size),
                        "target_size": [width, height],
                        "resize_mode": "native" if args.native_resize else settings["resize_mode"],
                        "content_size": None if args.native_resize else settings.get("content_size"),
                        "resample": "native_processor" if args.native_resize else "PIL.BICUBIC",
                        "letterbox_alignment": None if args.native_resize else "center",
                        "padding_color": None if args.native_resize else "white",
                        "exif_rotation": False,
                        "processor_do_resize": args.native_resize,
                        "region_recovery": region.get("crop_recovery") if region else None,
                    },
                    "prompt": settings["prompt"],
                    "max_new_tokens": budget,
                    "context_max_length": context_limit,
                    "do_sample": False,
                    "dtype": args.dtype,
                    "device": args.device,
                    "input_ids": input_ids,
                    "input_tensors": {
                        key: {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "sha256": hashlib.sha256(
                                value.contiguous().view(torch.uint8).numpy().tobytes()
                            ).hexdigest(),
                        }
                        for key, value in inputs.items()
                        if torch.is_tensor(value)
                    },
                    "image_grid_thw": inputs["image_grid_thw"].tolist(),
                }
            )
            if args.native_resize:
                result["preprocessing"].update(
                    prepared_pixels_stage="source_crop_before_native_resize",
                    native_image_processor=processor.image_processor.to_dict(),
                )
            _write_json(sample_dir / "result.json", result)
            inputs = {
                key: value.to(device=args.device, dtype=dtype if value.is_floating_point() else value.dtype)
                if torch.is_tensor(value)
                else value
                for key, value in inputs.items()
            }
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize(args.device)
            generated_at = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=budget, do_sample=False, **cache_kwargs)
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize(args.device)
            generation_seconds = time.monotonic() - generated_at
            sequence = generated.sequences if hasattr(generated, "sequences") else generated
            sequence = sequence[0].tolist()
            eos_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
            if eos_ids is None:
                eos_ids = processor.tokenizer.eos_token_id
            result.update(generation_summary(input_ids, sequence, eos_ids, budget))
            text = processor.tokenizer.decode(
                result["output_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            raw = processor.tokenizer.decode(
                result["output_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            (sample_dir / "output.txt").write_text(text, encoding="utf-8")
            (sample_dir / "raw_output.txt").write_text(raw, encoding="utf-8")
            result.update(
                {
                    "status": "completed",
                    "generation_seconds": generation_seconds,
                    "output_text": text,
                    "raw_output_text": raw,
                    "diagnostics": analyze_output(text, labels.get(case["id"])),
                }
            )
            if region:
                result["diagnostics"]["partial_visual_anchors"]["scope"] += (
                    " Labels are page-level; missing anchors may lie outside this region."
                )
        except Exception as error:
            failures += 1
            result.update({"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
        result["elapsed_seconds"] = time.monotonic() - began
        _write_json(sample_dir / "result.json", result)
        results.append(result)
        _write_json(
            output_dir / "summary.json",
            {"arguments": vars(args), "model_load_seconds": load_seconds, "results": results, "failures": failures},
        )
        print(f"{sample['id']}: {result['status']} ({result['elapsed_seconds']:.1f}s)", flush=True)
    if failures:
        raise RuntimeError(f"{failures} case(s) failed; inspect result.json under {output_dir}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=("hf", "hmonnx"), default="hf")
    parser.add_argument("--meta-file")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--case", action="append", help="Repeat to select cases; default is all three.")
    parser.add_argument("--include-regions", action="store_true", help="Also run the primary problem regions.")
    parser.add_argument(
        "--native-resize",
        action="store_true",
        help="Use native aspect-dependent resize instead of manifest resizing; HMONNX requires matching profiles.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        help="Defaults to bfloat16 for HF, float16 for the HMONNX KV-cache ABI.",
    )
    parser.add_argument(
        "--hf-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="HF only: override KV caching; omitted follows the checkpoint generation configuration.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, help="Override manifest budgets (OCR=1024; table=4096).")
    args = parser.parse_args(argv)
    if args.backend == "hmonnx" and not args.meta_file:
        parser.error("--meta-file is required for --backend hmonnx")
    if args.backend == "hmonnx" and args.hf_cache is not None:
        parser.error("--hf-cache/--no-hf-cache only apply to --backend hf")
    if args.backend == "hmonnx" and args.dtype == "bfloat16":
        parser.error("HMONNX requires --dtype float16 for its fixed FP16 KV-cache ABI")
    if args.dtype is None:
        args.dtype = "float16" if args.backend == "hmonnx" else "bfloat16"
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Use a new output directory to preserve prior experiments: {output_dir}")
    _write_json(output_dir / "arguments.json", vars(args))
    with (output_dir / "evaluate.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                evaluate(args)
            except Exception:
                traceback.print_exc()
                raise
    print(f"Results saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
