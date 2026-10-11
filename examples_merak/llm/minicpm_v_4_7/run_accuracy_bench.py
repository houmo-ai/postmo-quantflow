#!/usr/bin/env python3
"""MiniCPM-V-4.7 Torch vs HMONNX accuracy bench.

Reproduces the MiniCPM-V-4.6 comparison protocol:
same processor, dynamic slicing, downsample 4x, max_slice_nums=9, greedy.

Datasets:
  - OCRBench (1000) -> official count score
  - ChartQA_TEST (2500) -> relaxed accuracy (%)
  - AI2D_TEST Direct (3088) -> accuracy (%)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


DEFAULT_OCRBENCH = "echo840/OCRBench"
DEFAULT_CHARTQA = "/data01/home/xuchen/llada_compress/huggingface/lmms-lab/ChartQA"
DEFAULT_AI2D = "/data01/home/xuchen/llada_compress/huggingface/lmms-lab/ai2d"
DEFAULT_HF_MODEL = "/data01/datasets/Minicpm-v4.7/weights/1b"
DEFAULT_EXPORT_META = "work_dirs/minicpm_v_4_7_1b_export/export_meta_info.json"


@dataclass
class SampleResult:
    dataset: str
    index: int
    question: str
    prediction: str
    answers: list[str]
    correct: bool
    elapsed_s: float
    extra: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "hmonnx"), required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("ocrbench", "chartqa", "ai2d"),
        default=("ocrbench", "chartqa", "ai2d"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_HF_MODEL))
    parser.add_argument("--export-meta", type=Path, default=Path(DEFAULT_EXPORT_META))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--downsample-mode", choices=("4x", "16x"), default="4x")
    parser.add_argument("--max-slice-nums", type=int, default=9)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="0 means full split")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ocrbench", default=DEFAULT_OCRBENCH)
    parser.add_argument("--chartqa", default=DEFAULT_CHARTQA)
    parser.add_argument("--ai2d", default=DEFAULT_AI2D)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _save_temp_image(image: Image.Image, tmp_dir: Path, index: int) -> Path:
    path = tmp_dir / f"sample_{index:06d}.png"
    _to_rgb(image).save(path)
    return path


def strip_thinking(text: str) -> str:
    """Drop MiniCPM-V-4.7 reasoning blocks if present."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def ocrbench_match(prediction: str, answers: Iterable[str]) -> bool:
    pred = strip_thinking(prediction).lower().strip()
    for ans in answers:
        if str(ans).lower().strip() and str(ans).lower().strip() in pred:
            return True
    return False


def _to_float(text: str) -> float | None:
    cleaned = text.strip().replace(",", "")
    cleaned = re.sub(r"[%$]", "", cleaned)
    try:
        return float(cleaned)
    except ValueError:
        return None


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def chartqa_relaxed_correct(prediction: str, answer: str) -> bool:
    """ChartQA relaxed accuracy: numeric within 5%, otherwise exact string."""
    pred = strip_thinking(prediction).strip()
    gold = str(answer).strip()
    gold_num = _to_float(gold)
    pred_num = _to_float(pred)
    if pred_num is None:
        nums = _NUM_RE.findall(pred.replace(",", ""))
        pred_num = _to_float(nums[0]) if nums else None
    if pred_num is not None and gold_num is not None:
        if gold_num == 0:
            return abs(pred_num) <= 1e-6
        return abs(pred_num - gold_num) / abs(gold_num) <= 0.05
    return pred.lower() == gold.lower()


_LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def ai2d_direct_correct(prediction: str, options: list[str], answer_idx: int) -> bool:
    gold_option = str(options[answer_idx]).strip()
    gold_letter = chr(ord("A") + answer_idx)
    pred = strip_thinking(prediction).strip()
    letters = _LETTER_RE.findall(pred)
    if letters:
        # Direct-mode answers are usually "B" / "B. ..." — prefer the first letter.
        return letters[0].upper() == gold_letter
    pred_l = pred.lower()
    if gold_option.lower() in pred_l:
        return True
    # Some models answer with the raw index.
    if re.fullmatch(r"[0-3]", pred):
        return int(pred) == answer_idx
    return False


def build_chartqa_prompt(question: str) -> str:
    # Match VLMEvalKit ImageVQADataset / ChartQA_TEST prompt protocol.
    return f"{question.strip()}\nAnswer the question using a single word or phrase."


def build_ai2d_prompt(question: str, options: list[str]) -> str:
    lines = [question.strip(), "", "Options:"]
    for idx, option in enumerate(options):
        lines.append(f"{chr(ord('A') + idx)}. {option}")
    lines.append("")
    lines.append("Answer with the option letter directly.")
    return "\n".join(lines)


class TorchGenerator:
    def __init__(self, args: argparse.Namespace):
        import torch
        from packaging.version import Version
        from transformers import AutoModelForImageTextToText
        from transformers import __version__ as transformers_version

        from xhmodel_merak.xh_llm.models.minicpm_v_4_7.processor_compat import (
            load_minicpm_v47_processor,
        )

        if Version(transformers_version) < Version("5.7.0"):
            raise RuntimeError(
                f"MiniCPM-V-4.7 requires transformers>=5.7.0; found {transformers_version}"
            )
        dtype_map = {
            "auto": "auto",
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        self.device = args.device
        self.downsample_mode = args.downsample_mode
        self.max_slice_nums = args.max_slice_nums
        self.max_new_tokens = args.max_new_tokens
        self.processor = load_minicpm_v47_processor(args.model_dir)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                args.model_dir,
                dtype=dtype_map[args.dtype],
                attn_implementation="eager",
                trust_remote_code=True,
            )
            .eval()
            .to(args.device)
        )

    def generate(self, image_path: Path, prompt: str) -> str:
        import torch

        content = [
            {"type": "image", "path": str(image_path.resolve())},
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            processor_kwargs={
                "images_kwargs": {
                    "downsample_mode": self.downsample_mode,
                    "max_slice_nums": self.max_slice_nums,
                },
            },
        )
        prompt_length = int(inputs["input_ids"].shape[-1])
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                downsample_mode=self.downsample_mode,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                disable_compile=True,
            )
        generated_ids = generated_ids[:, prompt_length:]
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return strip_thinking(text)


class HMONNXGenerator:
    def __init__(self, args: argparse.Namespace):
        from xhmodel_merak.xh_llm.models.minicpm_v_4_7.inference import (
            MiniCPMV47HMONNXRuntime,
        )

        if not args.export_meta.is_file():
            raise FileNotFoundError(f"Export metadata not found: {args.export_meta}")

        class _BenchRuntime(MiniCPMV47HMONNXRuntime):
            """Keep the vision session warm across many eval samples."""

            def prepare_multimodal(self, *pargs, **kwargs):  # type: ignore[override]
                # Reuse parent logic but skip per-sample vision teardown.
                release = self._release_vision_session
                self._release_vision_session = lambda *_a, **_k: None  # type: ignore[method-assign]
                try:
                    return MiniCPMV47HMONNXRuntime.prepare_multimodal(self, *pargs, **kwargs)
                finally:
                    self._release_vision_session = release  # type: ignore[method-assign]

        self.runtime = _BenchRuntime(args.export_meta, device=args.device)
        self.downsample_mode = args.downsample_mode
        self.max_slice_nums = args.max_slice_nums
        self.max_new_tokens = args.max_new_tokens

    def generate(self, image_path: Path, prompt: str) -> str:
        messages = {
            "text": prompt,
            "images": [str(image_path.resolve())],
        }
        return strip_thinking(
            self.runtime.generate(
                messages,
                downsample_mode=self.downsample_mode,
                max_slice_nums=self.max_slice_nums,
                max_new_tokens=self.max_new_tokens,
                enable_thinking=False,
            )
        )


def _load_completed(path: Path) -> dict[int, SampleResult]:
    completed: dict[int, SampleResult] = {}
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            completed[int(item["index"])] = SampleResult(**item)
    return completed


def _append_result(path: Path, result: SampleResult) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def eval_ocrbench(
    generator: Any,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    ds = load_dataset(args.ocrbench)["test"]
    end = len(ds) if args.limit <= 0 else min(len(ds), args.offset + args.limit)
    results_path = out_dir / "ocrbench_results.jsonl"
    completed = _load_completed(results_path) if args.resume else {}
    if not args.resume and results_path.exists():
        results_path.unlink()

    with tempfile.TemporaryDirectory(prefix="minicpm_v47_ocrbench_") as tmp:
        tmp_dir = Path(tmp)
        for idx in tqdm(range(args.offset, end), desc="OCRBench"):
            if idx in completed:
                continue
            row = ds[idx]
            image_path = _save_temp_image(row["image"], tmp_dir, idx)
            answers = row["answer"]
            if not isinstance(answers, list):
                answers = [answers]
            answers = [str(a) for a in answers]
            t0 = time.perf_counter()
            pred = generator.generate(image_path, str(row["question"]))
            elapsed = time.perf_counter() - t0
            result = SampleResult(
                dataset="OCRBench",
                index=idx,
                question=str(row["question"]),
                prediction=pred,
                answers=answers,
                correct=ocrbench_match(pred, answers),
                elapsed_s=elapsed,
                extra={"question_type": row.get("question_type"), "source": row.get("dataset")},
            )
            _append_result(results_path, result)
            completed[idx] = result

    results = [completed[i] for i in range(args.offset, end) if i in completed]
    by_cat: dict[str, list[SampleResult]] = defaultdict(list)
    for item in results:
        by_cat[str(item.extra.get("question_type", "unknown"))].append(item)
    summary = {
        "dataset": "OCRBench",
        "num_samples": len(results),
        "metric": "score",
        "score": sum(1 for r in results if r.correct),
        "accuracy": (sum(1 for r in results if r.correct) / len(results)) if results else 0.0,
        "per_category": {
            cat: {
                "correct": sum(1 for r in items if r.correct),
                "total": len(items),
            }
            for cat, items in by_cat.items()
        },
    }
    (out_dir / "ocrbench_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def eval_chartqa(
    generator: Any,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    ds = load_dataset(args.chartqa)["test"]
    end = len(ds) if args.limit <= 0 else min(len(ds), args.offset + args.limit)
    results_path = out_dir / "chartqa_results.jsonl"
    completed = _load_completed(results_path) if args.resume else {}
    if not args.resume and results_path.exists():
        results_path.unlink()

    with tempfile.TemporaryDirectory(prefix="minicpm_v47_chartqa_") as tmp:
        tmp_dir = Path(tmp)
        for idx in tqdm(range(args.offset, end), desc="ChartQA"):
            if idx in completed:
                continue
            row = ds[idx]
            image_path = _save_temp_image(row["image"], tmp_dir, idx)
            answer = str(row["answer"])
            prompt = build_chartqa_prompt(str(row["question"]))
            t0 = time.perf_counter()
            pred = generator.generate(image_path, prompt)
            elapsed = time.perf_counter() - t0
            result = SampleResult(
                dataset="ChartQA_TEST",
                index=idx,
                question=prompt,
                prediction=pred,
                answers=[answer],
                correct=chartqa_relaxed_correct(pred, answer),
                elapsed_s=elapsed,
                extra={"type": row.get("type")},
            )
            _append_result(results_path, result)
            completed[idx] = result

    results = [completed[i] for i in range(args.offset, end) if i in completed]
    summary = {
        "dataset": "ChartQA_TEST",
        "num_samples": len(results),
        "metric": "relaxed_accuracy_percent",
        "score": 100.0 * (sum(1 for r in results if r.correct) / len(results)) if results else 0.0,
        "correct": sum(1 for r in results if r.correct),
    }
    (out_dir / "chartqa_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def eval_ai2d(
    generator: Any,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    ds = load_dataset(args.ai2d)["test"]
    end = len(ds) if args.limit <= 0 else min(len(ds), args.offset + args.limit)
    results_path = out_dir / "ai2d_results.jsonl"
    completed = _load_completed(results_path) if args.resume else {}
    if not args.resume and results_path.exists():
        results_path.unlink()

    with tempfile.TemporaryDirectory(prefix="minicpm_v47_ai2d_") as tmp:
        tmp_dir = Path(tmp)
        for idx in tqdm(range(args.offset, end), desc="AI2D Direct"):
            if idx in completed:
                continue
            row = ds[idx]
            options = list(row["options"])
            answer_idx = int(row["answer"])
            prompt = build_ai2d_prompt(str(row["question"]), options)
            image_path = _save_temp_image(row["image"], tmp_dir, idx)
            t0 = time.perf_counter()
            pred = generator.generate(image_path, prompt)
            elapsed = time.perf_counter() - t0
            result = SampleResult(
                dataset="AI2D_TEST_Direct",
                index=idx,
                question=prompt,
                prediction=pred,
                answers=[str(options[answer_idx]), chr(ord("A") + answer_idx)],
                correct=ai2d_direct_correct(pred, options, answer_idx),
                elapsed_s=elapsed,
                extra={"answer_idx": answer_idx, "options": options},
            )
            _append_result(results_path, result)
            completed[idx] = result

    results = [completed[i] for i in range(args.offset, end) if i in completed]
    summary = {
        "dataset": "AI2D_TEST（Direct）",
        "num_samples": len(results),
        "metric": "accuracy_percent",
        "score": 100.0 * (sum(1 for r in results if r.correct) / len(results)) if results else 0.0,
        "correct": sum(1 for r in results if r.correct),
    }
    (out_dir / "ai2d_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "backend": args.backend,
        "device": args.device,
        "downsample_mode": args.downsample_mode,
        "max_slice_nums": args.max_slice_nums,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "offset": args.offset,
        "model_dir": str(args.model_dir),
        "export_meta": str(args.export_meta),
        "datasets": list(args.datasets),
    }
    (args.output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Loading {args.backend} backend on {args.device} ...")
    generator: Any
    if args.backend == "torch":
        generator = TorchGenerator(args)
    else:
        generator = HMONNXGenerator(args)

    summaries = []
    runners: dict[str, Callable[[Any, argparse.Namespace, Path], dict[str, Any]]] = {
        "ocrbench": eval_ocrbench,
        "chartqa": eval_chartqa,
        "ai2d": eval_ai2d,
    }
    for name in args.datasets:
        print(f"==== Evaluating {name} ====")
        summary = runners[name](generator, args, args.output_dir)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
