import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from xhmodel_merak.xh_other_model.models.hy_mt2.evaluation import (
    TranslationSample,
    build_translation_prompt,
    load_flores_samples,
    resolve_legacy_meta,
    summarize_predictions,
)


DEFAULT_DIRECTIONS = ("zho_Hans-eng_Latn", "eng_Latn-zho_Hans")


class HFBackend:
    def __init__(self, model_dir: str, device: str, dtype: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=torch_dtype,
            device_map={"": device},
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, prompt: str, generation: dict[str, Any]) -> str:
        return _generate(self.model, self.tokenizer, prompt, self.device, generation)


class HMONNXBackend:
    def __init__(
        self,
        model_dir: str,
        export_path: str,
        device: str,
        execution_device: str,
        fast: bool,
        retain_sessions: bool,
    ) -> None:
        from xhmodel_merak.xh_other_model.models.hy_mt2 import (
            HyMT2HFCompatible,
            HyMT2Inference,
        )

        legacy_meta = resolve_legacy_meta(export_path)
        self.device = torch.device(device)
        self.inference = HyMT2Inference(
            str(legacy_meta),
            fast_mode=fast,
            device=device,
            execution_device=execution_device,
            retain_sessions=retain_sessions,
        )
        self.tokenizer = self.inference.tokenizer
        self.model = HyMT2HFCompatible.to_hf_compatible(model_dir, self.inference)
        self.model.eval()
        self.model.to(self.device)

    def generate(self, prompt: str, generation: dict[str, Any]) -> str:
        return _generate(self.model, self.tokenizer, prompt, self.device, generation)


def _generate(model: Any, tokenizer: Any, prompt: str, device: torch.device, generation: dict[str, Any]) -> str:
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {name: value.to(device) for name, value in encoded.items()}
    input_length = int(model_inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        output_ids = model.generate(**model_inputs, **generation)
    return tokenizer.decode(output_ids[0, input_length:], skip_special_tokens=True).strip()


def _generation_config(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        config.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    return config


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = str(record["sample_id"])
        if sample_id in records:
            raise ValueError(f"duplicate sample_id {sample_id!r} at {path}:{line_number}")
        records[sample_id] = record
    return records


def _validate_existing_record(record: dict[str, Any], sample: TranslationSample, prompt: str) -> None:
    expected = {**sample.to_dict(), "prompt": prompt}
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            raise ValueError(
                f"existing record {sample.sample_id!r} has incompatible {field}: "
                f"{record.get(field)!r} != {expected_value!r}; use --overwrite"
            )


def _metric_deltas(
    summary: dict[str, Any],
    baseline_file: str | None,
    *,
    evaluation_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if baseline_file is None:
        return None
    baseline = json.loads(Path(baseline_file).read_text(encoding="utf-8"))
    if evaluation_context is not None:
        for field in ("dataset_dir", "directions", "limit_per_direction", "offset", "generation"):
            if baseline.get(field) != evaluation_context.get(field):
                raise ValueError(
                    f"baseline {field} is {baseline.get(field)!r}, "
                    f"but the current evaluation uses {evaluation_context.get(field)!r}"
                )
    baseline_metrics = baseline["metrics"]
    deltas: dict[str, Any] = {}
    for direction, metrics in summary.items():
        if direction not in baseline_metrics:
            continue
        baseline_direction = baseline_metrics[direction]
        if "directions" in metrics and metrics["directions"] != baseline_direction.get("directions"):
            raise ValueError(
                f"baseline direction count is {baseline_direction.get('directions')}, "
                f"but the current evaluation has {metrics['directions']}"
            )
        if "samples" in metrics and metrics["samples"] != baseline_direction.get("samples"):
            raise ValueError(
                f"baseline sample count for {direction} is {baseline_direction.get('samples')}, "
                f"but the current evaluation has {metrics['samples']}"
            )
        deltas[direction] = {
            "chrf_pp": metrics["chrf_pp"] - baseline_direction["chrf_pp"],
            "bleu": metrics["bleu"] - baseline_direction["bleu"],
        }
    return deltas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hy-MT2 on a reproducible FLORES-200 subset.")
    parser.add_argument("--backend", choices=("fp", "hmonnx"), required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--export-path", help="HMONNX export directory or export_meta_info.json.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing devtest.parquet.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--directions", nargs="+", default=list(DEFAULT_DIRECTIONS))
    parser.add_argument("--split", default="devtest")
    parser.add_argument("--limit-per-direction", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execution-device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--retain-sessions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep both prefill and decode graphs resident between samples (uses more GPU memory).",
    )
    parser.add_argument("--baseline-summary", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "hmonnx" and not args.export_path:
        raise ValueError("--export-path is required for the hmonnx backend")

    torch.manual_seed(args.seed)
    samples = load_flores_samples(
        args.dataset_dir,
        args.directions,
        split=args.split,
        limit_per_direction=args.limit_per_direction,
        offset=args.offset,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = output_dir / f"{args.backend}_predictions.jsonl"
    summary_file = output_dir / f"{args.backend}_summary.json"
    if args.overwrite:
        predictions_file.unlink(missing_ok=True)
        summary_file.unlink(missing_ok=True)
    existing = _load_existing(predictions_file)

    if args.backend == "fp":
        backend: Any = HFBackend(args.model_dir, args.device, args.dtype)
    else:
        backend = HMONNXBackend(
            args.model_dir,
            args.export_path,
            args.device,
            args.execution_device,
            args.fast,
            args.retain_sessions,
        )
    generation = _generation_config(args, backend.tokenizer)

    with predictions_file.open("a", encoding="utf-8") as output:
        for sample in tqdm(samples, desc=f"Hy-MT2 {args.backend} FLORES"):
            prompt = build_translation_prompt(sample.source, sample.source_code, sample.target_code)
            if sample.sample_id in existing:
                _validate_existing_record(existing[sample.sample_id], sample, prompt)
                continue
            started = time.perf_counter()
            prediction = backend.generate(prompt, generation)
            record = {
                **sample.to_dict(),
                "prompt": prompt,
                "prediction": prediction,
                "latency_seconds": time.perf_counter() - started,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            existing[sample.sample_id] = record

    ordered_records = [existing[sample.sample_id] for sample in samples]
    metrics = summarize_predictions(ordered_records)
    evaluation_context = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "directions": args.directions,
        "limit_per_direction": args.limit_per_direction,
        "offset": args.offset,
        "generation": generation,
    }
    summary = {
        "backend": args.backend,
        "model_dir": str(Path(args.model_dir).resolve()),
        "export_path": None if args.export_path is None else str(Path(args.export_path).resolve()),
        "dataset_dir": evaluation_context["dataset_dir"],
        "dataset": "FLORES-200 devtest public Parquet mirror",
        "directions": evaluation_context["directions"],
        "limit_per_direction": evaluation_context["limit_per_direction"],
        "offset": evaluation_context["offset"],
        "generation": evaluation_context["generation"],
        "metrics": metrics,
        "deltas_from_baseline": _metric_deltas(
            metrics,
            args.baseline_summary,
            evaluation_context=evaluation_context,
        ),
        "predictions": predictions_file.name,
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
