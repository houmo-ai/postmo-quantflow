# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Compare all delivered RM samples with native logits and production scores."""

import argparse
import json
from pathlib import Path


def main():
    import torch
    from transformers import AutoProcessor, Qwen3_5ForSequenceClassification
    from xhmodel_merak.model_artifact.validation import read_manifest, validate_manifest_files
    from xhmodel_merak.xh_llm.models.ocr_rm.processing import prepare_inputs
    from xhmodel_merak.xh_llm.models.ocr_rm.runtime import OcrRewardRuntime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--samples-dir", required=True, help="Directory containing samples/manifest.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--native-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--max-prob-error", type=float, default=0.005)
    parser.add_argument(
        "--reference-max-prob-error",
        type=float,
        default=0.03,
        help="Delivered verification protocol's cross-dtype tolerance",
    )
    args = parser.parse_args()
    root, manifest = read_manifest(args.artifact_dir)
    validate_manifest_files(root, manifest["artifact"])
    if manifest["model_id"] != "ocr-rm-2b-value":
        parser.error("Expected an ocr-rm-2b-value artifact")
    samples_dir = Path(args.samples_dir).resolve()
    samples = json.loads((samples_dir / "samples/manifest.json").read_text())["samples"]
    native = Qwen3_5ForSequenceClassification.from_pretrained(
        args.model_dir,
        num_labels=2,
        dtype=getattr(torch, args.native_dtype),
        attn_implementation="sdpa",
        device_map=args.device,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_dir)
    runtime = OcrRewardRuntime(root / manifest["artifact"]["metadata_file"], device=args.device)
    rows = []
    try:
        for sample in samples:
            original = samples_dir / sample["orig_image"]
            rendered = samples_dir / sample["render_image"]
            text = (samples_dir / sample["ocr_text_file"]).read_text().rstrip("\r\n")
            inputs = prepare_inputs(processor, original, rendered, text)
            inputs = {key: value.to(args.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            with torch.inference_mode():
                logits = native(**inputs).logits[0].float()
            actual = runtime.score(original, rendered, text)
            expected = float(logits.softmax(-1)[1])
            row = {
                "name": sample["name"],
                "category": sample["category"],
                "label": sample["label"],
                "native_prob_1": expected,
                "ref_prob_1": sample["ref_prob_1"],
                **actual,
            }
            row["native_abs_error"] = abs(actual["prob_1"] - expected)
            row["reference_abs_error"] = abs(actual["prob_1"] - sample["ref_prob_1"])
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        runtime.close()
    maximum = max(row["native_abs_error"] for row in rows)
    reference_maximum = max(row["reference_abs_error"] for row in rows)
    report = {
        "model_id": manifest["model_id"],
        "native_dtype": args.native_dtype,
        "target": "XH2a",
        "execution": "HMONNX GPU simulation",
        "sample_count": len(rows),
        "tolerance": args.max_prob_error,
        "max_native_abs_error": maximum,
        "mean_native_abs_error": sum(row["native_abs_error"] for row in rows) / len(rows),
        "max_reference_abs_error": reference_maximum,
        "native_probability_passed": maximum <= args.max_prob_error,
        "reference_tolerance": args.reference_max_prob_error,
        "reference_probability_passed": reference_maximum <= args.reference_max_prob_error,
        "native_decision_agreement": sum((row["prob_1"] >= 0.5) == (row["native_prob_1"] >= 0.5) for row in rows),
        "reference_decision_agreement": sum((row["prob_1"] >= 0.5) == (row["ref_prob_1"] >= 0.5) for row in rows),
        "samples": rows,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, ensure_ascii=False), flush=True)
    if not report["native_probability_passed"] or not report["reference_probability_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
