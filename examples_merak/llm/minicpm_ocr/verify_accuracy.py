# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Paired greedy OCR generation using the identical native processor protocol."""

import argparse
import json
from pathlib import Path


def main():
    import torch
    from PIL import Image
    from xhmodel_merak.xh_llm.models.minicpm_ocr.native import (
        MiniCPMOcrConfig,
        MiniCPMOcrNativeModel,
        load_processor,
    )
    from xhmodel_merak.model_artifact.validation import read_manifest, validate_manifest_files
    from xhmodel_merak.xh_llm.models.minicpm_ocr.runtime import MiniCPMOcrRuntime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--prompt", default="请识别图片中的文字。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, manifest = read_manifest(args.artifact_dir)
    validate_manifest_files(root, manifest["artifact"])
    if manifest["model_id"] != "minicpm-ocr-1.0":
        parser.error("Expected a minicpm-ocr-1.0 artifact")
    model = MiniCPMOcrNativeModel.from_pretrained(
        args.model_dir,
        config=MiniCPMOcrConfig.from_pretrained(args.model_dir),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=args.device,
    ).eval()
    processor = load_processor(args.model_dir)
    runtime = MiniCPMOcrRuntime(root / manifest["artifact"]["metadata_file"], device=args.device)
    rows = []
    try:
        for image in args.image:
            with Image.open(image) as source:
                pixels = source.convert("RGB")
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = processor(
                text=[prompt], images=[pixels], return_tensors="pt", downsample_mode="4x", max_slice_nums=9
            )
            inputs = {key: value.to(args.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            native_ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
            actual = runtime.generate(image, args.prompt, args.max_new_tokens)
            rows.append(
                {
                    "image": str(Path(image).resolve()),
                    "native_token_ids": native_ids,
                    "native_text": processor.tokenizer.decode(native_ids, skip_special_tokens=True),
                    "tokens_equal": native_ids == actual["token_ids"],
                    **actual,
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    finally:
        runtime.close()
    report = {
        "model_id": manifest["model_id"],
        "target": "XH2a",
        "execution": "HMONNX GPU simulation",
        "native_dtype": "bfloat16",
        "sample_count": len(rows),
        "exact_sequences": sum(row["tokens_equal"] for row in rows),
        "passed": all(row["tokens_equal"] for row in rows),
        "samples": rows,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
