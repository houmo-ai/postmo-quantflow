# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import torch

from .hf import load_paddleocr_processor
from .processing import prepare_inputs


def load_calibration_inputs(model_dir, jsonl_path, *, nsamples=None, width=896, height=896):
    """Prepare image/prompt/answer records, including response tokens in calibration."""
    path = Path(jsonl_path).resolve()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if nsamples is not None:
        if nsamples <= 0 or nsamples > len(records):
            raise ValueError("nsamples must be positive and not exceed the available calibration records")
        records = records[:nsamples]
    if not records:
        raise ValueError("Calibration JSONL is empty")
    processor = load_paddleocr_processor(model_dir)
    examples = []
    for record in records:
        image = Path(record["image"])
        if not image.is_absolute():
            image = path.parent / image
        inputs = dict(prepare_inputs(processor, image, record["prompt"], width=width, height=height))
        answer = record.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Each OCR calibration image must include a non-empty answer")
        answer_ids = processor.tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids
        if processor.tokenizer.eos_token_id is not None:
            answer_ids = torch.cat([answer_ids, torch.tensor([[processor.tokenizer.eos_token_id]])], dim=1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], answer_ids], dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = torch.cat(
                [inputs["mm_token_type_ids"], torch.zeros_like(answer_ids)], dim=1
            )
        examples.append(inputs)
    return examples
