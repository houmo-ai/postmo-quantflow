# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Load PaddleOCR using the installed Transformers implementation."""

import json
from pathlib import Path

from transformers import AutoProcessor, PaddleOCRVLConfig, PaddleOCRVLForConditionalGeneration


def load_paddleocr_config(model_dir):
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    config = PaddleOCRVLConfig.from_dict(raw)
    # Transformers 5.16 promotes the default text-config tying flag over an
    # explicit false in legacy flat configs. Preserve the checkpoint's two
    # independent embedding/head matrices before constructing any weights.
    tied = raw.get("tie_word_embeddings", raw.get("text_config", {}).get("tie_word_embeddings", True))
    config.tie_word_embeddings = bool(tied)
    config.text_config.tie_word_embeddings = bool(tied)
    return config


def load_paddleocr_model(model_dir, **kwargs):
    kwargs.setdefault("config", load_paddleocr_config(model_dir))
    kwargs.setdefault("attn_implementation", "eager")
    kwargs.setdefault("trust_remote_code", False)
    return PaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, **kwargs).eval()


def load_paddleocr_processor(model_dir):
    return AutoProcessor.from_pretrained(model_dir, trust_remote_code=False)
