# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Explicit public-variant inventory; adding a recipe must add a real case."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UnifiedCase:
    model_id: str
    variant: str
    fixture: str

    @property
    def id(self):
        return f"{self.model_id}/{self.variant}"


MODELS = {
    "deepseek-v4-flash-0731": ("deepseek_v4", ("full", "dspark")),
    "gemma4-12b-unified": ("gemma_unified", ("full", "mtp")),
    "gemma4-26b-a4b": ("gemma_moe", ("full", "mtp")),
    "gemma4-31b": ("gemma_dense", ("full", "mtp")),
    "gemma4-e2b": ("gemma_e2b", ("full", "mtp")),
    "gemma4-e4b": ("gemma_e4b", ("full", "mtp")),
    "glm-5.3-flash": ("glm", ("full",)),
    "kokoro-82m-v1.1-zh": ("kokoro", ("full",)),
    "ling-3.0-flash": ("ling", ("full",)),
    "macaron-v1-tall": ("macaron", ("full", "lora")),
    "minicpm-v-4.7-1b": ("minicpm_v47", ("full",)),
    "minicpm-v-4.7-35b-a3b": ("minicpm_v47", ("full",)),
    "minicpm5-2.6b": ("minicpm", ("full",)),
    "minimax-h3-fl2va": ("minimax", ("full",)),
    "moss-transcribe-diarize-0.9b": ("moss", ("full",)),
    "ornith-1.0-35b-a3b": ("qwen_moe", ("full",)),
    "paddleocr-vl-1.6": ("paddleocr", ("full",)),
    "pi0.5-libero": ("pi05", ("lerobot-h50",)),
    "qwen-image-2.1": ("qwen_image21", ("full",)),
    "qwen3-next-80b-a3b": ("qwen_next", ("full", "mtp")),
    "qwen3.5-0.8b": ("qwen_dense", ("full",)),
    "qwen3.5-122b-a10b": ("qwen_moe", ("full", "mtp", "dflash")),
    "qwen3.5-2b": ("qwen_dense", ("full",)),
    "qwen3.5-4b": ("qwen_dense", ("full",)),
    "qwen3.5-9b": ("qwen_dense", ("full", "mtp", "dflash")),
    "qwen3.6-27b": ("qwen_dense", ("full", "mtp", "dflash")),
    "qwen3.6-35b-a3b": ("qwen_moe", ("full", "mtp", "dflash", "lora", "lora_mtp")),
    "qwen3.8-27b": ("qwen_dense", ("full", "mtp", "dflash2")),
    "wemm-embedding-2b": ("qwen_dense", ("full",)),
    "wemm-embedding-4b": ("qwen_dense", ("full",)),
    "wemm-embedding-9b": ("qwen_dense", ("full",)),
}
CASES = tuple(UnifiedCase(mid, variant, fixture) for mid, (fixture, variants) in MODELS.items() for variant in variants)
