"""Prefill-decode constants, flash-attention YAML, and host lm_head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .encoder import LLM_HIDDEN_SIZE

PREFILL_LENGTH = 512
CACHE_LENGTH = 2048
MOSS_VOCAB_SIZE = 151936
FLASH_ATTENTION_BIT_NAMES = ("q_bits", "k_bits", "v_bits", "p_bits")


def parse_flash_attention(config: Any) -> dict[str, int] | None:
    """Read Qwen3.5-style ``export.model.flash_attention``; None means dense attention."""
    if config is None:
        return None
    getter = config.get if isinstance(config, Mapping) else None
    if getter is None:
        enabled = bool(getattr(config, "enable", False))

        def get_value(name: str, default: int) -> Any:
            return getattr(config, name, default)
    else:
        enabled = bool(getter("enable", False))

        def get_value(name: str, default: int) -> Any:
            return getter(name, default)

    if not enabled:
        return None
    bits = {name: int(get_value(name, 16)) for name in FLASH_ATTENTION_BIT_NAMES}
    invalid = {name: value for name, value in bits.items() if value not in (8, 16)}
    if invalid:
        detail = ", ".join(f"{name}={value}" for name, value in invalid.items())
        raise ValueError(f"MOSS flash_attention bits must be 8 or 16, got {detail}")
    return bits


class MossLMHead(nn.Module):
    """Untied view of MOSS's tied token embedding matrix for logits."""

    def __init__(self, hidden_size: int = LLM_HIDDEN_SIZE, vocab_size: int = MOSS_VOCAB_SIZE):
        super().__init__()
        self.projection = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden)


__all__ = [
    "CACHE_LENGTH",
    "FLASH_ATTENTION_BIT_NAMES",
    "LLM_HIDDEN_SIZE",
    "MOSS_VOCAB_SIZE",
    "PREFILL_LENGTH",
    "MossLMHead",
    "parse_flash_attention",
]
