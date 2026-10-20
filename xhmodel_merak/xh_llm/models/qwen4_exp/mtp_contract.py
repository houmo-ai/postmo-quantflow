"""Qwen4 target verification dimensions, independent of draft candidate count."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEFAULT_MTP_VERIFY_LENGTH = 4


def resolve_mtp_verify_length(model_config: Mapping[str, Any]) -> int:
    """Resolve the static target width without changing the single-draft policy."""
    length = model_config.get("verify_length", DEFAULT_MTP_VERIFY_LENGTH)
    if type(length) is not int or length < 2:
        raise ValueError("Qwen4 MTP verify_length must be an integer >= 2")
    return length


def mtp_verify_phase(length: int) -> str:
    """Return the graph role corresponding to an already validated static width."""
    length = resolve_mtp_verify_length({"verify_length": length})
    return f"decode_m{length}"
