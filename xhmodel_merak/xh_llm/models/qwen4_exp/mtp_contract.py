"""Qwen4 candidate count and static target verification dimensions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEFAULT_MTP_NUM_DRAFT_TOKENS = 4
DEFAULT_MTP_VERIFY_LENGTH = DEFAULT_MTP_NUM_DRAFT_TOKENS + 1


def resolve_mtp_num_draft_tokens(model_config: Mapping[str, Any]) -> int:
    count = model_config.get("num_draft_tokens", DEFAULT_MTP_NUM_DRAFT_TOKENS)
    if type(count) is not int or count < 1:
        raise ValueError("Qwen4 MTP num_draft_tokens must be a positive integer")
    return count


def resolve_mtp_verify_length(model_config: Mapping[str, Any]) -> int:
    """A verify call contains the pending target token and all N candidates."""
    count = resolve_mtp_num_draft_tokens(model_config)
    length = model_config.get("verify_length", count + 1)
    if type(length) is not int or length < count + 1:
        raise ValueError("Qwen4 MTP verify_length must be an integer >= num_draft_tokens + 1")
    return length


def mtp_verify_phase(length: int, *, runtime_version=5) -> str:
    """Return the graph role corresponding to an already validated static width."""
    if type(length) is not int or length < 2:
        raise ValueError("Qwen4 MTP physical verify length must be an integer >= 2")
    return "decode" if runtime_version >= 5 else f"decode_m{length}"
