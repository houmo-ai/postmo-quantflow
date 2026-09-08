"""CPU-only Chinese text frontend; acoustic checkpoints are not needed here."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np


def phonemes_to_tokens(phonemes: str, vocab: Mapping[str, int], *, max_tokens: int) -> np.ndarray:
    """Map phoneme characters and add upstream Kokoro's two boundary tokens.

    Unlike upstream's silent filtering, reject unknown characters so the demo
    cannot appear to synthesize text that the frontend actually dropped.
    """
    if not phonemes.strip() or not any(character.isalnum() for character in phonemes):
        raise ValueError("text must contain pronounceable content, not only punctuation")
    unknown = sorted(set(phonemes).difference(vocab))
    if unknown:
        raise ValueError(f"phonemes are missing from the Kokoro vocabulary: {unknown!r}")
    length = len(phonemes) + 2
    if length > max_tokens:
        raise ValueError(
            f"text needs {length} tokens (including boundaries), but the largest exported "
            f"route supports {max_tokens}; split the text at sentence boundaries"
        )
    return np.asarray([0, *(vocab[character] for character in phonemes), 0], dtype=np.int32)


def _reject_english(text: str) -> str:
    raise ValueError(f"this demo uses the Chinese v1.1 frontend; English text is not supported: {text!r}")


@lru_cache(maxsize=1)
def _chinese_g2p():
    try:
        from misaki.zh import ZHG2P
    except ImportError as error:
        raise ImportError(
            "--text requires the Chinese frontend; install it with: python -m pip install 'misaki[zh]==0.9.4'"
        ) from error
    return ZHG2P(version="1.1", en_callable=_reject_english)


def prepare_text(text: str, config_path: str | Path, *, max_tokens: int) -> tuple[np.ndarray, str]:
    """Run ZHG2P v1.1 and vocabulary lookup without loading a PyTorch model."""
    if not text.strip():
        raise ValueError("--text must not be empty")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    vocab = config.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"Kokoro config must contain a nonempty vocab mapping: {config_path}")
    phonemes, _ = _chinese_g2p()(text.strip())
    return phonemes_to_tokens(phonemes, vocab, max_tokens=max_tokens), phonemes
