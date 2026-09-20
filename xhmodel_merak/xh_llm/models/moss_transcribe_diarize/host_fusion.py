"""Windowed audio encoding and recording-level MOSS Host fusion."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .processing import (
    AUDIO_PAD_TOKEN,
    AUDIO_WINDOW_SAMPLES,
    AUDIO_WINDOW_SECONDS,
    SAMPLING_RATE,
    SupportedUtteranceExceededError,
    as_tokenizer,
    audio_span_ids,
    audio_token_length,
    build_prompt_text,
    expand_audio_token,
    layout_prompt_ids,
    load_audio,
    load_supported_utterance,
    pad_audio_window,
)


def fuse_embeddings(
    token_embeddings: np.ndarray,
    audio_embeddings: np.ndarray,
    audio_pad_mask: np.ndarray,
) -> np.ndarray:
    """Write audio tokens into audio-pad positions; leave time-anchor digits unchanged."""
    if token_embeddings.ndim != 2:
        raise ValueError(f"token_embeddings must be rank-2, got shape {token_embeddings.shape}")
    if audio_embeddings.ndim != 2:
        raise ValueError(f"audio_embeddings must be rank-2, got shape {audio_embeddings.shape}")
    if audio_pad_mask.shape != token_embeddings.shape[:1]:
        raise ValueError(
            "audio_pad_mask length must match the token sequence: "
            f"{audio_pad_mask.shape} vs {token_embeddings.shape[:1]}"
        )
    n_audio = int(np.count_nonzero(audio_pad_mask))
    if audio_embeddings.shape[0] != n_audio:
        raise ValueError(
            f"audio embeddings and audio-pad positions do not match: {audio_embeddings.shape[0]} vs {n_audio}"
        )
    if audio_embeddings.shape[1] != token_embeddings.shape[1]:
        raise ValueError(
            "audio embedding width must match token embedding width: "
            f"{audio_embeddings.shape[1]} vs {token_embeddings.shape[1]}"
        )
    fused = np.array(token_embeddings, copy=True)
    fused[np.asarray(audio_pad_mask, dtype=bool)] = audio_embeddings
    return fused


def transcribe_supported_utterance(
    audio: str | Path | np.ndarray,
    prompt: str | None = None,
    *,
    tokenizer: Any,
    encode_audio: Callable[[np.ndarray], np.ndarray],
    embed_tokens: Callable[[np.ndarray], np.ndarray],
    generate: Callable[[np.ndarray, np.ndarray], np.ndarray],
    build_prompt: Callable[[str | None], str] | None = None,
    validate_prompt: Callable[[int], object] | None = None,
) -> str:
    """Encode every window, then decode the complete recording exactly once."""
    waveform = load_audio(audio)
    token_len = audio_token_length(int(waveform.shape[0]))
    vocab = as_tokenizer(tokenizer)
    if build_prompt is None:
        input_ids = layout_prompt_ids(tokenizer, prompt, token_len)
    else:
        prompt_text = build_prompt(prompt)
        input_ids = np.asarray(expand_audio_token(prompt_text, token_len, vocab), dtype=np.int64)
    # Reject impossible recordings before spending time encoding every window.
    if validate_prompt is not None:
        validate_prompt(len(input_ids))
    parts = []
    for start in range(0, len(waveform), AUDIO_WINDOW_SAMPLES):
        chunk = waveform[start:start + AUDIO_WINDOW_SAMPLES]
        chunk_tokens = audio_token_length(len(chunk))
        window_audio = encode_audio(pad_audio_window(chunk))
        if window_audio.ndim != 2 or window_audio.shape[0] < chunk_tokens:
            raise ValueError(
                "encode_audio must return one Audio window of embeddings "
                f"with at least {chunk_tokens} tokens, got shape {window_audio.shape}"
            )
        parts.append(window_audio[:chunk_tokens])
    audio_embeddings = np.concatenate(parts, axis=0)
    audio_token_id = int(vocab.convert_tokens_to_ids(AUDIO_PAD_TOKEN))
    audio_pad_mask = input_ids == audio_token_id
    token_embeddings = embed_tokens(input_ids)
    fused = fuse_embeddings(token_embeddings, audio_embeddings, audio_pad_mask)
    attention_mask = np.ones(input_ids.shape[0], dtype=np.int64)
    generated = generate(fused, attention_mask)
    return str(vocab.decode(generated, skip_special_tokens=True)).strip()


__all__ = [
    "AUDIO_WINDOW_SAMPLES",
    "AUDIO_WINDOW_SECONDS",
    "SAMPLING_RATE",
    "SupportedUtteranceExceededError",
    "audio_span_ids",
    "audio_token_length",
    "fuse_embeddings",
    "load_supported_utterance",
    "transcribe_supported_utterance",
]
