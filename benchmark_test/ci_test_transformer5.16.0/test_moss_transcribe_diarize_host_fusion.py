"""CPU contracts for MOSS Host fusion on one Supported utterance."""

import numpy as np
import pytest

from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.host_fusion import (
    AUDIO_WINDOW_SAMPLES,
    AUDIO_WINDOW_SECONDS,
    SupportedUtteranceExceededError,
    fuse_embeddings,
    load_supported_utterance,
    transcribe_supported_utterance,
)
from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.processing import (
    AUDIO_PAD_TOKEN,
    SAMPLING_RATE,
    as_tokenizer,
    audio_token_length,
    build_prompt_text,
    expand_audio_token,
    layout_prompt_ids,
)


class _FakeTokenizer:
    """Minimal tokenizer: specials, digits, and one-id-per-remaining-character."""

    _SPECIALS = (
        "<|im_start|>",
        "<|im_end|>",
        "<|audio_start|>",
        "<|audio_pad|>",
        "<|audio_end|>",
    )

    def __init__(self):
        self._token_to_id = {token: index + 1 for index, token in enumerate(self._SPECIALS)}
        self._token_to_id.update({str(digit): 20 + digit for digit in range(10)})
        self._id_to_token = {index: token for token, index in self._token_to_id.items()}
        self._next_id = 40

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self._token_to_id[token])

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        index = 0
        while index < len(text):
            matched = next((token for token in self._SPECIALS if text.startswith(token, index)), None)
            if matched is not None:
                ids.append(self._token_to_id[matched])
                index += len(matched)
                continue
            character = text[index]
            if character not in self._token_to_id:
                self._token_to_id[character] = self._next_id
                self._id_to_token[self._next_id] = character
                self._next_id += 1
            ids.append(self._token_to_id[character])
            index += 1
        return ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        pieces = []
        specials = set(self._SPECIALS)
        for token_id in np.asarray(ids).tolist():
            token = self._id_to_token[int(token_id)]
            if skip_special_tokens and token in specials:
                continue
            pieces.append(token)
        return "".join(pieces)


def test_audio_longer_than_one_window_is_rejected():
    too_long = np.zeros(AUDIO_WINDOW_SAMPLES + 1, dtype=np.float32)

    with pytest.raises(SupportedUtteranceExceededError, match=str(AUDIO_WINDOW_SECONDS)):
        load_supported_utterance(too_long)


def test_exact_window_length_is_accepted():
    waveform = np.zeros(AUDIO_WINDOW_SAMPLES, dtype=np.float32)

    np.testing.assert_array_equal(load_supported_utterance(waveform), waveform)


def test_time_anchor_digits_are_not_replaced_by_audio_embeddings():
    tokenizer = _FakeTokenizer()
    audio_token_id = tokenizer.convert_tokens_to_ids(AUDIO_PAD_TOKEN)
    six_seconds = np.zeros(SAMPLING_RATE * 6, dtype=np.float32)
    token_len = audio_token_length(int(six_seconds.shape[0]))
    input_ids = np.asarray(expand_audio_token(build_prompt_text("go"), token_len, tokenizer), dtype=np.int64)
    audio_pad_mask = input_ids == audio_token_id
    hidden = 4
    token_embeddings = np.full((input_ids.shape[0], hidden), 100.0, dtype=np.float32)
    audio_embeddings = np.arange(int(np.count_nonzero(audio_pad_mask)) * hidden, dtype=np.float32).reshape(-1, hidden)

    fused = fuse_embeddings(token_embeddings, audio_embeddings, audio_pad_mask)

    np.testing.assert_array_equal(fused[audio_pad_mask], audio_embeddings)
    np.testing.assert_array_equal(fused[~audio_pad_mask], token_embeddings[~audio_pad_mask])
    assert int(np.count_nonzero(audio_pad_mask)) == token_len
    digit_five = tokenizer.convert_tokens_to_ids("5")
    assert digit_five in set(input_ids.tolist())
    assert not bool(((input_ids == digit_five) & audio_pad_mask).any())


def _write_silence_wav(path, num_samples: int) -> None:
    import wave

    pcm = np.zeros(num_samples, dtype=np.int16)
    with wave.open(str(path), "wb") as container:
        container.setnchannels(1)
        container.setsampwidth(2)
        container.setframerate(SAMPLING_RATE)
        container.writeframes(pcm.tobytes())


def test_audio_path_longer_than_one_window_is_rejected(tmp_path):
    path = tmp_path / "too_long.wav"
    _write_silence_wav(path, AUDIO_WINDOW_SAMPLES + 1)

    with pytest.raises(SupportedUtteranceExceededError, match=str(AUDIO_WINDOW_SECONDS)):
        load_supported_utterance(path)


def test_audio_path_accepts_16khz_mono_flac(tmp_path):
    sf = pytest.importorskip("soundfile")

    path = tmp_path / "short.flac"
    waveform = np.zeros(SAMPLING_RATE, dtype=np.float32)
    sf.write(path, waveform, SAMPLING_RATE, subtype="PCM_16", format="FLAC")

    np.testing.assert_array_equal(load_supported_utterance(path), waveform)


@pytest.mark.parametrize(
    ("num_samples", "prompt"),
    [
        (SAMPLING_RATE, "go"),
        (AUDIO_WINDOW_SAMPLES, None),
    ],
)
def test_supported_utterance_returns_transcript_text(num_samples, prompt):
    tokenizer = _FakeTokenizer()
    transcript = "[0.48][S01]hello[1.66]"
    transcript_ids = np.asarray(tokenizer.encode(transcript), dtype=np.int64)
    hidden = 8

    def encode_audio(window: np.ndarray) -> np.ndarray:
        assert window.shape == (AUDIO_WINDOW_SAMPLES,)
        return np.ones((375, hidden), dtype=np.float32)

    def embed_tokens(input_ids: np.ndarray) -> np.ndarray:
        return np.full((input_ids.shape[0], hidden), 2.0, dtype=np.float32)

    def generate(fused: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        assert fused.shape[0] == attention_mask.shape[0]
        assert fused.shape[1] == hidden
        assert np.any(np.isclose(fused[:, 0], 1.0))
        return transcript_ids

    text = transcribe_supported_utterance(
        np.zeros(num_samples, dtype=np.float32),
        prompt=prompt,
        tokenizer=tokenizer,
        encode_audio=encode_audio,
        embed_tokens=embed_tokens,
        generate=generate,
    )

    assert text == transcript


def test_supported_utterance_accepts_model_prompt_builder():
    tokenizer = _FakeTokenizer()
    called = []

    def build_prompt(prompt):
        called.append(prompt)
        return build_prompt_text(prompt)

    text = transcribe_supported_utterance(
        np.zeros(SAMPLING_RATE, dtype=np.float32),
        prompt="official",
        tokenizer=tokenizer,
        encode_audio=lambda window: np.ones((375, 8), dtype=np.float32),
        embed_tokens=lambda ids: np.ones((ids.shape[0], 8), dtype=np.float32),
        generate=lambda fused, mask: np.asarray(tokenizer.encode("[0][S01]ok[1]")),
        build_prompt=build_prompt,
    )

    assert text == "[0][S01]ok[1]"
    assert called == ["official"]


def test_wav_path_supported_utterance_returns_transcript_text(tmp_path):
    tokenizer = _FakeTokenizer()
    transcript = "[0.48][S01]hello[1.66]"
    transcript_ids = np.asarray(tokenizer.encode(transcript), dtype=np.int64)
    path = tmp_path / "ok.wav"
    _write_silence_wav(path, SAMPLING_RATE)
    hidden = 8

    def encode_audio(window: np.ndarray) -> np.ndarray:
        assert window.shape == (AUDIO_WINDOW_SAMPLES,)
        return np.ones((375, hidden), dtype=np.float32)

    def embed_tokens(input_ids: np.ndarray) -> np.ndarray:
        return np.full((input_ids.shape[0], hidden), 2.0, dtype=np.float32)

    def generate(fused: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        del attention_mask
        assert np.any(np.isclose(fused[:, 0], 1.0))
        return transcript_ids

    text = transcribe_supported_utterance(
        path,
        prompt="go",
        tokenizer=tokenizer,
        encode_audio=encode_audio,
        embed_tokens=embed_tokens,
        generate=generate,
    )

    assert text == transcript


@pytest.mark.parametrize("samples", [
    1, AUDIO_WINDOW_SAMPLES - 1, AUDIO_WINDOW_SAMPLES,
    AUDIO_WINDOW_SAMPLES + 1, 2 * AUDIO_WINDOW_SAMPLES,
    2 * AUDIO_WINDOW_SAMPLES + 1600, 4 * AUDIO_WINDOW_SAMPLES,
])
def test_recording_encodes_all_windows_and_generates_once(samples):
    """Stub encoder/LM cost only; exercise real slicing, anchors and fusion."""
    tokenizer = _FakeTokenizer()
    waveform = np.arange(samples, dtype=np.float32) / max(samples, 1)
    windows, generated = [], []
    token_ids = np.asarray(expand_audio_token(build_prompt_text("go"),
                           audio_token_length(samples), tokenizer))
    audio_mask = token_ids == tokenizer.convert_tokens_to_ids(AUDIO_PAD_TOKEN)
    expected_parts = []

    def encode(window):
        start = len(windows) * AUDIO_WINDOW_SAMPLES
        valid = min(samples - start, AUDIO_WINDOW_SAMPLES)
        np.testing.assert_array_equal(window[:valid], waveform[start:start + valid])
        assert np.all(window[valid:] == 0)
        windows.append(window)
        embeddings = np.full((375, 2), len(windows), dtype=np.float32)
        expected_parts.append(embeddings[:audio_token_length(valid)])
        return embeddings

    def generate(fused, mask):
        generated.append(fused)
        np.testing.assert_array_equal(fused[audio_mask], np.concatenate(expected_parts))
        np.testing.assert_array_equal(fused[~audio_mask, 0], token_ids[~audio_mask])
        np.testing.assert_array_equal(mask, np.ones(len(token_ids)))
        return np.asarray(tokenizer.encode("[0][S01]ok[1]"))

    result = transcribe_supported_utterance(
        waveform, "go", tokenizer=tokenizer, encode_audio=encode,
        embed_tokens=lambda ids: np.repeat(ids[:, None], 2, axis=1).astype(np.float32),
        generate=generate,
    )
    assert result == "[0][S01]ok[1]"
    assert len(windows) == (samples + AUDIO_WINDOW_SAMPLES - 1) // AUDIO_WINDOW_SAMPLES
    assert len(generated) == 1


@pytest.mark.parametrize("waveform", [
    np.array([], dtype=np.float32), np.array([np.nan]), np.array([np.inf]),
])
def test_invalid_recording_is_rejected_before_encoder(waveform):
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.processing import load_audio

    with pytest.raises(ValueError):
        load_audio(waveform)


class _NestedOfficialProcessor:
    """ProcessorMixin-shaped object: layout lives here, vocab lives on .tokenizer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []

    def _expand_audio_token(self, text, num_audio_tokens, max_length=131072):
        self.calls.append((text, int(num_audio_tokens), int(max_length)))
        return expand_audio_token(text, num_audio_tokens, self.tokenizer)


def test_as_tokenizer_unwraps_official_processor():
    tokenizer = _FakeTokenizer()
    processor = _NestedOfficialProcessor(tokenizer)

    assert not hasattr(processor, "convert_tokens_to_ids")
    assert as_tokenizer(processor) is tokenizer
    assert as_tokenizer(tokenizer) is tokenizer


def test_layout_and_transcribe_use_official_processor_expansion():
    tokenizer = _FakeTokenizer()
    processor = _NestedOfficialProcessor(tokenizer)
    token_len = audio_token_length(SAMPLING_RATE)
    ids = layout_prompt_ids(processor, "go", token_len)
    hidden = 8

    text = transcribe_supported_utterance(
        np.zeros(SAMPLING_RATE, dtype=np.float32),
        prompt="go",
        tokenizer=processor,
        encode_audio=lambda window: np.ones((375, hidden), dtype=np.float32),
        embed_tokens=lambda tokens: np.ones((tokens.shape[0], hidden), dtype=np.float32),
        generate=lambda fused, mask: np.asarray(tokenizer.encode("[0][S01]ok[1]")),
    )

    assert text == "[0][S01]ok[1]"
    assert processor.calls
    assert processor.calls[0][1] == token_len
    assert ids.shape[0] == layout_prompt_ids(processor, "go", token_len).shape[0]


def test_broken_chat_template_does_not_fall_back_to_in_repo_prompt():
    tokenizer = _FakeTokenizer()

    def boom(*args, **kwargs):
        raise RuntimeError("bad template")

    tokenizer.apply_chat_template = boom
    processor = _NestedOfficialProcessor(tokenizer)
    with pytest.raises(RuntimeError, match="bad template"):
        layout_prompt_ids(processor, "go", audio_token_length(SAMPLING_RATE))


def test_over_capacity_recording_is_rejected_before_any_encoder_work(tmp_path):
    from xhmodel_merak.errors import ModelError
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.transcribe import validate_prompt_budget

    (tmp_path / "generation_config.json").write_text('{"max_new_tokens":1024}')

    def unexpected(*args):
        pytest.fail("Over-capacity recordings must not execute encoder, embedding or generation")

    with pytest.raises(ModelError, match="output budget"):
        transcribe_supported_utterance(
            np.zeros(120 * SAMPLING_RATE, dtype=np.float32),
            tokenizer=_FakeTokenizer(), encode_audio=unexpected,
            embed_tokens=unexpected, generate=unexpected,
            validate_prompt=lambda n: validate_prompt_budget(tmp_path, n, 512, 2048, 1024),
        )
