"""Waveform windowing and audio-span token layout for one Supported utterance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SAMPLING_RATE = 16000
AUDIO_WINDOW_SECONDS = 30
AUDIO_WINDOW_SAMPLES = SAMPLING_RATE * AUDIO_WINDOW_SECONDS
AUDIO_TOKENS_PER_SECOND = 12.5
AUDIO_MERGE_SIZE = 4
WHISPER_ENCODER_STRIDE = 2
HOP_LENGTH = 160
TIME_MARKER_EVERY_SECONDS = 5
AUDIO_PAD_TOKEN = "<|audio_pad|>"
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"
_FEATURE_EXTRACTOR_CACHE: dict[str, Any] = {}

DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


class SupportedUtteranceExceededError(ValueError):
    """Audio longer than one Audio window cannot be truncated into a success."""


def _load_wave_path(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".wav":
        return _load_wav_pcm16(path)
    return _load_with_soundfile(path)


def _load_wav_pcm16(path: Path) -> np.ndarray:
    import wave

    try:
        with wave.open(str(path), "rb") as container:
            if container.getnchannels() != 1:
                raise ValueError(f"Expected mono audio, got {container.getnchannels()} channels")
            if container.getframerate() != SAMPLING_RATE:
                raise ValueError(
                    f"Expected {SAMPLING_RATE} Hz audio, got {container.getframerate()} Hz"
                )
            if container.getsampwidth() != 2:
                raise ValueError("Expected 16-bit PCM audio")
            frames = container.readframes(container.getnframes())
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Unable to read 16-bit PCM audio file: {path}") from error
    pcm = np.frombuffer(frames, dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def _load_with_soundfile(path: Path) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as error:
        raise ValueError(f"Unable to read {path.suffix} audio without soundfile: {path}") from error
    try:
        info = sf.info(str(path))
        if info.channels != 1:
            raise ValueError(f"Expected mono audio, got {info.channels} channels")
        if info.samplerate != SAMPLING_RATE:
            raise ValueError(f"Expected {SAMPLING_RATE} Hz audio, got {info.samplerate} Hz")
        if info.subtype != "PCM_16":
            raise ValueError("Expected 16-bit PCM audio")
        waveform, _ = sf.read(str(path), dtype="float32", always_2d=True)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Unable to read 16-bit PCM audio file: {path}") from error
    return np.asarray(waveform[:, 0], dtype=np.float32)


def _as_mono_float32(audio: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(audio, (str, Path)):
        return _load_wave_path(Path(audio))
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = np.squeeze(waveform)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono audio with shape (num_samples,), got {waveform.shape}")
    return waveform


def load_audio(audio: str | Path | np.ndarray) -> np.ndarray:
    """Read one complete mono recording without truncating its time axis."""
    waveform = _as_mono_float32(audio)
    if waveform.shape[0] == 0:
        raise ValueError("Audio must contain at least one sample")
    if not np.isfinite(waveform).all():
        raise ValueError("Audio must contain only finite samples")
    return waveform


def load_supported_utterance(audio: str | Path | np.ndarray) -> np.ndarray:
    waveform = load_audio(audio)
    if waveform.shape[0] > AUDIO_WINDOW_SAMPLES:
        raise SupportedUtteranceExceededError(
            f"Supported utterance is one {AUDIO_WINDOW_SECONDS}-second audio window "
            f"({AUDIO_WINDOW_SAMPLES} samples at {SAMPLING_RATE} Hz); got {int(waveform.shape[0])} samples"
        )
    return waveform


def pad_audio_window(waveform: np.ndarray) -> np.ndarray:
    waveform = load_supported_utterance(waveform)
    if waveform.shape[0] >= AUDIO_WINDOW_SAMPLES:
        return waveform[:AUDIO_WINDOW_SAMPLES]
    return np.pad(waveform, (0, AUDIO_WINDOW_SAMPLES - waveform.shape[0]))


def extract_moss_log_mel(
    audio: str | Path | np.ndarray,
    model_dir: str | Path,
) -> tuple[np.ndarray, int]:
    """Use the bundled Whisper feature extractor for one audio window."""
    waveform = load_supported_utterance(audio)
    from transformers import WhisperFeatureExtractor

    model_key = str(Path(model_dir).resolve())
    extractor = _FEATURE_EXTRACTOR_CACHE.get(model_key)
    if extractor is None:
        extractor = WhisperFeatureExtractor.from_pretrained(model_key, local_files_only=True)
        _FEATURE_EXTRACTOR_CACHE[model_key] = extractor
    features = extractor(
        pad_audio_window(waveform),
        sampling_rate=SAMPLING_RATE,
        padding="max_length",
        max_length=AUDIO_WINDOW_SAMPLES,
        truncation=True,
        return_tensors="np",
    )["input_features"][0]
    expected_shape = (int(extractor.feature_size), int(extractor.nb_max_frames))
    if tuple(features.shape) != expected_shape:
        raise ValueError(f"MOSS feature extractor returned {features.shape}, expected {expected_shape}")
    return np.asarray(features, dtype=np.float32), audio_token_length(int(waveform.shape[0]))


def audio_token_length(num_samples: int) -> int:
    stride = HOP_LENGTH * WHISPER_ENCODER_STRIDE * AUDIO_MERGE_SIZE
    return (int(num_samples) - 1) // stride + 1


def audio_span_ids(
    audio_seq_len: int,
    *,
    audio_token_id: int,
    digit_token_ids: dict[str, int],
) -> list[int]:
    audio_seq_len = int(audio_seq_len)
    if audio_seq_len <= 0 or TIME_MARKER_EVERY_SECONDS <= 0:
        return [audio_token_id] * max(audio_seq_len, 0)
    tokens_per_marker = int(AUDIO_TOKENS_PER_SECOND * TIME_MARKER_EVERY_SECONDS)
    if tokens_per_marker <= 0:
        return [audio_token_id] * audio_seq_len
    duration = audio_seq_len / float(AUDIO_TOKENS_PER_SECOND)
    output: list[int] = []
    consumed = 0
    for sec in range(TIME_MARKER_EVERY_SECONDS, int(duration) + 1, TIME_MARKER_EVERY_SECONDS):
        pos = (sec // TIME_MARKER_EVERY_SECONDS) * tokens_per_marker
        segment_len = pos - consumed
        if segment_len > 0:
            output.extend([audio_token_id] * segment_len)
            consumed += segment_len
        output.extend(digit_token_ids[digit] for digit in str(sec))
    remainder = audio_seq_len - consumed
    if remainder > 0:
        output.extend([audio_token_id] * remainder)
    return output


def build_prompt_text(prompt: str | None) -> str:
    instruction = (prompt or DEFAULT_PROMPT).strip() or DEFAULT_PROMPT
    return (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{AUDIO_START_TOKEN}{AUDIO_PAD_TOKEN}{AUDIO_END_TOKEN}\n"
        f"{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def expand_audio_token(text: str, num_audio_tokens: int, tokenizer: Any) -> list[int]:
    audio_token = AUDIO_PAD_TOKEN
    if text.count(audio_token) != 1:
        raise ValueError(f"Expected exactly one {audio_token!r} token per prompt, got {text.count(audio_token)}")
    audio_token_id = int(tokenizer.convert_tokens_to_ids(audio_token))
    digit_token_ids = {}
    for digit in "0123456789":
        ids = tokenizer.encode(digit, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Digit {digit!r} is not a single token: {ids}")
        digit_token_ids[digit] = int(ids[0])
    audio_ids = audio_span_ids(
        num_audio_tokens,
        audio_token_id=audio_token_id,
        digit_token_ids=digit_token_ids,
    )
    before_audio, after_audio = text.split(audio_token, maxsplit=1)
    return (
        list(tokenizer.encode(before_audio, add_special_tokens=False))
        + audio_ids
        + list(tokenizer.encode(after_audio, add_special_tokens=False))
    )


def as_tokenizer(processor_or_tokenizer: Any) -> Any:
    """Official processor is not a tokenizer; greedy decode needs convert_tokens_to_ids."""
    if processor_or_tokenizer is None:
        raise TypeError("Official processor must expose a tokenizer")
    if hasattr(processor_or_tokenizer, "convert_tokens_to_ids"):
        return processor_or_tokenizer
    inner = getattr(processor_or_tokenizer, "tokenizer", None)
    if inner is not None and inner is not processor_or_tokenizer and hasattr(inner, "convert_tokens_to_ids"):
        return inner
    raise TypeError("Official processor must expose a tokenizer with convert_tokens_to_ids")


def prompt_text_from_processor(processor: Any, prompt: str | None) -> str:
    """Prefer the checkpoint chat template; fall back to the in-repo default."""
    instruction = (prompt or DEFAULT_PROMPT).strip() or DEFAULT_PROMPT
    tokenizer = as_tokenizer(processor)
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        return apply(messages, tokenize=False, add_generation_prompt=True)
    return build_prompt_text(prompt)


def load_official_processor(model_dir: str | Path) -> Any:
    """Load the checkpoint processor; missing remote-code fails."""
    from transformers import AutoProcessor

    root = Path(model_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"MOSS Official processor requires a checkpoint directory: {root}")
    return AutoProcessor.from_pretrained(str(root), trust_remote_code=True, local_files_only=True)


def layout_prompt_ids(processor: Any, prompt: str | None, num_audio_tokens: int) -> Any:
    """Token layout comes from the Official processor (or a test double that implements it)."""
    import numpy as np

    bound = getattr(processor, "layout_prompt_ids", None)
    impl = getattr(bound, "__func__", bound)
    if callable(bound) and impl is not layout_prompt_ids:
        return np.asarray(bound(prompt, num_audio_tokens), dtype=np.int64)
    for name in ("expand_audio_tokens", "apply_audio_prompt"):
        fn = getattr(processor, name, None)
        if callable(fn):
            return np.asarray(fn(prompt, num_audio_tokens), dtype=np.int64)
    expand = getattr(processor, "_expand_audio_token", None)
    if callable(expand):
        text = prompt_text_from_processor(processor, prompt)
        tokenizer = as_tokenizer(processor)
        max_length = int(getattr(tokenizer, "model_max_length", 0) or 131072)
        try:
            ids = expand(text, int(num_audio_tokens), max_length)
        except TypeError:
            ids = expand(text, int(num_audio_tokens))
        return np.asarray(ids, dtype=np.int64)
    tokenizer = as_tokenizer(processor)
    if not hasattr(tokenizer, "encode"):
        raise TypeError("Official processor must layout prompt ids")
    return np.asarray(expand_audio_token(build_prompt_text(prompt), int(num_audio_tokens), tokenizer), dtype=np.int64)
