import pytest

from xhmodel_merak.errors import ModelError
from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.transcribe import (
    audio_generation_budget,
    completed_generation,
    generation_limit,
    validate_transcript_text,
)


def test_eos_completion_preserves_generated_ids():
    assert completed_generation([3, 7], 4, None).tolist() == [3, 7]
    assert completed_generation([], 4, None).tolist() == []


def test_transcript_validation_accepts_timestamped_speaker_segments():
    text = "[0.00][S01]hello[0.80] [0.80][S02]world[1.40]"

    assert validate_transcript_text(text, "moss") == text


def test_transcript_validation_accepts_canonical_adjacent_timestamps_and_overlap():
    text = "[0.48][S01]Welcome everyone[1.66][12.26][S02]The meeting starts[13.81][13.40][S01]overlap[14.36]"

    assert validate_transcript_text(text, "moss") == text


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "[1.0][S01]bad[0.5]",
        "[0.0][S01]ok[0.5] trailing",
        "[1.0][S01]one[2.0] [0.8][S02]out of order[1.4]",
        "[0.0][S01]late[30.1]",
    ],
)
def test_transcript_validation_rejects_unstructured_or_invalid_segments(text):
    with pytest.raises(ModelError) as caught:
        validate_transcript_text(text, "moss", max_duration=30.0)
    assert caught.value.details["raw_transcript"] == text


def test_generation_limit_honors_requested_tokens_without_overrunning_cache():
    with pytest.raises(ModelError, match="output budget"):
        generation_limit({}, 124, cache_length=2048, override=5120)
    assert generation_limit({}, 124, cache_length=2048, override=16) == 16
    assert generation_limit({}, 124, cache_length=2048, override=128) == 128
    assert generation_limit({}, 124, cache_length=2048, override=None) == 1924
    with pytest.raises(ModelError, match="re-export a larger context"):
        generation_limit({}, 72000, cache_length=131072, override=65536)


def test_timestamp_error_preserves_raw_transcript_diagnostics():
    text = "[0.0][S01]late[30.1]"
    with pytest.raises(ModelError) as caught:
        validate_transcript_text(text, "moss", max_duration=30.0)

    assert caught.value.details["raw_transcript"] == text
    assert caught.value.details["overflow_s"] == pytest.approx(0.1)


def test_timestamp_validation_allows_explicit_window_boundary_tolerance():
    text = "[0.0][S01]late[30.06]"

    assert validate_transcript_text(
        text,
        "moss",
        max_duration=30.0,
        max_duration_tolerance=0.1,
    ) == text


def test_generation_limit_rejects_non_positive_request_or_full_cache():
    with pytest.raises(ModelError):
        generation_limit({}, 124, cache_length=2048, override=0)
    with pytest.raises(ModelError):
        generation_limit({}, 2048, cache_length=2048, override=1)
    with pytest.raises(ModelError):
        generation_limit({}, 2048, cache_length=2048, override=None)


def test_audio_generation_budget_scales_with_duration_and_keeps_explicit_requests():
    assert audio_generation_budget(30.0, None) == 512
    assert audio_generation_budget(1800.0, None) == 18000
    assert audio_generation_budget(5400.0, None) == 54000
    assert audio_generation_budget(7200.0, None) == 65536
    assert audio_generation_budget(5400.0, 65536) == 65536
    assert audio_generation_budget(30.0, 16) == 16
    assert generation_limit({}, 24000, cache_length=65536, override=18000) == 18000


def test_thirty_minute_budget_requires_64k_cache_not_16k():
    prompt_length = 24000
    with pytest.raises(ModelError, match="KV cache"):
        generation_limit({}, prompt_length, cache_length=16384, override=None)
    with pytest.raises(ModelError, match="output budget"):
        generation_limit({}, 8000, cache_length=16384, override=18000)
    assert generation_limit({}, prompt_length, cache_length=65536, override=None) == 41536
