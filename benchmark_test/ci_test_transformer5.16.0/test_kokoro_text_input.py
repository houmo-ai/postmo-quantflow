"""Small CPU contracts; real ZHG2P + HMONNX is exercised by the demo separately."""

import json

import numpy as np
import pytest

from examples_merak.audio.kokoro.bucketed_inference_demo import _resolve_config_path, parse_args
from xhmodel_merak.xh_other_model.models.kokoro.host import DEFAULT_INPUT_IDS, DEFAULT_PHONEMES
from xhmodel_merak.xh_other_model.models.kokoro.text import (
    _reject_english,
    phonemes_to_tokens,
    prepare_text,
)


def test_kokoro_default_phoneme_mapping_matches_golden_tokens():
    vocab = dict(zip(DEFAULT_PHONEMES, DEFAULT_INPUT_IDS[1:-1], strict=True))
    actual = phonemes_to_tokens(DEFAULT_PHONEMES, vocab, max_tokens=32)
    np.testing.assert_array_equal(actual, DEFAULT_INPUT_IDS)
    assert actual.dtype == np.int32


@pytest.mark.parametrize("capacity", (32, 64, 128, 256))
def test_kokoro_token_capacity_counts_both_boundaries(capacity):
    actual = phonemes_to_tokens("a" * (capacity - 2), {"a": 17}, max_tokens=capacity)
    assert actual.shape == (capacity,)
    assert actual[0] == actual[-1] == 0
    np.testing.assert_array_equal(actual[1:-1], np.full(capacity - 2, 17))
    with pytest.raises(ValueError, match="split the text"):
        phonemes_to_tokens("a" * (capacity - 1), {"a": 17}, max_tokens=capacity)


@pytest.mark.parametrize("phonemes", ("", "  ", ".,!?"))
def test_kokoro_rejects_unpronounceable_phonemes(phonemes):
    with pytest.raises(ValueError, match="pronounceable"):
        phonemes_to_tokens(phonemes, {"a": 17}, max_tokens=32)


def test_kokoro_does_not_silently_drop_unknown_phonemes_or_english():
    with pytest.raises(ValueError, match="missing.*vocabulary"):
        phonemes_to_tokens("a❓", {"a": 17}, max_tokens=32)
    with pytest.raises(ValueError, match="English text is not supported"):
        _reject_english("hello")


def test_kokoro_empty_text_fails_before_loading_assets(tmp_path):
    with pytest.raises(ValueError, match="must not be empty"):
        prepare_text(" \n", tmp_path / "absent.json", max_tokens=32)


@pytest.mark.parametrize("config", ({}, {"vocab": {}}, {"vocab": []}))
def test_kokoro_invalid_config_fails_before_loading_g2p(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty vocab"):
        prepare_text("你好", path, max_tokens=32)


def test_kokoro_config_resolution_does_not_require_checkpoints(tmp_path):
    recorded = tmp_path / "recorded.json"
    recorded.write_text("{}", encoding="utf-8")
    explicit = tmp_path / "explicit.json"
    explicit.write_text("{}", encoding="utf-8")
    metadata = {"runtime_assets": {"config_file": recorded.name}}
    assert _resolve_config_path(tmp_path, metadata, config_path=None) == recorded
    assert _resolve_config_path(tmp_path, metadata, config_path=str(explicit)) == explicit
    with pytest.raises(FileNotFoundError, match="--config-path"):
        _resolve_config_path(tmp_path, metadata, config_path=str(tmp_path / "missing.json"))
    with pytest.raises(KeyError):
        _resolve_config_path(tmp_path, {}, config_path=None)


def test_kokoro_cli_accepts_text_and_preserves_default(monkeypatch):
    base = ["demo", "--export-dir", "export"]
    monkeypatch.setattr("sys.argv", base)
    assert parse_args().text is None
    monkeypatch.setattr("sys.argv", [*base, "--text", "你好，世界。", "--config-path", "config.json"])
    args = parse_args()
    assert args.text == "你好，世界。"
    assert args.config_path == "config.json"
    assert args.backend == "hmonnx"
