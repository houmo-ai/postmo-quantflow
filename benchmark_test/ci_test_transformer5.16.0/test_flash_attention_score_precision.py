"""FlashAttention configurations expose Q/K/V/P precision; S stays FP16."""

from pathlib import Path

import pytest
import yaml

from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.prefill_decode import parse_flash_attention

ROOT = Path(__file__).resolve().parents[2]


def _assert_no_score_bits(value, path):
    if isinstance(value, dict):
        assert "s_bits" not in value, str(path)
        for item in value.values():
            _assert_no_score_bits(item, path)
    elif isinstance(value, list):
        for item in value:
            _assert_no_score_bits(item, path)


def test_workflow_yaml_does_not_set_score_bits():
    for directory in (ROOT / "configs_merak", ROOT / "vllm_merak/test_profiles"):
        for path in directory.rglob("*.yaml"):
            _assert_no_score_bits(yaml.safe_load(path.read_text()), path)


@pytest.mark.parametrize("p_bits", [8, 16])
def test_model_config_contains_only_operand_precision(p_bits):
    bits = parse_flash_attention({"enable": True, "p_bits": p_bits})
    assert set(bits) == {"q_bits", "k_bits", "v_bits", "p_bits"}
    assert bits["p_bits"] == p_bits


def test_missing_probability_precision_uses_model_default():
    bits = parse_flash_attention({"enable": True})
    assert bits["p_bits"] == 16
