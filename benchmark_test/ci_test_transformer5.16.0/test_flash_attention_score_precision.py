"""FlashAttention configurations expose Q/K/V/P precision; S stays FP16."""

import json
from pathlib import Path

import pytest
import yaml

from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.prefill_decode import parse_flash_attention

ROOT = Path(__file__).resolve().parents[2]
IMPACT = Path(__file__).with_suffix(".impact.json")


def _assert_no_score_bits(value, path):
    if isinstance(value, dict):
        assert "s_bits" not in value, str(path)
        for item in value.values():
            _assert_no_score_bits(item, path)
    elif isinstance(value, list):
        for item in value:
            _assert_no_score_bits(item, path)


def _impact_yaml_paths():
    manifest = json.loads(IMPACT.read_text())
    paths = []
    for rule in manifest["rules"]:
        for pattern in rule["patterns"]:
            if pattern.endswith((".yaml", ".yml")):
                paths.append(ROOT / pattern)
    assert paths, f"no workflow yaml patterns in {IMPACT}"
    return paths


def test_workflow_yaml_does_not_set_score_bits():
    for path in _impact_yaml_paths():
        assert path.is_file(), str(path)
        _assert_no_score_bits(yaml.safe_load(path.read_text()), path)


@pytest.mark.parametrize("p_bits", [8, 16])
def test_model_config_contains_only_operand_precision(p_bits):
    bits = parse_flash_attention({"enable": True, "p_bits": p_bits})
    assert set(bits) == {"q_bits", "k_bits", "v_bits", "p_bits"}
    assert bits["p_bits"] == p_bits


def test_missing_probability_precision_uses_model_default():
    bits = parse_flash_attention({"enable": True})
    assert bits["p_bits"] == 16
