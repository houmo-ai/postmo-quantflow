"""CPU-only contract for the shipped MiniCPM5 calibration resource."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / (
    "configs_merak/workflows/xh2a/llm_models/minicpm5/15b_a2_5b/"
    "minicpm5_15b_a2_5b_xh2a_w4a8_autoround.yaml"
)
CALIBRATION_URI = "xh2modelzoo://data/calib_data/minicpm5_calibration.jsonl"
CALIBRATION_PATH = REPO_ROOT / "data/calib_data/minicpm5_calibration.jsonl"


@pytest.fixture
def quant_adapter():
    """Load the real adapter without importing unrelated model registries."""
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "xhmodel_merak" or name.startswith("xhmodel_merak.")
    }
    for name in saved:
        del sys.modules[name]

    packages = (
        "xhmodel_merak",
        "xhmodel_merak.xh_llm",
        "xhmodel_merak.xh_llm.workflows",
        "xhmodel_merak.xh_llm.models",
        "xhmodel_merak.xh_llm.models.minicpm5",
    )
    try:
        for name in packages:
            package = types.ModuleType(name)
            package.__path__ = [str(REPO_ROOT.joinpath(*name.split(".")))]
            package.__package__ = name
            sys.modules[name] = package
        yield importlib.import_module("xhmodel_merak.xh_llm.models.minicpm5.quant_adapter")
    finally:
        for name in list(sys.modules):
            if name == "xhmodel_merak" or name.startswith("xhmodel_merak."):
                del sys.modules[name]
        sys.modules.update(saved)


def test_minicpm5_profile_resolves_committed_calibration(quant_adapter) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    calibration = config["quant"]["calibration"]

    assert calibration == {
        "jsonl": CALIBRATION_URI,
        "text_key": "text",
        "nsamples": 8,
        "seqlen": 256,
    }
    assert quant_adapter._resolve_calibration_jsonl_path(calibration) == str(CALIBRATION_PATH.resolve())

    records = [json.loads(line) for line in CALIBRATION_PATH.read_text(encoding="utf-8").splitlines()]
    assert len(records) == calibration["nsamples"]
    assert all(set(record) == {calibration["text_key"]} for record in records)
    assert all(isinstance(record["text"], str) and record["text"].strip() for record in records)


def test_minicpm5_calibration_resolver_supports_portable_and_local_paths(
    quant_adapter, monkeypatch, tmp_path
) -> None:
    assert quant_adapter._resolve_calibration_jsonl("repo://data/calib_data/minicpm5_calibration.jsonl") == (
        CALIBRATION_PATH.resolve()
    )

    local_path = tmp_path / "calibration.jsonl"
    local_path.write_text('{"text": "sample"}\n', encoding="utf-8")
    monkeypatch.setenv("MINICPM5_CALIBRATION_JSONL", str(local_path))
    assert quant_adapter._resolve_calibration_jsonl("$MINICPM5_CALIBRATION_JSONL") == local_path.resolve()


def test_minicpm5_calibration_resolver_rejects_invalid_paths(quant_adapter) -> None:
    with pytest.raises(quant_adapter.MiniCPM5QuantConfigError, match="placeholder path"):
        quant_adapter._resolve_calibration_jsonl_path({"jsonl": "path-to-minicpm5-calibration-jsonl"})

    with pytest.raises(FileNotFoundError, match="MiniCPM5 GPTQ calibration JSONL does not exist"):
        quant_adapter._resolve_calibration_jsonl("xh2modelzoo://data/calib_data/missing.jsonl")
