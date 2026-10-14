"""Contracts for exporting supported Qwen3-ASR sizes to YueHui."""

import importlib.util
from pathlib import Path

import pytest
import yaml
from torch import nn

from xhquant.api import DeviceType, QuantScheme, create_quant_config
from xhquant.quantization.yuehui._yuehui_quant_schema import YHQuantSchema


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / "xhmodel_merak/xh_other_model/models/qwen3_asr/workflow.py"
CONFIG_PATHS = {
    "0.6B": Path(
        "configs_merak/workflows/yuehui/other_models/qwen3_asr/0_6b/"
        "qwen3_asr_0_6b.yaml"
    ),
    "1.7B": Path(
        "configs_merak/workflows/yuehui/other_models/qwen3_asr/1_7b/"
        "qwen3_asr_1_7b.yaml"
    ),
}
EXPECTED_QUANT_TYPES = {
    "encoder": "w8a8_sefp",
    "prefill_decode": "w8a8_sefp",
}


def _load_workflow_module():
    spec = importlib.util.spec_from_file_location("_ci_qwen3_asr_workflow", WORKFLOW_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Qwen3-ASR workflow from {WORKFLOW_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen3_asr_accepts_yuehui_device_and_quant_schemes() -> None:
    to_xh_device_type = _load_workflow_module()._to_xh_device_type

    assert to_xh_device_type("XH2a", DeviceType) is DeviceType.XH2a
    assert to_xh_device_type("YueHui", DeviceType) is DeviceType.YueHui

    for quant_type in EXPECTED_QUANT_TYPES.values():
        quant_config = create_quant_config(
            QuantScheme(target_device=DeviceType.YueHui, quant_type=quant_type)
        )
        assert quant_config.arch == "YueHui"
        effective = YHQuantSchema.from_schema(quant_config).get_quant_config(module_type=nn.Linear)
        assert effective.w_cfg.quantizer.qspec.nshare == 64


@pytest.mark.parametrize(("model_size", "config_path"), CONFIG_PATHS.items())
def test_qwen3_asr_yuehui_workflow_contract(
    model_size: str,
    config_path: Path,
) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    export = config["export"]

    assert model_size in {"0.6B", "1.7B"}
    assert export["target_device"] == "YueHui"
    assert export["model"]["type"] == "XHQwen3ASRLLMModel"
    assert export["audio"]["max_audio_length"] == 1500
    assert {
        name: component["quant_type"]
        for name, component in export["components"].items()
    } == EXPECTED_QUANT_TYPES
