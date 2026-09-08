from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import yaml

from xhmodel_merak.xh_other_model.llm_onnx_model import LLMONNXModel
from xhmodel_merak.xh_other_model.models.cosyvoice3._export_utils import build_quant_config, to_xh_device_type
from xhmodel_merak.xh_other_model.models.cosyvoice3.llm_hmonnx_model import XHQwen2HMONNXModel
from xhquant.api import DeviceType
from xhquant.quantization.yuehui.yuehui_qconfig import YueHuiQuantConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
YUEHUI_CONFIG = REPO_ROOT / "configs_merak/workflows/yuehui/other_models/cosyvoice3/cosyvoice3_0_5b.yaml"


def test_cosyvoice3_maps_supported_target_devices():
    assert to_xh_device_type("XH2a") is DeviceType.XH2a
    assert to_xh_device_type("YueHui") is DeviceType.YueHui

    with pytest.raises(ValueError, match="Unsupported CosyVoice3 target_device"):
        to_xh_device_type("XH3")


def test_cosyvoice3_builds_yuehui_w8a16_quant_config():
    config = build_quant_config("YueHui", "w8a16_sefp")
    effective = YueHuiQuantConfig.from_schema(config)

    assert effective.w_cfg.bits == 8
    assert effective.a_cfg.bits == 16


def test_cosyvoice3_yuehui_workflow_config_selects_verified_w8_baseline():
    config = yaml.safe_load(YUEHUI_CONFIG.read_text(encoding="utf-8"))
    export = config["export"]

    assert export["target_device"] == "YueHui"
    assert "weight_formats" not in export
    assert set(export["quant_types"]) == set(export["components"])
    assert export["quant_types"]["llm"] == "w8a16_sefp"
    assert export["quant_types"]["campplus"] == "w8a16_sefp"
    assert export["quant_types"]["flow_decoder"] == "w8a16_sefp"
    assert export["quant_types"]["hift"] == "w8a16_sefp"


def test_cosyvoice3_llm_decoder_session_follows_runtime_device():
    model = object.__new__(XHQwen2HMONNXModel)
    decoder_session = Mock()
    object.__setattr__(model, "llm_decoder_session", decoder_session)
    device = torch.device("cuda:0")

    with patch.object(LLMONNXModel, "_set_device", return_value=model) as base_set_device:
        result = model._set_device(device)

    base_set_device.assert_called_once_with(device)
    decoder_session.to.assert_called_once_with(device)
    assert result is model

    with patch.object(LLMONNXModel, "_set_exec_device") as base_set_exec_device:
        model._set_exec_device(device)

    base_set_exec_device.assert_called_once_with(device)
    assert decoder_session.exec_device == device
