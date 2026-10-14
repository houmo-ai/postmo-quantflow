"""Contracts for exporting Fun-Audio-Chat-8B to YueHui."""

from pathlib import Path

import pytest
import torch
import yaml

from xhmodel_merak.xh_other_model.models.funaudiochat._export_utils import to_device_type
from xhmodel_merak.xh_other_model.models.funaudiochat._static_export import FunAudioChatStaticAudioTowerWarp
from xhmodel_merak.xh_other_model.models.funaudiochat.workflow import _build_export_args
from xhquant.api import DeviceType, QuantScheme, create_quant_config
from xhquant.quantization.yuehui._yuehui_quant_schema import YHQuantSchema


REPO_ROOT = Path(__file__).resolve().parents[2]
YUEHUI_CONFIG = (
    REPO_ROOT
    / "configs_merak/workflows/yuehui/other_models/funaudiochat/funaudiochat.yaml"
)
EXPECTED_QUANT_TYPES = {
    "audio_tower": "w8a8h1_sefp",
    "audio_encoder": "w8a8h1_sefp",
    "qwen3": "w8a8h1_sefp",
    "audio_decoder": "w8a8h1_sefp",
}


class _DummyAudioTower(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)

    def forward(self, speech_ids: torch.Tensor):
        return (self.proj(speech_ids.float()),)


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_tower = _DummyAudioTower()
        self.unrelated = torch.nn.Linear(2, 2)


def test_funaudiochat_maps_supported_target_devices() -> None:
    assert to_device_type("XH2a") is DeviceType.XH2a
    assert to_device_type("YueHui") is DeviceType.YueHui

    with pytest.raises(ValueError, match="Unsupported FunAudioChat target_device"):
        to_device_type("XH3")


def test_audio_tower_export_model_is_isolated_from_full_model() -> None:
    model = _DummyModel()
    export_model = FunAudioChatStaticAudioTowerWarp(model).eval()

    assert export_model.audio_tower is not model.audio_tower
    assert export_model.audio_tower.proj.weight.data_ptr() != model.audio_tower.proj.weight.data_ptr()


@pytest.mark.parametrize("quant_type", sorted(set(EXPECTED_QUANT_TYPES.values())))
def test_funaudiochat_builds_yuehui_group_64_quant_config(quant_type: str) -> None:
    quant_config = create_quant_config(
        QuantScheme(target_device=DeviceType.YueHui, quant_type=quant_type)
    )
    effective = YHQuantSchema.from_schema(quant_config).get_quant_config(module_type=torch.nn.Linear)

    assert quant_config.arch == "YueHui"
    assert effective.w_cfg.quantizer.qspec.nshare == 64
    assert effective.i_cfg.quantizer.qspec.nshare == 64


def test_funaudiochat_yuehui_workflow_contract() -> None:
    config = yaml.safe_load(YUEHUI_CONFIG.read_text(encoding="utf-8"))
    export = config["export"]

    assert export["target_device"] == "YueHui"
    assert export["model"]["type"] == "XHFunAudioChatModel"
    assert "weight_formats" not in export
    assert {
        name: component["quant_type"]
        for name, component in export["components"].items()
        if component["enabled"]
    } == EXPECTED_QUANT_TYPES

    args = _build_export_args(export, export["runtime"])
    assert args.target_device is DeviceType.YueHui
