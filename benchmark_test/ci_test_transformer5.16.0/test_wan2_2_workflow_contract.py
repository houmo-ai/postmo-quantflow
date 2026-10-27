"""CPU-only contract for Wan2.2 normalize_force_fp32 wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import yaml

from xhquant.api import QuantScheme
from xhquant.utils.config import ConfigDict
from xhmodel_merak.xh_other_model.models.wan2_2.model import XHWan22Model
from xhmodel_merak.xh_other_model.models.wan2_2.wan2_2_converter import (
    Wan22ConvertConfig,
    Wan22Converter,
)
from xhmodel_merak.xh_other_model.models.wan2_2.workflow import (
    Wan22Workflow,
    _build_convert_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
I2V_CONFIG = REPO_ROOT / (
    "configs_merak/workflows/xh2a/other_models/wan2_2/i2v_A14B/"
    "wan2_2_i2v_A14B.yaml"
)
T2V_CONFIG = REPO_ROOT / (
    "configs_merak/workflows/xh2a/other_models/wan2_2/t2v_A14B/"
    "wan2_2_t2v_A14B.yaml"
)


def test_wan2_2_model_binds_workflow_cls() -> None:
    assert XHWan22Model.WORKFLOW_CLS == (
        "xhmodel_merak.xh_other_model.models.wan2_2.workflow:Wan22Workflow"
    )
    assert Wan22Workflow.__name__ == "Wan22Workflow"


def test_wan2_2_build_convert_config_parses_normalize_force_fp32() -> None:
    convert_cfg = _build_convert_config(
        {
            "target_device": "XH2a",
            "wan2_2": {
                "components": ["t5"],
                "quant_types": {"t5": "w8a8_sefp"},
                "normalize_force_fp32": False,
            },
        }
    )

    assert convert_cfg.normalize_force_fp32 is False
    assert Wan22ConvertConfig().normalize_force_fp32 is False


def test_wan2_2_yaml_sets_normalize_force_fp32_false() -> None:
    for config_path in (I2V_CONFIG, T2V_CONFIG):
        export_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))["export"]
        convert_cfg = _build_convert_config(export_cfg)

        assert export_cfg["model"]["type"] == "XHWan22Model"
        assert export_cfg["wan2_2"]["normalize_force_fp32"] is False
        assert convert_cfg.normalize_force_fp32 is False


def test_wan2_2_i2v_yaml_keeps_default_export_size() -> None:
    export_cfg = yaml.safe_load(I2V_CONFIG.read_text(encoding="utf-8"))["export"]
    convert_cfg = _build_convert_config(export_cfg)

    assert export_cfg["wan2_2"]["size"] == [480, 832]
    assert convert_cfg.size == (480, 832)


def test_wan2_2_export_injects_rmsnorm_force_fp32(tmp_path, monkeypatch) -> None:
    captured = {}

    def _fake_create_quant_config(quant_scheme):
        del quant_scheme
        return ConfigDict({"ops_cfg": ConfigDict()})

    def _fake_convert_dynamo_model_to_quanted_model(model, inputs, device, quant_config):
        del model, inputs, device
        captured["quant_config"] = quant_config
        return MagicMock(name="quanted_model")

    def _fake_convert_quanted_model_to_hmonnx(*args, **kwargs):
        del args, kwargs
        return None

    monkeypatch.setattr(
        "xhmodel_merak.xh_other_model.models.wan2_2.wan2_2_converter.create_quant_config",
        _fake_create_quant_config,
    )
    monkeypatch.setattr(
        "xhmodel_merak.xh_other_model.models.wan2_2.wan2_2_converter.convert_dynamo_model_to_quanted_model",
        _fake_convert_dynamo_model_to_quanted_model,
    )
    monkeypatch.setattr(
        "xhmodel_merak.xh_other_model.models.wan2_2.wan2_2_converter.convert_quanted_model_to_hmonnx",
        _fake_convert_quanted_model_to_hmonnx,
    )

    converter = Wan22Converter(
        pretrained_model_path=str(tmp_path),
        convert_config=Wan22ConvertConfig(
            quant_scheme=QuantScheme(target_device="XH2a", quant_type="w8a8_sefp"),
            normalize_force_fp32=False,
        ),
    )
    converter.device = "cpu"
    converter.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    onnx_file = tmp_path / "dummy.onnx"
    converter._export_component(
        model=torch.nn.Identity(),
        inputs=(torch.zeros(1, 4),),
        input_names=("x",),
        output_names=("y",),
        onnx_file=onnx_file,
        golden_dir=tmp_path / "golden",
        export_golden=False,
    )

    quant_config = captured["quant_config"]
    assert quant_config["ops_cfg"]["RMSNorm"]["force_fp32"] is False
