"""Qwen3-TTS YueHui workflow configuration contracts."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / "xhmodel_merak/xh_other_model/models/qwen3_tts/workflow.py"
CONFIGS = (
    "0_6b_base/qwen3_tts_12hz_0_6b_base.yaml",
    "0_6b_customvoice/qwen3_tts_12hz_0_6b_customvoice.yaml",
    "1_7b_voicedesign/qwen3_tts_12hz_1_7b_voicedesign.yaml",
)


@lru_cache(maxsize=1)
def _workflow_module():
    spec = importlib.util.spec_from_file_location("_ci_qwen3_tts_workflow", WORKFLOW_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Qwen3-TTS workflow from {WORKFLOW_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("target_device", ("XH2a", "YueHui"))
def test_qwen3_tts_maps_supported_devices(target_device):
    from xhquant.api import DeviceType

    actual = _workflow_module()._to_xh_device_type(target_device)
    assert actual is getattr(DeviceType, target_device)


def test_qwen3_tts_rejects_unsupported_devices():
    with pytest.raises(ValueError, match="XH2a.*YueHui.*M66"):
        _workflow_module()._to_xh_device_type("M66")


@pytest.mark.parametrize("relative_path", CONFIGS)
def test_qwen3_tts_yuehui_configs_only_change_target_device(relative_path):
    root = REPO_ROOT / "configs_merak/workflows"
    xh2_text = (root / "xh2a/other_models/qwen3_tts" / relative_path).read_text(encoding="utf-8")
    yuehui_text = (root / "yuehui/other_models/qwen3_tts" / relative_path).read_text(encoding="utf-8")

    assert xh2_text.count("target_device: XH2a") == 1
    assert yuehui_text == xh2_text.replace("target_device: XH2a", "target_device: YueHui", 1)
