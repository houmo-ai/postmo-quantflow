"""Gemma4 side graphs must quantize on the same chip as the parent model."""

from __future__ import annotations

import pytest

from xhmodel_merak.xh_llm.models.gemma4_series.modality_contract import (
    Gemma4SeriesModalityContract,
)
from xhmodel_merak.xh_llm.models.gemma4_series.xh_gemma4_series_config import (
    XHGemma4SeriesAudioConfig,
    XHGemma4SeriesModelConfig,
    XHGemma4SeriesVisualConfig,
    XHGemma4UnifiedAudioConfig,
    XHGemma4UnifiedVisualConfig,
)


def _config(chip_arch: str, **nested: dict) -> XHGemma4SeriesModelConfig:
    return XHGemma4SeriesModelConfig(
        model_name="gemma_4_side_chip",
        chip_arch=chip_arch,
        hf_model=None,
        **nested,
    )


def _unified_contract() -> Gemma4SeriesModalityContract:
    return Gemma4SeriesModalityContract(
        frontend_kind="encoder_free",
        image_soft_tokens=280,
        video_soft_tokens_per_frame=70,
        audio_soft_tokens=750,
        vision_patch_dim=6912,
        audio_feature_dim=640,
        position_capacity=1024,
        sampling_rate=16000,
        config_hash="side-graph-chip",
    )


@pytest.mark.parametrize("chip_arch", ["YueHui", "XH2a"])
def test_missing_side_chip_arch_inherits_parent(chip_arch: str) -> None:
    config = _config(
        chip_arch,
        visual_config={"export_mode": "padded", "input_modality": "image"},
        audio_config={"sampling_rate": 16000},
    )

    assert config.chip_arch == chip_arch
    assert config.visual_config.chip_arch == chip_arch
    assert config.video_visual_config.chip_arch == chip_arch
    assert config.audio_config.chip_arch == chip_arch


def test_explicit_side_chip_arch_is_not_overwritten() -> None:
    config = _config(
        "YueHui",
        visual_config={"chip_arch": "XH2a", "input_modality": "image"},
        video_visual_config={"chip_arch": "XH2a", "input_modality": "video"},
        audio_config={"chip_arch": "XH2a"},
    )

    assert config.chip_arch == "YueHui"
    assert config.visual_config.chip_arch == "XH2a"
    assert config.video_visual_config.chip_arch == "XH2a"
    assert config.audio_config.chip_arch == "XH2a"


def test_constructed_side_configs_keep_their_own_chip() -> None:
    visual = XHGemma4SeriesVisualConfig(model_name="visual", chip_arch="XH2a")
    audio = XHGemma4SeriesAudioConfig(model_name="audio", chip_arch="XH2a")
    config = _config("YueHui", visual_config=visual, audio_config=audio)

    assert config.visual_config is visual
    assert config.audio_config is audio
    assert visual.chip_arch == "XH2a"
    assert audio.chip_arch == "XH2a"
    assert config.video_visual_config.chip_arch == "YueHui"


@pytest.mark.parametrize("chip_arch", ["YueHui", "XH2a"])
def test_unified_side_graphs_inherit_parent_chip(chip_arch: str) -> None:
    contract = _unified_contract()
    visual = XHGemma4SeriesModelConfig._build_unified_visual_config(
        {"quant_scheme": {"quant_type": "w8a8h1_sefp", "ops": {}}},
        model_name="gemma_4_12b_unified",
        hf_model=None,
        contract=contract,
        input_modality="image",
        quant_scheme={"quant_type": "w8a8h1_sefp", "ops": {}},
        chip_arch=chip_arch,
    )
    video = XHGemma4SeriesModelConfig._build_unified_visual_config(
        {},
        model_name="gemma_4_12b_unified",
        hf_model=None,
        contract=contract,
        input_modality="video",
        quant_scheme={"quant_type": "w8a8h1_sefp", "ops": {}},
        chip_arch=chip_arch,
    )
    audio = XHGemma4SeriesModelConfig._build_unified_audio_config(
        {},
        model_name="gemma_4_12b_unified",
        hf_model=None,
        contract=contract,
        quant_scheme={"quant_type": "w8a8h1_sefp", "ops": {}},
        chip_arch=chip_arch,
    )

    assert isinstance(visual, XHGemma4UnifiedVisualConfig)
    assert isinstance(video, XHGemma4UnifiedVisualConfig)
    assert isinstance(audio, XHGemma4UnifiedAudioConfig)
    assert visual.chip_arch == chip_arch
    assert video.chip_arch == chip_arch
    assert audio.chip_arch == chip_arch


def test_unified_explicit_chip_arch_is_not_overwritten() -> None:
    contract = _unified_contract()
    visual = XHGemma4SeriesModelConfig._build_unified_visual_config(
        {"chip_arch": "XH2a"},
        model_name="gemma_4_12b_unified",
        hf_model=None,
        contract=contract,
        input_modality="image",
        quant_scheme={"quant_type": "w8a8h1_sefp", "ops": {}},
        chip_arch="YueHui",
    )
    audio = XHGemma4SeriesModelConfig._build_unified_audio_config(
        {"chip_arch": "XH2a"},
        model_name="gemma_4_12b_unified",
        hf_model=None,
        contract=contract,
        quant_scheme={"quant_type": "w8a8h1_sefp", "ops": {}},
        chip_arch="YueHui",
    )

    assert visual.chip_arch == "XH2a"
    assert audio.chip_arch == "XH2a"
