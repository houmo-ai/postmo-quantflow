"""Contracts for Gemma4 Series YueHui eval workflows.

A variant may ship several YueHui YAMLs (attention or decoding variants);
every one of them must be registered here and match its model card.
"""

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
YUEHUI_GEMMA4 = REPO_ROOT / "configs_merak/workflows/yuehui/llm_models/gemma4_series"

CONFIGS = (
    {
        "variant": "e2b",
        "method": "autoround",
        "path": YUEHUI_GEMMA4 / "e2b/gemma4_e2b_autoround.yaml",
        "model_name": "gemma_4_e2b",
        "model_type": "Gemma4ForConditionalGeneration",
        "context_max_length": 131072,
        "padded_vit": True,
        "moe": False,
        "audio_config": False,
    },
    {
        "variant": "e4b",
        "method": "gptq",
        "path": YUEHUI_GEMMA4 / "e4b/gemma4_e4b_full.yaml",
        "model_name": "gemma_4_e4b",
        "model_type": "Gemma4ForConditionalGeneration",
        "context_max_length": 131072,
        "padded_vit": True,
        "moe": False,
        "audio_config": False,
    },
    {
        "variant": "26b_a4b",
        "method": "gptq",
        "path": YUEHUI_GEMMA4 / "26b_a4b/gemma4_26b_a4b_full.yaml",
        "model_name": "gemma_4_26b_a4b",
        "model_type": "Gemma4ForConditionalGeneration",
        "context_max_length": 262144,
        "padded_vit": True,
        "moe": True,
        "audio_config": False,
    },
    {
        "variant": "31b",
        "method": "autoround",
        "path": YUEHUI_GEMMA4 / "31b/gemma4_31b_autoround.yaml",
        "model_name": "gemma_4_31b",
        "model_type": "Gemma4ForConditionalGeneration",
        "context_max_length": 262144,
        "padded_vit": True,
        "moe": False,
        "audio_config": False,
    },
    {
        # FlashAttention on the full layers; sliding layers keep small-window matmul.
        "variant": "31b",
        "method": "autoround",
        "path": YUEHUI_GEMMA4 / "31b/gemma4_31b_autoround_flash_attention.yaml",
        "model_name": "gemma_4_31b",
        "model_type": "Gemma4ForConditionalGeneration",
        "context_max_length": 262144,
        "padded_vit": True,
        "moe": False,
        "audio_config": False,
    },
    {
        "variant": "12b_unified",
        "method": "gptq",
        "path": YUEHUI_GEMMA4 / "12b_unified/gemma4_12b_unified_full.yaml",
        "model_name": "gemma_4_12b_unified",
        "model_type": "Gemma4UnifiedForConditionalGeneration",
        "context_max_length": 2048,
        "padded_vit": False,
        "moe": False,
        "audio_config": True,
    },
)


def test_gemma4_yuehui_registers_every_workflow() -> None:
    """No YueHui gemma4 YAML escapes the per-config contract below."""
    expected_variants = {item["variant"] for item in CONFIGS}
    actual_dirs = {path.name for path in YUEHUI_GEMMA4.iterdir() if path.is_dir()}
    assert actual_dirs == expected_variants

    registered = {item["path"].resolve() for item in CONFIGS}
    on_disk = {path.resolve() for path in YUEHUI_GEMMA4.rglob("*.yaml")}
    assert on_disk == registered


@pytest.mark.parametrize("spec", CONFIGS, ids=lambda spec: spec["path"].stem)
def test_gemma4_yuehui_eval_workflow_matches_m50_card(spec: dict) -> None:
    config = yaml.safe_load(spec["path"].read_text(encoding="utf-8"))
    quant = config["quant"]
    model = config["export"]["model"]

    assert quant["algorithm"] == "gptqmodel"
    assert quant["method"] == spec["method"]
    assert quant["bits"] == 4
    assert quant["group_size"] == 64

    assert model["chip_arch"] == "YueHui"
    for side_name in ("visual_config", "video_visual_config", "audio_config"):
        side = model.get(side_name)
        if isinstance(side, dict):
            assert "chip_arch" not in side, (
                f"{spec['path'].name} {side_name} must inherit export.model.chip_arch"
            )
    assert model["model_type"] == spec["model_type"]
    assert model["model_name"] == spec["model_name"]
    assert model["context_max_length"] == spec["context_max_length"]
    assert model["prefill_chunk_length"] == 320
    assert model["sliding_kv_cache_input_mode"] == "slice_window"
    assert model["quant_scheme"]["quant_type"] == "w8a8h1_sefp"
    assert model["only_first_block"] is False
    assert ("audio_config" in model) is spec["audio_config"]

    if spec["padded_vit"]:
        assert model["visual_config"]["image_seq_length"] == 280
        assert model["visual_config"]["max_patches"] == 2520
        assert model["video_visual_config"]["image_seq_length"] == 70
        assert model["video_visual_config"]["max_patches"] == 630
    else:
        assert "image_seq_length" not in model["visual_config"]
        assert "max_patches" not in model["visual_config"]

    if spec["moe"]:
        assert quant["moe"]["routing"] == "bypass"
    else:
        assert "moe" not in quant
