# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Customer size plans exercise real capacity rules and an offline tokenizer."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from examples_merak.aigc.qwen_image21.capacity_plan import describe_capacity, load_config, main


EXAMPLES = Path(__file__).resolve().parents[2] / "examples_merak/aigc/qwen_image21/capacity_configs"


@pytest.fixture
def processor():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    special = ["[UNK]", "<|im_start|>", "<|im_end|>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]
    backend = Tokenizer(models.WordLevel(dict(zip(special, range(len(special)), strict=True)), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return SimpleNamespace(
        tokenizer=PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", additional_special_tokens=special[1:]
        )
    )


@pytest.mark.parametrize(
    "name,target,reference,feature_changed,graphs",
    [
        (
            "output_2048_reference_1024",
            16384,
            4352,
            False,
            {"vae_encode", "vae_decode", "denoise_text", "denoise_image"},
        ),
        (
            "output_1024_reference_2048",
            4352,
            16384,
            True,
            {"text_encode", "visual", "vae_encode", "vae_decode", "denoise_text", "denoise_image"},
        ),
        (
            "output_2048_reference_2048",
            16384,
            16384,
            True,
            {"text_encode", "visual", "vae_encode", "vae_decode", "denoise_text", "denoise_image"},
        ),
        ("output_512x2048_reference_1024", 4352, 4352, False, {"vae_encode", "vae_decode"}),
    ],
)
def test_customer_examples_separate_target_reference_and_canvas(
    processor, name, target, reference, feature_changed, graphs
):
    config = load_config(EXAMPLES / (name + ".yaml"))
    baseline = {
        **config,
        "output_size": [1024, 1024],
        "reference_size": [1024, 1024],
        "canvas_size": [1536, 1536],
        "target_capacity": 4352,
        "reference_capacity": 4352,
        "visual_patch_capacity": 4352,
    }
    plan = describe_capacity(config, processor, baseline=baseline, reference_input=(2048, 2048))
    original = describe_capacity(baseline, processor)
    assert plan["deployment_config"]["target_capacity"] == target
    assert plan["deployment_config"]["reference_capacity"] == reference
    assert (plan["text_budget"]["feature_capacity"] != original["text_budget"]["feature_capacity"]) == feature_changed
    assert plan["text_budget"]["dit_text_capacity"] == original["text_budget"]["dit_text_capacity"]
    assert plan["deployment_config"]["max_position"] == 32768
    assert set(plan["changed_graph_boundaries"]) == graphs
    assert plan["dit_gear"]["image_tokens"] == max(target, reference)
    assert plan["dit_gear"]["cache_tokens"] == plan["text_budget"]["prefix_capacity"] + target
    expected_reference = [2048, 2048] if reference == 16384 else [1024, 1024]
    assert plan["reference_input"]["fitted_hw"] == expected_reference
    assert plan["status"] == "configuration_checked_not_exported"


def test_image_cache_can_exceed_text_rope_position_table(processor):
    config = load_config(EXAMPLES / "output_2048_reference_2048.yaml")
    plan = describe_capacity(config, processor)
    assert plan["dit_gear"]["cache_tokens"] > 32768
    assert plan["text_budget"]["feature_capacity"] < 32768


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")), ids=lambda path: path.stem)
def test_example_enters_real_configured_workflow_without_separate_checkpoint_quantization(path, tmp_path):
    from xhmodel_merak.workflows import AutoWorkflow

    workflow = AutoWorkflow.from_config(model_dir=str(tmp_path), config_path=str(path))
    assert type(workflow).__name__ == "QwenImage21Workflow"
    result = workflow.quant(output_dir=str(tmp_path / "output"), device="cpu")
    assert result.skipped and result.raw_model_dir == str(tmp_path)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"target_capacity": 4352}, "token capacity"),
        ({"canvas_size": [1536, 1536]}, "spatial canvas"),
        ({"feature_capacity": 99999}, "derived"),
        ({"max_position": 64}, "max_position"),
    ],
)
def test_invalid_expansion_is_rejected(processor, change, match):
    config = load_config(EXAMPLES / "output_2048_reference_1024.yaml")
    with pytest.raises(ValueError, match=match):
        describe_capacity({**config, **change}, processor)


def test_inspection_cli_uses_saved_offline_processor_and_writes_report(processor, tmp_path, capsys):
    source = tmp_path / "source"
    processor.tokenizer.save_pretrained(source / "processor")
    # A real saved tokenizer is sufficient; no model weights or GPU are needed.
    config_path = tmp_path / "custom.yaml"
    config = load_config(EXAMPLES / "output_2048_reference_1024.yaml")
    config_path.write_text(yaml.safe_dump({"quant": None, "export": config}))
    report_path = tmp_path / "report.json"
    main(["--model-dir", str(source), "--config", str(config_path), "--report", str(report_path)])
    report = json.loads(report_path.read_text())
    assert report["plans"][0]["deployment_config"]["target_capacity"] == 16384
    assert "not_exported" in capsys.readouterr().out
