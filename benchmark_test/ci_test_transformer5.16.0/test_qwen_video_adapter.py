# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace

import pytest

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_catalog.registry import get_recipe
from xhmodel_merak.model_inference.inputs import validate_inputs
from xhmodel_merak.model_integrations.qwen.inference import adapter


@pytest.mark.parametrize("model", ["qwen3.5-9b", "qwen3.6-27b", "qwen3.6-35b-a3b"])
def test_video_recipe_reuses_visual_component(model):
    recipe = get_recipe(model)
    assert "video" in recipe.modalities
    assert "video" in recipe.input_schema["properties"]
    assert recipe.components == ("visual", "prefill", "decode")
    assert any("torchcodec==0.7.0" in item for item in recipe.dependencies.system)
    assert recipe.dependencies.inference == ("torchcodec==0.7.0; extra == 'video'",)
    with pytest.raises(ModelError):
        validate_inputs(recipe, {"prompt": "hello", "image": "a.png", "video": "b.mp4"})


@pytest.mark.parametrize("variant", ["full", "mtp", "dflash"])
def test_adapter_routes_video_directory_and_rejects_speculation(tmp_path, monkeypatch, variant):
    from xhmodel_merak.xh_llm.models.qwen3_5 import hmonnx_validation
    calls = []
    def generate(meta, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(latency_s=1, output_text="ok", output_tokens=2, tokens_per_second=2)
    monkeypatch.setattr(hmonnx_validation, "quick_test_hmonnx", generate)
    kwargs = dict(recipe=get_recipe("qwen3.6-35b-a3b"), root=tmp_path,
                  manifest={"recipe": {"variant": variant}}, meta_file=tmp_path / "runtime_meta_info.json",
                  inputs={"prompt": "hello", "video": str(tmp_path), "max_new_tokens": 2}, device="cpu",
                  components=("visual", "prefill", "decode"), selected_golden=(), golden_artifacts={})
    if variant != "full":
        with pytest.raises(ModelError, match="text input only"):
            adapter.run(**kwargs)
        assert not calls
    else:
        assert adapter.run(**kwargs).output_tokens == 2
        assert calls[0]["video_path"] == str(tmp_path)


def test_video_fixture_is_private_to_qwen36_full(tmp_path, monkeypatch):
    from _unified_cases import CASES
    from _unified_fixtures.qwen36_video import prepare
    from transformers import AutoVideoProcessor
    case = next(case for case in CASES if case.model_id == "qwen3.6-35b-a3b" and case.variant == "full")
    kwargs, inputs = prepare(case, tmp_path, monkeypatch)
    assert Path(tmp_path / "video.mp4").is_file()
    assert "video" not in inputs  # Text inference remains covered before the video assertion.
    processor = AutoVideoProcessor.from_pretrained(tmp_path / "source")
    assert processor.patch_size == 16 and processor.temporal_patch_size == 2


def test_video_decoder_requirement_does_not_block_text_or_frame_directory(tmp_path, monkeypatch):
    import xhmodel_merak._requirements as requirements
    from xhmodel_merak.xh_llm.models.qwen3_5 import hmonnx_validation
    original = requirements._installed_distribution_version
    monkeypatch.setattr(requirements, "_installed_distribution_version",
                        lambda name: None if name == "torchcodec" else original(name))
    monkeypatch.setattr(requirements, "_installed_module_version", lambda name: None)
    monkeypatch.setattr(hmonnx_validation, "quick_test_hmonnx", lambda *args, **kwargs:
                        SimpleNamespace(latency_s=1, output_text="ok", output_tokens=2, tokens_per_second=2))
    kwargs = dict(recipe=get_recipe("qwen3.6-35b-a3b"), root=tmp_path,
                  manifest={"recipe": {"variant": "full"}}, meta_file=tmp_path / "meta.json", device="cpu",
                  components=("visual", "prefill", "decode"), selected_golden=(), golden_artifacts={})
    for inputs in ({"prompt": "hello"}, {"prompt": "hello", "video": str(tmp_path)}):
        assert adapter.run(**kwargs, inputs=inputs).output_tokens == 2
    video = tmp_path / "video.mp4"
    video.touch()
    with pytest.raises(ModelError, match="registered dependency profile"):
        adapter.run(**kwargs, inputs={"prompt": "hello", "video": str(video)})
