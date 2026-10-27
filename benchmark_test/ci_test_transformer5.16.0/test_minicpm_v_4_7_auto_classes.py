"""Real HF loads with MiniCPM Auto mappings absent, without full checkpoints."""

from __future__ import annotations

import sys

from _minicpm_v47_fixture import minicpmv47_modeling  # noqa: F401

import pytest


@pytest.fixture
def auto_environment(monkeypatch):
    from transformers import AutoModelForImageTextToText
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.minicpmv4_7 import modeling_minicpmv4_7 as modeling

    # Isolate the process-wide registries, including the helper's additions.
    monkeypatch.setattr(CONFIG_MAPPING, "_extra_content", dict(CONFIG_MAPPING._extra_content))
    mapping = AutoModelForImageTextToText._model_mapping
    monkeypatch.setattr(mapping, "_extra_content", dict(mapping._extra_content))
    return modeling, CONFIG_MAPPING, mapping


def _remove_auto_mappings(monkeypatch, environment, *, local):
    modeling, configs, models = environment
    config_class = modeling.MiniCPMV4_7Config
    model_type = config_class.model_type
    monkeypatch.delitem(configs._extra_content, model_type, raising=False)
    monkeypatch.delitem(configs._mapping, model_type, raising=False)
    monkeypatch.delitem(models._extra_content, config_class, raising=False)
    monkeypatch.delitem(models._model_mapping, model_type, raising=False)
    if local:
        # Keep the real implementation and weights. Only module availability
        # and class provenance emulate a locally supplied Python module.
        monkeypatch.setattr(config_class, "__module__", "configuration_minicpmv4_7")
        monkeypatch.setitem(sys.modules, "modeling_minicpmv4_7", modeling)
        monkeypatch.setitem(
            sys.modules, "transformers.models.minicpmv4_7.modeling_minicpmv4_7", None
        )


def _tiny_model(modeling, *, moe):
    text = dict(
        model_type="qwen3_5_moe_text" if moe else "qwen3_5_text",
        vocab_size=128, hidden_size=128, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        layer_types=["full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32,
        rope_parameters=dict(
            rope_theta=10000000., partial_rotary_factor=.25,
            mrope_interleaved=True, mrope_section=[3, 3, 2],
        ),
        pad_token_id=0, eos_token_id=127, tie_word_embeddings=False,
    )
    if moe:
        text.update(num_experts=4, num_experts_per_tok=2,
                    moe_intermediate_size=64, shared_expert_intermediate_size=64)
    else:
        text["intermediate_size"] = 256
    config = modeling.MiniCPMV4_7Config(
        text_config=text,
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, image_size=56, patch_size=14,
        ),
        insert_layer_id=0, image_token_id=125, video_token_id=126,
        image_size=56, tie_word_embeddings=False,
    )
    return modeling.MiniCPMV4_7ForConditionalGeneration(config).eval()


@pytest.mark.parametrize("local", (False, True), ids=("native", "local"))
@pytest.mark.parametrize("moe", (False, True), ids=("dense", "moe"))
def test_text_loader_registers_before_auto_config(
    auto_environment, monkeypatch, tmp_path, restore_model_initialization_state, local, moe
):
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.auto_classes import (
        register_minicpmv47_auto_classes,
    )
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.text_model import (
        MiniCPMV47MoeTextModel, MiniCPMV47TextModel,
    )

    modeling = auto_environment[0]
    torch.manual_seed(47)
    native = _tiny_model(modeling, moe=moe)
    native.save_pretrained(tmp_path)
    _remove_auto_mappings(monkeypatch, auto_environment, local=local)
    with pytest.raises(ValueError, match="minicpmv4_7"):
        AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    with pytest.raises(ValueError, match="Unrecognized configuration"):
        AutoModelForImageTextToText.from_config(native.config)

    adapter = MiniCPMV47MoeTextModel if moe else MiniCPMV47TextModel
    with restore_model_initialization_state():
        loaded = adapter.get_hf_model(
            str(tmp_path), dtype=torch.float32, local_files_only=True,
            attn_implementation="eager",
        )
    assert AutoConfig.from_pretrained(tmp_path).__class__ is native.config.__class__
    assert AutoModelForImageTextToText._model_mapping[type(native.config)] is type(native)
    assert loaded.config.text_config.model_type == native.config.text_config.model_type
    expected = native.model.language_model.state_dict()
    actual = loaded.model.language_model.state_dict()
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(loaded.lm_head.weight, native.lm_head.weight, rtol=0, atol=0)

    # Repeated entry through a processor or another vision gear must be safe.
    register_minicpmv47_auto_classes()
    register_minicpmv47_auto_classes()
    assert AutoModelForImageTextToText._model_mapping[type(native.config)] is type(native)


def test_missing_implementation_reports_installation_requirement(auto_environment, monkeypatch):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.auto_classes import (
        register_minicpmv47_auto_classes,
    )

    monkeypatch.setitem(sys.modules, "transformers.models.minicpmv4_7.modeling_minicpmv4_7", None)
    monkeypatch.setitem(sys.modules, "modeling_minicpmv4_7", None)
    with pytest.raises(ImportError, match="Auto registration alone cannot supply"):
        register_minicpmv47_auto_classes()


@pytest.mark.parametrize("local", (False, True), ids=("native", "local"))
def test_dependency_import_errors_are_preserved(auto_environment, monkeypatch, local):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7 import auto_classes

    failure = ModuleNotFoundError("missing model dependency", name="missing_model_dependency")

    def unavailable(name):
        if local and name.startswith("transformers."):
            raise ModuleNotFoundError(name=name)
        raise failure

    monkeypatch.setattr(auto_classes, "import_module", unavailable)
    with pytest.raises(ModuleNotFoundError) as error:
        auto_classes.register_minicpmv47_auto_classes()
    assert error.value is failure
