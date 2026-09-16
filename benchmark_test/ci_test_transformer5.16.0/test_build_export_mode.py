"""Public export mode resolution and the environment consumed by the exporter."""

import json
import os
from dataclasses import replace

import pytest

from xhmodel_merak._environment import scoped_environment
from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_build import describe_model
from xhmodel_merak.model_build.resolver import _resolve_low_memory, resolve_build_plan
from xhmodel_merak.model_catalog import get_recipe, list_recipes
from xhmodel_merak.xh_llm.utils import is_huge_model_export_enabled


@pytest.mark.parametrize("recipe", list_recipes(), ids=lambda recipe: recipe.model_id)
def test_every_registered_variant_resolves_only_its_export_modes(recipe):
    for variant in recipe.variants:
        assert _resolve_low_memory(recipe.model_id, variant, None) == (variant.export_modes[0] == "low_memory")
        for enabled in (False, True):
            mode = "low_memory" if enabled else "standard"
            if mode in variant.export_modes:
                assert _resolve_low_memory(recipe.model_id, variant, enabled) is enabled
            else:
                with pytest.raises(ModelError) as error:
                    _resolve_low_memory(recipe.model_id, variant, enabled)
                assert error.value.code == ErrorCode.UNSUPPORTED_COMBINATION


@pytest.mark.parametrize("value", [0, 1, "true", "false", [], {}])
def test_export_mode_rejects_non_bool_requests(value):
    with pytest.raises(ModelError) as error:
        _resolve_low_memory("qwen3.5-9b", get_recipe("qwen3.5-9b").default_variant, value)
    assert error.value.code == ErrorCode.INVALID_PARAMETER


@pytest.mark.parametrize("modes", [(), ("auto",), ("standard", "standard")])
def test_variant_rejects_invalid_export_modes(modes):
    with pytest.raises(ValueError, match="export_modes"):
        replace(get_recipe("qwen3.5-9b").default_variant, export_modes=modes)


@pytest.mark.parametrize("inherited", [None, "0", "1"])
@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_public_plan_binds_actual_exporter_switch_and_restores_environment(
    tmp_path, monkeypatch, inherited, enabled, fail
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"hidden_size": 4096, "num_hidden_layers": 32},
            }
        )
    )
    variable = "HUGE_MODEL_EXPORT_ENABLED"
    if inherited is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, inherited)
    plan = resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(source),
        output_dir=str(tmp_path / "output"),
        low_memory=enabled,
    )
    assert os.environ.get(variable) == inherited
    assert plan.low_memory is bool(enabled)
    assert plan.field_sources["low_memory"] == ("recipe_default" if enabled is None else "user_explicit")
    try:
        with scoped_environment(plan.environment):
            assert is_huge_model_export_enabled() is bool(enabled)
            if fail:
                raise RuntimeError("export failed")
    except RuntimeError:
        assert fail
    assert os.environ.get(variable) == inherited


def test_lora_rejected_before_loading_adapter_weights(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(ModelError, match="Export mode is not registered"):
        resolve_build_plan(
            model_id="qwen3.6-35b-a3b",
            model_dir=str(source),
            output_dir=str(tmp_path / "output"),
            low_memory=True,
            lora={"paths": [str(adapter)]},
        )


def test_description_explains_variant_defaults_and_does_not_advertise_gemma():
    qwen = describe_model("qwen3.6-35b-a3b")["low_memory"]
    assert qwen["default"] is None
    assert qwen["variants"]["full"] == {"default": False, "choices": [False, True]}
    assert qwen["variants"]["lora"] == {"default": False, "choices": [False]}
    assert describe_model("ling-3.0-flash")["low_memory"]["variants"]["full"]["default"] is True
    assert "low_memory" not in describe_model("gemma4-e2b")
