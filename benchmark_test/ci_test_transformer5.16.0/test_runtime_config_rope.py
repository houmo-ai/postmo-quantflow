"""Checkpoint comparison accepts explicit duplicate RoPE values, never guessed defaults."""

from copy import deepcopy

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from xhmodel_merak.model_build.runtime_assets import behavior_config_mismatches


def _source():
    return {
        "model_type": "qwen3_5_text",
        "head_dim": 256,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_rope_duplicate_matches_native_config_and_rotary_frequencies(reverse):
    source = _source()
    checkpoint = {**deepcopy(source), "partial_rotary_factor": 0.25}
    a = Qwen3_5TextRotaryEmbedding(Qwen3_5TextConfig(**deepcopy(source)))
    b = Qwen3_5TextRotaryEmbedding(Qwen3_5TextConfig(**deepcopy(checkpoint)))
    torch.testing.assert_close(a.inv_freq, b.inv_freq, rtol=0, atol=0)
    expected, actual = (checkpoint, source) if reverse else (source, checkpoint)
    assert behavior_config_mismatches({"text_config": expected}, {"text_config": actual}) == []


@pytest.mark.parametrize("mutation", ["conflicting_duplicate", "changed_nested", "missing_nested", "guessed_default"])
def test_changed_or_unproven_rope_factor_remains_rejected(mutation):
    source = _source()
    checkpoint = {**deepcopy(source), "partial_rotary_factor": 0.25}
    if mutation == "conflicting_duplicate":
        checkpoint["partial_rotary_factor"] = 0.5
    elif mutation == "changed_nested":
        checkpoint["rope_parameters"]["partial_rotary_factor"] = 0.5
    elif mutation == "missing_nested":
        checkpoint.pop("rope_parameters")
    else:
        source["rope_parameters"].pop("partial_rotary_factor")
        checkpoint["rope_parameters"].pop("partial_rotary_factor")
    assert behavior_config_mismatches(source, checkpoint)


def test_rope_comparison_does_not_modify_configs_or_ignore_other_critical_fields():
    source = _source()
    checkpoint = {**deepcopy(source), "partial_rotary_factor": 0.25, "sliding_window": 1024}
    before = deepcopy((source, checkpoint))
    mismatches = behavior_config_mismatches(source, checkpoint)
    assert [item["field"] for item in mismatches] == ["sliding_window"]
    assert (source, checkpoint) == before


def _next_source():
    return {
        "model_type": "qwen3_next",
        "num_hidden_layers": 8,
        "full_attention_interval": 4,
        "rope_theta": 10000000,
        "rope_scaling": None,
        "partial_rotary_factor": 0.25,
    }


def _next_checkpoint():
    result = _next_source()
    result.pop("rope_theta")
    result.pop("rope_scaling")
    result["rope_parameters"] = {"rope_theta": 10000000, "rope_type": "default", "partial_rotary_factor": 0.25}
    result["layer_types"] = ["linear_attention"] * 3 + ["full_attention"]
    result["layer_types"] *= 2
    return result


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_legacy_next_config_matches_native_rope_and_layer_pattern(reverse):
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

    from xhmodel_merak.model_build.checkpoint import _model_config_identity

    source, checkpoint = _next_source(), _next_checkpoint()
    before = deepcopy((source, checkpoint))
    a, b = Qwen3NextConfig(**deepcopy(source)), Qwen3NextConfig(**deepcopy(checkpoint))
    assert a.layer_types == b.layer_types
    torch.testing.assert_close(
        Qwen3NextRotaryEmbedding(a).inv_freq, Qwen3NextRotaryEmbedding(b).inv_freq, rtol=0, atol=0
    )
    assert _model_config_identity(source) == _model_config_identity(checkpoint)
    assert behavior_config_mismatches(*(reversed(before) if reverse else before)) == []
    assert (source, checkpoint) == before


@pytest.mark.parametrize(
    "mutation", ["theta", "scaled", "missing_theta", "missing_scaling", "layer", "interval", "missing_interval"]
)
def test_next_config_normalization_rejects_changes_and_missing_evidence(mutation):
    source, checkpoint = _next_source(), _next_checkpoint()
    if mutation == "theta":
        checkpoint["rope_parameters"]["rope_theta"] = 10000
    elif mutation == "scaled":
        source["rope_scaling"] = {"rope_type": "linear", "factor": 2.0}
    elif mutation == "missing_theta":
        checkpoint["rope_parameters"].pop("rope_theta")
    elif mutation == "missing_scaling":
        source.pop("rope_scaling")
    elif mutation == "layer":
        checkpoint["layer_types"][0] = "full_attention"
    elif mutation == "interval":
        checkpoint["full_attention_interval"] = 2
    else:
        source.pop("full_attention_interval")
    assert behavior_config_mismatches(source, checkpoint)


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_composite_use_cache_duplicate_requires_explicit_equal_text_value(value, reverse):
    source = {"text_config": {"use_cache": value}, "use_cache": value}
    checkpoint = {"text_config": {"use_cache": value}}
    pair = (checkpoint, source) if reverse else (source, checkpoint)
    assert behavior_config_mismatches(*pair) == []
    checkpoint["text_config"]["use_cache"] = not value
    assert behavior_config_mismatches(*pair)
    checkpoint["text_config"].pop("use_cache")
    assert behavior_config_mismatches(*pair)


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_composite_cache_duplicate_is_not_ignored(reverse):
    source = {"text_config": {"use_cache": False}, "use_cache": True}
    checkpoint = {"text_config": {"use_cache": False}}
    assert behavior_config_mismatches(*((checkpoint, source) if reverse else (source, checkpoint)))


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_recipe_owned_vision_type_alias_matches_native_config(moe, reverse):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

    from xhmodel_merak.model_catalog import get_recipe

    name = "qwen3_5_moe" if moe else "qwen3_5"
    cls = Qwen3_5MoeConfig if moe else Qwen3_5Config
    recipe = get_recipe("qwen3.6-35b-a3b" if moe else "qwen3.8-27b")
    source = {"vision_config": {"model_type": name, "hidden_size": 128}}
    checkpoint = {"vision_config": {"model_type": name + "_vision", "hidden_size": 128}}
    a, b = cls(**deepcopy(source)), cls(**deepcopy(checkpoint))
    assert type(a.vision_config) is type(b.vision_config)
    assert a.vision_config.to_dict() == b.vision_config.to_dict()
    expected, actual = (checkpoint, source) if reverse else (source, checkpoint)
    assert behavior_config_mismatches(expected, actual)
    assert behavior_config_mismatches(expected, actual, type_aliases=recipe.config_type_aliases) == []
    checkpoint["vision_config"]["hidden_size"] = 256
    assert behavior_config_mismatches(expected, actual, type_aliases=recipe.config_type_aliases)
    checkpoint["vision_config"].update(hidden_size=128, model_type="unrelated_vision")
    assert behavior_config_mismatches(expected, actual, type_aliases=recipe.config_type_aliases)
    assert behavior_config_mismatches(
        {"model_type": name}, {"model_type": name + "_vision"}, type_aliases=recipe.config_type_aliases
    )


@pytest.mark.parametrize(
    "aliases",
    [
        (("model_type", "a", "b"),),
        (("vision_config.hidden_size", "a", "b"),),
        (("vision_config.model_type", "a", "a"),),
        (("vision_config.model_type", "a", "b"), ("vision_config.model_type", "a", "c")),
    ],
)
def test_recipe_type_aliases_reject_ambiguous_or_non_type_rules(aliases):
    from dataclasses import replace

    from xhmodel_merak.model_catalog import get_recipe

    with pytest.raises(ValueError, match="config type aliases"):
        replace(get_recipe("qwen3.8-27b"), config_type_aliases=aliases)
