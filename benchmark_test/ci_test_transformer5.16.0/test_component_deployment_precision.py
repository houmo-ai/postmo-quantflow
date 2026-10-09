"""Exercise graph precision through registered YAML and the real build config consumer."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
import yaml

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_build.workflow import _set_recipe_path, materialize_workflow_config
from xhmodel_merak.model_catalog import get_recipe, list_recipes, resolve_workflow_config
from xhmodel_merak.model_catalog.identity import build_artifact_name
from xhmodel_merak.model_catalog.precision import (
    deployment_precision_from_workflow,
    deployment_precision_replacements,
    resolve_deployment_precision,
)
from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig


def _workflows():
    for recipe in list_recipes():
        for variant in recipe.variants:
            paths = {variant.workflow_config, *(path for _, path in variant.context_workflows)}
            for path in sorted(paths):
                yield pytest.param(recipe, path, id=f"{recipe.model_id}/{variant.name}/{path.rsplit('/', 1)[-1]}")


def _get(data, path):
    for part in path.split("."):
        data = data[part]
    return data


@pytest.mark.parametrize("deployment,token", [
    ({"image": "w16a16_sefp"}, "w16a16"),
    ({"image": "w8a8_sefp"}, "w8a8"),
    ({"encode": "w8a8_sefp", "decode": "w8a8_sefp"}, "w8a8"),
    ({"encode": "w8a8_sefp", "decode": "w16a16_sefp"}, "wmix_amix"),
])
def test_artifact_template_uses_graph_precision_without_changing_checkpoint(deployment, token):
    original = get_recipe("qwen-image-2.1")
    recipe = replace(original, artifact_name_template="hmquant_xh2_image_{deployment_precision}_fixed")
    name = build_artifact_name(recipe, recipe.default_variant, recipe.export_name,
                               recipe.default_precision, None, None, deployment_precision=deployment)
    assert name == f"hmquant_xh2_image_{token}_fixed"
    assert recipe.default_precision.weight == "w16"


@pytest.mark.parametrize("template", ["hmquant_fixed", "../{deployment_precision}",
                                     "hm_{deployment_precision}_{deployment_precision}", "hm_{unknown}"])
def test_artifact_template_rejects_ambiguous_or_unsafe_names(template):
    with pytest.raises(ValueError, match="artifact name template"):
        replace(get_recipe("qwen-image-2.1"), artifact_name_template=template)


def test_artifact_template_cannot_conflict_with_a_fixed_name():
    with pytest.raises(ValueError, match="combined with a fixed artifact name"):
        replace(get_recipe("qwen-image-2.1"), fixed_artifact_name="hmquant_xh2_fixed")


@pytest.mark.parametrize("recipe,path", list(_workflows()))
def test_registered_workflow_precision_choices_preserve_operator_exceptions(recipe, path):
    original = yaml.safe_load(resolve_workflow_config(path).read_text())
    before = deepcopy(original)
    defaults = deployment_precision_from_workflow(recipe, original)
    assert deployment_precision_replacements(recipe, original, defaults) == {}
    for binding in recipe.deployment_precision_bindings:
        for choice in binding.choices:
            resolved = resolve_deployment_precision(recipe, {binding.components[0]: choice}, defaults=defaults)
            effective = deepcopy(original)
            for key, value in deployment_precision_replacements(recipe, original, resolved).items():
                _set_recipe_path(effective, key, value, model=recipe.model_id)
            assert deployment_precision_from_workflow(recipe, effective) == resolved
            for other in recipe.deployment_precision_bindings:
                try:
                    old_scheme = _get(original, other.config_path)
                except (KeyError, TypeError):
                    old_scheme = None
                if old_scheme is None:
                    old_scheme = _get(original, other.inherit_from)
                try:
                    new_scheme = _get(effective, other.config_path)
                except (KeyError, TypeError):
                    new_scheme = _get(effective, other.inherit_from)
                if new_scheme is None:
                    new_scheme = _get(effective, other.inherit_from)
                assert {k: v for k, v in old_scheme.items() if k != "quant_type"} == {
                    k: v for k, v in new_scheme.items() if k != "quant_type"
                }
                if other != binding:
                    assert resolved[other.components[0]] == defaults[other.components[0]]
    assert original == before


@pytest.mark.parametrize(
    "precision_request",
    [
        [],
        "w8a16h1_sefp",
        {"visual": None},
        {"visual": "fp32"},
        {"unknown": "w8a8h1_sefp"},
        {1: "w8a8h1_sefp"},
        {"prefill": "w8a8h1_sefp", "decode": "w8a16h1_sefp"},
    ],
)
def test_invalid_precision_requests_are_rejected(precision_request):
    with pytest.raises(ValueError):
        resolve_deployment_precision(get_recipe("qwen3.5-9b"), precision_request)


@pytest.mark.parametrize("model_id", ["kokoro-82m-v1.1-zh", "pi0.5-libero"])
def test_fixed_pipeline_precision_cannot_be_overridden(model_id):
    recipe = get_recipe(model_id)
    with pytest.raises(ValueError, match="not configurable"):
        resolve_deployment_precision(recipe, {recipe.components[0]: "w8a8h1_sefp"})
    assert deployment_precision_replacements(recipe, {}, recipe.deployment_precision) == {}


def test_gemma_inherited_audio_does_not_follow_language_override():
    recipe = get_recipe("gemma4-e2b")
    workflow = yaml.safe_load(resolve_workflow_config(recipe.default_variant.workflow_config).read_text())
    model = workflow["export"]["model"]
    # Exercise the exporter's supported inheritance, independent of explicit YAML defaults.
    model.pop("audio_config", None)
    model.pop("video_visual_config", None)
    defaults = deployment_precision_from_workflow(recipe, workflow)
    resolved = resolve_deployment_precision(recipe, {"decode": "w8a16h1_sefp"}, defaults=defaults)
    for key, value in deployment_precision_replacements(recipe, workflow, resolved).items():
        _set_recipe_path(workflow, key, value, model=recipe.model_id)
    assert model["quant_scheme"]["quant_type"] == "w8a16h1_sefp"
    assert model["audio_config"]["quant_scheme"]["quant_type"] == defaults["audio"]
    assert model["video_visual_config"]["quant_scheme"] == model["visual_config"]["quant_scheme"]
    assert resolved["prefill"] == resolved["decode"] == "w8a16h1_sefp"


@pytest.mark.parametrize("precision_request", [{"visual": "w8a16h1_sefp"}, {"decode": "w8a16h1_sefp"}])
def test_public_precision_reaches_real_workflow_config(tmp_path, precision_request):
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
    plan = resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(source),
        output_dir=str(tmp_path / "artifact"),
        deployment_precision=precision_request,
        device="cpu",
    )
    path = tmp_path / "effective.yaml"
    materialize_workflow_config(plan, path)
    config = WorkflowConfig.from_file(str(path)).with_overrides(plan.config_overrides)
    assert config.export["model"]["visual_config"]["quant_scheme"]["quant_type"] == plan.deployment_precision["visual"]
    assert config.export["model"]["quant_scheme"]["quant_type"] == plan.deployment_precision["decode"]
    assert plan.deployment_precision["prefill"] == plan.deployment_precision["decode"]
    assert plan.precision.weight == "w4"
    with pytest.raises(ModelError) as caught:
        resolve_build_plan(
            model_id="qwen3.5-9b",
            model_dir=str(source),
            output_dir=str(tmp_path / "artifact"),
            deployment_precision={"visual": "invalid"},
            device="cpu",
        )
    assert caught.value.code == ErrorCode.INVALID_PARAMETER
