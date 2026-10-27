import hashlib
import json
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from xhmodel_merak.model_build import describe_model, export_model_dependencies
from xhmodel_merak.model_catalog import get_recipe, list_recipes
from xhmodel_merak.model_catalog.dependencies import DependencySpec, dependency_profile, merge_requirements
from xhmodel_merak.model_integrations._dependencies import GPTQ_REQUIREMENTS, runtime_dependencies


@pytest.mark.parametrize("package", ["numpy", "datasets"])
def test_export_package_requirements_match_tested_quantization_versions(package):
    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    export = {
        requirement.name: requirement
        for value in project["project"]["optional-dependencies"]["export"]
        for requirement in [Requirement(value)]
    }
    tested = {
        requirement.name: requirement
        for value in (*runtime_dependencies().common, *GPTQ_REQUIREMENTS)
        for requirement in [Requirement(value)]
    }
    version = next(iter(tested[package].specifier)).version
    assert version in export[package].specifier, (
        f"export extra rejects tested {package}=={version}: {export[package]}"
    )


def test_all_registered_variants_have_explicit_dependency_profiles():
    assert list_recipes(), "The public model catalog must not be empty"
    for recipe in list_recipes():
        description = describe_model(recipe.model_id, full=True)
        profiles = description["dependencies"]["profiles"]
        assert set(profiles) == {v.name for v in recipe.variants}
        for variant in description["workflow_variants"]:
            profile = profiles[variant["name"]]
            assert variant["dependencies"] == profile
            assert profile["build"]["pip"] and profile["inference"]["pip"]
            assert profile["status"] == "blocked_distribution"
            assert not profile["validation"]["clean_install"]
            assert set(profile["quantization"]) == set(recipe.quant_methods)
        assert "dependencies" not in describe_model(recipe.model_id)


def test_dependencies_are_the_only_recipe_version_declaration():
    for recipe in list_recipes():
        description = describe_model(recipe.model_id, full=True)
        for obsolete in ("worker_requirements", "quant_requirements", "quant_checkpoint_requirements"):
            assert not hasattr(recipe, obsolete)
            assert obsolete not in description


def test_query_does_not_load_heavy_dependencies():
    code = """
import sys
from xhmodel_merak.model_build import list_models, describe_model
for model in list_models():
    describe_model(model['model_id'], full=True)
assert not {'torch', 'xhquant', 'transformers', 'gptqmodel'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_phase_and_variant_dependencies_are_separate():
    profiles = describe_model("gemma4-12b-unified", full=True)["dependencies"]["profiles"]
    assert "torchcodec==0.7.0" in profiles["full"]["inference"]["pip"]
    assert "torchcodec==0.7.0" not in profiles["mtp"]["inference"]["pip"]
    assert "torchcodec==0.7.0" in profiles["mtp"]["build"]["pip"]
    assert not any("gptqmodel" in r for r in profiles["full"]["inference"]["pip"])
    assert any("gptqmodel" in r for r in profiles["full"]["quantization"]["gptq"]["pip"])
    assert "datasets==3.6.0" in profiles["full"]["quantization"]["gptq"]["pip"]
    assert "datasets==3.6.0" not in profiles["full"]["inference"]["pip"]
    pi = describe_model("pi0.5-libero", full=True)["dependencies"]["profiles"]["lerobot-h50"]
    assert "transformers==5.16.1" in pi["inference"]["pip"]
    assert "lerobot==0.6.0" in pi["inference"]["pip"]


def test_variant_requirements_merge_without_mutating_recipe():
    recipe = get_recipe("qwen3.5-0.8b")
    variant = replace(recipe.default_variant, dependencies=DependencySpec(build=("peft==0.19.1",)))
    profile = dependency_profile(recipe, variant)
    assert "peft==0.19.1" in profile["build"]["pip"]
    assert "peft==0.19.1" not in profile["inference"]["pip"]
    assert not recipe.default_variant.dependencies.build


@pytest.mark.parametrize(
    "requirements",
    [
        ("torch==2.8.0", "torch>=2.9"),
        ("some-pkg==1", "some_pkg==2"),
        ("thing @ file:///tmp/thing.whl",),
        ("thing @ https://user:password@example.com/thing.whl",),
        ("thing @ git+https://example.com/thing.git@main",),
    ],
)
def test_invalid_or_conflicting_requirements_fail(requirements):
    with pytest.raises(ValueError):
        merge_requirements(requirements)


def test_markers_extras_and_remote_sources_are_preserved():
    values = merge_requirements(
        (
            'thing==1; python_version < "3.12"',
            'thing==2; python_version >= "3.12"',
            "misaki[en,zh]==0.9.4",
            "other @ https://example.com/other.whl",
        )
    )
    assert len(values) == 4
    assert "misaki[en,zh]==0.9.4" in values


def test_export_is_portable_hashed_and_refuses_overwrite(tmp_path):
    root = tmp_path / "deps"
    result = export_model_dependencies(model_id="gemma4-12b-unified", output_dir=str(root))
    assert json.loads((root / "environment.json").read_text()) == result
    assert result["variant"] == "full"
    for phase in [result[k] for k in ("build", "inference", "checkpoint_build")] + list(
        result["quantization"].values()
    ):
        path = root / phase["requirements_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == phase["sha256"]
        assert str(tmp_path) not in path.read_text()
    with pytest.raises(FileExistsError):
        export_model_dependencies(model_id="gemma4-12b-unified", output_dir=str(root))
    with pytest.raises(ValueError, match="Unknown variant"):
        export_model_dependencies(model_id="gemma4-12b-unified", variant="../invalid", output_dir=str(tmp_path / "bad"))
    assert not (tmp_path / "bad").exists()


def test_qwen38_exposes_dflash2_dependency_profile():
    description = describe_model("qwen3.8-27b", full=True)
    assert set(description["dependencies"]["profiles"]) == {"full", "mtp", "dflash2"}


def test_registered_dependencies_pin_tested_versions():
    for recipe in list_recipes():
        transformers_version = "5.18.0" if recipe.family == "minicpm-v-4.7" else "5.16.1"
        for variant in recipe.variants:
            profile = dependency_profile(recipe, variant)
            assert profile["python"] == "==3.12.13"
            for phase in (
                profile["build"],
                profile["inference"],
                profile["checkpoint_build"],
                *profile["quantization"].values(),
            ):
                assert [r for r in phase["pip"] if r.startswith("transformers==")] == [
                    f"transformers=={transformers_version}"
                ]
                for requirement in phase["pip"]:
                    parsed = Requirement(requirement)
                    if parsed.url:
                        assert parsed.url.startswith("git+https://")
                        revision = parsed.url.rsplit("@", 1)[-1]
                        assert len(revision) == 40 and all(char in "0123456789abcdef" for char in revision)
                        continue
                    pins = list(parsed.specifier)
                    assert len(pins) == 1 and pins[0].operator == "==", requirement
                    assert "*" not in pins[0].version, requirement


def test_variant_quantization_and_checkpoint_dependencies_are_merged():
    recipe = get_recipe("qwen3.5-0.8b")
    variant = replace(
        recipe.default_variant,
        dependencies=DependencySpec(
            quantization=(("autoround", ("quant-helper==1.0",)),),
            checkpoint_build=("checkpoint-helper==2.0",),
        ),
    )
    profile = dependency_profile(recipe, variant)
    assert "quant-helper==1.0" in profile["quantization"]["autoround"]["pip"]
    assert "checkpoint-helper==2.0" in profile["checkpoint_build"]["pip"]
    for phase in ("build", "inference"):
        assert "quant-helper==1.0" not in profile[phase]["pip"]
        assert "checkpoint-helper==2.0" not in profile[phase]["pip"]
    conflicting = replace(
        variant,
        dependencies=DependencySpec(
            quantization=(("autoround", ("gptqmodel==0.0.1",)),),
        ),
    )
    with pytest.raises(ValueError, match="Conflicting dependency versions"):
        dependency_profile(recipe, conflicting)
