from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from _recipe_coverage import MODEL_ROOTS, check_recipe_coverage, load_legacy_gaps, recipe_gaps, scan_implementations

from xhmodel_merak.model_catalog.contracts import VariantRecipe
from xhmodel_merak.model_catalog.discovery import build_builtin_catalog
from xhmodel_merak.model_catalog.registry import get_recipe


REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.ci_policy


def test_all_merak_models_have_registered_recipes():
    # Fresh discovery matters: a plugin file on disk is not itself registration.
    check_recipe_coverage(REPO_ROOT, build_builtin_catalog().list_recipes(), load_legacy_gaps())


@pytest.fixture
def model_repo(tmp_path):
    """Real AST scanners + tiny source/YAML inputs; no SDK, weights, or Git."""
    for family in MODEL_ROOTS:
        destination = tmp_path / "xhmodel_merak" / family
        (destination / "models").mkdir(parents=True)
        shutil.copyfile(
            REPO_ROOT / "xhmodel_merak" / family / "scan_model_types.py", destination / "scan_model_types.py"
        )
    return tmp_path


def _model(root, family="xh_llm", package="new_model", name="NewModel", *, master=True, force=False):
    path = root / "xhmodel_merak" / family / "models" / package / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'@{MODEL_ROOTS[family]}("{name}", master={master}, force={force})\nclass {name}:\n    pass\n',
        encoding="utf-8",
    )
    return path


def _recipe(root, *, name="NewModel", family="xh_llm", filename="full.yaml"):
    path = root / "configs_merak/workflows/new_model" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    field = "model_type" if family == "xh_llm" else "type"
    path.write_text(f"export:\n  model:\n    {field}: {name}\n", encoding="utf-8")
    return replace(
        get_recipe("ling-3.0-flash"),
        model_id=f"example-{name}",
        variants=(VariantRecipe("full", path.relative_to(root).as_posix()),),
    )


@pytest.mark.parametrize("family", MODEL_ROOTS)
def test_new_model_without_recipe_is_rejected(model_repo, family):
    _model(model_repo, family)
    with pytest.raises(ValueError, match=f"{family}/new_model:NewModel"):
        check_recipe_coverage(model_repo, (), set())


def test_new_directory_without_any_registration_is_rejected(model_repo):
    path = _model(model_repo)
    path.write_text("class NewModel: pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="xh_llm/new_model/"):
        check_recipe_coverage(model_repo, (), set())


def test_export_test_and_impact_declaration_do_not_replace_a_recipe(model_repo):
    from select_tests import POLICY_RELATIVE, SUITE_RELATIVE, select_for_changes

    source = _model(model_repo).relative_to(model_repo).as_posix()
    suite = model_repo / SUITE_RELATIVE
    suite.mkdir(parents=True)
    policy_path = model_repo / POLICY_RELATIVE
    policy = json.loads((REPO_ROOT / POLICY_RELATIVE).read_text())
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    for gate in policy["always_tests"]:
        (model_repo / gate).touch()
    test = suite / "test_new_model.py"
    test.write_text("def test_export(): pass\n", encoding="utf-8")
    test.with_suffix(".impact.json").write_text(
        json.dumps({"version": 1, "rules": [{"id": "new-model", "patterns": [source]}]}), encoding="utf-8"
    )
    selection = select_for_changes([source], repo_root=model_repo)
    assert set(selection.tests) == {*policy["always_tests"], test.relative_to(model_repo).as_posix()}
    assert not selection.uncovered_files
    with pytest.raises(ValueError, match="NewModel"):
        check_recipe_coverage(model_repo, (), set())


@pytest.mark.parametrize("family", MODEL_ROOTS)
def test_existing_recipe_does_not_cover_a_second_master_in_the_same_package(model_repo, family):
    _model(model_repo, family)
    recipe = _recipe(model_repo, family=family)
    _model(model_repo, family, name="AnotherModel")
    with pytest.raises(ValueError, match="new_model:AnotherModel"):
        check_recipe_coverage(model_repo, (recipe,), set())


@pytest.mark.parametrize("family", MODEL_ROOTS)
def test_complete_onboarding_needs_no_legacy_update_or_git(model_repo, family):
    _model(model_repo, family)
    _model(model_repo, family, name="VisualBranch", master=False)
    check_recipe_coverage(model_repo, (_recipe(model_repo, family=family),), set())
    assert not (model_repo / ".git").exists()


def test_auxiliary_model_cannot_be_used_as_a_public_recipe_target(model_repo):
    _model(model_repo, master=False)
    with pytest.raises(ValueError, match="not a registered master model"):
        check_recipe_coverage(model_repo, (_recipe(model_repo),), set())


def test_legacy_exceptions_are_exact_and_must_be_removed_when_resolved(model_repo):
    _model(model_repo)
    legacy = {"xh_llm/new_model/", "xh_llm/new_model:NewModel"}
    check_recipe_coverage(model_repo, (), legacy)
    _model(model_repo, name="NewGeneration")
    with pytest.raises(ValueError, match="NewGeneration"):
        check_recipe_coverage(model_repo, (), legacy)
    recipes = (_recipe(model_repo), _recipe(model_repo, name="NewGeneration", filename="next.yaml"))
    with pytest.raises(ValueError, match="Remove resolved/deleted"):
        check_recipe_coverage(model_repo, recipes, legacy)
    check_recipe_coverage(model_repo, recipes, set())


def test_deleted_model_must_not_leave_a_legacy_exception(model_repo):
    with pytest.raises(
        ValueError,
        match="Remove resolved/deleted.*",
    ):
        check_recipe_coverage(model_repo, (), {"xh_llm/retired/"})


@pytest.mark.parametrize(
    "declaration",
    [
        "@register_llm_model(MODEL_TYPE)\nclass Model: pass\n",
        '@register_llm_model("NewModel", master=FLAG)\nclass Model: pass\n',
        '@register_llm_model("NewModel", force=FLAG)\nclass Model: pass\n',
        '@register_llm_model("NewModel", **options)\nclass Model: pass\n',
        'register_llm_model("NewModel", module=Model)\n',
        "from some.builder import register_llm_model as register\n",
    ],
)
def test_registration_cannot_silently_evade_static_discovery(model_repo, declaration):
    _model(model_repo).write_text(declaration, encoding="utf-8")
    with pytest.raises(ValueError, match="literal"):
        recipe_gaps(model_repo, ())


def test_forced_registration_keeps_production_scanner_precedence(model_repo):
    _model(model_repo, package="a_new", force=True)
    _model(model_repo, package="z_legacy")
    _, types = scan_implementations(model_repo)
    assert types["xh_llm", "NewModel"] == "xh_llm/a_new"
    assert recipe_gaps(model_repo, (_recipe(model_repo),)) == {"xh_llm/z_legacy/"}


@pytest.mark.parametrize(
    "config,expected",
    [
        ("export: {}", "exactly one"),
        ("export: {model: {model_type: NewModel, type: NewModel}}", "exactly one"),
        ("export: {model: {model_type: MissingModel}}", "not a registered master"),
        ("export: {model: {model_type: null}}", "non-empty string"),
    ],
)
def test_recipe_workflow_must_resolve_to_a_real_model(model_repo, config, expected):
    _model(model_repo)
    recipe = _recipe(model_repo)
    (model_repo / recipe.default_variant.workflow_config).write_text(config, encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        check_recipe_coverage(model_repo, (recipe,), set())


@pytest.mark.parametrize("path", ["missing.yaml", "../outside.yaml", "/tmp/outside.yaml"])
def test_missing_or_outside_workflow_is_rejected(model_repo, path):
    _model(model_repo)
    recipe = replace(_recipe(model_repo), variants=(VariantRecipe("full", path),))
    with pytest.raises(ValueError, match="configs_merak/workflows"):
        check_recipe_coverage(model_repo, (recipe,), set())


def test_all_variants_and_context_workflows_are_checked(model_repo):
    _model(model_repo)
    recipe = _recipe(model_repo)
    missing = "configs_merak/workflows/new_model/missing.yaml"
    for variants in (
        (recipe.default_variant, VariantRecipe("mtp", missing)),
        (replace(recipe.default_variant, context_workflows=((8192, missing),)),),
    ):
        with pytest.raises(ValueError, match="does not exist"):
            check_recipe_coverage(model_repo, (replace(recipe, variants=variants),), set())


def test_plugin_without_package_init_is_not_mistaken_for_a_registered_recipe(model_repo, monkeypatch):
    import xhmodel_merak.model_integrations as integrations

    _model(model_repo)
    plugin = model_repo / "integrations/not_discovered/plugin.py"
    plugin.parent.mkdir(parents=True)
    plugin.write_text("def register(catalog):\n    raise AssertionError('not discoverable')\n", encoding="utf-8")
    monkeypatch.setattr(integrations, "__path__", [str(plugin.parent.parent)])
    catalog = build_builtin_catalog()
    assert catalog.list_recipes() == ()
    with pytest.raises(ValueError, match="NewModel"):
        check_recipe_coverage(model_repo, catalog.list_recipes(), set())


@pytest.mark.parametrize("gaps", [["xh_llm/*/"], ["xh_llm/a/", "xh_llm/a/"], [42], "xh_llm/a/"])
def test_legacy_inventory_rejects_wildcards_duplicates_and_bad_types(tmp_path, gaps):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"version": 1, "frozen_at": "0" * 40, "missing_recipes": gaps}))
    with pytest.raises(ValueError, match="unique exact"):
        load_legacy_gaps(path)


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"version": True, "frozen_at": "0" * 40, "missing_recipes": []},
        {"version": 2, "frozen_at": "0" * 40, "missing_recipes": []},
        {"version": 1, "frozen_at": "HEAD", "missing_recipes": []},
        {"version": 1, "frozen_at": "0" * 40, "missing_recipes": [], "allow_new": True},
    ],
)
def test_legacy_inventory_rejects_invalid_schema(tmp_path, data):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid frozen"):
        load_legacy_gaps(path)


def test_legacy_inventory_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text('{"version": 1, "version": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_legacy_gaps(path)


def test_recipe_example_matches_the_real_contract_without_polluting_catalog():
    from examples_merak.unified_api.recipe import example_catalog

    before = build_builtin_catalog().list_recipes()
    example = example_catalog()
    assert example.list_recipes() == (get_recipe("ling-3.0-flash"),)
    assert build_builtin_catalog().list_recipes() == before


def test_discovery_coverage_and_example_do_not_import_torch_sdk_or_models():
    script = """
import importlib.abc
import os
import sys
from pathlib import Path
class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "xhquant", "transformers"} or fullname.startswith(
            ("xhmodel_merak.xh_llm", "xhmodel_merak.xh_other_model")
        ):
            raise AssertionError("registration imported " + fullname)
sys.meta_path.insert(0, BlockHeavyImports())
before = dict(os.environ)
from _recipe_coverage import check_recipe_coverage, load_legacy_gaps
from xhmodel_merak.model_catalog.discovery import build_builtin_catalog
from examples_merak.unified_api.recipe import example_catalog, main
check_recipe_coverage(Path.cwd(), build_builtin_catalog().list_recipes(), load_legacy_gaps())
assert example_catalog().get_recipe("ling-3.0-flash").model_id == "ling-3.0-flash"
main()
assert dict(os.environ) == before
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(Path(__file__).parent), str(REPO_ROOT))))
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["model_id"] == "ling-3.0-flash"
