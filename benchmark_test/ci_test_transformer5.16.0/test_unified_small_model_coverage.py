# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Fail closed when public variants outgrow the executable integration matrix."""

import pytest
from _unified_cases import CASES

from xhmodel_merak.model_catalog import list_recipes


@pytest.mark.parametrize("variant", ["full", "mtp"])
def test_next_ci_gptq_plan_uses_local_data_and_short_validation(tmp_path, monkeypatch, variant):
    from _unified_e2e_worker import prepare

    import xhmodel_merak.model_build.api as api

    case = next(c for c in CASES if c.fixture == "qwen_next" and c.variant == variant)
    kwargs, inputs = prepare(case, tmp_path, monkeypatch)
    plan = api.resolve_build_plan(**kwargs)
    assert plan.quant_method == "gptq"
    replacements = plan.config_replacements
    assert replacements["quant.calibration.jsonl"] == str(tmp_path / "calibration.jsonl")
    assert replacements["quant.calibration.nsamples"] == 2
    assert replacements["quant.validation.skip_ppl"] is True
    assert replacements["quant.validation.max_new_tokens"] == 2
    assert inputs["max_new_tokens"] == 2


def test_every_registered_variant_has_a_unique_integration_case():
    actual = [(c.model_id, c.variant) for c in CASES]
    expected = {(r.model_id, v.name) for r in list_recipes() for v in r.variants}
    assert len(actual) == len(set(actual)), "Duplicate unified integration cases"
    assert set(actual) == expected


def test_variants_have_compatible_dependencies_within_their_transformers_environment():
    from xhmodel_merak.model_catalog.dependencies import dependency_profile, merge_requirements

    requirements = {}
    for recipe in list_recipes():
        for variant in recipe.variants:
            profile = dependency_profile(recipe, variant)
            for phase in ("build", "inference"):
                values = profile[phase]["pip"]
                version = next(r for r in values if r.startswith("transformers=="))
                requirements.setdefault(version, []).extend(values)
    for values in requirements.values():
        merge_requirements(tuple(values))


def _selected_cases(paths):
    from pathlib import Path

    from select_tests import SUITE_RELATIVE, select_for_changes

    result = select_for_changes(paths, repo_root=Path(__file__).resolve().parents[2])
    return {
        case.id
        for case in CASES
        if f"{SUITE_RELATIVE}/{case.owner}" in result.tests or f"{SUITE_RELATIVE}/{case.node_id}" in result.tests
    }


def test_each_registered_workflow_selects_only_its_model_variants():
    for recipe in list_recipes():
        expected = {c.id for c in CASES if c.model_id == recipe.model_id}
        for variant in recipe.variants:
            for config in [variant.workflow_config, *(p for _, p in variant.context_workflows)]:
                assert _selected_cases([config]) == expected, config


def test_private_model_and_adapter_changes_select_related_cases():
    for directory, prefix in (
        ("gemma4_series", "gemma4-"),
        ("ling_3_flash", "ling-"),
        ("macaron_v1_tall", "macaron-"),
        ("wemm_embedding", "wemm-"),
    ):
        assert _selected_cases([f"xhmodel_merak/xh_llm/models/{directory}/model.py"]) == {
            c.id for c in CASES if c.model_id.startswith(prefix)
        }
    assert _selected_cases(["xhmodel_merak/model_integrations/pi05/inference.py"]) == {"pi0.5-libero/lerobot-h50"}
    assert _selected_cases(["xhmodel_merak/xh_llm/models/qwen3_5_moe/_moe_model.py"]) == {
        c.id for c in CASES if c.fixture in {"qwen_moe", "macaron"}
    }
    assert _selected_cases(["xhmodel_merak/xh_llm/models/qwen3_5/qwen3_5_llm_model.py"]) == {
        c.id
        for c in CASES
        if (c.model_id.startswith(("qwen3", "ornith", "macaron", "wemm")) and c.fixture != "qwen3_vl_reranker")
        or c.fixture in {"minicpm_ocr", "ocr_rm"}
    }
    assert _selected_cases(["xhmodel_merak/xh_llm/models/qwen3_vl_reranker/model.py"]) == {"qwen3-vl-reranker-2b/full"}


def test_public_unified_changes_select_all_cases_and_mixed_private_changes_union():
    for path in (
        "model_build/resolver.py",
        "model_inference/api.py",
        "model_catalog/dependencies.py",
        "model_integrations/_dependencies.py",
        "model_artifact/validation.py",
    ):
        assert _selected_cases(["xhmodel_merak/" + path]) == {c.id for c in CASES}
    assert _selected_cases(
        ["xhmodel_merak/model_integrations/pi05/plugin.py", "xhmodel_merak/model_integrations/kokoro/inference.py"]
    ) == {
        "pi0.5-libero/lerobot-h50",
        "kokoro-82m-v1.1-zh/full",
    }


def test_impact_parameter_nodes_match_real_pytest_collection():
    import subprocess
    import sys
    from pathlib import Path

    suite = Path(__file__).resolve().parent
    # Collect entire owner files so unexpected or missing parameters also fail.
    targets = [str(suite / owner) for owner in sorted({case.owner for case in CASES})]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(suite / "pytest.ini"),
            "--confcutdir=" + str(suite),
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            *targets,
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{len(CASES)} tests collected" in result.stdout or f"collected {len(CASES)} items" in result.stdout

    actual = {
        line.strip() for line in result.stdout.splitlines() if "::test_public_build_relocate_and_inference[" in line
    }
    assert actual == {case.node_id for case in CASES}


def test_model_declarations_and_manifests_select_only_their_owner():
    from select_tests import SUITE_RELATIVE

    for owner in {case.owner for case in CASES}:
        expected = {case.id for case in CASES if case.owner == owner}
        for path in (owner, owner.replace(".py", ".impact.json")):
            assert _selected_cases([f"{SUITE_RELATIVE}/{path}"]) == expected


def test_helpers_select_only_actual_model_consumers():
    from select_tests import SUITE_RELATIVE

    def selected(helper):
        return _selected_cases([f"{SUITE_RELATIVE}/{helper}"])

    assert selected("_unified_reporting.py") == set()
    assert selected("_unified_assets.py") == {
        case.id for case in CASES if case.fixture in {"kokoro", "moss", "depthsplat", "minicpmo"}
    }
    assert selected("_unified_fixtures/deepseek_v4.py") == {case.id for case in CASES if case.fixture == "deepseek_v4"}
    assert selected("_unified_fixtures/minicpm_v47.py") == {case.id for case in CASES if case.fixture == "minicpm_v47"}
    assert selected("fixtures/unified/qwen_dense.json") == {case.id for case in CASES if case.fixture == "qwen_dense"}
    assert selected("fixtures/depthsplat/release.json") == {case.id for case in CASES if case.fixture == "depthsplat"}
    for helper in ("_unified_runner.py", "_unified_e2e_worker.py", "_unified_preparation.py"):
        assert selected(helper) == {case.id for case in CASES}


@pytest.mark.parametrize(
    "declaration",
    [
        "",
        "CASE_SPEC = ('model', 'fixture', 'module', ())",
        "CASE_SPEC = ('model', 'fixture', 'module', ('full', 'full'))",
    ],
)
def test_discovery_rejects_missing_invalid_or_duplicate_variants(tmp_path, declaration):
    from _unified_cases import discover_cases

    (tmp_path / "test_unified_model_a.py").write_text(declaration)
    with pytest.raises(ValueError):
        discover_cases(tmp_path)


def test_discovery_rejects_duplicate_model_owners(tmp_path):
    from _unified_cases import discover_cases

    for name in ("a", "b"):
        (tmp_path / f"test_unified_model_{name}.py").write_text("CASE_SPEC = ('model', 'fixture', 'module', ('full',))")
    with pytest.raises(ValueError, match="duplicate"):
        discover_cases(tmp_path)


def test_selected_cases_span_modules_and_reject_duplicate_execution():
    from types import SimpleNamespace

    from _unified_runner import selected_cases

    def item(case):
        return SimpleNamespace(callspec=SimpleNamespace(params={"case": case}))

    cases = [CASES[0], next(case for case in CASES if case.owner != CASES[0].owner)]
    assert selected_cases([item(cases[0]), SimpleNamespace(), item("unrelated"), item(cases[1])]) == cases
    with pytest.raises(AssertionError, match="Duplicate"):
        selected_cases([item(cases[0]), item(cases[0])])


def test_session_scheduler_runs_each_selected_case_once_without_gpu_overlap(tmp_path, monkeypatch):
    """Exercise real child scheduling with a tiny CPU worker and synthetic GPU IDs."""
    import json
    from types import SimpleNamespace

    import _unified_runner as runner

    script = tmp_path / "fixture" / "suite" / "_unified_e2e_worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("""import json, os, sys, time
from pathlib import Path
folder = Path(sys.argv[2])
device = os.environ["CUDA_VISIBLE_DEVICES"]
lock = folder.parents[1] / ("device-" + device)
# Exclusive creation fails if another worker owns the same device.
with lock.open("x"):
    time.sleep(0.05)
    (folder / "result.json").write_text(json.dumps({
        "passed": True, "stage": "complete", "case": sys.argv[1], "device": device}))
lock.unlink()
""")
    monkeypatch.setattr(runner, "__file__", str(script.with_name("_unified_runner.py")))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "synthetic-0,synthetic-1")
    selected = [CASES[0], CASES[1], next(case for case in CASES if case.owner != CASES[0].owner)]
    items = [SimpleNamespace(callspec=SimpleNamespace(params={"case": case})) for case in selected]
    request = SimpleNamespace(session=SimpleNamespace(items=items))
    root = tmp_path / "runs"
    root.mkdir()
    factory = SimpleNamespace(mktemp=lambda _: root)
    generator = runner.unified_runs.__wrapped__(request, factory)
    try:
        futures = next(generator)
        assert set(futures) == {case.id for case in selected}
        for case in selected:
            runner.check_case_result(case, futures)
        records = [json.loads(path.read_text()) for path in root.glob("*/*/result.json")]
        assert sorted(record["case"] for record in records) == sorted(case.id for case in selected)
        assert {record["device"] for record in records} == {"synthetic-0", "synthetic-1"}
    finally:
        generator.close()
    assert not list(root.glob("device-*"))


def test_every_case_has_an_importable_preparation_entry():
    from importlib import import_module

    for module in {case.preparation for case in CASES}:
        assert callable(import_module(module).prepare), module


def test_pi05_preparation_preserves_public_build_and_action_inputs(tmp_path, monkeypatch):
    from _unified_e2e_worker import prepare

    from xhmodel_merak.model_catalog.registry import get_catalog

    case = next(case for case in CASES if case.fixture == "pi05")
    kwargs, inputs = prepare(case, tmp_path, monkeypatch)
    assert kwargs["model_id"] == case.model_id
    assert kwargs["resources"] == {"tokenizer": str(tmp_path / "tokenizer")}
    assert (tmp_path / "source" / "config.json").is_file()
    assert inputs["observation.state"] == [0.0] * 8
    assert inputs["seed"] == 1 and inputs["task"]
    assert dict(get_catalog()._recipes[case.model_id].config_replacements)["export.other.hidden_size"] == 64
