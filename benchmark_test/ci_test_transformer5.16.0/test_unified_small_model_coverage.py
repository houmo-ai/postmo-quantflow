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


def test_all_variants_share_the_ci_transformers_environment():
    from xhmodel_merak.model_catalog.dependencies import dependency_profile, merge_requirements

    requirements = []
    for recipe in list_recipes():
        for variant in recipe.variants:
            profile = dependency_profile(recipe, variant)
            for phase in ("build", "inference"):
                values = profile[phase]["pip"]
                assert [r for r in values if r.startswith("transformers==")] == ["transformers==5.16.1"]
                requirements.extend(values)
    merge_requirements(tuple(requirements))


def _selected_cases(paths):
    from pathlib import Path
    from select_tests import select_for_changes, SUITE_RELATIVE

    owner = f"{SUITE_RELATIVE}/test_unified_small_models.py"
    result = select_for_changes(paths, repo_root=Path(__file__).resolve().parents[2])
    if owner in result.tests:
        return {c.id for c in CASES}
    return {t.split("[", 1)[1][:-1] for t in result.tests if t.startswith(owner + "::")}


def test_each_registered_workflow_selects_only_its_model_variants():
    for recipe in list_recipes():
        expected = {c.id for c in CASES if c.model_id == recipe.model_id}
        for variant in recipe.variants:
            for config in [variant.workflow_config, *(p for _, p in variant.context_workflows)]:
                assert _selected_cases([config]) == expected, config


def test_private_model_and_adapter_changes_select_related_cases():
    for directory, prefix in (("gemma4_series", "gemma4-"), ("ling_3_flash", "ling-"),
                              ("macaron_v1_tall", "macaron-"), ("wemm_embedding", "wemm-")):
        assert _selected_cases([f"xhmodel_merak/xh_llm/models/{directory}/model.py"]) == {
            c.id for c in CASES if c.model_id.startswith(prefix)
        }
    assert _selected_cases(["xhmodel_merak/model_integrations/pi05/inference.py"]) == {"pi0.5-libero/lerobot-h50"}
    assert _selected_cases(["xhmodel_merak/xh_llm/models/qwen3_5_moe/_moe_model.py"]) == {
        c.id for c in CASES if c.fixture in {"qwen_moe", "macaron"}
    }
    assert _selected_cases(["xhmodel_merak/xh_llm/models/qwen3_5/qwen3_5_llm_model.py"]) == {
        c.id for c in CASES if c.model_id.startswith(("qwen", "ornith", "macaron", "wemm"))
    }


def test_public_unified_changes_select_all_cases_and_mixed_private_changes_union():
    for path in ("model_build/resolver.py", "model_inference/api.py", "model_catalog/dependencies.py",
                 "model_integrations/_dependencies.py", "model_artifact/validation.py"):
        assert _selected_cases(["xhmodel_merak/" + path]) == {c.id for c in CASES}
    assert _selected_cases(["xhmodel_merak/model_integrations/pi05/plugin.py",
                            "xhmodel_merak/model_integrations/kokoro/inference.py"]) == {
        "pi0.5-libero/lerobot-h50", "kokoro-82m-v1.1-zh/full",
    }


def test_impact_parameter_nodes_match_real_pytest_collection():
    import json
    import subprocess
    import sys
    from pathlib import Path

    suite = Path(__file__).resolve().parent
    declaration = json.loads((suite / "test_unified_small_models.impact.json").read_text())
    nodes = {node for rule in declaration["rules"] for node in rule.get("nodes", [])}
    expected = {f"test_public_build_relocate_and_inference[{c.id}]" for c in CASES}
    assert nodes == expected
    # Exercise pytest's real node-id parser without running GPU fixtures.
    targets = [str(suite / "test_unified_small_models.py") + "::" + node for node in sorted(nodes)]
    result = subprocess.run([sys.executable, "-m", "pytest", "-c", str(suite / "pytest.ini"),
                             "--confcutdir=" + str(suite), "--collect-only", "-q", *targets],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{len(CASES)} tests collected" in result.stdout or f"collected {len(CASES)} items" in result.stdout
