"""Dedicated entry plumbing; checkpoint semantics have separate native tests."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_build_cli_help_needs_no_model_or_patch_installation():
    result = subprocess.run(
        [sys.executable, "-m", "examples_merak.llm.qwen4_exp.build", "--help"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert "--quant-checkpoint" in result.stdout
    assert "--speculative" in result.stdout
    assert "--deployment-precision" in result.stdout


def test_entry_rejects_other_models_before_loading_the_builder():
    from xhmodel_merak.xh_llm.models.qwen4_exp.build_entry import build_qwen4_exp_model

    with pytest.raises(ValueError, match="requires model_id"):
        build_qwen4_exp_model(model_id="qwen3.5-9b", model_dir="does-not-exist")


@pytest.mark.parametrize("raise_build", [False, True])
def test_entry_scopes_resolver_executor_and_restores_functions(monkeypatch, raise_build):
    from dataclasses import dataclass

    from xhmodel_merak.model_build import checkpoint, executor, resolver
    from xhmodel_merak.xh_llm.models.qwen4_exp.build_entry import build_qwen4_exp_model

    original = tuple(getattr(checkpoint, name) for name in (
        "_read_quant_metadata", "_verify_named_checkpoint_tensors", "_checkpoint_method",
    ))
    seen = []
    result = object()

    @dataclass
    class Plan:
        quant_method: str
        config_replacements: dict

    def resolve(**kwargs):
        assert checkpoint._read_quant_metadata is not original[0]
        seen.append(kwargs)
        return Plan("autoround", {"unrelated": True})

    def execute(plan):
        assert checkpoint._verify_named_checkpoint_tensors is not original[1]
        assert plan.config_replacements == {
            "unrelated": True, "quant.method": "autoround", "quant.iters": 200,
            "quant.disable_opt_rtn": None,
        }
        if raise_build:
            raise RuntimeError("export failure")
        return result

    monkeypatch.setattr(resolver, "resolve_build_plan", resolve)
    monkeypatch.setattr(executor, "execute_build", execute)
    kwargs = dict(model_dir="native", quant_checkpoint="packed", speculative={"mode": "mtp", "num_draft_tokens": 1})
    if raise_build:
        with pytest.raises(RuntimeError, match="export failure"):
            build_qwen4_exp_model(**kwargs)
    else:
        assert build_qwen4_exp_model(**kwargs) is result
    assert seen == [{"model_id": "qwen3.8-flash-next", **kwargs}]
    assert tuple(getattr(checkpoint, name) for name in (
        "_read_quant_metadata", "_verify_named_checkpoint_tensors", "_checkpoint_method",
    )) == original


@pytest.mark.parametrize("method", [None, "rtn", "autoround", "none"])
def test_cli_preserves_inferred_or_explicit_quant_method(monkeypatch, capsys, method):
    from examples_merak.llm.qwen4_exp.build import main
    from xhmodel_merak.xh_llm.models.qwen4_exp import build_entry

    seen = []

    def build(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {"model_id": "qwen3.8-flash-next"})

    monkeypatch.setattr(build_entry, "build_qwen4_exp_model", build)
    argv = [
        "--model-dir", "native", "--quant-checkpoint", "packed",
        "--speculative", '{"mode":"mtp","num_draft_tokens":1}',
        "--precision", '{"weight":"w4","activation":"a8"}',
        "--context-length", "262144", "--prefill-chunk-length", "256",
    ]
    if method is not None:
        argv += ["--quant-method", method]
    main(argv)
    assert json.loads(capsys.readouterr().out) == {"model_id": "qwen3.8-flash-next"}
    assert seen[0]["speculative"] == {"mode": "mtp", "num_draft_tokens": 1}
    assert seen[0]["precision"] == {"weight": "w4", "activation": "a8"}
    assert seen[0]["context_length"] == 262144
    if method is None:
        assert "quant_method" not in seen[0]
    else:
        assert seen[0]["quant_method"] == (None if method == "none" else method)


@pytest.mark.parametrize("requested,method,iterations,disable", [
    (None, "rtn", 0, True), ("rtn", "rtn", 0, True), ("autoround", "autoround", 200, None),
])
def test_real_plan_binds_selected_algorithm_to_effective_workflow(
    tmp_path, monkeypatch, requested, method, iterations, disable,
):
    import yaml

    # Use the same real source layout as the publication cases, without
    # replacing the resolver, config materialization or dependency publisher.
    from test_qwen4_exp_unified_api import _config

    from xhmodel_merak.model_build import executor
    from xhmodel_merak.model_build.workflow import materialize_workflow_config
    from xhmodel_merak.xh_llm.models.qwen4_exp.build_entry import build_qwen4_exp_model

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(_config()))
    seen = []
    original_writer = executor.write_dependency_files

    def execute(plan):
        path = tmp_path / "effective.yaml"
        materialize_workflow_config(plan, path)
        quant = yaml.safe_load(path.read_text())["quant"]
        assert plan.quant_method == quant["method"] == method
        assert quant["iters"] == iterations
        assert quant["disable_opt_rtn"] is disable
        folder = tmp_path / "environment"
        folder.mkdir()
        record = executor.write_dependency_files(folder, plan.model_id, plan.variant)
        assert set(record["quantization"]) == {"rtn", "autoround"}
        assert (folder / "environments/quant-rtn.txt").is_file()
        seen.append(plan)
        return plan

    monkeypatch.setattr(executor, "execute_build", execute)
    options = {} if requested is None else {"quant_method": requested}
    build_qwen4_exp_model(model_dir=str(source), output_dir=str(tmp_path / "artifact"), **options)
    assert len(seen) == 1
    assert executor.write_dependency_files is original_writer


def test_dependency_compatibility_rejects_upstream_drift_without_installing(monkeypatch):
    from xhmodel_merak.model_build import executor
    from xhmodel_merak.xh_llm.models.qwen4_exp.build_entry import _rtn_dependency_scope

    def changed_writer(root, model_id, variant):
        return None

    monkeypatch.setattr(executor, "write_dependency_files", changed_writer)
    with pytest.raises(RuntimeError, match="incompatible"):
        with _rtn_dependency_scope():
            pytest.fail("An incompatible publisher must not be installed")
    assert executor.write_dependency_files is changed_writer
