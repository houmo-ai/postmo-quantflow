"""CPU contracts for the isolated ZTE example; private attachments not needed."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples_merak/customer-specific/zte/m50_train"


@pytest.fixture
def common():
    spec = importlib.util.spec_from_file_location("_zte_example_common_contract", EXAMPLE/"common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_help_without_site_packages_from_another_directory(tmp_path):
    for script in ("train.py", "diagnose.py", "plot.py"):
        result = subprocess.run([sys.executable, "-S", str(EXAMPLE/script), "--help"],
                                cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_config_precedence_and_repo_relative_paths(common, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = "examples_merak/customer-specific/zte/m50_train/configs/c_sefp8_fp16_update.json"
    args = common.parse_training_args(["--config", config, "--seed", "23", "--steps", "2",
                                       "--output", "work_dirs/zte_train/runs/contract_example"])
    assert (args.variant, args.bits, args.seed, args.steps) == ("C", 8, 23, 2)
    assert args.output == REPO/"work_dirs/zte_train/runs/contract_example"
    assert args.config == REPO/config
    assert args.customer_sources == EXAMPLE/"customer"
    assert args.relu_path == "nl" and args.diagnostic_override is None


@pytest.mark.parametrize("payload", [
    {"lr": .01}, {"steps": True}, {"steps": 0}, {"bits": 4},
    {"variant": "NPU"}, {"check_only": "false"}, {"customer_sources": ""},
])
def test_invalid_config_fails_before_cuda_import(common, tmp_path, payload):
    path = tmp_path/"invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as error:
        common.parse_training_args(["--config", str(path)])
    assert error.value.code == 2


def test_duplicate_config_keys_are_rejected(common, tmp_path):
    path = tmp_path/"duplicate.json"
    path.write_text('{"steps": 300, "steps": 2}')
    with pytest.raises(SystemExit):
        common.parse_training_args(["--config", str(path)])


def test_failed_source_check_does_not_create_results(tmp_path):
    output = tmp_path/"should_not_exist"
    result = subprocess.run([sys.executable, "-S", str(EXAMPLE/"train.py"), "--check-only",
                             "--customer-sources", str(tmp_path/"missing"), "--output", str(output)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Missing customer reference" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not output.exists()


def test_material_layout_hashes_and_corruption(common, tmp_path, monkeypatch):
    content = {"reference": b"reference fixture", "autograd": b"autograd fixture"}
    spec = {"files": {"reference": {"path": "torch_native/model.py", "archive_layout": "torch_native/torch_native/model.py",
                                      "sha256": hashlib.sha256(content["reference"]).hexdigest()},
                       "autograd": {"path": "torch_integer/engine/bf16_autograd.py", "archive_layout": "torch_integer/torch_integer/engine/bf16_autograd.py",
                                     "sha256": hashlib.sha256(content["autograd"]).hexdigest()}}}
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps(spec))
    monkeypatch.setattr(common, "MANIFEST", manifest)
    root = tmp_path/"customer"
    for name, rule in spec["files"].items():
        p = root/rule["archive_layout"]
        p.parent.mkdir(parents=True)
        p.write_bytes(content[name])
    files = common.verify_customer_sources(root)
    assert set(files) == {"reference", "autograd"}
    files["reference"].write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        common.verify_customer_sources(root)


def test_output_protects_existing_evidence(common, tmp_path):
    output = tmp_path/"result"
    common.prepare_output(output)
    (output/"protocol.json").write_text("original")
    with pytest.raises(FileExistsError):
        common.prepare_output(output)
    assert (output/"protocol.json").read_text() == "original"


def test_sdk_without_git_metadata_still_has_usable_provenance(common, tmp_path):
    result = common.git_revision(tmp_path)
    assert result["revision"] is None
    assert result["error"]


def test_plot_labels_cannot_hide_diagnostic_overrides(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    spec = importlib.util.spec_from_file_location("_zte_plot_contract", EXAMPLE/"plot.py")
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    run = tmp_path/"diagnostic"
    run.mkdir()
    (run/"summary.json").write_text(json.dumps({"protocol":{"variant":"B","matmul_bits":16,"seed":17,
                                                                  "diagnostic_override":"backward_matmul_fp32","relu_path":"vp"}}))
    (run/"initial_metrics.json").write_text("{}")
    (run/"curve.csv").write_text("step,train_class_loss\n1,2.0\n")
    label = plot.load_run(run)["label"]
    assert "DIAGNOSTIC" in label
    assert "backward_matmul_fp32" in label and "relu=vp" in label


def test_data_is_learnable_reproducible_and_rng_isolated(common):
    import torch
    torch.manual_seed(11)
    before = torch.get_rng_state().clone()
    x, y = common.dataset(18, 20261022)
    assert torch.equal(before, torch.get_rng_state())
    repeat_x, repeat_y = common.dataset(18, 20261022)
    different_x, _ = common.dataset(18, 20261023)
    assert torch.equal(x, repeat_x) and torch.equal(y, repeat_y)
    assert not torch.equal(x, different_x)
    assert x.shape == (18, 9, 13)
    assert torch.equal(x[:, :, :9].argmax(-1), y[:, None].expand(-1, 9))
    assert y.bincount(minlength=9).tolist() == [2]*9


def test_all_published_configs_keep_the_main_experiment_contract(common):
    configs = sorted((EXAMPLE/"configs").glob("*.json"))
    assert len(configs) == 5
    observed = set()
    for path in configs:
        args = common.parse_training_args(["--config", str(path)])
        observed.add((args.variant, args.bits))
        assert (args.seed, args.steps, args.relu_path, args.diagnostic_override) == (17, 300, "nl", None)
    assert observed == {("A",16),("B",16),("C",16),("B",8),("C",8)}


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    spec = importlib.util.spec_from_file_location("_zte_runtime_contract", EXAMPLE/"runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_float_runtime_linear_forward_and_backward(runtime):
    """Exercise actual adapter arithmetic; no SDK, customer files or mocks."""
    import torch
    rt = runtime.FloatRuntime()
    a = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
    w = torch.tensor([[1.,-1.],[2.,0.]], requires_grad=True)
    reference = a @ w
    (reference.square().sum()/2).backward()
    out = rt.matmul(rt.wrap(a.detach()), rt.wrap(w.detach()))
    assert torch.equal(out.tensor, torch.tensor([[5.,-1.],[11.,-3.]]))
    da = rt.matmul(out, rt.wrap(w.detach()), tb=True)
    dw = rt.matmul(rt.wrap(a.detach()), out, ta=True)
    assert torch.equal(da.tensor, a.grad)
    assert torch.equal(dw.tensor, w.grad)
    db = rt.reduce(out, 0)
    assert torch.equal(db.tensor, torch.tensor([[16.,-4.]]))
    broadcast = rt.binary(out, rt.wrap(torch.tensor([[1.,2.]])), 0)
    assert torch.equal(broadcast.tensor, torch.tensor([[6.,1.],[12.,-1.]]))
    assert rt.counts["forward:matmul"] == 3
    assert runtime._xh2_backend.cache_info().currsize == 0


def test_runtime_masks_views_trace_and_state_assignment(runtime):
    import torch
    rt = runtime.FloatRuntime()
    x = rt.wrap(torch.tensor([[-1.,0.,1.],[2.,3.,4.]]))
    assert torch.equal(rt.unary(x,8).tensor,torch.tensor([[0.,0.,1.],[1.,1.,1.]]))
    assert torch.equal(rt.reduce(x,1,maximum=True).tensor,torch.tensor([[1.],[4.]]))
    assert torch.equal(rt.slice(x,1,2).tensor,torch.tensor([[0.,1.],[3.,4.]]))
    sequence = rt.wrap(torch.arange(18*13).reshape(18,13))
    assert torch.equal(rt.time(sequence,8).tensor,sequence.tensor[[8,17]])
    target = rt.wrap(torch.zeros_like(x.tensor))
    target.assign(x)
    x.tensor.fill_(7)
    assert torch.equal(target.tensor,torch.tensor([[-1.,0.,1.],[2.,3.,4.]]))
    rt.trace = []
    rt.phase = "backward"
    rt.matmul(target,rt.wrap(torch.ones(3,1)))
    assert rt.trace[0][0] == "backward:matmul"
    assert torch.equal(rt.trace[0][1],torch.tensor([[0.],[9.]]))


def test_real_trainer_metrics_and_checkpoint_comparison_without_customer(monkeypatch):
    import torch
    monkeypatch.syspath_prepend(str(EXAMPLE))
    spec = importlib.util.spec_from_file_location("_zte_trainer_contract", EXAMPLE/"trainer.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    metric = trainer.measure(torch.tensor([3.,4.]),torch.tensor([0.,0.]))
    assert metric["max_abs"] == 4
    assert torch.isfinite(torch.tensor(metric["relative_l2"]))
    assert metric["relative_l2"] > 1e12
    finite = trainer.stats([torch.tensor([0.,-1.,65504.])])
    assert finite["finite"]
    assert finite["zero_fraction"] == pytest.approx(1/3)
    assert finite["saturation_fraction"] == pytest.approx(1/3)
    assert not trainer.stats([torch.tensor([float("nan"),float("inf")])])["finite"]
    state = {"step":3,"optimizer":[torch.tensor([1.,2.]),{"rng":torch.tensor([3,4])}]}
    clone = {"step":3,"optimizer":[torch.tensor([1.,2.]),{"rng":torch.tensor([3,4])}]}
    assert trainer.equal_nested(state,clone)
    clone["optimizer"][0][1] = 5
    assert not trainer.equal_nested(state,clone)


def test_entire_amended_example_has_focused_ci_coverage():
    """Protect the exact full-patchset path set that Jenkins consumes."""
    from select_tests import SUITE_RELATIVE, select_for_changes
    owner = f"{SUITE_RELATIVE}/test_zte_m50_train_example.py"
    changed = [str(path.relative_to(REPO)) for path in EXAMPLE.rglob("*")
               if path.is_file() and "customer" not in path.relative_to(EXAMPLE).parts
               and "__pycache__" not in path.parts]
    changed += [owner,f"{SUITE_RELATIVE}/test_zte_m50_train_example.impact.json"]
    selection = select_for_changes(changed,repo_root=REPO)
    assert not selection.uncovered_files
    assert not selection.full_suite
    assert set(selection.tests) == {
        owner,f"{SUITE_RELATIVE}/test_ci_impact_selector.py",
        f"{SUITE_RELATIVE}/test_recipe_coverage.py",
    }
