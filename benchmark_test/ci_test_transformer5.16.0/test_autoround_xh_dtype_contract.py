"""Real workflow/adapter contracts for the positive-scale AutoRound default.

All families use real GPTQModel recipes in dry-run mode. These contracts
are not evidence of full-model accuracy or weight round-trips.
No full checkpoints are loaded and no GPU quantization is launched.
"""

from __future__ import annotations

import copy
import importlib
import sys
import types
from pathlib import Path

import pytest
import test_transformer516_recipe_contract as contracts
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models"
FAMILIES = ("qwen3_5", "qwen3_5_moe", "qwen3_next", "ling_3_flash")
PROFILE_PATHS = tuple(sorted(path for family in FAMILIES for path in (CONFIG_ROOT / family).rglob("*.yaml")))


@pytest.fixture
def modules():
    """Load actual adapters/workflows, avoiding unrelated exporter registries."""
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "xhmodel_merak" or name.startswith("xhmodel_merak.")
    }
    for name in saved:
        del sys.modules[name]
    packages = (
        "xhmodel_merak",
        "xhmodel_merak.xh_llm",
        "xhmodel_merak.xh_llm.workflows",
        "xhmodel_merak.xh_llm.models",
        *(f"xhmodel_merak.xh_llm.models.{family}" for family in ("qwen3_5", "qwen3_next", "ling_3_flash")),
    )
    try:
        for name in packages:
            package = types.ModuleType(name)
            package.__path__ = [str(REPO_ROOT.joinpath(*name.split(".")))]
            package.__package__ = name
            sys.modules[name] = package
        yield types.SimpleNamespace(
            qwen=importlib.import_module("xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter"),
            next=importlib.import_module("xhmodel_merak.xh_llm.models.qwen3_next.workflow"),
            ling=importlib.import_module("xhmodel_merak.xh_llm.models.ling_3_flash.quant_adapter"),
            ling_workflow=importlib.import_module("xhmodel_merak.xh_llm.models.ling_3_flash.workflow"),
        )
    finally:
        for name in list(sys.modules):
            if name == "xhmodel_merak" or name.startswith("xhmodel_merak."):
                del sys.modules[name]
        sys.modules.update(saved)


@pytest.mark.parametrize("path", PROFILE_PATHS, ids=lambda path: str(path.relative_to(CONFIG_ROOT)))
def test_family_yaml_dtype_is_autoround_only(path):
    quant = yaml.safe_load(path.read_text(encoding="utf-8")).get("quant")
    if quant is None:
        return
    if quant.get("method") == "autoround":
        expected_dtype = (
            "int_sym_xh_hessian"
            if path == CONFIG_ROOT / "qwen3_5/27b/qwen3_8_27b_full_xh_hessian.yaml"
            else "int_sym_xh"
        )
        assert quant["data_type"] == expected_dtype
        assert quant["sym"] is True
        assert quant["calibration"]["nsamples"] == 256
        assert quant["iters"] == 400
    else:
        assert "data_type" not in quant


@pytest.mark.parametrize("config_path", (contracts.QWEN_DENSE_AR_9B, contracts.QWEN_MOE_AR_35B))
@pytest.mark.parametrize("dtype", (None, "int_sym_xh", "int_sym_gptq", "int_sym"))
def test_qwen_dtype_reaches_real_recipe_without_changing_training(modules, tmp_path, config_path, dtype):
    config = yaml.safe_load((REPO_ROOT / config_path).read_text(encoding="utf-8"))
    quant = copy.deepcopy(config["quant"])
    quant.pop("data_type")
    if dtype is not None:
        quant["data_type"] = dtype
    builder = modules.qwen.build_qwen35_autoround_kwargs
    common = dict(
        hf_model_dir=str(tmp_path / "model"),
        output_dir=str(tmp_path / "out"),
        device="cuda:0",
        export_model_cfg=config["export"]["model"],
        workflow_seed=42,
    )
    kwargs = builder(quant_cfg=quant, **common)
    previous = builder(quant_cfg={**quant, "data_type": "int_sym_gptq"}, **common)
    assert kwargs["data_type"] == (dtype or "int_sym_xh")
    assert {key: value for key, value in kwargs.items() if key != "data_type"} == {
        key: value for key, value in previous.items() if key != "data_type"
    }
    assert (kwargs["nsamples"], kwargs["iters"], kwargs["batch_size"]) == (256, 400, 8)
    assert kwargs["deterministic"] is False
    assert kwargs["dataset"].endswith("data/calib_data/NeelNanda-pile-10k.jsonl")
    recipe = contracts._require_callable("qwen35_autoround", "quantize_qwen35_autoround")
    kwargs["dry_run"] = True
    contracts._assert_signature_accepts(recipe, kwargs)
    result = recipe(**kwargs)
    command = result.provenance["command"]
    assert command[command.index("--data_type") + 1] == (dtype or "int_sym_xh")
    contracts._assert_command_matches_parser(command)


@pytest.mark.parametrize(
    "method,dtype,fa_suffix",
    [
        *(
            ("autoround", dtype, suffix)
            for dtype in (None, "int_sym_xh", "int_sym_gptq", "int_sym")
            for suffix in ("", "_fa_all16")
        ),
        ("gptq", None, ""),
    ],
)
def test_next_workflow_reaches_real_recipe_and_parser(modules, tmp_path, method, dtype, fa_suffix):
    from gptqmodel.recipes.qwen3_next_autoround import build_parser

    suffix = "_gptq" if method == "gptq" else ""
    path = CONFIG_ROOT / f"qwen3_next/80b_a3b/qwen3_next_80b_a3b_full{fa_suffix}{suffix}.yaml"
    workflow = modules.next.Qwen3NextWorkflow(str(tmp_path / "model"), str(path))
    quant = workflow.workflow_config.quant
    quant.pop("data_type", None)
    if dtype is not None:
        quant["data_type"] = dtype
    quant["runtime"]["dry_run"] = True
    result = workflow.quant(str(tmp_path / "out"), "cuda:0")
    assert result.quanted_model_dir == str(tmp_path / "out")
    assert result.meta.method == method
    assert not (tmp_path / "out").exists()
    command = result.meta.provenance["command"]
    if method == "autoround":
        captured = vars(build_parser().parse_args(command[2:]))
        assert captured["data_type"] == (dtype or "int_sym_xh")
        assert (
            captured["nsamples"],
            captured["iters"],
            captured["batch_size"],
            captured["gradient_accumulate_steps"],
            captured["deterministic"],
        ) == (256, 400, 8, 1, False)
        assert captured["self_attn_bits"] == quant["moe"]["self_attn_bits"]
        assert captured["expert_bits"] == 4
        assert captured["linear_attn_bits"] == captured["shared_expert_bits"] == 8
        assert captured["dataset"] == "NeelNanda/pile-10k"
        assert Path(command[1]).name == "example_qwen3_next_autoround.py"
    else:
        assert "--data-type" not in command
        with pytest.raises(ValueError, match="keys not consumed"):
            modules.next.Qwen3NextWorkflow._validate_qwen3_next_quant_config({**quant, "data_type": "int_sym_xh"})


@pytest.mark.parametrize("method,dtype", (("autoround", None), ("autoround", "int_sym_gptq"), ("gptq", None)))
def test_ling_workflow_dry_run_reports_effective_dtype(modules, tmp_path, method, dtype):
    path = CONFIG_ROOT / f"ling_3_flash/ling_3_flash_{method}.yaml"
    workflow = modules.ling_workflow.Ling3FlashWorkflow(str(tmp_path / "model"), str(path))
    quant = workflow.workflow_config.quant
    quant.pop("data_type", None)
    quant["runtime"]["dry_run"] = True
    if dtype is not None:
        quant["data_type"] = dtype
    result = workflow.quant(str(tmp_path / "out"), "cuda:0")
    assert not (tmp_path / "out").exists()
    assert result.meta["precision"]["routed_experts"] == "W4"
    assert result.meta["precision"]["all_other_quantized_linears"] == "W8"
    if method == "autoround":
        assert result.meta["data_type"] == (dtype or "int_sym_xh")
        assert result.meta["calibration"]["nsamples"] == 256
        assert result.meta["provenance"]["iters"] == 400
        assert result.meta["provenance"]["backend"] == "auto_round"
    else:
        assert "data_type" not in result.meta
        assert result.meta["calibration"]["nsamples"] == 128
    assert result.meta["provenance"]["mla_weight_layout"] == "xh2_absorbed_v1"


@pytest.mark.parametrize("dtype", ("int_sym_xh", "int_sym_gptq"))
def test_ling_cli_passes_dtype_to_spec(modules, monkeypatch, dtype):
    cli = contracts._load_source_module("_ci_ling_quantize", REPO_ROOT / "examples_merak/llm/ling_3_flash/quantize.py")
    argv = ["quantize.py", "--method", "autoround", "--output", "/unused"]
    if dtype != "int_sym_xh":
        argv.extend(["--data-type", dtype])
    monkeypatch.setattr(sys, "argv", argv)
    spec = cli._build_spec(cli.parse_args())
    spec.validate()
    assert spec.data_type == dtype
    assert (spec.nsamples, spec.iters, spec.batch_size) == (256, 400, 1)


@pytest.mark.parametrize("dry_run", (False, True))
def test_ling_plan_preserves_requested_run_mode(modules, tmp_path, dry_run):
    spec = modules.ling.Ling3FlashQuantSpec(
        model_dir=str(tmp_path / "model"),
        output_dir=str(tmp_path / "out"),
        method="autoround",
        dry_run=dry_run,
    )
    plan = modules.ling.quantization_plan(spec)
    assert plan["provenance"]["dry_run"] is dry_run
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("section", (None, "calibration", "runtime", "moe"))
def test_ling_workflow_rejects_unused_config_keys(modules, tmp_path, section):
    path = CONFIG_ROOT / "ling_3_flash/ling_3_flash_autoround.yaml"
    workflow = modules.ling_workflow.Ling3FlashWorkflow(str(tmp_path / "model"), str(path))
    quant = workflow.workflow_config.quant
    quant["runtime"]["dry_run"] = True
    target = quant if section is None else quant.setdefault(section, {})
    target["unused_setting"] = 1
    with pytest.raises(ValueError, match="keys not consumed"):
        workflow.quant(str(tmp_path / "out"), "cuda:0")
    assert not (tmp_path / "out").exists()


def test_ling_workflow_reports_the_backend_output_path(modules, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    path = CONFIG_ROOT / "ling_3_flash/ling_3_flash_autoround.yaml"
    workflow = modules.ling_workflow.Ling3FlashWorkflow(str(tmp_path / "model"), str(path))
    result = workflow.quant(str(alias / "quantized"), "cuda:0", config_overrides={"quant.runtime.dry_run": True})
    assert result.quanted_model_dir == result.meta["output_dir"] == str(target / "quantized")
    assert not (target / "quantized").exists()
