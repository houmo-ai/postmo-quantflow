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
FAMILIES = ("qwen3_5", "qwen3_5_moe", "qwen3_next")
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
        *(f"xhmodel_merak.xh_llm.models.{family}" for family in ("qwen3_5", "qwen3_next")),
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
        assert quant["data_type"] == "int_sym_xh"
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
