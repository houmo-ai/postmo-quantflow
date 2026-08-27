"""Qwen3.5-MoE ACEP dynamic-pruning configuration contracts."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_CONFIG = (
    ROOT
    / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/"
    "qwen3_6_35b_a3b_dynamic_prune.yaml"
)


def test_dynamic_prune_workflow_config_uses_acep_top_p_85():
    model_config = yaml.safe_load(WORKFLOW_CONFIG.read_text(encoding="utf-8"))["export"]["model"]

    assert model_config["dynamic_prune_threshold"] == pytest.approx(0.85)
    assert model_config["dynamic_prune"]["threshold"] == pytest.approx(0.85)


def test_dynamic_prune_cli_parses_acep_coverage():
    from examples_merak.llm.qwen3_5.qwen3_5_dynamic_prune_workflow import _acep_coverage

    assert _acep_coverage("0.85") == pytest.approx(0.85)
    with pytest.raises(argparse.ArgumentTypeError, match=r"must be in \(0, 1\]"):
        _acep_coverage("0")


def test_dynamic_prune_config_defaults_to_valid_acep_coverage():
    from xhmodel_merak.xh_llm.models.qwen3_5_moe_dynamic_prune.xh_qwen3_5_moe_dynamic_prune_config import (
        XHQwen3_5MoeDynamicPruneModelConfig,
    )

    config = XHQwen3_5MoeDynamicPruneModelConfig(model_name="dynamic-prune-test")

    assert config.dynamic_prune_threshold == pytest.approx(0.85)


@pytest.mark.parametrize("coverage", [0.0, -0.1, 1.01])
def test_dynamic_prune_config_rejects_invalid_acep_coverage(coverage):
    from xhmodel_merak.xh_llm.models.qwen3_5_moe_dynamic_prune.xh_qwen3_5_moe_dynamic_prune_config import (
        XHQwen3_5MoeDynamicPruneModelConfig,
    )

    with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
        XHQwen3_5MoeDynamicPruneModelConfig(
            model_name="dynamic-prune-test",
            dynamic_prune_threshold=coverage,
        )


def test_dynamic_prune_block_applies_default_coverage_and_expert_scalars(monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen3_5_moe._moe_model import _Qwen3_5MoeSparseMoeBlock
    from xhmodel_merak.xh_llm.models.qwen3_5_moe_dynamic_prune._dynamic_prune_moe import (
        Qwen35MoeDynamicPruneBlock,
    )

    monkeypatch.setattr(_Qwen3_5MoeSparseMoeBlock, "_setup", lambda self, cfg=None: self)
    block = object.__new__(Qwen35MoeDynamicPruneBlock)
    torch.nn.Module.__init__(block)
    block.moeblock = SimpleNamespace(expert_gate_proj_weight=torch.empty(4, 2))

    assert block._setup() is block
    assert block.moeblock.prune_threshold == pytest.approx(0.85)
    torch.testing.assert_close(block.moeblock.s_scalar, torch.ones(4))


def test_dynamic_prune_workflow_maps_default_coverage_to_model_override(monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow
    from xhmodel_merak.xh_llm.models.qwen3_5_moe_dynamic_prune.qwen3_5_moe_dynamic_prune_model import (
        Qwen35DynamicPruneWorkflow,
    )

    class WorkflowConfig:
        export = {"model": {}}

        def with_overrides(self, overrides):
            assert overrides is None
            return self

    def capture_export(self, quant_result, output_dir, device, config_overrides=None):
        return config_overrides

    monkeypatch.setattr(Qwen35Workflow, "export", capture_export)
    workflow = object.__new__(Qwen35DynamicPruneWorkflow)
    workflow.workflow_config = WorkflowConfig()
    workflow.model_dir = "unused"

    overrides = workflow.export(object(), "unused", "cpu")

    assert overrides["export.model.dynamic_prune_threshold"] == pytest.approx(0.85)
    assert overrides["export.model.dynamic_prune_s_scalar_path"] is None
