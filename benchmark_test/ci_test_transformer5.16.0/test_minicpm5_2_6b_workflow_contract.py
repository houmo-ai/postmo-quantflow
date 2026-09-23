"""CPU-only contract for MiniCPM5-2.6B dense workflow registration."""

from __future__ import annotations

from pathlib import Path

import yaml

from xhmodel_merak.model_catalog.registry import get_recipe
from xhmodel_merak.xh_llm import AutoLLMConfig
from xhmodel_merak.xh_llm.builder import get_model_class
from xhmodel_merak.xh_llm.models.minicpm5_dense.minicpm5_model import (
    XHMiniCPM5Model,
    XHMiniCPM5ModelConfig,
)
from xhmodel_merak.xh_llm.models.minicpm5_dense.workflow import XHMiniCPM5Workflow
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow
from xhquant.api import Config


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / (
    "configs_merak/workflows/xh2a/llm_models/minicpm5/2_6b/"
    "minicpm5_2_6b_xh2a_w8a8.yaml"
)
EXPECTED_WORKFLOW_CLS = (
    "xhmodel_merak.xh_llm.models.minicpm5_dense.workflow:XHMiniCPM5Workflow"
)


def test_minicpm5_2_6b_recipe_points_at_dense_workflow() -> None:
    recipe = get_recipe("minicpm5-2.6b")
    assert recipe.export_name == "minicpm5_2_6b"
    assert recipe.family == "minicpm5-dense"
    assert recipe.default_variant.workflow_config == str(
        CONFIG_PATH.relative_to(REPO_ROOT)
    )


def test_minicpm5_2_6b_yaml_export_contract() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model = raw["export"]["model"]
    assert model["chip_arch"] == "XH2a"
    assert model["model_type"] == "LlamaForCausalLM"
    assert model["model_name"] == "minicpm5_2_6b"
    assert model["context_max_length"] == 2048
    assert model["prefill_chunk_length"] == 256
    assert model["quant_scheme"]["quant_type"] == "w8a8_sefp"
    assert model["quant_scheme"]["nodes"]["lm_head"]["quant_type"] == "w8a8_sefp"
    assert raw["quant"]["method"] == "gptq"
    assert raw["quant"]["algorithm"] == "gptqmodel"

    cfg = Config.fromfile(str(CONFIG_PATH))
    assert cfg.quant.bits == 8
    assert cfg.quant.method == "gptq"
    assert cfg.export.model.model_type == "LlamaForCausalLM"


def test_minicpm5_2_6b_resolves_dense_adapter_and_workflow(tmp_path) -> None:
    hf_model_dir = tmp_path / "hf"
    hf_model_dir.mkdir()

    cfg = Config.fromfile(str(CONFIG_PATH)).export.model
    cfg.hf_model = str(hf_model_dir)

    model_cls = get_model_class(cfg)
    assert model_cls is XHMiniCPM5Model
    assert model_cls.__module__.startswith("xhmodel_merak.xh_llm.models.minicpm5_dense.")
    assert model_cls.WORKFLOW_CLS == EXPECTED_WORKFLOW_CLS
    assert "minicpm5.workflow" not in model_cls.WORKFLOW_CLS

    model_cfg = AutoLLMConfig.from_pretrained(cfg)
    assert type(model_cfg) is XHMiniCPM5ModelConfig

    workflow = AutoLLMWorkflow.from_config(str(hf_model_dir), str(CONFIG_PATH))
    assert type(workflow) is XHMiniCPM5Workflow
    assert workflow.expected_model_cls_name == "XHMiniCPM5Model"
    assert workflow.expected_model_config_cls_name == "XHMiniCPM5ModelConfig"
