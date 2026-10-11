"""CPU-only contract for MiniCPM-V-4.7 recipes, workflows, and example helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from xhmodel_merak.model_catalog.registry import get_recipe
from xhmodel_merak.model_integrations._shared import (
    MINICPM_V_4_7_BUILD,
    MINICPM_V_4_7_INFERENCE,
)
from xhmodel_merak.workflows import AutoWorkflow


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples_merak/llm/minicpm_v_4_7"
DENSE_CONFIG = REPO_ROOT / (
    "configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/1b/"
    "minicpm_v_4_7_xh2a_w8a8.yaml"
)
MOE_CONFIG = REPO_ROOT / (
    "configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/35b_a3b/"
    "minicpm_v_4_7_moe_xh2a_w8a8.yaml"
)
COMPONENTS = ("vision_4x", "vision_16x", "llm")


def _load_example(module_name: str, filename: str):
    path = EXAMPLE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("model_id", "export_name", "config_path", "model_type"),
    (
        (
            "minicpm-v-4.7-1b",
            "minicpm_v_4_7_1b",
            DENSE_CONFIG,
            "MiniCPMV47ForConditionalGeneration",
        ),
        (
            "minicpm-v-4.7-35b-a3b",
            "minicpm_v_4_7_35b_a3b",
            MOE_CONFIG,
            "MiniCPMV47MoeForConditionalGeneration",
        ),
    ),
)
def test_minicpm_v47_recipes_point_at_root_metadata_workflows(
    model_id: str,
    export_name: str,
    config_path: Path,
    model_type: str,
) -> None:
    import xhmodel_merak.model_integrations.minicpm_v_4_7 as package

    assert "MiniCPM-V-4.7" in (package.__doc__ or "")

    recipe = get_recipe(model_id)
    assert recipe.family == "minicpm-v-4.7"
    assert recipe.export_name == export_name
    assert recipe.artifact_format == "minicpm_v_4_7"
    assert recipe.build_adapter == MINICPM_V_4_7_BUILD
    assert recipe.inference_adapter == MINICPM_V_4_7_INFERENCE
    assert recipe.components == COMPONENTS
    assert recipe.default_variant.workflow_config == str(config_path.relative_to(REPO_ROOT))

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    export = raw["export"]
    assert export["model"]["chip_arch"] == "XH2a"
    assert export["model"]["model_type"] == model_type
    assert set(export["components"]) == set(COMPONENTS)
    assert export["vision"]["token_capacity"] == 1536

    workflow = AutoWorkflow.from_config("/tmp/minicpm-v-4-7", str(config_path))
    assert type(workflow).__name__ == "MiniCPMV47Workflow"
    assert set(workflow.SUPPORTED_COMPONENTS) == set(COMPONENTS)


def test_minicpm_v47_example_helpers_are_importable() -> None:
    workflow = _load_example("_ci_minicpm_v47_workflow", "minicpm_v_4_7_workflow.py")
    summarize = _load_example("_ci_minicpm_v47_summarize", "summarize_accuracy_table.py")
    accuracy = _load_example("_ci_minicpm_v47_accuracy", "run_accuracy_bench.py")
    float_demo = _load_example("_ci_minicpm_v47_float_demo", "float_demo.py")
    hmonnx_demo = _load_example("_ci_minicpm_v47_hmonnx_demo", "hmonnx_demo.py")

    overrides = workflow._overrides(
        SimpleNamespace(
            token_capacity=64,
            context_length=1024,
            prefill_length=128,
            quant_type="w8a8h1_sefp",
        )
    )
    assert overrides == {
        "export.vision.token_capacity": 64,
        "export.model.context_max_length": 1024,
        "export.model.prefill_chunk_length": 128,
        "export.components.vision_4x.quant_type": "w8a8h1_sefp",
        "export.components.vision_16x.quant_type": "w8a8h1_sefp",
        "export.components.llm.quant_type": "w8a8h1_sefp",
    }

    assert summarize._fmt("OCRBench", 123.4) == "123"
    assert summarize._delta("ChartQA_TEST", 10.0, 9.5) == "-0.50 pp"
    assert accuracy.strip_thinking("a<think>x</think>b") == "b"
    assert accuracy.ocrbench_match("answer is hello", ["Hello"]) is True
    assert str(float_demo._dtype("fp16")).endswith("float16")
    assert callable(hmonnx_demo.parse_args)
    assert callable(float_demo.parse_args)
