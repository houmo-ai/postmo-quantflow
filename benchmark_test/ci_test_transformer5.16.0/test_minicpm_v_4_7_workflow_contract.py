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


@pytest.mark.parametrize("flag", ("--dspark-model-dir", "--dflash-model-dir"))
def test_dspark_cli_emits_canonical_settings(monkeypatch, flag):
    module = _load_example("minicpm_dspark_cli", "minicpm_v_4_7_workflow.py")
    monkeypatch.setattr(sys, "argv", ["workflow", "--model-dir", "target", flag, "draft",
                                    "--confidence-threshold", "0.3"])
    args = module.parse_args()
    overrides = module._overrides(args)
    assert overrides["export.model.spec_decode_mode"] == "dspark"
    assert overrides["export.model.dspark_config.hf_model"] == "draft"
    assert overrides["export.model.dspark_config.enable_confidence_truncation"] is True
    assert not any("dflash" in key for key in overrides)


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
    assert export["vision"]["patch_token_gears"] == [384, 784, 1536, 2816, 6144]
    assert "image_token_gears" not in export["vision"]
    assert "token_capacity" not in export["vision"]

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
        "export.vision.patch_token_gears": None,
        "export.vision.image_token_gears": None,
        "export.vision.token_capacity": 64,
        "export.model.context_max_length": 1024,
        "export.model.prefill_chunk_length": 128,
        "export.components.vision_4x.quant_type": "w8a8h1_sefp",
        "export.components.vision_16x.quant_type": "w8a8h1_sefp",
        "export.components.llm.quant_type": "w8a8h1_sefp",
    }
    gears = workflow._overrides(SimpleNamespace(
        token_capacity=None, image_token_gears=[96, 196, 384, 704, 1536],
        context_length=None, prefill_length=None, quant_type=None,
    ))
    assert gears == {
        "export.vision.patch_token_gears": None,
        "export.vision.token_capacity": None,
        "export.vision.image_token_gears": [96, 196, 384, 704, 1536],
    }
    patches = workflow._overrides(SimpleNamespace(
        token_capacity=None, patch_token_gears=[384, 784, 1536, 2816, 6144],
        context_length=None, prefill_length=None, quant_type=None,
    ))
    assert patches == {
        "export.vision.image_token_gears": None,
        "export.vision.token_capacity": None,
        "export.vision.patch_token_gears": [384, 784, 1536, 2816, 6144],
    }

    assert summarize._fmt("OCRBench", 123.4) == "123"
    assert summarize._delta("ChartQA_TEST", 10.0, 9.5) == "-0.50 pp"
    assert accuracy.strip_thinking("a<think>x</think>b") == "b"
    assert accuracy.ocrbench_match("answer is hello", ["Hello"]) is True
    assert str(float_demo._dtype("fp16")).endswith("float16")
    assert callable(hmonnx_demo.parse_args)
    assert callable(float_demo.parse_args)


@pytest.mark.parametrize("dspark", (False, True))
def test_minicpm35b_autoround_release_configs_keep_dynamic_vision(dspark):
    suffix = "_dspark" if dspark else ""
    path = MOE_CONFIG.parent / f"minicpm_v_4_7_moe_xh2a_w4a8_autoround{suffix}.yaml"
    raw = yaml.safe_load(path.read_text())
    assert raw["quant"]["method"] == "autoround"
    assert raw["quant"]["bits"] == 4
    assert raw["quant"]["iters"] == 400
    assert raw["export"]["vision"]["patch_token_gears"] == [384, 784, 1536, 2816, 6144]
    assert raw["export"]["model"]["quant_scheme"]["quant_type"] == "w4a8h0_ssfp"
    assert raw["export"]["components"]["vision_4x"]["quant_type"] == "w8a8h1_sefp"
    if dspark:
        assert raw["export"]["model"]["spec_decode_mode"] == "dspark"
        assert raw["export"]["model"]["num_draft_tokens"] == 3
    else:
        assert "spec_decode_mode" not in raw["export"]["model"]


@pytest.mark.parametrize("dspark", (False, True))
def test_minicpm35b_float_qdq_export_preserves_mixed_bits(dspark, tmp_path):
    """The SDK must retain W4 experts without lowering W8 host projections."""
    import onnx
    import torch
    from xhquant.nn import MoeBlock
    from xhquant.api import (
        PrecisionMode, convert_quanted_model_to_hmonnx, ptq_quantize,
        to_frontend_graph, to_quant_graph,
    )

    suffix = "_dspark" if dspark else ""
    path = MOE_CONFIG.parent / f"minicpm_v_4_7_moe_xh2a_w4a8_autoround{suffix}.yaml"
    scheme = yaml.safe_load(path.read_text())["export"]["model"]["quant_scheme"]

    class Projections(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = torch.nn.Linear(64, 64, bias=False)
            self.conv = torch.nn.Conv2d(64, 64, (1, 3), padding=(0, 1), groups=64, bias=False)
            self.experts = MoeBlock(
                "silu", 2,
                expert_gate_proj_weight=torch.randn(4, 64, 64) * 0.02,
                expert_up_proj_weight=torch.randn(4, 64, 64) * 0.02,
                expert_down_proj_weight=torch.randn(4, 64, 64) * 0.02,
            )

        def forward(self, hidden, scores):
            return (
                self.attention(hidden), self.conv(hidden.transpose(1, 2).unsqueeze(2)),
                self.experts(hidden, scores), torch.matmul(hidden, hidden.transpose(1, 2)),
            )

    inputs = [torch.randn(1, 2, 64).half(), torch.randn(1, 2, 4).softmax(-1).half()]
    frontend = to_frontend_graph(Projections().half().eval(), "TorchFX", inputs)
    quantized = to_quant_graph(frontend, "XH2a", scheme)
    assert quantized.attention.w_cfg.qspec.man_bit == 8
    assert quantized.conv.w_cfg.qspec.man_bit == 8
    assert quantized.experts.w_cfg.qspec.man_bit == 4
    ptq_quantize(quantized, [inputs], PrecisionMode.ALIGNED, ["cpu"],
                 auto_release_unused_parameters=True, infer_shape=False)
    output = tmp_path / "mixed.onnx"
    convert_quanted_model_to_hmonnx(quantized, inputs, str(output))
    graph = onnx.load(output)
    observed = {}
    matmuls = []
    for node in graph.graph.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type in ("Linear", "Conv2d", "MoeBlock"):
            observed[node.op_type] = attrs["hmfp_weight_man_bit"]
        if node.op_type == "MatMul":
            # HMONNX names the second MatMul activation as the weight operand.
            matmuls.append((attrs["hmfp_act_man_bit"], attrs["hmfp_weight_man_bit"]))
    assert observed == {"Linear": 8, "Conv2d": 8, "MoeBlock": 4}
    assert matmuls == [(16, 16)]
