from __future__ import annotations

import torch
import torch.nn as nn
from torch.fx import Graph, GraphModule

from xhmodel_merak.xh_llm.models.qwen3_5_moe import XHQwen3_5MoeCausalLMModel
from xhmodel_merak.xh_llm.models.qwen3_5_moe.xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig
from xhquant.api import ConfigDict
from xhquant.core.graph import FrontendGraph
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.quantization.chain_api.to_quant_graph import to_quant_graph


def _linear_with_autoround_codes(bits: int) -> nn.Linear:
    linear = nn.Linear(4, 4, bias=False)
    low, high = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    codes = torch.arange(16, dtype=torch.int16).reshape(4, 4)
    codes = (codes.remainder(high - low + 1) + low).to(torch.int8)
    linear.register_buffer("quant_weight", codes, persistent=False)
    linear._autoround_bits = bits
    linear._autoround_group_size = 64
    linear._xhquant_weight_origin = "autoround-gptq-v2"
    return linear


def _tiny_mixed_frontend() -> FrontendGraph:
    root = nn.Module()
    root.w4 = _linear_with_autoround_codes(4)
    root.w8 = _linear_with_autoround_codes(8)
    root.float_proj = nn.Linear(4, 4, bias=False)
    root.lm_head = nn.Linear(4, 4, bias=False)
    root.lm_head._xhquant_weight_origin = "explicit-base-lm-head"
    root.moe = MoeBlock(
        "silu",
        1,
        expert_gate_proj_weight=torch.randn(2, 4, 4),
        expert_up_proj_weight=torch.randn(2, 4, 4),
        expert_down_proj_weight=torch.randn(2, 4, 4),
    )
    for projection in ("gate_proj", "up_proj", "down_proj"):
        root.moe.register_buffer(
            f"expert_{projection}_quant_weight",
            torch.randint(-8, 8, (2, 4, 4), dtype=torch.int8),
            persistent=False,
        )
    root.moe._autoround_bits = 4
    root.moe._autoround_group_size = 64
    root.moe._xhquant_weight_origin = "autoround-gptq-v2"

    graph = Graph()
    hidden = graph.placeholder("hidden")
    routing = graph.placeholder("routing")
    selected = graph.placeholder("selected")
    w4 = graph.call_module("w4", (hidden,))
    w8 = graph.call_module("w8", (w4,))
    float_proj = graph.call_module("float_proj", (w8,))
    lm_head = graph.call_module("lm_head", (float_proj,))
    moe = graph.call_module("moe", (hidden, routing, selected), {"fast_mode": False})
    graph.output((lm_head, moe))
    return FrontendGraph.convert(GraphModule(root, graph))


def _weight_schema(module) -> tuple[str, int, bool, int]:
    qspec = module.w_cfg.qspec
    return qspec.fp_mode.lower(), qspec.man_bit, bool(qspec.hidden_bit), qspec.nshare


def test_autoround_frontend_policy_produces_strict_effective_qmodule_schemas():
    cfg = XHQwen3_5MoeModelConfig(
        model_name="qwen3_5_moe_autoround_test",
        model_type="Qwen3_5MoeForCausalLM",
        visual_config=None,
        quant_scheme=dict(
            quant_type="w4a8h0_ssfp",
            nodes=dict(lm_head=dict(quant_type="w8a8h1_sefp")),
        ),
    )
    model = XHQwen3_5MoeCausalLMModel(cfg)
    frontend = _tiny_mixed_frontend()

    model._apply_autoround_linear_w_schemas(frontend)

    nodes = {node.target: node for node in frontend.graph.nodes if node.op == "call_module"}
    assert nodes["w4"].meta["quant_config"]["w_schema"] == {
        "bits": 4,
        "fp_mode": "ssfp",
        "hidden_bit": False,
        "nshare": 64,
    }
    assert nodes["w8"].meta["quant_config"]["w_schema"] == {
        "bits": 8,
        "fp_mode": "ssfp",
        "hidden_bit": False,
        "nshare": 64,
    }
    assert "quant_config" not in nodes["float_proj"].meta
    assert "quant_config" not in nodes["lm_head"].meta
    assert nodes["moe"].meta["quant_config"]["w_schema"] == {
        "bits": 4,
        "fp_mode": "ssfp",
        "hidden_bit": False,
        "nshare": 64,
    }

    quanted = to_quant_graph(frontend, "XH2a", ConfigDict(model.get_quant_cfg()))

    assert _weight_schema(quanted.w4) == ("ssfp", 4, False, 64)
    assert _weight_schema(quanted.w8) == ("ssfp", 8, False, 64)
    assert quanted.w4.logical_bits == 4
    assert quanted.w8.logical_bits == 8
    assert _weight_schema(quanted.float_proj) == ("ssfp", 4, False, 64)
    assert _weight_schema(quanted.lm_head) == ("sefp", 8, True, 64)
    assert _weight_schema(quanted.moe) == ("ssfp", 4, False, 64)


def test_autoround_frontend_policy_fails_closed_when_codes_are_missing():
    cfg = XHQwen3_5MoeModelConfig(
        model_name="qwen3_5_moe_autoround_test",
        model_type="Qwen3_5MoeForCausalLM",
        visual_config=None,
        quant_scheme=dict(quant_type="w4a8h0_ssfp"),
    )
    model = XHQwen3_5MoeCausalLMModel(cfg)
    frontend = _tiny_mixed_frontend()
    del frontend.w4.quant_weight

    try:
        model._apply_autoround_linear_w_schemas(frontend)
    except RuntimeError as exc:
        assert "without quant_weight" in str(exc)
    else:
        raise AssertionError("Missing AutoRound codes must fail closed")


def test_autoround_frontend_policy_fails_closed_for_non_g64_source_codes():
    cfg = XHQwen3_5MoeModelConfig(
        model_name="qwen3_5_moe_autoround_test",
        model_type="Qwen3_5MoeForCausalLM",
        visual_config=None,
        quant_scheme=dict(quant_type="w4a8h0_ssfp"),
    )
    model = XHQwen3_5MoeCausalLMModel(cfg)
    frontend = _tiny_mixed_frontend()
    frontend.w4._autoround_group_size = 128

    try:
        model._apply_autoround_linear_w_schemas(frontend)
    except RuntimeError as exc:
        assert "expected source group_size=64" in str(exc)
    else:
        raise AssertionError("Non-G64 AutoRound source codes must fail closed")


def test_autoround_frontend_accepts_checkpoint_with_only_w8_linear_codes():
    root = nn.Module()
    root.proj = _linear_with_autoround_codes(8)
    graph = Graph()
    hidden = graph.placeholder("hidden")
    graph.output(graph.call_module("proj", (hidden,)))
    frontend = FrontendGraph.convert(GraphModule(root, graph))
    model = XHQwen3_5MoeCausalLMModel(XHQwen3_5MoeModelConfig(
        model_name="qwen35_w8_checkpoint", model_type="Qwen3_5MoeForCausalLM", visual_config=None,
    ))
    model._apply_autoround_linear_w_schemas(frontend)
    node = next(node for node in frontend.graph.nodes if node.op == "call_module")
    assert node.meta["quant_config"]["w_schema"]["bits"] == 8


def test_packed_head_uses_its_checkpoint_schema_instead_of_float_head_default():
    root = nn.Module()
    root.lm_head = _linear_with_autoround_codes(4)
    graph = Graph()
    hidden = graph.placeholder("hidden")
    graph.output(graph.call_module("lm_head", (hidden,)))
    frontend = FrontendGraph.convert(GraphModule(root, graph))
    model = XHQwen3_5MoeCausalLMModel(XHQwen3_5MoeModelConfig(
        model_name="qwen35_packed_head", model_type="Qwen3_5MoeForCausalLM", visual_config=None,
        quant_scheme=dict(quant_type="w4a8h0_ssfp", nodes=dict(lm_head=dict(quant_type="w8a8h1_sefp"))),
    ))
    model._apply_autoround_linear_w_schemas(frontend)
    quanted = to_quant_graph(frontend, "XH2a", ConfigDict(model.get_quant_cfg()))
    assert _weight_schema(quanted.lm_head) == ("ssfp", 4, False, 64)
