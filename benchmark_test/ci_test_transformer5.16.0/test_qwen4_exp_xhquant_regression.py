from __future__ import annotations

import torch

import xhquant.nn as xhnn
from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import (
    Qwen4ExpExportableGatedDeltaNet,
    _bind_qwen4_exp_packed_output_dtype_anchor,
    _build_qwen4_exp_gptq_packed_moeblock,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.xhquant_extensions import (
    qwen4_exp_xhquant_extension_scope,
)
from xhquant.frontend.fx_transform.op_normalizers.op_normalizer_impl import OP_NORMALIZER
from xhquant.quantization.xh2a.builder import NONE_QUANTED_MODULES
from xhquant.xhonnxruntime.builder import HMONNX_PARSERS


def _packed_linear(in_features: int, out_features: int) -> xhnn.GPTQPackedLinear:
    bits = 4
    pack_factor = 8
    packed_weight = xhnn.GPTQPackedWeight(
        qweight=torch.zeros(in_features // pack_factor, out_features, dtype=torch.int32),
        qzeros=torch.zeros(in_features // 8, out_features // pack_factor, dtype=torch.int32),
        scales=torch.ones(in_features // 8, out_features),
        g_idx=torch.arange(in_features, dtype=torch.int32) // 8,
        bits=bits,
        group_size=8,
        in_features=in_features,
        out_features=out_features,
        qzero_format=2,
    )
    return xhnn.GPTQPackedLinear(packed_weight)


def test_defused_gptq_experts_become_one_packed_moe_block() -> None:
    experts = torch.nn.ModuleList()
    for _ in range(2):
        expert = torch.nn.Module()
        expert.gate_proj = _packed_linear(8, 8)
        expert.up_proj = _packed_linear(8, 8)
        expert.down_proj = _packed_linear(8, 8)
        experts.append(expert)

    block = _build_qwen4_exp_gptq_packed_moeblock(experts, "silu", 1)

    assert isinstance(block, xhnn.GPTQPackedMoeBlock)
    assert len(block.expert_gate_proj_packed) == 2
    assert len(block.expert_up_proj_packed) == 2
    assert len(block.expert_down_proj_packed) == 2
    assert block.expert_gate_proj_weight.shape == ()
    assert block.expert_gate_proj_weight.dtype is block.expert_gate_proj_packed[0].scales.dtype
    assert "expert_gate_proj_weight" not in block._buffers
    assert "expert_gate_proj_weight" not in block.state_dict()


def test_gated_delta_setup_dtype_comes_from_packed_scales() -> None:
    gated_delta = Qwen4ExpExportableGatedDeltaNet()
    gated_delta.split_conv_cache = False
    gated_delta.in_proj_qkv = _packed_linear(8, 8)

    assert gated_delta.in_proj_qkv.weight is None
    assert gated_delta._input_projection_dtype() is gated_delta.in_proj_qkv.packed_weight.scales.dtype


def test_gated_delta_packed_output_exposes_only_an_unregistered_dtype_anchor() -> None:
    projection = _packed_linear(8, 8)
    packed_state_keys = tuple(projection.state_dict())

    _bind_qwen4_exp_packed_output_dtype_anchor(projection)

    assert type(projection) is xhnn.GPTQPackedLinear
    assert projection.weight.shape == ()
    assert projection.weight.dtype is projection.packed_weight.scales.dtype
    assert "weight" not in projection._parameters
    assert tuple(projection.state_dict()) == packed_state_keys


def test_qwen4_extension_scope_restores_shared_xhquant_registries() -> None:
    normalizers_before = dict(OP_NORMALIZER.D)
    none_quanted_before = list(NONE_QUANTED_MODULES)
    parsers_before = dict(HMONNX_PARSERS.module_dict)

    with qwen4_exp_xhquant_extension_scope():
        for name in ("Cast", "BitShift", "Clip", "Expand", "GatherElements", "XH2a::Conv1d"):
            assert HMONNX_PARSERS.module_dict[name] is parsers_before[name]
        assert xhnn.FloorDiv in NONE_QUANTED_MODULES

    assert OP_NORMALIZER.D == normalizers_before
    assert NONE_QUANTED_MODULES == none_quanted_before
    assert HMONNX_PARSERS.module_dict == parsers_before
