from __future__ import annotations

import pytest
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


def _packed_linear(in_features: int, out_features: int, group_size: int = 8) -> xhnn.GPTQPackedLinear:
    bits = 4
    pack_factor = 8
    packed_weight = xhnn.GPTQPackedWeight(
        qweight=torch.zeros(in_features // pack_factor, out_features, dtype=torch.int32),
        qzeros=torch.zeros(in_features // group_size, out_features // pack_factor, dtype=torch.int32),
        scales=torch.ones(in_features // group_size, out_features),
        g_idx=torch.arange(in_features, dtype=torch.int32) // group_size,
        bits=bits,
        group_size=group_size,
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mrope_cache_matches_native_dtype_and_default_eager_values(dtype):
    from test_qwen4_exp_workflow_quant_adapter import _tiny_stateful_model
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding

    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import build_qwen4_exp_text_export_wrapper
    from xhmodel_merak.xh_llm.mrope import StaticMRotaryEmbedding

    native = _tiny_stateful_model().to(dtype=dtype)
    rotary = native.model.rotary_emb
    inv_freq, scale = Qwen4ExpTextRotaryEmbedding.compute_default_rope_parameters(native.config, device="cpu")
    reference = StaticMRotaryEmbedding(inv_freq, scale, native.config.max_position_embeddings, rotary.mrope_section)
    wrapper = build_qwen4_exp_text_export_wrapper(native, input_sequence_length=4, cache_length=8)
    actual = wrapper.native_text_model.rotary_emb
    assert actual.cos_cached.dtype == actual.sin_cached.dtype == dtype
    args = (
        torch.zeros(1, 4, 16, dtype=dtype), torch.tensor([[0, 2, 7, 15]], dtype=torch.int32),
        torch.tensor([[1, 3, 8, 14]], dtype=torch.int32), torch.tensor([[2, 4, 9, 13]], dtype=torch.int32),
    )
    for observed, expected in zip(actual(*args), reference(*args), strict=True):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_fp16_mrope_tables_remain_shared_through_standard_torchfx_ptq():
    from test_qwen4_exp_workflow_quant_adapter import _tiny_stateful_model

    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import build_qwen4_exp_text_export_wrapper
    from xhquant.api import ConfigDict, FrontendType, PrecisionMode, ptq_quantize, to_frontend_graph, to_quant_graph

    native = _tiny_stateful_model().half()
    wrapper = build_qwen4_exp_text_export_wrapper(native, input_sequence_length=4, cache_length=8)
    args = (torch.zeros(1, 4, 16).half(), *(torch.arange(4, dtype=torch.int32).view(1, 4) for _ in range(3)))
    with qwen4_exp_xhquant_extension_scope():
        frontend = to_frontend_graph(wrapper.native_text_model.rotary_emb, FrontendType.TorchFX, args)
        quanted = to_quant_graph(frontend, "xh2a", ConfigDict({"quant_type": "w16a16h1_sefp"}))
        ptq_quantize(quanted, [list(args)], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    tables = [value for name, value in quanted.named_buffers(remove_duplicate=False)
              if "cached" in name and tuple(value.shape) == (16, 1, 4)]
    assert len(tables) >= 6  # Each T/H/W Gather owns a registration of its source table.
    assert len({id(value) for value in tables}) == 2
    assert {value.dtype for value in tables} == {torch.float16}


@pytest.mark.parametrize("grouped_input", [False, True], ids=["flat-input", "grouped-input"])
def test_grouped_norm_keeps_batch_and_sequence_axes(grouped_input):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import Qwen4ExpExportableRMSNorm

    native = Qwen4ExpTextRMSNorm(32, group_size=16, eps=1e-5).eval()
    with torch.no_grad():
        native.weight.normal_(mean=0, std=0.1)
    adapted = Qwen4ExpExportableRMSNorm(native).eval()
    adapted.input_is_grouped = grouped_input
    observed = []
    handle = adapted.norm.register_forward_pre_hook(lambda module, args: observed.append(tuple(args[0].shape)))
    values = torch.randn(1, 4, 32)
    try:
        actual = adapted(values.reshape(1, 4, 2, 16) if grouped_input else values)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, native(values))
    assert observed == [(1, 4, 2, 16)]


@pytest.mark.parametrize("length", [4, 1], ids=["prefill", "decode"])
@pytest.mark.parametrize("value_head_repeat", [1, 3], ids=["equal-heads", "threefold-value-heads"])
def test_gdr_merged_projection_feeds_convolution_without_qkv_roundtrip(length, value_head_repeat):
    import copy
    from types import SimpleNamespace

    from test_qwen4_exp_workflow_quant_adapter import _tiny_stateful_model

    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import adapt_qwen4_exp_rms_norms
    from xhquant.api import FrontendType, to_frontend_graph
    from xhquant.core import CacheTensor

    model = _tiny_stateful_model()
    config = copy.deepcopy(model.config)
    config.linear_num_value_heads *= value_head_repeat
    native = type(model.model.layers[0].linear_attn)(config, layer_idx=0).eval()
    adapt_qwen4_exp_rms_norms(native)
    layer = Qwen4ExpExportableGatedDeltaNet.from_native(native, batch_size=1, sequence_length=length)
    layer.enable_fused_gdr()

    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = layer

        def forward(self, hidden, conv, recurrent, current, mask):
            cache = SimpleNamespace(
                layers=[SimpleNamespace(conv_states=[conv], recurrent_states=[recurrent])],
                current_input_length=current,
            )
            return self.layer(hidden, cache_params=cache, attention_mask=mask)

    args = (
        torch.randn(1, length, 16), torch.zeros(1, 16 + 8 * value_head_repeat, 4),
        CacheTensor(torch.zeros(1, 2 * value_head_repeat, 4, 4)),
        torch.tensor([length], dtype=torch.int32), torch.ones(1, length),
    )
    with qwen4_exp_xhquant_extension_scope():
        frontend = to_frontend_graph(Step().eval(), FrontendType.TorchFX, args)
    projection, = [node for node in frontend.graph.nodes
                    if node.op == "call_module" and str(node.target).endswith("in_proj_qkv")]
    consumer, = projection.users
    assert type(frontend.get_submodule(str(consumer.target))).__name__ in {"Permute", "Transpose"}


def test_grouped_hyper_connection_entry_preserves_flat_residual():
    import copy

    from test_qwen4_exp_workflow_quant_adapter import _tiny_stateful_model

    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import (
        Qwen4ExpExportableGatedResidual,
        adapt_qwen4_exp_rms_norms,
    )
    from xhquant.api import FrontendType, to_frontend_graph

    native = _tiny_stateful_model().model.layers[0].attn_hyper_connection.eval()
    layer = copy.deepcopy(native)
    adapt_qwen4_exp_rms_norms(layer)
    layer.__class__ = Qwen4ExpExportableGatedResidual
    layer.hc_norm.input_is_grouped = True
    grouped = torch.randn(1, 4, 2, 16)
    expected = native(grouped.flatten(-2))
    with qwen4_exp_xhquant_extension_scope():
        frontend = to_frontend_graph(layer, FrontendType.TorchFX, (grouped,))
    actual = frontend(grouped)
    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, reference)
    norm, = [node for node in frontend.graph.nodes if node.op == "call_module" and node.target == "hc_norm.norm"]
    source = norm.args[0]
    while source.op == "call_module" and isinstance(frontend.get_submodule(str(source.target)), xhnn.Cast):
        source = source.args[0]
    assert source.op == "placeholder"


@pytest.mark.parametrize("assistant", [False, True], ids=["target", "mtp"])
@pytest.mark.parametrize("length", [4, 1], ids=["prefill", "decode"])
def test_static_wrapper_torchfx_preserves_logits_and_every_state(assistant, length):
    """Trace real HC/MoE/QSA and target PLE/GDR through the ordinary frontend."""
    from test_qwen4_exp_workflow_quant_adapter import _tiny_stateful_model

    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import resolve_qwen4_exp_cache_shape
    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import build_qwen4_exp_text_export_wrapper
    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_model import Qwen4ExpMTPExportWrapper, Qwen4ExpMTPModel
    from xhmodel_merak.xh_llm.models.qwen4_exp.qsa_host_controls import build_qsa_host_controls
    from xhmodel_merak.xh_llm.utils import unfold_args
    from xhquant.api import FrontendType, to_frontend_graph
    from xhquant.core import CacheTensor

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1701)
        source = _tiny_stateful_model()
        positions = torch.arange(length, dtype=torch.int32).reshape(1, -1)
        fixed = [torch.randn(1, length, 16)]
        if assistant:
            source.config.mtp = {"num_hidden_layers": 1, "layer_types": ["full_attention"]}
            wrapper = Qwen4ExpMTPExportWrapper(
                Qwen4ExpMTPModel(source.config).eval(), torch.nn.Linear(16, 32, bias=False).eval(),
                input_sequence_length=length, cache_length=8,
            ).eval()
            fixed.append(torch.randn(1, length, 32))
        else:
            wrapper = build_qwen4_exp_text_export_wrapper(
                source, input_sequence_length=length, cache_length=8,
            ).eval()
        fixed.extend((torch.tensor([0], dtype=torch.int32), torch.tensor([length], dtype=torch.int32)))
        if not assistant:
            fixed.append(torch.arange(2, 2 + length, dtype=torch.int32).reshape(1, -1))
        fixed.extend(positions.clone() for _ in range(3))
        fixed.extend(torch.zeros(1, (length + 3) // 4, dtype=torch.int32) for _ in range(3))
        state = [
            torch.zeros(
                resolve_qwen4_exp_cache_shape(spec, batch_size=1, context_length=8),
                dtype=torch.int32 if spec.dtype == "integer" else torch.float32,
            )
            for spec in wrapper.cache_contract.tensors
        ]
        controls = build_qsa_host_controls(0, length, length, 8, 1, device="cpu", dtype=torch.float32)

        def inputs():
            caches = tuple(
                CacheTensor(value.clone()) if spec.state_kind in {"kv_key", "kv_value", "qsa_indexer_key"}
                else value.clone()
                for spec, value in zip(wrapper.cache_contract.tensors, state, strict=True)
            )
            return (*[value.clone() for value in fixed], (*caches, *[value.clone() for value in controls]))

        with torch.no_grad():
            eager_inputs, traced_inputs = inputs(), inputs()
            expected = tuple(value.detach().clone() for value in wrapper(*eager_inputs))
            with qwen4_exp_xhquant_extension_scope():
                frontend = to_frontend_graph(wrapper, FrontendType.TorchFX, inputs())
            actual = frontend(*unfold_args(traced_inputs))
        core = wrapper.core if assistant else wrapper
        assert len(actual) == len(expected) == len(core.output_cache_specs()) + (2 if assistant else 1)
        for output, reference in zip(actual, expected, strict=True):
            # Frontend affine/constant fusion changes FP32 rounding slightly;
            # integer history and cache dtypes still have to agree exactly.
            torch.testing.assert_close(output, reference, rtol=1e-5, atol=3e-6)
        for index, spec in enumerate(wrapper.cache_contract.tensors):
            if spec.state_kind in {"kv_key", "kv_value", "qsa_indexer_key"}:
                torch.testing.assert_close(traced_inputs[-1][index], eager_inputs[-1][index], rtol=1e-5, atol=3e-6)


def test_packed_moe_uses_standard_frontend_without_capture_adapter(tmp_path):
    from pathlib import Path

    from xhmodel_merak.xh_llm.models.qwen4_exp.static_export import export_qwen4_exp_hmonnx
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    experts = torch.nn.ModuleList()
    for index in range(2):
        expert = torch.nn.Module()
        for role in ("gate_proj", "up_proj", "down_proj"):
            projection = _packed_linear(64, 64, group_size=64).half()
            projection.packed_weight.qweight.fill_(0x11111111 * (index + 1))
            projection.packed_weight.scales.fill_(0.125)
            setattr(expert, role, projection)
        experts.append(expert)

    class PackedStep(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = _build_qwen4_exp_gptq_packed_moeblock(experts, "silu", 1)

        def forward(self, values, routes):
            return self.block(values, routes)

    module = PackedStep().eval()
    inputs = torch.linspace(-0.2, 0.3, 128).reshape(1, 2, 64).half()
    routes = torch.tensor([[[0.8, 0.2], [0.1, 0.9]]], dtype=torch.float16)
    with torch.no_grad():
        expected = module(inputs, routes)
    assert expected.count_nonzero().item() > 0
    original = {name: value.clone() for name, value in module.state_dict().items()}
    result = export_qwen4_exp_hmonnx(
        module, tmp_path, sample_inputs=(inputs, routes), export_wrapper=module,
        export_input_names=("values", "routes"), export_output_names=("output",),
        ple_quant_type="fp16", quant_config={"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True},
        validate_export_graph_parity=True,
    )
    session = HMONNXInferenceV2(
        str(Path(result.hmonnx_path)), session_config=HMONNXInferenceConfig(exec_devices=["cpu"])
    )
    actual = session.run({"values": inputs, "routes": routes})
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
