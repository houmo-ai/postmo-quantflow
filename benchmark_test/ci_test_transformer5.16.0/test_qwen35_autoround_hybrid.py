"""Producer payloads survive native hybrid setup in model-owned adapters."""

from math import prod

import pytest
import torch
from transformers import Qwen3_5MoeTextConfig, Qwen4ExpTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeRMSNormGated,
    Qwen3_5MoeSparseMoeBlock,
)
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedDeltaNet, Qwen4ExpTextSparseMoeBlock

from xhmodel_merak.xh_llm.models.qwen3_5._hybrid_gated_delta_net import HybridRMSNormGatedMixin
from xhmodel_merak.xh_llm.models.qwen3_5_moe._moe_model import (
    _Qwen3_5MoeGatedDeltaNet,
    _Qwen3_5MoeSparseMoeBlock,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import (
    Qwen4ExpExportableGatedDeltaNet,
    Qwen4ExpExportableSparseMoe,
)
from xhmodel_merak.xh_llm.wrap_model import wrap_llm_model
from xhquant.api import ConfigDict


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_shared_gated_norm_retains_qwen35_silu_semantics(dtype):
    class AdaptedNorm(HybridRMSNormGatedMixin, torch.nn.Module):
        def __init__(self, native):
            super().__init__()
            self.weight = native.weight
            self.variance_epsilon = native.variance_epsilon
            self._setup()

    torch.manual_seed(19)
    native = Qwen3_5MoeRMSNormGated(16).to(dtype).eval()
    native.weight.data.uniform_(0.5, 1.5)
    adapted = AdaptedNorm(native).eval()
    hidden_states = torch.randn(2, 3, 16, dtype=dtype)
    gate = torch.randn_like(hidden_states)
    torch.testing.assert_close(adapted(hidden_states, gate), native(hidden_states, gate), rtol=2e-3, atol=2e-3)


def _config(config_cls):
    return config_cls(
        hidden_size=64, linear_num_key_heads=1, linear_key_head_dim=16,
        linear_num_value_heads=2, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, num_experts=2, num_experts_per_tok=1,
        moe_intermediate_size=64, shared_expert_intermediate_size=64,
    )


def _payload(module, shape, *, prefix="", bits=4):
    codes = ((torch.arange(prod(shape)).reshape(shape) % (1 << bits)) - (1 << (bits - 1)))
    codes = codes.to(torch.int8)
    scale_shape = (*shape[:-2], shape[-1] // 64, shape[-2])
    scales = torch.arange(prod(scale_shape), dtype=torch.float16).reshape(scale_shape) / 16 + 1
    module.register_buffer(prefix + "quant_weight", codes, persistent=False)
    module.register_buffer(prefix + "quant_scale", scales, persistent=False)
    return codes, scales


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("family", ["qwen35", "qwen4"])
def test_native_split_gdn_retains_exact_codes_scales_and_source_metadata(family, bits):
    if family == "qwen35":
        native = Qwen3_5MoeGatedDeltaNet(_config(Qwen3_5MoeTextConfig), layer_idx=0).half()
    else:
        native = Qwen4ExpTextGatedDeltaNet(_config(Qwen4ExpTextConfig), layer_idx=0).half()
    source = native.in_proj_qkv
    codes, scales = _payload(source, source.weight.shape, bits=bits)
    source._autoround_bits = bits
    source._autoround_group_size = 64
    source._xhquant_weight_origin = "autoround-gptq-v2"
    source_weights = source.weight.detach().clone()
    cfg = ConfigDict(use_cache=False, split_conv_cache=True, input_sequence_length=2, linear_chunk_size=8, batch_size=1)
    if family == "qwen35":
        wrapped = wrap_llm_model(native, cfg)
        assert isinstance(wrapped, _Qwen3_5MoeGatedDeltaNet)
    else:
        native.__class__ = Qwen4ExpExportableGatedDeltaNet
        wrapped = native
        wrapped._setup(cfg)
    widths = [wrapped.key_dim, wrapped.key_dim, wrapped.value_dim]
    for target, expected_codes, expected_scales, expected_weights in zip(
        (wrapped.in_proj_q, wrapped.in_proj_k, wrapped.in_proj_v),
        codes.split(widths, dim=0), scales.split(widths, dim=1), source_weights.split(widths, dim=0), strict=True,
    ):
        torch.testing.assert_close(target.quant_weight, expected_codes, rtol=0, atol=0)
        torch.testing.assert_close(target.quant_scale, expected_scales, rtol=0, atol=0)
        torch.testing.assert_close(target.weight, expected_weights, rtol=0, atol=0)
        assert target._autoround_bits == bits
        assert target._autoround_group_size == 64
        assert target._xhquant_weight_origin == "autoround-gptq-v2"
        assert target._autoround_split_from == "in_proj_qkv"
        assert "quant_weight" not in target.state_dict()
        assert "quant_scale" not in target.state_dict()


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("family", ["qwen35", "qwen4"])
def test_native_packed_experts_transfer_exact_payloads_without_copying_weights(family, bits):
    if family == "qwen35":
        native = Qwen3_5MoeSparseMoeBlock(_config(Qwen3_5MoeTextConfig)).half()
    else:
        native = Qwen4ExpTextSparseMoeBlock(_config(Qwen4ExpTextConfig)).half()
    experts = native.experts
    expected = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        shape = (2, 64, 64)
        expected[projection] = _payload(experts, shape, prefix=projection + "_", bits=bits)
    experts._autoround_expert_bits = bits
    experts._autoround_group_size = 64
    experts._xhquant_weight_origin = "autoround-gptq-v2"
    if family == "qwen35":
        wrapped = wrap_llm_model(native, ConfigDict(batch_size=1, input_sequence_length=2))
        assert isinstance(wrapped, _Qwen3_5MoeSparseMoeBlock)
    else:
        wrapped = Qwen4ExpExportableSparseMoe.from_native(native, batch_size=1, sequence_length=2)
    for projection, (codes, scales) in expected.items():
        assert getattr(wrapped.moeblock, f"expert_{projection}_quant_weight") is codes
        assert getattr(wrapped.moeblock, f"expert_{projection}_quant_scale") is scales
        assert getattr(experts, f"{projection}_quant_weight").device.type == "meta"
        assert getattr(experts, f"{projection}_quant_scale").device.type == "meta"
    assert wrapped.moeblock._autoround_bits == bits
    assert wrapped.moeblock._autoround_group_size == 64
    assert wrapped.moeblock._xhquant_weight_origin == "autoround-gptq-v2"


@pytest.mark.parametrize("family", ["qwen35", "qwen4"])
def test_invalid_split_payload_is_rejected_before_setup_discards_source(family):
    if family == "qwen35":
        native = Qwen3_5MoeGatedDeltaNet(_config(Qwen3_5MoeTextConfig), layer_idx=0).half()
    else:
        native = Qwen4ExpTextGatedDeltaNet(_config(Qwen4ExpTextConfig), layer_idx=0).half()
    source = native.in_proj_qkv
    source.register_buffer("quant_weight", torch.zeros(source.weight.shape[0] - 1, 64, dtype=torch.int8))
    cfg = ConfigDict(use_cache=False, split_conv_cache=True, input_sequence_length=2, linear_chunk_size=8, batch_size=1)
    with pytest.raises(ValueError, match="quant_weight shape.*does not match"):
        if family == "qwen35":
            wrap_llm_model(native, cfg)
        else:
            native.__class__ = Qwen4ExpExportableGatedDeltaNet
            native._setup(cfg)
    assert native.in_proj_qkv is source
    assert not hasattr(native, "in_proj_q")


def test_static_gdn_length_remains_traceable_by_strict_dynamo_export():
    from xhmodel_merak.xh_llm.models.qwen3_5._hybrid_gated_delta_net import _resolve_python_int_length

    class StaticLength(torch.nn.Module):
        def forward(self, hidden):
            return hidden + _resolve_python_int_length(hidden.shape[1])

    hidden = torch.ones(2, 3)
    exported = torch.export.export(StaticLength(), (hidden,), strict=True)
    torch.testing.assert_close(exported.module()(hidden), hidden + 3, rtol=0, atol=0)
