"""Consume producer-absorbed Xing weights and preserve their packed codes."""

import copy

import pytest
import torch
from torch import nn

from types import SimpleNamespace


def _attention():
    from xhmodel_merak.xh_llm.models.xing4_0._model import _Xing4_0Attention

    module = object.__new__(_Xing4_0Attention)
    nn.Module.__init__(module)
    module.config = SimpleNamespace(num_attention_heads=2, hidden_size=64)
    for name, value in dict(
        q_lora_rank=64, qk_nope_head_dim=32, qk_rope_head_dim=32,
        qk_head_dim=64, kv_lora_rank=64, v_head_dim=32, scaling=0.125,
    ).items():
        setattr(module, name, value)
    for index, (name, shape) in enumerate({
        "q_a_proj": (64, 64), "q_b_proj": (64, 128), "kv_a_proj_with_mqa": (64, 96),
        "kv_b_proj": (64, 128), "o_proj": (64, 64),
    }.items()):
        linear = nn.Linear(*shape, bias=False, dtype=torch.float32, device="cpu")
        values = torch.arange(linear.weight.numel(), dtype=torch.float32).reshape_as(linear.weight)
        with torch.no_grad():
            linear.weight.copy_(torch.sin(values.square() * 0.013 + index) * 0.1)
        setattr(module, name, linear)
    return module


def _packed_w8(linear):
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from xhmodel_merak.xh_llm.base_llm_model import gptqmodel_torch_qlinear_converter

    groups = linear.weight.detach().reshape(-1, 64)
    scales = (groups.abs().amax(-1) / 127).clamp_min(1e-5).reshape(linear.out_features, -1)
    packed = TorchQuantLinear(
        bits=8, group_size=64, sym=True, desc_act=False,
        in_features=linear.in_features, out_features=linear.out_features, bias=False,
    )
    packed.pack(linear, scales, torch.full_like(scales, 128), torch.arange(linear.in_features, dtype=torch.int32) // 64)
    packed._init_wf_unsqueeze_buffers()
    packed.dequant_dtype = torch.int16
    gptqmodel_torch_qlinear_converter(packed)
    return packed.to(torch.float32)


def _producer_absorbed():
    from gptqmodel.models.definitions.xing4_0_mla import _build_absorbed

    attention = _attention()
    attention.num_heads = 2
    _build_absorbed(attention, from_weights=True)
    attention.config.mla_weight_layout = "xh2_absorbed_v1"
    attention.config.mla_transform = {
        "version": 1,
        "matmul_dtype": "float32",
        "cache_layout": "separate_k_latent_rope_v_latent",
    }
    return attention


def test_preabsorbed_setup_keeps_linear_payload_and_separate_caches(monkeypatch):
    from xhquant.backend.xh2a.functions.te.linear import (
        linear_unfold_weight_xh2a_default,
    )

    attention = _producer_absorbed()
    expected = {}
    for name in ("q_absorbed_proj", "q_rope_proj", "o_proj", "kv_a_proj_with_mqa"):
        linear = _packed_w8(getattr(attention, name))
        setattr(attention, name, linear)
        expected[name] = linear, linear.weight.clone(), linear.quant_weight.clone()

    def no_absorption(*args, **kwargs):
        raise AssertionError("Consumer must not absorb producer weights again")

    monkeypatch.setattr(torch, "einsum", no_absorption)
    attention._setup({"use_cache": True, "enable_rope": False})
    assert attention.k_cache is not attention.v_cache
    assert attention.kv_scale == attention.scaling
    for name in ("q_absorbed_proj", "q_rope_proj", "o_proj"):
        linear, weight, codes = expected[name]
        assert getattr(attention, name) is linear
        assert torch.equal(linear.weight, weight) and torch.equal(
            linear.quant_weight, codes
        )
        linear.half()
        linear.weight.quant_val = linear.quant_weight.clone()
        packed = linear_unfold_weight_xh2a_default(
            linear.weight, mode="ssfp", bit=8, nshare=64
        )
        integers, scales, bits = packed.unpack(is_scale=True)
        assert bits == 8
        assert torch.equal(
            integers.reshape(linear.in_features, linear.out_features).T, codes
        )
        assert torch.isfinite(scales).all() and (scales > 0).all()
    kv_codes = expected["kv_a_proj_with_mqa"][2]
    assert torch.equal(attention.kv_a_proj_latent.quant_weight, kv_codes[:64])
    assert torch.equal(attention.kv_a_proj_rope.quant_weight, kv_codes[64:])


@pytest.mark.parametrize("broken", ["missing", "mixed", "version", "relabel", "shape"])
def test_absorbed_consumer_rejects_invalid_layout(broken):
    attention = _producer_absorbed()
    if broken == "missing":
        del attention.q_absorbed_proj
    elif broken == "mixed":
        attention.kv_b_proj = nn.Linear(64, 128)
    elif broken == "version":
        attention.config.mla_transform["version"] = 2
    elif broken == "shape":
        attention.o_proj = nn.Linear(64, 64)
    else:
        attention = _attention()
        attention.config.mla_weight_layout = "xh2_absorbed_v1"
    with pytest.raises(ValueError, match="Xing MLA"):
        attention._setup({"use_cache": False})


def test_k_v_equal_latents_have_independent_quantization_domains():
    from xhquant.backend.hmops.sefp.sefp import sefp

    latent = torch.ones(1, 1, 64, 64, dtype=torch.float16)
    latent[..., 0, 0] = 1000
    key, k_exp = sefp(latent.clone(), dim=-1, manbit=7, nshare=64, assemble=True)
    value, v_exp = sefp(latent.clone(), dim=-2, manbit=7, nshare=64, assemble=True)
    assert not torch.equal(key, value)
    assert k_exp.shape == (1, 1, 64, 1) and v_exp.shape == (1, 1, 1, 64)


def test_preabsorbed_graph_forward_matches_legacy_float():
    attention = _attention()
    attention.num_heads = 2
    attention.config.rope_interleave = True
    attention.q_a_layernorm = nn.RMSNorm(64)
    attention.kv_a_layernorm = nn.RMSNorm(64)
    # Install the producer's exact floating boundaries before consumption.
    from gptqmodel.models.definitions.xing4_0_mla import _build_absorbed

    absorbed = copy.deepcopy(attention)
    _build_absorbed(absorbed, from_weights=True)
    absorbed.config.mla_weight_layout = "xh2_absorbed_v1"
    absorbed.config.mla_transform = {
        "version": 1,
        "matmul_dtype": "float32",
        "cache_layout": "separate_k_latent_rope_v_latent",
    }
    for module in (attention, absorbed):
        module._setup({"use_cache": False, "enable_rope": False})
    hidden = torch.randn(1, 3, 64)
    cos = torch.ones(1, 1, 3, 32)
    sin = torch.zeros_like(cos)
    torch.testing.assert_close(
        absorbed.graph_forward(
            hidden, past_seq_length=torch.tensor([0]), position_embeddings=(cos, sin)
        )[0],
        attention.graph_forward(
            hidden, past_seq_length=torch.tensor([0]), position_embeddings=(cos, sin)
        )[0],
        atol=2e-6,
        rtol=1e-5,
    )


def test_preabsorbed_attention_hmonnx_round_trip(tmp_path):
    from xhquant.nn import RMSNorm
    from xhquant.api import (
        PrecisionMode,
        ptq_quantize,
        to_export_hmonnx_from_quanted_graph,
        to_frontend_graph,
        to_quant_graph,
    )
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import (
        HMONNXInferenceConfig,
        HMONNXInferenceV2,
    )

    attention = _producer_absorbed()
    attention.config.rope_interleave = True
    attention.q_a_layernorm = RMSNorm(64)
    attention.kv_a_layernorm = RMSNorm(64)
    for name in (
        "q_a_proj",
        "q_absorbed_proj",
        "q_rope_proj",
        "o_proj",
        "kv_a_proj_with_mqa",
    ):
        setattr(attention, name, _packed_w8(getattr(attention, name)))
    attention._setup({"use_cache": False, "enable_rope": False})

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = attention

        def forward(self, hidden, past, cos, sin):
            return self.attention.graph_forward(
                hidden, past_seq_length=past, position_embeddings=(cos, sin)
            )[0]

    args = [
        torch.linspace(-1, 1, 192).half().reshape(1, 3, 64),
        torch.tensor([0], dtype=torch.int32),
        torch.ones(1, 1, 3, 32).half(),
        torch.zeros(1, 1, 3, 32).half(),
    ]
    graph = to_quant_graph(
        to_frontend_graph(Attention().eval().half(), "TorchFX", args),
        "XH2a",
        {"quant_type": "w4a8h0_ssfp"},
    )
    ptq_quantize(graph, [args], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    expected = graph(*args)
    path = to_export_hmonnx_from_quanted_graph(
        graph,
        args,
        str(tmp_path / "attention.onnx"),
        save_as_external_data=True,
    )
    session = HMONNXInferenceV2(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))
    actual = session(*args)
    if isinstance(actual, (tuple, list)):
        (actual,) = actual
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
