"""CUDA SEFP VAE single heads: full dot, tiled values and true-length masking."""

import pytest
import torch


@pytest.mark.parametrize("head_dim", [768, 1152])
@pytest.mark.parametrize("quant_type, bits", [("w16a16_sefp", 16), ("w8a8_sefp", 8)])
def test_vae_large_head_hmonnx_export_load_and_execute(tmp_path, head_dim, quant_type, bits):
    import onnx

    from xhmodel_merak.xh_other_model.models.qwen_image21.export import export_cpu_graph
    from xhquant.api import ConfigDict, DeviceType, QuantScheme, create_quant_config
    from xhquant.backend.hmops.fused_ops.flash_attention.xh2.flash_attention_xh2 import flash_attention_xh2
    from xhquant.nn import FlashAttention
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = FlashAttention(num_heads=1, is_causal=False, scale=head_dim**-0.5)

        def forward(self, q, k, v, length):
            return self.attention(q, k, v, current_input_length=length)

    torch.manual_seed(12)
    values = [torch.randn(1, 1, 72, head_dim) for _ in range(3)]
    length = torch.tensor([69], dtype=torch.int32)
    path = tmp_path / "attention.onnx"
    export_cpu_graph(
        Attention(),
        (*values, length),
        path,
        ["q", "k", "v", "current_input_length"],
        ["attention_output"],
        target_device=DeviceType.XH2a,
        quant_config=ConfigDict(create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type))),
    )
    graph = onnx.load(path).graph
    assert [node.op_type for node in graph.node] == ["FlashAttention"]
    attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in graph.node[0].attribute}
    assert attributes["num_heads"] == attributes["num_kv_heads"] == 1
    assert all(attributes[name + "_bits"] == bits for name in ("q", "k", "v", "s", "p"))
    runtime = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
    q, k, v = [value.cuda().half() for value in values]
    length = length.cuda()
    actual = runtime(q, k, v, length).clone()
    reference = flash_attention_xh2(
        q,
        k,
        v,
        is_causal=False,
        current_input_length=length,
        num_heads=1,
        num_kv_heads=1,
        head_size=head_dim,
        **{name + "_manbit": bits - 1 for name in ("q", "k", "v", "s", "p")},
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("head_dim", [768, 1152])
@pytest.mark.parametrize("mantissas", [(7, 7, 7, 7, 7), (15, 15, 15, 15, 15), (15, 7, 15, 7, 15)])
def test_vae_flash_large_single_head_matches_hmfp_reference(head_dim, mantissas):
    from xhquant.backend.hmops.fused_ops.flash_attention.xh2._flash_attention_xh2_torch import (
        flash_attention_xh2_torch_flash,
    )
    from xhquant.backend.hmops.fused_ops.flash_attention.xh2.flash_attention_xh2 import flash_attention_xh2

    assert torch.cuda.is_available(), "CUDA is required to guard the deployed VAE large-head kernel"
    torch.manual_seed(7)
    q, k, v = [torch.randn(1, 1, 72, head_dim, device="cuda", dtype=torch.float16) for _ in range(3)]
    length = torch.tensor([69], dtype=torch.int32, device="cuda")
    bits = dict(zip(("q_manbit", "k_manbit", "v_manbit", "s_manbit", "p_manbit"), mantissas, strict=True))
    actual = flash_attention_xh2(
        q, k, v, is_causal=False, current_input_length=length, num_heads=1, num_kv_heads=1, head_size=head_dim, **bits
    )
    reference = flash_attention_xh2_torch_flash(
        q, k, v, length, length, False, None, head_dim**-0.5, 1, None, None, use_hmfp=True, **bits
    )
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, :, 69:]) == 0
    k[:, :, 69:] = 1000
    v[:, :, 69:] = -1000
    poisoned = flash_attention_xh2(
        q, k, v, is_causal=False, current_input_length=length, num_heads=1, num_kv_heads=1, head_size=head_dim, **bits
    )
    torch.testing.assert_close(poisoned, actual, atol=0, rtol=0)
