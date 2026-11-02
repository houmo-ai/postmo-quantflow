"""Effective-prefix attention semantics and real W8A8 export/golden agreement."""

import pytest
import torch

from xhmodel_merak.xh_other_model.models.qwen_image21.length_attention import LengthMaskedAttention


@pytest.mark.parametrize("head_dim,physical_dim", [(72, 128), (128, 128), (768, 768), (1152, 1152)])
@pytest.mark.parametrize("query_capacity,kv_capacity,query_length,kv_length", [
    (64, 512, 1, 1), (64, 512, 17, 97), (64, 512, 64, 512),
    (320, 768, 273, 521),
    # Empty effective prefixes are operator boundaries, not public empty-image support.
    (64, 512, 0, 0), (64, 512, 0, 97), (64, 512, 17, 0),
])
@pytest.mark.parametrize("native_attention", [False, True])
def test_length_attention_matches_sliced_reference_and_ignores_padding(
    head_dim, physical_dim, query_capacity, kv_capacity, query_length, kv_length, native_attention
):
    torch.manual_seed(42)
    q = torch.randn(1, 1, query_capacity, head_dim)
    k, v = [torch.randn(1, 1, kv_capacity, head_dim) for _ in range(2)]
    q, k, v = [torch.nn.functional.pad(x, (0, physical_dim - head_dim)) for x in (q, k, v)]
    lengths = [torch.tensor([n], dtype=torch.int32) for n in (query_length, kv_length)]
    module = LengthMaskedAttention(
        query_capacity, kv_capacity, scale=head_dim**-0.5, use_flash_attention=native_attention
    )
    actual = module(q, k, v, *lengths)
    expected = torch.zeros_like(actual)
    scores = q[:, :, :query_length] @ k[:, :, :kv_length].transpose(-1, -2) * head_dim**-0.5
    expected[:, :, :query_length] = scores.softmax(-1) @ v[:, :, :kv_length]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, :, query_length:]) == 0
    q[:, :, query_length:] = 1000
    k[:, :, kv_length:] = 1000
    v[:, :, kv_length:] = -1000
    torch.testing.assert_close(module(q, k, v, *lengths), actual, atol=0, rtol=0)


@pytest.mark.parametrize("quant_type", ["w16a16_sefp", "w8a16_sefp", "w8a8_sefp"])
@pytest.mark.parametrize("native_attention", [False, True])
@pytest.mark.parametrize("head_dim", [128, 768, 1152])
def test_large_qk_export_preserves_score_differences(tmp_path, quant_type, native_attention, head_dim):
    """Independent reference catches saturation shared by quanted/runtime golden."""
    import onnx

    from xhquant.api import (
        ConfigDict,
        DeviceType,
        QuantScheme,
        convert_fx_model_to_quanted_model,
        convert_quanted_model_to_hmonnx,
        create_quant_config,
    )
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    # Exercise the compiler FlashAttention boundary once; the other existing
    # precision/head-width cases retain their small, short-KV exports.
    capacity = 512 if native_attention and head_dim == 128 and quant_type == "w16a16_sefp" else 64
    q = torch.full((1, 1, capacity, head_dim), 24.0)
    k, v = torch.zeros_like(q), torch.zeros_like(q)
    for index, key in enumerate((24, 27, 30)):
        k[:, :, index] = key
        v[:, :, index] = index
    lengths = (torch.tensor([1], dtype=torch.int32), torch.tensor([3], dtype=torch.int32))
    inputs = (q, k, v, *lengths)
    reference = ((q.double()[:, :, :1] @ k.double()[:, :, :3].transpose(-1, -2)) * head_dim**-0.5).softmax(-1)
    reference = (reference @ v.double()[:, :, :3]).half()
    module = LengthMaskedAttention(
        capacity, capacity, scale=head_dim**-0.5, use_flash_attention=native_attention
    ).eval()
    config = ConfigDict(create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)))
    quanted = convert_fx_model_to_quanted_model(module, inputs, DeviceType.XH2a, config)
    path = tmp_path / "attention.onnx"
    convert_quanted_model_to_hmonnx(
        quanted, inputs, str(path), ["q", "k", "v", "query_length", "kv_length"], ["output"]
    )
    graph = onnx.load(path).graph
    producers = {output: node for node in graph.node for output in node.output}
    flash = [node for node in graph.node if node.op_type == "FlashAttention"]
    assert len(flash) == int(module.use_flash_attention)
    if module.use_flash_attention:
        assert len(flash[0].input) == 5 and flash[0].input[4]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in flash[0].attribute}
        assert attrs["is_causal"] == 0
        assert attrs["scale"] == pytest.approx(head_dim**-0.5)
    else:
        qk = next(node for node in graph.node if node.op_type == "MatMul")
        # QK must consume a scalar-scaled Q, rather than scaling its FP16 output.
        mul = producers[qk.input[0]]
        assert mul.op_type == "Mul"
        constants = {entry.name: onnx.numpy_helper.to_array(entry) for entry in graph.initializer}
        assert any(name in constants and constants[name].size == 1
                   and float(constants[name].item()) == pytest.approx(head_dim**-0.5, rel=1e-3)
                   for name in mul.input)
    runtime = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cpu"]))
    payload = [value.half() if value.is_floating_point() else value for value in inputs]
    with torch.inference_mode():
        output = runtime(*payload).clone()
        torch.testing.assert_close(output[:, :, :1], reference, atol=0, rtol=0)
        assert torch.count_nonzero(output[:, :, 1:]) == 0
        payload[0][:, :, 1:] = 1000
        payload[1][:, :, 3:] = 1000
        payload[2][:, :, 3:] = -1000
        torch.testing.assert_close(runtime(*payload), output, atol=0, rtol=0)


@pytest.mark.parametrize("head_dim", [128, 768, 1152])
def test_w8a8_length_attention_export_matches_quantized_golden(tmp_path, head_dim):
    assert torch.cuda.is_available(), "W8A8 golden comparison requires a visible CUDA GPU"
    import onnx

    from xhquant.api import (
        ConfigDict,
        DeviceType,
        QuantScheme,
        convert_fx_model_to_quanted_model,
        convert_quanted_model_to_hmonnx,
        create_quant_config,
    )
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.manual_seed(9)
    query_capacity, kv_capacity = 320, 768
    values = [torch.randn(1, 1, n, head_dim) for n in (query_capacity, kv_capacity, kv_capacity)]
    lengths = [torch.tensor([n], dtype=torch.int32) for n in (17, 97)]
    config = ConfigDict(create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type="w8a8_sefp")))
    module = LengthMaskedAttention(query_capacity, kv_capacity, scale=head_dim**-0.5).eval()
    quanted = convert_fx_model_to_quanted_model(module, (*values, *lengths), DeviceType.XH2a, config)
    path = tmp_path / "attention.onnx"
    convert_quanted_model_to_hmonnx(
        quanted, (*values, *lengths), str(path), ["q", "k", "v", "query_length", "kv_length"], ["output"]
    )
    graph = onnx.load(path).graph
    assert not any(node.op_type == "FlashAttention" for node in graph.node)
    assert sum(node.op_type == "MatMul" for node in graph.node) == 2
    assert sum(node.op_type == "Softmax" for node in graph.node) == 1
    for node in graph.node:
        if node.op_type == "MatMul":
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            assert attrs["hmfp_act_man_bit"] == attrs["hmfp_weight_man_bit"] == 8
    runtime = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
    quanted.cuda().half()
    gpu_values = [x.cuda().half() for x in values]
    # Reuse one export per head width while varying the runtime lengths.
    cases = [(17, 97), (0, 0), (0, 97), (17, 0), (query_capacity, kv_capacity), (256, 512), (273, 521)]
    with torch.inference_mode():
        for query_length, kv_length in cases:
            current_lengths = [torch.tensor([n], dtype=torch.int32, device="cuda:0") for n in (query_length, kv_length)]
            inputs = [*gpu_values, *current_lengths]
            expected = quanted(*inputs).clone()
            actual = runtime(*inputs).clone()
            context = f"head_dim={head_dim}, query_length={query_length}, kv_length={kv_length}"
            torch.testing.assert_close(actual, expected, atol=0, rtol=0, msg=context)
            assert torch.isfinite(actual).all(), context
            assert torch.count_nonzero(actual[:, :, query_length:]) == 0, context
            if kv_length == 0:
                assert torch.count_nonzero(actual) == 0, context
            poisoned = [value.clone() for value in gpu_values]
            poisoned[0][:, :, query_length:] = 1000
            poisoned[1][:, :, kv_length:] = 1000
            poisoned[2][:, :, kv_length:] = -1000
            torch.testing.assert_close(
                runtime(*poisoned, *current_lengths).clone(), actual, atol=0, rtol=0, msg=context
            )
