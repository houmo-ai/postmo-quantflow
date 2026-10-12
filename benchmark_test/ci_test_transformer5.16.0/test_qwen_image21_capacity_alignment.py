"""Real W8A8 exports preserve effective outputs when causal Q capacity grows."""

from copy import deepcopy

import onnx
import pytest
import torch
from torch import nn
from torch.nn import functional as F


@pytest.mark.parametrize("role,capacities", [("feature", (1664, 1792)), ("dit_text", (576, 768))])
def test_w8a8_capacity_alignment_preserves_outputs_and_cache(tmp_path, role, capacities):
    assert torch.cuda.is_available(), "W8A8 golden comparison requires a visible CUDA GPU"
    from xhmodel_merak.xh_other_model.models.qwen_image21.export import make_text_projection
    from xhmodel_merak.xh_other_model.models.qwen_image21.feature_graph import (
        CAPACITY_FEATURE_INPUT_NAMES,
        CapacityFeatureSession,
        CapacityTextFeatures,
    )
    from xhmodel_merak.xh_other_model.models.qwen_image21.full_graph import SegmentDenoiser
    from xhquant.api import (
        ConfigDict,
        DeviceType,
        QuantScheme,
        create_quant_config,
        convert_fx_model_to_quanted_model,
        convert_quanted_model_to_hmonnx,
    )
    from xhquant.core.cache_tensor import CacheTensor
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.manual_seed(47)
    length, past, cache_capacity = 129, 37, 1024
    if role == "feature":
        from test_qwen_image21_feature_encoder import tiny_config
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        config = tiny_config()
        config.text_config.hidden_size = 512
        config.text_config.num_attention_heads = 4
        config.text_config.num_key_value_heads = 1
        config.text_config.num_hidden_layers = 1
        source = Qwen3VLForConditionalGeneration(config).eval().model.language_model
        tokens = torch.randn(1, length, 512) * 0.2
        deepstack = [torch.randn_like(tokens) * 0.03 for _ in range(3)]
        positions = torch.arange(length, dtype=torch.int32).expand(3, -1).clone()
        names = CAPACITY_FEATURE_INPUT_NAMES
    else:
        from test_qwen_image21_full_graph import source as make_source

        source = make_source()
        source.txt_in = nn.Module()
        source.txt_in.text_norm = nn.RMSNorm(96, eps=1e-6)
        source.txt_in.in_layer = nn.Linear(96, 128)
        source.txt_in.act = nn.GELU(approximate="tanh")
        source.txt_in.out_layer = nn.Linear(128, 128)
        tokens = torch.randn(1, length, 96) * 0.2
        angles = torch.arange(length * 32).reshape(1, 1, length, 32).float() / 113
        initial_caches = [torch.zeros(1, 2, cache_capacity, 64) for _ in range(4)]
        for cache in initial_caches:
            cache[:, :, :past] = torch.randn_like(cache[:, :, :past]) * 0.2
        names = ["tokens", "time_features", "cos", "sin", "past_seq_length", "current_input_length"]
        names += [f"cache_{index}" for index in range(4)]

    def to_gpu(value):
        if isinstance(value, CacheTensor):
            return CacheTensor(value.data.cuda().half().clone())
        if isinstance(value, tuple):
            return tuple(to_gpu(item) for item in value)
        return value.cuda().half() if value.is_floating_point() else value.cuda()

    outputs, caches = [], []
    for capacity in capacities:
        if role == "feature":
            module = CapacityTextFeatures(source, capacity, 2048).eval()
            boundary = CapacityFeatureSession(None, capacity, 512, 2048, "cpu")
            inputs = boundary.prepare_inputs(tokens, positions, deepstack)
            inputs = tuple(value.float() if value.is_floating_point() else value for value in inputs)
            assert inputs[4].item() == length
        else:
            current_source = deepcopy(source)
            module = SegmentDenoiser(
                current_source, capacity, text_projection=make_text_projection(current_source)
            ).eval()
            inputs = (
                F.pad(tokens, (0, 0, 0, capacity - length)),
                torch.randn(1, 256, generator=torch.Generator().manual_seed(53)),
                F.pad(angles.cos(), (0, 0, 0, capacity - length), value=1),
                F.pad(angles.sin(), (0, 0, 0, capacity - length)),
                torch.tensor([past], dtype=torch.int32),
                torch.tensor([length], dtype=torch.int32),
                tuple(CacheTensor(value.clone()) for value in initial_caches),
            )
        quant = ConfigDict(create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type="w8a8_sefp")))
        quanted = convert_fx_model_to_quanted_model(module, inputs, DeviceType.XH2a, quant)
        path = tmp_path / f"{role}_{capacity}.onnx"
        convert_quanted_model_to_hmonnx(quanted, inputs, str(path), names, ["hidden"])
        graph = onnx.load(path).graph
        values = {value.name: value for value in (*graph.input, *graph.value_info, *graph.output)}
        attentions = [node for node in graph.node if node.op_type == "FlashAttention"]
        assert len(attentions) == (1 if role == "feature" else 2)
        for node in attentions:
            assert values[node.input[0]].type.tensor_type.shape.dim[2].dim_value == capacity
            assert node.input[5] == ("input_length" if role == "feature" else "current_input_length")
            assert values[node.input[5]].type.tensor_type.elem_type == onnx.TensorProto.INT32
            if role == "feature":
                past_value = next(value for value in graph.initializer if value.name == node.input[4])
                assert past_value.data_type == onnx.TensorProto.INT32
                assert onnx.numpy_helper.to_array(past_value).tolist() == [0]
            else:
                assert node.input[4] == "past_seq_length"
                assert values[node.input[4]].type.tensor_type.elem_type == onnx.TensorProto.INT32
            attrs = {value.name: onnx.helper.get_attribute_value(value) for value in node.attribute}
            assert attrs["is_causal"] == 1
            assert "s_bits" not in attrs
            assert all(attrs[name] == 8 for name in ("q_bits", "k_bits", "v_bits", "p_bits"))
        for node in graph.node:
            if node.op_type == "Linear":
                attrs = {value.name: onnx.helper.get_attribute_value(value) for value in node.attribute}
                assert attrs["hmfp_weight_man_bit"] == attrs["hmfp_act_man_bit"] == 8
        runtime = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
        if role == "dit_text":
            assert runtime.enable_inplace_cache_updates() == 4
        quanted.cuda().half()
        golden_inputs, runtime_inputs = to_gpu(inputs), to_gpu(inputs)

        def run(values):
            ports = values if role == "feature" else (*values[:6], *values[6])
            return runtime(*ports).clone()

        with torch.inference_mode():
            golden_ports = golden_inputs if role == "feature" else (*golden_inputs[:6], *golden_inputs[6])
            expected = quanted(*golden_ports).clone()
            actual = run(runtime_inputs)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert torch.isfinite(actual).all()
            outputs.append(actual[:, :length].clone())
            if role == "feature":
                assert torch.count_nonzero(actual[:, length:]) == 0
            else:
                caches.append([value.data.clone() for value in runtime_inputs[6]])
                for value, initial in zip(runtime_inputs[6], initial_caches, strict=True):
                    torch.testing.assert_close(
                        value.data[:, :, :past], initial[:, :, :past].cuda().half(), atol=0, rtol=0
                    )
                    assert torch.count_nonzero(value.data[:, :, past + length :]) == 0
            poisoned = to_gpu(inputs)
            poisoned[0][:, length:] = 1000
            if role == "feature":
                for value in poisoned[5:]:
                    value[:, length:] = -1000
            torch.testing.assert_close(run(poisoned)[:, :length], actual[:, :length], atol=0, rtol=0)
            # The same exported causal graph must also honor a full-capacity request.
            full_inputs, full_golden = to_gpu(inputs), to_gpu(inputs)
            length_index = 4 if role == "feature" else 5
            full_inputs[length_index].fill_(capacity)
            full_golden[length_index].fill_(capacity)
            full_ports = full_golden if role == "feature" else (*full_golden[:6], *full_golden[6])
            full_expected = quanted(*full_ports).clone()
            full_actual = run(full_inputs)
            assert full_actual.shape[1] == capacity and torch.isfinite(full_actual).all()
            torch.testing.assert_close(full_actual, full_expected, atol=0, rtol=0)
        del quanted, runtime, golden_inputs, runtime_inputs, poisoned
    torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)
    if role == "dit_text":
        for after, before in zip(caches[1], caches[0], strict=True):
            torch.testing.assert_close(after, before, atol=0, rtol=0)
