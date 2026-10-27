"""Export target FA8 and explicitly masked non-causal DSpark attention."""

import pytest
import torch


def _attention(causal):
    from xhquant.api import ConfigDict

    precision = dict(enable=True, q_bits=8, k_bits=8, v_bits=8, p_bits=8)
    if not causal:
        from xhmodel_merak.xh_llm.models.qwen3_5._dflash_model_impl import DFlashCrossAttention

        wrapped = DFlashCrossAttention(
            hidden_size=128, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
            rms_norm_eps=1e-6, input_sequence_length=15, max_pe_length=128,
            rope_theta=10000, use_cache=True, flash_attention=precision,
        )
    else:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
        from xhmodel_merak.xh_llm.models.qwen3_5._llm_model_impl import register_wrap_modules
        from xhmodel_merak.xh_llm.wrap_model import wrap_llm_model

        register_wrap_modules()
        native = Qwen3_5Attention(Qwen3_5TextConfig(
            hidden_size=128, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        ), layer_idx=0).eval()
        cfg = ConfigDict(dict(
            use_cache=True, kv_cache=dict(cache_axis=2), context_max_length=128,
            input_sequence_length=4, linear_attention_mode="recurrent",
            mask_causal_values=True, flash_attention=precision,
        ))
        wrapped = wrap_llm_model(native, cfg)
        from xhmodel_merak.xh_llm.models.minicpm_v_4_7.causal_values import enable_minicpm_causal_values

        enable_minicpm_causal_values(wrapped)
        wrapped._update_cfg(cfg)
        assert wrapped.causal_value_product is None
    assert wrapped.use_flash_attention
    return wrapped.flash_attn


@pytest.mark.parametrize("causal", (True, False), ids=("target", "dspark"))
def test_fa8_hmonnx_preserves_gqa_lengths_and_causal_prefix(causal, tmp_path):
    import onnx
    from xhquant.api import (PrecisionMode, convert_quanted_model_to_hmonnx,
                             ptq_quantize, to_frontend_graph, to_quant_graph)
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.set_num_threads(2)
    torch.manual_seed(47)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = _attention(causal)

        def forward(self, query, key, value, past, current):
            if causal:
                return self.attention(query, key, value, past_seq_length=past, current_input_length=current)
            return self.attention(query, key, value, current_input_length=current, kv_valid_length=past + current)

    length = 4 if causal else 15
    query = torch.randn(1, 2, length, 64).half() * .1
    key = torch.randn(1, 1, 128, 64).half() * .1
    value = torch.randn_like(key) * .1
    sessions = {}
    for size in ((1, length) if causal else (length,)):
        inputs = [query[:, :, :size], key, value,
                  torch.tensor([61], dtype=torch.int32), torch.tensor([size], dtype=torch.int32)]
        graph = to_quant_graph(to_frontend_graph(Attention(), "TorchFX", inputs),
                               "XH2a", {"quant_type": "w8a8h1_sefp"})
        ptq_quantize(graph, [inputs], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
        path = tmp_path / f"fa8_{size}.onnx"
        convert_quanted_model_to_hmonnx(graph, inputs, str(path))
        model = onnx.load(path)
        nodes = [node for node in model.graph.node if node.op_type == "FlashAttention"]
        assert len(nodes) == 1
        attributes = {a.name: onnx.helper.get_attribute_value(a) for a in nodes[0].attribute}
        assert attributes["is_causal"] == causal
        for name in ("q_bits", "k_bits", "v_bits", "p_bits"):
            assert attributes[name] == 8
        sessions[size] = HMONNXInferenceV2(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))

    with torch.inference_mode():
        for past in (0, 61, 65):
            offset = torch.tensor([past], dtype=torch.int32)
            current = torch.tensor([length], dtype=torch.int32)
            baseline = sessions[length](query, key, value, offset, current)
            assert torch.isfinite(baseline).all()
            for index in (range(length) if causal else (length - 1,)):
                visible = past + index + 1
                poisoned_key, poisoned_value = key.clone(), value.clone()
                poisoned_key[:, :, visible:] = 512
                poisoned_value[:, :, visible:] = -512
                if causal:
                    actual = sessions[1](query[:, :, index:index + 1], poisoned_key, poisoned_value,
                                         offset + index, torch.ones_like(current))
                    clean = sessions[1](query[:, :, index:index + 1], key, value,
                                        offset + index, torch.ones_like(current))
                    # Masked cache entries must have no effect. A different
                    # query capacity may introduce one FP16 rounding step in
                    # the CPU reference, including its smallest subnormal.
                    torch.testing.assert_close(actual, clean, rtol=0, atol=0)
                    fp16 = torch.finfo(torch.float16)
                    torch.testing.assert_close(clean, baseline[:, :, index:index + 1],
                                               rtol=fp16.eps, atol=fp16.tiny * fp16.eps)
                else:
                    actual = sessions[length](query, poisoned_key, poisoned_value, offset, current)
                    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)


def test_dspark_unfused_attention_exports_mask_and_excludes_unused_cache(tmp_path):
    import onnx
    from xhmodel_merak.xh_llm.models.qwen3_5._dflash_model_impl import DFlashCrossAttention
    from xhquant.api import (PrecisionMode, convert_quanted_model_to_hmonnx,
                             ptq_quantize, to_frontend_graph, to_quant_graph)
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2
    from xhquant.core.cache_tensor import CacheTensor

    torch.set_num_threads(2)
    torch.manual_seed(47)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DFlashCrossAttention(
                hidden_size=128, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                rms_norm_eps=1e-6, input_sequence_length=15, max_pe_length=128,
                rope_theta=10000, use_cache=True, flash_attention={"enable": False},
            ).half().eval()

        def forward(self, noise, past, current, key, value, mask):
            return self.attention.forward_decode(noise, past, current, key, value, mask)

    module = Attention().eval()
    noise = torch.randn(1, 15, 128).half() * .1
    key = torch.randn(1, 1, 128, 64).half() * .1
    value = torch.randn_like(key) * .1

    def inputs(past, poison=False):
        mask = torch.full((1, 128), -65504, dtype=torch.float16)
        mask[:, :past + 15] = 0
        k, v = key.clone(), value.clone()
        if poison:
            k[:, :, past + 15:] = 512
            v[:, :, past + 15:] = -512
        return [noise, torch.tensor([past], dtype=torch.int32),
                torch.tensor([15], dtype=torch.int32), k, v, mask]

    sample = inputs(61)
    graph = to_quant_graph(to_frontend_graph(module, "TorchFX", sample),
                           "XH2a", {"quant_type": "w8a8h1_sefp"})
    ptq_quantize(graph, [sample], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    path = tmp_path / "draft.onnx"
    convert_quanted_model_to_hmonnx(graph, sample, str(path))
    model = onnx.load(path)
    ops = [node.op_type for node in model.graph.node]
    assert "FlashAttention" not in ops
    assert "Softmax" in ops and sum(ops.count(op) for op in ("MatMul", "GroupMatMul")) == 2
    session = HMONNXInferenceV2(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))

    def run(values):
        values[3:5] = [CacheTensor(t) for t in values[3:5]]
        return session(*values)

    with torch.inference_mode():
        for past in (0, 61, 65):
            clean = run(inputs(past))
            poisoned = run(inputs(past, poison=True))
            assert torch.isfinite(clean).all()
            torch.testing.assert_close(poisoned, clean, rtol=0, atol=0)
        # DSpark's 15 queries see the whole block, including later mask queries.
        before = run(inputs(61))
        changed = inputs(61)
        changed[0] = noise.clone()
        changed[0][:, -1] += .5
        after = run(changed)
        assert not torch.equal(before[:, 0], after[:, 0])
