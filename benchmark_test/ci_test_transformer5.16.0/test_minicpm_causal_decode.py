"""Causal V quantization in real MiniCPM decode/verify export graphs."""

import pytest
import torch

from xhmodel_merak.xh_llm.models.minicpm_v_4_7.causal_values import (
    _CausalValueProduct,
    enable_minicpm_causal_values,
)


@pytest.mark.parametrize("moe", (False, True))
def test_minicpm_target_enables_causal_value_quantization(moe):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.text_model import (
        MiniCPMV47MoeTextModel, MiniCPMV47TextModel,
    )
    from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5ModelConfig

    cls = MiniCPMV47MoeTextModel if moe else MiniCPMV47TextModel
    model = cls(XHQwen3_5ModelConfig(model_name="minicpm", visual_config=None))
    assert model.wrap_cfg.mask_causal_values is True


@pytest.mark.parametrize("mode,enabled", (("chunk", True), ("recurrent", True), ("recurrent", False)))
def test_real_attention_wrapper_limits_causal_value_split_to_decode(mode, enabled):
    import copy

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
    from xhmodel_merak.xh_llm.models.qwen3_5._llm_model_impl import register_wrap_modules
    from xhmodel_merak.xh_llm.wrap_model import wrap_llm_model
    from xhquant.api import ConfigDict

    register_wrap_modules()
    native = Qwen3_5Attention(Qwen3_5TextConfig(
        hidden_size=128, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
    ), layer_idx=0).eval()
    cfg = ConfigDict(dict(
        use_cache=True, kv_cache=dict(cache_axis=2), context_max_length=128,
        input_sequence_length=4, linear_attention_mode=mode, mask_causal_values=enabled,
    ))
    wrapped = wrap_llm_model(native, cfg)
    original_class = type(wrapped)
    enable_minicpm_causal_values(wrapped)
    wrapped._update_cfg(cfg)
    product = wrapped.causal_value_product
    if enabled and mode == "recurrent":
        assert isinstance(product, _CausalValueProduct)
        assert product.query_capacity == 4 and product.kv_groups == 2
    else:
        assert product is None
    # Export starts in prefill, then copies and updates this same wrapper for verify.
    cfg.mask_causal_values = True
    for next_mode, length in (("recurrent", 1), ("recurrent", 4), ("chunk", 64)):
        cfg.linear_attention_mode = next_mode
        cfg.input_sequence_length = length
        wrapped._update_cfg(cfg)
        if next_mode == "chunk":
            assert wrapped.causal_value_product is None
        else:
            assert wrapped.causal_value_product.query_capacity == length
    copied = copy.deepcopy(wrapped)
    cfg.linear_attention_mode = "recurrent"
    cfg.input_sequence_length = 3
    copied._update_cfg(cfg)
    assert copied.causal_value_product.query_capacity == 3
    assert wrapped.causal_value_product is None
    assert not hasattr(original_class, "causal_value_product")


@pytest.mark.parametrize("bits", (8, 16))
def test_hmonnx_verify_ignores_future_and_rejected_v_cache(bits, tmp_path):
    """Preserve GQA, the 64-token scale boundary, and poisoned future cache."""
    import onnx
    from xhquant.api import (
        PrecisionMode, convert_quanted_model_to_hmonnx, ptq_quantize,
        to_frontend_graph, to_quant_graph,
    )
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2
    from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader

    torch.set_num_threads(2)
    torch.manual_seed(47)
    values = torch.randn(1, 1, 128, 64).half() * 0.1
    probabilities = torch.randn(1, 2, 4, 128).float()
    scheme = {"quant_type": "w8a8h1_sefp", "ops": {"MatMul": {
        "act_scheme": {"bits": bits, "fp_mode": "sefp"},
        "act_schema_2": {"bits": bits, "fp_mode": "sefp"},
    }}}
    sessions = {}
    for length in (1, 4):
        module = _CausalValueProduct(128, length, 2, device="cpu").eval()
        inputs = [probabilities[:, :, :length].half(), values, torch.tensor([61], dtype=torch.int32)]
        quantized = to_quant_graph(to_frontend_graph(module, "TorchFX", inputs), "XH2a", scheme)
        ptq_quantize(quantized, [inputs], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
        path = tmp_path / f"decode_{length}.onnx"
        convert_quanted_model_to_hmonnx(quantized, inputs, str(path))
        graph = onnx.load(path)
        assert sum(node.op_type == "Where" for node in graph.graph.node) == length
        assert sum(node.op_type in ("MatMul", "GroupMatMul") for node in graph.graph.node) == length
        loader = MultiHMONNXLoader({"graph": str(path)})
        sessions[length] = HMONNXInferenceV2.from_onnx_graph(
            str(path), loader.graphs["graph"], HMONNXInferenceConfig(exec_devices=["cpu"])
        )

    with torch.inference_mode():
        for past in (61, 65):
            logits = probabilities.clone()
            for index in range(4):
                logits[:, :, index, past + index + 1:] = float("-inf")
            weights = logits.softmax(-1).half()
            baseline = sessions[4](weights, values, torch.tensor([past], dtype=torch.int32))
            for index in range(4):
                poisoned = values.clone()
                # Includes other proposals and stale entries from an earlier rejected block.
                poisoned[:, :, past + index + 1:, :] = 512
                ordinary = sessions[1](weights[:, :, index:index + 1], poisoned,
                                       torch.tensor([past + index], dtype=torch.int32))
                torch.testing.assert_close(baseline[:, :, index:index + 1], ordinary, rtol=0, atol=0)
            poisoned = values.clone()
            poisoned[:, :, past + 1:, :] = -512
            verify = sessions[4](weights, poisoned, torch.tensor([past], dtype=torch.int32))
            torch.testing.assert_close(verify[:, :, :1], baseline[:, :, :1], rtol=0, atol=0)
