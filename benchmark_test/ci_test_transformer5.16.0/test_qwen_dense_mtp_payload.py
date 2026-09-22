"""Consume real high-order dense MTP payloads through the deployment frontend."""
import json
import pytest
import torch
from safetensors.torch import save_file


@pytest.mark.parametrize("bits", [4, 5, 8])
def test_dense_mtp_payload_forward_and_graph(tmp_path, bits):
    from transformers import Qwen3_5TextConfig
    from gptqmodel.quantization.qwen35_mtp import Qwen35MTP, MTPQuantizationConfig, calibrate_mtp_experts, save_mtp_checkpoint
    from xhmodel_merak.xh_llm.models.qwen3_5._mtp_model_impl import MTPModel
    from xhmodel_merak.xh_llm.models.qwen3_5.mtp_checkpoint import configure_packed_mtp_scheme
    from xhquant.api import to_frontend_graph, to_quant_graph, ptq_quantize, PrecisionMode
    torch.set_num_threads(2)
    torch.manual_seed(21)
    config = {"model_type": "qwen3_5", "text_config": Qwen3_5TextConfig(
        hidden_size=128, intermediate_size=256, vocab_size=256, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, layer_types=["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": 0.5}).to_dict()}
    source = tmp_path / "source"
    source.mkdir()
    model = Qwen35MTP(config).half().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.zeros_like(p) if "norm" in name else torch.randn_like(p) * 0.03)
    state = {"mtp." + k: v.contiguous() for k, v in model.state_dict().items()}
    state["lm_head.weight"] = torch.randn(256, 128).half() * 0.03
    save_file(state, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps(config))
    model = Qwen35MTP.from_pretrained(source)
    records = [{"embedding": torch.randn(1, 16, 128).half(), "hidden": torch.randn(1, 16, 128).half()}]
    options = MTPQuantizationConfig(block_bits=bits, algorithm="autoround", iters=2)
    payload, _ = calibrate_mtp_experts(model, records, options, device="cpu")
    target = tmp_path / "packed"
    save_mtp_checkpoint(source, target, payload, options, {})
    wrapped = MTPModel.from_pretrained(str(target), max_pe_length=32, input_sequence_length=4, use_cache=True)
    inputs = [records[0]["embedding"][:, :4], records[0]["hidden"][:, :4], torch.tensor([0], dtype=torch.int32),
              torch.tensor([4], dtype=torch.int32), torch.zeros(1, 2, 32, 32).half(), torch.zeros(1, 2, 32, 32).half()]
    from xhquant.core.cache_tensor import CacheTensor
    with torch.no_grad():
        hidden = wrapped(*inputs[:4], *[CacheTensor(x.clone()) for x in inputs[4:]])[1]
        expected = model(*inputs[:2])
    torch.testing.assert_close(hidden, expected, atol=0.006, rtol=0.008)
    assert wrapped._mtp_block_bits == bits
    scheme = configure_packed_mtp_scheme({"quant_type": "w8a8h1_sefp"}, wrapped, explicit=False)
    graph = to_quant_graph(to_frontend_graph(wrapped, "TorchFX", inputs), "XH2a", scheme)
    ptq_quantize(graph, [inputs], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    linears = [m for name, m in graph.named_modules() if hasattr(m, "qweight") and name.startswith(("layer_", "layer."))]
    assert len(linears) == 7
    assert all(m.w_cfg.qspec.man_bit == bits for m in linears)
    assert graph.fc.w_cfg.qspec.man_bit == 8
    assert graph.lm_head.w_cfg.qspec.man_bit == 8
