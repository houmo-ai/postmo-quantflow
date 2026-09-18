"""Real MTP packed experts/shared head -> Merak forward and HMONNX export."""
import json
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture(params=[("autoround", 4), ("gptq", 4), ("autoround", 5), ("gptq", 5)])
def checkpoint(tmp_path, request):
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from gptqmodel.quantization.qwen35_mtp import MTPQuantizationConfig, Qwen35MTP

    torch.set_num_threads(2)
    algorithm, head_bits = request.param
    config = {"text_config": {"model_type": "qwen3_5_moe_text", "hidden_size": 128,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
        "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64, "vocab_size": 256, "mtp_num_hidden_layers": 1,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
        "tie_word_embeddings": False}, "tie_word_embeddings": False,
        "quantization_config": {"quant_method": "gptq", "checkpoint_format": "gptq_v2",
            "bits": 4, "group_size": 64, "sym": True, "desc_act": False, "lm_head": True,
            "dynamic": {"+:^lm_head$": {"bits": head_bits}}},
        "mtp_quantization": {**asdict(MTPQuantizationConfig(algorithm=algorithm)), "version": 1, "format": "gptq_v2",
                             "scope": "routed_experts", "other_weights": "float"}}
    if algorithm == "autoround":
        config["quantization_config"]["provider"] = "auto-round"
    model = Qwen35MTP(config).half()
    generator = torch.Generator().manual_seed(14)
    with torch.no_grad():
        for name, weight in model.named_parameters():
            weight.copy_(torch.randn(weight.shape, generator=generator) * 0.02)
            if "norm" in name:
                weight.zero_()
    state = {"mtp." + k: v.contiguous() for k, v in model.state_dict().items() if ".mlp.experts." not in k}
    codes = {}

    def pack(prefix, ins, outs, bits=4):
        midpoint = 1 << (bits - 1)
        signed = torch.randint(-midpoint, midpoint, (outs, ins), generator=generator).to(torch.int8)
        linear = torch.nn.Linear(ins, outs, bias=False)
        linear.weight.data.copy_(signed.float() / 512)
        module = TorchQuantLinear(bits=bits, group_size=64, sym=True, desc_act=False,
                                 in_features=ins, out_features=outs)
        scale = torch.full((outs, ins // 64), 1 / 512)
        module.pack_original(
            linear, scale, torch.full_like(scale, midpoint), torch.arange(ins, dtype=torch.int32) // 64
        )
        if bits == 5:
            assert module.bits == 8 and module.logical_bits == 5
        state.update({prefix + "." + k: v for k, v in module.state_dict().items()})
        codes[prefix] = signed

    for expert in range(4):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            pack(f"mtp.layers.0.mlp.experts.{expert}.{proj}", 64 if proj == "down_proj" else 128,
                 128 if proj == "down_proj" else 64)
    pack("lm_head", 128, 256, bits=head_bits)
    # Deliberately different: an embedding fallback would fail the assertions.
    state["model.language_model.embed_tokens.weight"] = torch.ones(256, 128, dtype=torch.float16)
    root = tmp_path / "model"
    root.mkdir()
    save_file(state, root / "model.safetensors")
    if algorithm == "gptq":
        (root / "quantize_config.json").write_text(json.dumps(config.pop("quantization_config")))
    (root / "config.json").write_text(json.dumps(config))
    return root, codes, head_bits


@pytest.mark.parametrize("length", [1, 4])
def test_packed_mtp_forward_and_export(checkpoint, length, tmp_path):
    import onnx
    from gptqmodel.quantization.qwen35_mtp import Qwen35MTP

    from xhmodel_merak.xh_llm.models.qwen3_5._mtp_model_impl import MTPModel
    from xhmodel_merak.xh_llm.models.qwen3_5.mtp_checkpoint import configure_packed_mtp_scheme
    from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import build_spec_draft_quant_scheme
    from xhquant.api import (
        PrecisionMode,
        convert_quanted_model_to_hmonnx,
        ptq_quantize,
        to_frontend_graph,
        to_quant_graph,
    )

    root, codes, head_bits = checkpoint
    native = Qwen35MTP.from_pretrained(root)
    wrapped = MTPModel.from_pretrained(str(root), input_sequence_length=length, max_pe_length=32, use_cache=True)
    assert wrapped._mtp_expert_bits == 4 and wrapped._mtp_head_bits == head_bits
    assert torch.equal(wrapped.lm_head.quant_weight, codes["lm_head"])
    expected = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        expected[proj] = torch.stack([codes[f"mtp.layers.0.mlp.experts.{i}.{proj}"] for i in range(4)])
        assert torch.equal(getattr(wrapped.layer.mlp.moeblock, f"expert_{proj}_quant_weight"), expected[proj])
    generator = torch.Generator().manual_seed(8)
    inputs = [torch.randn(1, length, 128, generator=generator).half() for _ in range(2)]
    inputs.extend([torch.tensor([0]), torch.tensor([length])])
    inputs.extend([torch.zeros(1, 2, 32, 32, dtype=torch.float16) for _ in range(2)])
    from xhquant.core.cache_tensor import CacheTensor
    with torch.no_grad():
        logits, hidden = wrapped(*inputs[:4], *[CacheTensor(x.clone()) for x in inputs[4:]])
        reference = native(*inputs[:2])
    torch.testing.assert_close(hidden, reference, rtol=0.015, atol=0.006)
    torch.testing.assert_close(logits, torch.nn.functional.linear(hidden, codes["lm_head"].half() / 512))
    scheme = configure_packed_mtp_scheme(build_spec_draft_quant_scheme(head_bits), wrapped, explicit=False)
    front = to_frontend_graph(wrapped.eval(), "TorchFX", inputs)
    quant = to_quant_graph(front, "XH2a", scheme)
    ptq_quantize(quant, [inputs], PrecisionMode.ALIGNED, ["cpu"],
                 auto_release_unused_parameters=True, infer_shape=False)
    moe = next(m for m in quant.modules() if hasattr(m, "expert_gate_proj_qweight"))
    assert moe.w_cfg.qspec.man_bit == 4
    for proj, value in expected.items():
        actual = getattr(moe, f"expert_{proj}_qweight").reshape(4, value.shape[2], value.shape[1]).transpose(1, 2)
        assert torch.equal(actual, value)
    assert quant.lm_head.w_cfg.qspec.man_bit == head_bits
    assert quant.fc.w_cfg.qspec.man_bit == 8
    attention = next(m for n, m in quant.named_modules() if n.endswith("q_proj"))
    assert attention.w_cfg.qspec.man_bit == 8
    path = tmp_path / "mtp.onnx"
    convert_quanted_model_to_hmonnx(quant, inputs, str(path))
    graph = onnx.load(path)
    for node in graph.graph.node:
        if node.op_type == "MoeBlock" or node.name == "lm_head":
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            assert attrs["hmfp_weight_man_bit"] == (head_bits if node.name == "lm_head" else 4)
    assert any(n.op_type == "MoeBlock" for n in graph.graph.node)


def test_mtp_scheme_rejects_conflicting_override(checkpoint):
    from types import SimpleNamespace

    from xhmodel_merak.xh_llm.models.qwen3_5.mtp_checkpoint import configure_packed_mtp_scheme
    with pytest.raises(ValueError, match="requires"):
        configure_packed_mtp_scheme({"quant_type": "w8a8h1_sefp", "nodes": {
            "layer_mlp_moeblock": {"quant_type": "w8a8h1_sefp"}}},
            SimpleNamespace(_mtp_expert_bits=4), explicit=True)


def test_public_draft_wrapper_accepts_serialized_default_scheme(checkpoint):
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_mtp_model import XHQwen3_5MTPDraftModel
    from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import (
        XHQwen3_5_MTPConfig,
        build_spec_draft_quant_scheme,
    )
    root, _, head_bits = checkpoint
    cfg = XHQwen3_5_MTPConfig(model_name="tiny_mtp", hf_model=str(root), chip_arch="XH2a", hidden_size=128,
        num_key_value_heads=2, head_dim=32, context_max_length=32, max_pe_length=32,
        draft_head_weight_bits=head_bits, quant_scheme=build_spec_draft_quant_scheme(head_bits))
    draft = XHQwen3_5MTPDraftModel(cfg)
    draft.to_wrap()
    assert draft.config.quant_scheme["nodes"]["lm_head"]["quant_type"] == f"w{head_bits}a8h1_sefp"
    assert draft.config.quant_scheme["nodes"]["layer.mlp.moeblock"]["quant_type"] == "w4a8h1_sefp"


def test_adapter_passes_mtp_without_changing_legacy_defaults(tmp_path):
    from pathlib import Path

    import yaml

    from xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter import build_qwen35_autoround_kwargs

    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    common = dict(hf_model_dir=str(tmp_path), output_dir=str(tmp_path / "out"), device="cpu",
                  quant_cfg=cfg["quant"], export_model_cfg=cfg["export"]["model"], workflow_seed=42)
    assert "mtp" not in build_qwen35_autoround_kwargs(**common)
    cfg["quant"]["mtp"] = {"expert_bits": 4, "iters": 400}
    assert build_qwen35_autoround_kwargs(**common)["mtp"] == cfg["quant"]["mtp"]
