"""Expert LoRA uses the same pre-quantization path as ordinary Qwen LoRA."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import save_file, load_file

from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.frontend.torchfx import xh_fx
from xhmodel_merak.xh_llm.models.qwen3_5.lora import (
    attach_lora_buffers,
    apply_lora_to_frontend,
    inspect_lora_adapter,
)


class TinyExpertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.moeblock = MoeBlock(
            "silu",
            2,
            expert_gate_proj_weight=torch.randn(3, 128, 64) * 0.02,
            expert_up_proj_weight=torch.randn(3, 128, 64) * 0.02,
            expert_down_proj_weight=torch.randn(3, 64, 128) * 0.02,
        )

    def forward(self, x, routes):
        return self.model.layers[0].mlp.moeblock(x, routes)


def write_adapter(path, seed=1):
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 2,
                "lora_alpha": 4,
                "bias": "none",
                "target_parameters": ["experts.gate_up_proj", "experts.down_proj"],
            }
        )
    )
    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    for suffix, inputs, out in (("base_layer", 64, 256), ("", 128, 64)):
        stem = "base_model.model.model.layers.0.mlp.experts" + ("." + suffix if suffix else "")
        tensors[stem + ".lora_A.weight"] = torch.randn(6, inputs, generator=gen) * 0.1
        tensors[stem + ".lora_B.weight"] = torch.randn(out, 6, generator=gen) * 0.1
    save_file(tensors, path / "adapter_model.safetensors")
    return inspect_lora_adapter(str(path))


@pytest.mark.parametrize("tokens", [1, 3])
def test_expert_lora_matches_explicit_per_expert_reference(tmp_path, tokens):
    torch.manual_seed(4891)
    model = TinyExpertModel()
    pristine = copy.deepcopy(model.state_dict())
    adapter = write_adapter(tmp_path / "L0")
    assert attach_lora_buffers(model, adapter) == 2
    moe = model.model.layers[0].mlp.moeblock
    x = torch.randn(1, tokens, 64)
    routes = torch.tensor([0.2, 0.5, 0.3]).expand(1, tokens, 3)
    result = model(x, routes)
    tensors = load_file(tmp_path / "L0" / "adapter_model.safetensors")
    deltas = {}
    for pair in adapter.pairs:
        a, b = tensors[pair.a_key], tensors[pair.b_key]
        # PEFT ParamWrapper contracts: B[output, rank, expert], A[expert, rank, input].
        deltas[pair.target_parameter] = (
            torch.einsum(
                "ore,eri->eoi",
                b.reshape(-1, 2, 3),
                a.reshape(3, 2, a.shape[1]),
            )
            * adapter.scale
        )
    expected = torch.zeros_like(x)
    for expert, probability in ((1, 0.625), (2, 0.375)):

        def project(name, value):
            base = nn.functional.linear(value, getattr(moe, f"expert_{name}_proj_weight")[expert])
            delta = (
                deltas["down"][expert] if name == "down" else deltas["gate_up"][expert].chunk(2, dim=0)[name == "up"]
            )
            return base + nn.functional.linear(value, delta)

        expected += project("down", nn.functional.silu(project("gate", x)) * project("up", x)) * probability
    torch.testing.assert_close(result, expected)
    for key, weight in pristine.items():
        torch.testing.assert_close(model.state_dict()[key], weight, rtol=0, atol=0)


@pytest.mark.parametrize("override", [None, {"bits": 8, "fp_mode": "ssfp", "hidden_bit": False}])
@pytest.mark.parametrize("scheme", ["w4a8h0_ssfp", "w8a8h1_sefp"])
def test_expert_lora_quantization_reaches_existing_moeblock_abi(tmp_path, scheme, override):
    from xhquant.api import FrontendGraph, to_quant_graph, ptq_quantize, PrecisionMode

    model = TinyExpertModel().half()
    adapter = write_adapter(tmp_path / "L0")
    attach_lora_buffers(model, adapter)
    frontend = SimpleNamespace(prefill=xh_fx.symbolic_trace(model), decode=xh_fx.symbolic_trace(model))
    apply_lora_to_frontend(frontend, adapter, override)
    quanted = to_quant_graph(FrontendGraph.convert(frontend.prefill), "XH2a", {"quant_type": scheme})
    device = "cuda:0"
    x = torch.randn(1, 2, 64, device=device, dtype=torch.float16)
    routes = torch.tensor([0.2, 0.5, 0.3], device=device, dtype=torch.float16).expand(1, 2, 3).contiguous()
    ptq_quantize(quanted, [(x, routes)], PrecisionMode.ALIGNED, [device], infer_shape=False)
    quanted.to(device)
    result = quanted(x, routes)
    assert torch.isfinite(result).all()
    blocks = [m for m in quanted.modules() if hasattr(m, "_lora_operands") and m.has_lora]
    assert len(blocks) == 1
    operands = blocks[0]._lora_operands()
    assert len([k for k in operands if k.endswith(("_qweight", "_scale"))]) == 12
    assert operands["lora_scaling"] == 2

    # Use the production two-stage exporter, which clears frontend-only float
    # buffers. Direct module execution alone would miss dropped LoRA operands.
    import onnx
    from xhquant.api import to_export_graph_v2, to_export_hmonnx_v2
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceV2, HMONNXInferenceConfig

    values = (x.cpu(), routes.cpu())
    exported = to_export_graph_v2(quanted.cpu(), values)
    path = tmp_path / "experts.onnx"
    to_export_hmonnx_v2(exported, values, str(path), save_as_external_data=True)
    nodes = [n for n in onnx.load(path, load_external_data=False).graph.node if n.op_type == "MoeBlock"]
    assert len(nodes) == 1 and len(nodes[0].input) == 28
    from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader

    loader = MultiHMONNXLoader({"prefill": str(path)})
    loaded = HMONNXInferenceV2.from_onnx_graph(
        str(path),
        loader.graphs["prefill"],
        HMONNXInferenceConfig(exec_devices=[device]),
    )
    torch.testing.assert_close(loaded(x, routes), result, rtol=0, atol=0)


@pytest.mark.parametrize("transposed", [False, True])
def test_split_qkv_lora_keeps_packed_projection_semantics(tmp_path, transposed):
    class SplitProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            attn = self.layers[0].linear_attn = nn.Module()
            for suffix, outputs in (("q", 3), ("k", 3), ("v", 6)):
                setattr(attn, "in_proj_" + suffix, nn.Linear(4, outputs, bias=False))

        def forward(self, x):
            attn = self.layers[0].linear_attn
            return torch.cat([attn.in_proj_q(x), attn.in_proj_k(x), attn.in_proj_v(x)], dim=-1)

    path = tmp_path / "split"
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 2,
                "lora_alpha": 4,
                "bias": "none",
            }
        )
    )
    stem = "base_model.model.model.layers.0.linear_attn.in_proj_qkv"
    a, b = torch.randn(2, 4), torch.randn(12, 2)
    save_file(
        {
            stem + ".lora_A.weight": a.T.contiguous() if transposed else a,
            stem + ".lora_B.weight": b.T.contiguous() if transposed else b,
        },
        path / "adapter_model.safetensors",
    )
    adapter = inspect_lora_adapter(str(path))
    model = SplitProjection()
    x = torch.randn(2, 4)
    expected = model(x) + nn.functional.linear(nn.functional.linear(x, a), b) * 2
    assert attach_lora_buffers(model, adapter) == 1
    frontend = SimpleNamespace(prefill=xh_fx.symbolic_trace(model), decode=xh_fx.symbolic_trace(model))
    apply_lora_to_frontend(frontend, adapter, None)
    torch.testing.assert_close(frontend.prefill(x), expected)
    torch.testing.assert_close(frontend.decode(x), expected)


@pytest.mark.parametrize("mode", ["mtp", "dflash", "dflash2"])
def test_streamed_speculative_lora_fails_before_creating_artifacts(tmp_path, monkeypatch, mode):
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model

    adapter = write_adapter(tmp_path / "L0")
    model = object.__new__(XHQwen3_5Model)
    model.config = SimpleNamespace(lora={"path": [adapter.source_path]}, spec_decode_mode=mode)
    monkeypatch.setenv("HUGE_MODEL_EXPORT_ENABLED", "1")
    output = tmp_path / "export"
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        model.export_hmonnx(str(output))
    assert not output.exists()
