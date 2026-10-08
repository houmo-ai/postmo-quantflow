"""Compiler-aligned GDR inverse: real VP primitives, export ABI, and replay."""
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
import torch

from xhquant.backend.xh2a.functions import gdr_block_tri_inverse as implementation
from xhquant.common.types import PrecisionMode
from xhquant.frontend.onnx.parsers.hmops.gdr_block_tri_inverse import GDRBlockTriInverse as Frontend
from xhquant.quantization.xh2a.builder import build_quant_module
from xhquant.quantization.xh2a.qmodules._gdr_reference import reference_gdr_block_tri_inverse_chain
from xhquant.quantization.xh2a.xh2a_quant_config import XH2AQuantConfig
from xhquant.xhonnxruntime.parsers.gdr_block_tri_inverse import GDRBlockTriInverse as Runtime
from xhmodel_merak.xh_llm.models.qwen3_5._gdr_ops import GDRBlockTriInverse


def inputs(chunk_size, device="cpu"):
    generator = torch.Generator().manual_seed(489)
    attn = torch.randn(1, 2, 2, chunk_size, chunk_size, generator=generator).half() * 0.025
    attn = torch.tril(attn, diagonal=-1).to(device)
    eye = torch.eye(8, dtype=torch.float16, device=device).expand(4 * chunk_size // 8, 8, 8).contiguous()
    return attn, eye


@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("mode", ["aligned", "fast"])
def test_all_products_use_vp_and_match_inverse(monkeypatch, chunk_size, mode):
    def reject_te(*args, **kwargs):
        raise AssertionError("GDR inverse must not dispatch a TE MatMul")

    monkeypatch.setattr(implementation, "_get_ir_matmul", reject_te)
    actual_vp = implementation._vp_matmul_8x8_chain
    shapes = []

    def observe_vp(mul_fn, a, b):
        shapes.append((tuple(a.shape[-2:]), tuple(b.shape[-2:])))
        return actual_vp(mul_fn, a, b)

    monkeypatch.setattr(implementation, "_vp_matmul_8x8_chain", observe_vp)
    attn, eye = inputs(chunk_size)
    fn = (implementation.gdr_block_tri_inverse_xh2a_default if mode == "aligned"
          else implementation.gdr_block_tri_inverse_xh2a_fast)
    actual = fn(attn, eye, chunk_size, 8)
    assert len(shapes) == 4 + 2 * (chunk_size // 8 - 1)
    assert ((8, chunk_size - 8), (chunk_size - 8, chunk_size - 8)) in shapes
    reference = reference_gdr_block_tri_inverse_chain(attn, eye, chunk_size, 8, mode=mode)
    assert torch.equal(actual, reference)
    oracle = torch.linalg.inv(torch.eye(chunk_size) - attn.float())
    torch.testing.assert_close(actual.float(), oracle, rtol=0.003, atol=0.0002)


@pytest.mark.parametrize("bits", [8, 16])
def test_quant_runtime_and_frontend_have_no_operand_bit(bits):
    attn, eye = inputs(16)
    base = GDRBlockTriInverse(16, 8)
    assert not hasattr(base, "matmul_input_bit")
    module = build_quant_module(base, XH2AQuantConfig(dict(act_schema=dict(bits=bits))).get_quant_config())
    module._precision_mode = PrecisionMode.ALIGNED
    assert not hasattr(module, "matmul_input_bit")
    variables = [gs.Variable("attn", np.float16), gs.Variable("eye", np.float16)]
    node = gs.Node("GDRBlockTriInverse", attrs=dict(chunk_size=16, block_size=8),
                   inputs=variables, outputs=[gs.Variable("out", np.float16)])
    frontend = Frontend().parse(gs.Graph(nodes=[node]), node)
    runtime = Runtime.from_onnx_node(node, None)
    assert not hasattr(frontend, "matmul_input_bit")
    assert not hasattr(runtime, "matmul_input_bit")
    expected = reference_gdr_block_tri_inverse_chain(attn, eye, 16, 8)
    assert torch.equal(module._hardward_forward(attn, eye), expected)
    assert torch.equal(runtime._forward_aligned(attn, eye)["out"], expected)


@pytest.mark.parametrize("bits", [8, 16])
def test_real_hmonnx_export_omits_operand_bit(tmp_path, bits):
    from xhquant.api import DeviceType, HMONNXInference, QuantScheme, convert_fx_model_to_hmonnx, create_quant_config

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inverse = GDRBlockTriInverse(16, 8)

        def forward(self, attn, eye):
            return self.inverse(attn, eye)

    attn, eye = inputs(16)
    config = create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=f"w8a{bits}h0_sefp"))
    path = str(tmp_path / "inverse.onnx")
    convert_fx_model_to_hmonnx(Model().eval(), [attn, eye], DeviceType.XH2a, path, quant_config=config.to_dict())
    nodes = [n for n in onnx.load(path).graph.node if n.op_type == "GDRBlockTriInverse"]
    assert len(nodes) == 1
    assert {a.name for a in nodes[0].attribute} == {"chunk_size", "block_size"}
    runtime = HMONNXInference(path)
    expected = reference_gdr_block_tri_inverse_chain(attn, eye, 16, 8)
    assert torch.equal(runtime(attn, eye), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA replay needs a GPU")
@pytest.mark.parametrize("chunk_size", [16, 64])
def test_cuda_graph_replays_changed_inputs(chunk_size):
    attn, eye = inputs(chunk_size, "cuda")
    runtime = Runtime(chunk_size, 8)
    runtime.output_names = ["out"]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            runtime._forward_aligned(attn, eye)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = runtime._forward_aligned(attn, eye)["out"]
    for scale in (0.7, 1.3):
        attn.mul_(scale)
        expected = runtime._forward_aligned(attn, eye)["out"]
        graph.replay()
        assert torch.equal(actual, expected)
