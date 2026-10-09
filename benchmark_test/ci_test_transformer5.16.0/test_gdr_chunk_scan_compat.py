"""Old GDRChunkScan graphs omit precision; explicit precision overrides it."""

import onnx
import onnx_graphsurgeon as gs
import pytest
import torch
from torch import nn

from xhquant.api import DeviceType, HMONNXInference, convert_fx_model_to_hmonnx
from xhquant.backend.xh2a.ir import gdr_chunk_scan_xh2a_default
from xhquant.core import CacheTensor
from xhquant.export.onnx.onnx_opset._impl.schemas.opset1.gdr_chunk_scan_schema import gdr_chunk_scan_schema
from xhquant.frontend.onnx.parsers.hmops.gdr_chunk_scan import GDRChunkScan as FrontendScan
from xhquant.nn import GDRChunkScan
from xhquant.ops.xh.gdr_chunk_scan import gdr_chunk_scan_default
from xhquant.xhonnxruntime.parsers.gdr_chunk_scan import GDRChunkScan as RuntimeScan


@pytest.fixture(scope="module")
def exported_scan(tmp_path_factory):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.scan = GDRChunkScan(1, 1, 64, 64, 8, matmul_input_bit=16)

        def forward(self, query, key, value, k_cumdecay, decay_mask, mask_incl, g, initial_state):
            return self.scan(query, key, value, k_cumdecay, decay_mask, mask_incl, g, initial_state)

    generator = torch.Generator().manual_seed(489)
    shape = (1, 1, 1, 8, 64)
    q, k, v, w = [torch.randn(shape, generator=generator) * 0.1 for _ in range(4)]
    g = -torch.rand(1, 1, 1, 8, generator=generator) * 0.1
    pair = (g.unsqueeze(-1) - g.unsqueeze(-2)).exp()
    mask = torch.ones(8, 8).tril()
    state = torch.randn(1, 1, 64, 64, generator=generator) * 0.1
    inputs = (q, k, v, w, pair, mask, g, state)
    path = tmp_path_factory.mktemp("gdr_compat") / "scan.onnx"
    convert_fx_model_to_hmonnx(Model().eval(), [*inputs[:-1], CacheTensor(state.clone())], DeviceType.XH2a, str(path))
    return onnx.load(path, load_external_data=False), inputs


def test_gdr_chunk_scan_precision_is_optional_and_defaults_to_eight():
    assert not gdr_chunk_scan_schema.attributes["matmul_input_bit"].required
    assert GDRChunkScan(1, 1, 64, 64).matmul_input_bit == 8


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scan_default_is_bit_exact_a8_and_a16_tracks_float_reference(exported_scan, device):
    graph, inputs = exported_scan
    node = next(n for n in graph.graph.node if n.op_type == "GDRChunkScan")
    tensors = tuple(x.to(device=device, dtype=torch.float16) for x in inputs)
    initializers = {
        value.name: torch.from_numpy(onnx.numpy_helper.to_array(value).copy()).to(device=device, dtype=torch.float16)
        for value in graph.graph.initializer
    }
    lut = tuple(initializers[name] for name in node.input[8:11])
    results = []
    for precision in (None, 8, 16):
        cache = CacheTensor(tensors[-1].clone())
        args = (*tensors[:-1], cache, *lut, 1, 1, 64, 64, 8)
        output = (
            gdr_chunk_scan_xh2a_default(*args) if precision is None else gdr_chunk_scan_xh2a_default(*args, precision)
        )
        results.append((output, cache.data.clone()))
    for implicit, explicit in zip(results[0], results[1], strict=True):
        assert torch.equal(implicit.view(torch.int16), explicit.view(torch.int16))
    # Bound A16 error independently of A8. The fixture retains nonzero
    # recurrent state and all five internal MatMul paths at 64 features.
    floats = tuple(value.float() for value in tensors)
    reference = gdr_chunk_scan_default(*floats[:-1], CacheTensor(floats[-1].clone()), 1, 1, 64, 64, 8)
    for actual, expected in zip(results[2], reference, strict=True):
        difference = actual.float() - expected.float()
        assert difference.square().mean().sqrt() < 3e-4
        assert difference.abs().max() < 2e-3


@pytest.mark.parametrize("attribute", [None, 8, 16], ids=["old_graph", "explicit_a8", "explicit_a16"])
def test_old_and_explicit_scan_graphs_parse_and_execute(exported_scan, tmp_path, attribute):
    original, inputs = exported_scan
    graph = onnx.ModelProto()
    graph.CopyFrom(original)
    node = next(n for n in graph.graph.node if n.op_type == "GDRChunkScan")
    existing = next(a for a in node.attribute if a.name == "matmul_input_bit")
    if attribute is None:
        node.attribute.remove(existing)
    else:
        existing.i = attribute
    onnx.checker.check_model(graph)
    path = tmp_path / "scan.onnx"
    onnx.save(graph, path)
    imported = gs.import_onnx(graph)
    frontend_node = next(n for n in imported.nodes if n.op == "GDRChunkScan")
    expected_bits = 8 if attribute is None else attribute
    assert FrontendScan().parse(imported, frontend_node).matmul_input_bit == expected_bits
    assert RuntimeScan.from_onnx_node(frontend_node, None).matmul_input_bit == expected_bits

    tensors = tuple(x.half() for x in inputs)
    initializers = {
        value.name: torch.from_numpy(onnx.numpy_helper.to_array(value).copy()).half()
        for value in graph.graph.initializer
    }
    lut = tuple(initializers[name] for name in node.input[8:11])
    expected_cache = CacheTensor(tensors[-1].clone())
    args = (*tensors[:-1], expected_cache, *lut, 1, 1, 64, 64, 8)
    # The old graph is compared to the pre-existing call signature, without
    # passing a bit argument at all. Explicit attributes use the override.
    expected = (
        gdr_chunk_scan_xh2a_default(*args) if attribute is None else gdr_chunk_scan_xh2a_default(*args, attribute)
    )
    cache = CacheTensor(tensors[-1].clone())
    session = HMONNXInference(str(path))
    output = session(*tensors[:-1], cache)
    if isinstance(output, dict):
        actual = next(iter(output.values()))
    elif isinstance(output, torch.Tensor):
        actual = output
    else:
        actual = output[0]
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(cache.data.view(torch.int16), expected_cache.data.view(torch.int16))
