"""The CUDA optimization must preserve VP rounding, selection and replay."""

import onnx
import onnx_graphsurgeon as gs
import pytest
import torch

from xhquant.api import HMONNXInference
from xhquant.backend.xh2a.functions.vp.sparse_selectkv_opt import sparse_selectkv_dots_opt
from xhquant.backend.xh2a.ir.operators.activate import generate_relu_lut_table
from xhquant.backend.xh2a.ir.operators.mul import mul_xh2a_default
from xhquant.backend.xh2a.ir.operators.reduce import reducesum_xh2a_default
from xhquant.backend.xh2a.ir.operators.sparse_attention_select_kv import (
    sparse_attention_select_kv_xh2a_default as select,
)
from xhquant.backend.xh2a.ir.operators.sparse_attention_select_kv import (
    sparse_attention_select_kv_xh2a_default_opt as optimized,
)
from xhquant.backend.xh2a.ir.operators.sparse_attention_select_kv import (
    sparse_attention_select_kv_xh2a_default_original as original,
)
from xhquant.core.opset import XH2a_DOMAIN
from xhquant.frontend.onnx.parsers.hmops.sparse_attention_select_kv import SparseAttentionSelectKV as FrontendSelect
from xhquant.xhonnxruntime.parsers.sparse_attention_select_kv import SparseAttentionSelectKV as RuntimeSelect


def _inputs(device, *, dim=31, tokens=4, capacity=65):
    generator = torch.Generator(device=device).manual_seed(489)

    def rand(shape):
        return torch.randn(shape, device=device, dtype=torch.float16, generator=generator)

    # Real model queries and head weights are transposed, not contiguous.
    query = rand((1, tokens, 4, dim)).transpose(1, 2)
    keys = rand((1, 1, capacity, dim * 2))[..., ::2]
    payload = torch.arange(capacity * 4 * 3, device=device, dtype=torch.float16).reshape(1, 1, capacity * 4, 3)
    length = torch.tensor([capacity], device=device, dtype=torch.int32)
    weights = rand((1, tokens, 4, 1)).transpose(1, 2)
    per_query = torch.tensor([[0, 3, 16, capacity]], device=device, dtype=torch.int64)[:, :tokens].clone()
    if tokens == 1:
        per_query.fill_(capacity)
    lut = tuple(torch.as_tensor(x, device=device, dtype=torch.float16) for x in generate_relu_lut_table())
    return query, keys, payload, length, weights, 16, dim**-0.5, *lut, 4, per_query


def _assert_bits(actual, expected):
    assert actual.dtype == expected.dtype == torch.float16
    assert torch.equal(actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16))


def test_selectkv_default_dispatch_and_cpu_fallback():
    assert select is optimized
    args = _inputs("cpu")
    _assert_bits(select(*args), original(*args))


def _legacy_node():
    names = ("query", "index_key_cache", "payload_cache", "valid_length")
    shapes = ((1, 4, 1, 31), (1, 1, 65, 31), (1, 1, 65, 3), (1,))
    inputs = [
        gs.Variable(name, dtype="int32" if name == "valid_length" else "float16", shape=shape)
        for name, shape in zip(names, shapes, strict=True)
    ]
    cut, table, scale = generate_relu_lut_table()
    return gs.Node(
        "SparseAttentionSelectKV",
        domain=XH2a_DOMAIN,
        attrs={
            "candidate_topk": 16,
            "score_scale": 31**-0.5,
            "lut_cut_points": list(cut),
            "lut_table": list(table),
            "lut_scale": list(scale),
        },
        inputs=inputs,
        outputs=[gs.Variable("selected_kv", dtype="float16", shape=(1, 1, 16, 3))],
    )


def test_old_four_input_single_output_selectkv_graph(tmp_path):
    node = _legacy_node()
    graph = gs.Graph(nodes=[node], inputs=node.inputs, outputs=node.outputs, opset=18)
    model = gs.export_onnx(graph)
    # Preserve the old contract: no pool_size, score_mode, head weights or
    # per-query length. Its four inputs must still select individual rows.
    model.opset_import.append(onnx.helper.make_opsetid(XH2a_DOMAIN, 1))
    onnx.checker.check_model(model)
    path = tmp_path / "legacy_selectkv.onnx"
    onnx.save(model, path)
    assert FrontendSelect().parse(graph, node).pool_size == 1
    assert RuntimeSelect.from_onnx_node(node, None).pool_size == 1
    args = list(_inputs("cpu", tokens=1))
    args[2] = args[2][:, :, :65]
    args[4] = None
    session = HMONNXInference(str(path))
    for length in (0, 3, 65):
        args[3].fill_(length)
        expected = original(*args[:10])
        result = session(*args[:4])
        actual = next(iter(result.values())) if isinstance(result, dict) else result
        if isinstance(actual, (tuple, list)):
            actual = actual[0]
        assert actual.shape == (1, 1, 16, 3)
        _assert_bits(actual, expected)
        assert torch.count_nonzero(actual[:, :, min(length, 16) :]) == 0


@pytest.mark.parametrize("mode,outputs", [("hmfp_pool", 1), ("unknown", 1), ("vp", 2)])
def test_selectkv_parsers_reject_removed_or_unknown_contracts(mode, outputs):
    node = _legacy_node()
    node.attrs["score_mode"] = mode
    if outputs == 2:
        node.outputs.append(gs.Variable("old_mask", dtype="float16", shape=(1, 1, 16)))
    graph = gs.Graph(nodes=[node], inputs=node.inputs, outputs=node.outputs)
    with pytest.raises(ValueError, match="VP mode and one output"):
        FrontendSelect().parse(graph, node)
    with pytest.raises(ValueError, match="score_mode|requires 1 output"):
        RuntimeSelect.from_onnx_node(node, None)


@pytest.mark.parametrize("dim", [31, 128])
@pytest.mark.parametrize("scale", [2.0**-12, 0.125, 16.0])
def test_fused_vp_dot_rounding_and_saturation(dim, scale):
    assert torch.cuda.is_available(), "VP CUDA optimization regression requires a GPU"
    query, keys, _, length, *_ = _inputs("cuda", dim=dim, tokens=1)
    query.mul_(scale)
    keys.mul_(scale)
    # Include signed zero, subnormals, cancellation, overflow and non-finite
    # products. Compare to the existing CUDA ACCU, including its conversions.
    query[0, 0, 0, :8] = torch.tensor(
        [0.0, -0.0, 2.0**-24, -(2.0**-24), 65504.0, -65504.0, float("inf"), float("nan")],
        device="cuda",
        dtype=torch.float16,
    )
    for valid in [0, 3, 33, keys.shape[2]]:
        length.fill_(valid)
        expected = reducesum_xh2a_default(mul_xh2a_default(query, keys), [-1], True)
        actual = sparse_selectkv_dots_opt(query, keys, length)
        _assert_bits(actual[:, :, :valid], expected[:, :, :valid])
        assert torch.count_nonzero(actual[:, :, valid:]) == 0


@pytest.mark.parametrize("pool_size", [1, 4])
def test_selectkv_stable_selection_and_inputs_unchanged(pool_size):
    args = list(_inputs("cuda"))
    args[-2] = pool_size
    if pool_size == 1:
        args[2] = args[2][:, :, :65]
    snapshots = [(value, value.clone()) for value in args if isinstance(value, torch.Tensor)]
    for tied in [False, True]:
        _assert_bits(select(*args), original(*args))
        if not tied:
            for value, snapshot in snapshots:
                assert torch.equal(value, snapshot)
        # All scores tie; valid pools must come before zero padding.
        args[0].zero_()
        args[4].fill_(-1)
    # Finite VP weighted sums overflow to -inf during score scaling. Stable
    # ties must still keep valid candidates ahead of the masked -inf suffix.
    args[0].fill_(1)
    args[1].fill_(1)
    args[6] = 65504.0
    _assert_bits(select(*args), original(*args))
    expected = args[2][0, 0, : 3 * pool_size]
    _assert_bits(select(*args)[0, 1, : 3 * pool_size], expected)
    assert torch.count_nonzero(select(*args)[0, 1, 3 * pool_size :]) == 0
    args[4] = None
    _assert_bits(select(*args), original(*args))


def test_selectkv_graph_replay_reads_current_lengths_and_cache():
    args = list(_inputs("cuda"))
    for _ in range(3):
        select(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = select(*args)
    for valid in [0, 3, 16, 33, 65, 5]:
        args[3].fill_(valid)
        args[-1].copy_(torch.tensor([[0, 3, 16, 65]], device="cuda"))
        args[0].neg_()
        args[1].add_(0.0625)
        args[2].add_(0.125)
        args[4].neg_()
        expected = original(*args)
        graph.replay()
        _assert_bits(captured, expected)
