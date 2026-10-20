"""Packed PLE encoding and real SDK lookup/host placement contracts."""

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_runtime import (
    Qwen4ExpHostPLEInferenceV2,
    find_host_ple_gathers,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.ple_sefp4 import (
    PLE_SEFP4_OP,
    lookup_ple_sefp4,
    quantize_ple_sefp4_rows,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.xhquant_extensions import qwen4_exp_xhquant_extension_scope
from xhquant.quantization.xh2a.quantizer._sefp_utils import sefp
from xhquant.xhonnxruntime.builder import HMONNX_PARSERS
from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig


def _weight(width=160):
    return torch.linspace(-0.1, 0.1, 5 * width, dtype=torch.float16).reshape(5, width)


@pytest.mark.parametrize("width", [1, 64, 65, 160])
def test_storage_is_packed_and_lookup_matches_sdk_assemble(width):
    weight = _weight(width)
    before = weight.clone()
    packed, exp = quantize_ple_sefp4_rows(weight)
    assert torch.equal(weight, before)
    groups = (width + 63) // 64
    assert packed.dtype == torch.int8 and packed.shape == (5, groups, 32)
    assert exp.dtype == torch.int16 and exp.shape == (5, groups, 1)
    if width >= 64:
        assert packed.nbytes + exp.nbytes < weight.nbytes
    expected, _ = sefp(weight.clone(), dim=1, manbit=3, nshare=64, assemble=True)
    ids = torch.tensor([[0, -1, 2, 2, -5]], dtype=torch.int32)
    torch.testing.assert_close(
        lookup_ple_sefp4(packed, exp, ids, embedding_dim=width), expected[ids.long()], rtol=0, atol=0,
    )
    empty = lookup_ple_sefp4(packed, exp, torch.empty(2, 0, dtype=torch.int64), embedding_dim=width)
    assert empty.shape == (2, 0, width)
    for index in (-6, 5):
        with pytest.raises(IndexError, match="out of bounds"):
            lookup_ple_sefp4(packed, exp, torch.tensor(index), embedding_dim=width)
    with pytest.raises(TypeError, match="int32 or int64"):
        lookup_ple_sefp4(packed, exp, torch.tensor([1.0]), embedding_dim=width)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_source_rejected_without_mutation(value):
    weight = _weight()
    weight[0, 0] = value
    with pytest.raises(ValueError, match="NaN or Inf"):
        quantize_ple_sefp4_rows(weight)
    assert (torch.isnan(weight[0, 0]) if np.isnan(value) else weight[0, 0] == value)


def _graph_file(tmp_path):
    weight = _weight()
    packed, exp = quantize_ple_sefp4_rows(weight)
    graph = helper.make_graph(
        [
            helper.make_node(
                PLE_SEFP4_OP, ["ple.packed", "ple.exponent", "ids"], ["embedding"],
                name="ple_lookup", domain="ai.houmo.xh2a", embedding_dim=160,
                nshare=64, manbit=3, keep_bit_for_exp=1,
            ),
            helper.make_node("TopK", ["embedding", "k"], ["values", "top_ids"], name="top_k", axis=-1),
        ],
        "packed_ple",
        [helper.make_tensor_value_info("ids", TensorProto.INT32, [1, 3])],
        [helper.make_tensor_value_info("values", TensorProto.FLOAT16, [1, 3, 2])],
        initializer=[
            numpy_helper.from_array(packed.numpy(), "ple.packed"),
            numpy_helper.from_array(exp.numpy(), "ple.exponent"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "k"),
        ],
    )
    path = tmp_path / "packed.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 17), helper.make_opsetid("ai.houmo.xh2a", 1),
    ]), path)
    return path, packed, exp


class _TinyHostSession(Qwen4ExpHostPLEInferenceV2):
    # Original FP16 is 1600 bytes; packed storage is only 510 bytes.
    HOST_PLE_MIN_BYTES = 1000


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_real_scoped_sdk_load_lookup_and_two_weight_host_placement(tmp_path, device):
    path, packed, exp = _graph_file(tmp_path)
    before = dict(HMONNX_PARSERS.module_dict)
    with qwen4_exp_xhquant_extension_scope():
        session = _TinyHostSession(str(path), HMONNXInferenceConfig(
            exec_devices=[device], enable_auto_offload=device.startswith("cuda"),
        ))
    assert HMONNX_PARSERS.module_dict == before
    assert find_host_ple_gathers(session.graph_module, min_nbytes=1000) == {"ple_lookup": 510}
    ids = torch.tensor([[0, 4, 2]], dtype=torch.int32, device=device)
    actual = session.run({"ids": ids})
    if isinstance(actual, (tuple, list)):
        actual = actual[0]
    expected = torch.topk(lookup_ple_sefp4(packed, exp, ids.cpu(), embedding_dim=160), 2, dim=-1).values
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    buffers = dict(session.graph_module.named_buffers())
    assert all(t.device.type == "cpu" for n, t in buffers.items() if "ple" in n)
    if device.startswith("cuda"):
        interpreter = session.interpreter
        assert interpreter.node_device_map["ple_lookup"].device == "cpu"
        assert interpreter.node_device_map["ple_lookup"].size_in_bytes == 510
        assert interpreter._get_node_memory_footprints()["ple_lookup"] == 0
        assert interpreter.node_device_map["top_k"].device == device


@pytest.mark.parametrize("attribute,value", [("nshare", 32), ("manbit", 7), ("keep_bit_for_exp", 0)])
def test_sdk_rejects_incompatible_quantization_abi_and_restores_registry(tmp_path, attribute, value):
    path, _, _ = _graph_file(tmp_path)
    model = onnx.load(path)
    node = model.graph.node[0]
    attributes = [item for item in node.attribute if item.name != attribute]
    del node.attribute[:]
    node.attribute.extend(attributes + [helper.make_attribute(attribute, value)])
    onnx.save(model, path)
    before = dict(HMONNX_PARSERS.module_dict)
    with pytest.raises(ValueError, match=f"requires {attribute}="):
        with qwen4_exp_xhquant_extension_scope():
            _TinyHostSession(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))
    assert HMONNX_PARSERS.module_dict == before


def test_corrupt_layout_rejected_before_lookup():
    packed, exp = quantize_ple_sefp4_rows(_weight())
    with pytest.raises(ValueError, match="packed storage"):
        lookup_ple_sefp4(packed[:, :, :-1], exp, torch.tensor([0]), embedding_dim=160)
    with pytest.raises(ValueError, match="exponent storage"):
        lookup_ple_sefp4(packed, exp.to(torch.int8), torch.tensor([0]), embedding_dim=160)


def test_bfloat16_sdk_lookup_preserves_dtype_and_small_exponents(tmp_path):
    # Rows with values beyond FP16's exponent range must survive lookup.
    weight = _weight().to(torch.bfloat16)
    weight[0] *= 2 ** -60
    weight[1] *= 2 ** 40
    packed, exp = quantize_ple_sefp4_rows(weight)
    expected, _ = sefp(weight.clone(), dim=1, manbit=3, nshare=64, assemble=True)
    ids = torch.tensor([[0, 1, 4]], dtype=torch.int32)
    direct = lookup_ple_sefp4(packed, exp, ids, embedding_dim=160, output_dtype=torch.bfloat16)
    torch.testing.assert_close(direct, expected[ids.long()], rtol=0, atol=0)
    assert direct.dtype == torch.bfloat16
    assert torch.count_nonzero(direct[0, 0]) > 0
    assert direct[0, 1].abs().max() > torch.finfo(torch.float16).max
    path, _, _ = _graph_file(tmp_path)
    model = onnx.load(path)
    del model.graph.node[1:]
    model.graph.node[0].attribute.append(helper.make_attribute("output_dtype", TensorProto.BFLOAT16))
    model.graph.output[0].name = "embedding"
    model.graph.output[0].type.tensor_type.elem_type = TensorProto.BFLOAT16
    model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value = 160
    del model.graph.initializer[:]
    model.graph.initializer.extend([
        numpy_helper.from_array(packed.numpy(), "ple.packed"),
        numpy_helper.from_array(exp.numpy(), "ple.exponent"),
    ])
    onnx.save(model, path)
    with qwen4_exp_xhquant_extension_scope():
        session = _TinyHostSession(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))
    actual = session.run({"ids": ids})
    if isinstance(actual, (tuple, list)):
        actual = actual[0]
    torch.testing.assert_close(actual, expected[ids.long()], rtol=0, atol=0)
    with pytest.raises(ValueError, match="output_dtype"):
        lookup_ple_sefp4(packed, exp, ids, embedding_dim=160, output_dtype=torch.float32)


def test_sdk_rejects_unsupported_output_dtype(tmp_path):
    path, _, _ = _graph_file(tmp_path)
    model = onnx.load(path)
    model.graph.node[0].attribute.append(helper.make_attribute("output_dtype", TensorProto.FLOAT))
    onnx.save(model, path)
    with pytest.raises(ValueError, match="output_dtype requires"):
        with qwen4_exp_xhquant_extension_scope():
            _TinyHostSession(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))
