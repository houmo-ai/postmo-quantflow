"""The published cache ABI preserves storage, state types and external assets."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import convert_qwen4_exp_kv_to_llm_cache


def _cache_graph():
    cache_shape = [1, 1, 8, 2]
    current_shape = [1, 1, 1, 2]
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape) for name, shape in (
        ("key", current_shape), ("value", current_shape),
        ("past_key_cache_0", cache_shape), ("past_value_cache_0", cache_shape),
    )]
    inputs += [helper.make_tensor_value_info(name, TensorProto.INT32, [1])
               for name in ("past_seq_length", "current_input_length")]
    nodes = [helper.make_node("KVcache", [kind, "past_seq_length", "current_input_length", f"past_{kind}_cache_0"],
                              [f"present_past_{kind}_cache_0"], domain="ai.houmo.xh2a", axis=2)
             for kind in ("key", "value")]
    nodes.append(helper.make_node("Add", ["key", "weight"], ["logits"]))
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT16, current_shape)]
    outputs += [helper.make_tensor_value_info(f"present_past_{kind}_cache_0", TensorProto.FLOAT16, cache_shape)
                for kind in ("key", "value")]
    return helper.make_model(helper.make_graph(nodes, "cache", inputs, outputs,
        [numpy_helper.from_array(np.asarray([1, 2], dtype=np.float16), "weight")]),
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("ai.houmo.xh2a", 1)])


def test_cache_publication_keeps_writers_types_and_external_bytes(tmp_path):
    source = tmp_path / "source" / "raw.onnx"
    source.parent.mkdir()
    onnx.save_model(_cache_graph(), source, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="weights.bin", size_threshold=0)
    before = source.read_bytes()
    published = tmp_path / "published" / "cache.onnx"
    convert_qwen4_exp_kv_to_llm_cache(source, published, context_length=8, num_kv_heads=1, head_dim=2)
    model = onnx.load(published, load_external_data=False)
    onnx.checker.check_model(str(published))
    assert [value.name for value in model.graph.output] == ["logits"]
    writers = [node for node in model.graph.node if node.op_type == "KVcache"]
    assert len(writers) == 2
    for node in writers:
        assert {attr.name: helper.get_attribute_value(attr) for attr in node.attribute} == {
            "axis": 2, "attention_max_length": -1,
        }
        info = next(value for value in model.graph.value_info if value.name == node.output[0])
        assert info.type.tensor_type.elem_type == TensorProto.FLOAT16
        assert [dim.dim_value for dim in info.type.tensor_type.shape.dim] == [1, 1, 8, 2]
    assert source.read_bytes() == before
    assert (source.parent / "weights.bin").samefile(published.parent / "weights.bin")


def test_cache_publication_rejects_missing_paired_writer(tmp_path):
    model = _cache_graph()
    del model.graph.node[1]
    source = tmp_path / "raw.onnx"
    onnx.save(model, source)
    with pytest.raises(ValueError, match="all paired K/V writers"):
        convert_qwen4_exp_kv_to_llm_cache(source, tmp_path / "published.onnx", context_length=8,
                                       num_kv_heads=1, head_dim=2)
    assert not (tmp_path / "published.onnx").exists()
