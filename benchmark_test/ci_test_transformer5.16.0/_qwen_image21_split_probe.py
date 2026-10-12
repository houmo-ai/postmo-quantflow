"""Check the actual exported Split tuple ABI and execute it on CPU."""

import math

import onnx
import torch


def assert_split_probe(model, sections, input_shape, path, *, split_output):
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    graph = model.graph
    splits = [node for node in graph.node if node.op_type == "Split" and split_output in node.output]
    assert len(splits) == 1, f"expected one Split producing {split_output!r}, found {len(splits)}"
    split, = splits
    assert len(split.input) == 1 and split.input[0], "Split probe requires exactly one non-empty tensor input"
    assert len(split.output) == 1
    types = {entry.name: entry.type for entry in split.attribute}
    assert types.get("indices_or_sections") == onnx.AttributeProto.INTS
    assert types.get("axis") == onnx.AttributeProto.INT
    attrs = {entry.name: onnx.helper.get_attribute_value(entry) for entry in split.attribute}
    assert "num_outputs" not in attrs
    sections = list(sections)
    assert list(attrs["indices_or_sections"]) == sections
    axis = int(attrs["axis"])
    assert -len(input_shape) <= axis < len(input_shape)
    assert axis % len(input_shape) == len(input_shape) - 1
    parts = [node for node in graph.node if node.op_type == "TupleGetItem" and node.input[0] == split.output[0]]
    indexed = {next(a.i for a in node.attribute if a.name == "index"): node for node in parts}
    assert len(parts) == len(indexed) == len(sections), "TupleGetItem indices must be unique and cover every section"
    assert set(indexed) == set(range(len(sections)))
    values = {value.name: value for value in (*graph.input, *graph.value_info, *graph.output)}
    assert split.input[0] in values, "Split input tensor metadata is missing"
    input_info = values[split.input[0]]
    dtypes = {onnx.TensorProto.FLOAT16: torch.float16, onnx.TensorProto.FLOAT: torch.float32}
    elem_type = input_info.type.tensor_type.elem_type
    assert elem_type in dtypes, "Split probe only supports FLOAT or FLOAT16 tensors"
    assert [dim.dim_value for dim in input_info.type.tensor_type.shape.dim] == input_shape
    assert input_shape[-1] == sum(sections)
    outputs = [values[indexed[index].output[0]] for index in range(len(sections))]
    for output, size in zip(outputs, sections, strict=True):
        assert [dim.dim_value for dim in output.type.tensor_type.shape.dim] == [*input_shape[:-1], size]
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    probe.graph.CopyFrom(onnx.helper.make_graph([split, *indexed.values()], "split_probe", [input_info], outputs))
    onnx.save(probe, path)
    payload = torch.arange(math.prod(input_shape)).remainder(257).reshape(input_shape).to(dtypes[elem_type])
    runtime = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cpu"]))
    actual = runtime(payload)
    expected = torch.split(payload, sections, dim=-1)
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)
