import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

TOOL_PATH = Path("tools/fuse_hmonnx_parallel_linears.py")


def _load_tool():
    spec = importlib.util.spec_from_file_location("fuse_hmonnx_parallel_linears", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _linear(
    name: str,
    input_name: str,
    out_features: int,
    *,
    act_man_bit: int = 8,
) -> onnx.NodeProto:
    return helper.make_node(
        "Linear",
        [input_name, f"{name}.qweight", f"{name}.scale"],
        [f"{name}.output"],
        name=name,
        domain="ai.houmo.xh2a",
        have_bias=0,
        in_features=4,
        out_features=out_features,
        mode="ssfp",
        hmfp_act_exp_bit=5,
        hmfp_act_man_bit=act_man_bit,
        hmfp_act_nshare=2,
        hmfp_act_rounding="RNE",
        hmfp_act_hidden_bit=1,
        hmfp_weight_exp_bit=5,
        hmfp_weight_man_bit=8,
        hmfp_weight_nshare=2,
        hmfp_weight_rounding="RNE",
        hmfp_weight_hidden_bit=0,
        hmfp_psum_dtype="fp24",
        hmfp_psum_round_mode="trunc",
        output_dtype="float16",
    )


def _graph(linears: list[onnx.NodeProto]) -> onnx.GraphProto:
    initializers = []
    for index, node in enumerate(linears, 1):
        attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
        out_features = int(attributes["out_features"])
        qweight = np.arange(4 * out_features, dtype=np.int8).reshape(2, 2, out_features) + index
        scale = np.full((2, 1, out_features), index - 2, dtype=np.int8)
        initializers.extend(
            [
                numpy_helper.from_array(qweight, node.input[1]),
                numpy_helper.from_array(scale, node.input[2]),
            ]
        )
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, 2, 2])
        for name in dict.fromkeys(node.input[0] for node in linears)
    ]
    outputs = [helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT16, [1, None]) for node in linears]
    return helper.make_graph(linears, "parallel_linears", inputs, outputs, initializers)


def test_fusion_requires_same_input_and_sefp_format() -> None:
    tool = _load_tool()
    linears = [
        _linear("a", "hidden", 2),
        _linear("b", "hidden", 3),
        _linear("other_input", "other", 2),
        _linear("other_sefp", "hidden", 2, act_man_bit=7),
    ]
    graph = _graph(linears)

    groups = tool._find_fusion_groups(graph, tool._initializer_map(graph))

    assert [[node.name for node in group] for group in groups] == [["a", "b"]]


def test_default_output_and_external_data_names(monkeypatch, tmp_path: Path) -> None:
    tool = _load_tool()
    input_path = tmp_path / "model.onnx"
    model = helper.make_model(
        _graph([_linear("a", "hidden", 2), _linear("b", "hidden", 3)]),
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("ai.houmo.xh2a", 1)],
    )
    onnx.save(model, input_path)
    monkeypatch.setattr(sys, "argv", [str(TOOL_PATH), str(input_path)])

    tool.main()

    output_path = tmp_path / "model_linear_fused.onnx"
    external_path = tmp_path / "model_linear_fused_external_data"
    assert output_path.is_file()
    assert external_path.is_file()
    fused = onnx.load(output_path, load_external_data=False)
    fused_initializers = [
        tensor for tensor in fused.graph.initializer if tensor.name.startswith("parallel_linear_fusion")
    ]
    assert fused_initializers
    assert {
        next(entry.value for entry in tensor.external_data if entry.key == "location") for tensor in fused_initializers
    } == {external_path.name}


def test_fused_linear_split_is_numerically_equivalent(tmp_path: Path) -> None:
    tool = _load_tool()
    linears = [_linear("a", "hidden", 2), _linear("b", "hidden", 3)]
    graph = _graph(linears)
    initializers = tool._initializer_map(graph)
    reader = tool._ExternalTensorReader(tmp_path)
    external_path = tmp_path / "fused.data"
    try:
        with external_path.open("wb") as output:
            fused_qweight, _ = tool._stream_concat_last_axis(
                [initializers["a.qweight"], initializers["b.qweight"]],
                reader,
                output,
                output_name="fused.qweight",
                external_location=external_path.name,
            )
            fused_scale, _ = tool._stream_concat_last_axis(
                [initializers["a.scale"], initializers["b.scale"]],
                reader,
                output,
                output_name="fused.scale",
                external_location=external_path.name,
            )
    finally:
        reader.close()

    tool._validate_fused_tensor_equivalence([linears], initializers, [(fused_qweight, fused_scale)], tmp_path)
    hidden = np.array([[[1.0, -2.0], [0.5, 3.0]]], dtype=np.float32)

    def reference(qweight_name: str, scale_name: str) -> np.ndarray:
        qweight = numpy_helper.to_array(initializers[qweight_name]).astype(np.float32)
        scale = numpy_helper.to_array(initializers[scale_name]).astype(np.float32)
        return np.einsum("bgs,gso->bo", hidden, qweight * np.exp2(scale))

    originals = [reference("a.qweight", "a.scale"), reference("b.qweight", "b.scale")]
    fused_q = np.concatenate(
        [numpy_helper.to_array(initializers["a.qweight"]), numpy_helper.to_array(initializers["b.qweight"])],
        axis=-1,
    ).astype(np.float32)
    fused_s = np.concatenate(
        [numpy_helper.to_array(initializers["a.scale"]), numpy_helper.to_array(initializers["b.scale"])],
        axis=-1,
    ).astype(np.float32)
    fused_output = np.einsum("bgs,gso->bo", hidden, fused_q * np.exp2(fused_s))

    for original, split in zip(originals, np.split(fused_output, [2], axis=-1), strict=True):
        np.testing.assert_array_equal(split, original)
