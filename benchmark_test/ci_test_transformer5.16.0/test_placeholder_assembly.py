"""Real SDK tuple ports and HMONNX assembly; preserve SDK debug metadata."""

import operator

import onnx
import pytest
import torch

from xhmodel_merak.xh_llm.big_hf_model_helper import replace_placeholder_node
from xhquant.api import (
    PrecisionMode,
    QuantScheme,
    create_quant_config,
    ptq_quantize,
    to_export_hmonnx_from_quanted_graph,
    to_frontend_graph,
    to_quant_graph,
)
from xhquant.core.graph import FrontendGraph
from xhquant.nn.modules import PlaceHolderModule


class _Main(torch.nn.Module):
    def __init__(self, indices, count):
        super().__init__()
        self.indices = indices
        self.block = PlaceHolderModule(
            output_shapes=[[1, 2]] * count,
            output_dtypes=[torch.float32] * count,
            num_outputs=count,
            content="block",
        )

    def forward(self, value):
        outputs = self.block(value)
        return tuple(outputs[index] for index in self.indices)


def _main_graph(indices, count, normalized=False):
    root = _Main(indices, count)
    if normalized:
        return to_frontend_graph(root, "TorchFX", [torch.ones(1, 2)])
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    block = graph.call_module("block", (value,))
    graph.output(tuple(graph.call_function(operator.getitem, (block, index)) for index in indices))
    return FrontendGraph.convert(torch.fx.GraphModule(root, graph), {})


def _quantize(frontend, values):
    config = create_quant_config(QuantScheme(target_device="XH2a", quant_type="w16a16_sefp"))
    model = to_quant_graph(frontend, "XH2a", config)
    ptq_quantize(model, [values], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    return model


class _Outputs(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.count = count

    def forward(self, value):
        return tuple(value + index + 1 for index in range(self.count))


@pytest.mark.parametrize("indices,count", [([0], 1), ([2, 0, 3, 1], 4), ([3, 0], 5)])
@pytest.mark.parametrize("normalized", [False, True], ids=["fx-getitem", "normalized-qgetitem"])
def test_real_sdk_placeholder_tuple_assembly_preserves_metadata(tmp_path, indices, count, normalized):
    values = [torch.tensor([[0.25, -0.5]])]
    main = _quantize(_main_graph(indices, count, normalized), values)
    sub = _quantize(to_frontend_graph(_Outputs(count), "TorchFX", values), values)
    main_file, sub_file = tmp_path / "main.onnx", tmp_path / "sub.onnx"
    to_export_hmonnx_from_quanted_graph(main, values, str(main_file), save_as_external_data=True)
    to_export_hmonnx_from_quanted_graph(sub, values, str(sub_file), save_as_external_data=True)
    model = onnx.load(main_file, load_external_data=False)
    sdk_metadata = [(entry.key, entry.value) for entry in model.metadata_props]
    runtime_placeholder = next(node for node in model.graph.node if node.op_type == "PlaceHolder")
    replace_placeholder_node(
        runtime_placeholder, model, sub_file, "main_external_data", tmp_path / "main_external_data"
    )
    onnx.checker.check_model(model)
    assert [(entry.key, entry.value) for entry in model.metadata_props] == sdk_metadata
    assert not any(node.op_type in {"PlaceHolder", "TupleGetItem"} for node in model.graph.node)

    # Execute the assembled artifact, not just its metadata or an FX substitute.
    from xhquant.api import HMONNXInference

    onnx.save(model, main_file)
    session = HMONNXInference(str(main_file))
    actual = session(*(value.half() for value in values))
    if isinstance(actual, torch.Tensor):
        actual = (actual,)
    expected = _Outputs(count)(*values)
    torch.testing.assert_close(tuple(actual), tuple(expected[index].half() for index in indices))



@pytest.mark.parametrize("invalid_index", [0, 99, -1])
def test_assembly_rejects_invalid_executable_tuple_indices(tmp_path, invalid_index):
    values = [torch.tensor([[0.25, -0.5]])]
    main = _quantize(_main_graph([0, 1], 2), values)
    sub = _quantize(to_frontend_graph(_Outputs(2), "TorchFX", values), values)
    main_file, sub_file = tmp_path / "main.onnx", tmp_path / "sub.onnx"
    to_export_hmonnx_from_quanted_graph(main, values, str(main_file), save_as_external_data=True)
    to_export_hmonnx_from_quanted_graph(sub, values, str(sub_file), save_as_external_data=True)
    model = onnx.load(main_file, load_external_data=False)
    accesses = [node for node in model.graph.node if node.op_type == "TupleGetItem"]
    assert len(accesses) == 2
    index = next(attr for attr in accesses[1].attribute if attr.name == "index")
    index.i = invalid_index
    placeholder = next(node for node in model.graph.node if node.op_type == "PlaceHolder")
    with pytest.raises(RuntimeError, match="index|indices"):
        replace_placeholder_node(placeholder, model, sub_file, "main_external_data", tmp_path / "main_external_data")


def _opset_fixture(version, op="HMIdentity", domain="ai.houmo.xh2a"):
    import importlib

    importlib.import_module("xhquant.export.onnx.onnx_opset")
    return onnx.helper.make_model(
        onnx.helper.make_graph([onnx.helper.make_node(op, ["x"], ["y"], domain=domain)], "opset", [], []),
        opset_imports=[onnx.helper.make_opsetid(domain, version)],
    )


@pytest.mark.parametrize("main_version,block_version", [(1, 2), (2, 1)])
def test_assembly_rejects_mixed_opsets_even_for_unchanged_operators(main_version, block_version):
    from xhmodel_merak.xh_llm.big_hf_model_helper import _merge_opset_imports

    main, block = _opset_fixture(main_version), _opset_fixture(block_version)
    higher = main if main_version == 2 else block
    higher.graph.node[0].op_type = "TopK"
    with pytest.raises(RuntimeError, match="Incompatible ONNX opset"):
        _merge_opset_imports(main, block)
    assert main.opset_import[0].version == main_version
    assert block.opset_import[0].version == block_version


@pytest.mark.parametrize("main_version,block_version", [(1, 2), (2, 1)])
@pytest.mark.parametrize("op,domain", [("TopK", "ai.houmo.xh2a"),
                                     ("UnknownOperator", "ai.houmo.xh2a"), ("Identity", "private")])
def test_assembly_rejects_changed_or_unproven_opset_upgrade(main_version, block_version, op, domain):
    from xhmodel_merak.xh_llm.big_hf_model_helper import _merge_opset_imports

    main, block = _opset_fixture(main_version, op, domain), _opset_fixture(block_version, op, domain)
    with pytest.raises(RuntimeError, match="Incompatible ONNX opset"):
        _merge_opset_imports(main, block)
    assert main.opset_import[0].version == main_version


@pytest.mark.parametrize("scope", ["function", "subgraph"])
def test_assembly_rejects_opset_upgrade_with_unproven_nested_scope(scope):
    from xhmodel_merak.xh_llm.big_hf_model_helper import _merge_opset_imports

    main, block = _opset_fixture(1), _opset_fixture(2)
    if scope == "function":
        main.functions.append(onnx.helper.make_function("private", "fn", ["x"], ["y"], [], []))
    else:
        main.graph.node[0].attribute.append(onnx.helper.make_attribute("body", block.graph))
    with pytest.raises(RuntimeError, match="Incompatible ONNX opset"):
        _merge_opset_imports(main, block)
