"""Real SDK tuple metadata and HMONNX assembly, without a checkpoint."""

import operator

import onnx
import pytest
import torch

from xhmodel_merak.xh_llm.big_hf_model_helper import replace_placeholder_node
from xhmodel_merak.xh_llm.hmonnx.deterministic_export import (
    _decode_quant_metadata_model,
    flatten_placeholder_quant_metadata,
)
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
def test_real_sdk_placeholder_tuple_metadata_and_hmonnx_assembly(tmp_path, indices, count, normalized):
    values = [torch.tensor([[0.25, -0.5]])]
    main = _quantize(_main_graph(indices, count, normalized), values)
    # QuantedInfo really loses literal indices; never infer them from node order.
    original = main.dump_quant_info_to_onnx_model("QuantedInfo")
    accesses = [node for node in original.graph.node if list(node.input) == ["block"]]
    assert len(accesses) == len(indices)
    assert all(not any(attr.name == "index" for attr in node.attribute) for node in accesses)
    assert flatten_placeholder_quant_metadata(main) == 1
    sub = _quantize(to_frontend_graph(_Outputs(count), "TorchFX", values), values)
    main_file, sub_file = tmp_path / "main.onnx", tmp_path / "sub.onnx"
    to_export_hmonnx_from_quanted_graph(main, values, str(main_file), save_as_external_data=True)
    to_export_hmonnx_from_quanted_graph(sub, values, str(sub_file), save_as_external_data=True)
    model = onnx.load(main_file, load_external_data=False)
    _, metadata = _decode_quant_metadata_model(model)
    placeholder = next(node for node in metadata.graph.node if node.op_type.endswith("QPlaceHolderModule"))
    assert len(placeholder.output) == count
    for ordinal, index in enumerate(indices):
        assert placeholder.output[index] == accesses[ordinal].output[0]
    runtime_placeholder = next(node for node in model.graph.node if node.op_type == "PlaceHolder")
    replace_placeholder_node(
        runtime_placeholder, model, sub_file, "main_external_data", tmp_path / "main_external_data"
    )
    onnx.checker.check_model(model)
    _, metadata = _decode_quant_metadata_model(model)
    assert not any(node.op_type.endswith("QPlaceHolderModule") for node in metadata.graph.node)
    assert not any(node.op_type in {"PlaceHolder", "TupleGetItem"} for node in model.graph.node)
    known = {value.name for value in metadata.graph.input}
    for node in metadata.graph.node:
        assert set(node.input) <= known
        known.update(node.output)
    assert all(value.name in known for value in metadata.graph.output)

    # Execute the assembled artifact, not just its metadata or an FX substitute.
    from xhquant.api import HMONNXInference

    onnx.save(model, main_file)
    session = HMONNXInference(str(main_file))
    actual = session(*(value.half() for value in values))
    if isinstance(actual, torch.Tensor):
        actual = (actual,)
    expected = _Outputs(count)(*values)
    torch.testing.assert_close(tuple(actual), tuple(expected[index].half() for index in indices))


@pytest.mark.parametrize("indices,count", [([0, 0], 1), ([2], 2), ([-1], 2)])
def test_metadata_rejects_ambiguous_or_invalid_tuple_access(indices, count):
    model = to_quant_graph(_main_graph(indices, count), "XH2a", {}).disable_quant()
    model.create_quant_meta()
    before = dict(model.graph_meta.quant_meta)
    with pytest.raises(RuntimeError, match="Invalid placeholder tuple access"):
        flatten_placeholder_quant_metadata(model)
    assert model.graph_meta.quant_meta == before
