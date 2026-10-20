from __future__ import annotations

import functools
import json
import operator
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import xhquant.nn as xhnn
from xhmodel_merak.xh_llm.models.qwen3_5._hybrid_gated_delta_net import HybridGatedDeltaNetMixin
from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.hmonnx_export import (
    _align_graph_constants_with_consumers,
    _build_qwen4_exp_gptq_export_device_map,
    _dynamo_export_compatibility_scope,
    _normalize_frontend_dynamic_slice_modules,
    _normalize_frontend_gather_overloads,
    _normalize_frontend_llm_cache_modules,
    _normalize_frontend_rmsnorm_modules,
    _normalize_frontend_rope_modules,
    _normalize_functional_conv1d_modules,
    _normalize_functional_linear_modules,
    _normalize_quantized_moe_call_signatures,
    _opaque_gptq_modules_for_dynamo,
    _remove_device_only_transfer_nodes,
    _remove_export_identity_nodes,
    _restore_gptq_modules_after_dynamo,
    build_qwen4_exp_serving_sample_inputs,
    export_qwen4_exp_hmonnx,
    preflight_qwen4_exp_hmonnx_export,
    run_qwen4_exp_hmonnx_golden,
    sanitize_qwen4_exp_frontend_graph,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import (
    Qwen4ExpExportableGatedDeltaNet,
    Qwen4ExpExportUnsupportedError,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.ple_sharding import (
    group_qwen4_exp_ple_shard_placement_tags,
    shard_qwen4_exp_ple_gather,
)


def test_export_identity_removal_preserves_fanout_and_graph_outputs():
    from xhquant.quantization.xh2a.qmodules.identity import QIdentity

    class ExportIdentity(QIdentity):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def forward(self, value):
            return value

    owner = torch.nn.Module()
    owner.add_module("identity", ExportIdentity())
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    identity = graph.call_module("identity", (source,))
    identity = graph.call_function(torch.ops.xh2a.HMIdentity.default, (identity,))
    doubled = graph.call_function(torch.add, (identity, identity))
    graph.output((identity, doubled))
    model = torch.fx.GraphModule(owner, graph)
    values = torch.randn(3, 4)
    assert _remove_export_identity_nodes(model) == ("identity", "hmidentity_default")
    after = model(values)
    for actual, expected in zip(after, (values, values + values), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(node.op != "call_module" for node in model.graph.nodes)


def _flash_next_text_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="qwen4_exp_text",
        num_hidden_layers=2,
        layer_types=[LINEAR_ATTENTION, FULL_ATTENTION],
        hidden_size=32,
        hc_count=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        indexer_head_dim=4,
        indexer_budget=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        ple_conv_kernel_size=4,
        ngram_size=3,
        ple_layer_ids=[1],
    )


def _isolated_sdk_registry(test):
    """Keep synthetic custom-op graphs independent of process-wide SDK registration.

    Export tests elsewhere register real ONNX schemas and generated quant-module
    classes globally. These structural fixtures deliberately supply only inputs
    needed by the rewrite under test; run them with a fresh SDK registry while
    preserving the normal checker and every transform assertion.
    """

    @functools.wraps(test)
    def run(*args, **kwargs):
        marker = "QWEN4_ISOLATED_EXPORT_TEST"
        if os.environ.get(marker) == test.__name__:
            return test(*args, **kwargs)
        env = dict(os.environ, **{marker: test.__name__})
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-c",
                str(Path(__file__).resolve().parent / "pytest.ini"),
                "--confcutdir",
                str(Path(__file__).resolve().parent),
                f"{Path(__file__).resolve()}::{test.__name__}",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    return run


def test_hmonnx_preflight_rejects_sdk_without_public_export_api(monkeypatch):
    import xhquant.api
    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import inspect_qwen4_exp_export_capability

    monkeypatch.delattr(xhquant.api, "to_export_hmonnx_v2")
    capability = inspect_qwen4_exp_export_capability(_flash_next_text_config(), require_hmonnx_runtime=True)
    assert not capability.supported
    assert any("public v2 export API" in reason for reason in capability.reasons)


@pytest.mark.parametrize("overload", [False, True])
@pytest.mark.parametrize("parameter", [False, True])
def test_inlined_sdk_rmsnorm_preserves_weight_storage_and_values(overload, parameter):
    root = torch.nn.Module()
    weight = torch.linspace(0.5, 1.5, 16)
    if parameter:
        root.register_parameter("scale", torch.nn.Parameter(weight))
    else:
        root.register_buffer("scale", weight)
    graph = torch.fx.Graph()
    inputs = graph.placeholder("inputs")
    scale = graph.get_attr("scale")
    target = torch.ops.xh.RMSNorm.default if overload else torch.ops.xh.RMSNorm
    output = graph.call_function(target, (inputs, scale, 1e-5))
    graph.output(output)
    model = torch.fx.GraphModule(root, graph)
    values = torch.randn(2, 3, 16)
    expected = model(values)
    assert len(_normalize_frontend_rmsnorm_modules(model)) == 1
    norm, = [module for module in model.modules() if isinstance(module, xhnn.RMSNorm)]
    assert norm.weight is model.scale
    assert norm.variance_epsilon == 1e-5
    torch.testing.assert_close(model(values), expected, rtol=0, atol=0)
    assert _normalize_frontend_rmsnorm_modules(model) == ()


def test_inlined_sdk_rmsnorm_rejects_dynamic_weight():
    graph = torch.fx.Graph()
    inputs, scale = graph.placeholder("inputs"), graph.placeholder("scale")
    output = graph.call_function(torch.ops.xh.RMSNorm, (inputs, scale, 1e-5))
    graph.output(output)
    model = torch.fx.GraphModule(torch.nn.Module(), graph)
    with pytest.raises(ValueError, match="static get_attr"):
        _normalize_frontend_rmsnorm_modules(model)


@pytest.mark.parametrize("overload", [False, True])
def test_inlined_sdk_rope_keeps_fused_operator_and_values(overload):
    graph = torch.fx.Graph()
    inputs = tuple(graph.placeholder(name) for name in ("inputs", "cos", "sin"))
    target = torch.ops.xh.Rope.default if overload else torch.ops.xh.Rope
    output = graph.call_function(target, inputs)
    graph.output(output)
    model = torch.fx.GraphModule(torch.nn.Module(), graph)
    values = torch.randn(1, 2, 3, 8), torch.randn(1, 1, 3, 8), torch.randn(1, 1, 3, 8)
    expected = model(*values)
    assert len(_normalize_frontend_rope_modules(model)) == 1
    assert any(isinstance(module, xhnn.Rope) for module in model.modules())
    torch.testing.assert_close(model(*values), expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["linear", "moe"])
def test_gptq_dynamo_workaround_restores_packed_linear_call_module(kind) -> None:
    from xhmodel_merak.xh_llm.models.qwen4_exp.hmonnx_export import _isolate_frontend_module_ownership
    from xhquant.api import FrontendType, to_frontend_graph

    width = 8 if kind == "linear" else 64

    packed_weight = xhnn.GPTQPackedWeight(
        qweight=torch.zeros((8, width), dtype=torch.int32),
        qzeros=torch.zeros((1, width // 8), dtype=torch.int32),
        scales=torch.ones((1, width), dtype=torch.float16),
        g_idx=torch.zeros((64,), dtype=torch.int32),
        bits=4,
        group_size=64,
        in_features=64,
        out_features=width,
        qzero_format=2,
    )

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = (
                xhnn.GPTQPackedLinear(packed_weight) if kind == "linear" else xhnn.GPTQPackedMoeBlock(
                    activation_type="silu", k=1, gate_proj=[packed_weight] * 2,
                    up_proj=[packed_weight] * 2, down_proj=[packed_weight] * 2,
                )
            )
            self.register_buffer("routing", torch.full((2, 2), 0.5, dtype=torch.float16))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.projection(inputs) if kind == "linear" else self.projection(inputs, self.routing)

    model = TinyModel().eval()
    sample = torch.zeros((2, 64), dtype=torch.float16)
    with _opaque_gptq_modules_for_dynamo(model) as (originals, kinds):
        frontend = to_frontend_graph(model, FrontendType.DynamoFX, (sample,), enable_fuse=False)

    restored = _restore_gptq_modules_after_dynamo(frontend, originals, kinds)

    assert restored
    packed_nodes = [node for node in frontend.graph.nodes if node.op == "call_module"]
    assert len(packed_nodes) == 1
    restored_projection = frontend.get_submodule(str(packed_nodes[0].target))
    expected_class = xhnn.GPTQPackedLinear if kind == "linear" else xhnn.GPTQPackedMoeBlock
    assert isinstance(restored_projection, expected_class)
    assert restored_projection is not model.projection
    assert isinstance(model.projection, expected_class)
    before = dict(restored_projection.named_buffers(remove_duplicate=False))
    assert before and all(any(part in name for part in ("qweight", "qzeros", "scales", "g_idx"))
                          for name in before)
    snapshots = {name: value.clone() for name, value in before.items()}
    isolated = _isolate_frontend_module_ownership(frontend)
    owned_projection = isolated.get_submodule(str(packed_nodes[0].target))
    assert owned_projection is not restored_projection
    owned_weights = [module for module in owned_projection.modules() if isinstance(module, xhnn.GPTQPackedWeight)]
    original_weights = [module for module in restored_projection.modules() if isinstance(module, xhnn.GPTQPackedWeight)]
    assert len(owned_weights) == len(original_weights) > 0
    assert all(owned is not original for owned, original in zip(owned_weights, original_weights, strict=True))
    for name, value in owned_projection.named_buffers(remove_duplicate=False):
        assert value is not before[name] and value.data_ptr() == before[name].data_ptr()
        assert value.dtype == before[name].dtype
    owned_projection.float()
    owned_projection.__class__ = torch.nn.Identity
    if kind == "linear":
        owned_projection.packed_weight = None
    else:
        owned_projection.expert_gate_proj_packed = torch.nn.ModuleList()
    assert type(restored_projection) is type(model.projection) is expected_class
    for name, value in restored_projection.named_buffers(remove_duplicate=False):
        assert value is before[name]
        torch.testing.assert_close(value, snapshots[name], rtol=0, atol=0)
    assert packed_weight.scales.dtype == torch.float16



def test_frontend_ptq_ownership_borrows_storage_without_mutating_source():
    from xhmodel_merak.xh_llm.models.qwen4_exp.hmonnx_export import _isolate_frontend_module_ownership

    source = torch.nn.Module()
    source.linear = torch.nn.Linear(4, 8).half()
    source.embedding = torch.nn.Embedding(8, 4).half()
    source.embedding.weight = source.linear.weight
    source.register_buffer("left", torch.arange(4))
    source.register_buffer("right", source.left)
    original = source.linear.weight.detach().clone()
    pointer = source.linear.weight.data_ptr()
    frontend = torch.nn.Module()
    frontend.linear = source.linear
    frontend.embedding = source.embedding
    frontend.register_buffer("left", source.left)
    frontend.register_buffer("right", source.right)
    owned = _isolate_frontend_module_ownership(frontend)
    assert owned.linear is not source.linear
    assert owned.linear.weight is not source.linear.weight
    assert owned.linear.weight is owned.embedding.weight
    assert owned.linear.weight.data_ptr() == pointer
    assert owned.left is owned.right and owned.left is not source.left
    assert owned.left.data_ptr() == source.left.data_ptr()
    owned.linear.float()  # Module.to must not rebind the source Parameter's data.
    owned.linear.weight = torch.nn.Parameter(torch.empty(8, 4, device="meta"))
    assert source.linear.weight.dtype == torch.float16 and source.linear.weight.data_ptr() == pointer
    torch.testing.assert_close(source.linear.weight, original, rtol=0, atol=0)


def test_dynamo_export_compatibility_scope_restores_cuda_query() -> None:
    original = torch.cuda.is_current_stream_capturing
    with _dynamo_export_compatibility_scope():
        assert torch.cuda.is_current_stream_capturing() is False
    assert torch.cuda.is_current_stream_capturing is original


def test_functional_linear_normalizer_reifies_regular_linear() -> None:
    from torch.fx import symbolic_trace

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(4, 3, bias=True)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.linear(inputs, self.projection.weight, self.projection.bias)

    traced = symbolic_trace(TinyModel().eval())
    # symbolic_trace emits a functional linear with get_attr weight/bias.
    assert any(node.op == "call_function" and node.target is torch.nn.functional.linear for node in traced.graph.nodes)
    normalized = _normalize_functional_linear_modules(traced)

    assert normalized
    linear_nodes = [node for node in traced.graph.nodes if node.op == "call_module"]
    assert len(linear_nodes) == 1
    module = traced.get_submodule(str(linear_nodes[0].target))
    assert isinstance(module, xhnn.XHLinear)
    sample = torch.randn(2, 4)
    torch.testing.assert_close(traced(sample), torch.nn.functional.linear(sample, module.weight, module.bias))


def test_dynamic_slice_normalizer_reifies_custom_op() -> None:
    from torch.fx import Graph, GraphModule

    graph = Graph()
    data = graph.placeholder("data")
    starts = graph.placeholder("starts")
    sliced = graph.call_function(torch.ops.xh.DynamicSlice.default, (data, starts, [4], [2], [1]))
    graph.output(sliced)
    traced = GraphModule(torch.nn.Module(), graph)

    normalized = _normalize_frontend_dynamic_slice_modules(traced)

    assert normalized == ("dynamic_slice_default",)
    slice_nodes = [node for node in traced.graph.nodes if node.op == "call_module"]
    assert len(slice_nodes) == 1
    assert isinstance(traced.get_submodule(str(slice_nodes[0].target)), xhnn.DynamicSlice)


def test_llm_cache_normalizer_reifies_custom_op() -> None:
    from torch.fx import Graph, GraphModule

    graph = Graph()
    states = graph.placeholder("states")
    past_seq_length = graph.placeholder("past_seq_length")
    current_seq_length = graph.placeholder("current_seq_length")
    past_cache = graph.placeholder("past_cache")
    cached = graph.call_function(
        torch.ops.xh.LLMCache.default,
        (states, past_seq_length, current_seq_length, past_cache, 2, -1, False),
    )
    graph.output(cached)
    traced = GraphModule(torch.nn.Module(), graph)

    normalized = _normalize_frontend_llm_cache_modules(traced)

    assert normalized == ("llmcache_default",)
    cache_nodes = [node for node in traced.graph.nodes if node.op == "call_module"]
    assert len(cache_nodes) == 1
    assert isinstance(traced.get_submodule(str(cache_nodes[0].target)), xhnn.LLMCacheV2)


@pytest.mark.parametrize("channels,groups,kernel,padding,dilation", [(2, 1, 3, 0, 1), (4, 4, 4, 3, 1), (4, 4, 4, 0, 3)])
def test_functional_conv1d_normalizer_reifies_weighted_conv(channels, groups, kernel, padding, dilation) -> None:
    from torch.fx import Graph, GraphModule

    from xhquant.frontend.fx_transform.conv1d_to_conv2d import Conv1dToConv2d

    root = torch.nn.Module()
    root.register_parameter("weight", torch.nn.Parameter(torch.randn(4, channels // groups, kernel)))
    root.register_parameter("bias", torch.nn.Parameter(torch.randn(4)))
    graph = Graph()
    inputs = graph.placeholder("inputs")
    weight = graph.get_attr("weight")
    bias = graph.get_attr("bias")
    convolved = graph.call_function(torch.conv1d, (inputs, weight, bias, 1, padding, dilation, groups))
    graph.output(convolved)
    traced = GraphModule(root, graph)
    normalized = _normalize_functional_conv1d_modules(traced)
    assert normalized == ("conv1d",)
    conv_nodes = [node for node in traced.graph.nodes if node.op == "call_module"]
    assert len(conv_nodes) == 1
    module = traced.get_submodule(str(conv_nodes[0].target))
    assert isinstance(module, torch.nn.Conv1d)
    sample = torch.randn(2, channels, 15)
    expected = torch.conv1d(sample, root.weight, root.bias, 1, padding, dilation, groups)
    traced = Conv1dToConv2d()(traced, (sample,))
    assert not any(isinstance(module, torch.nn.Conv1d) for module in traced.modules())
    torch.testing.assert_close(traced(sample), expected, rtol=0, atol=0)


@pytest.mark.parametrize("target", [torch.ops.xh.Gather, torch.ops.xh.Gather.default])
def test_frontend_gather_overload_keeps_int32_index_semantics(target) -> None:
    from torch.fx import Graph, GraphModule

    graph = Graph()
    values, indices = graph.placeholder("values"), graph.placeholder("indices")
    gathered = graph.call_function(target, (values, 0, indices))
    graph.output(gathered)
    module = GraphModule(torch.nn.Module(), graph)
    args = gathered.args
    changed = _normalize_frontend_gather_overloads(module)
    assert changed == (gathered.name,)
    replacement = next(node for node in module.graph.nodes if node.op == "call_module")
    assert isinstance(module.get_submodule(replacement.target), xhnn.Gather)
    assert replacement.args == (args[0], args[2])
    inputs = torch.arange(24).reshape(6, 4).float()
    index = torch.tensor([4, 1, 5], dtype=torch.int32)
    torch.testing.assert_close(module(inputs, index), inputs[index], rtol=0, atol=0)
    assert index.dtype == torch.int32


def test_quantized_moe_normalizer_moves_fast_mode_to_keyword() -> None:
    from torch.fx import Graph, GraphModule

    class XH2aQuantQGPTQPackedMoeBlock(torch.nn.Module):
        def forward(self, hidden_states, routing_weights, selected_experts=None, **kwargs):
            return hidden_states

    root = torch.nn.Module()
    root.add_module("moe", XH2aQuantQGPTQPackedMoeBlock())
    graph = Graph()
    hidden_states = graph.placeholder("hidden_states")
    routing_weights = graph.placeholder("routing_weights")
    moe = graph.call_module("moe", (hidden_states, routing_weights, None, True))
    graph.output(moe)
    traced = GraphModule(root, graph)
    normalized = _normalize_quantized_moe_call_signatures(traced)

    assert normalized == ("moe",)
    moe_node = next(node for node in traced.graph.nodes if node.op == "call_module")
    assert len(moe_node.args) == 3
    assert moe_node.kwargs["fast_mode"] is True


def test_ple_gather_sharding_reuses_external_bytes_and_preserves_outputs(tmp_path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnx.reference import ReferenceEvaluator

    weights = np.arange(16, dtype=np.float16).reshape(8, 2)
    weight_name = "model.layers.1.ple.ple_embedding.ngram_embedding_weight"
    graph = helper.make_graph(
        [helper.make_node("Gather", [weight_name, "indices"], ["output"], name="ple_gather", axis=0)],
        "ple_gather",
        [helper.make_tensor_value_info("indices", TensorProto.INT32, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 4, 2])],
        [numpy_helper.from_array(weights, name=weight_name)],
    )
    source_path = tmp_path / "source.onnx"
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
        source_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="source.onnx.data",
        size_threshold=0,
    )
    source_data_path = tmp_path / "source.onnx.data"
    source_stat = source_data_path.stat()
    output_path = tmp_path / "sharded.onnx"

    result = shard_qwen4_exp_ple_gather(source_path, output_path, shard_count=4)

    assert result.shard_rows == (2, 2, 2, 2)
    assert result.shard_nbytes == (8, 8, 8, 8)
    assert source_data_path.stat().st_ino == source_stat.st_ino
    assert source_data_path.stat().st_size == source_stat.st_size

    metadata = onnx.load(output_path, load_external_data=False)
    initializers = {initializer.name: initializer for initializer in metadata.graph.initializer}
    assert weight_name not in initializers
    shard_initializers = [
        initializer for name, initializer in initializers.items() if name.startswith(f"{weight_name}.row_shard_")
    ]
    assert [list(initializer.dims) for initializer in shard_initializers] == [[2, 2]] * 4
    assert [node.op_type for node in metadata.graph.node].count("Gather") == 4
    assert [node.op_type for node in metadata.graph.node].count("Clip") == 0
    assert [node.op_type for node in metadata.graph.node].count("Where") == 7
    tags = [node for node in metadata.graph.node if node.op_type == "Tag"]
    assert len(tags) == 4
    assert [node.domain for node in tags] == ["ai.houmo.xh2a"] * 4

    original = onnx.load(source_path, load_external_data=True)
    sharded = onnx.load(output_path, load_external_data=True)
    for node in sharded.graph.node:
        if node.op_type == "Tag":
            node.op_type = "Identity"
            node.domain = ""
            del node.attribute[:]
    indices = np.asarray([[0, 2, 5, 7]], dtype=np.int32)
    expected = ReferenceEvaluator(original).run(None, {"indices": indices})[0]
    actual = ReferenceEvaluator(sharded).run(None, {"indices": indices})[0]
    np.testing.assert_array_equal(actual, expected)


def test_ple_shard_placement_tags_group_adjacent_shards(tmp_path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight_name = "model.layers.1.ple.ple_embedding.ngram_embedding_weight"
    source_path = tmp_path / "source.onnx"
    sharded_path = tmp_path / "sharded.onnx"
    grouped_path = tmp_path / "grouped.onnx"
    graph = helper.make_graph(
        [helper.make_node("Gather", [weight_name, "indices"], ["output"], name="ple_gather", axis=0)],
        "ple_gather",
        [helper.make_tensor_value_info("indices", TensorProto.INT32, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 4, 2])],
        [numpy_helper.from_array(np.arange(16, dtype=np.float16).reshape(8, 2), name=weight_name)],
    )
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
        source_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="group_source.onnx.data",
        size_threshold=0,
    )
    shard_qwen4_exp_ple_gather(source_path, sharded_path, shard_count=4)

    result = group_qwen4_exp_ple_shard_placement_tags(
        sharded_path,
        grouped_path,
        placement_group_size=2,
    )
    grouped = onnx.load(grouped_path, load_external_data=False)
    tag_types = []
    for node in grouped.graph.node:
        if node.op_type == "Tag":
            attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
            tag_types.append(attributes["tag_type"])

    assert result.shard_count == 4
    assert result.llm_boundary_shards == (1, 3)
    assert tag_types == [b"PLE", b"LLM", b"PLE", b"LLM"]


def test_gptq_export_device_map_keeps_oversized_ple_table_on_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    config = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=5, ple_layer_ids=[2]),
    )

    device_map = _build_qwen4_exp_gptq_export_device_map(config)

    assert device_map[""] == "cpu"
    assert device_map["model.language_model.layers.0"] == "cuda:0"
    assert device_map["model.language_model.layers.1"] == "cpu"
    assert device_map["model.language_model.layers.2"] == "cuda:2"
    assert "model.language_model.layers.1.ple.ple_embedding.ngram_embedding" not in device_map
    assert device_map["model.visual"] == "cuda:2"


def test_preflight_accepts_ple_qsa_and_hyperconnection_architecture() -> None:
    capability = preflight_qwen4_exp_hmonnx_export(_flash_next_text_config(), require_hmonnx_runtime=False)

    assert capability.supported is True
    assert capability.has_transformers_qwen4_exp is True
    assert capability.reasons == ()
    assert capability.details["has_ple_layers"] is True
    assert capability.details["has_qsa_indexer_layers"] is True
    assert capability.cache_contract is not None


def test_export_writes_preflight_before_refusing_unknown_layer_type(tmp_path: Path) -> None:
    model = torch.nn.Linear(32, 32)
    config = _flash_next_text_config()
    config.layer_types = [LINEAR_ATTENTION, "unsupported_attention"]

    with pytest.raises(Qwen4ExpExportUnsupportedError, match="unsupported layer_types"):
        export_qwen4_exp_hmonnx(
            model,
            tmp_path,
            text_config=config,
            sample_inputs=(),
            convert_to_hmonnx=False,
        )

    preflight = tmp_path / "qwen4_exp_hmonnx_preflight.json"
    data = json.loads(preflight.read_text(encoding="utf-8"))
    assert data["supported"] is False
    assert any("unsupported layer_types" in reason for reason in data["reasons"])


def test_gated_delta_net_uses_runtime_valid_length(monkeypatch) -> None:
    captured = {}

    def fake_forward(
        _module,
        hidden_states,
        *,
        conv_cache,
        recurrent_state,
        linear_attn_mask,
        current_input_length,
    ):
        captured["current_input_length"] = current_input_length
        return hidden_states, conv_cache, recurrent_state

    monkeypatch.setattr(HybridGatedDeltaNetMixin, "forward", fake_forward)
    conv_cache = torch.zeros((1, 4, 3))
    recurrent_state = torch.zeros((1, 2, 4, 4))
    cache = SimpleNamespace(
        current_input_length=torch.tensor([3], dtype=torch.int32),
        layers=[SimpleNamespace(conv_states=[conv_cache], recurrent_states=[recurrent_state])],
    )

    Qwen4ExpExportableGatedDeltaNet.forward(
        SimpleNamespace(layer_idx=0),
        torch.zeros((1, 64, 8)),
        cache_params=cache,
    )

    assert captured["current_input_length"] is cache.current_input_length


def test_frontend_sanitizer_materializes_functional_lm_head() -> None:
    from xhquant.quantization.xh2a.builder import XH2A_QMODULES

    root = torch.nn.Module()
    root.register_parameter("self___lm_head__parameters__weight", torch.nn.Parameter(torch.randn(5, 8)))
    graph = torch.fx.Graph()
    hidden_states = graph.placeholder("hidden_states")
    weight = graph.get_attr("self___lm_head__parameters__weight")
    logits = graph.call_function(torch._C._nn.linear, (hidden_states, weight, None))
    graph.output(logits)
    module = torch.fx.GraphModule(root, graph).eval()
    sample = torch.randn(1, 2, 8)
    reference = module(sample)

    removed = sanitize_qwen4_exp_frontend_graph(module)

    leaf_nodes = [node for node in module.graph.nodes if node.op == "call_module"]
    assert removed == (logits.name,)
    assert len(leaf_nodes) == 1
    assert leaf_nodes[0].target == "lm_head"
    leaf_module = module.get_submodule(leaf_nodes[0].target)
    assert type(leaf_module) is xhnn.XHLinear
    assert XH2A_QMODULES.get(type(leaf_module)) is not None
    assert leaf_module.weight is root.self___lm_head__parameters__weight
    torch.testing.assert_close(module(sample), reference)


def test_frontend_sanitizer_functionalizes_cache_copy_and_removes_grad_state() -> None:
    graph = torch.fx.Graph()
    target = graph.placeholder("target")
    source = graph.placeholder("source")
    copied = graph.call_method("copy_", (target, source))
    grad_state = graph.call_function(torch._C._set_grad_enabled, (False,))
    graph.output(copied)
    module = torch.fx.GraphModule(torch.nn.Module(), graph)

    removed = sanitize_qwen4_exp_frontend_graph(module)

    assert set(removed) == {copied.name, grad_state.name}
    assert module(torch.tensor([1.0]), torch.tensor([2.0])).item() == 2.0


def test_frontend_sanitizer_materializes_where_scalar_as_tensor() -> None:
    from xhquant.nn.modules.onnx_style_modules import Where

    root = torch.nn.Module()
    root.where = Where()
    graph = torch.fx.Graph()
    condition = graph.placeholder("condition")
    value = graph.placeholder("value")
    selected = graph.call_module("where", (condition, value, 17))
    graph.output(selected)
    module = torch.fx.GraphModule(root, graph)

    sanitize_qwen4_exp_frontend_graph(module)

    where_node = next(node for node in module.graph.nodes if node.op == "call_module")
    assert isinstance(where_node.args[2], torch.fx.Node)
    torch.testing.assert_close(
        module(torch.tensor([True, False]), torch.tensor([3, 4])),
        torch.tensor([3, 17]),
    )


def test_portable_graph_removes_device_transfer_but_preserves_dtype_cast() -> None:
    class To(torch.nn.Module):
        def forward(self, value: torch.Tensor, target: object) -> torch.Tensor:
            return value.to(target)

    root = torch.nn.Module()
    root.device_to = To()
    root.dtype_to = To()
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    value.meta["val"] = torch.empty(1, dtype=torch.float16)
    device_copy = graph.call_module("device_to", (value, torch.device("cpu")))
    device_copy.meta["val"] = torch.empty(1, dtype=torch.float16)
    dtype_cast = graph.call_module("dtype_to", (device_copy, torch.float32))
    dtype_cast.meta["val"] = torch.empty(1, dtype=torch.float32)
    graph.output(dtype_cast)
    module = torch.fx.GraphModule(root, graph)

    removed = _remove_device_only_transfer_nodes(module)

    assert removed == (device_copy.name,)
    assert module(torch.tensor([1.0], dtype=torch.float16)).dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device placement")
def test_graph_constants_follow_their_sharded_consumer_device() -> None:
    root = torch.nn.Module()
    root.register_buffer("constant", torch.tensor([2.0]))
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    value.meta["val"] = torch.empty(1, device="cuda:0")
    constant = graph.get_attr("constant")
    copied_constant = graph.call_method("clone", (constant,))
    result = graph.call_function(torch.add, (value, copied_constant))
    graph.output(result)
    module = torch.fx.GraphModule(root, graph)

    relocated = _align_graph_constants_with_consumers(module)

    assert relocated == ("constant",)
    assert module.constant.device == torch.device("cuda:0")






def test_golden_runner_casts_inputs_and_writes_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeInfo:
        def __init__(self, dtype: torch.dtype):
            self.dtype = dtype

    class FakeSession:
        def __init__(self, path: str):
            self.path = path
            self.exec_device = None
            self.save_golden = False
            self.golden_dir = None
            self.step = None

        def initialize(self) -> None:
            pass

        def get_input_names(self) -> list[str]:
            return ["tokens", "state"]

        def get_input(self, name: str) -> FakeInfo:
            return FakeInfo(torch.int32 if name == "tokens" else torch.float16)

        def get_output_names(self) -> list[str]:
            return ["hidden", "present_state"]

        def run(self, feed):
            assert feed["tokens"].dtype == torch.int32
            assert feed["state"].dtype == torch.float16
            return feed["state"] + 1, feed["state"]

    import xhquant.api

    monkeypatch.setattr(xhquant.api, "HMONNXGoldenInference", FakeSession)
    outputs = run_qwen4_exp_hmonnx_golden(
        tmp_path / "model.onnx",
        (torch.tensor([1], dtype=torch.int64), torch.tensor([2.0], dtype=torch.float32)),
        tmp_path / "golden",
    )

    assert tuple(outputs) == ("hidden", "present_state")
    summary = json.loads((tmp_path / "golden/step_0/qwen4_exp_runtime_summary.json").read_text())
    assert [item["dtype"] for item in summary["inputs"]] == ["int32", "float16"]


@pytest.mark.parametrize("with_ple", [False, True, "packed"], ids=["gdr_conv", "gdr_and_ple_conv", "packed_gdr_conv"])
@pytest.mark.parametrize("verify_width", [2, 4])
def test_verify_recurrent_graph_matches_sdk_decode_steps(tmp_path, monkeypatch, with_ple, verify_width):
    import numpy as np
    import onnx
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    from xhmodel_merak.exec_backends import HMONNXInferenceConfig
    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_runtime import Qwen4ExpHostPLEInferenceV2
    from xhmodel_merak.xh_llm.models.qwen4_exp.xhquant_extensions import qwen4_exp_xhquant_extension_scope

    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    monkeypatch.setattr(torch._dynamo.config, "disable", False)
    packed_projection = with_ple == "packed"
    with_ple = with_ple is True
    torch.manual_seed(53)
    config = Qwen4ExpTextConfig(
        max_position_embeddings=16,
        vocab_size=32, hidden_size=64 if packed_projection else 16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, indexer_head_dim=8, linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_conv_kernel_dim=4,
        moe_intermediate_size=8, shared_expert_intermediate_size=8, num_experts=2,
        num_experts_per_tok=1, hc_count=2, hc_lowrank=4, layer_types=["linear_attention"],
        indexer_n_heads=2, indexer_kv_heads=1, indexer_budget=4, indexer_compress_ratio=2,
        ple_layer_ids=[], rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5},
    )
    if with_ple:
        config.ple_layer_ids = [1]
        config.ple_embed_dim = 64
        config.ple_conv_kernel_size = 4
        config.ngram_size = 3
        config.heads_per_ngram = 1
        config.ngram_vocab_size_base = 5
        config.make_ngram_vocab_size_divisible_by = 8
        config.split_ngram_parts = 1
        config.eos_token_id = 0
    native = Qwen4ExpForCausalLM(config).half().eval()
    if packed_projection:
        native.model.layers[0].linear_attn.in_proj_qkv = xhnn.GPTQPackedLinear(xhnn.GPTQPackedWeight(
            qweight=torch.full((8, 24), 0x11111111, dtype=torch.int32),
            qzeros=torch.zeros(1, 3, dtype=torch.int32), scales=torch.full((1, 24), 0.125).half(),
            g_idx=torch.zeros(64, dtype=torch.int32), bits=4, group_size=64,
            in_features=64, out_features=24, qzero_format=2,
        ))
    retained = {name: (parameter, parameter.data_ptr(), parameter.detach().clone())
                for name, parameter in native.named_parameters()
                if "hyper_connection" in name or name == "lm_head.weight" or "ngram_embedding" in name}
    retained_classes = {name: type(module) for name, module in native.named_modules()
                        if "hyper_connection" in name and isinstance(module, torch.nn.Linear)}
    sessions, samples = {}, {}
    for width in (verify_width, 1):
        model = native
        _, _, sample = build_qwen4_exp_serving_sample_inputs(model, input_sequence_length=width, context_length=8)
        exported = export_qwen4_exp_hmonnx(
            model, tmp_path / f"seq{width}", text_config=config, sample_inputs=sample,
            input_sequence_length=width, cache_length=8, quant_device="cpu",
            quant_config={"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True},
            use_fused_gdr=True, linear_attention_mode="recurrent", return_hidden_states=True, ple_quant_type="fp16",
        )
        current = dict(native.named_parameters())
        for name, (owner, pointer, original) in retained.items():
            assert current[name] is owner and owner.device.type == "cpu" and owner.data_ptr() == pointer
            torch.testing.assert_close(owner, original, rtol=0, atol=0)
        assert type(native.lm_head) is torch.nn.Linear
        current_modules = dict(native.named_modules())
        for name, cls in retained_classes.items():
            assert type(current_modules[name]) is cls
        graph = onnx.load(exported.hmonnx_path, load_external_data=False)
        assert len(json.loads(Path(exported.io_manifest_path).read_text())["removed_gdr_projection_roundtrips"]) == 1
        # Default SDK optimization removes internal aliases while public
        # output aliases may be needed to preserve the serving ABI.
        public_outputs = {value.name for value in graph.graph.output}
        assert all(
            set(node.output) <= public_outputs
            for node in graph.graph.node if node.op_type in {"Identity", "HMIdentity"}
        )
        assert not any(node.op_type == "Conv1d" for node in graph.graph.node)
        convolutions = [node for node in graph.graph.node if node.op_type == "Conv2d"]
        assert len(convolutions) == 1 + int(with_ple)
        geometry = [{attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
                    for node in convolutions]
        assert any(item["dilations"] == [1, 1] and item["pads"] == [3, 0] for item in geometry)
        producers = {name: node for node in graph.graph.node for name in node.output}
        skip = {"Cast", "Transpose", "Reshape", "Unsqueeze", "Squeeze", "Identity", "HMIdentity"}

        def data_source(name, producers=producers, skip=skip):
            while name in producers and producers[name].op_type in skip:
                name = producers[name].input[0]
            return producers[name]

        gdr_conv = next(node for node, attrs in zip(convolutions, geometry, strict=True) if attrs["pads"] == [3, 0])
        temporal_concat = data_source(gdr_conv.input[0])
        assert temporal_concat.op_type == "Concat" and len(temporal_concat.input) == 2
        assert data_source(temporal_concat.input[1]).op_type == "Linear"
        assert sum(node.op_type == "Split" for node in graph.graph.node) == 1  # Required post-convolution Q/K/V split.
        if with_ple:
            # The SDK's initial frontend pass may expand dilation into zero
            # taps before our post-normalization pass sees this module.
            ple_node = next(node for node, item in zip(convolutions, geometry, strict=True)
                            if item["pads"] == [0, 0])
            ple_geometry = {attr.name: onnx.helper.get_attribute_value(attr) for attr in ple_node.attribute}
            # SDK versions may retain native dilation or expand it into
            # zero taps. Both must describe the same ten-position PLE window.
            dilation, width_dilation = ple_geometry["dilations"]
            kernel, kernel_width = ple_geometry["kernel_shape"]
            assert width_dilation == kernel_width == 1
            assert (kernel - 1) * dilation + 1 == 10
            weight = next(value for value in graph.graph.initializer if value.name == ple_node.input[3])
            taps = onnx.numpy_helper.to_array(weight, base_dir=str(Path(exported.hmonnx_path).parent))
            expanded = np.zeros((*taps.shape[:2], 10, 1), dtype=taps.dtype)
            expanded[:, :, ::dilation, :] = taps
            assert not expanded[:, :, [1, 2, 4, 5, 7, 8], :].any()
        assert sum(node.op_type == "GDRRecurrentScan" for node in graph.graph.node) == width
        assert not any(node.op_type in {"GDRChunkScan", "GDRBlockTriInverse"} for node in graph.graph.node)
        with qwen4_exp_xhquant_extension_scope():
            session = Qwen4ExpHostPLEInferenceV2(str(exported.hmonnx_path), HMONNXInferenceConfig(exec_devices=["cpu"]))
            session.initialize()
        sessions[width] = session
        samples[width] = dict(zip(session.get_input_names(), sample, strict=True))
    embeds = torch.randn(1, verify_width + 1, config.hidden_size).half()
    token_ids = torch.arange(3, verify_width + 4, dtype=torch.int32).unsqueeze(0)
    from xhmodel_merak.xh_llm.mrope import MROPE_CURRENT_POSITION_NAMES, MROPE_POOL_POSITION_NAMES

    data_names = {"inputs_embeds", "past_seq_length", "current_input_length", "ple_input_ids",
                  *MROPE_CURRENT_POSITION_NAMES, *MROPE_POOL_POSITION_NAMES}
    state_names = [name for name in samples[1] if name not in data_names]
    initial = {name: value.clone() for name, value in samples[1].items() if name in state_names}
    initial['delta_recurrent_state_0'] = torch.randn_like(initial['delta_recurrent_state_0']) * 0.1
    def execute(width, offset, states, valid_length=None):
        valid_length = width if valid_length is None else valid_length
        data = dict(samples[width], **states)
        data.update(inputs_embeds=embeds[:, offset:offset + width],
                    past_seq_length=torch.tensor([offset], dtype=torch.int32),
                    current_input_length=torch.tensor([valid_length], dtype=torch.int32),
                    ple_input_ids=token_ids[:, offset:offset + width])
        for name in MROPE_CURRENT_POSITION_NAMES:
            data[name] = torch.arange(offset, offset + width, dtype=torch.int32).unsqueeze(0)
        for name in MROPE_POOL_POSITION_NAMES:
            data[name] = (torch.arange((width + 3) // 4, dtype=torch.int32) * 4 + offset // 4 * 4).unsqueeze(0)
        # Export needs Dynamo capture, but this regression compares SDK
        # arithmetic/state semantics rather than Inductor's C++ code generation.
        # Scope the runtime-only disable so machines without g++ remain usable.
        with torch._dynamo.config.patch(disable=True):
            result = dict(zip(sessions[width].get_output_names(), sessions[width].run(data), strict=True))
        # Match serving commit_outputs' copy into the original cache dtype.
        committed = {name: result[f'present_{name}'].to(initial[name].dtype).clone() for name in state_names}
        return result, committed

    def assert_states(actual, expected):
        for name in state_names:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    # A width-four verify graph still receives two real positions for the
    # current single-candidate policy. Padding must leave recurrent/conv/PLE
    # state unchanged, including when the next single-token decode consumes it.
    for valid_length in sorted({2, verify_width}):
        batch, batch_state = execute(
            verify_width, 0, {name: value.clone() for name, value in initial.items()}, valid_length,
        )
        sequential_state = {name: value.clone() for name, value in initial.items()}
        sequential = []
        for offset in range(valid_length):
            result, sequential_state = execute(1, offset, sequential_state)
            sequential.append(result)
        for name in ('logits', 'target_hidden_states'):
            torch.testing.assert_close(
                batch[name][:, :valid_length], torch.cat([result[name] for result in sequential], dim=1),
                rtol=0, atol=0,
            )
        assert_states(batch_state, sequential_state)
        after_batch, batch_state = execute(1, valid_length, batch_state)
        after_sequential, sequential_state = execute(1, valid_length, sequential_state)
        for name in ('logits', 'target_hidden_states'):
            torch.testing.assert_close(after_batch[name], after_sequential[name], rtol=0, atol=0)
        assert_states(batch_state, sequential_state)
    sessions.clear()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width,group_size", [(16, None), (10240, 2560)])
def test_qwen4_rmsnorm_preserves_native_grouped_fp32_affine(dtype, width, group_size):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import Qwen4ExpExportableRMSNorm

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(34)
        native = Qwen4ExpTextRMSNorm(width, group_size=group_size, eps=1e-5).to(dtype).eval()
        native.weight.data.uniform_(-0.9, 0.9)
        value = torch.randn(1, 2, width, dtype=dtype)
        if group_size is not None:
            # Distinct HC stream magnitudes reject a whole-10240 reduction.
            value = (value.reshape(1, 2, 4, group_size)
                     * torch.tensor([0.03125, 0.5, 8, 128], dtype=dtype).reshape(1, 1, 4, 1)).flatten(-2)
    expected = native(value)
    adapted = Qwen4ExpExportableRMSNorm(native).eval()
    assert adapted.weight is native.weight
    assert adapted.norm.weight.shape == (group_size or width,)
    assert torch.equal(adapted.norm.weight, torch.ones_like(adapted.norm.weight))
    assert dict(adapted.named_parameters()) == {"weight": native.weight}
    assert set(adapted.state_dict()) == {"weight"}
    torch.testing.assert_close(adapted(value), expected, rtol=0, atol=0)
    assert adapted(value).dtype == dtype
    # Reapplying dtype placement must keep the unit scale harmless too.
    torch.testing.assert_close(adapted.to(dtype)(value), expected, rtol=0, atol=0)


def test_qwen4_rmsnorm_adapter_is_idempotent_and_preserves_checkpoint_layout():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextRMSNorm,
        Qwen4ExpTextRMSNormGated,
    )

    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import (
        Qwen4ExpExportableRMSNorm,
        Qwen4ExpExportableRMSNormGated,
        adapt_qwen4_exp_rms_norms,
    )

    model = torch.nn.ModuleDict({
        "plain": Qwen4ExpTextRMSNorm(8),
        "nested": torch.nn.ModuleList([Qwen4ExpTextRMSNorm(16, group_size=4)]),
        "gated": Qwen4ExpTextRMSNormGated(8),
    }).half().eval()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    parameters = dict(model.named_parameters())
    assert adapt_qwen4_exp_rms_norms(model) == ("plain", "nested.0", "gated")
    assert isinstance(model["plain"], Qwen4ExpExportableRMSNorm)
    assert isinstance(model["gated"], Qwen4ExpExportableRMSNormGated)
    assert adapt_qwen4_exp_rms_norms(model) == ()
    assert set(model.state_dict()) == set(state)
    assert all(dict(model.named_parameters())[name] is value for name, value in parameters.items())
    model.load_state_dict(state, strict=True)
    model.float()
    assert model["gated"].norm.weight is model["gated"].weight
    assert set(model.state_dict()) == set(state)


@pytest.mark.parametrize("group_size,gate", [(None, None), (4, None), (None, "sigmoid"), (None, "silu")])
def test_qwen4_rmsnorm_real_sdk_export_and_replay(tmp_path, group_size, gate):
    import onnx
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm, Qwen4ExpTextRMSNormGated

    from xhmodel_merak.xh_llm.models.qwen4_exp.hmonnx_export import _export_qwen4_exp_hmonnx_program
    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import (
        Qwen4ExpExportableRMSNorm,
        Qwen4ExpExportableRMSNormGated,
    )
    from xhquant.api import (
        ConfigDict,
        FrontendType,
        PrecisionMode,
        ptq_quantize,
        to_export_graph,
        to_frontend_graph,
        to_quant_graph,
    )
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.manual_seed(34)
    native = (Qwen4ExpTextRMSNorm(16, group_size=group_size) if gate is None
              else Qwen4ExpTextRMSNormGated(16, activation=gate)).half().eval()
    native.weight.data.uniform_(-0.75, 0.75)
    adapted = (Qwen4ExpExportableRMSNorm(native) if gate is None
               else Qwen4ExpExportableRMSNormGated(native)).eval()
    sample = (torch.randn(1, 2, 16).half(),)
    if gate is not None:
        sample += (torch.randn_like(sample[0]),)
    with _dynamo_export_compatibility_scope():
        frontend = to_frontend_graph(adapted, FrontendType.DynamoFX, sample, enable_fuse=False)
    norms = [module for module in frontend.modules() if isinstance(module, xhnn.RMSNorm)]
    assert len(norms) == 1 and norms[0].weight.shape == (group_size or 16,)
    quant = to_quant_graph(frontend, "xh2a", ConfigDict({"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True}))
    ptq_quantize(quant, [list(sample)], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    quant.fixed()
    exported = to_export_graph(quant, list(sample))
    path = tmp_path / "norm.onnx"
    _export_qwen4_exp_hmonnx_program(exported, sample, path,
                                   input_names=("values",) if gate is None else ("values", "gate"),
                                   output_names=("normalized",))
    graph = onnx.load(path, load_external_data=False)
    norm, = [node for node in graph.graph.node if node.op_type == "RMSNorm"]
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in norm.attribute}
    assert attrs["axis"] == -1 and attrs["force_fp32"] == 1
    infos = {value.name: value for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
    # XH2 QCast lowers the FP32 reference boundary to its FP16 deployment
    # dtype. The fused operator retains FP32 variance accumulation.
    assert infos[norm.input[0]].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert infos[norm.output[0]].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    weight = next(item for item in graph.graph.initializer if item.name == norm.input[1])
    assert list(weight.dims) == [group_size or 16]
    assert not any(node.op_type in {"Pow", "ReduceMean", "Sqrt"} for node in graph.graph.node)
    session = HMONNXInferenceV2(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))
    # Replay an independent input after tracing, including distinct HC groups.
    value = sample[0] * torch.linspace(0.125, 4, 16).half()
    with torch._dynamo.config.patch(disable=True):
        result = session.run({"values": value, **({"gate": sample[1]} if gate is not None else {})})
    actual = result[0] if isinstance(result, (tuple, list)) else result
    assert actual.dtype == torch.float16
    assert actual.shape == value.shape and torch.isfinite(actual).all()


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_qwen4_gated_rmsnorm_preserves_native_scale_gate_and_checkpoint(dtype, activation):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNormGated

    from xhmodel_merak.xh_llm.models.qwen3_5._hybrid_gated_delta_net import HybridRMSNormGatedMixin
    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import Qwen4ExpExportableRMSNormGated

    torch.manual_seed(68)
    native = Qwen4ExpTextRMSNormGated(16, activation=activation).to(dtype).eval()
    native.weight.data.uniform_(-1.5, 1.5)
    value = torch.randn(1, 3, 2, 16, dtype=dtype)
    value[:, :, 0] *= 0.03125
    value[:, :, 1] *= 16
    gate = torch.randn_like(value) * 2
    adapted = Qwen4ExpExportableRMSNormGated(native)
    assert isinstance(adapted, HybridRMSNormGatedMixin)
    assert adapted.weight is native.weight and adapted.norm.weight is native.weight
    assert set(adapted.state_dict()) == {"weight"}
    torch.testing.assert_close(adapted(value, gate), native(value, gate), rtol=0, atol=0)


def _gdr_projection_roundtrip_fixture(sequence_length, variant="identity"):
    """Executable SDK leaves around the GDR projection and cache convolution."""
    from torch.fx import Graph, GraphModule, Interpreter

    root = torch.nn.Module()
    root.linear_attn = torch.nn.Module()
    sections = [8, 8, 8] if variant == "wrong_axis" else [8, 8, 12]
    channels = sum(sections)
    projection = "in_proj_qkvz" if variant == "packed_name" else "in_proj_qkv"
    setattr(root.linear_attn, projection, torch.nn.Linear(8, channels, bias=False).half())
    root.split = xhnn.Split(sections, axis=-1)
    root.cat = xhnn.Concat(dim=1 if variant == "wrong_axis" else -1)
    root.conv = torch.nn.Conv1d(channels, channels, 4, padding=3, groups=channels, bias=False).half()
    graph = Graph()
    hidden, cache = graph.placeholder("hidden"), graph.placeholder("cache")
    split = None
    if variant != "split_conv":
        mixed = graph.call_module(f"linear_attn.{projection}", (hidden,))
        split = graph.call_module("split", (mixed,))
    lanes, heads = [], []
    for index in range(3):
        root.add_module(f"item{index}", xhnn.GetItem())
        root.add_module(f"heads{index}", xhnn.Reshape())
        root.add_module(f"flat{index}", xhnn.Reshape())
        if variant == "split_conv":
            root.linear_attn.add_module(f"in_proj_{index}", torch.nn.Linear(8, sections[index], bias=False).half())
            item = graph.call_module(f"linear_attn.in_proj_{index}", (hidden,))
        else:
            item = graph.call_module(f"item{index}", (split, index))
        head = graph.call_module(f"heads{index}", (item, (1, sequence_length, sections[index] // 4, 4)))
        heads.append(head)
        if variant == "transpose" and index == 0:
            head = graph.call_method("transpose", (head, -1, -2))
        if variant == "cast" and index == 0:
            head = graph.call_method("float", (head,))
        lanes.append(graph.call_module(f"flat{index}", (head, (1, sequence_length, sections[index]))))
    if variant == "reordered":
        lanes = [lanes[1], lanes[0], lanes[2]]
    cat = graph.call_module("cat", tuple(lanes))
    if variant in {"wrong_axis", "cast"}:
        graph.output(cat)
    else:
        transposed = graph.call_method("transpose", (cat, 1, 2))
        temporal = graph.call_function(torch.cat, ((cache, transposed),), {"dim": -1})
        conv = graph.call_module("conv", (temporal,))
        cache_out = graph.call_function(operator.getitem, (temporal, (..., slice(-4, None))))
        root.post_split = xhnn.Split(sections, axis=1)
        post = graph.call_module("post_split", (conv,))
        graph.output((post, cache_out, heads[0]) if variant == "shared" else (post, cache_out))
    module = GraphModule(root, graph)
    sample = (torch.randn(1, sequence_length, 8).half(), torch.randn(1, channels, 4).half())

    class Metadata(Interpreter):
        def run_node(self, node):
            result = super().run_node(node)
            node.meta["val"] = result
            return result

    Metadata(module).run(*sample)
    return module, sample


@pytest.mark.parametrize("sequence_length", [1, 2, 4], ids=["decode", "verify", "prefill"])
def test_gdr_projection_roundtrip_removal_preserves_conv_outputs_and_cache(sequence_length):
    from xhmodel_merak.xh_llm.models.qwen4_exp.hmonnx_export import _remove_gdr_qkv_projection_roundtrips

    graph, sample = _gdr_projection_roundtrip_fixture(sequence_length)
    expected = graph(*sample)
    before = len(tuple(graph.graph.nodes))
    assert _remove_gdr_qkv_projection_roundtrips(graph) == ("cat",)
    assert len(tuple(graph.graph.nodes)) == before - 11
    assert not any(node.op == "call_module" and node.target == "split" for node in graph.graph.nodes)
    assert any(node.op == "call_module" and node.target == "post_split" for node in graph.graph.nodes)
    torch.testing.assert_close(graph(*sample), expected, rtol=0, atol=0)
    assert _remove_gdr_qkv_projection_roundtrips(graph) == ()


@pytest.mark.parametrize(
    "variant", ["reordered", "shared", "transpose", "cast", "wrong_axis", "packed_name", "split_conv"],
)
def test_gdr_projection_roundtrip_removal_rejects_nonidentity_or_shared_lanes(variant):
    from xhmodel_merak.xh_llm.models.qwen4_exp.hmonnx_export import _remove_gdr_qkv_projection_roundtrips

    graph, sample = _gdr_projection_roundtrip_fixture(2, variant)
    expected, original = graph(*sample), graph.code
    assert _remove_gdr_qkv_projection_roundtrips(graph) == ()
    assert graph.code == original
    torch.testing.assert_close(graph(*sample), expected, rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
def test_gdr_projection_roundtrip_is_removed_after_real_packed_opaque_restore(packed):
    from xhmodel_merak.xh_llm.models.qwen4_exp.hmonnx_export import _remove_gdr_qkv_projection_roundtrips
    from xhquant.api import FrontendType, to_frontend_graph

    class PackedGDRProjection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = torch.nn.Module()
            payload = xhnn.GPTQPackedWeight(
                qweight=torch.zeros(8, 32, dtype=torch.int32),
                qzeros=torch.zeros(1, 4, dtype=torch.int32), scales=torch.ones(1, 32).half(),
                g_idx=torch.zeros(64, dtype=torch.int32), bits=4, group_size=64,
                in_features=64, out_features=32, qzero_format=2,
            )
            self.linear_attn.in_proj_qkv = (xhnn.GPTQPackedLinear(payload) if packed
                                           else torch.nn.Linear(64, 32, bias=False).half())

        def forward(self, hidden):
            mixed = self.linear_attn.in_proj_qkv(hidden)
            q, k, v = mixed.split([8, 8, 16], dim=-1)
            return torch.cat((q.reshape(1, 2, 2, 4).reshape(1, 2, 8),
                              k.reshape(1, 2, 2, 4).reshape(1, 2, 8),
                              v.reshape(1, 2, 4, 4).reshape(1, 2, 16)), dim=-1)

    model = PackedGDRProjection().eval()
    sample = (torch.randn(1, 2, 64).half(),)
    with _dynamo_export_compatibility_scope(), _opaque_gptq_modules_for_dynamo(model) as (originals, kinds):
        frontend = to_frontend_graph(model, FrontendType.DynamoFX, sample, enable_fuse=False)
    _restore_gptq_modules_after_dynamo(
        frontend, originals, kinds, original_module_paths={id(module): path for path, module in model.named_modules()},
    )
    projection, = [node for node in frontend.graph.nodes if node.op == "call_module"
                   and isinstance(frontend.get_submodule(str(node.target)), torch.nn.Linear)]
    if packed:
        assert str(projection.target).startswith("_xh2_gptq_packed_")
        assert projection.meta["qwen4_original_module_path"] == "linear_attn.in_proj_qkv"
    expected = frontend(*sample)
    assert len(_remove_gdr_qkv_projection_roundtrips(frontend)) == 1
    torch.testing.assert_close(frontend(*sample), expected, rtol=0, atol=0)


# RoPE deployment contracts: the SDK owns half-split rotation, while Qwen
# retains its partial rotary prefix and the four-dimensional table layout.
_QWEN4_ROPE_CASES = [
    ((1, 24, 2, 256), (1, 2, 64), 1),
    ((1, 2, 2, 256), (1, 2, 64), 1),
    ((1, 2, 4, 128), (1, 2, 64), 2),
    ((1, 1, 3, 128), (1, 3, 64), 1),
    ((1, 3, 2, 64), (1, 2, 64), 1),
    ((1, 2, 3, 128), (1, 2, 3, 64), None),
]
_QWEN4_ROPE_IDS = ["main_query", "main_key", "index_query", "block_key", "dense", "query_specific_blocks"]


class _Qwen4PartialRopeProbe(torch.nn.Module):
    def __init__(self, table_axis):
        super().__init__()
        self.table_axis = table_axis
        self.rope = xhnn.Rope()

    def forward(self, values, cos, sin):
        from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import _apply_rotary

        return _apply_rotary(values, cos, sin, unsqueeze_dim=self.table_axis, rope=self.rope)


@pytest.mark.parametrize("shape,table_shape,table_axis", _QWEN4_ROPE_CASES, ids=_QWEN4_ROPE_IDS)
def test_qwen4_partial_rope_matches_native_half_split(shape, table_shape, table_axis):
    torch.manual_seed(83)
    values, cos, sin = torch.randn(shape).half(), torch.randn(table_shape).half(), torch.randn(table_shape).half()
    tables = (cos, sin) if table_axis is None else (cos.unsqueeze(table_axis), sin.unsqueeze(table_axis))
    width = table_shape[-1]
    rotary, tail = values[..., :width], values[..., width:]
    expected = torch.cat((rotary * tables[0]
                          + torch.cat((-rotary[..., width // 2:], rotary[..., :width // 2]), -1) * tables[1],
                          tail), -1)
    model = _Qwen4PartialRopeProbe(table_axis)
    torch.testing.assert_close(model(values, cos, sin), expected, rtol=0, atol=0)
    torch.testing.assert_close(model(values, cos, sin)[..., width:], tail, rtol=0, atol=0)
    assert not model.state_dict()


@pytest.mark.parametrize("shape,table_shape,table_axis", _QWEN4_ROPE_CASES, ids=_QWEN4_ROPE_IDS)
def test_qwen4_partial_rope_real_sdk_hmonnx_replay(tmp_path, shape, table_shape, table_axis):
    import onnx

    from xhmodel_merak.xh_llm.models.qwen4_exp.hmonnx_export import _export_qwen4_exp_hmonnx_program
    from xhquant.api import (
        ConfigDict,
        FrontendType,
        PrecisionMode,
        ptq_quantize,
        to_export_graph,
        to_frontend_graph,
        to_quant_graph,
    )
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.manual_seed(83)
    sample = (torch.randn(shape).half(), torch.randn(table_shape).half(), torch.randn(table_shape).half())
    model = _Qwen4PartialRopeProbe(table_axis).eval()
    with _dynamo_export_compatibility_scope():
        frontend = to_frontend_graph(model, FrontendType.DynamoFX, sample, enable_fuse=False)
    assert sum(isinstance(module, xhnn.Rope) for module in frontend.modules()) == 1
    quant = to_quant_graph(frontend, "xh2a", ConfigDict({"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True}))
    ptq_quantize(quant, [list(sample)], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    quant.fixed()
    exported = to_export_graph(quant, list(sample))
    path = tmp_path / "partial_rope.onnx"
    _export_qwen4_exp_hmonnx_program(
        exported, sample, path, input_names=("values", "cos", "sin"), output_names=("rotated",),
    )
    graph = onnx.load(path, load_external_data=False)
    rope, = [node for node in graph.graph.node if node.op_type == "Rope"]
    infos = {value.name: value for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
    assert rope.domain == "ai.houmo.xh2a"
    assert not any(node.op_type in {"Neg", "Mul", "Add"} for node in graph.graph.node)
    for name in rope.input:
        tensor = infos[name].type.tensor_type
        assert tensor.elem_type == onnx.TensorProto.FLOAT16 and len(tensor.shape.dim) == 4
        assert tensor.shape.dim[-1].dim_value == table_shape[-1]
    session = HMONNXInferenceV2(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))
    replay = (sample[0] * 0.5, sample[1] * 0.25, sample[2] * 0.75)
    with torch._dynamo.config.patch(disable=True):
        expected = exported(*replay)
        actual = session.run(dict(zip(("values", "cos", "sin"), replay, strict=True)))
    expected = expected[0] if isinstance(expected, (tuple, list)) else expected
    actual = actual[0] if isinstance(actual, (tuple, list)) else actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[..., table_shape[-1]:], replay[0][..., table_shape[-1]:], rtol=0, atol=0)
