import operator

import pytest
import torch
from torch import nn
from torch.fx import GraphModule, Tracer

from xhmodel_merak.xh_llm.models.qwen3_5._hybrid_gated_delta_net import (
    HybridDecoderLayerMixin,
    _unpack_split_conv_cache_outputs,
)
from xhmodel_merak.xh_llm.models.qwen3_5._qwen3_5_big_export import _Qwen3_5GatedDeltaNet_PlaceHolder
from xhmodel_merak.xh_llm.models.qwen3_5_moe._qwen3_5_moe_big_export import _Qwen3_5MoeGatedDeltaNet_PlaceHolder
from xhquant.api import ConfigDict


def _cfg(mode, steps, suppress):
    return ConfigDict(
        dict(
            use_cache=True,
            split_conv_cache=True,
            linear_attention_mode=mode,
            input_sequence_length=steps,
            verify_output_intermediates=steps > 1 and mode == "recurrent",
            suppress_recurrent_state_outputs=suppress,
            linear_chunk_size=8,
            batch_size=1,
            fuse_gdr_ops=False,
            fuse_gdr_block_recurrent_ops=False,
        )
    )


class _Decoder(HybridDecoderLayerMixin, nn.Module):
    def __init__(self, leaf):
        super().__init__()
        self.layer_type = "linear_attention"
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = nn.Identity()
        self.linear_attn = leaf


class _LeafTracer(Tracer):
    def is_leaf_module(self, module, qualified_name):
        return "PlaceHolder" in type(module).__name__ or super().is_leaf_module(module, qualified_name)


@pytest.mark.parametrize("cls", [_Qwen3_5GatedDeltaNet_PlaceHolder, _Qwen3_5MoeGatedDeltaNet_PlaceHolder])
@pytest.mark.parametrize(
    "mode,steps,suppress", [("chunk", 8, True), ("chunk", 8, False), ("recurrent", 1, False), ("recurrent", 3, False)]
)
def test_real_placeholder_decoder_traces_static_cache_layout(cls, mode, steps, suppress):
    leaf = object.__new__(cls)
    nn.Module.__init__(leaf)
    leaf._setup(_cfg(mode, steps, suppress))
    decoder = _Decoder(leaf)
    graph = _LeafTracer().trace(decoder)
    traced = GraphModule(decoder, graph)
    hidden = torch.ones(1, steps, 8)
    caches = [torch.full((1, 4, 4), float(index)) for index in (1, 2, 3)]
    state = torch.full((1, 2, 2, 2), 4.0)
    actual = traced(hidden, past_conv_cache=caches, past_recurrent_state=state)
    expected = decoder(hidden, past_conv_cache=caches, past_recurrent_state=state)
    torch.testing.assert_close(actual, expected)
    count = steps if mode == "recurrent" else 1
    torch.testing.assert_close(actual[0], hidden * 4)
    for index, cache in enumerate(actual[1:-1]):
        torch.testing.assert_close(cache, caches[index // count])
    if suppress:
        assert actual[-1] is None
    elif count == 1:
        torch.testing.assert_close(actual[-1], state)
    else:
        assert len(actual[-1]) == count
        for snapshot in actual[-1]:
            torch.testing.assert_close(snapshot, state)
    leaf_node = next(node for node in graph.nodes if node.op == "call_module" and node.target == "linear_attn")
    accesses = [node.args[1] for node in leaf_node.users if node.target is operator.getitem]
    assert sorted(accesses) == list(range(1 + 3 * count + (0 if suppress else count)))


def _real_gdn_case(mode, steps, suppress):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    from xhmodel_merak.xh_llm.models.qwen3_5._llm_model_impl import register_wrap_modules
    from xhmodel_merak.xh_llm.wrap_model import wrap_llm_model

    register_wrap_modules()
    config = Qwen3_5TextConfig(
        hidden_size=8,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=4,
    )
    model = Qwen3_5GatedDeltaNet(config, 0).eval()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.copy_(torch.linspace(-0.1, 0.1, parameter.numel()).reshape(parameter.shape) + index * 0.001)
    model = wrap_llm_model(model, _cfg(mode, steps, suppress))
    hidden = torch.linspace(-0.3, 0.3, steps * 8).reshape(1, steps, 8)
    caches = tuple(torch.full((1, 4, 4), 0.01 * index) for index in (1, 2, 3))
    state = torch.full((1, 2, 2, 2), 0.04)
    args = (hidden, caches, state, torch.ones(1, steps), torch.tensor([steps], dtype=torch.int32))
    return model, args


@pytest.mark.parametrize("spec_mode", [None, "mtp", "dflash"])
@pytest.mark.parametrize("draft_tokens", [2, 4])
def test_streamed_decode_replays_main_graph_verify_cache_contract(monkeypatch, spec_mode, draft_tokens):
    from types import SimpleNamespace

    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model
    from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5ModelConfig
    from xhmodel_merak.xh_llm.types import LLMModelState, ModelSwitcher

    monkeypatch.setattr(XHQwen3_5Model, "_get_dflash_target_layer_ids", lambda self: [0])
    model = XHQwen3_5Model(
        XHQwen3_5ModelConfig(
            model_name="qwen3_5",
            spec_decode_mode=spec_mode,
            num_draft_tokens=draft_tokens,
            num_logits_to_keep=1,
            prefill_chunk_length=256,
            split_conv_cache=True,
            fuse_gdr_ops=False,
            fuse_gdr_block_recurrent_ops=False,
        )
    )
    model._data_processor = SimpleNamespace(input_sequence_length=256)
    monkeypatch.setattr(model, "get_data_preprocessor", lambda: model._data_processor)
    model._frontend_model = ModelSwitcher({"prefill": nn.Identity(), "decode": nn.Identity()})
    model._state = LLMModelState.FRONTED
    registered = []
    exporter = SimpleNamespace(register_layer_as_place_holder=registered.append)
    prefill_cfg, decode_cfg = model._prepare_placeholder_graphs(exporter)
    assert registered == [model._frontend_model.prefill, model._frontend_model.decode]
    assert prefill_cfg.input_sequence_length == 256
    assert prefill_cfg.verify_output_intermediates is False
    assert prefill_cfg.linear_attention_mode == "chunk"
    steps = draft_tokens + 1 if spec_mode else 1
    assert model.wrap_cfg.input_sequence_length == steps
    assert model._data_processor.input_sequence_length == steps
    assert model.wrap_cfg.verify_output_intermediates is bool(spec_mode)
    assert model.wrap_cfg.num_logits_to_keep == (0 if spec_mode else 1)
    assert bool(model.wrap_cfg.get("output_post_norm_hidden")) is (spec_mode == "mtp")
    leaf = object.__new__(_Qwen3_5GatedDeltaNet_PlaceHolder)
    nn.Module.__init__(leaf)
    leaf._setup(decode_cfg)
    assert leaf.split_conv_cache_output_layout == (3 * steps, steps)
    # Compare the captured configuration with an actual wrapped GDN, not just
    # output counts manufactured from the placeholder implementation.
    real, args = _real_gdn_case("recurrent", steps, False)
    from xhmodel_merak.xh_llm.wrap_model import wrap_llm_model

    real = wrap_llm_model(real, decode_cfg)
    with torch.no_grad():
        assert len(real.graph_forward(*args)) == 1 + 4 * steps


@pytest.mark.parametrize(
    "mode,steps,suppress", [("chunk", 8, True), ("chunk", 8, False), ("recurrent", 1, False), ("recurrent", 3, False)]
)
def test_real_gdn_graph_outputs_match_eager_values(mode, steps, suppress):
    model, args = _real_gdn_case(mode, steps, suppress)
    with torch.no_grad():
        eager = model(*args)
        flat = model.graph_forward(*args)
    recovered = _unpack_split_conv_cache_outputs(
        flat, owner="test GDN", flat_layout=model.split_conv_cache_output_layout
    )
    torch.testing.assert_close(recovered, eager)
    assert all(isinstance(value, torch.Tensor) for value in flat)
    assert len(flat) == 1 + sum(model.split_conv_cache_output_layout)


@pytest.mark.parametrize("mode,steps,suppress", [("chunk", 8, True), ("recurrent", 1, False), ("recurrent", 3, False)])
def test_real_gdn_layer_export_assembly_and_inference(tmp_path, mode, steps, suppress):
    import onnx

    from xhmodel_merak.xh_llm.big_hf_model_helper import (
        BigHFModelExportHelper,
        _module_name_to_filename,
        replace_placeholder_node,
    )
    from xhmodel_merak.xh_llm.hmonnx.deterministic_export import flatten_placeholder_quant_metadata
    from xhquant.api import (
        HMONNXInference,
        PrecisionMode,
        QuantScheme,
        create_quant_config,
        ptq_quantize,
        to_export_hmonnx_from_quanted_graph,
        to_quant_graph,
    )
    from xhquant.core.graph import FrontendGraph
    from xhquant.nn.modules import PlaceHolderModule

    model, args = _real_gdn_case(mode, steps, suppress)
    count = 1 + sum(model.split_conv_cache_output_layout)
    config = create_quant_config(QuantScheme(target_device="XH2a", quant_type="w16a16_sefp"))
    exporter = object.__new__(BigHFModelExportHelper)
    exporter._export_loaded_place_holder_layer_from_template(
        "block",
        model,
        args,
        "XH2a",
        _cfg(mode, steps, suppress),
        config,
        tmp_path,
        execution_device="cpu",
        expected_outputs=count,
    )
    sub_file = tmp_path / f"{_module_name_to_filename('block')}.onnx"
    sub = onnx.load(sub_file, load_external_data=False)
    assert len(sub.graph.output) == count
    flat_inputs = [args[0], *args[1], *args[2:]]
    values = []
    root = nn.Module()
    graph = torch.fx.Graph()
    inputs = []
    for value in sub.graph.input:
        assert value.name.startswith("arg") and value.name[3:].isdigit()
        tensor = flat_inputs[int(value.name[3:])]
        values.append(tensor.half() if tensor.is_floating_point() else tensor)
        inputs.append(graph.placeholder(value.name))
    root.block = PlaceHolderModule(
        output_shapes=[[dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in sub.graph.output],
        output_dtypes=[torch.float16] * count,
        num_outputs=count,
        content="block",
    )
    block = graph.call_module("block", tuple(inputs))
    graph.output(tuple(graph.call_function(operator.getitem, (block, index)) for index in range(count)))
    quanted = to_quant_graph(FrontendGraph.convert(GraphModule(root, graph), {}), "XH2a", config)
    ptq_quantize(quanted, [values], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    flatten_placeholder_quant_metadata(quanted)
    main_file = tmp_path / "main.onnx"
    to_export_hmonnx_from_quanted_graph(quanted, values, str(main_file), save_as_external_data=True)
    main = onnx.load(main_file, load_external_data=False)
    placeholder = next(node for node in main.graph.node if node.op_type == "PlaceHolder")
    replace_placeholder_node(placeholder, main, sub_file, "main_external_data", tmp_path / "main_external_data")
    onnx.save(main, main_file)
    onnx.checker.check_model(str(main_file))
    expected = HMONNXInference(str(sub_file))(*values)
    actual = HMONNXInference(str(main_file))(*values)
    assert len(actual) == count
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
