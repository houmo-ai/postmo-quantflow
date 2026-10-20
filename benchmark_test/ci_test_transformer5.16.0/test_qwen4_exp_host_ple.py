"""Real SDK placement keeps the PLE table on host and returns small GPU tensors."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from xhmodel_merak.xh_llm.models.qwen4_exp.golden_runtime import GoldenCapture
from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_runtime import (
    Qwen4ExpHostPLEInferenceV2,
    Qwen4ExpServingGraphModel,
    find_host_ple_gathers,
)
from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, LayerInfo


class TinyHostPLESession(Qwen4ExpHostPLEInferenceV2):
    HOST_PLE_MIN_BYTES = 1


def graph_file(tmp_path, *, weight_name="model.layers.1.ple.ple_embedding.ngram_embedding.weight", axis=0):
    weight = np.arange(20, dtype=np.float16).reshape(5, 4)
    graph = helper.make_graph(
        [
            helper.make_node("Gather", [weight_name, "ids"], ["embedding"], name="ple_lookup", axis=axis),
            helper.make_node(
                "TopK", ["embedding", "k"], ["values", "indices"], name="top_k", axis=-1, largest=1, sorted=1
            ),
        ],
        "host_ple",
        [helper.make_tensor_value_info("ids", TensorProto.INT32, [1, 2])],
        [
            helper.make_tensor_value_info("values", TensorProto.FLOAT16, [1, 2, 2]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 2, 2]),
        ],
        initializer=[
            numpy_helper.from_array(weight, weight_name),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "k"),
        ],
    )
    path = tmp_path / "host_ple.onnx"
    onnx.save_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    return path, torch.from_numpy(weight)


def test_proven_ple_remains_cpu_and_small_gather_output_reaches_gpu_with_golden(tmp_path):
    path, table = graph_file(tmp_path)
    # Incompatible prefill names must never override decode placement.
    config = HMONNXInferenceConfig(
        exec_devices=["cuda:0"],
        enable_auto_offload=True,
        layers=[LayerInfo(99, ["unrelated_prefill_node"], torch.device("cuda:0"), 0)],
    )
    with torch.cuda.device(0):
        session = TinyHostPLESession(str(path), config)
        capture = GoldenCapture(tmp_path / "golden", ["prefill"])
        ids = torch.tensor([[0, 4]], dtype=torch.int32, device="cuda:0")
        with capture.step("prefill", session):
            actual = session.run({"ids": ids})
        capture.finish()
        expected = torch.topk(table[ids.cpu().long()], 2, dim=-1)
        torch.testing.assert_close(actual[0].cpu(), expected.values)
        torch.testing.assert_close(actual[1].cpu(), expected.indices)
        assert all(value.device == torch.device("cuda:0") for value in actual)
        interpreter = session.interpreter
        assert interpreter.node_device_map["ple_lookup"].device == "cpu"
        assert interpreter._get_node_memory_footprints()["ple_lookup"] == 0
        assert interpreter.node_device_map["ple_lookup"].size_in_bytes == 40
        assert interpreter.node_device_map["top_k"].device == "cuda:0"
        assert all(
            node.name in interpreter.node_device_map
            for node in session.graph_module.graph.nodes
            if node.op not in {"get_attr", "placeholder"}
        )
        state = next(t for n, t in session.graph_module.named_buffers() if n.endswith("ngram_embedding.weight"))
        assert state.device.type == "cpu"
        assert interpreter.host_ple_nodes == {"ple_lookup": table.numel() * table.element_size()}


@pytest.mark.parametrize(
    "weight_name",
    [
        "model.layers.1.ple.ple_embedding.ngram_embedding.weight",
        "L__self___native_text_model_layers_1_ple_ple_embedding_ngram_embedding_weight",
    ],
)
def test_cpu_execution_remains_available_for_oversized_ple(tmp_path, weight_name):
    path, table = graph_file(tmp_path, weight_name=weight_name)
    session = TinyHostPLESession(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))
    assert find_host_ple_gathers(session.graph_module, min_nbytes=1) == {"ple_lookup": 40}
    ids = torch.tensor([[1, 3]], dtype=torch.int32)
    actual = session.run({"ids": ids})
    expected = torch.topk(table[ids.long()], 2, dim=-1)
    torch.testing.assert_close(actual[0], expected.values)
    torch.testing.assert_close(actual[1], expected.indices)


def test_ordinary_gather_and_small_ple_do_not_get_special_placement(tmp_path):
    path, _ = graph_file(tmp_path, weight_name="ordinary_embedding.weight")
    session = TinyHostPLESession(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))
    assert find_host_ple_gathers(session.graph_module, min_nbytes=1) == {}
    path, _ = graph_file(tmp_path)
    session = Qwen4ExpHostPLEInferenceV2(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))
    assert find_host_ple_gathers(session.graph_module) == {}


@pytest.mark.parametrize("from_graph", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_serving_wrapper_uses_model_session_with_bound_backend(tmp_path, monkeypatch, from_graph, device):
    from xhmodel_merak.exec_backends import get_backend

    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "hmonnx")
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    monkeypatch.setattr(Qwen4ExpHostPLEInferenceV2, "HOST_PLE_MIN_BYTES", 1)
    backend = get_backend()
    original_implementation = backend.resolve_class("HMONNXInferenceV2")
    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "invalid-after-selection")
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "0")
    path, table = graph_file(tmp_path)
    wrapper = Qwen4ExpServingGraphModel(
        str(path),
        onnx_graph=gs.import_onnx(onnx.load(path)) if from_graph else None,
        backend=backend,
        device_map=[device],
        enable_auto_offload=device.startswith("cuda"),
    )
    session = wrapper.hmonnx_session
    assert isinstance(session, Qwen4ExpHostPLEInferenceV2)
    assert wrapper._backend is backend
    assert backend.resolve_class("HMONNXInferenceV2") is original_implementation
    ids = torch.tensor([[0, 4]], dtype=torch.int32, device=device)
    actual = wrapper(ids)
    expected = torch.topk(table[ids.cpu().long()], 2, dim=-1)
    torch.testing.assert_close(actual[0].cpu(), expected.values)
    torch.testing.assert_close(actual[1].cpu(), expected.indices)
    if device.startswith("cuda"):
        assert session.interpreter.host_ple_nodes == {"ple_lookup": 40}
        assert session.interpreter.node_device_map["ple_lookup"].device == "cpu"
        assert session.interpreter.node_device_map["top_k"].device == device
        assert all(value.device == torch.device(device) for value in actual)


@pytest.mark.parametrize("explicit_backend", [False, True])
def test_host_ple_wrapper_rejects_tcim_without_creating_sdk_session(monkeypatch, explicit_backend):
    from xhmodel_merak.exec_backends import get_backend

    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "tcim")
    backend = get_backend() if explicit_backend else None
    if explicit_backend:
        monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "hmonnx")
    with pytest.raises(NotImplementedError, match="SDK HMONNX"):
        Qwen4ExpServingGraphModel("does-not-exist.onnx", backend=backend)


@pytest.mark.parametrize(
    "auto_offload,cuda_graph,message",
    [
        (False, False, "requires auto-offload"),
        (True, True, "CUDA Graph to be disabled"),
    ],
)
def test_gpu_oversized_ple_requires_supported_execution_mode(tmp_path, auto_offload, cuda_graph, message):
    path, _ = graph_file(tmp_path)
    config = HMONNXInferenceConfig(
        exec_devices=["cuda:0"], enable_auto_offload=auto_offload, enable_cuda_graph=cuda_graph
    )
    with pytest.raises(ValueError, match=message):
        TinyHostPLESession(str(path), config)


@pytest.mark.parametrize("fail_loading", [False, True])
@pytest.mark.parametrize("use_inference_v2", [False, True])
def test_serving_wrapper_installs_runtime_adapters_only_during_loading(monkeypatch, fail_loading, use_inference_v2):
    from xhmodel_merak.exec_backends import get_backend
    from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel
    from xhmodel_merak.xh_llm.models.qwen4_exp import host_ple_runtime
    from xhquant.xhonnxruntime.builder import HMONNX_PARSERS

    original = dict(HMONNX_PARSERS.module_dict)
    observed = []

    def parse(self, enable_golden, enable_auto_offload):
        observed.append(HMONNX_PARSERS.get("Clip"))
        expected = Qwen4ExpHostPLEInferenceV2 if use_inference_v2 else original_implementation
        assert self._backend.resolve_class("HMONNXInferenceV2") is expected
        if fail_loading:
            raise RuntimeError("invalid model graph")
        assert HMONNX_PARSERS.get("XH2a::Conv1d") is original["XH2a::Conv1d"]

    monkeypatch.setattr(HMONNXModel, "_create_hmonnx_session", parse)
    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "hmonnx")
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1" if use_inference_v2 else "0")
    wrapper = object.__new__(host_ple_runtime.Qwen4ExpServingGraphModel)
    backend = get_backend()
    original_implementation = backend.resolve_class("HMONNXInferenceV2")
    wrapper._backend = backend
    if fail_loading:
        with pytest.raises(RuntimeError, match="invalid model graph"):
            wrapper._create_hmonnx_session(False, True)
    else:
        wrapper._create_hmonnx_session(False, True)
    assert wrapper._backend is backend
    assert backend.resolve_class("HMONNXInferenceV2") is original_implementation
    assert observed == [original["Clip"]]
    assert HMONNX_PARSERS.module_dict == original


def test_golden_host_transfer_compacts_broadcast_without_mutating_cache(monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_runtime import HostPLEAutoOffloadInterpreter
    from xhquant.core import CacheTensor

    interpreter = object.__new__(HostPLEAutoOffloadInterpreter)
    base = torch.arange(6, device="cuda:0", dtype=torch.float16).reshape(1, 2, 3)
    cache = CacheTensor(base.expand(5, 2, 3))
    original_pointer = cache.data_ptr()
    original_device = cache.device
    cpu = torch.Tensor.cpu
    transferred = []

    def observed(value, *args, **kwargs):
        transferred.append((type(value), value.numel(), tuple(value.shape)))
        return cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", observed)
    actual = interpreter._to_cpu(cache)
    assert transferred == [(torch.Tensor, 6, (1, 2, 3))]
    assert actual.device.type == "cpu" and actual.is_contiguous()
    torch.testing.assert_close(actual, cpu(base).expand(5, 2, 3))
    assert cache.device == original_device and cache.data_ptr() == original_pointer
    assert cache.shape == (5, 2, 3) and cache.stride()[0] == 0
    cpu_tensor = torch.ones(2, 3)
    assert interpreter._to_cpu(cpu_tensor) is cpu_tensor


def test_actual_sdk_golden_captures_expanded_indices_on_host(tmp_path):
    from xhmodel_merak.xh_llm.models.qwen4_exp.xhquant_extensions import qwen4_exp_xhquant_extension_scope

    path, table = graph_file(tmp_path)
    model = onnx.load(path)
    del model.graph.node[1:]
    model.graph.node.extend([
        helper.make_node("Expand", ["indices", "shape"], ["expanded"], name="expand_indices"),
        helper.make_node("GatherElements", ["embedding", "expanded"], ["result"], name="select", axis=2),
    ])
    model.graph.input.append(helper.make_tensor_value_info("indices", TensorProto.INT32, [1, 1, 2]))
    model.graph.initializer.append(numpy_helper.from_array(np.array([1, 2, 2], dtype=np.int64), "shape"))
    del model.graph.output[:]
    model.graph.output.append(helper.make_tensor_value_info("result", TensorProto.FLOAT16, [1, 2, 2]))
    onnx.save(model, path)
    config = HMONNXInferenceConfig(exec_devices=["cuda:0"], enable_auto_offload=True)
    with torch.cuda.device(0), qwen4_exp_xhquant_extension_scope():
        session = TinyHostPLESession(str(path), config)
    capture = GoldenCapture(tmp_path / "expanded_golden", ["prefill"])
    ids = torch.tensor([[0, 4]], dtype=torch.int32, device="cuda:0")
    indices = torch.tensor([[[0, 3]]], dtype=torch.int32, device="cuda:0")
    with capture.step("prefill", session):
        actual = session.run({"ids": ids, "indices": indices})
    capture.finish()
    if isinstance(actual, (tuple, list)):
        actual = actual[0]
    expected = torch.gather(table[ids.cpu().long()], 2, indices.cpu().long().expand(1, 2, 2))
    torch.testing.assert_close(actual.cpu(), expected)
    saved = np.load(tmp_path / "expanded_golden/prefill/step_0/expanded.npy")
    np.testing.assert_array_equal(saved, np.array([[[0, 3], [0, 3]]], dtype=np.int32))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("mtp", [False, True])
def test_model_graph_loader_binds_host_ple_backend_to_shared_components(tmp_path, monkeypatch, device, mtp):
    from xhmodel_merak.exec_backends import get_backend
    from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel
    from xhmodel_merak.xh_llm.models.qwen4_exp.qwen4_exp_hmonnx_inference import XHQwen4ExpHMONNXModel
    from xhquant.xhonnxruntime.builder import HMONNX_PARSERS

    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "hmonnx")
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    monkeypatch.setattr(Qwen4ExpHostPLEInferenceV2, "HOST_PLE_MIN_BYTES", 1)
    original_backend = get_backend()
    original_session = original_backend.resolve_class("HMONNXInferenceV2")
    registry = dict(HMONNX_PARSERS.module_dict)
    runtime = object.__new__(XHQwen4ExpHMONNXModel)
    runtime._backend = original_backend
    path, table = graph_file(tmp_path)
    runtime._qwen4_spec = {
        "mode": "mtp", "runtime_contract_version": 3,
        "draft": {"decode_hmonnx": str(path), "prefill_hmonnx": str(path)},
        "target": {"verify_hmonnx": str(path), "verify_chunk_length": 2},
    } if mtp else None
    prefill, decode, _ = runtime._load_hmonnx_graphs(str(path), str(path))
    assert runtime._backend.resolve_class("HMONNXInferenceV2") is Qwen4ExpHostPLEInferenceV2
    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "invalid-after-selection")
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "0")
    ids = torch.tensor([[0, 4]], dtype=torch.int32, device=device)
    expected = torch.topk(table[ids.cpu().long()], 2, dim=-1)
    for graph in (prefill, decode):
        component = HMONNXModel(
            str(path), onnx_graph=graph, backend=runtime._backend,
            device_map=[device], enable_auto_offload=device.startswith("cuda"),
        )
        assert type(component) is HMONNXModel
        assert isinstance(component.hmonnx_session, Qwen4ExpHostPLEInferenceV2)
        actual = component(ids)
        torch.testing.assert_close(actual[0].cpu(), expected.values)
        torch.testing.assert_close(actual[1].cpu(), expected.indices)
    if mtp:
        for graph in (runtime._qwen4_verify_graph, runtime._qwen4_mtp_prefill_graph, runtime._qwen4_mtp_graph):
            component = Qwen4ExpServingGraphModel(
                str(path), onnx_graph=graph, backend=runtime._backend,
                device_map=[device], enable_auto_offload=device.startswith("cuda"),
            )
            assert isinstance(component.hmonnx_session, Qwen4ExpHostPLEInferenceV2)
            actual = component(ids)
            torch.testing.assert_close(actual[0].cpu(), expected.values)
            torch.testing.assert_close(actual[1].cpu(), expected.indices)
    assert original_backend.resolve_class("HMONNXInferenceV2") is original_session
    assert HMONNX_PARSERS.module_dict == registry


def test_model_backend_restores_adapters_when_shared_session_loading_fails(tmp_path, monkeypatch):
    from xhmodel_merak.exec_backends import get_backend
    from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel
    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_runtime import qwen4_exp_execution_backend
    from xhquant.xhonnxruntime.builder import HMONNX_PARSERS

    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "hmonnx")
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    backend = get_backend()
    registry = dict(HMONNX_PARSERS.module_dict)
    path, _ = graph_file(tmp_path)
    graph = onnx.load(path)
    graph.graph.node[0].op_type = "UnknownQwen4Op"
    onnx.save(graph, path)
    with pytest.raises(ValueError, match="Unsupported op"):
        HMONNXModel(str(path), backend=qwen4_exp_execution_backend(backend), device_map=["cpu"])
    assert HMONNX_PARSERS.module_dict == registry
    assert backend.resolve_class("HMONNXInferenceV2") is not Qwen4ExpHostPLEInferenceV2


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_full_serving_constructor_binds_shared_prefill_and_decode_sessions(tmp_path, monkeypatch, device):
    from transformers import Qwen4ExpTextConfig

    from xhmodel_merak.exec_backends import get_backend
    from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel
    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import Qwen4ExpCacheContract
    from xhmodel_merak.xh_llm.models.qwen4_exp.qwen4_exp_hmonnx_inference import XHQwen4ExpHMONNXModel
    from xhmodel_merak.xh_llm.types import BaseLLMModelConfig, KVCacheConfig, LLMModelMeta

    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "hmonnx")
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    monkeypatch.setattr(Qwen4ExpHostPLEInferenceV2, "HOST_PLE_MIN_BYTES", 1)
    config = Qwen4ExpTextConfig(
        max_position_embeddings=16,
        vocab_size=5, hidden_size=4, num_hidden_layers=0,
        num_attention_heads=1, num_key_value_heads=1, head_dim=4,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=1, linear_num_value_heads=1,
        hc_count=2, hc_lowrank=2, layer_types=[], ple_layer_ids=[],
        indexer_n_heads=1, indexer_kv_heads=1, indexer_head_dim=4,
        indexer_budget=2, indexer_compress_ratio=2,
    )
    hf_dir = tmp_path / "hf"
    config.save_pretrained(hf_dir)
    path, table = graph_file(tmp_path)
    embedding = tmp_path / "embedding.pt"
    torch.save({"weight": table}, embedding)
    meta = LLMModelMeta(
        model_config=BaseLLMModelConfig(
            model_name="tiny_qwen4_host_ple", hf_model=str(hf_dir),
            context_max_length=8, prefill_chunk_length=2,
        ),
        hf_config=str(hf_dir), quant_embedding=str(embedding),
        kv_cache=KVCacheConfig(num_layers=0, kv_cache_shape=[1, 1, 8, 4]),
        prefill_hmonnx=str(path), decode_hmonnx=str(path),
        qwen4_exp_cache_contract=Qwen4ExpCacheContract.from_text_config(config).to_dict(),
    )
    runtime = XHQwen4ExpHMONNXModel(
        meta, device_map=[device], enable_auto_offload=device.startswith("cuda"),
    )
    assert runtime.embed_tokens.weight.device == torch.device(device)
    assert runtime._models["prefill_model"] is runtime.prefill_model
    assert runtime._models["decode_model"] is runtime.decode_model
    ids = torch.tensor([[0, 4]], dtype=torch.int32, device=device)
    expected = torch.topk(table[ids.cpu().long()], 2, dim=-1)
    for component in (runtime.prefill_model, runtime.decode_model):
        assert type(component) is HMONNXModel
        assert isinstance(component.hmonnx_session, Qwen4ExpHostPLEInferenceV2)
        assert component._backend is runtime._backend
        actual = component(ids)
        torch.testing.assert_close(actual[0].cpu(), expected.values)
        torch.testing.assert_close(actual[1].cpu(), expected.indices)
    assert get_backend().resolve_class("HMONNXInferenceV2") is not Qwen4ExpHostPLEInferenceV2
