from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_llm.builder import get_model_class
from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    QWEN_SPARSE_ATTENTION,
    Qwen4ExpCacheContract,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.qwen4_exp_hmonnx_inference import (
    Qwen4ExpServingDataPreprocess,
    Qwen4ExpServingState,
    XHQwen4ExpHMONNXModel,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.qwen4_exp_model import XHQwen4ExpModel
from xhmodel_merak.xh_llm.models.qwen4_exp.serving_export import build_qwen4_exp_serving_metadata
from xhquant.core import CacheTensor


def _text_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="qwen4_exp_text",
        num_hidden_layers=3,
        layer_types=[LINEAR_ATTENTION, QWEN_SPARSE_ATTENTION, FULL_ATTENTION],
        hidden_size=16,
        hc_count=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        indexer_head_dim=4,
        indexer_budget=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        ple_conv_kernel_size=3,
        ngram_size=3,
        ple_layer_ids=[1],
        eos_token_id=151643,
    )


def test_qwen4_exp_model_registers_hmonnx_runtime_class() -> None:
    assert (
        get_model_class({"model_name": "qwen3_8_flash_next", "model_type": "Qwen4ExpForConditionalGeneration"})
        is XHQwen4ExpModel
    )
    assert XHQwen4ExpModel.HMONNXINFERENCE_CLS.__name__ == "XHQwen4ExpHMONNXModel"


def test_serving_state_keeps_only_non_page_kv_tensors_and_resets() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    state = Qwen4ExpServingState(
        contract,
        batch_size=1,
        context_length=9,
        dtype=torch.float16,
        device="cpu",
        ple_eos_token_id=151643,
    )

    assert "key_cache_1" not in state.tensors
    assert "value_cache_1" not in state.tensors
    assert "qsa_indexer_key_cache_1" in state.tensors
    assert isinstance(state.tensors["qsa_indexer_key_cache_1"], CacheTensor)
    assert state.tensors["qsa_indexer_key_cache_1"].shape == (1, 1, 2, 4)
    assert state.tensors["qsa_indexer_key_tail_1"].shape == (1, 4, 4)
    assert state.group_start_position.shape == (3, 1, 1)
    state.group_start_position.fill_(9)
    state.tensors["qsa_indexer_key_cache_1"].fill_(3)
    state.reset()
    assert state.group_start_position.count_nonzero().item() == 0
    assert state.tensors["qsa_indexer_key_cache_1"].sum().item() == 0
    assert state.tensors["ple_ngram_history_0"].tolist() == [[151643, 151643]]



def test_explicit_cache_state_owns_full_capacity_kv_and_preserves_storage():
    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import (
        qwen4_exp_cache_input_specs,
        resolve_qwen4_exp_cache_abi,
    )

    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    state = Qwen4ExpServingState(contract, batch_size=1, context_length=9,
                                dtype=torch.float16, device="cpu", own_kv=True)
    specs = qwen4_exp_cache_input_specs(contract)
    assert specs["past_key_cache_0"].name == "key_cache_1"
    assert specs["past_key_cache_1"].name == "key_cache_2"
    values = state.inputs_in_contract_order(past_key_caches=None, past_value_caches=None,
                                           enable_page_attention=False)
    for spec, value in zip(contract.tensors, values, strict=True):
        assert value is state.tensors[spec.name]
        if spec.state_kind in {"kv_key", "kv_value"}:
            assert isinstance(value, CacheTensor)
            assert value.shape == (1, 2, 9, 4)
    owner = state.tensors["key_cache_1"]
    owner[:, :, :2].fill_(3)
    state.commit_outputs((torch.zeros(1), owner[:, :, :2]),
                         ("logits", "present_past_key_cache_0"))
    assert state.tensors["key_cache_1"] is owner
    assert owner[:, :, :2].sum() == 48
    assert resolve_qwen4_exp_cache_abi() == "page_kv_v1"
    assert resolve_qwen4_exp_cache_abi("llm_cache_inplace_v1") == "llm_cache_inplace_v1"
    import pytest
    with pytest.raises(ValueError, match="Unsupported Qwen4 cache ABI"):
        resolve_qwen4_exp_cache_abi("unknown")

def test_runtime_exposes_vllm_adapter_reset_contract() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    runtime = XHQwen4ExpHMONNXModel.__new__(XHQwen4ExpHMONNXModel)
    runtime._qwen4_state = Qwen4ExpServingState(
        contract,
        batch_size=1,
        context_length=9,
        dtype=torch.float16,
        device="cpu",
    )
    runtime._qwen4_state.tensors["qsa_indexer_key_cache_1"].fill_(5)

    runtime.reset_internal_state()

    assert runtime._qwen4_state.tensors["qsa_indexer_key_cache_1"].sum().item() == 0
    assert callable(runtime.reset_qwen4_state)


def test_recurrent_cache_survives_fused_updates_and_unfused_outputs() -> None:
    state = Qwen4ExpServingState(
        Qwen4ExpCacheContract.from_text_config(_text_config()),
        batch_size=1,
        context_length=9,
        dtype=torch.float16,
        device="cpu",
    )
    recurrent = state.tensors["delta_recurrent_state_0"]
    assert isinstance(recurrent, CacheTensor)
    inputs = state.inputs_in_contract_order(
        past_key_caches=None, past_value_caches=None, enable_page_attention=True
    )
    assert any(value is recurrent for value in inputs)
    logits = torch.zeros(1, 1, 8)
    # Fused scan publishes its final state in place, even without a state output.
    recurrent.fill_(2)
    state.commit_outputs((logits,), ("logits",))
    assert state.tensors["delta_recurrent_state_0"] is recurrent
    assert recurrent.sum().item() == recurrent.numel() * 2
    # Unfused serving graphs publish a plain output tensor for the same state.
    updated = torch.full(recurrent.shape, 3, dtype=recurrent.dtype)
    assert not isinstance(updated, CacheTensor)
    state.commit_outputs((logits, updated), ("logits", "present_delta_recurrent_state_0"))
    assert state.tensors["delta_recurrent_state_0"] is recurrent
    torch.testing.assert_close(recurrent, updated)
    state.to(dtype=torch.float32)
    assert state.tensors["delta_recurrent_state_0"] is recurrent
    assert isinstance(recurrent, CacheTensor) and recurrent.dtype == torch.float32
    state.reset()
    assert isinstance(state.tensors["delta_recurrent_state_0"], CacheTensor)
    assert state.tensors["delta_recurrent_state_0"].count_nonzero().item() == 0


def test_serving_preprocessor_preserves_valid_length_for_padded_prefill() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    state = Qwen4ExpServingState(
        contract,
        batch_size=1,
        context_length=256,
        dtype=torch.float16,
        device="cpu",
    )
    processor = Qwen4ExpServingDataPreprocess(
        token_embedding=torch.nn.Embedding(32, 16),
        input_sequence_length=64,
        state=state,
        past_key_caches=None,
        past_value_caches=None,
        enable_page_attention=True,
        pad_token_id=0,
    )

    net_inputs = processor(
        {
            "input_ids": torch.ones((1, 3), dtype=torch.long),
            "past_seq_length": 128,
            "current_input_length": 3,
        }
    )

    assert net_inputs[1].tolist() == [128]
    assert net_inputs[2].tolist() == [3]
    assert net_inputs[0].shape == (1, 64, 16)
    assert net_inputs[3].shape == (1, 64)
    assert net_inputs[3][0, :3].tolist() == [1, 1, 1]
    assert net_inputs[3][0, 3:].tolist() == [0] * 61


def test_serving_preprocessor_prepares_host_qsa_controls_for_padded_prefill() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    state = Qwen4ExpServingState(contract, batch_size=1, context_length=64, dtype=torch.float16, device="cpu")
    processor = Qwen4ExpServingDataPreprocess(
        token_embedding=torch.nn.Embedding(32, 16), input_sequence_length=4, state=state,
        past_key_caches=None, past_value_caches=None, enable_page_attention=True, pad_token_id=0,
        qsa_control_spec=(2, torch.float16),
    )
    values = processor({"input_ids": torch.tensor([[1, 2]]), "past_seq_length": 3})
    offset, new_count, write_start, query_valid, mask = values[-5:]
    assert (offset.item(), new_count.item(), write_start.item()) == (1, 1, 0)
    assert query_valid.tolist() == [[True, True, False, False]]
    assert mask.shape == (1, 4, 1, 11) and mask.dtype == torch.float16
    assert (mask[0, 2, 0] == 0).nonzero().flatten().tolist() == []


def test_qwen38_export_cli_defaults_to_256_token_prefill(monkeypatch) -> None:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "examples_merak/llm/qwen4_exp/debug_scripts/export_qwen4_exp_serving_hmonnx.py"
    )
    spec = importlib.util.spec_from_file_location("qwen38_serving_export_cli", script_path)
    assert spec is not None and spec.loader is not None
    export_qwen38_next_serving_hmonnx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_qwen38_next_serving_hmonnx)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_qwen4_exp_serving_hmonnx.py",
            "--model-dir",
            "/tmp/model",
            "--output-dir",
            "/tmp/export",
            "--config",
            "/tmp/config.yaml",
        ],
    )

    args = export_qwen38_next_serving_hmonnx._parse_args()

    assert args.prefill_chunk_length == 256


def test_serving_metadata_writes_standard_auto_hmonnx_contract(tmp_path) -> None:
    import onnx
    from onnx import TensorProto, helper

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text('{"model_type":"qwen4_exp"}\n', encoding="utf-8")
    prefill = tmp_path / "prefill.hmonnx"
    decode = tmp_path / "decode.hmonnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])], "metadata_fixture",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 1, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 1, 16])],
    )
    for path in (prefill, decode):
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)

    meta_path = build_qwen4_exp_serving_metadata(
        output_dir=tmp_path / "package",
        model_name="qwen3_8_flash_next",
        hf_model_dir=hf_dir,
        text_config=_text_config(),
        model_config={"context_max_length": 10240, "batch_size": 1},
        prefill_hmonnx=prefill,
        decode_hmonnx=decode,
        embedding_state={"weight": torch.zeros(8, 16, dtype=torch.float16)},
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["model_config"]["model_type"] == "Qwen4ExpForConditionalGeneration"
    assert meta["model_config"]["context_max_length"] == 10240
    assert meta["model_config"]["prefill_chunk_length"] == 256
    assert meta["kv_cache"]["num_layers"] == 2
    assert meta["kv_cache"]["kv_cache_shape"] == [[1, 2, 10240, 4], [1, 2, 10240, 4]]
    assert meta["qwen4_exp_serving"]["qsa_attention"] == "selected_kv_matmul_attention"
    assert "qwen4_exp_cache_contract" in meta
    merak_config = json.loads((meta_path.parent / "merak_config.json").read_text(encoding="utf-8"))
    assert merak_config["architectures"] == ["MerakForCausalLM"]
    assert merak_config["config_format"] == "merak_llm"
    assert merak_config["load_format"] == "merak_llm"
    assert merak_config["xh_model"] == {"model_type": "hmonnx", "meta_info": "golden_meta_info.json"}
    assert merak_config["model_type"] == "merak_llm"
    assert merak_config["enable_page_attention"] is False
    assert meta["qwen4_exp_cache_abi"] == "llm_cache_inplace_v1"
    assert merak_config["disable_prefix_caching"] is True


@pytest.mark.parametrize("verify_length", [2, 4])
def test_verify_dispatches_independent_target_sdk_graph(tmp_path, verify_length):
    import onnx
    from onnx import TensorProto, helper

    from xhmodel_merak.exec_backends import HMONNXInferenceConfig, HMONNXInferenceV2

    path = tmp_path / "verify.onnx"
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "x"], ["logits"], name="verify_logits"),
         helper.make_node("Identity", ["x"], ["target_hidden_states"], name="verify_hidden")],
        "independent_verify", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, verify_length, 4])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, verify_length, 4]),
         helper.make_tensor_value_info("target_hidden_states", TensorProto.FLOAT, [1, verify_length, 4])],
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    proto.ir_version = 9
    onnx.save(proto, path)
    session = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cpu"]))
    model = object.__new__(XHQwen4ExpHMONNXModel)
    model.verify_model = SimpleNamespace(hmonnx_session=session, forward=session.forward)
    model.prefill_model = model.decode_model = None
    model._qwen4_verify_length = verify_length
    model._data_processor = None
    model._qwen4_state = Qwen4ExpServingState(
        Qwen4ExpCacheContract.from_text_config(_text_config()), batch_size=1,
        context_length=8, dtype=torch.float32, device="cpu",
    )
    model.set_verify()
    assert model.get_input_sequence_length() == verify_length
    assert model.active_graph() is model.verify_model
    assert model.active_component() == f"decode_m{verify_length}"
    inputs = torch.arange(verify_length * 4, dtype=torch.float32).view(1, verify_length, 4)
    logits = model.forward(inputs)
    torch.testing.assert_close(logits, inputs * 2)
    torch.testing.assert_close(model.last_hidden_states, inputs)


@pytest.mark.parametrize("verify_length", [2, 4])
@pytest.mark.parametrize("declared_role", [True, False])
def test_verify_loader_resolves_metadata_width_and_legacy_role(tmp_path, verify_length, declared_role):
    import onnx
    from onnx import TensorProto, helper

    from xhmodel_merak.exec_backends import get_backend

    paths = {}
    role = f"decode_m{verify_length}"
    for phase, width in (("prefill", 8), ("decode", 1), ("mtp_prefill", 8), ("mtp_decode", 1),
                         (role, verify_length)):
        path = tmp_path / phase / "model.onnx"
        path.parent.mkdir()
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"], name=f"{phase}_identity")], phase,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, width, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, width, 4])],
        )
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
        paths[phase] = path
    model = object.__new__(XHQwen4ExpHMONNXModel)
    model._backend = get_backend()
    model._qwen4_spec = {
        "mode": "mtp", "runtime_contract_version": 3,
        "target": {"verify_hmonnx": f"{role}/model.onnx", "verify_chunk_length": verify_length},
        "draft": {"prefill_hmonnx": "mtp_prefill/model.onnx", "decode_hmonnx": "mtp_decode/model.onnx"},
    }
    if declared_role:
        model._qwen4_spec["target_verify_graph"] = role
    prefill, decode, _ = model._load_hmonnx_graphs(paths["prefill"], paths["decode"])
    assert prefill is not decode
    assert model._qwen4_verify_length == verify_length
    assert model._qwen4_verify_phase == role
    assert set(model.external_ple) == {"prefill", "decode", role}
    assert model._qwen4_verify_graph is not None
    model._qwen4_spec["target_verify_graph"] = "decode_m4" if verify_length == 2 else "decode_m2"
    with pytest.raises(ValueError, match="role disagrees"):
        model._load_hmonnx_graphs(paths["prefill"], paths["decode"])


@pytest.mark.parametrize("cache_op", ["KVcache", "LLMCache"])
def test_explicit_kv_writer_validation_cannot_count_qsa_as_value_cache(tmp_path, cache_op):
    import onnx
    import pytest
    from onnx import TensorProto, helper

    from xhmodel_merak.exec_backends import HMONNXInferenceConfig, MultiHMONNXLoader
    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_runtime import (
        Qwen4ExpHostPLEInferenceV2,
        enable_qwen4_exp_inplace_kv,
    )

    def session_for(names):
        inputs = [helper.make_tensor_value_info("states", TensorProto.FLOAT16, [1, 1, 1, 8]),
                  helper.make_tensor_value_info("past", TensorProto.INT32, [1]),
                  helper.make_tensor_value_info("valid", TensorProto.INT32, [1])]
        nodes, outputs = [], []
        for name in names:
            inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, 1, 8, 8]))
            nodes.append(helper.make_node(cache_op, ["states", "past", "valid", name], [f"out_{name}"],
                                          name=f"writer_{name}", domain="ai.houmo.xh2a", axis=2,
                                          attention_max_length=-1))
            outputs.append(helper.make_tensor_value_info(f"out_{name}", TensorProto.FLOAT16, [1, 1, 8, 8]))
        path = tmp_path / f"cache_{len(names)}.onnx"
        onnx.save(helper.make_model(helper.make_graph(nodes, "cache", inputs, outputs), opset_imports=[
            helper.make_opsetid("", 17), helper.make_opsetid("ai.houmo.xh2a", 1)]), path)
        loader = MultiHMONNXLoader({"cache": path})
        session = Qwen4ExpHostPLEInferenceV2.from_onnx_graph(
            str(path), loader.graphs["cache"], HMONNXInferenceConfig(exec_devices=["cpu"]),
        )
        session.initialize()
        return session

    key, value, qsa = "past_key_cache_0", "past_value_cache_0", "qsa_indexer_key_cache_0"
    with pytest.raises(ValueError, match="paired K/V"):
        enable_qwen4_exp_inplace_kv(session_for([key, qsa]), {key, value})
    session = session_for([key, value, qsa])
    enable_qwen4_exp_inplace_kv(session, {key, value})
    feed = {name: CacheTensor(torch.zeros(1, 1, 8, 8, dtype=torch.float16)) for name in (key, value, qsa)}
    feed.update(states=torch.ones(1, 1, 1, 8, dtype=torch.float16), past=torch.tensor([2], dtype=torch.int32),
                valid=torch.ones(1, dtype=torch.int32))
    session.run(feed)
    for name in (key, value):
        assert feed[name][:, :, 2].sum() == 8
        assert feed[name][:, :, :2].sum() == feed[name][:, :, 3:].sum() == 0
