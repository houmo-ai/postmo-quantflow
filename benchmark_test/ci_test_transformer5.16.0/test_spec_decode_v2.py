"""Real CPU ONNX sessions: loading/sharing and ABI, not generation accuracy."""
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import _patches_runtime_scope
from xhmodel_merak.xh_llm.models.qwen3_5.spec_decode_v2 import Qwen3_5SpecDecodeV2Model
from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceV2


def _write_graph(path: Path):
    inputs = [
        helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [1, 2, 4]),
        helper.make_tensor_value_info("past_seq_length", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("current_input_length", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("linear_attn_mask", TensorProto.FLOAT, [1, 2]),
    ]
    graph = helper.make_graph(
        [helper.make_node("Add", ["inputs_embeds", "shared_weight"], ["logits"], name="add_shared")],
        "tiny_spec_session", inputs,
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, 4])],
        [numpy_helper.from_array(np.arange(4, dtype=np.float32), "shared_weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)
    return {"onnx": str(path)}


@pytest.mark.parametrize("mode", ["mtp", "dflash", "dflash2"])
def test_real_v2_shared_target_and_draft_sessions(tmp_path, mode):
    files = {name: _write_graph(tmp_path / f"{name}.onnx")
             for name in ("prefill", "decode", "draft_a", "draft_b")}
    draft = {"prefill" if mode == "mtp" else "context": files["draft_a"], "decode": files["draft_b"]}
    runtime = Qwen3_5SpecDecodeV2Model(
        prefill=files["prefill"], decode=files["decode"], draft=draft,
        spec_decode_mode=mode, dflash_noise_token_id=0 if mode in {"dflash", "dflash2"} else None,
        session_device="cpu", auto_offload=False,
    )
    runtime.to("cpu").set_exec_device("cpu")
    sessions = [runtime.prefill_session, runtime.decode_session, *runtime._iter_unique_draft_sessions()]
    assert len(sessions) == 4
    for session in sessions:
        assert isinstance(session, HMONNXInferenceV2)
        assert not session._session_config.enable_cuda_graph
        feed = {
            "inputs_embeds": torch.ones(1, 2, 4),
            "past_seq_length": torch.tensor([0]),
            "current_input_length": torch.tensor([2]),
            "linear_attn_mask": torch.ones(1, 2),
        }
        _, output = runtime._run_hmonnx(session, feed)
        torch.testing.assert_close(output["logits"], feed["inputs_embeds"] + torch.arange(4))
    summary = runtime.runtime_summary["initializer_pool"]
    assert summary["unique_loaded_initializers"] == 1
    assert summary["shared_initializer_references"] == 3
    weights = [s.constants["shared_weight"] for s in sessions]
    assert len({w.data_ptr() for w in weights}) == 1
    assert not runtime._v2_graphs
    if mode in {"dflash", "dflash2"}:
        assert runtime.draft_context_decode_session is runtime.draft_context_session
    assert all(s["backend"] == "HMONNXInferenceV2" for s in runtime.get_cuda_graph_status().values()
               if s["backend"] is not None)
    with pytest.raises(ValueError, match="cannot move"):
        runtime.to("meta")
    with pytest.raises(ValueError, match="cannot move"):
        runtime.set_exec_device("meta")


@pytest.mark.parametrize("option,value", [
    ("resource_tight_mode", True),
    ("auto_offload_max_memory", {0: "1GiB"}),
    ("prefill_auto_offload_max_memory", {0: "1GiB"}),
    ("decode_auto_offload_max_memory", {0: "1GiB"}),
])
def test_v2_rejects_legacy_placement_before_loading(option, value):
    with pytest.raises(ValueError, match=option):
        Qwen3_5SpecDecodeV2Model(prefill=None, decode=None, draft=None, session_device="cpu", **{option: value})


def test_public_runtime_scope_enables_packing_and_restores_on_failure(monkeypatch):
    import os

    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "0")
    monkeypatch.setenv("XHQUANT_PACKED_W4", "0")
    with pytest.raises(RuntimeError), _patches_runtime_scope():
        assert os.environ["ENABLE_HMINFERENCE_V2"] == "1"
        assert os.environ["XHQUANT_PACKED_W4"] == "1"
        raise RuntimeError("test cleanup")
    assert os.environ["ENABLE_HMINFERENCE_V2"] == "0"
    assert os.environ["XHQUANT_PACKED_W4"] == "0"


def test_shared_loader_packs_w4_before_session_construction(tmp_path):
    from xhquant.backend.xh2a.functions.te.xh2a_matmul_triton import unpack_signed_w4_mma_tiled
    from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader

    source = (np.arange(2048) % 16 - 8).astype(np.int8).reshape(1, 64, 32)
    files = {}
    for stage in ("prefill", "decode", "draft_decode"):
        node = helper.make_node("Linear", ["x", "weight"], ["y"], name="linear",
                                domain="com.houmo", hmfp_weight_man_bit=4, hmfp_act_man_bit=8)
        graph = helper.make_graph(
            [node], stage, [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 64])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 32])],
            [numpy_helper.from_array(source, "weight")],
        )
        path = tmp_path / f"{stage}.onnx"
        onnx.save(helper.make_model(graph), path)
        files[stage] = str(path)
    with _patches_runtime_scope():
        loader = MultiHMONNXLoader(files)
    summary = loader.initializer_pool.summary()
    assert summary["runtime_w4_packed_references"] == 3
    assert summary["runtime_w4_packed_initializers"] == 1
    assert summary["runtime_w4_packed_saved_nbytes"] == 1024
    packed = [loader.initializer_pool.graphs[name]["weight"].values for name in files]
    assert len({w.data_ptr() for w in packed}) == 1
    torch.testing.assert_close(unpack_signed_w4_mma_tiled(packed[0], 32), torch.from_numpy(source))
