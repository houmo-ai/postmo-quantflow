"""Real SDK Golden hooks, component coverage and mutable-state snapshots."""

import json
from collections import ChainMap

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

from xhmodel_merak.xh_llm.models.qwen4_exp.golden_runtime import GoldenCapture
from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2


@pytest.fixture
def session(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "x"], ["y"], name="actual_add")],
        "golden_real_sdk",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "actual_graph.onnx"
    onnx.save(model, path)
    return HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cpu"]))


def test_real_sdk_all_components_and_page_state_capture(session, tmp_path):
    root = tmp_path / "golden"
    capture = GoldenCapture(root, ["prefill", "decode", "visual", "mtp"])
    state = torch.zeros(2, 1, 1, 2, 2)
    for index, component in enumerate(capture.components):
        inputs = torch.tensor([[index + 1.0, index + 2.0]])
        with capture.step(component, session, runtime_state={"physical_page_kv": state}):
            result = session(inputs)
            state.add_(1)
        step = root / component / "step_0"
        np.testing.assert_array_equal(np.load(step / "x.npy"), inputs.numpy())
        np.testing.assert_array_equal(np.load(step / "y.npy"), result.numpy())
        np.testing.assert_array_equal(result.numpy(), inputs.numpy() * 2)
        np.testing.assert_array_equal(np.load(step / "runtime_before_0.npy"), np.full(state.shape, index))
        np.testing.assert_array_equal(np.load(step / "runtime_after_0.npy"), np.full(state.shape, index + 1))
    artifacts = capture.finish()
    assert set(artifacts) == set(capture.components)
    assert json.loads(capture.manifest_path.read_text())["status"] == "complete"
    assert session.save_golden is False
    assert session.interpreter._enable_golden is False


def test_distinct_actual_steps_do_not_overwrite(session, tmp_path):
    capture = GoldenCapture(tmp_path / "golden", ["decode"])
    for value in (3.0, 7.0):
        with capture.step("decode", session):
            session(torch.full((1, 2), value))
    result = capture.finish()
    assert len(result["decode"]) == 2
    np.testing.assert_array_equal(np.load(tmp_path / "golden/decode/step_0/y.npy"), [[6.0, 6.0]])
    np.testing.assert_array_equal(np.load(tmp_path / "golden/decode/step_1/y.npy"), [[14.0, 14.0]])
    with pytest.raises(FileExistsError):
        GoldenCapture(tmp_path / "golden", ["decode"])


def test_missing_real_execution_is_not_golden(session, tmp_path):
    capture = GoldenCapture(tmp_path / "golden", ["prefill"])
    with pytest.raises(ValueError, match="SDK did not capture"):
        with capture.step("prefill", session):
            pass
    assert json.loads(capture.manifest_path.read_text())["status"] == "failed"
    assert session.save_golden is False
    assert session.interpreter._enable_golden is False


def test_missing_selected_component_rejects_completion(session, tmp_path):
    capture = GoldenCapture(tmp_path / "golden", ["prefill", "mtp"])
    with capture.step("prefill", session):
        session(torch.ones(1, 2))
    with pytest.raises(ValueError, match="mtp"):
        capture.finish()


def test_execution_exception_restores_sdk_flags(session, tmp_path):
    capture = GoldenCapture(tmp_path / "golden", ["decode"])
    before = session.step
    with pytest.raises(AssertionError, match="shape mismatch"):
        with capture.step("decode", session):
            session(torch.ones(1, 3))
    assert session.step == before
    assert session.save_golden is False
    assert session.interpreter._enable_golden is False
    assert json.loads(capture.manifest_path.read_text())["status"] == "failed"


def test_recurrent_cache_and_replaced_ple_state_are_snapshotted_live(session, tmp_path):
    from xhquant.core import CacheTensor

    states = {"recurrent": CacheTensor(torch.zeros(2)), "ple": torch.zeros(2)}
    pages = {"page_kv": torch.zeros(2)}
    capture = GoldenCapture(tmp_path / "golden", ["prefill"])
    with capture.step("prefill", session, runtime_state=ChainMap(pages, states)):
        session(torch.ones(1, 2))
        states["recurrent"].add_(2)
        states["ple"] = torch.full((2,), 3.0)
    capture.finish()
    step = capture.steps[0]
    for prefix, expected in (("runtime_before", {"recurrent": 0, "ple": 0}),
                             ("runtime_after", {"recurrent": 2, "ple": 3})):
        arrays = {item["name"]: np.load(capture.root / step["directory"] / item["file"])
                  for item in step[prefix]}
        for name, value in expected.items():
            np.testing.assert_array_equal(arrays[name], np.full((2,), value))
    assert isinstance(states["recurrent"], CacheTensor)


@pytest.mark.parametrize("verify_phase", ["decode_m2", "decode_m4"])
def test_external_ple_demo_replays_actual_sdk_golden_after_relocation(tmp_path, verify_phase):
    import shutil
    import subprocess
    import sys

    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_inputs import HostPLEInputs
    from xhmodel_merak.xh_llm.models.qwen4_exp.ple_sefp4 import quantize_ple_sefp4_rows

    source = tmp_path / "source"
    source.mkdir()
    weight = torch.linspace(-0.07, 0.09, 12 * 65, dtype=torch.float16).reshape(12, 65)
    packed, exponent = quantize_ple_sefp4_rows(weight)
    spec = {
        "layer_idx": 0, "input_name": "ple_embeddings_0", "embedding_dim": 65,
        "output_dtype": "float16", "ngram_size": 2, "heads_per_ngram": 2,
        "eos_token_id": 0, "multipliers": [7, 13], "vocab_sizes": [5, 7], "offsets": [0, 5],
    }
    for kind, tensor in (("packed", packed), ("exponent", exponent)):
        tensor.numpy().tofile(source / f"{kind}.bin")
        spec[kind] = {"path": f"{kind}.bin", "shape": list(tensor.shape), "dtype": str(tensor.numpy().dtype)}
    graph = helper.make_graph(
        [helper.make_node("Identity", ["ple_embeddings_0"], ["output"], name="actual_identity")],
        "external_ple", [helper.make_tensor_value_info("ple_embeddings_0", TensorProto.FLOAT16, [1, 3, 2, 65])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 3, 2, 65])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    helper.set_model_props(model, {"qwen4_exp_ple": json.dumps({"version": 1, "layers": [spec]})})
    graph_path = source / "model.onnx"
    onnx.save(model, graph_path)
    host = HostPLEInputs.from_graph(graph_path)
    session = HMONNXInferenceV2(str(graph_path), session_config=HMONNXInferenceConfig(exec_devices=["cpu"]))
    capture = GoldenCapture(tmp_path / "golden", ["prefill", "decode", "mtp"])
    capture.add_ple_resources({"prefill": host, "decode": host, verify_phase: host})
    state = {"ple_ngram_history_0": torch.tensor([[0]], dtype=torch.int32)}
    for phase, token in (("prefill", 2), ("decode", 3), (verify_phase, 4)):
        (embedding,) = host.prepare(torch.tensor([[token, 1, 99]]), state, 2, device="cpu")
        assert embedding.shape == (1, 3, 2, 65)
        component = "mtp" if phase == verify_phase else phase
        with capture.step(component, session, graph_name=phase, runtime_state=state, ple_indices=host.last_indices):
            session(embedding)
    capture.finish()
    shutil.rmtree(source)
    relocated = tmp_path / "moved"
    shutil.move(capture.root, relocated)
    for manifest in ("manifest.json", "decode_manifest.json", f"{verify_phase}_manifest.json"):
        result = subprocess.run(
            [sys.executable, "-I", str(relocated / "ple_demo.py"), "--manifest", str(relocated / "ple" / manifest)],
            cwd=tmp_path, capture_output=True, text=True, check=True,
        )
        assert json.loads(result.stdout)["matched_expected"] is True
    assert (relocated / "ple/prefill/layer_0.packed.bin").read_bytes() == packed.numpy().tobytes()
    assert (relocated / "ple/prefill/layer_0.exponent.bin").read_bytes() == exponent.numpy().tobytes()
    assert not (relocated / "mtp").exists()
    verify_manifest = json.loads((relocated / f"ple/{verify_phase}_manifest.json").read_text())
    assert verify_manifest["samples"][0]["component"] == verify_phase
    assert (relocated / f"ple/{verify_phase}/layer_0.packed.bin").read_bytes() == packed.numpy().tobytes()


@pytest.mark.parametrize("verify_phase", ["decode_m2", "decode_m4"])
def test_mtp_group_requires_and_captures_all_three_graphs(session, tmp_path, verify_phase):
    capture = GoldenCapture(tmp_path / "golden", ["mtp"])
    capture.required_graphs.update({"mtp_prefill", "mtp_decode", verify_phase})
    cache = torch.zeros(2, 1, 1, 64, 4)
    for name in ("mtp_prefill", "mtp_decode", verify_phase):
        with capture.step("mtp", session, graph_name=name, runtime_state={"page_kv_0": cache}):
            session(torch.ones(1, 2))
            cache.add_(1)
    artifacts = capture.finish()
    assert len(artifacts["mtp"]) == 3
    records = json.loads(capture.manifest_path.read_text())["steps"]
    assert {row["graph"] for row in records} == capture.required_graphs
    assert all(row["runtime_before"] and row["runtime_after"] for row in records)


@pytest.mark.parametrize("verify_phase", ["decode_m2", "decode_m4"])
def test_mtp_group_rejects_missing_prefill_or_verify_golden(session, tmp_path, verify_phase):
    capture = GoldenCapture(tmp_path / "golden", ["mtp"])
    capture.required_graphs.update({"mtp_prefill", "mtp_decode", verify_phase})
    with capture.step("mtp", session, graph_name="mtp_decode"):
        session(torch.ones(1, 2))
    with pytest.raises(ValueError, match=f"{verify_phase}.*mtp_prefill"):
        capture.finish()


def _selected_kv_graph(tmp_path, inputs, outputs):
    node = helper.make_node(
        "SparseAttentionSelectKV", inputs, outputs, name="select_kv", domain="ai.houmo.xh2a",
        q_bit=16, k_bit=16, compression_ratio=4, pool_size=4, candidate_topk=2, score_scale=0.125,
    )
    graph = helper.make_graph([node], "selected_kv_ports", [], [])
    path = tmp_path / "selected.onnx"
    onnx.save(helper.make_model(graph), path)
    return path


def test_selected_kv_golden_proof_records_cache_abi_and_ports(tmp_path):
    from xhmodel_merak.xh_llm.models.qwen4_exp.golden_package import _selected_kv_proof

    inputs = ["query", "index_keys", "key_cache", "past", "", "current", "value_cache"]
    outputs = ["selected_key", "selected_value"]
    path = _selected_kv_graph(tmp_path, inputs, outputs)
    proof = _selected_kv_proof(path, expected_score_mode="te")
    assert proof["node_count"] == 1
    record, = proof["nodes"]
    assert record["valid_length_input"] == "past"
    assert record["current_length_input"] == "current"
    assert record["abi"] == "raw_history_current_length_gqa"
    assert record["key_cache_input"] == "key_cache"
    assert record["value_cache_input"] == "value_cache"
    assert record["selected_key_output"] == "selected_key"
    assert record["selected_value_output"] == "selected_value"


@pytest.mark.parametrize(
    "inputs,outputs",
    [
        (["q", "index", "key", "past", "", "current", "value"], ["key_out"]),
        (["q", "index", "key", "past", "", "current", ""], ["key_out", "value_out"]),
        (["q", "index", "key", "past", "", "", "value"], ["key_out", "value_out"]),
        (["q", "index", "key", "", "", "current", "value"], ["key_out", "value_out"]),
        (["q", "index", "", "past", "", "current", "value"], ["key_out", "value_out"]),
        (["q", "index", "key", "past", "", "current", "value"], ["key_out", ""]),
    ],
    ids=["gqa_one_output", "missing_value", "gqa_missing_current",
         "missing_history", "missing_key", "missing_value_output"],
)
def test_selected_kv_golden_proof_rejects_inconsistent_cache_ports(tmp_path, inputs, outputs):
    from xhmodel_merak.xh_llm.models.qwen4_exp.golden_package import _selected_kv_proof

    path = _selected_kv_graph(tmp_path, inputs, outputs)
    with pytest.raises(ValueError, match="SelectedKV requires"):
        _selected_kv_proof(path, expected_score_mode="te")


@pytest.mark.parametrize("golden,explicit_dir", [(False, True), (True, False), (False, False)])
def test_qwen4_generation_report_stays_model_owned(tmp_path, monkeypatch, golden, explicit_dir):
    from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import HMONNXQuickTestResult
    from xhmodel_merak.xh_llm.models.qwen4_exp import hmonnx_validation as validation

    result = HMONNXQuickTestResult(
        meta_file=str(tmp_path / "golden_meta_info.json"), output_text="回答", output_tokens=2,
        latency_s=0.5, tokens_per_second=4.0,
    )
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(validation, "_generate", generate)
    output = tmp_path / "report" if explicit_dir else None
    actual = validation.hmonnx_generate(
        meta_file=result.meta_file, prompt="问题", golden=golden,
        golden_output_dir=output, auto_offload=True,
    )
    assert actual is result
    assert calls == [{"meta_file": result.meta_file, "prompt": "问题", "golden": golden, "auto_offload": True}]
    target = output or tmp_path / "golden"
    if golden or explicit_dir:
        report = json.loads((target / "golden_output.json").read_text())
        assert report["output_text"] == "回答" and report["prompt"] == "问题"
        assert report["auto_offload"] is True and report["golden"] is golden
        assert report["latency_s"] == 0.5 and report["output_tokens"] == 2
        assert (target / "golden_output.txt").read_text() == "回答\n"
    else:
        assert not target.exists()
