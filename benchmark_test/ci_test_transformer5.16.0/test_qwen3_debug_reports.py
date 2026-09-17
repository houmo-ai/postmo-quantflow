"""Small-tensor regression tests for Qwen3 diagnostic capture and reports."""
import csv
import importlib
import json
import queue

import pytest
import torch

pytestmark = pytest.mark.transformer516
PREFIX = "examples_merak.llm.qwen3.debug_scripts."
REPORTS = ["qwen3_layerwise_diff", "qwen3_layerwise_diff_2", "qwen3_layerwise_diff_3", "qwen3_quanted_hmonnx_layerwise_diff"]


@pytest.mark.parametrize("name", REPORTS)
def test_capture_report_detects_numeric_missing_and_shape_errors(name, tmp_path):
    module = importlib.import_module(PREFIX + name)
    reference, candidate = module.CaptureStore(), module.CaptureStore()
    layer = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2))
    if name == REPORTS[-1]:
        hook = module._tag_hook(reference, 0)
    else:
        hook = module._module_output_hook(reference, "layer.000.output")
    handle = layer.register_forward_hook(hook, with_kwargs=name != REPORTS[-1])
    try:
        layer(torch.tensor([[[1., 2.]]]))
    finally:
        handle.remove()
    candidate.record("layer.000.output", torch.tensor([[[1., 2.]]]))
    assert module._build_report(reference, candidate, metadata={})["summary"]["exact"]
    reference.record("layer.001.output", torch.tensor([[[2., 4.]]]))
    candidate.record("layer.001.output", torch.tensor([[[2., 6.]]]))
    reference.record("missing", torch.ones(1))
    reference.record("shape", torch.ones(2))
    candidate.record("shape", torch.ones(3))
    reference.record("logits", torch.tensor([[[1., 3.]]]))
    candidate.record("logits", torch.tensor([[[4., 1.]]]))
    report = module._build_report(reference, candidate, metadata={})
    rows = {row["name"]: row for row in report["comparisons"]}
    assert not report["summary"]["exact"]
    assert report["summary"]["first_bad_layer"]["prefill"] == 1
    assert rows["prefill.layer.001.output"]["max_abs"] == 2
    assert rows["prefill.missing"]["status"] == "missing"
    assert rows["prefill.shape"]["status"] == "shape_mismatch"
    assert rows["prefill.logits"]["argmax_equal"] is False
    json_path, csv_path = module._write_report(report, tmp_path)
    assert json.loads(json_path.read_text())["summary"] == report["summary"]
    with csv_path.open() as stream:
        assert len(list(csv.DictReader(stream))) == len(rows)


@pytest.mark.parametrize("name", REPORTS[1:])
def test_reports_identify_output_token_divergence(name):
    module = importlib.import_module(PREFIX + name)
    mismatch = module._first_output_token_mismatch([4, 5, 6], [4, 7], input_length=3, max_decode_steps=2)
    assert mismatch["generation_index"] == 1
    assert mismatch["position_id"] == 4
    assert mismatch["prediction_stage"] == "decode.0000"
    assert module._first_output_token_mismatch([4], [4], input_length=3, max_decode_steps=1) is None


def test_teacher_token_stream_propagates_failure_and_stops():
    module = importlib.import_module(PREFIX + "qwen3_layerwise_diff_2")
    items = queue.Queue()
    items.put(4)
    items.put(module._TOKEN_STREAM_END)
    assert list(module._iter_token_queue(items)) == [4]
    error = ValueError("producer failed")
    items.put(module._TokenStreamError(error))
    with pytest.raises(RuntimeError, match="native token stream failed") as caught:
        list(module._iter_token_queue(items))
    assert caught.value.__cause__ is error
    assert module._generation_budget(input_length=3, max_total_tokens=3) == 0
    with pytest.raises(ValueError, match="exceeds"):
        module._generation_budget(input_length=4, max_total_tokens=3)


def test_generate_hooks_label_prefill_decode_and_reject_extra_calls():
    module = importlib.import_module(PREFIX + "qwen3_layerwise_diff_3")
    root = torch.nn.Identity()
    store = module.CaptureStore()
    snapshots = []
    handles, state = module._register_generation_capture_hooks(
        root, store, input_length=3, max_decode_steps=1,
        cache_recorder=lambda output, length, start: snapshots.append((length, start)),
    )
    try:
        root(torch.ones(1, 3, 4))
        root(torch.ones(1, 1, 4) * 2)
        assert state.calls == 2
        assert snapshots == [(3, 0), (4, 3)]
        assert set(store.tensors) == {"prefill.logits", "decode.logits"}
        assert store.tensors["prefill.logits"].shape == (1, 1, 4)
        with pytest.raises(RuntimeError, match="unexpected decode"):
            root(torch.ones(1, 1, 4))
    finally:
        module._remove_hooks(handles)
    root(torch.ones(1, 1, 4))
    assert state.calls == 2


def test_quanted_capture_trims_padding_and_rejects_incomplete_layers():
    module = importlib.import_module(PREFIX + REPORTS[-1])
    assert module._prefill_chunks(6, 4) == [("prefill.0000", 0, 4), ("prefill.0001", 4, 6)]
    with pytest.raises(ValueError):
        module._prefill_chunks(6, 0)
    store = module.CaptureStore()
    store.set_stage("prefill.0001", valid_sequence_length=2)
    tensor = torch.arange(12.).reshape(1, 4, 3)
    module._record_layer_boundary(store, 0, tensor)
    torch.testing.assert_close(store.tensors["prefill.0001.layer.000.output"].tensor, tensor[:, :2])
    torch.testing.assert_close(module._last_valid_token_tensor(store, tensor, label="logits"), tensor[:, 1:2])
    module._validate_captured_layers(store, stages=[store.stage], expected_layers=1, label="test")
    with pytest.raises(RuntimeError, match="missing"):
        module._validate_captured_layers(store, stages=[store.stage], expected_layers=2, label="test")
    report = {"summary": {"exact": True}}
    module._apply_report_integrity(report, quantization_contract_exact=False, lm_head_parameters_exact=True)
    assert report["summary"]["exact"] is False
