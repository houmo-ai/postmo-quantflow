from __future__ import annotations

import json

import torch

from vllm_merak.determinism_trace import (
    TRACE_ENV,
    emit_event,
    emit_events,
    fingerprint,
    tensor_fingerprint,
)


def test_tensor_fingerprint_is_byte_exact() -> None:
    first = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    same = first.clone()
    changed = first.clone()
    changed[0, 1] = 3.0

    assert tensor_fingerprint(first)["sha256"] == tensor_fingerprint(same)["sha256"]
    assert tensor_fingerprint(first)["sha256"] != tensor_fingerprint(changed)["sha256"]


def test_tensor_fingerprint_handles_noncontiguous_subclass() -> None:
    class TensorSubclass(torch.Tensor):
        pass

    value = torch.arange(12, dtype=torch.float16).reshape(3, 4).T
    subclass = value.as_subclass(TensorSubclass)

    result = tensor_fingerprint(subclass)

    assert result["shape"] == [4, 3]
    assert result["nbytes"] == 24
    assert result["sha256"] == tensor_fingerprint(value)["sha256"]


def test_tensor_fingerprint_handles_size_one_final_dimension_stride() -> None:
    value = torch.empty_strided(
        (2, 3, 1),
        (3, 1, 99),
        dtype=torch.float16,
    )
    value.copy_(torch.arange(6, dtype=torch.float16).reshape(2, 3, 1))

    result = tensor_fingerprint(value)

    assert value.is_contiguous()
    assert result["nbytes"] == 12
    assert result["sha256"] == tensor_fingerprint(value.clone())["sha256"]


def test_nested_fingerprint_preserves_names_and_container_shape() -> None:
    result = fingerprint({"b": [torch.tensor([2])], "a": (torch.tensor([1]),)})

    assert list(result) == ["a", "b"]
    assert result["a"]["kind"] == "tuple"
    assert result["b"]["kind"] == "list"


def test_emit_event_is_disabled_without_sink(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(TRACE_ENV, raising=False)
    assert not emit_event("disabled", payload={"value": 1})
    assert not (tmp_path / "trace.jsonl").exists()


def test_emit_event_appends_jsonl(monkeypatch, tmp_path) -> None:
    sink = tmp_path / "trace.jsonl"
    monkeypatch.setenv(TRACE_ENV, str(sink))

    assert emit_event("first", request_id="req-1", payload={"value": 1})
    assert emit_event("second", request_id="req-2", payload={"value": 2})

    rows = [json.loads(line) for line in sink.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["first", "second"]
    assert [row["request_id"] for row in rows] == ["req-1", "req-2"]
    assert rows[0]["sequence"] < rows[1]["sequence"]


def test_emit_events_supports_an_alternate_sink(monkeypatch, tmp_path) -> None:
    sink = tmp_path / "nodes.jsonl"
    monkeypatch.setenv("NODE_TRACE_SINK", str(sink))

    assert (
        emit_events(
            [
                ("node", "req-1", {"index": 0}),
                ("node", "req-1", {"index": 1}),
            ],
            sink_env="NODE_TRACE_SINK",
        )
        == 2
    )

    rows = [json.loads(line) for line in sink.read_text().splitlines()]
    assert [row["payload"]["index"] for row in rows] == [0, 1]
