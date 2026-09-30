from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import fx, nn

import vllm_merak.hmonnx_op_trace as trace


class _ToyModule(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(value + 1)


def _reset_trace_state() -> None:
    trace._stage_requests.clear()
    trace._request_invocations.clear()
    trace._anonymous_invocations.clear()
    trace._expected_nodes.clear()


def test_node_trace_records_every_operational_fx_node(monkeypatch, tmp_path) -> None:
    _reset_trace_state()
    sink = tmp_path / "nodes.jsonl"
    monkeypatch.setenv(trace.OP_TRACE_ENV, str(sink))
    graph_module = fx.symbolic_trace(_ToyModule())
    interpreter = trace._TracingFXInterpreter(
        graph_module,
        device=torch.device("cpu"),
        stage="prefill",
    )

    with trace.hmonnx_op_trace_scope("prefill", request_id="req-1"):
        output = interpreter(torch.tensor([-2.0, 1.0]))

    assert torch.equal(output, torch.tensor([0.0, 2.0]))
    rows = [json.loads(line) for line in sink.read_text().splitlines()]
    node_rows = [
        row for row in rows if row["event"] == "hmonnx_node_output"
    ]
    summary = rows[-1]["payload"]
    assert [row["payload"]["node_op"] for row in node_rows] == [
        "call_function",
        "call_function",
    ]
    assert summary["expected_nodes"] == 2
    assert summary["recorded_nodes"] == 2


def test_install_rejects_cuda_graph(monkeypatch, tmp_path) -> None:
    _reset_trace_state()
    monkeypatch.setenv(trace.OP_TRACE_ENV, str(tmp_path / "nodes.jsonl"))
    component = SimpleNamespace(
        hmonnx_session=SimpleNamespace(
            graph_module=fx.symbolic_trace(_ToyModule()),
            _session_config=SimpleNamespace(
                enable_cuda_graph=True,
                enable_auto_offload=False,
            ),
        )
    )

    with pytest.raises(RuntimeError, match="CUDA Graph"):
        trace._install_session("prefill", component)
