"""Request-scoped, per-FX-node HMONNX determinism tracing.

This diagnostic path is intentionally opt-in.  It replaces the HMONNX V2
execution wrapper with HMONNX's own FX interpreter only when
``VLLM_MERAK_HMONNX_OP_TRACE_JSONL`` is set, then fingerprints every
call_module/call_function/call_method result immediately after the node runs.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from torch import fx
from xhquant.xhonnxruntime.fx_interpreter import FXInterpreter
from xhquant.xhonnxruntime.hook import Hook

from .determinism_trace import emit_events, fingerprint

OP_TRACE_ENV = "VLLM_MERAK_HMONNX_OP_TRACE_JSONL"
OP_TRACE_MAX_REQUESTS_ENV = "VLLM_MERAK_HMONNX_OP_TRACE_MAX_REQUESTS"


@dataclass
class _TraceScope:
    stage: str
    request_id: str | None
    invocation: int
    expected_nodes: int
    rows: list[tuple[str, str | None, dict[str, Any]]] = field(
        default_factory=list
    )


_active_scope: ContextVar[_TraceScope | None] = ContextVar(
    "vllm_merak_hmonnx_node_trace_scope",
    default=None,
)
_claim_lock = threading.Lock()
_stage_requests: dict[str, list[str]] = {}
_request_invocations: dict[tuple[str, str], int] = {}
_anonymous_invocations: dict[str, int] = {}
_expected_nodes: dict[str, int] = {}


def op_trace_enabled() -> bool:
    return bool(os.environ.get(OP_TRACE_ENV, "").strip())


def _max_requests() -> int:
    raw = os.environ.get(OP_TRACE_MAX_REQUESTS_ENV, "2").strip()
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ValueError(
            f"{OP_TRACE_MAX_REQUESTS_ENV} must be an integer, got {raw!r}"
        ) from exc


def _claim_invocation(
    stage: str,
    request_id: str | None,
) -> int | None:
    with _claim_lock:
        if request_id is None:
            invocation = _anonymous_invocations.get(stage, 0)
            if invocation >= _max_requests():
                return None
            _anonymous_invocations[stage] = invocation + 1
            return invocation

        requests = _stage_requests.setdefault(stage, [])
        if request_id not in requests:
            if len(requests) >= _max_requests():
                return None
            requests.append(request_id)
        key = (stage, request_id)
        invocation = _request_invocations.get(key, 0)
        # One prefill/visual invocation and the first decode invocation are
        # sufficient for a cross-request, same-logical-state comparison.
        if invocation >= 1:
            return None
        _request_invocations[key] = invocation + 1
        return requests.index(request_id)


class _NodeFingerprintHook(Hook):
    priority = 0

    def __init__(self, stage: str, node_indices: dict[str, int]) -> None:
        self.stage = stage
        self.node_indices = node_indices

    def after_node_run(
        self,
        graph_module: fx.GraphModule,
        graph: fx.Graph,
        node: fx.Node,
        output: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        del graph, args, kwargs
        scope = _active_scope.get()
        if scope is None or scope.stage != self.stage:
            return output
        if node.op not in {"call_module", "call_function", "call_method"}:
            return output

        backend_type = None
        output_names: list[str] = []
        if node.op == "call_module":
            module = graph_module.get_submodule(str(node.target))
            backend_type = type(module).__name__
            output_names = [
                str(name) for name in getattr(module, "output_names", ())
            ]

        scope.rows.append(
            (
                "hmonnx_node_output",
                scope.request_id,
                {
                    "stage": scope.stage,
                    "invocation": scope.invocation,
                    "node_index": self.node_indices[node.name],
                    "node_name": node.name,
                    "node_op": node.op,
                    "target": str(node.target),
                    "backend_type": backend_type,
                    "output_names": output_names,
                    "output": fingerprint(output),
                },
            )
        )
        return output


class _TracingFXInterpreter(FXInterpreter):
    def __init__(
        self,
        graph_module: fx.GraphModule,
        *,
        device: Any,
        stage: str,
    ) -> None:
        super().__init__(graph_module, exec_device=device)
        self._device = device
        operational_nodes = [
            node
            for node in graph_module.graph.nodes
            if node.op in {"call_module", "call_function", "call_method"}
        ]
        node_indices = {
            node.name: index for index, node in enumerate(operational_nodes)
        }
        _expected_nodes[stage] = len(operational_nodes)
        self.register_hooks([_NodeFingerprintHook(stage, node_indices)])

    @property
    def device(self) -> Any:
        return self._device


def _install_session(stage: str, component: Any) -> bool:
    session = getattr(component, "hmonnx_session", None)
    graph_module = getattr(session, "graph_module", None)
    if session is None or graph_module is None:
        return False
    session_config = getattr(session, "_session_config", None)
    if bool(getattr(session_config, "enable_cuda_graph", False)):
        raise RuntimeError(
            "Per-node HMONNX tracing requires CUDA Graph to be disabled."
        )
    if bool(getattr(session_config, "enable_auto_offload", False)):
        raise RuntimeError(
            "Per-node HMONNX tracing requires auto-offload to be disabled."
        )
    current_interpreter = getattr(session, "interpreter", None)
    device = getattr(current_interpreter, "device", getattr(component, "device", None))
    session.interpreter = _TracingFXInterpreter(
        graph_module,
        device=device,
        stage=stage,
    )
    return True


def install_hmonnx_op_tracing(hmonnx_model: Any) -> None:
    if not op_trace_enabled():
        return
    installed = {
        "prefill": _install_session(
            "prefill", getattr(hmonnx_model, "prefill_model", None)
        ),
        "decode": _install_session(
            "decode", getattr(hmonnx_model, "decode_model", None)
        ),
        "visual": _install_session(
            "visual", getattr(hmonnx_model, "visual", None)
        ),
    }
    missing = sorted(stage for stage, present in installed.items() if not present)
    if missing:
        raise RuntimeError(
            "HMONNX per-node tracing could not install stages: "
            + ", ".join(missing)
        )


@contextmanager
def hmonnx_op_trace_scope(
    stage: str,
    *,
    request_id: str | None,
) -> Iterator[None]:
    if not op_trace_enabled():
        yield
        return
    invocation = _claim_invocation(stage, request_id)
    if invocation is None:
        yield
        return

    scope = _TraceScope(
        stage=stage,
        request_id=request_id,
        invocation=invocation,
        expected_nodes=_expected_nodes.get(stage, 0),
    )
    token = _active_scope.set(scope)
    try:
        yield
    finally:
        _active_scope.reset(token)
        scope.rows.append(
            (
                "hmonnx_node_trace_summary",
                request_id,
                {
                    "stage": stage,
                    "invocation": invocation,
                    "expected_nodes": scope.expected_nodes,
                    "recorded_nodes": len(scope.rows),
                },
            )
        )
        emit_events(scope.rows, sink_env=OP_TRACE_ENV)


__all__ = [
    "OP_TRACE_ENV",
    "hmonnx_op_trace_scope",
    "install_hmonnx_op_tracing",
    "op_trace_enabled",
]
