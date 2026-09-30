"""Opt-in, run-scoped JSONL evidence for Gemma4 hardware validation.

Trace events are serialized into memory on the request path and written only by
an explicit :func:`flush_traces` call (the harness flushes after engine
shutdown). With no sink configured, no envelope or JSON payload is built.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import threading
import time
from collections.abc import Mapping
from typing import Any

GEMMA4_TRACE_ENV = "VLLM_MERAK_GEMMA4_TRACE_JSONL"
KV_TRACE_ENV = "VLLM_MERAK_KV_TRACE_JSONL"
TRACE_RUN_ID_ENV = "VLLM_MERAK_TRACE_RUN_ID"
TRACE_SCHEMA = "vllm-merak.gemma4.trace/v1"

_buffers: dict[str, bytearray] = {}
_buffers_lock = threading.Lock()


def trace_enabled(env_name: str) -> bool:
    """Return whether an explicit non-empty JSONL sink is configured."""

    return bool(os.environ.get(env_name, "").strip())


def _rank(env_name: str) -> int:
    value = os.environ.get(env_name, "").strip()
    return int(value) if value else -1


def emit_trace(
    env_name: str,
    *,
    component: str,
    event: str,
    payload: Mapping[str, Any],
) -> bool:
    """Buffer one stable-envelope JSON object when ``env_name`` is enabled."""

    sink = os.environ.get(env_name, "").strip()
    if not sink:
        return False
    run_id = os.environ.get(TRACE_RUN_ID_ENV, "").strip()
    if not run_id:
        raise RuntimeError("Gemma4 tracing requires an explicit run ID")

    row = {
        "schema": TRACE_SCHEMA,
        "timestamp_ns": time.time_ns(),
        "pid": os.getpid(),
        "rank": _rank("RANK"),
        "local_rank": _rank("LOCAL_RANK"),
        "run_id": run_id,
        "component": component,
        "event": event,
        **payload,
    }
    encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
    with _buffers_lock:
        _buffers.setdefault(sink, bytearray()).extend(encoded)
    return True


def _write_all(fd: int, data: bytes | bytearray) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("trace sink write made no forward progress")
        view = view[written:]


def flush_traces() -> None:
    """Drain all process-local trace buffers, tolerating short writes."""

    with _buffers_lock:
        pending = tuple(_buffers.items())
        _buffers.clear()
    for sink, encoded in pending:
        if not encoded:
            continue
        fd = os.open(sink, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            _write_all(fd, encoded)
        finally:
            os.close(fd)


atexit.register(flush_traces)


__all__ = [
    "GEMMA4_TRACE_ENV",
    "KV_TRACE_ENV",
    "TRACE_RUN_ID_ENV",
    "emit_trace",
    "flush_traces",
    "trace_enabled",
]
