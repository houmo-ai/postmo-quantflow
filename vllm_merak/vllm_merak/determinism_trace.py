"""Opt-in tensor fingerprints for Merak determinism investigations.

The request path is unchanged unless ``VLLM_MERAK_DETERMINISM_TRACE_JSONL`` is
set.  Diagnostic runs intentionally synchronize tensors to the host so SHA256
fingerprints compare the exact bytes consumed by HMONNX.
"""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

TRACE_ENV = "VLLM_MERAK_DETERMINISM_TRACE_JSONL"
TRACE_SCHEMA = "vllm-merak.determinism.trace/v1"

_sequence = itertools.count()
_write_lock = threading.Lock()


def trace_enabled() -> bool:
    return bool(os.environ.get(TRACE_ENV, "").strip())


def _plain_tensor(value: torch.Tensor) -> torch.Tensor:
    """Unwrap tensor subclasses while preserving the represented bytes."""

    if type(value) is not torch.Tensor:
        try:
            value = value.as_subclass(torch.Tensor)
        except RuntimeError:
            pass
    data = getattr(value, "data", None)
    if isinstance(data, torch.Tensor) and type(value) is not torch.Tensor:
        value = data
    return value.detach()


def tensor_fingerprint(value: torch.Tensor) -> dict[str, Any]:
    tensor = _plain_tensor(value)
    cpu = tensor.to(device="cpu").contiguous()
    if type(cpu) is not torch.Tensor:
        cpu = cpu.as_subclass(torch.Tensor)
    if not cpu.is_contiguous():
        cpu = cpu.clone(memory_format=torch.contiguous_format)
    # A size-1 last dimension may legally retain a non-unit stride while
    # PyTorch still reports the tensor as contiguous. Flattening first gives
    # view(dtype=uint8) the unit final stride it requires.
    raw = cpu.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
    return {
        "kind": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "numel": tensor.numel(),
        "nbytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def fingerprint(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_fingerprint(value)
    qdata = getattr(value, "qdata", None)
    if isinstance(qdata, torch.Tensor):
        q_param = getattr(value, "q_param", None)
        return {
            "kind": type(value).__name__,
            "qdata": tensor_fingerprint(qdata),
            "scale": fingerprint(getattr(q_param, "scale", None)),
            "zero_point": fingerprint(getattr(q_param, "zero_point", None)),
            "quant_spec": repr(getattr(q_param, "quant_spec", None)),
        }
    data = getattr(value, "data", None)
    if isinstance(data, torch.Tensor):
        return {
            "kind": type(value).__name__,
            "data": tensor_fingerprint(data),
        }
    if isinstance(value, Mapping):
        return {
            str(key): fingerprint(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [fingerprint(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [fingerprint(item) for item in value]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"kind": type(value).__name__, "repr": repr(value)}


def emit_events(
    events: Sequence[tuple[str, str | None, Mapping[str, Any] | None]],
    *,
    sink_env: str = TRACE_ENV,
) -> int:
    sink = os.environ.get(sink_env, "").strip()
    if not sink:
        return 0

    encoded_rows = []
    for event, request_id, payload in events:
        row = {
            "schema": TRACE_SCHEMA,
            "sequence": next(_sequence),
            "timestamp_ns": time.time_ns(),
            "pid": os.getpid(),
            "event": event,
            "request_id": request_id,
            "payload": dict(payload or {}),
        }
        encoded_rows.append(
            (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        )
    if not encoded_rows:
        return 0

    Path(sink).parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        fd = os.open(sink, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            for encoded in encoded_rows:
                view = memoryview(encoded)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError(
                            "determinism trace write made no progress"
                        )
                    view = view[written:]
        finally:
            os.close(fd)
    return len(encoded_rows)


def emit_event(
    event: str,
    *,
    request_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
    sink_env: str = TRACE_ENV,
) -> bool:
    emitted = emit_events(
        [(event, request_id, payload)],
        sink_env=sink_env,
    )
    return emitted == 1


def fingerprint_named_tensors(
    values: Mapping[str, Any] | Sequence[Any],
) -> dict[str, Any]:
    if isinstance(values, Mapping):
        return {str(name): fingerprint(value) for name, value in values.items()}
    return {str(index): fingerprint(value) for index, value in enumerate(values)}


__all__ = [
    "TRACE_ENV",
    "emit_event",
    "emit_events",
    "fingerprint",
    "fingerprint_named_tensors",
    "tensor_fingerprint",
    "trace_enabled",
]
