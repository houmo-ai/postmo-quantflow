"""Lossless request and response auditing for the Merak vLLM server.

The middleware is deliberately implemented at the ASGI boundary. It observes
messages without buffering or replacing a StreamingResponse, so the bytes sent
to an OpenAI-compatible client remain unchanged.
"""

from __future__ import annotations

import base64
import codecs
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Awaitable, Callable
import uuid

AUDIT_LOG_MODE_ENV = "VLLM_MERAK_AUDIT_LOG_MODE"
AUDIT_LOG_DIR_ENV = "VLLM_MERAK_AUDIT_LOG_DIR"
AUDIT_LOG_MIDDLEWARE = "vllm_merak.audit_logging.AuditLogMiddleware"

_AUDIT_MODES = frozenset({"off", "aggregate", "sse", "both"})
_AUDIT_PATH = "/v1/chat/completions"
_EVENT_SEPARATOR = re.compile(r"\r?\n\r?\n")
_FILE_LOCKS: dict[Path, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()
logger = logging.getLogger(__name__)

ASGIApp = Callable[
    [
        dict[str, Any],
        Callable[[], Awaitable[dict[str, Any]]],
        Callable[[dict[str, Any]], Awaitable[None]],
    ],
    Awaitable[None],
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _utf8(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _parse_json_body(body: bytes) -> tuple[Any | None, str | None]:
    if not body:
        return None, None
    try:
        return json.loads(body), None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


class _JsonlWriter:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self._directory_ready = False
        self._directory_lock = threading.Lock()

    def _ensure_directory(self) -> None:
        if self._directory_ready:
            return
        with self._directory_lock:
            if self._directory_ready:
                return
            self.log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.log_dir, 0o700)
            self._directory_ready = True

    def write(self, stream: str, record: dict[str, Any]) -> None:
        self._ensure_directory()
        date = datetime.now().strftime("%Y%m%d")
        path = self.log_dir / f"audit-{stream}-{date}-p{os.getpid()}.jsonl"
        payload = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with _FILE_LOCKS_GUARD:
            lock = _FILE_LOCKS.setdefault(path, threading.Lock())
        with lock:
            fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    view = view[written:]
            finally:
                os.close(fd)


def _new_choice(index: int) -> dict[str, Any]:
    return {
        "index": index,
        "role": None,
        "content": "",
        "reasoning": "",
        "tool_calls": {},
        "finish_reason": None,
    }


class _SSEAggregator:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer = ""
        self.id: str | None = None
        self.model: str | None = None
        self.usage: Any | None = None
        self.done = False
        self.choices: dict[int, dict[str, Any]] = {}
        self.parse_errors: list[str] = []

    def feed(self, body: bytes) -> None:
        self._buffer += self._decoder.decode(body)
        self._drain_events()

    def finish(self) -> None:
        self._buffer += self._decoder.decode(b"", final=True)
        self._drain_events()
        if self._buffer.strip():
            self._consume_event(self._buffer)
        self._buffer = ""

    def _drain_events(self) -> None:
        while match := _EVENT_SEPARATOR.search(self._buffer):
            event = self._buffer[: match.start()]
            self._buffer = self._buffer[match.end() :]
            self._consume_event(event)

    def _consume_event(self, event: str) -> None:
        data_lines: list[str] = []
        for line in event.splitlines():
            if not line.startswith("data:"):
                continue
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
        if not data_lines:
            return
        data = "\n".join(data_lines)
        if data == "[DONE]":
            self.done = True
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            self.parse_errors.append(f"JSONDecodeError: {exc}")
            return
        if not isinstance(chunk, dict):
            return
        if isinstance(chunk.get("id"), str):
            self.id = chunk["id"]
        if isinstance(chunk.get("model"), str):
            self.model = chunk["model"]
        if chunk.get("usage") is not None:
            self.usage = chunk["usage"]
        for raw_choice in chunk.get("choices") or []:
            if not isinstance(raw_choice, dict):
                continue
            index = raw_choice.get("index", 0)
            if not isinstance(index, int):
                continue
            choice = self.choices.setdefault(index, _new_choice(index))
            delta = raw_choice.get("delta") or {}
            if isinstance(delta, dict):
                role = delta.get("role")
                if isinstance(role, str):
                    choice["role"] = role
                content = delta.get("content")
                if isinstance(content, str):
                    choice["content"] += content
                reasoning = delta.get("reasoning")
                if not isinstance(reasoning, str):
                    reasoning = delta.get("reasoning_content")
                if isinstance(reasoning, str):
                    choice["reasoning"] += reasoning
                self._append_tool_calls(choice, delta.get("tool_calls"))
            if raw_choice.get("finish_reason") is not None:
                choice["finish_reason"] = raw_choice["finish_reason"]

    @staticmethod
    def _append_tool_calls(choice: dict[str, Any], raw_tool_calls: Any) -> None:
        if not isinstance(raw_tool_calls, list):
            return
        tool_calls: dict[int, dict[str, Any]] = choice["tool_calls"]
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            index = raw_call.get("index", 0)
            if not isinstance(index, int):
                continue
            tool_call = tool_calls.setdefault(
                index,
                {
                    "index": index,
                    "id": "",
                    "type": "",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if isinstance(raw_call.get("id"), str):
                tool_call["id"] += raw_call["id"]
            if isinstance(raw_call.get("type"), str):
                tool_call["type"] += raw_call["type"]
            function = raw_call.get("function")
            if isinstance(function, dict):
                if isinstance(function.get("name"), str):
                    tool_call["function"]["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    tool_call["function"]["arguments"] += function["arguments"]

    def result(self) -> dict[str, Any]:
        choices: list[dict[str, Any]] = []
        for index in sorted(self.choices):
            raw_choice = self.choices[index]
            message: dict[str, Any] = {}
            if raw_choice["role"] is not None:
                message["role"] = raw_choice["role"]
            if raw_choice["content"]:
                message["content"] = raw_choice["content"]
            if raw_choice["reasoning"]:
                message["reasoning"] = raw_choice["reasoning"]
            if raw_choice["tool_calls"]:
                message["tool_calls"] = [
                    raw_choice["tool_calls"][tool_index]
                    for tool_index in sorted(raw_choice["tool_calls"])
                ]
            choices.append(
                {
                    "index": index,
                    "message": message,
                    "finish_reason": raw_choice["finish_reason"],
                }
            )
        result: dict[str, Any] = {
            "stream": True,
            "id": self.id,
            "model": self.model,
            "choices": choices,
            "usage": self.usage,
            "done": self.done,
        }
        if self.parse_errors:
            result["parse_errors"] = self.parse_errors
        return result


class AuditLogMiddleware:
    """Record OpenAI chat requests and responses without changing ASGI output."""

    def __init__(
        self,
        app: ASGIApp,
        mode: str | None = None,
        log_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.app = app
        self.mode = mode or os.getenv(AUDIT_LOG_MODE_ENV, "off")
        if self.mode not in _AUDIT_MODES:
            raise ValueError(
                f"invalid audit log mode {self.mode!r}; expected one of "
                f"{sorted(_AUDIT_MODES)}"
            )
        configured_dir = log_dir or os.getenv(AUDIT_LOG_DIR_ENV, "logs/audit")
        self.writer = _JsonlWriter(Path(configured_dir))
        self._write_failed = False

    def _write(self, stream: str, record: dict[str, Any]) -> None:
        if self._write_failed:
            return
        try:
            self.writer.write(stream, record)
        except OSError:
            self._write_failed = True
            logger.exception(
                "Disabling Merak audit logging after a filesystem write failure"
            )

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if (
            self.mode == "off"
            or scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != _AUDIT_PATH
        ):
            await self.app(scope, receive, send)
            return

        audit_id = uuid.uuid4().hex
        started_at = _utc_now()
        started_ns = time.monotonic_ns()
        request_body = bytearray()
        response_body = bytearray()
        response_status: int | None = None
        response_headers: list[tuple[bytes, bytes]] = []
        is_stream = False
        response_started = False
        request_start_logged = False
        chunk_seq = 0
        error: str | None = None
        aggregator = _SSEAggregator()

        async def audit_receive() -> dict[str, Any]:
            message = await receive()
            if message.get("type") == "http.request":
                request_body.extend(message.get("body", b""))
            return message

        def elapsed_ms() -> float:
            return round((time.monotonic_ns() - started_ns) / 1_000_000, 3)

        def request_record() -> dict[str, Any]:
            body_json, json_error = _parse_json_body(bytes(request_body))
            record: dict[str, Any] = {
                "schema_version": 1,
                "record_type": "request_start",
                "audit_id": audit_id,
                "timestamp": started_at,
                "method": scope.get("method"),
                "path": scope.get("path"),
                "query_string_base64": _b64(scope.get("query_string", b"")),
                "client": list(scope["client"]) if scope.get("client") else None,
                "request_body_text": _utf8(bytes(request_body)),
                "request_body_json": body_json,
                "request_body_base64": _b64(bytes(request_body)),
            }
            if json_error is not None:
                record["request_json_error"] = json_error
            return record

        async def ensure_request_start() -> None:
            nonlocal request_start_logged
            if self.mode not in {"sse", "both"} or request_start_logged:
                return
            self._write("sse", request_record())
            request_start_logged = True

        async def audit_send(message: dict[str, Any]) -> None:
            nonlocal response_status, response_headers, is_stream
            nonlocal response_started, chunk_seq
            message_type = message.get("type")
            if message_type == "http.response.start":
                response_status = message.get("status")
                response_headers = list(message.get("headers") or [])
                content_type = next(
                    (
                        value.decode("latin-1")
                        for name, value in response_headers
                        if name.lower() == b"content-type"
                    ),
                    "",
                )
                is_stream = content_type.lower().startswith("text/event-stream")
                await send(message)
                response_started = True
                await ensure_request_start()
                if self.mode in {"sse", "both"}:
                    self._write(
                        "sse",
                        {
                            "schema_version": 1,
                            "record_type": "response_start",
                            "audit_id": audit_id,
                            "elapsed_ms": elapsed_ms(),
                            "status_code": response_status,
                            "headers": [
                                {
                                    "name": name.decode("latin-1"),
                                    "value": value.decode("latin-1"),
                                    "name_base64": _b64(name),
                                    "value_base64": _b64(value),
                                }
                                for name, value in response_headers
                            ],
                        },
                    )
                return

            if message_type != "http.response.body":
                await send(message)
                return

            body = message.get("body", b"")
            more_body = bool(message.get("more_body", False))
            await send(message)
            if self.mode in {"aggregate", "both"}:
                if is_stream:
                    aggregator.feed(body)
                else:
                    response_body.extend(body)
            if self.mode in {"sse", "both"}:
                self._write(
                    "sse",
                    {
                        "schema_version": 1,
                        "record_type": "response_chunk",
                        "audit_id": audit_id,
                        "seq": chunk_seq,
                        "elapsed_ms": elapsed_ms(),
                        "body_text": _utf8(body),
                        "body_base64": _b64(body),
                        "more_body": more_body,
                    },
                )
                chunk_seq += 1

        try:
            await self.app(scope, audit_receive, audit_send)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            await ensure_request_start()
            duration_ms = elapsed_ms()
            if self.mode in {"aggregate", "both"}:
                request_json, request_json_error = _parse_json_body(bytes(request_body))
                request_data: dict[str, Any] = {
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "query_string_base64": _b64(scope.get("query_string", b"")),
                    "client": list(scope["client"]) if scope.get("client") else None,
                    "body_text": _utf8(bytes(request_body)),
                    "body_json": request_json,
                    "body_base64": _b64(bytes(request_body)),
                }
                if request_json_error is not None:
                    request_data["json_error"] = request_json_error

                if is_stream:
                    aggregator.finish()
                    response_data = aggregator.result()
                else:
                    body_json, body_json_error = _parse_json_body(bytes(response_body))
                    response_data = {
                        "stream": False,
                        "body_json": body_json,
                    }
                    if body_json_error is not None:
                        response_data["body_base64"] = _b64(bytes(response_body))
                        response_data["json_error"] = body_json_error
                response_data["status_code"] = response_status
                record = {
                    "schema_version": 1,
                    "record_type": "aggregate",
                    "audit_id": audit_id,
                    "timestamp": started_at,
                    "duration_ms": duration_ms,
                    "request": request_data,
                    "response": response_data,
                }
                if error is not None:
                    record["error"] = error
                self._write("aggregate", record)

            if self.mode in {"sse", "both"}:
                complete_record = {
                    "schema_version": 1,
                    "record_type": "request_complete",
                    "audit_id": audit_id,
                    "elapsed_ms": duration_ms,
                    "response_started": response_started,
                    "status_code": response_status,
                    "response_chunks": chunk_seq,
                }
                if error is not None:
                    complete_record["error"] = error
                self._write("sse", complete_record)


__all__ = [
    "AUDIT_LOG_DIR_ENV",
    "AUDIT_LOG_MIDDLEWARE",
    "AUDIT_LOG_MODE_ENV",
    "AuditLogMiddleware",
]
