import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import pytest

from vllm_merak.audit_logging import AuditLogMiddleware

CHAT_PATH = "/v1/chat/completions"


async def _call_asgi(
    app,
    *,
    request_body: bytes,
    response_messages: list[dict[str, Any]],
    path: str = CHAT_PATH,
) -> list[dict[str, Any]]:
    request_messages = [
        {
            "type": "http.request",
            "body": request_body,
            "more_body": False,
        }
    ]
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if request_messages:
            return request_messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message.copy())

    async def downstream(scope, receive, send) -> None:
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        assert bytes(body) == request_body
        for response_message in response_messages:
            await send(response_message.copy())

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 43210),
        "server": ("127.0.0.1", 18006),
    }
    middleware = app(downstream)
    await middleware(scope, receive, send)
    return sent


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _response_start(content_type: bytes) -> dict[str, Any]:
    return {
        "type": "http.response.start",
        "status": 200,
        "headers": [(b"content-type", content_type)],
    }


def test_off_mode_does_not_create_audit_directory(tmp_path: Path) -> None:
    request = {"model": "qwen", "messages": [{"role": "user", "content": "你好"}]}
    response_messages = [
        _response_start(b"application/json"),
        {"type": "http.response.body", "body": b"{}", "more_body": False},
    ]

    sent = asyncio.run(
        _call_asgi(
            lambda downstream: AuditLogMiddleware(
                downstream, mode="off", log_dir=tmp_path / "audit"
            ),
            request_body=json.dumps(request).encode(),
            response_messages=response_messages,
        )
    )

    assert sent == response_messages
    assert not (tmp_path / "audit").exists()


def test_aggregate_mode_preserves_request_and_reconstructs_stream(
    tmp_path: Path,
) -> None:
    request = {
        "model": "qwen3_5_9b_merak",
        "messages": [
            {"role": "user", "content": "第一轮"},
            {"role": "assistant", "content": "第一轮回复"},
            {"role": "user", "content": "继续"},
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
        "stream": True,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    request_bytes = json.dumps(
        request, ensure_ascii=False, separators=(",", ":")
    ).encode()
    sse = (
        'data: {"id":"chatcmpl-1","model":"qwen","choices":[{"index":0,'
        '"delta":{"role":"assistant","reasoning":"先想"}}]}\n\n'
        'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":'
        '{"content":"答案","tool_calls":[{"index":0,"id":"call-1",'
        '"type":"function","function":{"name":"look","arguments":"{\\"q\\":"}}]}}]}\n\n'
        'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":'
        '{"content":"完成","tool_calls":[{"index":0,"function":'
        '{"name":"up","arguments":"\\"北京\\"}"}}]},"finish_reason":"tool_calls"}],'
        '"usage":{"prompt_tokens":20,"completion_tokens":4,"total_tokens":24}}\n\n'
        "data: [DONE]\n\n"
    ).encode()
    split = sse.index(b"tool_calls") + 5
    response_messages = [
        _response_start(b"text/event-stream; charset=utf-8"),
        {"type": "http.response.body", "body": sse[:split], "more_body": True},
        {"type": "http.response.body", "body": sse[split:], "more_body": False},
    ]

    sent = asyncio.run(
        _call_asgi(
            lambda downstream: AuditLogMiddleware(
                downstream, mode="aggregate", log_dir=tmp_path
            ),
            request_body=request_bytes,
            response_messages=response_messages,
        )
    )

    assert sent == response_messages
    files = list(tmp_path.glob("audit-aggregate-*.jsonl"))
    assert len(files) == 1
    records = _jsonl(files[0])
    assert len(records) == 1
    record = records[0]
    assert record["record_type"] == "aggregate"
    assert base64.b64decode(record["request"]["body_base64"]) == request_bytes
    assert record["request"]["body_text"] == request_bytes.decode("utf-8")
    assert record["request"]["body_json"] == request
    assert record["response"]["stream"] is True
    assert record["response"]["id"] == "chatcmpl-1"
    assert record["response"]["model"] == "qwen"
    assert record["response"]["choices"] == [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "答案完成",
                "reasoning": "先想",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":"北京"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ]
    assert record["response"]["usage"] == {
        "prompt_tokens": 20,
        "completion_tokens": 4,
        "total_tokens": 24,
    }
    assert record["response"]["done"] is True


@pytest.mark.parametrize("mode", ["sse", "both"])
def test_sse_modes_preserve_exact_asgi_body_chunks(tmp_path: Path, mode: str) -> None:
    request = {"model": "qwen", "messages": [{"role": "user", "content": "你好"}]}
    request_bytes = json.dumps(request, ensure_ascii=False).encode()
    chunks = [
        b'data: {"choices":[{"delta":{"content":"\xe4\xbd\xa0"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"\xe5\xa5\xbd"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    response_messages = [
        _response_start(b"text/event-stream; charset=utf-8"),
        *[
            {
                "type": "http.response.body",
                "body": chunk,
                "more_body": index < len(chunks) - 1,
            }
            for index, chunk in enumerate(chunks)
        ],
    ]

    sent = asyncio.run(
        _call_asgi(
            lambda downstream: AuditLogMiddleware(
                downstream, mode=mode, log_dir=tmp_path
            ),
            request_body=request_bytes,
            response_messages=response_messages,
        )
    )

    assert sent == response_messages
    files = list(tmp_path.glob("audit-sse-*.jsonl"))
    assert len(files) == 1
    records = _jsonl(files[0])
    assert [record["record_type"] for record in records] == [
        "request_start",
        "response_start",
        "response_chunk",
        "response_chunk",
        "response_chunk",
        "request_complete",
    ]
    assert base64.b64decode(records[0]["request_body_base64"]) == request_bytes
    assert records[0]["request_body_text"] == request_bytes.decode("utf-8")
    assert records[1]["headers"] == [
        {
            "name": "content-type",
            "value": "text/event-stream; charset=utf-8",
            "name_base64": "Y29udGVudC10eXBl",
            "value_base64": "dGV4dC9ldmVudC1zdHJlYW07IGNoYXJzZXQ9dXRmLTg=",
        }
    ]
    logged_chunks = [
        base64.b64decode(record["body_base64"])
        for record in records
        if record["record_type"] == "response_chunk"
    ]
    assert logged_chunks == chunks
    assert [
        record["body_text"]
        for record in records
        if record["record_type"] == "response_chunk"
    ] == [chunk.decode("utf-8") for chunk in chunks]
    assert [
        record["seq"] for record in records if record["record_type"] == "response_chunk"
    ] == [0, 1, 2]


def test_non_target_endpoint_is_not_logged(tmp_path: Path) -> None:
    response_messages = [
        _response_start(b"application/json"),
        {"type": "http.response.body", "body": b'{"status":"ok"}'},
    ]

    sent = asyncio.run(
        _call_asgi(
            lambda downstream: AuditLogMiddleware(
                downstream, mode="both", log_dir=tmp_path
            ),
            request_body=b"{}",
            response_messages=response_messages,
            path="/health",
        )
    )

    assert sent == response_messages
    assert list(tmp_path.iterdir()) == []


def test_non_streaming_response_is_preserved_structurally(tmp_path: Path) -> None:
    request = {"model": "qwen", "messages": [{"role": "user", "content": "你好"}]}
    response = {
        "id": "chatcmpl-json",
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning": "思考",
                    "content": "回答",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    response_bytes = json.dumps(response, ensure_ascii=False).encode()
    response_messages = [
        _response_start(b"application/json"),
        {
            "type": "http.response.body",
            "body": response_bytes[:15],
            "more_body": True,
        },
        {
            "type": "http.response.body",
            "body": response_bytes[15:],
            "more_body": False,
        },
    ]

    asyncio.run(
        _call_asgi(
            lambda downstream: AuditLogMiddleware(
                downstream, mode="aggregate", log_dir=tmp_path
            ),
            request_body=json.dumps(request, ensure_ascii=False).encode(),
            response_messages=response_messages,
        )
    )

    record = _jsonl(next(tmp_path.glob("audit-aggregate-*.jsonl")))[0]
    assert record["response"]["stream"] is False
    assert record["response"]["body_json"] == response


def test_concurrent_requests_remain_separate(tmp_path: Path) -> None:
    async def run(content: str) -> None:
        request = {"model": "qwen", "messages": [{"role": "user", "content": content}]}
        response = json.dumps(
            {"choices": [{"message": {"content": f"reply:{content}"}}]},
            ensure_ascii=False,
        ).encode()
        await _call_asgi(
            lambda downstream: AuditLogMiddleware(
                downstream, mode="aggregate", log_dir=tmp_path
            ),
            request_body=json.dumps(request, ensure_ascii=False).encode(),
            response_messages=[
                _response_start(b"application/json"),
                {"type": "http.response.body", "body": response},
            ],
        )

    async def run_concurrently() -> None:
        await asyncio.gather(run("北京"), run("上海"))

    asyncio.run(run_concurrently())

    records = _jsonl(next(tmp_path.glob("audit-aggregate-*.jsonl")))
    pairs = {
        (
            record["request"]["body_json"]["messages"][0]["content"],
            record["response"]["body_json"]["choices"][0]["message"]["content"],
        )
        for record in records
    }
    assert pairs == {("北京", "reply:北京"), ("上海", "reply:上海")}


def test_invalid_mode_is_rejected(tmp_path: Path) -> None:
    async def downstream(scope, receive, send) -> None:
        raise AssertionError("not called")

    with pytest.raises(ValueError, match="audit log mode"):
        AuditLogMiddleware(downstream, mode="verbose", log_dir=tmp_path)


def test_audit_disk_failure_does_not_break_client_response(tmp_path: Path) -> None:
    response_messages = [
        _response_start(b"text/event-stream; charset=utf-8"),
        {
            "type": "http.response.body",
            "body": b'data: {"choices":[]}\n\n',
            "more_body": True,
        },
        {
            "type": "http.response.body",
            "body": b"data: [DONE]\n\n",
            "more_body": False,
        },
    ]

    def middleware_factory(downstream):
        middleware = AuditLogMiddleware(downstream, mode="both", log_dir=tmp_path)

        def fail_write(stream: str, record: dict[str, Any]) -> None:
            raise OSError("disk full")

        middleware.writer.write = fail_write
        return middleware

    sent = asyncio.run(
        _call_asgi(
            middleware_factory,
            request_body=b'{"model":"qwen","messages":[]}',
            response_messages=response_messages,
        )
    )

    assert sent == response_messages
