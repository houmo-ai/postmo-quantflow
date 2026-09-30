import base64
import json
import sys
from pathlib import Path

from vllm_merak import audit_log_viewer
from vllm_merak.audit_log_viewer import format_record, print_log


def test_format_record_decodes_legacy_base64_response_chunk() -> None:
    body = 'data: {"choices":[{"delta":{"content":"你好"}}]}\n\n'.encode()
    record = {
        "record_type": "response_chunk",
        "audit_id": "audit-1",
        "seq": 3,
        "elapsed_ms": 12.5,
        "body_base64": base64.b64encode(body).decode(),
    }

    assert format_record(record) == (
        "[audit-1 chunk=3 elapsed_ms=12.5]\n" + body.decode("utf-8")
    )


def test_format_record_prefers_readable_request_json() -> None:
    record = {
        "record_type": "request_start",
        "audit_id": "audit-2",
        "request_body_json": {
            "messages": [{"role": "user", "content": "多轮对话"}],
            "stream": True,
        },
        "request_body_base64": "ignored",
    }

    rendered = format_record(record)

    assert rendered.startswith("[audit-2 request]\n")
    assert '"content": "多轮对话"' in rendered


def test_print_log_decodes_existing_jsonl_file(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "audit-sse.jsonl"
    path.write_text(
        json.dumps(
            {
                "record_type": "response_chunk",
                "audit_id": "audit-3",
                "seq": 0,
                "elapsed_ms": 1.0,
                "body_base64": base64.b64encode(b"data: [DONE]\n\n").decode(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    print_log(path)

    assert "data: [DONE]" in capsys.readouterr().out


def test_main_ignores_broken_pipe(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "audit-sse.jsonl"

    def raise_broken_pipe(_: Path) -> None:
        raise BrokenPipeError

    monkeypatch.setattr(audit_log_viewer, "print_log", raise_broken_pipe)
    monkeypatch.setattr(sys, "argv", ["audit-log-viewer", str(path)])

    audit_log_viewer.main()
