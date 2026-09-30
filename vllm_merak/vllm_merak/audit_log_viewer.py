"""Human-readable viewer for Merak audit JSONL files."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any


def _decode_base64(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        return base64.b64decode(value).decode("utf-8", errors="replace")
    except ValueError:
        return "<invalid base64>"


def format_record(record: dict[str, Any]) -> str:
    """Render one aggregate or raw-SSE audit record for terminal reading."""
    record_type = record.get("record_type", "unknown")
    audit_id = record.get("audit_id", "unknown")

    if record_type == "request_start":
        body_json = record.get("request_body_json")
        if body_json is not None:
            body = json.dumps(body_json, ensure_ascii=False, indent=2)
        else:
            body = record.get("request_body_text") or _decode_base64(
                record.get("request_body_base64")
            )
        return f"[{audit_id} request]\n{body}"

    if record_type == "response_start":
        headers = []
        for header in record.get("headers") or []:
            name = header.get("name") or _decode_base64(header.get("name_base64"))
            value = header.get("value") or _decode_base64(header.get("value_base64"))
            headers.append(f"{name}: {value}")
        details = "\n".join(headers)
        return f"[{audit_id} response status={record.get('status_code')}]\n{details}"

    if record_type == "response_chunk":
        body = record.get("body_text")
        if body is None:
            body = _decode_base64(record.get("body_base64"))
        return (
            f"[{audit_id} chunk={record.get('seq')} "
            f"elapsed_ms={record.get('elapsed_ms')}]\n{body}"
        )

    if record_type == "request_complete":
        return (
            f"[{audit_id} complete status={record.get('status_code')} "
            f"elapsed_ms={record.get('elapsed_ms')} "
            f"chunks={record.get('response_chunks')}]"
        )

    return json.dumps(record, ensure_ascii=False, indent=2)


def print_log(path: Path) -> None:
    """Stream a JSONL audit file to stdout in human-readable form."""
    with path.open(encoding="utf-8") as audit_file:
        for line_number, line in enumerate(audit_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[line {line_number} invalid JSON: {exc}]")
                continue
            print(format_record(record))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Merak audit JSONL file")
    args = parser.parse_args()
    try:
        print_log(args.path)
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
