# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Preserve worker failures even when package installation metadata is absent."""

import json
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path

from xhmodel_merak.errors import ModelError


def record_error(record, exc):
    error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    if isinstance(exc, ModelError):
        error.update(exc.to_dict())
    record["error"] = error
    print("[unified-ci] failure: " + json.dumps(error, ensure_ascii=False, default=str),
          file=sys.stderr, flush=True)


def write_report(root, record, started):
    environment = {}
    missing = []
    for name in ("torch", "transformers", "gptqmodel", "lerobot"):
        try:
            environment[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            environment[name] = None
            missing.append(name)
    environment["python"] = sys.version.split()[0]
    record["environment"] = environment
    record["environment_metadata_missing"] = missing
    record["imported_modules"] = sorted({name.partition(".")[0] for name in sys.modules})
    record["elapsed_s"] = time.monotonic() - started
    (Path(root) / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
