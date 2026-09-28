# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Missing package metadata must not replace the original worker failure."""

import json
import time
from importlib import metadata

import pytest

from _unified_reporting import write_report
from xhmodel_merak.errors import ErrorCode, ModelError


def _missing_metadata(monkeypatch):
    original = metadata.version

    def version(name):
        if name in {"gptqmodel", "lerobot"}:
            raise metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(metadata, "version", version)


def test_worker_preserves_model_error_and_writes_report(monkeypatch, tmp_path, capsys):
    import _unified_e2e_worker as worker

    _missing_metadata(monkeypatch)
    error = ModelError(ErrorCode.RESOURCE_NOT_FOUND, "dependency precheck failed",
                       incompatible_requirements=[{"requirement": "sentencepiece==0.2.1", "installed": None}])

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(worker, "prepare", fail)
    with pytest.raises(ModelError) as captured:
        worker.run(worker.CASES[0], tmp_path)
    assert captured.value is error
    report = json.loads((tmp_path / "result.json").read_text())
    assert not report["passed"] and report["stage"] == "prepare"
    assert report["error"]["code"] == error.code
    assert report["error"]["details"] == error.details
    assert "sentencepiece==0.2.1" in capsys.readouterr().err
    assert report["environment"]["gptqmodel"] is None
    assert report["environment_metadata_missing"] == ["gptqmodel", "lerobot"]
    assert report["environment"]["torch"]


def test_missing_optional_metadata_does_not_fail_successful_record(monkeypatch, tmp_path):
    _missing_metadata(monkeypatch)
    record = {"passed": True, "stage": "complete"}
    write_report(tmp_path, record, time.monotonic())
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["passed"] and "error" not in result
    assert result["environment"]["lerobot"] is None
