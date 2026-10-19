# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""DepthSplat CI retrieves immutable real assets without a developer-local path."""

import functools
import hashlib
import http.server
import io
import json
import tarfile
import threading

import pytest

from _unified_fixtures import depthsplat


def _release(tmp_path, monkeypatch):
    """Small byte payloads exercise the asset transport, not model semantics."""
    source = tmp_path / "depthsplat-ci-source-v1"
    source.mkdir()
    files = {"config.json": b'{"profile":"2v_256x256"}', "encoder.onnx": b"asset transport payload"}
    manifest = {"archive": "depthsplat-ci-source-v1.tar.gz", "files": {}}
    for name, content in files.items():
        (source / name).write_bytes(content)
        manifest["files"][name] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    archive = tmp_path / manifest["archive"]
    with tarfile.open(archive, "w:gz") as payload:
        for name, content in files.items():
            member = tarfile.TarInfo(f"{source.name}/{name}")
            member.size = len(content)
            payload.addfile(member, io.BytesIO(content))
    manifest["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    release = tmp_path / "release.json"
    release.write_text(json.dumps(manifest))
    monkeypatch.setattr(depthsplat, "RELEASE_FILE", release)
    return source, archive, manifest


def test_download_then_offline_cache_and_corruption(tmp_path, monkeypatch):
    monkeypatch.delenv("DEPTHSPLAT_CI_BUNDLE", raising=False)
    source, archive, manifest = _release(tmp_path, monkeypatch)
    web = tmp_path / "web"
    remote = web / "unified_ci_assets/depthsplat" / manifest["sha256"]
    remote.mkdir(parents=True)
    (remote / archive.name).write_bytes(archive.read_bytes())
    monkeypatch.setenv("UNIFIED_CI_ASSET_CACHE", str(tmp_path / "cache"))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(web))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("UNIFIED_CI_ARTIFACTORY", f"http://127.0.0.1:{server.server_port}")
    try:
        cached = depthsplat.depthsplat_source()
        assert (cached / "encoder.onnx").read_bytes() == (source / "encoder.onnx").read_bytes()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert depthsplat.depthsplat_source() == cached
    path = cached / "encoder.onnx"
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        depthsplat.depthsplat_source()


def test_override_cannot_bypass_identity(tmp_path, monkeypatch):
    source, _, _ = _release(tmp_path, monkeypatch)
    monkeypatch.setenv("DEPTHSPLAT_CI_BUNDLE", str(source))
    assert depthsplat.depthsplat_source() == source
    config = source / "config.json"
    config.write_bytes(b"x" * config.stat().st_size)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        depthsplat.depthsplat_source()


@pytest.mark.parametrize("mode", ["missing", "symlink", "truncated"])
def test_override_rejects_invalid_file(tmp_path, monkeypatch, mode):
    source, _, _ = _release(tmp_path, monkeypatch)
    config = source / "config.json"
    original = config.read_bytes()
    config.unlink()
    if mode == "symlink":
        target = tmp_path / "outside.json"
        target.write_bytes(original)
        config.symlink_to(target)
    elif mode == "truncated":
        config.write_bytes(original[:-1])
    monkeypatch.setenv("DEPTHSPLAT_CI_BUNDLE", str(source))
    with pytest.raises(ValueError, match="Missing or invalid DepthSplat release asset"):
        depthsplat.depthsplat_source()


def test_checked_in_manifest_pins_final_host_prepost_export():
    manifest = json.loads(depthsplat.RELEASE_FILE.read_text())
    assert manifest["sha256"] == "c7a459085fc5fab67230e847e7cb5a34c35219660bf69f724f38b7d3635dbf1f"
    assert manifest["files"]["encoder.onnx"]["sha256"] == (
        "289da02862944aa28f51f7f0d07ea189618365863f1cc31e1d3fa754320421d0"
    )
    assert set(manifest["files"]) == {
        "config.json",
        "encoder.onnx",
        depthsplat.CALIBRATION,
        depthsplat.EVALUATION,
        "LICENSE.DepthSplat",
        "provenance.json",
    }
