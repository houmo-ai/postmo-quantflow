# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""CI downloads must be hash-pinned and extract only inside their cache."""

import hashlib
import io
import tarfile

import pytest

from _unified_assets import extract_verified


@pytest.mark.parametrize("name", ["model/config.json", "other/config.json", "model/../../escape"])
def test_archive_root_and_traversal(tmp_path, name):
    archive = tmp_path / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as payload:
        member = tarfile.TarInfo(name)
        member.size = 2
        payload.addfile(member, io.BytesIO(b"{}"))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    target = tmp_path / "model"
    if name == "model/config.json":
        extract_verified(archive, target, digest)
        assert (target / "config.json").read_bytes() == b"{}"
    else:
        with pytest.raises((ValueError, tarfile.FilterError)):
            extract_verified(archive, target, digest)
        assert not target.exists()


def test_checksum_precedes_extraction(tmp_path):
    archive = tmp_path / "model.tar.gz"
    archive.write_bytes(b"corrupt download")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        extract_verified(archive, tmp_path / "model", "0" * 64)
    assert not (tmp_path / "model").exists()


def test_download_cache_and_corruption_are_checked(tmp_path, monkeypatch):
    import functools
    import http.server
    import threading
    from _unified_assets import _cached_source

    web = tmp_path / "web"
    web.mkdir()
    archive = tmp_path / "moss-ci-source-v1.tar.gz"
    with tarfile.open(archive, "w:gz") as payload:
        info = tarfile.TarInfo("moss-ci-source-v1/config.json")
        info.size = 2
        payload.addfile(info, io.BytesIO(b"{}"))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    remote = web / "unified_ci_assets/moss" / digest
    remote.mkdir(parents=True)
    (remote / archive.name).write_bytes(archive.read_bytes())
    monkeypatch.setenv("UNIFIED_CI_ASSET_CACHE", str(tmp_path / "cache"))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(web))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("UNIFIED_CI_ARTIFACTORY", f"http://127.0.0.1:{server.server_port}")

    def verify(source):
        if (source / "config.json").read_bytes() != b"{}":
            raise ValueError("corrupt cached file")

    try:
        source = _cached_source("moss", digest, archive.name, verify)
        assert (source / "config.json").read_bytes() == b"{}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert _cached_source("moss", digest, archive.name, verify) == source
    (source / "config.json").write_bytes(b"broken")
    with pytest.raises(ValueError, match="corrupt cached file"):
        _cached_source("moss", digest, archive.name, verify)


def test_moss_override_is_verified_instead_of_bypassing_integrity(tmp_path, monkeypatch):
    from _unified_assets import moss_source

    monkeypatch.setenv("UNIFIED_CI_MOSS_SOURCE", str(tmp_path))
    with pytest.raises(ValueError, match="MOSS release asset"):
        moss_source()


def test_moss_override_rejects_same_size_corruption(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import _unified_assets as assets

    manifest = json.loads((Path(assets.__file__).parent / "fixtures/moss/release.json").read_text())
    name, expected = next(iter(manifest["files"].items()))
    (tmp_path / name).write_bytes(b"\0" * expected["size"])
    monkeypatch.setenv("UNIFIED_CI_MOSS_SOURCE", str(tmp_path))
    with pytest.raises(ValueError, match="MOSS release asset SHA-256 mismatch"):
        assets.moss_source()
