# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Pinned release inputs for unified cases that cannot use synthetic weights."""

import fcntl
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import ProxyHandler, build_opener


KOKORO_SHA256 = "704324aeaac896e830f964f36e7a19002c134440d51c9124e582ca8460f4e13c"
KOKORO_ARCHIVE = "kokoro-ci-source-v1.tar.gz"
MOSS_SHA256 = "1e58b1d08c1bdd93e1ec025b0b29dad8dcc37ce29318132ca5088f97b4946dc5"
MOSS_ARCHIVE = "moss-ci-source-v1.tar.gz"


def extract_verified(archive, destination, expected_sha256):
    """Validate before extracting; publish only a complete, safe directory."""
    archive, destination = Path(archive), Path(destination)
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"CI archive SHA-256 mismatch: {archive}")
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        with tarfile.open(archive, "r:gz") as payload:
            members = payload.getmembers()
            if any(not m.name.startswith(destination.name + "/") and m.name != destination.name
                   for m in members):
                raise ValueError("Unexpected CI archive root")
            payload.extractall(temporary, filter="data")
        Path(temporary, destination.name).rename(destination)


def kokoro_source():
    from xhmodel_merak.xh_other_model.models.kokoro.assets import resolve_model_assets, verify_release_assets

    override = os.environ.get("UNIFIED_CI_KOKORO_SOURCE")
    if override:
        source = Path(override).resolve()
        verify_release_assets(resolve_model_assets(source))
        return source
    return _cached_source("kokoro", KOKORO_SHA256, KOKORO_ARCHIVE,
                          lambda source: verify_release_assets(resolve_model_assets(source)))


def verify_moss_source(source):
    source = Path(source)
    manifest = json.loads((Path(__file__).parent / "fixtures/moss/release.json").read_text())
    for name, expected in manifest["files"].items():
        path = source / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["size"]:
            raise ValueError(f"Missing or invalid MOSS release asset: {path}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected["sha256"]:
            raise ValueError(f"MOSS release asset SHA-256 mismatch: {path}")


def moss_source():
    override = os.environ.get("UNIFIED_CI_MOSS_SOURCE")
    if override:
        source = Path(override).resolve()
        verify_moss_source(source)
        return source
    return _cached_source("moss", MOSS_SHA256, MOSS_ARCHIVE, verify_moss_source)


def _cached_source(family, sha256, archive_name, verify):
    cache = Path(os.environ.get("UNIFIED_CI_ASSET_CACHE",
                                Path(__file__).resolve().parents[2] / "work_dirs/unified_ci_assets"))
    cache = cache / sha256
    cache.mkdir(parents=True, exist_ok=True)
    source = cache / archive_name.removesuffix(".tar.gz")
    with (cache / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not source.exists():
            base = os.environ.get("UNIFIED_CI_ARTIFACTORY",
                                  "http://10.10.1.53:8082/artifactory/model_zoo2").rstrip("/")
            url = f"{base}/unified_ci_assets/{family}/{sha256}/{archive_name}"
            archive = cache / archive_name
            print(f"[unified-ci] downloading {url}", flush=True)
            try:
                with build_opener(ProxyHandler({})).open(url, timeout=120) as response, archive.open("wb") as target:
                    shutil.copyfileobj(response, target)
                extract_verified(archive, source, sha256)
            finally:
                archive.unlink(missing_ok=True)
        verify(source)
    return source
