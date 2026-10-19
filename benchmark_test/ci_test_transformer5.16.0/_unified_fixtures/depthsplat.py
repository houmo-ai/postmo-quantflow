# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Pinned real DepthSplat graph and held-out scene; no substitute model."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

RELEASE_FILE = Path(__file__).parents[1] / "fixtures/depthsplat/release.json"
CALIBRATION = "calibration-000c3ab189999a83.npz"
EVALUATION = "evaluation-6ca4076fadf3200e.npz"


def verify_source(source):
    """Check every immutable source file, including config and provenance."""
    source = Path(source)
    manifest = json.loads(RELEASE_FILE.read_text())
    for name, expected in manifest["files"].items():
        path = source / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["size"]:
            raise ValueError(f"Missing or invalid DepthSplat release asset: {path}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected["sha256"]:
            raise ValueError(f"DepthSplat release asset SHA-256 mismatch: {path}")


def depthsplat_source():
    """Retrieve the pinned source automatically, with an integrity-checked override."""
    override = os.environ.get("DEPTHSPLAT_CI_BUNDLE")
    if override:
        source = Path(override).resolve()
        verify_source(source)
        return source
    from _unified_assets import _cached_source

    manifest = json.loads(RELEASE_FILE.read_text())
    return _cached_source("depthsplat", manifest["sha256"], manifest["archive"], verify_source)


def prepare(case, root, monkeypatch):
    del monkeypatch
    source = depthsplat_source()
    # The worker deletes the model source before testing artifact relocation.
    staged = root / "source"
    staged.mkdir()
    for name in json.loads(RELEASE_FILE.read_text())["files"]:
        shutil.copy2(source / name, staged / name)
    sample = root / "inference.npz"
    shutil.copy2(staged / EVALUATION, sample)
    return {
        "model_id": case.model_id,
        "model_dir": str(staged),
        "output_dir": str(root / "artifact"),
        "device": "cuda:0",
    }, {"sample": str(sample)}


def validate_output(case, output, result, moved, inputs, root, record):
    del case, result, moved, inputs, root
    arrays = output.stats["outputs"]
    assert set(arrays) == {"means", "covariances", "harmonics", "opacities"}
    assert all(np.asarray(value).size > 0 and np.isfinite(value).all() for value in arrays.values())
    means, covariance, harmonics, opacity = (
        np.asarray(arrays[name]) for name in ("means", "covariances", "harmonics", "opacities")
    )
    assert means.shape == (1, 131072, 3)
    assert covariance.shape == (*means.shape[:2], 3, 3)
    assert harmonics.shape == (*means.shape[:2], 3, 9)
    assert opacity.shape == means.shape[:2]
    assert np.all((opacity >= 0) & (opacity <= 1))
    record["gaussian_count"] = means.shape[1]
    record["business_accuracy_verified"] = False
