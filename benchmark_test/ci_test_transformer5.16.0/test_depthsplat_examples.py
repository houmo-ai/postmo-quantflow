# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""DepthSplat CLI and real dataset selection contracts, without upstream weights."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from xhmodel_merak.xh_other_model.models.depthsplat.demo import load_scene, prepare_sample

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples_merak/cv/depthsplat"


@pytest.mark.parametrize("entry", ["workflow", "unified", "evaluate", "export_onnx", "onnx_pipeline", "demo"])
def test_cli_help_needs_no_weights_or_optional_renderer(entry, tmp_path):
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / f"{entry}.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("arguments", [["prepare"], ["render"], ["render", "--profile", "4v"]])
def test_demo_cli_rejects_incomplete_or_unsupported_requests(arguments):
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "demo.py"), *arguments],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr


def _scene(root):
    scene = {"key": "held-out", "images": [torch.tensor([index], dtype=torch.uint8) for index in range(8)]}
    torch.save([{"key": "calibration", "images": []}, scene], root / "chunk.torch")
    return scene


@pytest.mark.parametrize("layout", ["file", "split", "dataset"])
def test_scene_loader_selects_exact_scene_in_supported_chunk_layouts(tmp_path, layout):
    chunks = tmp_path / "test" if layout == "dataset" else tmp_path
    chunks.mkdir(exist_ok=True)
    expected = _scene(chunks)
    path = chunks / "chunk.torch" if layout == "file" else tmp_path
    actual = load_scene(path, "held-out")
    assert actual["key"] == expected["key"]
    assert [image.item() for image in actual["images"]] == list(range(8))
    with pytest.raises(ValueError, match="not found"):
        load_scene(path, "unknown-scene")


@pytest.mark.parametrize(
    "profile,contexts,targets,message",
    [
        ("2v", [0], None, "distinct context"),
        ("2v", [0, 0], None, "distinct context"),
        ("6v", [0, 1], None, "distinct context"),
        ("6v", [0, 1, 2, 3, 4, 4], None, "distinct context"),
        ("2v", [-1, 1], None, "outside"),
        ("2v", [0, 8], None, "outside"),
        ("2v", [0, 1], [], "empty"),
        ("2v", [0, 1], [8], "outside"),
    ],
)
def test_prepare_rejects_invalid_frame_selection_before_decoding(tmp_path, profile, contexts, targets, message):
    _scene(tmp_path)
    output = tmp_path / "never-written.npz"
    with pytest.raises(ValueError, match=message):
        prepare_sample(tmp_path, "held-out", profile, tmp_path / "no-upstream", output, contexts, targets)
    assert not output.exists()
