"""Real CPU matching geometry, independent of neural checkpoint and SDK."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "xhmodel_merak/xh_other_model/models/depthsplat/preprocess.py"
_SPEC = importlib.util.spec_from_file_location("depthsplat_preprocess", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
prepare_matching_grid = _MODULE.prepare_matching_grid


def fixture():
    return {
        "intrinsics": np.tile(np.eye(3, dtype=np.float32), (1, 2, 1, 1)),
        "extrinsics": np.tile(np.eye(4, dtype=np.float32), (1, 2, 1, 1)),
        "near": np.ones((1, 2), dtype=np.float32),
        "far": np.full((1, 2), 4, dtype=np.float32),
    }, {"type": "depthsplat_matching_grid_fp32_v1", "num_depth_candidates": 3, "feature_height": 2, "feature_width": 2}


def test_identity_projection_and_no_input_mutation():
    feed, config = fixture()
    before = {key: value.copy() for key, value in feed.items()}
    result = prepare_matching_grid(feed, config)
    expected = np.array([[[-1, -1], [1, -1]], [[-1, 1], [1, 1]]], dtype=np.float32)
    np.testing.assert_allclose(result.reshape(2, 3, 2, 2, 2), np.tile(expected, (2, 3, 1, 1, 1)))
    assert result.dtype == np.float32 and result.flags.c_contiguous
    for key in feed:
        np.testing.assert_array_equal(feed[key], before[key])


def test_opposite_view_translation_depends_on_inverse_depth():
    feed, config = fixture()
    feed["extrinsics"][0, 1, 0, 3] = 1
    grid = prepare_matching_grid(feed, config).reshape(2, 3, 2, 2, 2)
    np.testing.assert_allclose(grid[0, :, 0, 0, 0], [-2, -3.5, -5])
    np.testing.assert_allclose(grid[1, :, 0, 0, 0], [0, 1.5, 3])
    feed["far"][:] = 2
    changed = prepare_matching_grid(feed, config).reshape(2, 3, 2, 2, 2)
    assert changed[0, 0, 0, 0, 0] == -3
    assert changed[0, -1, 0, 0, 0] == grid[0, -1, 0, 0, 0]


def test_non_square_feature_geometry_and_camera_rotation():
    feed, config = fixture()
    config.update(feature_height=3, feature_width=4)
    feed["extrinsics"][0, 1, :3, :3] = np.diag([-1, -1, 1])
    grid = prepare_matching_grid(feed, config)
    assert grid.shape == (2, 9, 4, 2)
    np.testing.assert_allclose(grid[0, 2, 3], [-3, -3], atol=2e-7)


@pytest.mark.parametrize(
    "invalid", ["half", "shape", "nan", "near", "far", "singular_k", "singular_pose", "count", "width", "type"]
)
def test_reject_invalid_original_camera_contract(invalid):
    feed, config = fixture()
    if invalid == "half":
        feed["intrinsics"] = feed["intrinsics"].astype(np.float16)
    if invalid == "shape":
        feed["extrinsics"] = feed["extrinsics"][0]
    if invalid == "nan":
        feed["near"][0, 0] = np.nan
    if invalid == "near":
        feed["near"][0, 0] = 0
    if invalid == "far":
        feed["far"][0, 0] = 1
    if invalid == "singular_k":
        feed["intrinsics"][0, 0] = 0
    if invalid == "singular_pose":
        feed["extrinsics"][0, 0] = 0
    if invalid == "count":
        config["num_depth_candidates"] = True
    if invalid == "width":
        config["feature_width"] = 1
    if invalid == "type":
        config["type"] = "unknown"
    with pytest.raises(ValueError):
        prepare_matching_grid(feed, config)
