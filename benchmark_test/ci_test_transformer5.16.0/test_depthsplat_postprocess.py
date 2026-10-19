"""Small real CPU geometry checks for the neural/Host boundary."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "xhmodel_merak/xh_other_model/models/depthsplat/postprocess.py"
_SPEC = importlib.util.spec_from_file_location("depthsplat_postprocess", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
postprocess_gaussians = _MODULE.postprocess_gaussians


def fixture():
    feed = {
        "images": np.full((1, 2, 3, 1, 2), 0.5, dtype=np.float32),
        "intrinsics": np.tile(np.eye(3, dtype=np.float32), (1, 2, 1, 1)),
        "extrinsics": np.tile(np.eye(4, dtype=np.float32), (1, 2, 1, 1)),
    }
    for degree in range(3):
        feed[f"camera_{degree + 3}_sh{degree}"] = np.tile(
            np.eye(2 * degree + 1, dtype=np.float32), (1, 2, 1, 1, 1, 1, 1, 1)
        )
    raw = np.zeros((2, 37, 1, 2), dtype=np.float32)
    raw[:, 3:6] = 4 + np.log(np.expm1(np.float32(1)))
    raw[:, 9] = 1
    outputs = {"depths": np.full((1, 2, 2, 1, 1), 2, dtype=np.float32), "raw_gaussians": raw}
    config = dict(
        type="depthsplat_gaussian_fp32_v1", sh_degree=2, num_surfaces=1, gaussian_scale_min=1e-10, gaussian_scale_max=3
    )
    return outputs, feed, config


def test_pixel_depth_covariance_opacity_and_rgb():
    outputs, feed, config = fixture()
    feed["images"][:, :, 0] = 1
    result = postprocess_gaussians(outputs, feed, config)
    np.testing.assert_allclose(result["means"][0], [[0.5, 1, 2], [1.5, 1, 2]] * 2)
    np.testing.assert_allclose(result["covariances"], np.tile(np.eye(3), (1, 4, 1, 1)), atol=2e-6)
    np.testing.assert_allclose(result["opacities"], 0.5)
    np.testing.assert_allclose(result["harmonics"][..., 0, 0], 0.5 / 0.28209479177387814)
    assert all(value.dtype == np.float32 for value in result.values())


def test_original_cameras_drive_world_positions():
    outputs, feed, config = fixture()
    feed["extrinsics"][0, 1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    feed["extrinsics"][0, 1, :3, 3] = [3, 4, 5]
    feed["intrinsics"][0, 1, 0, 0] = 2
    result = postprocess_gaussians(outputs, feed, config)
    np.testing.assert_allclose(result["means"][0, 2:], [[2, 4.25, 7], [2, 4.75, 7]])


def test_quaternion_covariance_and_sh_band_rotation():
    outputs, feed, config = fixture()
    raw = outputs["raw_gaussians"]
    raw[:, 3:6] = (4 + np.log(np.expm1(np.array([1, 2, 3], dtype=np.float32))))[None, :, None, None]
    raw[:, 8:10] = np.sqrt(np.float32(0.5))
    raw[:, 10:] = 1
    feed["camera_4_sh1"][..., 0, 0] = -1
    feed["camera_4_sh1"][..., 2, 2] = -1
    result = postprocess_gaussians(outputs, feed, config)
    np.testing.assert_allclose(result["covariances"][0, 0], np.diag([4, 1, 9]), atol=4e-6)
    np.testing.assert_allclose(
        result["harmonics"][0, 0, 0],
        [1, -0.025, 0.025, -0.025, 0.00625, 0.00625, 0.00625, 0.00625, 0.00625],
        atol=1e-7,
    )


def test_extreme_logits_and_zero_quaternion_remain_finite():
    outputs, feed, config = fixture()
    outputs["raw_gaussians"][:, :6] = -1000
    outputs["raw_gaussians"][:, 6:10] = 0
    result = postprocess_gaussians(outputs, feed, config)
    assert all(np.isfinite(value).all() for value in result.values())
    np.testing.assert_array_equal(result["opacities"], 0)


@pytest.mark.parametrize(
    "bad", ["depth_shape", "raw_shape", "camera_shape", "half_camera", "nan", "sh_shape", "surfaces", "bounds"]
)
def test_contract_rejections(bad):
    outputs, feed, config = fixture()
    if bad == "depth_shape":
        outputs["depths"] = outputs["depths"][..., 0]
    if bad == "raw_shape":
        outputs["raw_gaussians"] = outputs["raw_gaussians"][:, :-1]
    if bad == "camera_shape":
        feed["intrinsics"] = feed["intrinsics"][0]
    if bad == "half_camera":
        feed["intrinsics"] = feed["intrinsics"].astype(np.float16)
    if bad == "nan":
        outputs["depths"].flat[0] = np.nan
    if bad == "sh_shape":
        feed["camera_4_sh1"] = np.eye(3, dtype=np.float32)
    if bad == "surfaces":
        config["num_surfaces"] = 2
    if bad == "bounds":
        config["gaussian_scale_min"] = 4
    with pytest.raises(ValueError):
        postprocess_gaussians(outputs, feed, config)
