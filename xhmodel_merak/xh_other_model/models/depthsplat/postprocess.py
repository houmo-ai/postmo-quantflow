# Copyright (c) 2024 Haofei Xu
# Copyright 2026 HOUMO AI (CPU FP32 adapter)
# SPDX-License-Identifier: MIT
# Behavioral adaptation of DepthSplat 2dad25a7b9ba7c5537aba08463ef88d8511a06be.
# See licenses/DepthSplat/LICENSE. No upstream runtime imports are required.
"""CPU FP32 camera geometry and Gaussian construction after neural inference."""

from collections.abc import Mapping

import numpy as np


def _array(value, name, *, original=False):
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{name} must be a floating tensor")
    if original and value.dtype != np.float32:
        raise ValueError(f"{name} must retain original float32 precision")
    value = value.astype(np.float32, copy=False)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")
    return value


def _sigmoid(value):
    # tanh gives a stable sigmoid without exp overflow for large logits.
    return np.float32(0.5) * (np.tanh(value * np.float32(0.5)) + np.float32(1))


def postprocess_gaussians(raw_outputs: Mapping, feed: Mapping, config: dict):
    """Construct world-space Gaussians; neural outputs never supply camera data.

    One Gaussian surface and SH degree two are the supported model contract.
    Small spatial dimensions are accepted for executable mathematical tests.
    """
    if config.get("type") != "depthsplat_gaussian_fp32_v1":
        raise ValueError("Unsupported Gaussian postprocess type")
    if config.get("sh_degree") != 2 or config.get("num_surfaces") != 1:
        raise ValueError("Gaussian postprocess requires sh_degree=2 and num_surfaces=1")
    lower, upper = float(config.get("gaussian_scale_min", 1e-10)), float(config.get("gaussian_scale_max", 3.0))
    if not np.isfinite([lower, upper]).all() or not 0 < lower <= upper:
        raise ValueError("Invalid Gaussian scale bounds")
    if set(raw_outputs) != {"depths", "raw_gaussians"}:
        raise ValueError("Expected depths and raw_gaussians neural outputs")
    image = _array(feed["images"], "images", original=True)
    if image.ndim != 5 or image.shape[2] != 3 or min(image.shape) <= 0:
        raise ValueError("images must have shape [B,V,3,H,W]")
    batch, views, _, height, width = image.shape
    pixels = height * width
    depths = _array(raw_outputs["depths"], "depths")
    raw = _array(raw_outputs["raw_gaussians"], "raw_gaussians")
    if depths.shape != (batch, views, pixels, 1, 1) or raw.shape != (batch * views, 37, height, width):
        raise ValueError("Neural output shapes do not match image batch/views/spatial dimensions")
    intrinsics = _array(feed["intrinsics"], "intrinsics", original=True)
    extrinsics = _array(feed["extrinsics"], "extrinsics", original=True)
    if intrinsics.shape != (batch, views, 3, 3) or extrinsics.shape != (batch, views, 4, 4):
        raise ValueError("Camera shapes do not match images")
    rotations = extrinsics[..., :3, :3]
    raw = raw.reshape(batch, views, 37, pixels).transpose(0, 1, 3, 2)
    opacity = _sigmoid(raw[..., 0])
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    centers = np.stack((xx, yy), axis=-1).reshape(pixels, 2)
    coordinates = (centers + _sigmoid(raw[..., 1:3])) / np.array([width, height], dtype=np.float32)
    homogeneous = np.concatenate((coordinates, np.ones((*coordinates.shape[:-1], 1), dtype=np.float32)), axis=-1)
    rays = (np.linalg.inv(intrinsics)[:, :, None] @ homogeneous[..., None])[..., 0]
    if np.any(rays[..., 2] == 0):
        raise ValueError("Camera rays have zero homogeneous depth")
    rays = rays / rays[..., 2:3]
    directions = (rotations[:, :, None] @ rays[..., None])[..., 0]
    means = extrinsics[:, :, None, :3, 3] + directions * depths[..., 0, 0, None]

    scales = np.clip(np.logaddexp(np.float32(0), raw[..., 3:6] - np.float32(4)), lower, upper)
    quaternion = raw[..., 6:10]
    quaternion = quaternion / (np.linalg.norm(quaternion, axis=-1, keepdims=True) + np.float32(1e-8))
    xyz, scalar = quaternion[..., :3], quaternion[..., 3]
    skew = np.zeros((*xyz.shape[:-1], 3, 3), dtype=np.float32)
    skew[..., 0, 1], skew[..., 0, 2] = -xyz[..., 2], xyz[..., 1]
    skew[..., 1, 0], skew[..., 1, 2] = xyz[..., 2], -xyz[..., 0]
    skew[..., 2, 0], skew[..., 2, 1] = -xyz[..., 1], xyz[..., 0]
    coefficient = np.float32(2) / (np.sum(quaternion * quaternion, axis=-1) + np.float32(1e-8))
    local_rotation = np.eye(3, dtype=np.float32) + coefficient[..., None, None] * (
        scalar[..., None, None] * skew + skew @ skew
    )
    scaled_rotation = local_rotation * scales[..., None, :]
    covariance = scaled_rotation @ scaled_rotation.swapaxes(-1, -2)
    world_rotation = rotations[:, :, None]
    covariance = world_rotation @ covariance @ world_rotation.swapaxes(-1, -2)

    harmonics = raw[..., 10:].reshape(batch, views, pixels, 3, 9).copy()
    for degree in range(1, 3):
        harmonics[..., degree * degree : (degree + 1) ** 2] *= np.float32(0.1 * 0.25**degree)
    rgb = image.transpose(0, 1, 3, 4, 2).reshape(batch, views, pixels, 3)
    harmonics[..., 0] += (rgb - np.float32(0.5)) / np.float32(0.28209479177387814)
    for degree in range(3):
        name = f"camera_{3 + degree}_sh{degree}"
        matrix = _array(feed[name], name, original=True)
        size = 2 * degree + 1
        if matrix.shape != (batch, views, 1, 1, 1, 1, size, size):
            raise ValueError(f"{name} has invalid SH rotation shape")
        matrix = matrix.reshape(batch, views, 1, 1, size, size)
        band = slice(degree * degree, (degree + 1) ** 2)
        harmonics[..., band] = (matrix @ harmonics[..., band, None])[..., 0]
    result = {
        "means": means.reshape(batch, views * pixels, 3),
        "covariances": covariance.reshape(batch, views * pixels, 3, 3),
        "harmonics": harmonics.reshape(batch, views * pixels, 3, 9),
        "opacities": opacity.reshape(batch, views * pixels),
    }
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Gaussian postprocessing produced nonfinite output")
    return {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in result.items()}
