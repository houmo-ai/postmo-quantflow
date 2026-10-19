# Copyright (c) 2024 Haofei Xu
# Copyright 2026 HOUMO AI (CPU FP32 camera adapter)
# SPDX-License-Identifier: MIT
# Behavioral adaptation of DepthSplat 2dad25a7b9ba7c5537aba08463ef88d8511a06be.
# See licenses/DepthSplat/LICENSE. No upstream runtime imports are required.
"""Construct the two-view matching grid from original cameras on CPU."""

from collections.abc import Mapping

import numpy as np


def prepare_matching_grid(feed: Mapping, config: dict) -> np.ndarray:
    """Project fixed inverse-depth candidates into the opposite source view.

    Camera intrinsics use normalized image coordinates and extrinsics are
    camera-to-world transforms. Coordinates target align_corners=True sampling.
    No neural prediction, captured intermediate, or dataset-specific constant is
    consumed. The function does not mutate the caller's input arrays.
    """
    if config.get("type") != "depthsplat_matching_grid_fp32_v1":
        raise ValueError("Unsupported matching-grid preprocessing type")
    dimensions = []
    for name in ("num_depth_candidates", "feature_height", "feature_width"):
        value = config.get(name)
        if type(value) is not int or value < 2:
            raise ValueError(f"{name} must be an integer greater than one")
        dimensions.append(value)
    count, height, width = dimensions
    arrays = {}
    for name, shape in (("intrinsics", (1, 2, 3, 3)), ("extrinsics", (1, 2, 4, 4)), ("near", (1, 2)), ("far", (1, 2))):
        value = np.asarray(feed[name])
        if value.dtype != np.float32 or value.shape != shape:
            raise ValueError(f"{name} must be original float32 with shape {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")
        arrays[name] = value
    near, far = arrays["near"].reshape(2, 1), arrays["far"].reshape(2, 1)
    if np.any(near <= 0) or np.any(far <= near):
        raise ValueError("Camera ranges require 0 < near < far")
    intrinsic = arrays["intrinsics"].reshape(2, 3, 3).copy()
    intrinsic[:, 0] *= np.float32(width)
    intrinsic[:, 1] *= np.float32(height)
    extrinsic = arrays["extrinsics"].reshape(2, 4, 4)
    try:
        inverse_intrinsic = np.linalg.inv(intrinsic)
        relative_pose = np.linalg.inv(extrinsic[::-1]) @ extrinsic
    except np.linalg.LinAlgError as error:
        raise ValueError("Matching grid requires nonsingular cameras") from error
    inverse_depth = np.float32(1) / far + np.linspace(0, 1, count, dtype=np.float32)[None] * (
        np.float32(1) / near - np.float32(1) / far
    )
    depth = np.float32(1) / inverse_depth
    y, x = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    pixels = np.stack((x, y, np.ones_like(x)), axis=0).reshape(1, 3, height * width)
    rays = inverse_intrinsic @ pixels
    rotated = relative_pose[:, :3, :3] @ rays
    points = rotated[:, :, None] * depth[:, None, :, None] + relative_pose[:, :3, 3, None, None]
    projected = (intrinsic @ points.reshape(2, 3, -1)).reshape(2, 3, count, height * width)
    xy = projected[:, :2] / np.maximum(projected[:, 2:3], np.float32(1e-3))
    normalized_x = np.float32(2) * xy[:, 0] / np.float32(width - 1) - np.float32(1)
    normalized_y = np.float32(2) * xy[:, 1] / np.float32(height - 1) - np.float32(1)
    grid = np.stack((normalized_x, normalized_y), axis=-1).reshape(2, count * height, width, 2)
    if not np.isfinite(grid).all():
        raise ValueError("Matching-grid preprocessing produced nonfinite coordinates")
    return np.ascontiguousarray(grid, dtype=np.float32)
