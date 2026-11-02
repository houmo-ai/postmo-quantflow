# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Aspect-aware reference sizes backed by a small set of static VAE canvases."""

import math


def image_gears(options):
    gears = options.get("vae_gears")
    if gears is None:
        return None
    if (
        not isinstance(gears, (list, tuple))
        or not gears
        or any(
            not isinstance(shape, (list, tuple))
            or len(shape) != 2
            or any(type(n) is not int or n < 64 or n % 32 for n in shape)
            for shape in gears
        )
    ):
        raise ValueError("reference image VAE gears must be positive 32-aligned height/width pairs")
    result = tuple(tuple(shape) for shape in gears)
    if len(set(result)) != len(result):
        raise ValueError("reference image VAE gears must be unique")
    return result


def video_gear(options):
    shape = options.get("vae_size")
    if shape is None:
        return None
    if not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(
        type(n) is not int or n < 64 or n % 32 for n in shape
    ):
        raise ValueError("reference video VAE size must be a positive 32-aligned height/width pair")
    return tuple(shape)


def image_size(height, width, gears, target_area):
    """Choose the fixed gear whose aspect ratio is closest to the source."""
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        raise ValueError("reference image dimensions must be positive integers")
    if type(target_area) is not int or target_area <= 0:
        raise ValueError("reference image target_area must be a positive integer")
    ratio = width / height
    if not 0.25 <= ratio <= 4:
        raise ValueError("reference images must have an aspect ratio between 1:4 and 4:1")
    if not gears:
        raise ValueError("reference image requires at least one VAE gear")
    return min(
        (tuple(shape) for shape in gears),
        key=lambda shape: (
            abs(math.log((shape[1] / shape[0]) / ratio)),
            abs(math.log(shape[0] * shape[1] / target_area)),
        ),
    )


def video_size(height, width, canvas):
    """Use the fixed spatial size exported for reference video."""
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        raise ValueError("reference video dimensions must be positive integers")
    if not 0.25 <= width / height <= 4:
        raise ValueError("reference videos must have an aspect ratio between 1:4 and 4:1")
    return tuple(canvas)
