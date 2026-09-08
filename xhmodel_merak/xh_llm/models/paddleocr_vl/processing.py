# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Shared, deterministic image preparation for PaddleOCR HF and HMONNX."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw


def load_rgb_image(image):
    """Read pixels without EXIF rotation or other geometric correction."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image or a local image path")
    return image.convert("RGB")


def prepare_image(image, resize_mode="stretch", width=896, height=896, *, content_size=None):
    """Resize once with bicubic interpolation; letterbox uses centered white padding.

    ``content_size`` preserves the effective dimensions recorded by historical
    letterbox diagnostics. Historical interpolation/centering were not archived;
    these explicit choices must be shared by both inference backends.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    image = load_rgb_image(image)
    if resize_mode == "stretch":
        if content_size is not None:
            raise ValueError("content_size is only valid for letterbox")
        return image.resize((width, height), Image.Resampling.BICUBIC)
    if resize_mode != "letterbox":
        raise ValueError(f"Unsupported resize mode: {resize_mode}")
    if content_size is None:
        scale = min(width / image.width, height / image.height)
        content_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    content_width, content_height = content_size
    if not (0 < content_width <= width and 0 < content_height <= height):
        raise ValueError("letterbox content_size must fit inside the target image")
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(
        image.resize((content_width, content_height), Image.Resampling.BICUBIC),
        ((width - content_width) // 2, (height - content_height) // 2),
    )
    return canvas


def prepare_inputs(processor, image, prompt, resize_mode="stretch", width=896, height=896):
    """Return native processor tensors for one fixed-size RGB image and prompt."""
    factor = processor.image_processor.patch_size * processor.image_processor.merge_size
    if width % factor or height % factor:
        raise ValueError(f"Target dimensions must be divisible by patch/merge factor {factor}")
    image = prepare_image(image, resize_mode, width, height)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Disable the processor's dynamic resize so it cannot change the locked
    # pixels/aspect ratio used for the HF-versus-HMONNX comparison.
    return processor(text=[text], images=[image], return_tensors="pt", do_resize=False)


def prepare_native_inputs(processor, image, prompt):
    """Use the native processor's aspect-dependent size and pixel limits.

    A static runtime must have a visual graph for the returned grid; this path
    does not pad, stretch, or silently choose another supported resolution.
    """
    image = load_rgb_image(image)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt", do_resize=True)


def _matching_polygon(source, coordinate):
    if isinstance(source, dict):
        candidate = source.get("coordinate")
        if candidate is not None and len(candidate) == len(coordinate):
            if all(math.isclose(a, b, abs_tol=1e-4) for a, b in zip(candidate, coordinate, strict=True)):
                if source.get("polygon_points"):
                    return source["polygon_points"]
        for value in source.values():
            polygon = _matching_polygon(value, coordinate)
            if polygon is not None:
                return polygon
    elif isinstance(source, list):
        for value in source:
            polygon = _matching_polygon(value, coordinate)
            if polygon is not None:
                return polygon
    return None


def recover_region_image(image, region, *, layout_source=None):
    """Recover archived padded, polygon-masked region pixels before resizing.

    The exact archived crop boxes take precedence over recomputing padding from
    fractional detector coordinates. Missing polygon data fail explicitly.
    Multi-part crops are stacked on white with the recorded horizontal alignment.
    """
    image = load_rgb_image(image)
    recovery = region["crop_recovery"]
    sources = recovery.get("source_crops", [])
    if not sources:
        raise ValueError("Region has no archived source_crops")
    parts = []
    for source in sources:
        box = source["crop_box"]
        if len(box) != 4 or any(int(value) != value for value in box):
            raise ValueError("crop_box must contain four integer pixel coordinates")
        left, top, right, bottom = map(int, box)
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            raise ValueError(f"crop_box {box} lies outside the source image")
        part = image.crop((left, top, right, bottom))
        if source.get("polygon_mask_applied"):
            polygon = source.get("polygon_points")
            if not polygon and len(sources) == 1:
                polygon = region.get("polygon_points")
            if not polygon:
                polygon = _matching_polygon(layout_source, source["coordinate"])
            if not polygon or len(polygon) < 3:
                raise ValueError("Polygon-masked crop requires its original polygon_points")
            mask = Image.new("L", part.size, 0)
            ImageDraw.Draw(mask).polygon([(x - left, y - top) for x, y in polygon], fill=255)
            part = Image.composite(part, Image.new("RGB", part.size, "white"), mask)
        if "crop_size" in source and list(part.size) != source["crop_size"]:
            raise ValueError("Recovered part dimensions differ from archived crop_size")
        parts.append(part)
    if len(parts) == 1:
        result = parts[0]
    else:
        if not recovery.get("merged_on_white_canvas"):
            raise ValueError("Multiple source crops require archived canvas merging")
        alignments = recovery.get("merge_aligns", [])
        alignment = alignments[0] if alignments else "left"
        if any(item != alignment for item in alignments) or alignment not in {"left", "center", "right"}:
            raise ValueError(f"Unsupported merged crop alignments: {alignments}")
        result = Image.new("RGB", (max(part.width for part in parts), sum(part.height for part in parts)), "white")
        top = 0
        for part in parts:
            left = {"left": 0, "center": (result.width - part.width) // 2, "right": result.width - part.width}[
                alignment
            ]
            result.paste(part, (left, top))
            top += part.height
    if "crop_size" in recovery and list(result.size) != recovery["crop_size"]:
        raise ValueError("Recovered region dimensions differ from archived crop_size")
    return result
