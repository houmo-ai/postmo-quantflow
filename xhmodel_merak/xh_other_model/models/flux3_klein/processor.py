# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Reference geometry and checkpoint-specific text inputs, without learned fallbacks."""

import math

import numpy as np
from PIL import Image, ImageOps


def _multi_reference_size(image, max_pixels, max_hw):
    width, height = image.size
    scale = min(1.0, math.sqrt(max_pixels / (width * height)), max_hw[1] / width, max_hw[0] / height)
    resized = (int(width * scale), int(height * scale)) if scale < 1 else image.size
    aligned = tuple(side // 16 * 16 for side in resized)
    if min(aligned) < 16:
        raise ValueError("reference budget is too small after 16-pixel centre crop")
    return resized, aligned


def _prepare_multi_references(images, max_pixels, max_hw, total_max_tokens):
    sizes = [_multi_reference_size(image, max_pixels, max_hw) for image in images]
    counts = [width * height // 256 for _, (width, height) in sizes]
    if total_max_tokens is not None and sum(counts) > total_max_tokens:
        if total_max_tokens < len(images):
            raise ValueError("reference token budget cannot fit all references")
        # Max-min allocation: keep small references and share the remainder.
        remaining, budgets = total_max_tokens, [0] * len(images)
        for rank, index in enumerate(sorted(range(len(images)), key=counts.__getitem__)):
            budgets[index] = min(counts[index], remaining // (len(images) - rank))
            remaining -= budgets[index]
        sizes = [
            _multi_reference_size(image, budget * 256, max_hw) if budget < count else size
            for image, budget, count, size in zip(images, budgets, counts, sizes, strict=True)
        ]
    result = []
    for image, (resized, (width, height)) in zip(images, sizes, strict=True):
        # Resize the original only once, after the final budget is known.
        if image.size != resized:
            image = image.resize(resized, Image.Resampling.LANCZOS)
        left, top = (image.width - width) // 2, (image.height - height) // 2
        result.append(image.crop((left, top, left + width, top + height)))
    return result


def prepare_references(
    images, *, single_max_pixels=4 * 1024**2, multi_max_pixels=1024**2, max_hw=(4096, 4096), total_max_tokens=None
):
    """Official single-image reflect padding and multi-image centre crop.

    Fit multi-reference inputs to an optional shared token budget, preserving
    fitting inputs. Returned crop is used only without explicit output sizes.
    """
    if single_max_pixels < min(64, max_hw[0]) * min(64, max_hw[1]):
        raise ValueError("single reference pixel budget cannot fit the minimum padded canvas")
    if len(images) > 10:
        raise ValueError("FLUX supports at most ten reference images")
    if total_max_tokens is not None and (type(total_max_tokens) is not int or total_max_tokens <= 0):
        raise ValueError("reference token budget must be a positive integer")
    normalized = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError("references must be PIL images")
        normalized.append(ImageOps.exif_transpose(image).convert("RGB"))
    if len(normalized) > 1:
        return _prepare_multi_references(normalized, multi_max_pixels, max_hw, total_max_tokens), None
    result, crop = [], None
    for image in normalized:
        width, height = image.size
        scale = min(1.0, math.sqrt(single_max_pixels / (width * height)), max_hw[1] / width, max_hw[0] / height)
        while True:
            fit = (max(1, round(width * scale)), max(1, round(height * scale)))
            canvas = (
                min(max_hw[1], max(64, math.ceil(fit[0] / 16) * 16)),
                min(max_hw[0], max(64, math.ceil(fit[1] / 16) * 16)),
            )
            if canvas[0] * canvas[1] <= single_max_pixels:
                break
            scale *= 0.995
        if fit != image.size:
            image = image.resize(fit, Image.Resampling.LANCZOS)
        if canvas != image.size:
            left, top = (canvas[0] - image.width) // 2, (canvas[1] - image.height) // 2
            pad = ((top, canvas[1] - image.height - top), (left, canvas[0] - image.width - left), (0, 0))
            crop = (left, top, left + image.width, top + image.height)
            image = Image.fromarray(
                np.pad(np.asarray(image), pad, mode="reflect" if min(image.size) > 16 else "symmetric")
            )
        result.append(image)
    return result, crop


def text_inputs(processor, prompt, capacity, *, device="cpu"):
    import torch

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty full scene caption")
    text = processor.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    tokenizer = processor.tokenizer
    real = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)["input_ids"].shape[1]
    length = min(math.ceil(real / 80) * 80, 4096)
    if length > capacity:
        raise ValueError("expanded prompt including official padding exceeds the text capacity")
    batch = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
    mask = batch["attention_mask"]
    positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
    ids = torch.nn.functional.pad(batch["input_ids"], (0, capacity - length), value=tokenizer.pad_token_id)
    positions = torch.nn.functional.pad(positions, (0, capacity - length), value=1)
    return ids.to(device), positions.to(device).int(), int(mask.sum()), length
