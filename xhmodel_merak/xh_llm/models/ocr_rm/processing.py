# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Delivered value-head input protocol (original image, text, rendered image)."""

import math


def build_prompt(ocr_text):
    image = "<|vision_start|><|image_pad|><|vision_end|>"
    return (
        f"<|im_start|>user\n{image}OCR文本：\n{ocr_text}\n{image}"
        "第一张图是原图，第二张图是上述 OCR 文本渲染得到的图片。\n"
        "请判断该 OCR 文本及其对应渲染图是否与原图内容一致。\n"
        "这是一个二分类任务：一致标为 1，不一致标为 0。<|im_end|>\n"
    )


def load_image(path):
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    if max(width, height) / min(width, height) > 199:
        size = (max(width, math.ceil(height / 199)), max(height, math.ceil(width / 199)))
        canvas = Image.new("RGB", size, "white")
        canvas.paste(image, ((size[0] - width) // 2, (size[1] - height) // 2))
        image = canvas
    return image


def prepare_inputs(processor, orig_image, render_image, ocr_text):
    # Preserve the native 64-token minimum and full maximum budget. Runtime
    # rejects images exceeding its exported gears instead of resizing silently.
    return processor(
        text=[build_prompt(ocr_text)], images=[load_image(orig_image), load_image(render_image)], return_tensors="pt"
    )
