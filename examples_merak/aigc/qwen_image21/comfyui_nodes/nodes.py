# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""ComfyUI image node delegating all learned forwards to the Merak worker."""

import json
from importlib import import_module

import numpy as np
import torch
from PIL import Image

from .client import encode_image, generate


class MerakQwenImage21TurboW8A8:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 -- required ComfyUI node API
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A red panda reading a book in a cozy library."}),
                "width": ("INT", {"default": 1024, "min": 32, "max": 1536, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 32, "max": 1536, "step": 32}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**63 - 1, "control_after_generate": True}),
            },
            "optional": {"reference_image": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Merak/Qwen Image"
    DESCRIPTION = "Qwen Image 2.1 Turbo W8A8 SEFP; 8 steps, CFG 1, one reference, published 1MP capacity."

    def run(self, prompt, width, height, seed, reference_image=None):
        values = {"prompt": prompt, "width": width, "height": height, "seed": seed}
        if reference_image is not None:
            if reference_image.ndim != 4 or reference_image.shape[0] != 1 or reference_image.shape[-1] not in (3, 4):
                raise ValueError("reference_image requires exactly one RGB/RGBA image")
            pixels = reference_image.detach().cpu().float()
            if not bool(torch.isfinite(pixels).all()):
                raise ValueError("reference_image contains nonfinite values")
            pixels = (pixels[0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            values["reference_image"] = encode_image(Image.fromarray(pixels))
        # The worker uses a separate Torch/SDK environment on the same GPU.
        # Native ComfyUI weights may still occupy VRAM after a BF16 workflow.
        management = import_module("comfy.model_management")
        management.unload_all_models()
        management.soft_empty_cache()
        image, stats = generate(values)
        output = torch.from_numpy(np.asarray(image).copy()).float().div_(255).unsqueeze(0)
        return {"ui": {"text": [json.dumps(stats, ensure_ascii=False)]}, "result": (output,)}
