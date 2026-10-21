# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Host flow-map sampling over four deployed neural graphs."""

import math

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .capacity import validate_capacity, validate_request
from .dit_graph import denoise_inputs, time_features
from .processor import prepare_references, text_inputs
from .text_graph import feature_inputs
from .vae_graph import spatial_inputs


class ImageRuntime:
    def __init__(self, sessions, processor, embedding, capacity, source_config, *, device="cuda:0", upsampler=None):
        self.sessions, self.processor, self.embedding = sessions, processor, embedding
        self.capacity, self.source_config = validate_capacity(capacity), source_config
        self.device = torch.device(device)
        self.num_inference_steps = 0
        self.last_geometry = None
        self.upsampler = upsampler
        self.last_caption = None
        self.last_upsampler_stats = None

    def _run(self, role, inputs):
        values = [x.to(device=self.device, dtype=torch.float16 if x.is_floating_point() else x.dtype) for x in inputs]
        value = self.sessions[role](*values)
        if isinstance(value, (tuple, list)):
            if len(value) != 1:
                raise ValueError(f"{role} returned an unexpected output inventory")
            value = value[0]
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{role} returned nonfinite values")
        return value.clone()

    def encode_text(self, prompt):
        ids, positions, real, length = text_inputs(self.processor, prompt, self.capacity["text_capacity"])
        embeddings = self.embedding(ids).float()
        values = self._run("text_encode", feature_inputs(embeddings, positions, real, length))
        expected = (1, self.capacity["text_capacity"], self.source_config["dit"]["in_channels"]["text"])
        if tuple(values.shape) != expected:
            raise ValueError("text graph output shape disagrees with metadata")
        return values[:, :length].to(torch.bfloat16).half().clone()

    def encode_reference(self, image):
        h, w = image.height, image.width
        ch, cw = self.capacity["canvas_size"]
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
        padded = F.pad(pixels, (0, cw - w, 0, ch - h))
        latent = self._run("vae_encode", (padded, *spatial_inputs((h, w), (ch, cw))))
        if tuple(latent.shape) != (1, 128, ch // 16, cw // 16):
            raise ValueError("VAE encoder returned an incompatible latent canvas")
        return latent[:, :, : h // 16, : w // 16].flatten(2).transpose(1, 2).to(torch.bfloat16).half().contiguous()

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        *,
        images=None,
        height=None,
        width=None,
        steps=None,
        seed=42,
        callback=None,
        prompt_upsampling=False,
    ):
        if type(prompt_upsampling) is not bool:
            raise ValueError("prompt_upsampling must be a bool")
        if prompt_upsampling and self.upsampler is None:
            raise ValueError("this bundle does not include a prompt upsampler")
        self.last_caption, self.last_upsampler_stats = None, None
        steps = self.source_config["sampler"]["steps"] if steps is None else steps
        if type(steps) is not int or steps <= 0:
            raise ValueError("steps must be a positive integer")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be a nonnegative signed 64-bit integer")
        images = [] if images is None else list(images)
        if len(images) > self.capacity["max_references"]:
            raise ValueError("reference count exceeds the published capacity")
        references, crop = prepare_references(
            images,
            single_max_pixels=min(
                4 * 1024**2, self.capacity["single_reference_max_pixels"], self.capacity["reference_capacity"] * 256
            ),
            multi_max_pixels=min(
                1024**2, self.capacity["multi_reference_max_pixels"], self.capacity["reference_capacity"] * 256
            ),
            max_hw=self.capacity["canvas_size"],
            total_max_tokens=self.capacity["reference_total_capacity"],
        )
        crop = crop if height is None and width is None else None
        default = (references[0].height, references[0].width) if references else self.capacity["output_size"]
        h, w = default[0] if height is None else height, default[1] if width is None else width
        shapes = [(image.height, image.width) for image in references]
        # Validate geometry/count before executing the potentially large VAE graphs.
        validate_request(self.capacity, (h, w), shapes, 1)
        if prompt_upsampling:
            prompt = self.upsampler(prompt, images, (h, w))
            self.last_upsampler_stats = dict(self.upsampler.last_stats)
        self.last_caption = prompt
        text = self.encode_text(prompt)
        counts = validate_request(self.capacity, (h, w), shapes, text.shape[1])
        encoded = [self.encode_reference(image) for image in references]
        ref = torch.cat(encoded, 1) if encoded else torch.empty(1, 0, 128, device=self.device, dtype=torch.float16)
        noise = torch.randn((counts[0], 128), generator=torch.Generator(device="cpu").manual_seed(seed))
        latent = noise.to(torch.bfloat16).float()[None].to(self.device)
        alpha = self.source_config["sampler"]["alpha"]
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("invalid packaged sampling alpha")
        time = torch.linspace(1.0, 0.0, steps + 1)
        schedule = alpha * time / (1 + (alpha - 1) * time)
        capacities = (
            self.capacity["target_capacity"],
            self.capacity["reference_total_capacity"],
            self.capacity["text_capacity"],
        )
        self.num_inference_steps = steps
        self.last_geometry = dict(
            target_hw=[h, w], references_hw=[list(x) for x in shapes], crop=crop, text_tokens=text.shape[1]
        )
        for index in range(steps):
            current, following = schedule[index].item(), schedule[index + 1].item()
            inputs = list(
                denoise_inputs(
                    latent.to(torch.bfloat16).half(), ref, text, (h, w), shapes, capacities, current, following
                )
            )
            # Preserve the reference BF16 time inputs and output/step rounding;
            # the persistent Euler state itself remains FP32.
            inputs[3] = (
                time_features([current, 0, 0], device=self.device, dtype=torch.bfloat16).to(torch.bfloat16).half()
            )
            inputs[4] = time_features([following], device=self.device, dtype=torch.bfloat16).to(torch.bfloat16).half()
            velocity = self._run("denoise", inputs)[:, : counts[0]].to(torch.bfloat16)
            if tuple(velocity.shape) != tuple(latent.shape):
                raise ValueError("DiT output does not match the request target")
            latent = latent + (following - current) * velocity
            if not bool(torch.isfinite(latent).all()):
                raise FloatingPointError(f"nonfinite latent at step {index}")
            if callback is not None:
                callback(index, current, latent.clone())
        ch, cw = self.capacity["canvas_size"]
        value = latent.transpose(1, 2).reshape(1, 128, h // 16, w // 16)
        value = F.pad(value, (0, (cw - w) // 16, 0, (ch - h) // 16))
        pixels = self._run("vae_decode", (value, *spatial_inputs((h, w), (ch, cw))))
        if tuple(pixels.shape) != (1, 3, ch, cw):
            raise ValueError("VAE decoder returned an incompatible RGB canvas")
        array = ((pixels[0, :, :h, :w].float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(array)
        return (image.crop(crop) if crop is not None else image), latent
