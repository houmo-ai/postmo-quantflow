# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Host image-generation loop over deployed encoder, DiT and VAE components."""

import math

import numpy as np
import torch
from PIL import Image

from .capacity import fit_reference_size, validate_capacity_config, validate_size
from .layout import prepare_layout
from .sampling import validate_sigmas


class ImageRuntime:
    """Batch-one generation with one immutable prefix cache per CFG branch.

    References are fitted to the published canvas/token capacities. The sampler
    and positional module are weightless upstream components; all learned
    components are supplied deployment sessions, never HF model fallbacks.
    """

    def __init__(self, encoder, denoiser, vae_encoder, vae_decoder, vae_config, scheduler_config, rope, capacity,
                 *, sample_sigmas=None):
        self.encoder = encoder
        self.denoiser = denoiser
        self.vae_encoder, self.vae_decoder = vae_encoder, vae_decoder
        self.capacity = validate_capacity_config(capacity)
        self.scheduler_config = dict(scheduler_config)
        self.sample_sigmas = None if sample_sigmas is None else validate_sigmas(sample_sigmas, field="sample_sigmas")
        self.num_inference_steps = 0
        self.rope = rope
        self.device = torch.device(denoiser.device)
        channels = denoiser.meta["latent_channels"]
        self.mean = torch.tensor(vae_config["latents_mean"], device=self.device, dtype=torch.float32)
        self.std = torch.tensor(vae_config["latents_std"], device=self.device, dtype=torch.float32)
        if (
            self.mean.numel() != channels
            or self.std.numel() != channels
            or bool((self.std <= 0).any())
            or not bool(torch.isfinite(self.mean).all())
            or not bool(torch.isfinite(self.std).all())
        ):
            raise ValueError("VAE normalization does not match the DiT latent channels")
        self.mean = self.mean.reshape(1, channels, 1, 1)
        self.std = self.std.reshape(1, channels, 1, 1)

    def _request(self, prompt, images, references, shapes):
        features, image_mask = self.encoder(prompt, images)
        layout = prepare_layout(
            image_mask,
            shapes,
            prefix_capacity=self.denoiser.gear.prefix_tokens,
            target_capacity=self.denoiser.gear.target_tokens,
        )
        rope = self.rope(shapes, layout.compact_image_mask.to(self.device), device=self.device)
        return self.denoiser.new_request(layout, rope, features, references)

    @torch.no_grad()
    def generate(
        self,
        prompt,
        *,
        height=None,
        width=None,
        images=None,
        steps=20,
        sigmas=None,
        seed=42,
        negative_prompt=None,
        guidance_scale=1.0,
        callback=None,
    ):
        from diffusers import FlowMatchEulerDiscreteScheduler

        if type(steps) is not int or steps <= 0:
            raise ValueError("steps must be a positive integer")
        selected_sigmas = validate_sigmas(sigmas) if sigmas is not None else self.sample_sigmas
        if selected_sigmas is None:
            selected_sigmas = np.linspace(1.0, 1 / steps, steps)
        height = self.capacity["output_size"][0] if height is None else height
        width = self.capacity["output_size"][1] if width is None else width
        validate_size((height, width), self.capacity["canvas_size"], self.capacity["target_capacity"])
        if not math.isfinite(guidance_scale) or guidance_scale < 1:
            raise ValueError("guidance_scale must be finite and at least one")
        if guidance_scale > 1 and negative_prompt is None:
            raise ValueError("guidance requires a negative prompt (an empty string is allowed)")
        images = [] if images is None else list(images)
        if len(images) > self.capacity["max_references"]:
            raise ValueError("reference count exceeds the published capacity")
        shapes, reference_latents = [], []
        for index, image in enumerate(images):
            if not isinstance(image, Image.Image):
                raise TypeError("reference images must be PIL images")
            fitted = fit_reference_size(
                image.height, image.width, self.capacity["canvas_size"], self.capacity["reference_capacity"]
            )
            if fitted != (image.height, image.width):
                image = image.resize((fitted[1], fitted[0]), Image.Resampling.LANCZOS)
            images[index] = image
            rgba = np.asarray(image.convert("RGBA")).copy()
            pixels = torch.from_numpy(rgba).permute(2, 0, 1)[None].to(self.device).float() / 127.5 - 1
            latent = self.vae_encoder(pixels.half()).float()
            expected = (1, self.mean.shape[1], image.height // 16, image.width // 16)
            if latent.shape != expected:
                raise ValueError("VAE encoder returned an incompatible latent shape")
            latent = (latent - self.mean) / self.std
            reference_latents.append(latent.flatten(2).transpose(1, 2).clone())
            shapes.append((1, image.height // 16, image.width // 16))
        shapes.append((1, height // 16, width // 16))
        channels = self.mean.shape[1]
        references = (
            torch.cat(reference_latents, dim=1)
            if reference_latents
            else torch.empty(1, 0, channels, device=self.device)
        )
        conditional = self._request(prompt, images, references, shapes)
        unconditional = None
        if guidance_scale > 1:
            unconditional = self._request(negative_prompt, images, references, shapes)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        # Preserve native channel-major random-number ordering before packing.
        latents = (
            torch.randn(
                (1, 1, channels, height // 16, width // 16),
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )
            .reshape(1, channels, -1)
            .transpose(1, 2)
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler_config)
        config = scheduler.config
        base, maximum = config.get("base_image_seq_len", 256), config.get("max_image_seq_len", 4096)
        if maximum <= base:
            raise ValueError("scheduler sequence-length bounds must be increasing")
        slope = (config.get("max_shift", 1.15) - config.get("base_shift", 0.5)) / (maximum - base)
        mu = slope * (latents.shape[1] - base) + config.get("base_shift", 0.5)
        scheduler.set_timesteps(sigmas=selected_sigmas, device=self.device, mu=mu)
        self.num_inference_steps = len(scheduler.timesteps)
        scheduler.set_begin_index(0)
        for index, timestep in enumerate(scheduler.timesteps):
            model_time = timestep.expand(1).to(latents.dtype) / 1000
            # Both branches share head sessions, whose output buffers may be
            # reused by the following invocation.
            noise = conditional(latents.half(), model_time).clone()
            if unconditional is not None:
                negative = unconditional(latents.half(), model_time)
                noise = negative + guidance_scale * (noise - negative)
            latents = scheduler.step(noise, timestep, latents, return_dict=False)[0]
            if not bool(torch.isfinite(latents).all()):
                raise FloatingPointError(f"nonfinite latents at denoising step {index}")
            if callback is not None:
                callback(index, timestep, latents)
        decoded_input = latents.float().transpose(1, 2).reshape(1, channels, height // 16, width // 16)
        decoded_input = decoded_input * self.std + self.mean
        decoded = self.vae_decoder(decoded_input.half()).float()
        if decoded.shape != (1, 4, height, width) or not bool(torch.isfinite(decoded).all()):
            raise ValueError("VAE decoder returned invalid RGBA output")
        pixels = ((decoded[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(pixels), latents.clone()
