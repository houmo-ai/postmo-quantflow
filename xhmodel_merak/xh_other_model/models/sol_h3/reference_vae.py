# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native H3 temporal chunks and spatial stitching around exported encoder tiles."""

import json
from pathlib import Path

import torch
from torch import nn

from .feature_graph import load_component_weights


def _vae_shell(config, *, keep_encoder):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    with torch.device("meta"):
        vae = AutoencoderKLMiniMaxH3.from_config(config)
    vae.decoder = vae.post_quant_conv = nn.Identity()
    if not keep_encoder:
        vae.encoder = vae.quant_conv = nn.Identity()
    return vae


def load_reference_vae(model_dir):
    root = Path(model_dir) / "vae"
    config = json.loads((root / "config.json").read_text())
    vae = _vae_shell(config, keep_encoder=True)
    load_component_weights(vae, root, "", weights_name="diffusion_pytorch_model.safetensors")
    return vae.eval().requires_grad_(False), config


def invoke_reference_tile(metadata, shape, pixels, invoke, *, mode=None):
    if (mode or metadata.get("mode")) in ("i2va", "l2va", "fl2va"):
        base = metadata["reference_vae_base"]
        if tuple(shape) != tuple(base["shape"]):
            raise ValueError(f"base reference VAE input {shape} does not match its exported geometry")
        return invoke(base["graph"], pixels)
    name = metadata["reference_vae_tiles"].get("x".join(map(str, shape)))
    if name is None:
        raise ValueError(f"reference VAE tile {shape} is outside the exported profile")
    if metadata.get("reference_vae_layout") == "fixed_v1":
        if tuple(pixels.shape[2:]) != tuple(shape):
            raise ValueError("reference VAE pixels do not match the fixed exported geometry")
        return invoke(name, pixels)
    policy = metadata["reference_vae_masks"][name]
    from .masked_vae import reference_mask_inputs

    moments = invoke(name, *reference_mask_inputs(pixels, **policy))
    temporal, spatial = policy["levels"][-1]
    return moments[:, :, : (shape[0] - 1) // temporal + 1, : shape[1] // spatial, : shape[2] // spatial]


class _TileSession(nn.Module):
    def __init__(self, invoke):
        super().__init__()
        self.invoke = invoke

    def forward(self, value):
        return self.invoke(tuple(value.shape[2:]), value).float()


class ReferenceVAEAdapter(nn.Module):
    def __init__(self, config, invoke, *, tile_size=(768, 768), overlap=(64, 64), untiled=False):
        super().__init__()
        from .h3_vae_wrappers import SolH3VAEEncoderWrapper

        vae = _vae_shell(config, keep_encoder=False)
        ratio = vae.spatial_compression_ratio
        if (
            len(tile_size) != 2
            or len(overlap) != 2
            or any(type(n) is not int or n <= 0 or n % ratio for n in tile_size)
            or any(type(n) is not int or n < 0 or n % ratio for n in overlap)
            or any(o >= t for o, t in zip(overlap, tile_size, strict=True))
        ):
            raise ValueError("reference VAE tiles and overlaps must be latent-aligned with positive stride")
        vae.enable_tiling(
            tile_sample_min_height=tile_size[0],
            tile_sample_min_width=tile_size[1],
            tile_sample_min_overlap_height=overlap[0],
            tile_sample_min_overlap_width=overlap[1],
        )
        # Native enable_tiling treats zero as 'keep default'; explicit zero is a
        # valid non-overlapping tile policy and must survive serialization.
        vae.tile_sample_min_overlap_height, vae.tile_sample_min_overlap_width = overlap
        if untiled:
            vae.disable_tiling()
        self.config, self.spatial_compression_ratio = vae.config, ratio
        self.encoder = SolH3VAEEncoderWrapper(vae, tile_encoder=_TileSession(invoke), wrap_modules=False)
        self.dtype = torch.float32

    def tile_shapes(self, spatial_shapes, *, video=False):
        """Plan exact graph gears with the native splitter, without pixel allocation.

        Short boundary tiles keep their actual dimensions: padding them to a
        larger graph would alter the encoder's spatial GroupNorm statistics.
        """
        vae, result = self.encoder.vae, set()
        frames = self.config.clip_length if video else 1
        for height, width in spatial_shapes:
            if any(type(n) is not int or n <= 0 or n % self.spatial_compression_ratio for n in (height, width)):
                raise ValueError("reference VAE geometry must be positive and latent-aligned")
            _, heights, _ = vae._split_tiles(height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)
            _, widths, _ = vae._split_tiles(width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width)
            result.update((frames, h, w) for h in heights for w in widths)
        return sorted(result)

    def encode(self, pixels, return_dict=True):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from diffusers.models.modeling_outputs import AutoencoderKLOutput

        if pixels.ndim != 5 or pixels.shape[0] != 1 or pixels.shape[1] != self.config.in_channels:
            raise ValueError("reference VAE requires batch-one [B, C, T, H, W] pixels")
        posterior = DiagonalGaussianDistribution(self.encoder(pixels.float()))
        return AutoencoderKLOutput(latent_dist=posterior) if return_dict else (posterior,)
