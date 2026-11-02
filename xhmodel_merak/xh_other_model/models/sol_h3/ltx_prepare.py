# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Build-only use of the author's positional grids and time embedding networks."""

from __future__ import annotations

from dataclasses import replace

import torch

from .conditioning import REFINER_SIGMAS


def cpu_lora_fuse_rule():
    """Keep native BF16 fusion math, but release each GPU result to the host."""
    from ltx_core.loader.fuse_loras import FuseRule, bf16_fuse_rule

    def fuse(key, weight, deltas, model_sd):
        return {name: value.cpu() for name, value in bf16_fuse_rule(key, weight, deltas, model_sd).items()}

    return FuseRule(aggregation_dtype=bf16_fuse_rule.aggregation_dtype, fuse_fn=fuse)


def native_tools(*, size=(768, 1344), frames=121, fps=24):
    from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
    from ltx_core.tools import AudioLatentTools, VideoLatentTools
    from ltx_core.types import VIDEO_SCALE_FACTORS, AudioLatentShape, VideoLatentShape, VideoPixelShape

    pixel = VideoPixelShape(batch=1, frames=frames, height=size[0], width=size[1], fps=fps)
    return (
        VideoLatentTools(
            VideoLatentPatchifier(patch_size=1),
            VideoLatentShape.from_pixel_shape(pixel),
            float(fps),
            scale_factors=VIDEO_SCALE_FACTORS,
        ),
        AudioLatentTools(AudioPatchifier(patch_size=1), AudioLatentShape.from_video_pixel_shape(pixel)),
    )


@torch.no_grad()
def prepare_conditioning(source, cache, *, device="cpu", size=(768, 1344), frames=121, fps=24, freeze_audio=False):
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_pipelines.utils.helpers import create_noised_state

    video_tools, audio_tools = native_tools(size=size, frames=frames, fps=fps)
    noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(42))
    video_state = create_noised_state(video_tools, [], noiser, torch.float16, device, REFINER_SIGMAS[0])
    audio_state = create_noised_state(audio_tools, [], noiser, torch.float16, device, REFINER_SIGMAS[0])
    if freeze_audio:
        audio_state = freeze_audio_state(audio_state)
    return _prepare_state_conditioning(source, cache, video_state, audio_state, device=device)


def freeze_audio_state(state):
    """Match upstream: consume the original noise draws, then freeze clean audio."""
    return replace(state, latent=state.clean_latent, denoise_mask=torch.zeros_like(state.denoise_mask), frozen=True)


@torch.no_grad()
def prepare_keyframe_conditioning(source, cache, video_state, audio_state, *, device="cpu"):
    """Fold two video time roles while retaining native target/guide positions.

    The clean role covers frozen first-frame tokens and appended last-frame
    guides. Prompt modulation and A/V gates use each modality's global sigma;
    frozen audio uses zero, including the audio-to-video gate conditioning.
    """
    mask = video_state.denoise_mask
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("the compact LTX keyframe wrapper requires binary video denoise masks")
    audio_mask = 0 if getattr(audio_state, "frozen", False) else 1
    if not bool((audio_state.denoise_mask == audio_mask).all()):
        raise ValueError("LTX audio must be fully frozen or fully generated")
    if getattr(video_state, "attention_mask", None) is not None:
        raise ValueError("the LTX keyframe wrapper requires unrestricted native guide attention")
    return _prepare_state_conditioning(source, cache, video_state, audio_state, device=device, video_time_roles=True)


def _prepare_state_conditioning(source, cache, video_state, audio_state, *, device, video_time_roles=False):
    from ltx_core.model.transformer.modality import Modality

    source.half().to(device)
    videos, audios = [], []
    for sigma in REFINER_SIGMAS[:-1]:

        def modality(state, context, *, time_roles=False, sigma=sigma):
            modality_sigma = 0.0 if getattr(state, "frozen", False) else sigma
            return Modality(
                latent=state.latent.to(device=device, dtype=torch.float16),
                sigma=torch.tensor([modality_sigma], device=device),
                timesteps=torch.tensor(
                    [[modality_sigma, 0.0]] if time_roles else [[modality_sigma]], device=device
                ),
                positions=state.positions.to(device=device),
                context=context.to(device=device, dtype=torch.float16),
                keyframes_mask=(
                    state.keyframes_mask.to(device=device)
                    if getattr(state, "keyframes_mask", None) is not None
                    else None
                ),
            )

        video, audio = (
            modality(video_state, cache["contexts"]["video"], time_roles=video_time_roles),
            modality(audio_state, cache["contexts"]["audio"]),
        )
        videos.append(source.video_args_preprocessor.prepare(video, audio))
        audios.append(source.audio_args_preprocessor.prepare(audio, video))
    rope = tuple(
        torch.cat((value, value), -1)
        for pe in (
            videos[0].positional_embeddings,
            audios[0].positional_embeddings,
            videos[0].cross_positional_embeddings,
            audios[0].cross_positional_embeddings,
        )
        for value in pe
    )
    return videos, audios, rope
