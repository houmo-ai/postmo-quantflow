# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Prompt/layout, signal preprocessing and FP32 solver state outside model graphs."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .conditioning import AUDIO_SIGMAS, REFINER_SIGMAS, VIDEO_SIGMAS
from .precision import require_finite


def initial_h3(seed, *, size=(384, 672), video_frames=37, audio_frames=207):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents

    generator = torch.Generator(device="cpu").manual_seed(seed)
    height, width = size
    video = torch.randn((1, 24, video_frames, height // 16, width // 16), generator=generator, dtype=torch.float32)
    video = patchify_video_latents(video, (1, 2, 2))[None]
    audio = torch.randn((1, 2 * audio_frames, 32), generator=generator, dtype=torch.float32)
    return video, audio


def h3_rope(text_length, *, text_capacity=512, size=(384, 672), video_frames=37, audio_frames=207):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    if not 0 < text_length <= text_capacity:
        raise ValueError("prompt length exceeds the H3 text gear")
    positions, _, vi, ai, ti, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.ones(text_length, dtype=torch.int64),
        video_frames, size[0] // 16, size[1] // 16, audio_frames, (1, 2, 2), 2, 2, 0,
    )
    positions = positions[torch.cat((vi, ai, ti))].float()
    positions = F.pad(positions, (0, 0, 0, text_capacity - text_length))
    inv = 1 / (10000 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
    phase = (positions[:, :, None] * inv).flatten(1)
    phase = torch.cat((phase, phase), -1)
    return phase.cos()[None, None], phase.sin()[None, None]


def h3_updates(invoke, video, audio, text, cos, sin, text_length):
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    video_scheduler, audio_scheduler = MiniMaxH3Scheduler(), MiniMaxH3Scheduler()
    video_scheduler.set_timesteps(sigmas=VIDEO_SIGMAS)
    audio_scheduler.set_timesteps(sigmas=AUDIO_SIGMAS)
    video, audio = video.float(), audio.float()
    length = torch.tensor([video.shape[1] + audio.shape[1] + text_length], dtype=torch.int32)
    for step in range(4):
        v, a = invoke(video, audio, text, torch.tensor([step], dtype=torch.int32), cos, sin, length)
        require_finite((v, a), name=f"H3 velocity step {step}")
        video = video_scheduler.step(v.float(), video_scheduler.timesteps[step], video, return_dict=False)[0]
        audio = audio_scheduler.step(a.float(), audio_scheduler.timesteps[step], audio, return_dict=False)[0]
        require_finite((video, audio), name=f"H3 solver step {step}")
    return video, audio


def unpatch_h3(video, *, size=(384, 672), frames=37):
    height, width = size[0] // 16, size[1] // 16
    return (
        video.reshape(1, frames, height // 2, width // 2, 24, 1, 2, 2)
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .reshape(1, 24, frames, height, width)
    )


def upscale_h3(invoke, latent, plans, statistics):
    mean, std = (
        torch.tensor(statistics[name], dtype=torch.float32).reshape(1, 24, 1, 1, 1) for name in ("mean", "std")
    )
    normalized = (latent.float() - mean) / std
    padded = F.pad(normalized, (0, 0, 0, 0, 5, 5), mode="replicate")
    output = torch.zeros(1, 24, latent.shape[2], latent.shape[3] * 2, latent.shape[4] * 2)
    weights = torch.zeros(1, 1, latent.shape[2], 1, 1)
    for plan in plans:
        segment = padded[:, :, plan["input_start"] : plan["input_end"]]
        value = invoke(f"h3_upscale_t{segment.shape[2]}", segment).float()
        value = value[:, :, plan["crop_start"] : plan["crop_end"]]
        weight = torch.tensor(plan["weights"]).reshape(1, 1, -1, 1, 1)
        start, end = plan["output_start"], plan["output_end"]
        output[:, :, start:end] += value * weight
        weights[:, :, start:end] += weight
    if bool((weights <= 0).any()):
        raise ValueError("upscaler chunk plan leaves frames uncovered")
    result = (output / weights) * std + mean
    require_finite(result, name="H3 x2 upscale")
    return result


def audio_mel(pcm, *, source_rate, config):
    import torchaudio

    waveform = pcm.detach().float().reshape(1, 2, -1)
    rate = config["sample_rate"]
    if source_rate != rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, rate)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=rate,
        n_fft=config["n_fft"],
        win_length=config["n_fft"],
        hop_length=config["mel_hop_length"],
        f_min=0,
        f_max=rate / 2,
        n_mels=config["mel_bins"],
        window_fn=torch.hann_window,
        center=True,
        pad_mode="reflect",
        power=1.0,
        mel_scale="slaney",
        norm="slaney",
    )
    result = transform(waveform).clamp_min(1e-5).log().permute(0, 1, 3, 2).contiguous()
    require_finite(result, name="LTX log-mel frontend")
    return result


def audio_for_video(pcm, *, frames, fps=24, sample_rate=32000):
    """Fit output PCM to the presentation duration of the decoded video."""
    if type(frames) is not int or frames <= 0 or fps <= 0 or sample_rate <= 0:
        raise ValueError("video frames, fps and audio sample rate must be positive")
    samples = round(frames * sample_rate / fps)
    if pcm.shape[-1] < samples:
        return F.pad(pcm, (0, samples - pcm.shape[-1]))
    return pcm[..., :samples]


def initial_ltx(video_latent, audio_latent, seed, *, device="cuda:0", freeze_audio=False):
    video = video_latent.float().permute(0, 2, 3, 4, 1).reshape(1, -1, 128)
    audio = audio_latent.float().permute(0, 2, 1, 3).reshape(1, audio_latent.shape[2], 128)
    generator = torch.Generator(device=device).manual_seed(seed)
    sigma = REFINER_SIGMAS[0]
    # Match the reference's CUDA BF16 random draw while retaining FP32 solver state.
    vn = torch.randn(video.shape, device=device, dtype=torch.bfloat16, generator=generator).float().cpu()
    an = torch.randn(audio.shape, device=device, dtype=torch.bfloat16, generator=generator).float().cpu()
    return (1 - sigma) * video + sigma * vn, audio.cpu() if freeze_audio else (1 - sigma) * audio + sigma * an


def ltx_updates(invoke, video, audio, rope=(), *, freeze_audio=False):
    for step, (sigma, next_sigma) in enumerate(zip(REFINER_SIGMAS[:-1], REFINER_SIGMAS[1:], strict=True)):
        v, a = invoke(video, audio, torch.tensor([step], dtype=torch.int32), *rope)
        require_finite((v, a), name=f"LTX velocity step {step}")
        # LTX predicts noise-ward velocity, opposite to H3's convention.
        video = video.float() + (next_sigma - sigma) * v.float()
        if not freeze_audio:
            audio = audio.float() + (next_sigma - sigma) * a.float()
        require_finite((video, audio), name=f"LTX solver step {step}")
    return video, audio


def unpatch_ltx(video, *, size=(768, 1344), frames=16):
    return video.reshape(1, frames, size[0] // 32, size[1] // 32, 128).permute(0, 4, 1, 2, 3).contiguous()


def conditioned_h3_layout(
    text_tags, *, size=(384, 672), anchors=(), references=(), visual_latents=(), audio_latents=(),
    video_frames=37, audio_frames=207,
):
    """Reuse native rotary positions, retaining the export's video/audio/text order."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3Ref2VAPrepareLayoutStep,
    )

    geometry = (video_frames, size[0] // 16, size[1] // 16, audio_frames, (1, 2, 2), 2, 2, 0)
    if references:
        if anchors:
            raise ValueError("reference and first/last frame inputs are mutually exclusive")
        layout = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
            text_tags,
            list(references),
            list(visual_latents),
            list(audio_latents),
            *geometry,
        )
    else:
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(text_tags, *geometry, keyframe_anchors=anchors)
    positions, tags, vi, ai, ti, cv, ca = layout
    order = torch.cat((vi, ai, ti))
    roles = tags.clone()
    roles[vi[:cv]] = 3
    roles[ai[:ca]] = 4
    return positions[order].float(), roles[order].int(), cv, ca


def condition_slot_layout(lengths, capacities, roles, positions):
    """Place valid modal rows in fixed slots and compact their attention sequence.

    Inputs follow video/audio/text order. RoPE positions are already in compact
    order from the native H3 layout; padding is appended only after valid rows.
    """
    if (
        len(lengths) != 3
        or len(capacities) != 3
        or any(
            type(n) is not int or type(c) is not int or not 0 <= n <= c or c <= 0
            for n, c in zip(lengths, capacities, strict=True)
        )
    ):
        raise ValueError("H3 video/audio/text row counts must fit their exported capacities")
    valid = sum(lengths)
    if roles.shape != (valid,) or positions.shape != (valid, 3):
        raise ValueError("native roles/positions must describe exactly the valid rows")
    padded_roles = torch.ones(sum(capacities), dtype=torch.int32)
    used, unused, source_offset, target_offset = [], [], 0, 0
    for count, capacity in zip(lengths, capacities, strict=True):
        padded_roles[target_offset : target_offset + count] = roles[source_offset : source_offset + count]
        used.extend(range(target_offset, target_offset + count))
        unused.extend(range(target_offset + count, target_offset + capacity))
        source_offset += count
        target_offset += capacity
    pack = torch.tensor(used + unused, dtype=torch.int32)
    unpack = torch.argsort(pack).to(torch.int32)
    return padded_roles, F.pad(positions, (0, 0, 0, sum(capacities) - valid)), pack, unpack


class ConditionedDraft:
    """One request's native noise/layout over a fixed-capacity conditioned graph."""

    def __init__(self, request, *, seed, capacities, rope_config):
        from .request import prepare_latents

        self.video, self.audio, roles, positions, self.cv, self.ca = prepare_latents(request, seed=seed)
        self.text_length = request.state.get("text_token_tags").numel()
        self._configure_slots(capacities, rope_config, roles, positions)

    @classmethod
    def from_text(cls, *, seed, size, text_length, capacities, rope_config, video_frames=37, audio_frames=207):
        """Use the same padded slots for T2VA without introducing media encoders."""
        result = cls.__new__(cls)
        result.video, result.audio = initial_h3(seed, size=size, video_frames=video_frames, audio_frames=audio_frames)
        result.text_length = text_length
        positions, roles, result.cv, result.ca = conditioned_h3_layout(
            torch.ones(text_length, dtype=torch.int64), size=size, video_frames=video_frames, audio_frames=audio_frames
        )
        result._configure_slots(capacities, rope_config, roles, positions)
        return result

    def _configure_slots(self, capacities, rope_config, roles, positions):
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

        self.lengths = (self.video.shape[1], self.audio.shape[1], self.text_length)
        self.capacities = tuple(capacities[key] for key in ("video_capacity", "audio_capacity", "text_capacity"))
        self.roles, padded_positions, self.pack, self.unpack = condition_slot_layout(
            self.lengths, self.capacities, roles, positions
        )
        cos, sin = MiniMaxH3RotaryPosEmbed(**rope_config)(padded_positions)
        self.cos, self.sin = cos[None, None], sin[None, None]
        self.valid = torch.tensor([sum(self.lengths)], dtype=torch.int32)

    def inputs(self, text, *, video=None, audio=None, step=None):
        video = self.video if video is None else video
        audio = self.audio if audio is None else audio
        if text.shape[1] not in (self.text_length, self.capacities[2]):
            raise ValueError("conditioned text must contain its valid rows or the exported capacity")
        padded = tuple(
            F.pad(value, (0, 0, 0, capacity - value.shape[1]))
            for value, capacity in zip((video, audio, text), self.capacities, strict=True)
        )
        return (
            *padded,
            torch.zeros(1, dtype=torch.int32) if step is None else step,
            self.cos,
            self.sin,
            self.valid,
            self.roles,
            self.pack,
            self.unpack,
        )

    def run(self, invoke, text, *, schedule=None):
        def velocities(video, audio, text, step, *_):
            v, a = invoke(*self.inputs(text, video=video, audio=audio, step=step))
            return v[:, : self.lengths[0]], a[:, : self.lengths[1]]

        return conditioned_h3_updates(
            velocities,
            self.video,
            self.audio,
            text,
            self.cos,
            self.sin,
            self.text_length,
            self.roles,
            schedule=schedule,
            condition_video_rows=self.cv,
            condition_audio_rows=self.ca,
        )


class RolePartitionedDraft(ConditionedDraft):
    """Native request state over fixed AdaLN role slots and block-valid holes."""

    def __init__(self, request, *, role_capacities, attention_heads, attention_block_size=256, **kwargs):
        self.role_capacities = tuple(role_capacities)
        self.attention_heads, self.attention_block_size = attention_heads, attention_block_size
        super().__init__(request, **kwargs)

    @classmethod
    def from_text(
        cls,
        *,
        seed,
        size,
        text_length,
        capacities,
        rope_config,
        role_capacities,
        attention_heads,
        attention_block_size=256,
        video_frames=37,
        audio_frames=207,
    ):
        result = cls.__new__(cls)
        result.role_capacities = tuple(role_capacities)
        result.attention_heads, result.attention_block_size = attention_heads, attention_block_size
        result.video, result.audio = initial_h3(seed, size=size, video_frames=video_frames, audio_frames=audio_frames)
        result.text_length = text_length
        positions, roles, result.cv, result.ca = conditioned_h3_layout(
            torch.ones(text_length, dtype=torch.int64), size=size, video_frames=video_frames, audio_frames=audio_frames
        )
        result._configure_slots(capacities, rope_config, roles, positions)
        return result

    def _configure_slots(self, capacities, rope_config, roles, positions):
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

        from .role_layout import make_role_layout

        self.lengths = (self.video.shape[1], self.audio.shape[1], self.text_length)
        self.capacities = tuple(capacities[key] for key in ("video_capacity", "audio_capacity", "text_capacity"))
        self.layout = make_role_layout(
            self.lengths,
            self.capacities,
            roles,
            positions,
            self.role_capacities,
            attention_heads=self.attention_heads,
            block_size=self.attention_block_size,
        )
        self.roles = self.layout.roles
        self.pack, self.head_unpack = self.layout.pack, self.layout.head_unpack
        self.key_mask = self.layout.key_mask
        cos, sin = MiniMaxH3RotaryPosEmbed(**rope_config)(self.layout.positions)
        self.cos, self.sin = cos[None, None], sin[None, None]
        self.valid = torch.tensor([sum(self.lengths)], dtype=torch.int32)

    def inputs(self, text, *, video=None, audio=None, step=None):
        video = self.video if video is None else video
        audio = self.audio if audio is None else audio
        if text.shape[1] not in (self.text_length, self.capacities[2]):
            raise ValueError("conditioned text must contain its valid rows or the exported capacity")
        padded = tuple(
            F.pad(value, (0, 0, 0, capacity - value.shape[1]))
            for value, capacity in zip((video, audio, text), self.capacities, strict=True)
        )
        return (
            *padded,
            torch.zeros(1, dtype=torch.int32) if step is None else step,
            self.cos,
            self.sin,
            self.layout.pack,
            self.layout.head_unpack,
            self.key_mask,
        )


def conditioned_h3_updates(
    invoke,
    video,
    audio,
    text,
    cos,
    sin,
    text_length,
    row_roles,
    *,
    condition_video_rows=0,
    condition_audio_rows=0,
    schedule=None,
):
    """Keep reference rows fixed, including their attention updates inside every DiT call."""
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    from .conditioning import H3Schedule

    schedule = schedule or H3Schedule()
    vs, aus = MiniMaxH3Scheduler(), MiniMaxH3Scheduler()
    vs.set_timesteps(sigmas=schedule.video_sigmas)
    aus.set_timesteps(sigmas=schedule.audio_sigmas)
    video, audio = video.float().clone(), audio.float().clone()
    cv, ca = condition_video_rows, condition_audio_rows
    length = torch.tensor([video.shape[1] + audio.shape[1] + text_length], dtype=torch.int32)
    for step in range(schedule.steps):
        v, a = invoke(video, audio, text, torch.tensor([step], dtype=torch.int32), cos, sin, length, row_roles)
        require_finite((v, a), name=f"conditioned H3 velocity step {step}")
        video[:, cv:] = vs.step(v[:, cv:].float(), vs.timesteps[step], video[:, cv:], return_dict=False)[0]
        audio[:, ca:] = aus.step(a[:, ca:].float(), aus.timesteps[step], audio[:, ca:], return_dict=False)[0]
        require_finite((video, audio), name=f"conditioned H3 solver step {step}")
    return video[:, cv:], audio[:, ca:]
