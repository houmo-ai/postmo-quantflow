# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native LTX image conditions with FP32 Sol-H3 solver state.

Frame zero replaces the target's first latent frame. Other pixel-frame indices
append independent image-guide tokens; they never replace a multi-frame latent.
The caller encodes images with the native LTX VideoEncoder before using this
module. Given keyframes use full strength; no RGB frame replacement is applied.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from .conditioning import REFINER_SIGMAS
from .precision import require_finite


def _validate_size(size):
    if not isinstance(size, Sequence) or len(size) != 2 or any(type(n) is not int or n <= 0 or n % 32 for n in size):
        raise ValueError("LTX keyframe size must be two positive multiples of 32")


def _validate_latent(value, shape, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor of shape {tuple(shape)}")
    require_finite(value, name=name)


def validate_keyframe_resolution(image, size):
    """Reject keyframes that require enlargement after native EXIF rotation."""
    from pathlib import Path

    from PIL import Image

    _validate_size(size)
    if isinstance(image, Image.Image):
        width, height = image.size
        orientation = image.getexif().get(274)
    else:
        with Image.open(Path(image).expanduser().resolve(strict=True)) as source:
            width, height = source.size
            orientation = source.getexif().get(274)
    if orientation in (6, 8):
        width, height = height, width
    if height < size[0] or width < size[1]:
        raise ValueError(
            f"LTX keyframe is {width}x{height}; this artifact requires at least {size[1]}x{size[0]} "
            "without enlargement. Supply a clear high-resolution image; resizing a blurred image is insufficient."
        )


def prepare_keyframe_pixels(image, size, crf) -> torch.Tensor:
    """Prepare an SDR PIL image or local image path for the LTX image encoder.

    CRF is checkpoint metadata and must be supplied by the caller. Match the
    native SDR path: supported EXIF rotations, sRGB color conversion, optional
    H.264 recompression, bilinear aspect-ratio-preserving center crop, and
    [-1, 1] normalization. Recompression alone crops odd dimensions to even.
    Returns a CPU FP32 tensor shaped ``(1, 3, 1, H, W)`` without author imports.
    """
    from io import BytesIO
    from pathlib import Path

    import numpy as np
    from PIL import Image, ImageCms

    _validate_size(size)
    if type(crf) is not int or not 0 <= crf <= 51:
        raise ValueError("LTX image conditioning CRF must be a resolved integer in [0, 51]")
    if isinstance(image, Image.Image):
        source = image.copy()
    else:
        path = Path(image).expanduser().resolve(strict=True)
        if path.suffix.lower() == ".exr":
            raise ValueError("LTX SDR keyframe preprocessing does not accept HDR EXR images")
        with Image.open(path) as opened:
            source = opened.copy()
    orientation = source.getexif().get(274)
    if orientation in (3, 6, 8):
        source = source.rotate({3: 180, 6: 270, 8: 90}[orientation], expand=True)
    profile = source.info.get("icc_profile")
    if source.mode == "RGBA":
        source = source.convert("RGB")
    elif source.mode == "LA":
        source = source.convert("L")
    if profile:
        try:
            source = ImageCms.profileToProfile(
                source,
                ImageCms.ImageCmsProfile(BytesIO(profile)),
                ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")),
                outputMode="RGB",
            )
        except (ImageCms.PyCMSError, OSError, ValueError):
            source = source.convert("RGB")
    else:
        source = source.convert("RGB")
    pixels = np.array(source, dtype=np.uint8)
    if crf and min(pixels.shape[:2]) >= 2:
        import av

        with BytesIO() as buffer:
            with av.open(buffer, "w", format="mp4") as container:
                stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
                height, width = pixels.shape[0] // 2 * 2, pixels.shape[1] // 2 * 2
                stream.height, stream.width = height, width
                frame = av.VideoFrame.from_ndarray(pixels[:height, :width], format="rgb24").reformat(format="yuv420p")
                container.mux(stream.encode(frame))
                container.mux(stream.encode())
            video = buffer.getvalue()
        with av.open(BytesIO(video)) as container:
            stream = next(stream for stream in container.streams if stream.type == "video")
            pixels = next(container.decode(stream)).to_ndarray(format="rgb24")
    value = torch.tensor(pixels, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    scale = max(size[0] / pixels.shape[0], size[1] / pixels.shape[1])
    height, width = math.ceil(pixels.shape[0] * scale), math.ceil(pixels.shape[1] * scale)
    value = torch.nn.functional.interpolate(value, size=(height, width), mode="bilinear", align_corners=False)
    top, left = (height - size[0]) // 2, (width - size[1]) // 2
    value = value[:, :, top : top + size[0], left : left + size[1]].unsqueeze(2)
    value = value / 127.5 - 1.0
    require_finite(value, name="LTX normalized keyframe pixels")
    return value


@dataclass(frozen=True)
class StaticKeyframeState:
    """Source-free solver state for the fixed 16-target plus one-guide graph.

    ``first_role`` selects noisy=0 or clean=1 for the first target frame.
    ``video_valid_length`` excludes an unused guide from attention. The guide
    remains present and frozen in solver storage, even when it is not valid.
    All floating tensors are FP32; only graph invocation casts model inputs.
    """

    video: torch.Tensor
    audio: torch.Tensor
    clean_video: torch.Tensor
    video_mask: torch.Tensor
    first_role: torch.Tensor
    video_valid_length: torch.Tensor
    size: tuple[int, int]
    target_frames: int
    audio_frozen: bool = False


@torch.no_grad()
def prepare_static_keyframe_state(
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    *,
    keyframe_latents: Sequence[tuple[int, torch.Tensor]] = (),
    seed: int,
    size=(768, 1344),
    refiner_frames=121,
    device="cpu",
    dtype=torch.float16,
    noise_dtype=torch.bfloat16,
    video_noise: torch.Tensor | None = None,
    audio_noise: torch.Tensor | None = None,
    freeze_audio: bool = False,
) -> StaticKeyframeState:
    """Prepare native-strength endpoint conditions without author libraries.

    The fixed package accepts the first and last pixel-frame indices. Target video
    noise is drawn before audio noise at the requested device, using BF16 by
    default, and no noise is drawn for the appended guide. The two FP32 lerps
    preserve native GaussianNoiser arithmetic, including conditioned rows.
    Replay noise, when provided, uses packed target BNC shapes.
    """
    _validate_size(size)
    if type(freeze_audio) is not bool:
        raise ValueError("LTX freeze_audio must be a boolean")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("LTX keyframe seed is outside the supported range")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("LTX keyframe dtype must be float16, bfloat16 or float32")
    if noise_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("LTX noise dtype must be float16, bfloat16 or float32")
    height, width = size[0] // 32, size[1] // 32
    frame_tokens = height * width
    if type(refiner_frames) is not int or refiner_frames < 9 or (refiner_frames - 1) % 8:
        raise ValueError("LTX output frame count must be a positive 8n+1 temporal gear")
    target_frames = (refiner_frames - 1) // 8 + 1
    audio_frames = round(refiner_frames / 24 * 25)
    last_frame = refiner_frames - 1
    video_shape = (1, 128, target_frames, height, width)
    audio_shape = (1, 8, audio_frames, 16)
    image_shape = (1, 128, 1, height, width)
    _validate_latent(video_latent, video_shape, "LTX target video latent")
    _validate_latent(audio_latent, audio_shape, "LTX target audio latent")
    if not isinstance(keyframe_latents, Sequence):
        raise ValueError("LTX keyframe latents must be a sequence of (pixel index, latent) pairs")
    images = {}
    for frame_idx, latent in keyframe_latents:
        if type(frame_idx) is not int or frame_idx not in (0, last_frame) or frame_idx in images:
            raise ValueError("static LTX keyframes require unique endpoint first/last indices")
        _validate_latent(latent, image_shape, "LTX image latent")
        images[frame_idx] = latent

    device = torch.device(device)
    video = video_latent.to(device=device, dtype=torch.float32).permute(0, 2, 3, 4, 1).reshape(1, -1, 128)
    audio = audio_latent.to(device=device, dtype=torch.float32).permute(0, 2, 1, 3).reshape(1, audio_frames, 128)
    generator = torch.Generator(device=device).manual_seed(seed)
    sampled = [
        torch.randn(value.shape, generator=generator, device=device, dtype=noise_dtype).float()
        for value in (video, audio)
    ]
    for index, (value, reference) in enumerate(zip((video_noise, audio_noise), (video, audio), strict=True)):
        if value is not None:
            _validate_latent(value, reference.shape, "LTX replay noise")
            sampled[index] = value.to(device=device, dtype=torch.float32)

    padded_video = torch.nn.functional.pad(video, (0, 0, 0, frame_tokens))
    clean = padded_video.clone()
    mask = torch.ones((1, frame_tokens * (target_frames + 1), 1), device=device, dtype=torch.float32)
    mask[:, frame_tokens * target_frames :] = 0
    for frame_idx, value in images.items():
        value = value.to(device=device, dtype=dtype).float().permute(0, 2, 3, 4, 1).reshape(1, frame_tokens, 128)
        begin = 0 if frame_idx == 0 else frame_tokens * target_frames
        clean[:, begin : begin + frame_tokens] = value
        mask[:, begin : begin + frame_tokens] = 0
    sigma = REFINER_SIGMAS[0]
    noised = torch.lerp(padded_video, torch.nn.functional.pad(sampled[0], (0, 0, 0, frame_tokens)), sigma)
    video = torch.lerp(clean, noised, mask)
    if not freeze_audio:
        audio = torch.lerp(audio, sampled[1], sigma)
    require_finite((video, audio), name="static LTX initial state")
    return StaticKeyframeState(
        video=video,
        audio=audio,
        clean_video=clean,
        video_mask=mask,
        first_role=torch.tensor([int(0 in images)], dtype=torch.int32, device=device),
        video_valid_length=torch.tensor(
            [frame_tokens * (target_frames + int(last_frame in images))], dtype=torch.int32, device=device
        ),
        size=tuple(size),
        target_frames=target_frames,
        audio_frozen=freeze_audio,
    )


@torch.no_grad()
def static_keyframe_step(state, video_velocity, audio_velocity, sigma: float, next_sigma: float):
    """Native X0/postprocess/Euler operation order, entirely in package Host.

    Keep the conversion to X0 and back to velocity: simplifying to a direct
    velocity update changes FP32 rounding and clean-reference semantics.
    """
    if not math.isfinite(sigma) or not math.isfinite(next_sigma) or not 0 <= next_sigma < sigma <= 1:
        raise ValueError("LTX Euler sigmas must decrease from (0, 1] towards zero")
    _validate_latent(video_velocity, state.video.shape, "LTX video velocity")
    _validate_latent(audio_velocity, state.audio.shape, "LTX audio velocity")
    sigmas = state.video.new_tensor([sigma, next_sigma], dtype=torch.float32)
    current, delta = sigmas[0], sigmas[1] - sigmas[0]
    video_x0 = state.video - video_velocity.float() * (state.video_mask * sigma)
    video_x0 = video_x0 * state.video_mask + state.clean_video * (1 - state.video_mask)
    video = state.video + ((state.video - video_x0) / current) * delta
    audio = state.audio
    if not state.audio_frozen:
        audio_x0 = state.audio - audio_velocity.float() * sigma
        audio = state.audio + ((state.audio - audio_x0) / current) * delta
    require_finite((video, audio), name="static LTX solver step")
    return replace(state, video=video, audio=audio)


def clear_static_keyframe_state(state: StaticKeyframeState) -> torch.Tensor:
    """Drop the guide and return normalized BCTHW target latents for decode."""
    height, width = state.size[0] // 32, state.size[1] // 32
    target_tokens = state.target_frames * height * width
    return (
        state.video[:, :target_tokens].reshape(1, state.target_frames, height, width, 128)
        .permute(0, 4, 1, 2, 3).contiguous()
    )


@torch.no_grad()
def prepare_keyframe_states(
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    *,
    keyframe_latents: Sequence[tuple[int, torch.Tensor]] = (),
    seed: int,
    size=(768, 1344),
    refiner_frames=121,
    device="cpu",
    dtype=torch.float16,
    noise_dtype=torch.bfloat16,
    video_noise: torch.Tensor | None = None,
    audio_noise: torch.Tensor | None = None,
    freeze_audio: bool = False,
):
    """Return ``(video_state, audio_state, video_tools, audio_tools)``.

    Input latents use the native LTX normalized convention. Image latents have
    shape ``(1, 128, 1, H/32, W/32)`` and valid output pixel indices. Reference
    latents cross the requested model ``dtype`` boundary; solver tensors remain
    FP32. ``state_modality`` casts a state for the model without changing it.

    Draw the original target video/audio noise before appending guide tokens.
    This preserves Sol-H3's seeded BF16 noise and audio RNG order, independent
    of the number of references. Optional replay noise is packed ``BNC`` in the
    original target shape. Native ``GaussianNoiser`` performs the actual mix;
    appended, fully frozen guide rows receive unused zero noise.
    """
    _validate_size(size)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("LTX keyframe seed is outside the supported range")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("LTX keyframe dtype must be float16, bfloat16 or float32")
    if noise_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("LTX noise dtype must be float16, bfloat16 or float32")
    if type(refiner_frames) is not int or refiner_frames < 9 or (refiner_frames - 1) % 8:
        raise ValueError("LTX output frame count must be a positive 8n+1 temporal gear")
    video_shape = (1, 128, (refiner_frames - 1) // 8 + 1, size[0] // 32, size[1] // 32)
    audio_shape = (1, 8, round(refiner_frames / 24 * 25), 16)
    image_shape = (1, 128, 1, *video_shape[-2:])
    _validate_latent(video_latent, video_shape, "LTX target video latent")
    _validate_latent(audio_latent, audio_shape, "LTX target audio latent")
    if not isinstance(keyframe_latents, Sequence):
        raise ValueError("LTX keyframe latents must be a sequence of (pixel index, latent) pairs")
    seen = set()
    for frame_idx, latent in keyframe_latents:
        if type(frame_idx) is not int or not 0 <= frame_idx < refiner_frames or frame_idx in seen:
            raise ValueError("LTX keyframe pixel indices must be unique and inside the output")
        seen.add(frame_idx)
        _validate_latent(latent, image_shape, "LTX image latent")

    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.conditioning import VideoConditionByKeyframeIndex, VideoConditionByLatentIndex
    from ltx_pipelines.utils.helpers import create_noised_state

    from .ltx_prepare import freeze_audio_state, native_tools

    video_tools, audio_tools = native_tools(size=size, frames=refiner_frames)
    device = torch.device(device)
    video = video_latent.to(device=device, dtype=torch.float32)
    audio = audio_latent.to(device=device, dtype=torch.float32)
    packed_video = video_tools.patchifier.patchify(video)
    packed_audio = audio_tools.patchifier.patchify(audio)
    generator = torch.Generator(device=device).manual_seed(seed)
    sampled = [
        torch.randn(value.shape, generator=generator, device=device, dtype=noise_dtype).float()
        for value in (packed_video, packed_audio)
    ]
    for index, (value, reference) in enumerate(
        zip((video_noise, audio_noise), (packed_video, packed_audio), strict=True)
    ):
        if value is not None:
            _validate_latent(value, reference.shape, "LTX replay noise")
            sampled[index] = value.to(device=device, dtype=torch.float32)

    class ReplayNoiser(GaussianNoiser):
        def __init__(self, noise):
            super().__init__(generator)
            self.noise = noise

        def _sample_noise(self, latent_state):
            extra = latent_state.latent.shape[1] - self.noise.shape[1]
            if extra < 0:
                raise ValueError("conditioned state is shorter than its target noise")
            return torch.nn.functional.pad(self.noise, (0, 0, 0, extra))

    conditions = []
    for frame_idx, latent in keyframe_latents:
        latent = latent.to(device=device, dtype=dtype).float()
        if frame_idx == 0:
            conditions.append(VideoConditionByLatentIndex(latent=latent, strength=1.0, latent_idx=0))
        else:
            conditions.append(VideoConditionByKeyframeIndex(keyframes=latent, frame_idx=frame_idx, strength=1.0))
    sigma = REFINER_SIGMAS[0]
    video_state = create_noised_state(
        video_tools, conditions, ReplayNoiser(sampled[0]), torch.float32, device, sigma, video
    )
    audio_state = create_noised_state(audio_tools, [], ReplayNoiser(sampled[1]), torch.float32, device, sigma, audio)
    if freeze_audio:
        audio_state = freeze_audio_state(audio_state)
    return video_state, audio_state, video_tools, audio_tools


def video_timestep_ranges(video_state) -> tuple[tuple[int, int, int], ...]:
    """Contiguous ``(start, end, role)`` spans: role 0=sigma, role 1=clean t=0.

    The folded engineering graph has exactly these two timestep categories.
    Refuse fractional masks instead of silently approximating soft conditioning.
    """
    mask = video_state.denoise_mask
    if mask.ndim != 3 or mask.shape[0] != 1 or mask.shape[-1] != 1 or mask.shape[1] == 0:
        raise ValueError("LTX timestep ranges require a nonempty batch-one BNC mask")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("LTX folded timestep ranges require a binary denoise mask")
    values, counts = torch.unique_consecutive(mask[0, :, 0], return_counts=True)
    start, ranges = 0, []
    for value, count in zip(values.tolist(), counts.tolist(), strict=True):
        ranges.append((start, start + count, 1 - int(value)))
        start += count
    return tuple(ranges)


def state_modality(state, context, sigma, *, dtype=torch.float16):
    """Native per-token ``mask * sigma``, positions and masks at model dtype."""
    from ltx_pipelines.utils.helpers import modality_from_latent_state

    model_state = replace(state, latent=state.latent.to(dtype=dtype))
    sigma = torch.as_tensor(sigma, device=state.latent.device, dtype=torch.float32).reshape(1)
    return modality_from_latent_state(model_state, context.to(device=state.latent.device, dtype=dtype), sigma)


@torch.no_grad()
def masked_ltx_step(state, velocity, sigma: float, next_sigma: float):
    """Native X0 conversion, clean-reference postprocess and Euler in FP32.

    Keep the per-token timestep in the X0 conversion and the global schedule in
    Euler. This also follows native soft-mask arithmetic; merely multiplying a
    velocity by a fractional mask would not reproduce that arithmetic.
    """
    if not math.isfinite(sigma) or not math.isfinite(next_sigma) or not 0 <= next_sigma < sigma <= 1:
        raise ValueError("LTX Euler sigmas must decrease from (0, 1] towards zero")
    _validate_latent(velocity, state.latent.shape, "LTX velocity")
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.utils import to_denoised
    from ltx_pipelines.utils.helpers import post_process_latent, timesteps_from_mask

    state = replace(state, latent=state.latent.float(), clean_latent=state.clean_latent.float())
    denoised = to_denoised(state.latent, velocity.float(), timesteps_from_mask(state.denoise_mask, sigma))
    denoised = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
    sigmas = state.latent.new_tensor([sigma, next_sigma])
    updated = EulerDiffusionStep().step(state.latent, denoised, sigmas, 0)
    require_finite(updated, name="conditioned LTX solver step")
    return replace(state, latent=updated)
