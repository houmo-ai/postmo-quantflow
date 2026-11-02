# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Sol-H3 request routing around the native H3 media preparation blocks."""

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


MODES = ("t2va", "i2va", "l2va", "fl2va", "ref2va")


@dataclass
class PreparedRequest:
    mode: str
    state: object
    components: object

    @property
    def keyframes(self):
        return self.state.get("keyframes") or []

    @property
    def anchors(self):
        return self.state.get("keyframe_anchors") or ()

    @property
    def references(self):
        return self.state.get("normalized_references") or []


def _local_image(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    with Image.open(Path(value).expanduser().resolve(strict=True)) as image:
        return image.convert("RGB")


def _reference_image(value):
    """Convert an already decoded native reference to an RGB PIL image."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().movedim(0, -1).numpy()
    value = np.asarray(value)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("reference images must contain RGB pixels")
    if value.dtype != np.uint8:
        value = (value * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(value, mode="RGB")


def _reference_video_shape(frames):
    if isinstance(frames, list):
        width, height = frames[0].size
        return height, width
    if isinstance(frames, torch.Tensor):
        return int(frames.shape[-2]), int(frames.shape[-1])
    return int(frames.shape[1]), int(frames.shape[2])


def _reference(value):
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3Reference,
        MiniMaxH3VideoReference,
    )

    if isinstance(value, MiniMaxH3Reference):
        return value
    if not isinstance(value, dict) or set(value) != {"kind", "path"}:
        raise ValueError("references must be native H3 references or {kind, path} dictionaries")
    classes = dict(image=MiniMaxH3ImageReference, video=MiniMaxH3VideoReference, audio=MiniMaxH3AudioReference)
    if value["kind"] not in classes:
        raise ValueError("reference kind must be image, video or audio")
    path = Path(value["path"]).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("reference path must name a local media file")
    reference = classes[value["kind"]].from_file(path)
    if reference.has_audio:
        _trim_file_audio_padding(reference, path)
    return reference


def _trim_file_audio_padding(reference, path):
    """Remove only a codec's partial final frame beyond the audio presentation.

    Native file decoding already skips encoder priming, but concatenates the
    final decoded frame in full. Stream duration is a length (independent of
    start_time), not the container/video duration or the configured budget.
    Unknown duration and larger discrepancies retain the decoded samples so
    normal duration validation can reject them instead of hiding real audio.
    """
    import av

    with av.open(str(path)) as container:
        # Match the native from_file loader's first audio stream selection.
        stream = container.streams.audio[0]
        frame_size = stream.codec_context.frame_size
        if stream.duration is None or stream.time_base is None or not frame_size:
            return
        samples = ceil(stream.duration * stream.time_base * reference.sample_rate)
        excess = reference.audio.shape[-1] - samples
        if samples > 0 and 0 < excess < frame_size:
            reference.audio = reference.audio[..., :samples]


def _validate_durations(references, limits=None):
    totals = dict(video=0.0, audio=0.0)
    if limits is not None:
        for kind in ("image", "video", "audio"):
            maximum = limits.get(kind, {}).get("max_count", 0)
            if sum(ref.kind == kind for ref in references) > maximum:
                raise ValueError(f"reference {kind} count exceeds configured maximum {maximum}")
    for reference in references:
        if reference.kind == "video":
            if not np.isfinite(reference.fps) or reference.fps <= 0:
                raise ValueError("reference video fps must be finite and positive")
            # All native layouts (THWC arrays, TCHW tensors, PIL lists) are T-first.
            duration = len(reference.frames) / reference.fps
            if not 2 <= duration <= 15:
                raise ValueError("each reference video must last between 2 and 15 seconds")
            if limits is not None and duration > limits.get("video", {}).get("max_duration_seconds", 5) + 1e-6:
                raise ValueError("reference video duration exceeds configured maximum")
            totals["video"] += duration
        if reference.has_audio:
            rate = 32000 if reference.sample_rate is None else reference.sample_rate
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("reference audio requires its finite positive sample_rate")
            duration = reference.audio.shape[-1] / rate
            if not 2 <= duration <= 15:
                raise ValueError("each reference soundtrack must last between 2 and 15 seconds")
            if limits is not None and duration > limits.get("audio", {}).get("max_duration_seconds", 5) + 1e-6:
                raise ValueError("reference soundtrack duration exceeds configured maximum")
            totals["audio"] += duration
    if any(total > 15 + 1e-6 for total in totals.values()):
        raise ValueError("reference videos and soundtracks each have a 15-second total budget")


def infer_mode(*, image=None, last_image=None, references=None):
    """Resolve media routing before opening any file or loading model weights."""
    if references is not None and (not isinstance(references, (list, tuple)) or not references):
        raise ValueError("references must be a nonempty ordered sequence")
    if references is not None and (image is not None or last_image is not None):
        raise ValueError("Ref2VA references cannot be combined with first/last keyframes")
    return (
        "ref2va"
        if references is not None
        else (
            "fl2va"
            if image is not None and last_image is not None
            else "i2va"
            if image is not None
            else "l2va"
            if last_image is not None
            else "t2va"
        )
    )


def prepare_request(
    prompt,
    *,
    size,
    image=None,
    last_image=None,
    references=None,
    mode=None,
    reference_image_short_edge=2048,
    num_frames=124,
    reference_limits=None,
):
    """Resolve five modes without reimplementing native crop, resample or labels.

    Source media keep their own rates. Ref2VA order is never regrouped by kind.
    The reference image edge is explicit because native H3 and Sol-H3 variants
    can use different conditioning resolutions.
    """
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep, MiniMaxH3ResizeStep
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Sol-H3 requires a nonempty prompt")
    if len(size) != 2 or any(type(n) is not int or n <= 0 or n % 32 for n in size):
        raise ValueError("Sol-H3 canvas height and width must be positive multiples of 32")
    inferred = infer_mode(image=image, last_image=last_image, references=references)
    if mode is not None and mode != inferred:
        raise ValueError(f"mode {mode!r} does not match inputs for {inferred}")
    if reference_limits and inferred != "ref2va":
        raise ValueError("reference limits apply only to ref2va")
    if (
        type(reference_image_short_edge) is not int
        or reference_image_short_edge <= 0
        or reference_image_short_edge % 32
    ):
        raise ValueError("reference_image_short_edge must be a positive multiple of 32")
    config = SimpleNamespace(
        canvas_short_edge=min(size),
        canvas_max_pixels=size[0] * size[1],
        reference_image_short_edge=reference_image_short_edge,
    )
    components = SimpleNamespace(
        config=config,
        image_processor=VaeImageProcessor(vae_scale_factor=16),
        canvas_multiple=32,
        fps=24,
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        # The 15-second Sol-H3 gear uses 362 aligned H3 frames, slightly
        # exceeding 15 seconds before the adapter crops to 361 output frames.
        min_duration=min(5, num_frames / 24),
        max_duration=max(15, num_frames / 24),
        audio_sampling_rate=32000,
    )
    components._execution_device = torch.device("cpu")
    for name in (
        "pixel_mean",
        "pixel_std",
        "keyframe_encode_seed",
        "keyframe_noise_aug",
        "text_tag",
        "video_tag",
        "audio_tag",
        "audio_channels",
        "audio_latent_channels",
    ):
        setattr(components, name, getattr(MiniMaxH3ModularPipeline, name).fget(components))
    state = PipelineState(values=dict(prompt=prompt, height=size[0], width=size[1], num_frames=num_frames))
    if inferred == "ref2va":
        from .reference_geometry import image_gears, image_size, video_gear, video_size

        entries = [_reference(value) for value in references]
        _validate_durations(entries, reference_limits)
        limits = reference_limits or {}
        gears = image_gears(limits.get("image", {}))
        canvas = video_gear(limits.get("video", {}))
        if gears:
            # The native setup normalizes references in order; its short-edge
            # result is replaced below by one direct aspect-aware resize.
            components.config.reference_image_short_edge = 32
        if limits.get("independent_duration", False):
            from diffusers.modular_pipelines.minimax_h3.modular_pipeline import align_num_frames

            reference_seconds = max(
                limits.get("video", {}).get("max_duration_seconds", 0),
                limits.get("audio", {}).get("max_duration_seconds", 0),
            )
            target_frames = align_num_frames(num_frames, 17, 5)
            reference_frames = align_num_frames(ceil(reference_seconds * 24), 17, 5) if reference_seconds else 0
            state.set("num_frames", max(target_frames, reference_frames))
            components.max_duration = max(components.max_duration, state.get("num_frames") / 24)
        state.set("references", entries)
        if canvas:
            class FixedCanvasSetup(MiniMaxH3Ref2VASetupStep):
                @staticmethod
                def _normalize_video_condition(frames, fps, count, multiple, short_edge, max_pixels, target_fps):
                    height, width = _reference_video_shape(frames)
                    wanted = video_size(height, width, canvas)
                    result = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
                        frames, fps, count, multiple, min(wanted), wanted[0] * wanted[1], target_fps
                    )
                    if tuple(result.shape[1:3]) != wanted:
                        result = np.stack([
                            np.asarray(Image.fromarray(frame).resize((wanted[1], wanted[0]), Image.Resampling.LANCZOS))
                            for frame in result
                        ])
                    return result

            setup = FixedCanvasSetup()
        else:
            setup = MiniMaxH3Ref2VASetupStep()
        setup(components, state)
        if limits.get("independent_duration", False):
            state.set("num_frames", target_frames)
        if gears:
            from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference

            normalized = list(state.get("normalized_references"))
            area = limits["image"].get("target_area", size[0] * size[1])
            for index, entry in enumerate(entries):
                if entry.kind == "image":
                    image_value = _reference_image(entry.image)
                    height, width = image_size(image_value.height, image_value.width, gears, area)
                    image_value = image_value.resize((width, height), Image.Resampling.LANCZOS)
                    normalized[index] = MiniMaxH3ImageReference(image=image_value)
            state.set("normalized_references", normalized)
    else:
        state.set("image", None if image is None else _local_image(image))
        state.set("last_image", None if last_image is None else _local_image(last_image))
        MiniMaxH3ResizeStep()(components, state)
    return PreparedRequest(inferred, state, components)


def tokenize_prompt(prompt, tokenizer, maximum):
    """Validate user text before media work, without truncation or chat templates."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Sol-H3 requires a nonempty prompt")
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not 0 < len(ids) <= maximum:
        raise ValueError(f"Sol-H3 prompt must contain between one and {maximum} tokens")
    return ids


def prepare_configured_request(prompt, *, config, tokenizer, image=None, last_image=None, references=None):
    """Apply an export profile's input contract to native media preparation.

    The caller supplies the validated export configuration saved with the bundle.
    Duration/count limits concern source media, before native truncation/resampling.
    Video soundtracks use the audio duration limit but not the standalone audio count.
    """
    tokenize_prompt(prompt, tokenizer, config.get("prompt", {}).get("max_tokens", config.get("text_capacity", 1024)))
    limits = config.get("references", {})
    return prepare_request(
        prompt,
        size=config["draft_size"],
        num_frames=config["draft_frames"],
        mode=config.get("mode", "t2va"),
        image=image,
        last_image=last_image,
        references=references,
        reference_limits=limits,
        reference_image_short_edge=limits.get("image", {}).get("vae_short_edge", 2048),
    )


def prepare_encoder_calibration(request, processor, config):
    """Collect real prepared media and derive the complete reference tile gears."""
    from .capacity import image_geometries, video_geometries
    from .reference_geometry import image_gears, video_gear

    images, videos, soundtracks = list(request.keyframes), [], []
    image_shapes, video_shapes = [tuple(config["draft_size"])], []
    if request.mode == "ref2va":
        images = [reference.image for reference in request.references if reference.kind == "image"]
        videos = [reference for reference in request.references if reference.kind == "video"]
        soundtracks = [reference.audio for reference in request.references if reference.has_audio]
        limits = config["references"]
        has_images = limits.get("image", {}).get("max_count", 0) > 0
        has_videos = limits.get("video", {}).get("max_count", 0) > 0
        has_audio = has_videos or limits.get("audio", {}).get("max_count", 0) > 0
        if has_videos and not videos:
            raise ValueError("reference video capacity requires a real calibration video")
        if has_audio and not soundtracks:
            raise ValueError("reference audio capacity requires a real calibration soundtrack")
        image_shapes = (
            (image_gears(limits["image"]) or image_geometries(limits["image"].get("vae_short_edge", 2048)))
            if has_images else []
        )
        video_shape = video_gear(limits.get("video", {})) if has_videos else None
        video_shapes = ([video_shape] if video_shape else video_geometries(config["draft_size"])) if has_videos else []
    visual_calibrations = []
    if images:
        vision = processor.image_processor(images=images, return_tensors="pt")
        visual_calibrations.append((vision["pixel_values"], vision["image_grid_thw"]))
        if request.mode == "ref2va" and image_gears(config["references"]["image"]):
            from PIL import ImageOps

            height, width = max(image_shapes, key=lambda shape: shape[0] * shape[1])
            boundary = ImageOps.fit(images[0], (width, height), method=Image.Resampling.LANCZOS)
            vision = processor.image_processor(images=[boundary], return_tensors="pt")
            visual_calibrations.append((vision["pixel_values"], vision["image_grid_thw"]))
    if videos:
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

        sampled, _ = MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames(
            videos[0].frames, 24, 2, processor.video_processor.temporal_patch_size
        )
        vision = processor.video_processor(videos=[np.stack(sampled)], do_sample_frames=False, return_tensors="pt")
        visual_calibrations.append((vision["pixel_values_videos"], vision["video_grid_thw"]))
    visual_pixels, grids = visual_calibrations[0]
    if videos:
        pixels = torch.from_numpy(videos[0].frames[:17].copy()).permute(3, 0, 1, 2)[None]
    else:
        pixels = torch.from_numpy(np.array(images[0])).permute(2, 0, 1)[None, :, None]
    mean, std = (
        torch.tensor(value).reshape(1, 3, 1, 1, 1)
        for value in (request.components.pixel_mean, request.components.pixel_std)
    )
    return dict(
        visual_pixels=visual_pixels,
        visual_calibrations=visual_calibrations,
        grids=grids,
        pixels=(pixels.float() / 255 - mean) / std,
        image_shapes=image_shapes,
        video_shapes=video_shapes,
        waveform=soundtracks[0][:, None] if soundtracks else None,
    )


def encode_request(request, *, text_encoder, tokenizer, processor, vae=None, audio_vae=None, text_encoder_layer=50):
    """Run native presentation/conditioning blocks with supplied encoder adapters.

    Adapters may own exported sessions or native modules. They expose the same
    small encode/forward protocol, so reference labeling and latent sampling
    remain owned by Diffusers rather than duplicated in the deployment runtime.
    """
    from diffusers.modular_pipelines.minimax_h3 import encoders

    components, state = request.components, request.state
    components.text_encoder, components.tokenizer, components.processor = text_encoder, tokenizer, processor
    components.vae, components.audio_vae = vae, audio_vae
    components.text_encoder_layer = text_encoder_layer
    if request.mode == "ref2va":
        if vae is None or audio_vae is None:
            raise ValueError("Ref2VA requires the reference video and audio VAE adapters")
        encoders.MiniMaxH3Ref2VATextEncoderStep()(components, state)
        encoders.MiniMaxH3Ref2VAReferenceEncoderStep()(components, state)
    else:
        encoders.MiniMaxH3FL2VATextEncoderStep()(components, state)
        if request.keyframes:
            if vae is None:
                raise ValueError("keyframes require the reference video VAE adapter")
            encoders.MiniMaxH3KeyframeVaeEncoderStep()(components, state)
        else:
            state.set("condition_latents", [])
        state.set("audio_condition_latents", [])
    return state.get("prompt_embeds"), state.get("text_token_tags")


def prepare_latents(request, *, seed, latent_channels=24):
    """Use native layout/noise/packing, including conditioning-first RNG order."""
    from diffusers.modular_pipelines.minimax_h3 import before_denoise
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Sol-H3 seed is outside the supported range")
    components, state = request.components, request.state
    components.scheduler = MiniMaxH3Scheduler()
    components.vae_spatial_compression_ratio = 16
    components.vae_latent_channels = latent_channels
    components.patch_size = (1, 2, 2)
    state.set("generator", torch.Generator(device="cpu").manual_seed(seed))
    layout = (
        before_denoise.MiniMaxH3Ref2VAPrepareLayoutStep
        if request.mode == "ref2va"
        else before_denoise.MiniMaxH3PrepareLayoutStep
    )
    layout()(components, state)
    if state.get("condition_latents"):
        before_denoise.MiniMaxH3PrepareConditionLatentsStep()(components, state)
    before_denoise.MiniMaxH3PrepareLatentsStep()(components, state)
    if request.mode == "ref2va":
        before_denoise.MiniMaxH3Ref2VAPrepareLatentsStep()(components, state)
    elif request.keyframes:
        before_denoise.MiniMaxH3FL2VAPrepareLatentsStep()(components, state)
    vi, ai, ti = (state.get(name) for name in ("video_indices", "audio_indices", "text_indices"))
    order = torch.cat((vi, ai, ti))
    roles = state.get("token_tags").clone()
    cv, ca = state.get("num_condition_video_rows"), state.get("num_condition_audio_rows")
    roles[vi[:cv]], roles[ai[:ca]] = 3, 4
    return (
        state.get("latents")[None],
        state.get("audio_latents")[None],
        roles[order].int(),
        state.get("position_ids")[order].float(),
        cv,
        ca,
    )
