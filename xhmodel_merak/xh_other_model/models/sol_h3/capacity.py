# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Conservative static capacities derived from native media and tokenizer rules."""

from copy import deepcopy
from math import ceil, isfinite, lcm, sqrt


MODES = ("t2va", "i2va", "l2va", "fl2va", "ref2va")


def mode_configs(config):
    """Resolve one unified profile into the existing native request contracts."""
    if config.get("mode") != "all":
        raise ValueError("unified capacity planning requires mode=all")
    result = {}
    for mode in MODES:
        value = deepcopy(config)
        value["mode"] = mode
        size = value["draft_size"]
        value["profile"] = ("" if mode == "t2va" else mode + "-") + f"dense-{size[0]}p-{size[0] * 2}p-5s"
        if "calibration_inputs" in value:
            inventory = value.pop("calibration_inputs")
            if not isinstance(inventory, dict) or any(key not in MODES for key in inventory):
                raise ValueError("unified calibration_inputs must be indexed by input mode")
            if mode in inventory:
                value["calibration_inputs"] = inventory[mode]
        if mode != "ref2va":
            value.pop("references", None)
        result[mode] = value
    return result


def plan_unified_capacities(config, **components):
    """Share encoder capacities and take independent maxima for each DiT branch.

    Native token counts remain per mode. Only the physical graph slots are
    promoted to the shared branch capacity, so a mode switch never changes ABI.
    """
    plans = {mode: plan_capacities(value, **components) for mode, value in mode_configs(config).items()}
    encoder = {
        field: max(plan[field] for plan in plans.values())
        for field in ("text_capacity", "kv_capacity", "qwen_visual_max_patches")
    }
    encoder["qwen_visual_patch_gears"] = sorted(
        {capacity for plan in plans.values() for capacity in plan["qwen_visual_patch_gears"]}
    )
    branches = {}
    for branch, modes in (("base", MODES[:-1]), ("ref", ("ref2va",))):
        slots = {
            field: max(plans[mode][field] for mode in modes)
            for field in ("text_capacity", "video_capacity", "audio_capacity")
        }
        slots["dit_capacity"] = sum(slots.values())
        branches[branch] = slots
        for mode in modes:
            plans[mode].update(slots)
            plans[mode]["kv_capacity"] = encoder["kv_capacity"]
            plans[mode]["qwen_visual_max_patches"] = encoder["qwen_visual_max_patches"]
            plans[mode]["qwen_visual_patch_gears"] = list(encoder["qwen_visual_patch_gears"])
    return dict(encoder=encoder, branches=branches, modes=plans)


def _integer(value, name, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum or 'unbounded'}]")
    return value


def _duration(value, name):
    if type(value) not in (int, float) or not isfinite(value) or not 2 <= value <= 15:
        raise ValueError(f"{name} must be between 2 and 15 seconds")
    return value


def configure_processor(processor, options):
    """Use the same processor settings for geometry planning and actual pixels."""
    result = deepcopy(processor)
    for name in ("image", "video"):
        component = getattr(result, f"{name}_processor")
        key = f"{name}_max_pixels"
        if key in options:
            maximum = _integer(options[key], key, minimum=component.size["shortest_edge"])
            component.size = type(component.size)(shortest_edge=component.size["shortest_edge"], longest_edge=maximum)
    return result


def image_geometries(short_edge):
    """All native 32-rounded images with a fixed short edge and aspect <= 4."""
    if short_edge % 32:
        raise ValueError("reference image short edge must be aligned to 32")
    return sorted(
        {
            shape
            for long in range(short_edge, short_edge * 4 + 1, 32)
            for shape in ((short_edge, long), (long, short_edge))
        }
    )


def video_geometries(size):
    """Native canvas rounding cells intersecting the aspect interval [1, 4].

    Before the area cap, H=S and W=S*r. After it, H=sqrt(A/r)
    and W=sqrt(A*r). Intersect each rounded cell with these curves;
    independent upper bounds would export hundreds of unreachable small canvases.
    Closed intervals conservatively retain ties at half-multiple boundaries.
    Portrait canvases are the transposes of the same cells.
    """
    short, area = min(size), size[0] * size[1]
    cap_ratio = area / short**2
    long_limit = ceil(min(4 * short, 2 * sqrt(area)) / 32) * 32
    shapes = set()
    for h in range(32, short + 1, 32):
        for w in range(h, long_limit + 1, 32):
            uncapped = h == short and max(1, (w - 16) / short) <= min(4, cap_ratio, (w + 16) / short)
            capped = max(1, cap_ratio, (w - 16) ** 2 / area, area / (h + 16) ** 2) <= min(
                4, (w + 16) ** 2 / area, area / (h - 16) ** 2
            )
            if uncapped or capped:
                shapes.update(((h, w), (w, h)))
    return sorted(shapes)


def plan_capacities(config, *, tokenizer, processor=None, max_position_embeddings):
    """Return upper bounds, not a claim that every maximum occurs simultaneously.

    Prompt tokens exclude native labels, delimiters and video timestamps. Audio
    waveforms do not enter Qwen; their labels do, and their VAE rows enter DiT.
    Video soundtracks reserve audio capacity in addition to audio-only references.
    Counts retain native geometry; exported slots round up to 256. The KV cache
    additionally rounds to a complete prefill chunk.
    """
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        audio_latent_num_frames,
        video_latent_num_frames,
    )

    mode = config.get("mode", "t2va")
    if mode not in MODES:
        raise ValueError(f"unsupported Sol-H3 mode: {mode}")
    prompt = _integer(
        config.get("prompt", {}).get("max_tokens", config.get("text_capacity", 1024)), "prompt.max_tokens"
    )
    options = config.get("qwen3_vl", {})
    chunk = _integer(options.get("prefill_chunk_length", 256), "prefill_chunk_length")
    size, frames = config["draft_size"], config["draft_frames"]
    if len(size) != 2 or any(type(n) is not int or n <= 0 or n % 32 for n in size):
        raise ValueError("draft_size must contain positive multiples of 32")
    frames = _integer(frames, "draft_frames")
    target_video = video_latent_num_frames(frames, 17, 5) * (size[0] // 32) * (size[1] // 32)
    target_audio = audio_latent_num_frames(frames) * 2
    refs = config.get("references", {})
    if mode != "ref2va" and refs:
        raise ValueError("references limits apply only to ref2va; keyframe count follows the mode")
    counts = {
        kind: _integer(refs.get(kind, {}).get("max_count", 0), f"{kind}.max_count", 0, limit)
        for kind, limit in (("image", 9), ("video", 3), ("audio", 3))
    }
    image_count = {"t2va": 0, "i2va": 1, "l2va": 1, "fl2va": 2}.get(mode, counts["image"])
    if mode == "ref2va" and counts["image"] + counts["video"] == 0:
        raise ValueError("ref2va must allow at least one image or video reference")
    if mode != "t2va" and processor is None:
        raise ValueError("multimodal capacity planning requires the native Qwen3-VL processor")
    processor = configure_processor(processor, options) if processor is not None else None

    def label(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    image_tokens = video_tokens = visual_patches = reference_video = reference_audio = overhead = 0
    image_patches = video_patches = 0
    if image_count:
        edge = _integer(refs.get("image", {}).get("vae_short_edge", 2048), "image.vae_short_edge")
        shapes = image_geometries(edge) if mode == "ref2va" else [tuple(size)]
        ip = processor.image_processor
        patches = max(ip.get_number_of_image_patches(h, w) for h, w in shapes)
        image_patches = visual_patches = patches
        image_tokens = patches // ip.merge_size**2
        overhead += sum(label(f"<Picture {i}>: ") + 2 for i in range(1, image_count + 1))
        reference_video += image_count * max(h * w // 1024 for h, w in shapes)
    if counts["video"]:
        import numpy as np
        from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

        seconds = _duration(refs["video"].get("max_duration_seconds", 5), "video.max_duration_seconds")
        frame_limit = min(frames, ceil(seconds * 24))
        shapes, vp = video_geometries(size), processor.video_processor
        timestamp_overhead = 0
        for num_frames in range(48, frame_limit + 1):
            sampled, timestamps = MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames(
                np.empty((num_frames, 1, 1, 3), dtype=np.uint8), 24, 2, vp.temporal_patch_size
            )
            # The processor's counting helper floor-divides T, while actual patchify
            # repeats the last frame to a whole temporal patch. Preserve resize's
            # original frame count, then correct only the temporal multiplicity.
            groups = len(sampled) // vp.temporal_patch_size
            per_block = max(vp.get_num_of_video_patches(len(sampled), h, w) // groups for h, w in shapes)
            video_patches = max(video_patches, per_block)
            visual_patches = max(visual_patches, per_block)
            video_tokens = max(video_tokens, per_block * len(timestamps) // vp.merge_size**2)
            timestamp_overhead = max(timestamp_overhead, sum(label(f"<{t:.1f} seconds>") + 2 for t in timestamps))
        overhead += sum(label(f"<Video {i}>: ") + timestamp_overhead for i in range(1, counts["video"] + 1))
        snapped = max(1, (frame_limit - 5) // 17) * 17 + 5
        reference_video += (
            counts["video"] * video_latent_num_frames(snapped, 17, 5) * max(h * w // 1024 for h, w in shapes)
        )
    if counts["video"] or counts["audio"]:
        audio_seconds = _duration(refs.get("audio", {}).get("max_duration_seconds", 5), "audio.max_duration_seconds")
        tracks = counts["audio"] + counts["video"]
        # Native preparation truncates every soundtrack to the target duration.
        # Reserve each separately; the request's aggregate 15s limit can only lower use.
        standalone = ceil(min(audio_seconds, frames / 24) * 40)
        reference_audio = 2 * tracks * standalone
        overhead += sum(label(f"<Audio {i}>: ") for i in range(1, tracks + 1))

    def aligned(value):
        return ceil(value / 256) * 256

    text = prompt + image_count * image_tokens + counts["video"] * video_tokens + overhead
    text_capacity = aligned(text)
    video_capacity = aligned(target_video + reference_video)
    audio_capacity = aligned(target_audio + reference_audio)
    cache_alignment = lcm(256, chunk)
    kv_capacity = ceil(text_capacity / cache_alignment) * cache_alignment
    if kv_capacity > max_position_embeddings:
        raise ValueError(f"derived Qwen context/cache {kv_capacity} exceeds native maximum {max_position_embeddings}")
    return dict(
        mode=mode,
        prompt_max_tokens=prompt,
        token_alignment=256,
        text_tokens=text,
        text_capacity=text_capacity,
        prefill_chunk_length=chunk,
        kv_capacity=kv_capacity,
        qwen_visual_max_patches=aligned(visual_patches),
        qwen_visual_patch_gears=sorted({aligned(n) for n in (image_patches, video_patches) if n}),
        image_qwen_tokens=image_tokens,
        video_qwen_tokens=video_tokens,
        presentation_overhead=overhead,
        target_video_tokens=target_video,
        target_audio_tokens=target_audio,
        reference_video_tokens=reference_video,
        reference_audio_tokens=reference_audio,
        video_capacity=video_capacity,
        audio_capacity=audio_capacity,
        dit_capacity=text_capacity + video_capacity + audio_capacity,
    )
