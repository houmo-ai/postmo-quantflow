# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Five native request modes and fixed-capacity reference attention."""

import numpy as np
import pytest
import torch
from PIL import Image

from xhmodel_merak.xh_other_model.models.sol_h3.request import prepare_request


@pytest.fixture
def request_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "scene": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


def test_configured_request_checks_prompt_before_opening_media(request_tokenizer):
    from xhmodel_merak.xh_other_model.models.sol_h3.request import prepare_configured_request

    config = dict(mode="i2va", prompt=dict(max_tokens=2), draft_size=[32, 64], draft_frames=124)
    with pytest.raises(ValueError, match="between one and 2 tokens"):
        prepare_configured_request(
            "scene scene scene", config=config, tokenizer=request_tokenizer, image="/nonexistent/never-opened.png"
        )
    request = prepare_configured_request(
        "scene scene", config=config, tokenizer=request_tokenizer, image=Image.new("RGB", (32, 32))
    )
    assert request.anchors == ("first",)
    assert request.keyframes[0].size == (64, 32)


def test_configured_references_enforce_counts_and_use_vae_edge(request_tokenizer):
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference

    from xhmodel_merak.xh_other_model.models.sol_h3.request import prepare_configured_request

    config = dict(
        mode="ref2va",
        prompt=dict(max_tokens=2),
        draft_size=[32, 64],
        draft_frames=124,
        references=dict(image=dict(max_count=1, vae_short_edge=64)),
    )
    image = MiniMaxH3ImageReference(Image.new("RGB", (64, 32)))
    with pytest.raises(ValueError, match="image count exceeds configured maximum 1"):
        prepare_configured_request("scene", config=config, tokenizer=request_tokenizer, references=[image, image])
    request = prepare_configured_request("scene", config=config, tokenizer=request_tokenizer, references=[image])
    assert request.references[0].image.size == (128, 64)


@pytest.mark.parametrize("excess", [None, "video", "soundtrack"])
def test_configured_video_soundtrack_has_separate_duration_budget(request_tokenizer, excess):
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3VideoReference

    from xhmodel_merak.xh_other_model.models.sol_h3.request import prepare_configured_request

    config = dict(
        mode="ref2va",
        prompt=dict(max_tokens=2),
        draft_size=[32, 64],
        draft_frames=124,
        references=dict(
            video=dict(max_count=1, max_duration_seconds=2), audio=dict(max_count=0, max_duration_seconds=2)
        ),
    )
    reference = MiniMaxH3VideoReference(
        frames=np.zeros((49 if excess == "video" else 48, 32, 64, 3), np.uint8),
        fps=24,
        audio=torch.zeros(2, 64001 if excess == "soundtrack" else 64000),
        sample_rate=32000,
    )
    if excess:
        with pytest.raises(ValueError, match=f"{excess} duration exceeds configured maximum"):
            prepare_configured_request("scene", config=config, tokenizer=request_tokenizer, references=[reference])
    else:
        request = prepare_configured_request(
            "scene", config=config, tokenizer=request_tokenizer, references=[reference]
        )
        assert len(request.references) == 1 and request.references[0].has_audio


@pytest.mark.parametrize("with_video,samples,start_pts", [(False, 160000, 0), (True, 163200, 0), (True, 158976, 64000)])
def test_file_aac_tail_padding_preserves_duration_budget(tmp_path, with_video, samples, start_pts):
    from fractions import Fraction

    import av

    from xhmodel_merak.xh_other_model.models.sol_h3.request import _reference, _validate_durations

    path = tmp_path / ("reference.mp4" if with_video else "reference.m4a")
    with av.open(str(path), "w") as container:
        audio = container.add_stream("aac", rate=32000)
        audio.layout = "mono"
        if with_video:
            video = container.add_stream("mpeg4", rate=24)
            video.width = video.height = 16
            video.pix_fmt = "yuv420p"
            for i in range(120):
                frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i, np.uint8), format="rgb24")
                frame.pts = i
                frame.time_base = Fraction(1, 24)
                for packet in video.encode(frame):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
        waveform = np.sin(np.arange(samples, dtype=np.float32) * 0.1)[None] * 0.1
        frame = av.AudioFrame.from_ndarray(waveform, format="fltp", layout="mono")
        frame.sample_rate = 32000
        frame.pts = start_pts
        frame.time_base = Fraction(1, 32000)
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        # With a delayed stream, MP4 includes the 1024-sample priming frame in
        # its presentation. Duration is a length, never an absolute end PTS.
        valid_samples = samples + (1024 if start_pts else 0)
        assert bool(stream.start_time) == bool(start_pts)
        assert stream.duration * stream.time_base == Fraction(valid_samples, 32000)
        decoded = np.concatenate([frame.to_ndarray() for frame in container.decode(stream)], axis=-1)
        assert 0 < decoded.shape[-1] - valid_samples < 1024  # Actual AAC partial-frame overhang.
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3AudioReference, MiniMaxH3VideoReference

    # Native video loading decodes video then seeks for audio; preserve that
    # decoder history, which can differ from a fresh audio-only decode.
    native = (MiniMaxH3VideoReference if with_video else MiniMaxH3AudioReference).from_file(path)
    reference = _reference(dict(kind="video" if with_video else "audio", path=str(path)))
    assert reference.audio.shape[-1] == valid_samples
    torch.testing.assert_close(reference.audio, native.audio[..., :valid_samples], rtol=0, atol=0)
    limits = dict(video=dict(max_count=1, max_duration_seconds=5), audio=dict(max_count=1, max_duration_seconds=5))
    if valid_samples == 160000:
        _validate_durations([reference], limits)
    else:
        # A longer audio track is not truncated to the five-second video/config.
        with pytest.raises(ValueError, match="soundtrack duration exceeds configured maximum"):
            _validate_durations([reference], limits)


@pytest.mark.parametrize(
    "first,last,mode,anchors",
    [
        (False, False, "t2va", ()),
        (True, False, "i2va", ("first",)),
        (False, True, "l2va", ("last",)),
        (True, True, "fl2va", ("first", "last")),
    ],
)
def test_keyframe_modes_preserve_native_crop_and_anchors(first, last, mode, anchors):
    array = np.arange(35 * 53 * 3, dtype=np.uint8).reshape(35, 53, 3)
    image = Image.fromarray(array)
    request = prepare_request(
        "scene", size=(32, 64), image=image if first else None, last_image=image if last else None, mode=mode
    )
    assert request.mode == mode and request.anchors == anchors
    assert all(frame.size == (64, 32) for frame in request.keyframes)
    if request.keyframes:
        np.testing.assert_array_equal(request.keyframes[0], image.resize((64, 32), Image.Resampling.LANCZOS))
    if first and last:
        resized = image.resize((64, 42), Image.Resampling.LANCZOS)
        np.testing.assert_array_equal(request.keyframes[1], resized.crop((0, 5, 64, 37)))


def test_mixed_references_preserve_order_rates_and_native_resize():
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    video = np.stack([np.full((32, 64, 3), i, np.uint8) for i in range(48)])
    references = [
        MiniMaxH3AudioReference(torch.ones(1, 64000), 32000),
        MiniMaxH3ImageReference(Image.new("RGB", (64, 32))),
        MiniMaxH3VideoReference(video, fps=24),
    ]
    request = prepare_request("scene", size=(32, 64), references=references, reference_image_short_edge=64)
    assert request.mode == "ref2va"
    assert [ref.kind for ref in request.references] == ["audio", "image", "video"]
    assert request.references[0].audio.shape == (2, 64000)
    assert request.references[1].image.size == (128, 64)
    np.testing.assert_array_equal(request.references[2].frames, video)
    assert request.references[2].fps == 24


def test_reference_validation_rejects_ambiguous_or_unsupported_requests():
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3AudioReference, MiniMaxH3ImageReference

    image = Image.new("RGB", (32, 32))
    reference = MiniMaxH3ImageReference(image)
    for kwargs in (
        dict(mode="i2va"),
        dict(image=image, references=[reference]),
        dict(references=[]),
        dict(references=[reference] * 10),
        dict(references=[MiniMaxH3AudioReference(torch.ones(1, 64000), 32000)]),
        dict(references=[reference, MiniMaxH3AudioReference(torch.ones(1, 16), 32000)]),
    ):
        with pytest.raises(ValueError):
            prepare_request("scene", size=(32, 64), **kwargs)


@pytest.mark.parametrize("mode", ["t2va", "i2va", "l2va", "fl2va", "ref2va"])
def test_native_latent_preparation_preserves_rng_order_and_reference_prefixes(mode):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3AudioReference, MiniMaxH3ImageReference
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    from xhmodel_merak.xh_other_model.models.sol_h3.request import prepare_latents

    image = Image.new("RGB", (64, 32))
    kwargs = {}
    if mode in ("i2va", "fl2va"):
        kwargs["image"] = image
    if mode in ("l2va", "fl2va"):
        kwargs["last_image"] = image
    if mode == "ref2va":
        kwargs["references"] = [MiniMaxH3AudioReference(torch.zeros(2, 64000), 32000), MiniMaxH3ImageReference(image)]
    request = prepare_request("scene", size=(32, 64), mode=mode, reference_image_short_edge=32, **kwargs)
    count = 1 if mode == "ref2va" else len(request.keyframes)
    conditions = [torch.full((1, 24, 1, 2, 4), float(i + 1)) for i in range(count)]
    audio_conditions = [torch.arange(6 * 32).reshape(6, 32).float()] if mode == "ref2va" else []
    request.state.set("condition_latents", conditions)
    request.state.set("audio_condition_latents", audio_conditions)
    request.state.set("text_token_tags", torch.tensor([1, 0, 1]))
    video, audio, roles, positions, cv, ca = prepare_latents(request, seed=123)

    # Independently replay the released conditioning -> video -> audio draw order.
    generator = torch.Generator().manual_seed(123)
    scheduler = MiniMaxH3Scheduler()
    expected_conditions = [
        patchify_video_latents(
            scheduler.scale_noise(value, 0.999, torch.randn(value.shape, generator=generator)), (1, 2, 2)
        )
        for value in conditions
    ]
    expected_video = patchify_video_latents(torch.randn((1, 24, 37, 2, 4), generator=generator), (1, 2, 2))
    expected_audio = torch.randn((414, 32), generator=generator)
    torch.testing.assert_close(video[0], torch.cat([*expected_conditions, expected_video]), rtol=0, atol=0)
    torch.testing.assert_close(audio[0], torch.cat([*audio_conditions, expected_audio]), rtol=0, atol=0)
    assert (cv, ca) == (count * 2, 6 if mode == "ref2va" else 0)
    assert positions.shape == (video.shape[1] + audio.shape[1] + 3, 3)
    assert roles.tolist() == [3] * cv + [0] * 74 + [4] * ca + [2] * 414 + [1, 0, 1]
