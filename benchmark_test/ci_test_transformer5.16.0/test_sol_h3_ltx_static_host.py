# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Fixed-guide Host against independent noise, Euler and image references."""

import builtins

import pytest
import torch
from einops import rearrange

from xhmodel_merak.xh_other_model.models.sol_h3.conditioning import REFINER_SIGMAS
from xhmodel_merak.xh_other_model.models.sol_h3.ltx_keyframes import (
    clear_static_keyframe_state,
    prepare_keyframe_pixels,
    prepare_static_keyframe_state,
    static_keyframe_step,
)


def inputs():
    generator = torch.Generator().manual_seed(915)
    return (
        torch.randn((1, 128, 16, 1, 2), generator=generator),
        torch.randn((1, 8, 126, 16), generator=generator),
        {index: torch.randn((1, 128, 1, 1, 2), generator=generator) for index in (0, 120)},
    )


@pytest.mark.parametrize("indices", [(), (0,), (120,), (0, 120)])
def test_static_initial_state_and_three_steps_match_independent_formulas(indices):
    video, audio, images = inputs()
    options = dict(keyframe_latents=[(i, images[i]) for i in indices], seed=42, size=(32, 64), device="cpu")
    state = prepare_static_keyframe_state(video, audio, **options)
    valid = 34 if 120 in indices else 32
    packed_video = rearrange(video, "b c t h w -> b (t h w) c").double()
    packed_audio = rearrange(audio, "b c t f -> b t (c f)").double()
    noise_generator = torch.Generator().manual_seed(42)
    video_noise = torch.randn(packed_video.shape, generator=noise_generator, dtype=torch.bfloat16).double()
    audio_noise = torch.randn(packed_audio.shape, generator=noise_generator, dtype=torch.bfloat16).double()
    sigma = REFINER_SIGMAS[0]
    expected_video = torch.zeros((1, 34, 128), dtype=torch.float64)
    expected_video[:, :32] = packed_video * (1 - sigma) + video_noise * sigma
    expected_audio = packed_audio * (1 - sigma) + audio_noise * sigma
    clean = torch.cat((packed_video.float(), torch.zeros((1, 2, 128))), dim=1)
    mask = torch.ones((1, 34, 1))
    mask[:, 32:] = 0
    for index in indices:
        start = 0 if index == 0 else 32
        image = rearrange(images[index].half().float(), "b c t h w -> b (t h w) c")
        clean[:, start : start + 2] = image
        expected_video[:, start : start + 2] = image.double()
        mask[:, start : start + 2] = 0
    assert state.video.shape == (1, 34, 128)
    assert state.video.dtype == state.audio.dtype == torch.float32
    assert state.first_role.dtype == state.video_valid_length.dtype == torch.int32
    assert state.first_role.tolist() == [int(0 in indices)]
    assert state.video_valid_length.tolist() == [valid]
    torch.testing.assert_close(state.video, expected_video.float(), rtol=1e-6, atol=5e-7)
    torch.testing.assert_close(state.audio, expected_audio.float(), rtol=1e-6, atol=5e-7)
    torch.testing.assert_close(state.clean_video, clean, rtol=0, atol=0)
    torch.testing.assert_close(state.video_mask, mask, rtol=0, atol=0)
    generator = torch.Generator().manual_seed(776)
    for sigma, next_sigma in zip(REFINER_SIGMAS[:-1], REFINER_SIGMAS[1:], strict=True):
        velocity_v = torch.randn(state.video.shape, generator=generator)
        velocity_a = torch.randn(state.audio.shape, generator=generator)
        state = static_keyframe_step(state, velocity_v, velocity_a, sigma, next_sigma)
        # For binary masks Euler integrates velocity only on generated rows.
        # Evaluate the closed form in FP64, independently of X0 roundtripping.
        expected_video += velocity_v.double() * mask.double() * (next_sigma - sigma)
        expected_audio += velocity_a.double() * (next_sigma - sigma)
        torch.testing.assert_close(state.video, expected_video.float(), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(state.audio, expected_audio.float(), rtol=2e-5, atol=2e-6)
        if 120 not in indices:
            assert torch.count_nonzero(state.video[:, 32:]) == 0
    expected = rearrange(expected_video[:, :32], "b (t h w) c -> b c t h w", t=16, h=1, w=2).float()
    torch.testing.assert_close(clear_static_keyframe_state(state), expected, rtol=2e-5, atol=2e-6)


def test_target_and_audio_noise_do_not_depend_on_endpoints_or_input_order():
    video, audio, images = inputs()
    plain = prepare_static_keyframe_state(video, audio, seed=79, size=(32, 64))
    ends = prepare_static_keyframe_state(
        video, audio, keyframe_latents=[(120, images[120]), (0, images[0])], seed=79, size=(32, 64)
    )
    assert torch.equal(plain.video[:, 2:32], ends.video[:, 2:32])
    assert torch.equal(plain.audio, ends.audio)
    # The target's last latent remains noisy/generated, separate from the guide.
    assert bool((ends.video_mask[:, 30:32] == 1).all())
    assert bool((ends.video_mask[:, 32:] == 0).all())
    assert torch.equal(ends.video[:, 32:], images[120].half().float().permute(0, 2, 3, 4, 1).reshape(1, 2, 128))


def test_runtime_static_path_never_imports_author_libraries(monkeypatch):
    original = builtins.__import__

    def source_free(name, *args, **kwargs):
        if name.startswith(("ltx_core", "ltx_pipelines")):
            raise AssertionError(f"Source-free Host imported author build dependency {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", source_free)
    video, audio, _ = inputs()
    state = prepare_static_keyframe_state(video, audio, seed=42, size=(32, 64))
    initial_target = state.video[:, :32].clone()
    initial_audio = state.audio.clone()
    for sigma, next_sigma in zip(REFINER_SIGMAS[:-1], REFINER_SIGMAS[1:], strict=True):
        velocity = torch.zeros_like(state.video)
        velocity[:, 32:] = 60000  # Ignored dummy-guide velocities must not enter target sampling.
        state = static_keyframe_step(state, velocity, torch.zeros_like(state.audio), sigma, next_sigma)
    assert torch.equal(state.video[:, :32], initial_target)
    assert torch.equal(state.audio, initial_audio)
    assert torch.count_nonzero(state.video[:, 32:]) == 0
    assert clear_static_keyframe_state(state).shape == video.shape


@pytest.mark.parametrize("indices", [(0, 0), (1,)])
def test_static_graph_accepts_only_unique_endpoint_conditions(indices):
    video, audio, images = inputs()
    with pytest.raises(ValueError, match="unique endpoint"):
        prepare_static_keyframe_state(
            video, audio, keyframe_latents=[(i, images[0]) for i in indices], seed=42, size=(32, 64)
        )


@pytest.mark.parametrize("sigma,next_sigma", [(0.0, 0.0), (0.4, 0.5)])
def test_static_solver_rejects_invalid_schedule(sigma, next_sigma):
    video, audio, _ = inputs()
    state = prepare_static_keyframe_state(video, audio, seed=42, size=(32, 64))
    with pytest.raises(ValueError, match="sigmas must decrease"):
        static_keyframe_step(state, torch.zeros_like(state.video), torch.zeros_like(state.audio), sigma, next_sigma)


@pytest.mark.parametrize("crf,mode", [(0, "RGB"), (33, "RGBA")])
def test_keyframe_pixels_preserve_sdr_colors_and_exif_rotation(tmp_path, crf, mode):
    import numpy as np
    from PIL import Image, ImageCms

    # Odd source dimensions exercise codec even-cropping, rotation and center crop.
    pixels = np.zeros((19, 29, 3), dtype=np.uint8)
    pixels[:, :14, 0] = 255
    pixels[:, 14:, 2] = 255
    image = Image.fromarray(pixels).convert(mode)
    exif = Image.Exif()
    exif[274] = 6
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = tmp_path / "condition.png"
    image.save(path, exif=exif, icc_profile=profile)
    actual = prepare_keyframe_pixels(path, (32, 64), crf)
    with Image.open(path) as opened:
        from_pil = prepare_keyframe_pixels(opened, (32, 64), crf)
    assert actual.dtype == torch.float32 and actual.device.type == "cpu"
    assert actual.shape == (1, 3, 1, 32, 64)
    torch.testing.assert_close(from_pil, actual, rtol=0, atol=0)
    # A clockwise quarter-turn moves the red left half above the blue half.
    # The uniform outer bands survive center-cropping and codec quantization.
    top = actual[:, :, :, :8].mean((0, 2, 3, 4))
    bottom = actual[:, :, :, -8:].mean((0, 2, 3, 4))
    if crf:
        # Lossy codec output is checked for color/orientation, not bit equality.
        assert top[0] > 0.8 and bool((top[1:] < -0.8).all())
        assert bottom[2] > 0.8 and bool((bottom[:2] < -0.8).all())
    else:
        torch.testing.assert_close(top, torch.tensor([1.0, -1.0, -1.0]), rtol=0, atol=1e-6)
        torch.testing.assert_close(bottom, torch.tensor([-1.0, -1.0, 1.0]), rtol=0, atol=1e-6)


@pytest.mark.parametrize("crf", [None])
def test_keyframe_pixels_require_checkpoint_crf(crf):
    with pytest.raises(ValueError, match="resolved integer"):
        prepare_keyframe_pixels("unused.png", (32, 64), crf)
