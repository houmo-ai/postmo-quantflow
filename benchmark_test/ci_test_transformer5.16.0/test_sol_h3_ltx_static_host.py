# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Fixed-guide Host against independent noise, Euler and image references."""

import builtins

import numpy as np
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
from xhmodel_merak.xh_other_model.models.sol_h3.runtime import SolH3Runtime, _resolve_refiner_options


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


def refiner_metadata(mode="ref2va", frames=121):
    frame_tokens = 2  # A complete 32x64 spatial grid, with production temporal gears.
    target = ((frames - 1) // 8 + 1) * frame_tokens
    names = [
        "video", "audio", "step", "vcos", "vsin", "acos", "asin", "vccos", "vcsin", "accos", "acsin",
        "first_role", "video_valid_length", "audio_mode",
    ]
    return {
        "schema_version": 9, "mode": mode, "refiner_frames": frames, "refiner_size": (32, 64),
        "ltx_rope_mode": "inputs", "resources": {"ltx_rope": "ltx_rope.npz"},
        "graphs": {"ltx_denoiser": {"input_names": names}},
        "ltx_reference_conditioning": {
            "enabled": True, "graph_mode": "static_maximum_guide", "first_frame_tokens": frame_tokens,
            "target_tokens": target, "video_tokens": target + frame_tokens,
        },
        "ltx_audio_policy": {
            "input_name": "audio_mode", "dtype": "int32", "shape": [1],
            "modes": {"joint": 0, "frozen": 1}, "default": "frozen",
        },
    }


def fake_refiner(tmp_path, meta):
    runtime = SolH3Runtime.__new__(SolH3Runtime)
    runtime.meta = runtime.bundle_meta = meta
    runtime.device = torch.device("cpu")
    runtime.close = lambda: None
    total = meta["ltx_reference_conditioning"]["video_tokens"]
    source = np.arange(total * 3, dtype=np.float32).reshape(1, 1, total, 3)
    arrays = {f"rope_{i}": source + i * 10000 for i in range(8)}
    rope_path = tmp_path / "rope.npz"
    np.savez(rope_path, **arrays)
    runtime.numpy_resource = lambda name: np.load(rope_path, allow_pickle=False)
    calls = []

    def invoke(name, video, audio, step, *args):
        assert name == "ltx_denoiser"
        assert step.tolist() == [len(calls)]
        rope = args[:8]
        first_role, valid, audio_mode = args[8:]
        calls.append((video.clone(), audio.clone(), [r.clone() for r in rope], first_role.clone(), valid.clone(),
                      audio_mode.clone()))
        velocity = torch.full_like(video, 2)
        velocity[:, -2:] = 10000  # The guide's velocity must never update its clean slot.
        return velocity, torch.ones_like(audio)

    runtime.invoke = invoke
    return runtime, calls, arrays


@pytest.mark.parametrize("frames,guide_frames,audio_mode", [
    (121, (4, 10), None), (361, (12, 30), None),
    (121, (4, 10), "frozen"),
])
def test_ref2va_midguide_tracks_clean_bridge_rope_and_releases_final_step(
    tmp_path, monkeypatch, frames, guide_frames, audio_mode
):
    from xhmodel_merak.xh_other_model.models.sol_h3 import ltx_keyframes

    meta = refiner_metadata(frames=frames)
    runtime, calls, original_rope = fake_refiner(tmp_path, meta)
    runtime.ltx_image_encode = lambda **kwargs: pytest.fail("midguide must not invoke the LTX image encoder")
    real_step = ltx_keyframes.static_keyframe_step
    target = meta["ltx_reference_conditioning"]["target_tokens"]

    def check_solver(state, *args):
        assert bool((state.video_mask[:, :target] == 1).all())
        assert bool((state.video_mask[:, target:] == 0).all())
        before = state.video[:, target:].clone()
        result = real_step(state, *args)
        torch.testing.assert_close(result.video[:, target:], before, rtol=0, atol=0)
        return result

    monkeypatch.setattr(ltx_keyframes, "static_keyframe_step", check_solver)
    nframes = (frames - 1) // 8 + 1
    video = torch.arange(128 * nframes * 2, dtype=torch.float32).reshape(1, 128, nframes, 1, 2) / 10000
    audio = torch.ones(1, 8, round(frames / 24 * 25), 16)
    video_before, audio_before = video.clone(), audio.clone()
    output = runtime.refine(video, audio, 42, ltx_audio_mode=audio_mode)
    assert output.shape == video.shape and len(calls) == 3
    assert torch.equal(video, video_before) and torch.equal(audio, audio_before)
    clean = rearrange(video, "b c t h w -> b (t h w) c")
    for step, (model_video, model_audio, rope, role, valid, mode_input) in enumerate(calls):
        assert role.tolist() == [0]
        assert valid.tolist() == [target + (2 if step < 2 else 0)]
        assert mode_input.tolist() == [int(audio_mode == "frozen")]
        if step < 2:
            start = guide_frames[step] * 2
            torch.testing.assert_close(model_video[:, target:], clean[:, start:start + 2], rtol=0, atol=0)
            for i in (0, 1, 4, 5):
                np.testing.assert_array_equal(
                    rope[i][:, :, target:].numpy(), original_rope[f"rope_{i}"][:, :, start:start + 2]
                )
        else:
            assert torch.count_nonzero(model_video[:, target:]) == 0
            for i in (0, 1, 4, 5):
                np.testing.assert_array_equal(rope[i][:, :, target:].numpy(), original_rope[f"rope_{i}"][:, :, target:])
        for i in (0, 1, 4, 5):
            np.testing.assert_array_equal(rope[i][:, :, :target].numpy(), original_rope[f"rope_{i}"][:, :, :target])
        for i in (2, 3, 6, 7):
            np.testing.assert_array_equal(rope[i].numpy(), original_rope[f"rope_{i}"])
        if step:
            delta = REFINER_SIGMAS[step] - REFINER_SIGMAS[step - 1]
            torch.testing.assert_close(model_video[:, :target], calls[step - 1][0][:, :target] + 2 * delta,
                                       rtol=1e-5, atol=1e-6)
            expected_audio = calls[step - 1][1] + (0 if audio_mode == "frozen" else delta)
            torch.testing.assert_close(model_audio, expected_audio, rtol=1e-5, atol=1e-6)
    with runtime.numpy_resource("ltx_rope") as saved:
        for name, array in original_rope.items():
            np.testing.assert_array_equal(saved[name], array)


@pytest.mark.parametrize("mode,endpoint_indices", [
    ("t2va", ()), ("i2va", (0,)), ("l2va", (120,)), ("fl2va", (0, 120)), ("ref2va", ()),
])
def test_other_modes_and_explicit_none_keep_endpoint_guide_behavior(tmp_path, mode, endpoint_indices):
    runtime, calls, original_rope = fake_refiner(tmp_path, refiner_metadata(mode))
    video, audio, images = inputs()
    runtime.ltx_image_encode = lambda **kwargs: [(index, images[index]) for index in endpoint_indices]
    runtime.refine(video, audio, 42, ltx_guide_policy="none" if mode == "ref2va" else None)
    assert len(calls) == 3
    for model_video, model_audio, rope, role, valid, audio_mode in calls:
        assert role.tolist() == [int(0 in endpoint_indices)]
        assert valid.tolist() == [34 if 120 in endpoint_indices else 32]
        assert audio_mode.tolist() == [0 if mode == "ref2va" else 1]
        if mode != "ref2va":
            torch.testing.assert_close(model_audio, calls[0][1], rtol=0, atol=0)
        if 0 in endpoint_indices:
            expected = rearrange(images[0].half().float(), "b c t h w -> b (t h w) c")
            torch.testing.assert_close(model_video[:, :2], expected, rtol=0, atol=0)
        if 120 in endpoint_indices:
            expected = rearrange(images[120].half().float(), "b c t h w -> b (t h w) c")
            torch.testing.assert_close(model_video[:, 32:], expected, rtol=0, atol=0)
        for i in (0, 1, 4, 5):
            np.testing.assert_array_equal(rope[i].numpy(), original_rope[f"rope_{i}"])


def test_midguide_rejects_unsupported_modes_gears_and_graphs():
    meta = refiner_metadata()
    runtime = SolH3Runtime.__new__(SolH3Runtime)
    runtime.meta = meta
    for field in ("image", "last_image"):
        with pytest.raises(ValueError, match="does not accept first/last keyframes"):
            runtime.refine(None, None, 42, **{field: object()})
    with pytest.raises(ValueError, match="Ref2VA"):
        _resolve_refiner_options(refiner_metadata("t2va"), "midguide", None)
    with pytest.raises(ValueError, match="5s or 15s"):
        _resolve_refiner_options(refiner_metadata(frames=201), None, None)
    for field, value in (("graph_mode", "target_only"), ("enabled", False)):
        bad = refiner_metadata()
        bad["ltx_reference_conditioning"][field] = value
        with pytest.raises(ValueError, match="static_maximum_guide"):
            _resolve_refiner_options(bad, None, None)
    for field, value in (("ltx_rope_mode", "constant"), ("resources", {}), ("graphs", {})):
        bad = {**meta, field: value}
        with pytest.raises(ValueError, match="input RoPE"):
            _resolve_refiner_options(bad, None, None)
    bad = refiner_metadata()
    bad["ltx_reference_conditioning"]["video_tokens"] -= 1
    with pytest.raises(ValueError, match="one full guide frame"):
        _resolve_refiner_options(bad, None, None)
    with pytest.raises(ValueError, match="ltx_guide_policy"):
        _resolve_refiner_options(meta, "other", None)


def test_ref2va_legacy_frozen_audio_requires_explicit_override():
    legacy = refiner_metadata()
    legacy.update(schema_version=8, ltx_audio_frozen=True)
    del legacy["ltx_audio_policy"]
    legacy["ltx_reference_conditioning"]["graph_mode"] = "target_only"
    with pytest.raises(ValueError, match="static_maximum_guide"):
        _resolve_refiner_options(legacy, None, None)
    with pytest.raises(ValueError, match="supports only LTX audio mode 'frozen'"):
        _resolve_refiner_options(legacy, "none", None)
    assert _resolve_refiner_options(legacy, "none", "frozen") == ("none", "frozen")


def test_legacy_target_only_refine_can_explicitly_return_to_original_path(tmp_path):
    meta = refiner_metadata()
    meta.update(schema_version=8, ltx_audio_frozen=True)
    del meta["ltx_audio_policy"]
    meta["ltx_reference_conditioning"]["video_tokens"] = meta["ltx_reference_conditioning"]["target_tokens"]
    runtime, _, _ = fake_refiner(tmp_path, meta)
    del meta["ltx_reference_conditioning"]  # Older metadata need not declare guide support.
    video, audio, _ = inputs()
    calls = []

    def invoke(name, packed_video, packed_audio, step, *rope):
        calls.append((packed_video.clone(), packed_audio.clone(), step.item()))
        assert name == "ltx_denoiser" and len(rope) == 8
        return torch.ones_like(packed_video), torch.ones_like(packed_audio)

    runtime.invoke = invoke
    runtime.ltx_image_encode = lambda **kwargs: pytest.fail("target-only graph cannot encode keyframes")
    result = runtime.refine(video, audio, 42, ltx_guide_policy="none", ltx_audio_mode="frozen")
    assert result.shape == video.shape and [call[2] for call in calls] == [0, 1, 2]
    for _, actual_audio, _ in calls:
        torch.testing.assert_close(actual_audio, calls[0][1], rtol=0, atol=0)


@pytest.mark.parametrize("media,audio_option,guide_option,expected_audio,expected_guide", [
    ({}, None, None, "frozen", "none"),
    ({"image": "first"}, None, None, "frozen", "none"),
    ({"last_image": "last"}, None, None, "frozen", "none"),
    ({"image": "first", "last_image": "last"}, None, None, "frozen", "none"),
    ({"references": ["reference"]}, None, None, "joint", "midguide"),
    ({"references": ["reference"]}, "frozen", "none", "frozen", "none"),
    ({}, "joint", None, "joint", "none"),
])
def test_generate_routes_refiner_options_after_input_mode_selection(
    tmp_path, monkeypatch, media, audio_option, guide_option, expected_audio, expected_guide
):
    from xhmodel_merak.xh_other_model.models.sol_h3 import bundle
    from xhmodel_merak.xh_other_model.models.sol_h3 import runtime as runtime_module

    meta = refiner_metadata("all")
    meta.update(quant_type="w8a8_sefp", refiner_cache_policy="per_sigma_exact_fixed_prompt")
    runtime = SolH3Runtime.__new__(SolH3Runtime)
    runtime.meta = runtime.bundle_meta = meta
    runtime.device = torch.device("cuda:0")
    runtime.close = lambda: None
    monkeypatch.setattr(bundle, "mode_metadata", lambda data, mode: {**data, "mode": mode})
    monkeypatch.setattr(runtime_module, "write_video", lambda *args: None)
    runtime.text = lambda prompt: (None, 1)
    runtime.condition = lambda *args, **kwargs: (None, 1, object())
    runtime.draft = lambda *args, **kwargs: (None, None)
    runtime.audio_decode = lambda audio: None
    runtime.bridge = lambda video: None
    runtime.audio_encode = lambda pcm: None
    runtime.video_decode = lambda video: np.zeros((1, 1, 1, 3), dtype=np.uint8)
    received = []
    runtime.refine = lambda *args, **kwargs: received.append(kwargs)
    result = runtime.generate(
        "prompt", output_path=tmp_path / "out.mp4", ltx_audio_mode=audio_option,
        ltx_guide_policy=guide_option, **media,
    )
    assert received[0]["ltx_audio_mode"] == result["stats"]["ltx_audio_mode"] == expected_audio
    assert received[0]["ltx_guide_policy"] == result["stats"]["ltx_guide_policy"] == expected_guide
