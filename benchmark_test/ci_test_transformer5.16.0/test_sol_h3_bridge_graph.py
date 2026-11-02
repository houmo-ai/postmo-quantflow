# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from xhmodel_merak.xh_other_model.models.sol_h3.bridge_graph import TemporalAlignment, upscaler_segments
from xhmodel_merak.xh_other_model.models.sol_h3.conv_graph import prepare_static_shapes
from xhmodel_merak.xh_other_model.models.sol_h3.duration import duration_gear
from xhmodel_merak.xh_other_model.models.sol_h3.host import audio_for_video, initial_h3, unpatch_h3


def test_fake_shape_discovery_handles_conv_after_a_norm_and_activation():
    module = torch.nn.Sequential(
        torch.nn.Conv3d(4, 4, 3, padding=1),
        torch.nn.GroupNorm(2, 4),
        torch.nn.SiLU(),
        torch.nn.Conv3d(4, 4, 3, padding=1),
    )
    value = torch.randn(1, 4, 5, 6, 8)
    expected = module(value)
    prepare_static_shapes(module, (value,))
    assert sum(isinstance(layer, torch.nn.Conv3d) for layer in module.modules()) == 2
    torch.testing.assert_close(module(value), expected, atol=5e-7, rtol=2e-5)


@pytest.mark.parametrize(
    "frames,windows",
    [(37, [(0, 42), (22, 42)]), (107, [(0, 42), (22, 74), (54, 106), (86, 112)])],
)
def test_upscaler_chunks_cover_every_frame_and_overlap_the_author_windows(frames, windows):
    plans = upscaler_segments(frames)
    assert [(p["input_start"], p["input_end"]) for p in plans] == windows
    total = torch.zeros(frames)
    for plan in plans:
        total[plan["output_start"] : plan["output_end"]] += torch.tensor(plan["weights"])
    assert bool((total > 0).all())
    assert plans[1]["crop_start"] == 10


def test_upscaler_lowering_choice_is_scoped_to_both_segment_exports(monkeypatch, tmp_path):
    from _sol_h3_tiny_components import bridge_sources

    from xhmodel_merak.xh_other_model.models.sol_h3 import source_io
    from xhmodel_merak.xh_other_model.models.sol_h3.build import BundleExporter, SourceAssets, build_bundle
    from xhquant.frontend import convert_impl
    from xhquant.frontend.fx_transform import conv3d_to_conv2d

    def reject_source_access(*_):
        raise AssertionError("unsupported Conv3d options reached asset or processor loading")

    with monkeypatch.context() as patch:
        patch.setattr(SourceAssets, "read", reject_source_access)
        with pytest.raises(ValueError, match="upscaler_conv3d_fastpath must be a boolean"):
            build_bundle(tmp_path, tmp_path / "output", {"upscaler_conv3d_fastpath": None})

    # Both halves of the SDK configuration path must exist before any media or
    # source is read, including the unified-package dispatch.
    for transformer, builder, lowering in (
        (lambda: None, lambda enable_fuse=True, quant_config=None: None, "gather"),
        (lambda mode="slice", enable_fastpath=False: None, lambda enable_fuse=True: None, "gather"),
        (lambda mode="slice": None, lambda enable_fuse=True, quant_config=None: None, "slice"),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(conv3d_to_conv2d, "Conv3dToConv2d", transformer)
            patch.setattr(convert_impl, "build_fx_transformer", builder)
            patch.setattr(SourceAssets, "read", reject_source_access)
            for mode in ("i2va", "all"):
                with pytest.raises(ValueError, match="SDK with configurable Conv3dToConv2d"):
                    build_bundle(tmp_path, tmp_path / "output", {"mode": mode, "upscaler_conv3d_mode": lowering})
            assert not (tmp_path / "output").exists()

    upscaler, adapter = bridge_sources()
    monkeypatch.setattr(source_io, "upscaler", lambda *_: (upscaler, {}))
    monkeypatch.setattr(source_io, "adapter", lambda *_: adapter)
    runner = object.__new__(BundleExporter)
    runner.assets, runner.device = None, "cpu"
    runner.cfg = dict(
        draft_size=[32, 32], refiner_size=[64, 64], draft_frames=124, refiner_frames=121, fps=24,
        calibration_seed=42,
    )
    runner.progress = lambda *_: None
    runner.resource = lambda *_: None
    calls = []

    def record(name, _directory, _module, _values, _inputs, _outputs, **options):
        # SDK lowering/numerics have their own tests; this checks model dispatch.
        calls.append((name, options.get("conv3d_mode"), options.get("conv3d_fastpath")))

    runner.graph = record
    for mode, fastpath in (("slice", False), ("slice", True), ("gather", False)):
        runner.cfg["upscaler_conv3d_mode"] = mode
        runner.cfg["upscaler_conv3d_fastpath"] = fastpath
        runner.meta = {"upscaler_graphs": []}
        calls.clear()
        runner.bridge()
        assert calls == [
            ("h3_upscale_t20", mode, fastpath),
            ("h3_upscale_t42", mode, fastpath),
            ("h3_to_ltx", None, None),
        ]


@pytest.mark.parametrize(
    "draft,output,h3_video,h3_audio,ltx_video,ltx_audio,samples",
    [(124, 121, 37, 207, 16, 126, 161333), (362, 361, 107, 603, 46, 376, 481333)],
)
def test_alignment_preserves_nearest_slot_channel_order_and_empty_slots(
    draft, output, h3_video, h3_audio, ltx_video, ltx_audio, samples
):
    gear = duration_gear(draft, output)
    assert (gear.h3_video_frames, gear.h3_audio_frames, gear.ltx_video_frames, gear.ltx_audio_frames) == (
        h3_video, h3_audio, ltx_video, ltx_audio
    )
    alignment = TemporalAlignment(pixel_frames=draft)
    latent = torch.arange(24 * h3_video * 4 * 6, dtype=torch.float32).reshape(1, 24, h3_video, 4, 6)
    actual = alignment(latent)
    assert actual.shape == (1, 384, ltx_video + 1, 2, 3)
    # First target time's first nearest slot must contain the original frame zero.
    expected = torch.nn.functional.pixel_unshuffle(latent[:, :, 0], 2)
    torch.testing.assert_close(actual[:, 96:192, 0], expected)
    assert not bool(actual[:, 288:384, -1].any())
    pcm = torch.arange(samples + 2, dtype=torch.float32).view(1, -1)
    assert torch.equal(audio_for_video(pcm, frames=output), pcm[..., :samples])
    padded = audio_for_video(pcm[..., : samples - 2], frames=output)
    assert torch.equal(padded[..., : samples - 2], pcm[..., : samples - 2]) and not bool(padded[..., -2:].any())


def test_h3_seed_draw_order_and_patch_roundtrip():
    video, audio = initial_h3(42)
    generator = torch.Generator().manual_seed(42)
    expected_video = torch.randn(1, 24, 37, 24, 42, generator=generator)
    expected_audio = torch.randn(1, 414, 32, generator=generator)
    torch.testing.assert_close(unpatch_h3(video), expected_video, rtol=0, atol=0)
    torch.testing.assert_close(audio, expected_audio, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["upscaler", "adapter"])
@torch.no_grad()
def test_native_bridge_wrappers_preserve_native_outputs(kind):
    from _sol_h3_tiny_components import bridge_sources

    from xhmodel_merak.xh_other_model.models.sol_h3.bridge_graph import AdapterGraph, UpscalerSegment

    upscaler, adapter = bridge_sources()
    if kind == "upscaler":
        graph, value = UpscalerSegment(upscaler).eval(), torch.randn(1, 24, 5, 4, 6)
        torch.testing.assert_close(graph(value), upscaler(value), rtol=2e-5, atol=2e-6)
    else:
        graph, value = AdapterGraph(adapter, sdk_ops=True).eval(), torch.randn(1, 24, 37, 4, 6)
        torch.testing.assert_close(graph(value), adapter(TemporalAlignment()(value))[:, :, :16], rtol=0, atol=0)
