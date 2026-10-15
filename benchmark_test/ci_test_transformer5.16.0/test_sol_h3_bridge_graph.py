# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from xhmodel_merak.xh_other_model.models.sol_h3.bridge_graph import TemporalAlignment, upscaler_segments
from xhmodel_merak.xh_other_model.models.sol_h3.conv_graph import prepare_static_shapes
from xhmodel_merak.xh_other_model.models.sol_h3.host import initial_h3, unpatch_h3


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


def test_upscaler_chunks_cover_every_frame_and_overlap_the_author_windows():
    plans = upscaler_segments()
    assert [(p["input_start"], p["input_end"]) for p in plans] == [(0, 42), (22, 42)]
    total = torch.zeros(37)
    for plan in plans:
        total[plan["output_start"] : plan["output_end"]] += torch.tensor(plan["weights"])
    assert bool((total > 0).all())
    assert plans[1]["crop_start"] == 10


def test_alignment_preserves_nearest_slot_channel_order_and_empty_slots():
    alignment = TemporalAlignment()
    latent = torch.arange(24 * 37 * 4 * 6, dtype=torch.float32).reshape(1, 24, 37, 4, 6)
    actual = alignment(latent)
    assert actual.shape == (1, 384, 17, 2, 3)
    # First target time's first nearest slot must contain the original frame zero.
    expected = torch.nn.functional.pixel_unshuffle(latent[:, :, 0], 2)
    torch.testing.assert_close(actual[:, 96:192, 0], expected)
    assert not bool(actual[:, 288:384, -1].any())


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
