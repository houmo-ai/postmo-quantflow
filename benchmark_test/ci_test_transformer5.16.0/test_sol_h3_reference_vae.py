# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native tiled/chunked reference posterior and static image/video encoder graphs."""

import copy

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

from xhmodel_merak.xh_other_model.models.sol_h3.h3_vae_wrappers import (
    MiniMaxH3VAEEncoderTileWrapper,
    wrap_vae_encoder,
)
from xhmodel_merak.xh_other_model.models.sol_h3.reference_vae import ReferenceVAEAdapter, load_reference_vae


def tiny_vae():
    return AutoencoderKLMiniMaxH3(
        latent_channels=4,
        block_out_channels=(32,) * 6,
        layers_per_block=1,
        decoder_num_layers=1,
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=64,
        latents_mean=(0.0,) * 4,
        latents_std=(1.0,) * 4,
    ).eval()


@torch.no_grad()
@pytest.mark.parametrize("frames", [1, 22])
def test_reference_adapter_preserves_native_tiling_temporal_tail_and_posterior(frames):
    native = tiny_vae()
    native.enable_tiling(
        tile_sample_min_height=32,
        tile_sample_min_width=32,
        tile_sample_min_overlap_height=16,
        tile_sample_min_overlap_width=16,
    )
    wrapped = wrap_vae_encoder(copy.deepcopy(native))
    tile = MiniMaxH3VAEEncoderTileWrapper(wrapped.encoder, wrapped.quant_conv)
    observed = set()

    def invoke(shape, value):
        observed.add(shape)
        return tile(value)

    adapter = ReferenceVAEAdapter(dict(native.config), invoke, tile_size=(32, 32), overlap=(16, 16))
    pixels = torch.randn(1, 3, frames, 48, 48)
    expected, actual = native.encode(pixels).latent_dist, adapter.encode(pixels).latent_dist
    torch.testing.assert_close(actual.mean, expected.mean, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual.logvar, expected.logvar, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(
        actual.sample(generator=torch.Generator().manual_seed(42)),
        expected.sample(generator=torch.Generator().manual_seed(42)),
        rtol=3e-5,
        atol=3e-6,
    )
    assert actual.mean.shape == (1, 4, 1 if frames == 1 else 7, 3, 3)
    assert observed == set(adapter.tile_shapes([(48, 48)], video=frames > 1))
    assert observed == {(1 if frames == 1 else 17, 32, 32)}
    assert list(adapter.parameters()) == []


@torch.no_grad()
def test_plain_reference_adapter_encodes_full_frame_larger_than_ref_tile():
    native = tiny_vae()
    wrapped = wrap_vae_encoder(copy.deepcopy(native))
    tile = MiniMaxH3VAEEncoderTileWrapper(wrapped.encoder, wrapped.quant_conv)
    observed = []

    def invoke(shape, pixels):
        observed.append(shape)
        return tile(pixels)

    adapter = ReferenceVAEAdapter(dict(native.config), invoke, tile_size=(32, 32), overlap=(16, 16), untiled=True)
    pixels = torch.randn(1, 3, 1, 48, 64)
    expected = native.encode(pixels).latent_dist.mean
    actual = adapter.encode(pixels).latent_dist.mean
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert observed == [(1, 48, 64)]


@torch.no_grad()
def test_reference_encoder_loads_without_decoder_and_exports_static_tiles(tmp_path):
    frames = 1
    from xhmodel_merak.xh_other_model.models.sol_h3.export import export_graph
    from xhmodel_merak.xh_other_model.models.sol_h3.graph_audit import audit_graph

    native = tiny_vae()
    native.save_pretrained(tmp_path / "vae")
    source, _ = load_reference_vae(tmp_path)
    assert not any(name.startswith(("decoder.", "post_quant_conv.")) for name in source.state_dict())
    source = wrap_vae_encoder(source)
    graph = MiniMaxH3VAEEncoderTileWrapper(source.encoder, source.quant_conv)
    pixels = torch.randn(1, 3, frames, 32, 32)
    expected = native.quant_conv(native.encoder(pixels))
    torch.testing.assert_close(graph(pixels), expected, rtol=3e-5, atol=3e-6)
    path = export_graph(
        graph,
        (pixels,),
        tmp_path / "encoder.onnx",
        ["pixels"],
        ["moments"],
        quant_scheme=dict(target_device="XH2a", quant_type="w8a8_sefp"),
        dynamo=True,
    )
    audit = audit_graph(path, quant_type="w8a8_sefp", inputs=["pixels"], outputs=["moments"])
    assert audit["output_shapes"]["moments"] == [1, 8, 1 if frames == 1 else 5, 2, 2]
    if frames == 1:
        import onnx

        exported = onnx.load(str(path), load_external_data=False)
        assert not any(node.op_type == "Gather" for node in exported.graph.node)
        pads = [node for node in exported.graph.node if node.op_type == "Pad"]
        assert pads
        # Reflect spatial boundaries remain; causal all-zero history is gone.
        assert all(
            next(onnx.helper.get_attribute_value(a) for a in node.attribute if a.name == "mode") == b"reflect"
            for node in pads
        )


@torch.no_grad()
def test_masked_reference_encoder_preserves_native_reflect_groupnorm_and_causal_tail(tmp_path):
    from xhmodel_merak.xh_other_model.models.sol_h3.conv_graph import prepare_static_shapes
    from xhmodel_merak.xh_other_model.models.sol_h3.export import export_graph
    from xhmodel_merak.xh_other_model.models.sol_h3.masked_vae import (
        MaskedReferenceEncodeGraph,
        reference_mask_inputs,
        reference_mask_names,
    )
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.manual_seed(37)
    native = tiny_vae()
    graph = MaskedReferenceEncodeGraph(copy.deepcopy(native))
    pixels = torch.randn(1, 3, 17, 48, 48)
    args = reference_mask_inputs(pixels, (17, 48, 48), graph.levels, graph.spatial_padding_mode)
    lowered = prepare_static_shapes(copy.deepcopy(graph), args)

    names = reference_mask_names(graph.levels, graph.spatial_padding_mode)
    path = export_graph(
        graph,
        args,
        tmp_path / "masked.onnx",
        names,
        ["moments"],
        quant_scheme=dict(target_device="XH2a", quant_type="w8a8_sefp"),
        dynamo=True,
    )
    import onnx

    exported = onnx.load(str(path), load_external_data=False)
    assert any(node.op_type == "Gather" for node in exported.graph.node)
    assert any(node.op_type == "Add" for node in exported.graph.node)
    runner = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
    for frames, h, w in ((1, 32, 32), (1, 32, 48), (17, 48, 32), (17, 48, 48)):
        sample = pixels[:, :, :frames, :h, :w]
        expected = native.quant_conv(native.encoder(sample))
        args = reference_mask_inputs(sample, (17, 48, 48), graph.levels, graph.spatial_padding_mode)
        t = expected.shape[2]
        eager = graph(*args)[:, :, :t, : h // 16, : w // 16]
        # Masked reductions run over the padded canvas, unlike native GroupNorm.
        # FP32 cancellation at near-zero outputs reached 1.8e-5 before any
        # Conv3d lowering. Check the global error as well as each coordinate.
        assert (eager - expected).norm() / expected.norm() < 5e-5
        torch.testing.assert_close(eager, expected, rtol=2e-4, atol=3e-5)
        actual = lowered(*args)[:, :, :t, : h // 16, : w // 16]
        # Static norm preparation changes FP32 reduction order; the small
        # 2x2 final GroupNorm amplifies its near-zero coordinate differences.
        assert (actual - expected).norm() / expected.norm() < 5e-5
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)
        inputs = {
            name: value.to(device="cuda:0", dtype=runner.get_input(name).dtype)
            for name, value in zip(names, args, strict=True)
        }
        result = runner.run(inputs)
        result = result[0] if isinstance(result, (tuple, list)) else result
        result = result.float().cpu()[:, :, :t, : h // 16, : w // 16]
        assert torch.isfinite(result).all()
        assert (result - expected).norm() / expected.norm() < 0.2
