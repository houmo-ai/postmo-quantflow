# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Real native ViT decoder, partial RoPE, and overlapping temporal/spatial tiles."""

import pytest
import torch

from xhmodel_merak.xh_other_model.models.sol_h3.native_video import (
    NativeVideoDecodeAdapter,
    NativeVideoDecodeGraph,
    load_native_video_vae,
)


def _source(head_dim=8):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    torch.manual_seed(19)
    source = AutoencoderKLMiniMaxH3(
        latent_channels=2,
        block_out_channels=(4, 4, 4),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2, 1),
        temporal_downsample_factors=(1, 2, 2),
        norm_num_groups=1,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=head_dim,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=2,
        latents_mean=(0.2, -0.3),
        latents_std=(0.8, 1.3),
    ).eval()
    # Native initializes residual gains to zero; exercise the actual attention/FFN.
    with torch.no_grad():
        source.decoder.transformer_blocks[0].scale1.fill_(0.4)
        source.decoder.transformer_blocks[0].scale2.fill_(0.3)
        source.decoder.register_tokens.normal_(std=0.2)
    source.enable_tiling(12, 12, 4, 4)
    return source


@torch.no_grad()
def test_native_decoder_tile_matches_native_partial_rotary():
    shape = (9, 2, 3)
    source = _source()
    latent = torch.randn(1, 2, *shape)
    expected = source.decoder(source.post_quant_conv(latent))
    actual = NativeVideoDecodeGraph(source, shape)(latent)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("frames", [7, 12, 13])
@torch.no_grad()
def test_native_host_chunk_and_spatial_stitch_match_original(frames):
    source = _source()
    latent = torch.randn(1, 2, frames, 5, 4)
    seen, graphs = set(), {}

    def invoke(shape, value):
        seen.add(shape)
        if shape not in graphs:
            graphs[shape] = NativeVideoDecodeGraph(source, shape)
        return graphs[shape](value)

    adapter = NativeVideoDecodeAdapter(
        dict(source.config), invoke, tile_size=(12, 12), overlap=(4, 4), dtype=torch.float32
    )
    planned = adapter.tile_shapes([(frames, 5, 4)])
    actual = adapter.decode(latent, return_dict=False)[0]
    expected = source.decode(latent, return_dict=False)[0]
    assert set(planned) == seen
    assert all(shape[1:] == (3, 3) for shape in planned)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    mean = latent.new_tensor(source.config.latents_mean).view(1, 2, 1, 1, 1)
    std = latent.new_tensor(source.config.latents_std).view(1, 2, 1, 1, 1)
    pixels = adapter((latent - mean) / std)
    expected = (
        expected * expected.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        + expected.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
    ).clamp(0, 1)
    torch.testing.assert_close(pixels, expected, atol=2e-6, rtol=3e-5)
    assert pixels.dtype == torch.float32


@torch.no_grad()
def test_decoder_only_loader_materializes_nonpersistent_rope(tmp_path):
    source = _source()
    source.save_pretrained(tmp_path / "vae")
    loaded, config = load_native_video_vae(tmp_path)
    assert isinstance(loaded.encoder, torch.nn.Identity)
    assert not any(value.is_meta for value in loaded.parameters())
    assert not loaded.decoder.rope.inv_freq.is_meta
    latent = torch.randn(1, 2, 7, 3, 3)
    torch.testing.assert_close(
        loaded.decoder(loaded.post_quant_conv(latent)), source.decoder(source.post_quant_conv(latent))
    )


def test_native_decoder_sdk_export_uses_short_kv_matmul_attention(tmp_path):
    import onnx

    from xhmodel_merak.xh_other_model.models.sol_h3.export import export_graph

    source = _source(head_dim=64)
    graph = NativeVideoDecodeGraph(source, (7, 2, 2), sdk_ops=True).eval()
    latent = torch.randn(1, 2, 7, 2, 2)
    path = export_graph(
        graph,
        (latent,),
        tmp_path / "native_decode.onnx",
        ["latent"],
        ["pixels"],
        quant_scheme=dict(target_device="XH2a", quant_type="w8a8_sefp"),
        dynamo=True,
    )
    nodes = onnx.load(str(path), load_external_data=False).graph.node
    attention = [node for node in nodes if node.op_type == "FlashAttention"]
    assert not attention
    assert sum(node.op_type == "MatMul" for node in nodes) == 2
    assert sum(node.op_type == "Softmax" for node in nodes) == 1
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    runner = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
    with torch.no_grad():
        expected = source.decoder(source.post_quant_conv(latent))
        result = runner.run({"latent": latent.half().cuda()})
        result = result[0] if isinstance(result, (tuple, list)) else result
        result = torch.as_tensor(result).float().cpu()
        assert torch.isfinite(result).all()
        assert (result - expected).norm() / expected.norm() < 0.08
