# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Reference audio uses one static length-aware graph without changing short tails."""


import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio

from xhmodel_merak.xh_other_model.models.sol_h3.reference_audio import ReferenceAudioAdapter, ReferenceAudioGraph


def tiny_audio():
    return AutoencoderKLMiniMaxH3Audio(
        encoder_dim=2,
        latent_dim=32,
        latent_channels=32,
        num_attention_heads=2,
        decoder_dim=32,
        decoder_rates=(20, 10, 4),
        decoder_kernel_sizes=(40, 20, 8),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3, 5),),
        latents_mean=(0.0,) * 32,
        latents_std=(1.0,) * 32,
    ).eval()


@torch.no_grad()
def test_reference_audio_load_export_and_source_free_variable_length_execution(tmp_path):
    import shutil

    from xhmodel_merak.xh_other_model.models.sol_h3.export import export_graph
    from xhmodel_merak.xh_other_model.models.sol_h3.graph_audit import audit_graph
    from xhmodel_merak.xh_other_model.models.sol_h3.reference_audio import load_reference_audio
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.manual_seed(52)
    native = tiny_audio()
    native.save_pretrained(tmp_path / "audio_vae")
    source, config = load_reference_audio(tmp_path)
    assert not any(key.startswith(("decoder.", "dec_in_proj.", "logs_proj.")) for key in source.state_dict())
    graph = ReferenceAudioGraph(source, max_frames=5)
    waveform = torch.randn(2, 1, 4000)
    expected = native.encode(waveform).latent_dist.mode()
    torch.testing.assert_close(graph(waveform, torch.ones(1, 1, 5)), expected, rtol=3e-5, atol=3e-6)
    path = export_graph(
        graph,
        (waveform, torch.ones(1, 1, 5)),
        tmp_path / "encode.onnx",
        ["waveform", "frame_mask"],
        ["mean"],
        quant_scheme=dict(target_device="XH2a", quant_type="w8a8_sefp"),
        dynamo=True,
    )
    audit = audit_graph(path, quant_type="w8a8_sefp", inputs=["waveform", "frame_mask"], outputs=["mean"])
    assert audit["output_shapes"]["mean"] == [2, 32, 5]
    shutil.rmtree(tmp_path / "audio_vae")
    session = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))

    def invoke(pixels, frames):
        result = session.run(
            {"waveform": pixels.to("cuda:0", torch.float16), "frame_mask": frames.to("cuda:0", torch.float16)}
        )
        return (result[0] if isinstance(result, (tuple, list)) else result).cpu()

    adapter = ReferenceAudioAdapter(config, invoke, max_frames=5)
    for count in (799, 801, 4000):
        expected = native.encode(waveform[:, :, :count]).latent_dist.mode()
        actual = adapter.encode(waveform[:, :, :count]).latent_dist.mode()
        assert actual.shape == expected.shape and torch.isfinite(actual).all()
        assert (actual - expected).norm() / expected.norm() < 0.15
    with pytest.raises(ValueError, match="frame capacity"):
        adapter.encode(torch.zeros(2, 1, 4001))
