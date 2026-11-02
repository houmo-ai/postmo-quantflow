# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native H3 source fixture and adapter path ownership."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image


@pytest.mark.parametrize("weight", ["../outside.safetensors", "/tmp/outside.safetensors", "linked.safetensors"])
def test_native_adapter_rejects_unfingerprinted_external_weights(tmp_path, weight):
    from xhmodel_merak.xh_other_model.models.sol_h3.native_assets import read_native_adapters

    root = tmp_path / "adapter"
    root.mkdir()
    external = tmp_path / "outside.safetensors"
    external.write_bytes(b"not-loaded")
    (root / "linked.safetensors").symlink_to(external)
    (root / "h3_adapter.json").write_text(
        json.dumps(
            dict(
                name="turbo8",
                branches=dict(base=weight),
                sampling=dict(steps=8, video_shift=6, audio_shift=3),
            )
        )
    )
    with pytest.raises(ValueError, match="relative|symlink"):
        read_native_adapters([root])


def native_source(root):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from test_qwen_image21_feature_encoder import tiny_config
    from test_sol_h3_native_lora import adapter_fixture
    from test_sol_h3_reference_vae import tiny_vae
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessorFast,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    from xhmodel_merak.xh_other_model.models.sol_h3.runtime import write_video

    source = root / "source"
    source.mkdir()
    h3 = source / "h3"
    adapter = source / "adapter"
    adapter.mkdir()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(917)
        Qwen3VLForConditionalGeneration(tiny_config()).eval().save_pretrained(h3 / "text_encoder")
        transformer = MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=128,
            hidden_size=256,
            num_layers=1,
            num_refiner_layers=1,
            ffn_dim=512,
            in_channels=24,
            audio_in_channels=32,
            patch_size=(1, 2, 2),
            text_dim=256,
            freq_dim=16,
            time_embed_hidden_dim=128,
            time_embed_dim=128,
            rope_freq_dim=16,
        ).eval()
        transformer.save_pretrained(h3 / "transformer")
        adapter_fixture(adapter / "base.safetensors", transformer)
        with torch.no_grad():
            transformer.context_embedder.weight.mul_(0.9)
        transformer.save_pretrained(h3 / "transformer_ref")
        adapter_fixture(adapter / "ref.safetensors", transformer)
        vae_config = dict(tiny_vae().config)
        vae_config.update(latent_channels=24, latents_mean=[0.0] * 24, latents_std=[1.0] * 24)
        vae = AutoencoderKLMiniMaxH3.from_config(vae_config).eval()
        with torch.no_grad():
            vae.decoder.transformer_blocks[0].scale1.fill_(0.3)
            vae.decoder.transformer_blocks[0].scale2.fill_(0.3)
        vae.save_pretrained(h3 / "vae")
        AutoencoderKLMiniMaxH3Audio(
            encoder_dim=2,
            latent_dim=32,
            latent_channels=32,
            num_attention_heads=2,
            decoder_dim=32,
            decoder_rates=(20, 10, 4),
            decoder_kernel_sizes=(40, 20, 8),
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1, 3, 5),),
            latents_mean=[0.0] * 32,
            latents_std=[1.0] * 32,
        ).eval().save_pretrained(h3 / "audio_vae")
    (adapter / "h3_adapter.json").write_text(
        json.dumps(
            dict(
                name="turbo8",
                branches=dict(base="base.safetensors", ref="ref.safetensors"),
                sampling=dict(steps=8, video_shift=6.0, audio_shift=3.0),
            )
        )
    )
    (source / "checkpoint-paths.json").write_text(json.dumps(dict(h3_model="h3")))
    (source / "config.json").write_bytes((h3 / "text_encoder/config.json").read_bytes())
    vocab = {f"token{i}": i for i in range(64)}
    special = {
        "[UNK]": 0,
        "[PAD]": 1,
        "hello": 2,
        "world": 3,
        "<|image_pad|>": 60,
        "<|video_pad|>": 61,
        "<|vision_start|>": 62,
        "<|vision_end|>": 63,
    }
    for token, index in special.items():
        del vocab[f"token{index}"]
        vocab[token] = index
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=[name for name in special if name.startswith("<|")],
    )
    tokenizer.save_pretrained(h3 / "tokenizer")
    processor_args = dict(
        patch_size=16, temporal_patch_size=2, merge_size=2, size={"shortest_edge": 1024, "longest_edge": 1024}
    )
    Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessorFast(**processor_args),
        video_processor=Qwen3VLVideoProcessor(**processor_args),
    ).save_pretrained(h3 / "processor")
    media = {}
    for key, color in (("image", (50, 150, 190)), ("last_image", (190, 100, 50))):
        path = root / f"{key}.png"
        Image.new("RGB", (32, 32), color).save(path)
        media[key] = str(path)
    video_path = root / "reference.mp4"
    audio_path = root / "reference.wav"
    samples = (np.sin(np.arange(64000) * (2 * np.pi * 220 / 32000)) * 3000).astype("<i2")
    pixels = np.stack([np.full((32, 32, 3), 50 + index, np.uint8) for index in range(48)])
    # The embedded soundtrack and standalone WAV share the audio budget;
    # a video soundtrack must not count as a second standalone audio input.
    write_video(video_path, pixels, torch.from_numpy(np.stack((samples, samples))).float() / 32768)
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(32000)
        output.writeframes(np.stack((samples, samples), -1).tobytes())
    inputs = dict(
        t2va={},
        i2va=dict(image=media["image"]),
        l2va=dict(last_image=media["last_image"]),
        fl2va=media,
        ref2va=dict(
            references=[
                dict(kind="image", path=media["image"]),
                dict(kind="video", path=str(video_path)),
                dict(kind="audio", path=str(audio_path)),
            ]
        ),
    )
    (source / "calibration_inputs.json").write_text(json.dumps(inputs))
    config = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "configs_merak/workflows/xh2a/other_models/h3/h3_native_768p_w4.yaml"
        ).read_text()
    )["export"]
    config.update(
        profile="all-native-32p-5s",
        draft_size=[32, 32],
        prompt=dict(max_tokens=8),
        text_layers=2,
        text_layers_per_graph=1,
        conditioner_quant_type="w8a8_sefp",
        qwen3_vl=dict(prefill_chunk_length=4, image_max_pixels=1024, video_max_pixels=1024),
        references=dict(
            image=dict(max_count=1, vae_short_edge=32),
            video=dict(max_count=1, max_duration_seconds=2),
            audio=dict(max_count=1, max_duration_seconds=2),
        ),
        video_tile=[32, 32],
        video_overlap=[0, 0],
        calibration_prompt="hello world",
        native_sampling=dict(steps=2, video_shift=12.0, audio_shift=3.0),
    )
    config["model"]["lora"] = dict(path=[str(adapter)])
    return source, config, inputs
