# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Constructed small checkpoints through real Sol-H3 graph export and runtime.

Native H3/Qwen come from installed Diffusers/Transformers. Bridge and LTX-shaped
components are small real PyTorch networks, injected only at source I/O and
structural adaptation boundaries. No author checkout, downloaded code, mocked
forward or release weights are needed. This is not native LTX-2.5 equivalence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def prepare_sol_h3(source, root, catalog, recipe, monkeypatch):
    import torch
    from _sol_h3_tiny_components import (
        audio_encoder_source,
        bridge_sources,
        refiner_source,
        video_decoder_source,
        video_encoder_source,
    )
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from safetensors.torch import save_file
    from test_qwen_image21_feature_encoder import tiny_config
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3VLForConditionalGeneration

    source.mkdir()
    h3_root = source / "h3"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(917)
        Qwen3VLForConditionalGeneration(tiny_config()).eval().save_pretrained(
            h3_root / "text_encoder", max_shard_size="1MB"
        )
        (source / "config.json").write_bytes((h3_root / "text_encoder/config.json").read_bytes())
        h3 = MiniMaxH3Transformer3DModel(
            num_attention_heads=2,
            attention_head_dim=128,
            hidden_size=256,
            num_layers=2,
            num_refiner_layers=2,
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
        h3.save_pretrained(h3_root / "transformer")
        # BigVGAN/Snake/anti-alias filters are retained; the hop product is 800.
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
        ).eval().save_pretrained(h3_root / "audio_vae")
        name = "transformer_blocks.0.adaln_proj.linear"
        layer = h3.get_submodule(name)
        save_file(
            {
                name + ".lora_A.weight": torch.randn(2, layer.in_features) * 0.001,
                name + ".lora_B.weight": torch.randn(layer.out_features, 2) * 0.001,
                "context_embedder.diff_b": torch.randn(256) * 0.001,
            },
            source / "dense.safetensors",
            metadata=dict(
                format="fastvideo-lora-v2", rank="2", low_rank_tensors="2", diff_tensors="1", set_weight_tensors="0"
            ),
        )
        del h3

        upscaler, adapter_model = bridge_sources()
        save_file(upscaler.state_dict(), source / "upscaler.safetensors")
        adapter = source / "adapter"
        adapter.mkdir()
        save_file(adapter_model.state_dict(), adapter / "model.safetensors")
        (adapter / "config.json").write_text(json.dumps({"format": "constructed-ci-component-v1"}))
        ltx = refiner_source()
        save_file(ltx.state_dict(), source / "ltx.safetensors")
        projection = ltx.transformer_blocks[0].attn1.to_q
        save_file(
            {
                "transformer_blocks.0.attn1.to_q.lora_A.weight": torch.randn(2, projection.in_features) * 0.01,
                "transformer_blocks.0.attn1.to_q.lora_B.weight": torch.randn(projection.out_features, 2) * 0.01,
            },
            source / "ltx-lora.safetensors",
        )
        encoder, decoder = video_encoder_source(), video_decoder_source()
        save_file(
            {
                **{"encoder." + name: value.contiguous() for name, value in encoder.state_dict().items()},
                **{"decoder." + name: value.contiguous() for name, value in decoder.state_dict().items()},
            },
            source / "video-vae.safetensors",
        )
        save_file(audio_encoder_source().state_dict(), source / "audio-vae.safetensors")
        # An actual fixture config is the source input; no fake author package
        # directories or Python modules are created to satisfy imports.
        (source / "components.json").write_text(json.dumps({"format": "constructed-ci-component-v1"}))
        recipe_data = {"kind": "constructed-small-components", "prompt": "refined detail"}
        fingerprint = hashlib.sha256(json.dumps(recipe_data, sort_keys=True).encode()).hexdigest()
        torch.save(
            dict(
                schema_version=1,
                prompt="refined detail",
                recipe=recipe_data,
                fingerprint=fingerprint,
                prompt_stats={},
                contexts=dict(
                    video=torch.randn(1, 1024, 128).bfloat16() * 0.01,
                    audio=torch.randn(1, 1024, 128).bfloat16() * 0.01,
                ),
            ),
            source / "prompt.pt",
        )

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(
        h3_root / "tokenizer"
    )
    inventory = dict(
        h3_model=str(h3_root),
        dense_lora=str(source / "dense.safetensors"),
        transformer=str(source / "ltx.safetensors"),
        refiner_lora=str(source / "ltx-lora.safetensors"),
        output_video_vae=str(source / "video-vae.safetensors"),
        audio_vae=str(source / "audio-vae.safetensors"),
        h3_upscaler_checkpoint=str(source / "upscaler.safetensors"),
        adapter_dir=str(source / "adapter"),
        refiner_prompt_cache=str(source / "prompt.pt"),
        sana_source=str(source),
        ltx_source=str(source),
        upscaler_source=str(source / "components.json"),
    )
    (source / "checkpoint-paths.json").write_text(json.dumps(inventory, indent=2))
    configure_component_loaders(monkeypatch)
    configure_small_recipe(catalog, recipe, monkeypatch)
    return {"prompt": "hello world", "seed": 7, "output_path": str(root / "result.mp4")}


def configure_component_loaders(monkeypatch):
    """Inject real checkpoint I/O, never model forward or SDK conversion."""
    from types import SimpleNamespace

    import _sol_h3_tiny_components as components
    import torch
    from safetensors.torch import load_file

    from xhmodel_merak.xh_other_model.models.sol_h3 import source_io
    from xhmodel_merak.xh_other_model.models.sol_h3.assets import SourceAssets, read_prompt_cache

    def load(factory, path, prefix=""):
        model = factory()
        payload = load_file(str(path))
        if prefix:
            payload = {name[len(prefix) :]: value for name, value in payload.items() if name.startswith(prefix)}
        assert payload and any(name.endswith("weight") for name in payload)
        model.load_state_dict(payload, strict=True)
        return model.eval().requires_grad_(False)

    def upscale(assets, device):
        model = load(components.SmallUpscaler, assets["h3_upscaler_checkpoint"]).bfloat16()
        return model, {"fixture": "constructed-small-network"}

    def refiner(assets, config, video_latent, audio_latent):
        model = load(components.refiner_source, assets["transformer"])
        from xhquant.nn import RMSNorm

        # The native Torch RMS leaf is not an SDK frontend module. Reuse the
        # production normalization adapter, preserving its weight and epsilon.
        for name in ("norm_out", "audio_norm_out"):
            native = getattr(model, name)
            adapted = RMSNorm(native.normalized_shape[-1], native.eps)
            adapted.weight = torch.nn.Parameter(native.weight.detach().clone(), requires_grad=False)
            model.set_submodule(name, adapted)
        adapter = load_file(str(assets["refiner_lora"]))
        with torch.no_grad():
            for name, value in adapter.items():
                if name.endswith(".lora_A.weight"):
                    target = name.removesuffix(".lora_A.weight")
                    model.get_submodule(target).weight.addmm_(adapter[target + ".lora_B.weight"], value, alpha=0.8)
        cache = read_prompt_cache(assets["refiner_prompt_cache"], dimensions=(128, 128))
        video, audio, rope, mask = components.refiner_conditioning(
            model,
            cache,
            size=config["refiner_size"],
            keyframe=config.get("ltx_reference_conditioning", {}).get("enabled", True),
            freeze_audio=True,
        )
        joint_video, joint_audio, _, _ = components.refiner_conditioning(
            model,
            cache,
            size=config["refiner_size"],
            keyframe=config.get("ltx_reference_conditioning", {}).get("enabled", True),
            freeze_audio=False,
        )
        return model, cache, video, audio, rope, mask, joint_video, joint_audio

    def decoder(assets, latent, config):
        source = load(components.video_decoder_source, assets["output_video_vae"], "decoder.")
        shape = (1, 3, 121, *config["refiner_size"])
        tile = SimpleNamespace(
            in_coords=tuple(slice(0, size) for size in latent.shape),
            out_coords=tuple(slice(0, size) for size in shape),
            masks_1d=tuple(torch.ones(size) for size in (1, 1, *shape[2:])),
        )
        return source, [tile], True

    def encoder_adaptation(source, scale, epsilon):
        # The constructed downsamplers have a linear learned branch and an
        # averaged skip. Rescale the stream while keeping their weights shared.
        with torch.no_grad():
            source.conv_in.convolution.weight.div_(scale)
            source.conv_in.convolution.bias.div_(scale)
            for block in source.down_blocks:
                block.conv.convolution.bias.div_(scale)
        source.conv_norm_out.eps = max(source.conv_norm_out.eps / scale**2, epsilon or 0)
        return source

    # Source directories are not third-party packages. The injected loaders
    # consume the generated checkpoint/config inventory instead of sys.path.
    monkeypatch.setattr(SourceAssets, "activate_build_sources", lambda self: None)
    monkeypatch.setattr(source_io, "upscaler", upscale)
    monkeypatch.setattr(
        source_io, "adapter", lambda assets: load(components.SmallAdapter, assets["adapter_dir"] / "model.safetensors")
    )
    monkeypatch.setattr(
        source_io, "audio_encoder", lambda assets: load(components.audio_encoder_source, assets["audio_vae"])
    )
    monkeypatch.setattr(
        source_io,
        "video_encoder",
        lambda assets: (load(components.video_encoder_source, assets["output_video_vae"], "encoder."), 0),
    )
    monkeypatch.setattr(source_io, "refiner", refiner)
    monkeypatch.setattr(source_io, "video_decoder", decoder)
    monkeypatch.setattr(source_io, "prepare_video_encoder", encoder_adaptation)
    monkeypatch.setattr(source_io, "prepare_video_decoder", lambda source: source)
    monkeypatch.setattr(source_io, "prepare_audio_encoder", lambda source: source)


def configure_small_recipe(catalog, recipe, monkeypatch):
    monkeypatch.setitem(
        catalog._recipes,
        recipe.model_id,
        replace(
            recipe,
            # These local Torch checkpoint loaders do not use the author's
            # OIIO/SciPy I/O. Keep production requirements and validation intact.
            dependencies=replace(
                recipe.dependencies,
                build=tuple(
                    value
                    for value in recipe.dependencies.build
                    if canonicalize_name(Requirement(value).name) not in {"openimageio", "scipy"}
                ),
            ),
            default_profile="all-dense-32p-64p-5s",
            workflows_by_profile=(),
            profiles=tuple(profile.replace("384p-768p", "32p-64p") for profile in recipe.profiles),
            artifact_name_template="hmquant_xh2_sol_h3_{deployment_precision}_small",
            config_replacements=(
                *recipe.config_replacements,
                ("export.profile", "all-dense-32p-64p-5s"),
                ("export.draft_size", [32, 32]),
                ("export.refiner_size", [64, 64]),
                ("export.prompt.max_tokens", 8),
                ("export.qwen3_vl.prefill_chunk_length", 1),
                ("export.text_layers", 2),
                ("export.text_layers_per_graph", 1),
                ("export.video_tile", [128, 64, 64]),
                ("export.video_overlap", [24, 0, 0]),
                ("export.calibration_prompt", "hello world"),
            ),
        ),
    )


def prepare_conditioned_source(tmp_path, monkeypatch, mode):
    import av
    import numpy as np
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
    from PIL import Image
    from test_sol_h3_reference_vae import tiny_vae
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessorFast, Qwen3VLProcessor, Qwen3VLVideoProcessor

    from xhmodel_merak.model_catalog.registry import get_catalog

    catalog = get_catalog()
    source = tmp_path / "source"
    inputs = prepare_sol_h3(source, tmp_path, catalog, catalog.get_recipe("sol-h3"), monkeypatch)
    h3 = source / "h3"
    config = dict(tiny_vae().config)
    config.update(latent_channels=24, latents_mean=[0.0] * 24, latents_std=[1.0] * 24)
    AutoencoderKLMiniMaxH3.from_config(config).eval().save_pretrained(h3 / "vae")
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
    Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessorFast(
            patch_size=16, temporal_patch_size=2, merge_size=2, size={"shortest_edge": 1024, "longest_edge": 1024}
        ),
        video_processor=Qwen3VLVideoProcessor(
            patch_size=16, temporal_patch_size=2, merge_size=2, size={"shortest_edge": 1024, "longest_edge": 1024}
        ),
    ).save_pretrained(h3 / "processor")
    media = {}
    for key, color in (("image", (50, 150, 190)), ("last_image", (190, 100, 50))):
        if (mode == "i2va" and key == "last_image") or (mode == "l2va" and key == "image"):
            continue
        path = tmp_path / f"{key}.png"
        Image.new("RGB", (64, 64), color).save(path)
        media[key] = str(path)
    keyframes = dict(media)
    extra = ()
    if mode in ("ref2va", "all"):
        import wave

        import numpy as np
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
        from safetensors.torch import save_file

        from xhmodel_merak.xh_other_model.models.minimax_h3.lora import LORA_TARGET_MODULES

        ref = MiniMaxH3Transformer3DModel.from_pretrained(h3 / "transformer").eval()
        with torch.no_grad():
            next(ref.parameters()).mul_(0.9)
        ref.save_pretrained(h3 / "transformer_ref")
        pairs = {}
        for name, module in ref.named_modules():
            if isinstance(module, torch.nn.Linear) and name.endswith(LORA_TARGET_MODULES):
                pairs[name + ".lora_A.default.weight"] = torch.randn(2, module.in_features) * 0.001
                pairs[name + ".lora_B.default.weight"] = torch.randn(module.out_features, 2) * 0.001
        lora = source / "reference.safetensors"
        save_file(pairs, lora, metadata={"alpha": "8", "format": "pt"})
        inventory = source / "checkpoint-paths.json"
        paths = json.loads(inventory.read_text())
        paths["ref2va_lora"] = str(lora)
        inventory.write_text(json.dumps(paths))
        video_path = tmp_path / "reference.mp4"
        with av.open(str(video_path), mode="w") as container:
            stream = container.add_stream("libx264", rate=24)
            stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
            for index in range(48):
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 50 + index, np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        audio_path = tmp_path / "reference.wav"
        samples = (np.sin(np.arange(64000) * (2 * np.pi * 220 / 32000)) * 3000).astype("<i2")
        with wave.open(str(audio_path), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(32000)
            output.writeframes(np.stack((samples, samples), -1).tobytes())
        media = {
            "references": [
                {"kind": "image", "path": media["image"]},
                {"kind": "video", "path": str(video_path)},
                {"kind": "audio", "path": str(audio_path)},
            ]
        }
        extra = (
            (
                "export.references",
                dict(
                    image=dict(max_count=1, vae_short_edge=32),
                    video=dict(max_count=1, max_duration_seconds=2),
                    audio=dict(max_count=1, max_duration_seconds=2),
                ),
            ),
        )
    profile = f"{mode}-dense-32p-64p-5s"
    inventory = {mode: media}
    if mode == "all":
        inventory = {
            "i2va": {"image": keyframes["image"]},
            "l2va": {"last_image": keyframes["last_image"]},
            "fl2va": keyframes,
            "ref2va": media,
        }
    (source / "calibration_inputs.json").write_text(json.dumps(inventory))
    recipe = catalog.get_recipe("sol-h3")
    monkeypatch.setitem(
        catalog._recipes,
        "sol-h3",
        replace(
            recipe,
            config_replacements=(
                *recipe.config_replacements,
                ("export.profile", profile),
                ("export.mode", mode),
                ("export.qwen3_vl.prefill_chunk_length", 4),
                ("export.qwen3_vl.image_max_pixels", 1024),
                ("export.qwen3_vl.video_max_pixels", 1024),
                *extra,
            ),
        ),
    )
    return source, inputs, media, inventory, profile


def assert_all_export_layout(source, artifact, inputs, profile):
    from pathlib import Path

    from xhmodel_merak.xh_other_model.models.sol_h3.artifacts import read_metadata
    from xhmodel_merak.xh_other_model.models.sol_h3.bundle import mode_metadata

    meta = read_metadata(artifact)
    assert meta["complete"] and meta["mode"] == "all" and meta["profile"] == profile
    assert source.is_dir() and not Path(inputs["output_path"]).exists()
    assert (artifact / "export_meta_info.json").is_file() and (artifact / "model_build_manifest.json").is_file()
    assert len(meta["feature_graphs"]) == 2
    assert len(meta["upscaler_graphs"]) == 2 and meta["video_decoder_graphs"] == ["ltx_video_decode"]
    assert len(meta["reference_vae_masks"]) == 2 and meta["reference_vae_base"]["graph"] in meta["graphs"]
    expected = {
        "h3_text_refiner_base",
        "h3_text_refiner_ref",
        "h3_denoiser_base",
        "h3_denoiser_ref",
        "h3_reference_audio_encode",
        "h3_to_ltx",
        "h3_audio_decode",
        "ltx_audio_encode",
        "ltx_image_encode",
        "ltx_denoiser",
        meta["reference_vae_base"]["graph"],
        *meta["reference_vae_masks"],
        *meta["feature_graphs"],
        *meta["upscaler_graphs"],
        *meta["video_decoder_graphs"],
        *(gear["graph"] for gear in meta["visual_graphs"]),
    }
    assert set(meta["graphs"]) == expected
    assert len(meta["graphs"]) == 18 + len(meta["visual_graphs"])
    assert [gear["capacity"] for gear in meta["visual_graphs"]] == meta["capacity_plan"]["qwen_visual_patch_gears"]
    h3_inputs = [
        "video_tokens",
        "audio_tokens",
        "text_features",
        "step",
        "cos",
        "sin",
        "role_pack",
        "head_unpack",
        "key_mask",
    ]
    ltx_inputs = [
        "video",
        "audio",
        "step",
        "vcos",
        "vsin",
        "acos",
        "asin",
        "vccos",
        "vcsin",
        "accos",
        "acsin",
        "first_role",
        "video_valid_length",
        "audio_mode",
    ]
    for mode in ("t2va", "i2va", "l2va", "fl2va", "ref2va"):
        selected = mode_metadata(meta, mode)
        assert selected["graphs"]["h3_denoiser"]["input_names"] == h3_inputs
        assert selected["graphs"]["ltx_denoiser"]["input_names"] == ltx_inputs
        assert selected["h3_role_layout"]["format"] == "fixed_adaln_role_slots_v1"
