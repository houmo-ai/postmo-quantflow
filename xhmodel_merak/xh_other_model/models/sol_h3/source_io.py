# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Build-time component loading and native-source adaptation boundaries.

Deployment graph forward methods do not import or depend on author packages.
The default loaders retain the native checkpoint builders and conditioning.
"""


def upscaler(assets, device):
    import torch
    from runtime.stage2_ops.h3_upscale import load_upscaler

    upscaler, receipt = load_upscaler(
        assets["upscaler_source"], assets["h3_upscaler_checkpoint"], device=device, torch_module=torch
    )
    return upscaler, receipt


def adapter(assets):
    from runtime.stage2_ops.h3_ltx_adapter import H3ToLTXAdapter

    return H3ToLTXAdapter.from_pretrained(assets["adapter_dir"], device="cpu").model


def audio_encoder(assets):
    import torch
    from ltx_pipelines.utils.blocks import AudioConditioner

    block = AudioConditioner(str(assets["audio_vae"]), torch.bfloat16, torch.device("cpu"))
    return block._encoder_builder.build(device="cpu", dtype=torch.bfloat16).eval().requires_grad_(False)


def video_encoder(assets):
    import torch
    from ltx_pipelines.utils.blocks import ImageConditioner

    block = ImageConditioner(str(assets["output_video_vae"]), torch.bfloat16, torch.device("cpu"))
    source = block._encoder_builder.build(device=torch.device("cpu"), dtype=torch.bfloat16).eval()
    return source, block.default_image_crf


def refiner(assets, config, video_latent, audio_latent):
    import torch
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
    from ltx_pipelines.utils.blocks import DiffusionStage

    from .assets import read_prompt_cache
    from .ltx_keyframes import prepare_keyframe_states
    from .ltx_prepare import (
        cpu_lora_fuse_rule,
        freeze_audio_state,
        native_tools,
        prepare_conditioning,
        prepare_keyframe_conditioning,
    )

    size = config["refiner_size"]
    stage = DiffusionStage.from_checkpoint(
        str(assets["transformer"]),
        torch.bfloat16,
        torch.device("cpu"),
        loras=(LoraPathStrengthAndSDOps(str(assets["refiner_lora"]), 0.8, LTXV_LORA_COMFY_RENAMING_MAP),),
    )
    stage = stage.with_builder(stage._prepared_builder().with_fuse_rule(cpu_lora_fuse_rule()))
    frames, fps = config["refiner_frames"], config["fps"]
    video_tools, _ = native_tools(size=size, frames=frames, fps=fps)
    with torch.no_grad():
        source = stage._build_transformer(video_tools=video_tools).velocity_model.eval().requires_grad_(False)
    cache = read_prompt_cache(
        assets["refiner_prompt_cache"],
        dimensions=(source.patchify_proj.out_features, source.audio_patchify_proj.out_features),
    )
    if config.get("ltx_reference_conditioning", {}).get("enabled", True):
        image_shape = (1, 128, 1, size[0] // 32, size[1] // 32)
        video, audio, _, _ = prepare_keyframe_states(
            video_latent,
            audio_latent,
            keyframe_latents=[(frames - 1, torch.zeros(image_shape))],
            seed=config["calibration_seed"],
            size=size,
            refiner_frames=frames,
            freeze_audio=False,
        )
        video_args, audio_args, rope = prepare_keyframe_conditioning(source, cache, video, freeze_audio_state(audio))
        joint_video_args, joint_audio_args, _ = prepare_keyframe_conditioning(source, cache, video, audio)
        mask = video.keyframes_mask
    else:
        video_args, audio_args, rope = prepare_conditioning(
            source, cache, size=size, frames=frames, fps=fps, freeze_audio=True
        )
        joint_video_args, joint_audio_args, _ = prepare_conditioning(
            source, cache, size=size, frames=frames, fps=fps, freeze_audio=False
        )
        mask = video_tools.create_initial_state(device="cpu", dtype=torch.float16).keyframes_mask
    return source, cache, video_args, audio_args, rope, mask, joint_video_args, joint_audio_args


def video_decoder(assets, latent, config):
    import torch
    from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig, masks_are_complementary
    from ltx_pipelines.utils.blocks import VideoDecoder

    block = VideoDecoder(str(assets["output_video_vae"]), torch.bfloat16, torch.device("cpu"))
    decoder = block._decoder_builder.build(device="cpu", dtype=torch.bfloat16).eval().requires_grad_(False)
    if decoder.timestep_conditioning:
        raise ValueError("this Sol-H3 decoder gear requires the deterministic conv VAE")
    tiles = TileSizeConfig(
        **{
            name: DimensionSizeConfig(tile_size=size, overlap=overlap)
            for name, size, overlap in zip(
                ("frames", "height", "width"), config["video_tile"], config["video_overlap"], strict=True
            )
        }
    )
    native = decoder._prepare_tiles(latent, tiles)
    complementary = masks_are_complementary(native, (1, 3, config["refiner_frames"], *config["refiner_size"]))
    return decoder, native, complementary


def prepare_video_encoder(source, residual_scale, normalization_epsilon):
    from ltx_core.model.video_vae.enums import LogVarianceType

    from .vae_graph import _adapt_vae_norms_and_convolutions, _scale_encoder_residuals

    if source.latent_log_var not in (LogVarianceType.UNIFORM, LogVarianceType.CONSTANT, LogVarianceType.PER_CHANNEL):
        raise ValueError("LTX encode requires a checkpoint with explicit posterior mean channels")
    _scale_encoder_residuals(source, residual_scale)
    if normalization_epsilon is not None:
        from ltx_core.model.common.normalization import PixelNorm

        for module in source.modules():
            if isinstance(module, PixelNorm):
                module.eps = max(module.eps, normalization_epsilon)
    _adapt_vae_norms_and_convolutions(source, static_causal=True)
    return source


def prepare_video_decoder(source):
    from ltx_core.model.video_vae.memory_efficient_decode import disable_memory_efficient_decode
    from ltx_core.model.video_vae.sampling import DepthToSpaceUpsample

    from .vae_graph import DepthExpansion, _adapt_vae_norms_and_convolutions

    source = disable_memory_efficient_decode(source)
    _adapt_vae_norms_and_convolutions(source, static_causal=source.causal)
    for name, module in list(source.named_modules()):
        if isinstance(module, DepthToSpaceUpsample):
            source.set_submodule(name, DepthExpansion(module))
    return source


def prepare_audio_encoder(source):
    from ltx_core.model.common.normalization import PixelNorm

    from .vae_graph import ChannelPixelNorm

    for name, module in list(source.named_modules()):
        if isinstance(module, PixelNorm):
            if module.dim != 1:
                raise ValueError("audio VAE expects per-location channel normalization")
            source.set_submodule(name, ChannelPixelNorm(module.eps))
    return source
