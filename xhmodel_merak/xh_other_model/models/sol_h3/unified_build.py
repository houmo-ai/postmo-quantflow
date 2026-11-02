# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Export shared encoders/codecs once and the two distinct H3 weight branches."""

import json
from copy import deepcopy


def _replay_conditioning(exporter, request, processor, vae, audio_vae):
    """Calibrate another DiT using the already exported common feature graphs."""
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from .build import _example_latent, _example_vision_features
    from .conditioner import H3FeatureEncoder
    from .prefill import prefill_feature_graphs
    from .request import encode_request
    from .vision import VisionFeatureAdapter

    config = json.loads((exporter.root / exporter.meta["resources"]["qwen_config"]).read_text())
    divisor = exporter.meta["text_feature_divisor"]
    export_only = exporter.meta["kind"] == "sol_h3"
    embedding = (
        torch.from_numpy(np.load(exporter.root / exporter.meta["resources"]["text_embedding"])).float() * divisor
    )
    vision = VisionFeatureAdapter(
        (
            (
                lambda name, *values: _example_vision_features(
                    values[0], config["vision_config"], seed=exporter.cfg["calibration_seed"]
                )
            )
            if export_only
            else lambda name, *values: exporter.infer(name, values)
        ),
        config["vision_config"],
        gears=exporter.meta["visual_graphs"],
        device="cpu" if export_only else exporter.device,
    )

    def features(hidden, cos, sin, additions):
        if export_only:
            return _example_latent(hidden.shape, seed=exporter.cfg["calibration_seed"]) / divisor
        return prefill_feature_graphs(
            hidden,
            cos,
            sin,
            additions,
            metadata=exporter.meta,
            invoke=lambda name, *values: exporter.infer(name, values),
            close=exporter.close,
            device=exporter.device,
        )

    encoder = H3FeatureEncoder(
        config, embedding, vision, features, feature_layer=exporter.cfg["text_layers"], divisor=divisor
    )
    tokenizer = AutoTokenizer.from_pretrained(exporter.root / "tokenizer", local_files_only=True)
    hidden, _ = encode_request(
        request,
        text_encoder=encoder,
        tokenizer=tokenizer,
        processor=processor,
        vae=vae,
        audio_vae=audio_vae,
        text_encoder_layer=exporter.cfg["text_layers"],
    )
    exporter.request = request
    exporter.text_length = hidden.shape[1]
    exporter.text_features = torch.nn.functional.pad(
        hidden.float() / divisor, (0, 0, 0, exporter.cfg["text_capacity"] - hidden.shape[1])
    )


def build_unified_bundle(model_dir, output_dir, config, *, device, conditioner_checkpoint):
    import torch
    from transformers import AutoProcessor, AutoTokenizer

    from .artifacts import export_entries, read_metadata
    from .assets import SourceAssets
    from .build import BundleExporter, _example_reference_moments, calibration_inputs, validate_build_config
    from .bundle import BRANCH_GRAPHS
    from .capacity import configure_processor, mode_configs, plan_unified_capacities
    from .reference_vae import ReferenceVAEAdapter, invoke_reference_tile
    from .request import prepare_configured_request, prepare_encoder_calibration
    from .role_layout import role_capacities_from_plan

    cfg, assets = validate_build_config(config), SourceAssets.read(model_dir)
    actual = "w4a8h0_ssfp" if conditioner_checkpoint is not None else cfg["model"]["quant_scheme"]["quant_type"]
    if cfg.get("conditioner_quant_type") not in (None, actual):
        raise ValueError(f"conditioner_quant_type requires matching weights; got {actual}")
    if (
        "ref2va_lora" not in assets.paths
        or not assets["ref2va_lora"].is_file()
        or not (assets["h3_model"] / "transformer_ref/config.json").is_file()
    ):
        raise ValueError("unified Sol-H3 requires both H3 checkpoints and their adapters")
    tokenizer = AutoTokenizer.from_pretrained(assets["h3_model"] / "tokenizer", local_files_only=True)
    processor = configure_processor(
        AutoProcessor.from_pretrained(assets["h3_model"] / "processor", local_files_only=True), cfg.get("qwen3_vl", {})
    )
    text_config = json.loads((assets["h3_model"] / "text_encoder/config.json").read_text())["text_config"]
    plans = plan_unified_capacities(
        cfg, tokenizer=tokenizer, processor=processor, max_position_embeddings=text_config["max_position_embeddings"]
    )
    configs = mode_configs(cfg)
    role_capacities = {
        branch: tuple(max(role_capacities_from_plan(plans["modes"][mode])[role] for mode in modes) for role in range(5))
        for branch, modes in (("base", ("t2va", "i2va", "l2va", "fl2va")), ("ref", ("ref2va",)))
    }
    requests = {
        mode: prepare_configured_request(
            cfg["calibration_prompt"],
            config=configs[mode],
            tokenizer=tokenizer,
            **calibration_inputs(model_dir, configs[mode]),
        )
        for mode in ("ref2va", "fl2va")
    }
    calibrations = {
        mode: prepare_encoder_calibration(request, processor, configs[mode]) for mode, request in requests.items()
    }
    common_plan = deepcopy(plans["modes"]["ref2va"])
    common_plan.update(mode="all", **plans["encoder"])
    for field in ("video_capacity", "audio_capacity"):
        common_plan[field] = max(branch[field] for branch in plans["branches"].values())
    common_plan["dit_capacity"] = sum(common_plan[key] for key in ("text_capacity", "video_capacity", "audio_capacity"))
    cfg["text_capacity"] = common_plan["text_capacity"]
    assets.activate_build_sources()
    exporter = BundleExporter(assets, output_dir, cfg, device, conditioner_checkpoint)
    exporter.meta.update(
        mode="all",
        capacity_plan=common_plan,
        prompt_max_tokens=common_plan["prompt_max_tokens"],
        branches={},
        mode_profiles={},
    )
    for mode, mode_cfg in configs.items():
        name = "input_config_" + mode
        exporter.resource(
            name,
            {
                key: mode_cfg[key]
                for key in ("mode", "prompt", "draft_size", "draft_frames", "references", "qwen3_vl")
                if key in mode_cfg
            },
        )
        exporter.meta["mode_profiles"][mode] = dict(
            branch="ref" if mode == "ref2va" else "base",
            profile=mode_cfg["profile"],
            capacity_plan=plans["modes"][mode],
            input_config=name,
        )
    processor.save_pretrained(exporter.root / "processor")
    exporter.meta["processor_files"] = [
        path.relative_to(exporter.root).as_posix()
        for path in sorted((exporter.root / "processor").iterdir())
        if path.is_file()
    ]
    try:
        calibration = calibrations["ref2va"]
        visual_calibrations = [
            sample for mode in ("fl2va", "ref2va") for sample in calibrations[mode]["visual_calibrations"]
        ]
        exporter.vision(*visual_calibrations[0], additional_calibrations=visual_calibrations[1:])
        exporter.reference_vae(
            calibration["pixels"],
            base_image_shape=tuple(cfg["draft_size"]),
            image_shapes=calibration["image_shapes"],
            video_shapes=calibration["video_shapes"],
        )
        vae_config = json.loads((exporter.root / exporter.meta["resources"]["reference_vae_config"]).read_text())

        def encode_tile(shape, value):
            return invoke_reference_tile(
                exporter.meta,
                shape,
                value,
                lambda name, *args: _example_reference_moments(
                    args[0].shape[2:], exporter._reference_vae_geometry, seed=cfg["calibration_seed"]
                ),
                mode="fl2va" if exporter.branch == "base" else "ref2va",
            )

        vae = ReferenceVAEAdapter(vae_config, encode_tile, **exporter.meta["reference_vae_policy"])
        audio_vae = exporter.reference_audio(calibration["waveform"])
        exporter.text(requests["ref2va"], processor=processor, vae=vae, audio_vae=audio_vae)
        exporter.close()
        for branch, mode in (("ref", "ref2va"), ("base", "fl2va")):
            exporter.cfg = {**configs[mode], "text_capacity": plans["modes"][mode]["text_capacity"]}
            exporter.meta.update(text_capacity=exporter.cfg["text_capacity"], capacity_plan=plans["modes"][mode])
            exporter.branch = branch
            exporter.role_capacities = role_capacities[branch]
            if branch == "base":
                base_vae = ReferenceVAEAdapter(
                    vae_config, encode_tile, untiled=True, **exporter.meta["reference_vae_policy"]
                )
                _replay_conditioning(exporter, requests[mode], processor, base_vae, None)
            else:
                exporter.text_features = torch.nn.functional.pad(
                    exporter.text_features[:, : exporter.text_length],
                    (0, 0, 0, exporter.cfg["text_capacity"] - exporter.text_length),
                )
            exporter.draft()
            exporter.close()
            exporter.meta["branches"][branch] = dict(
                graphs={name: exporter.graph_name(name) for name in BRANCH_GRAPHS},
                **{
                    key: deepcopy(exporter.meta[key])
                    for key in (
                        "h3_adapter",
                        "h3_rope_config",
                        "h3_role_layout",
                        "folded_h3_projection_parameters",
                    )
                },
            )
        exporter.branch = None
        for stage in (exporter.bridge, exporter.audio, exporter.refiner, exporter.video_vae):
            try:
                stage()
            finally:
                exporter.close()
    finally:
        exporter.close()
    exporter.cfg = cfg
    exporter.meta.update(text_capacity=common_plan["text_capacity"], capacity_plan=common_plan, complete=True)
    for key in ("h3_adapter", "h3_rope_config", "h3_role_layout", "folded_h3_projection_parameters"):
        exporter.meta.pop(key)
    (exporter.root / "runtime_meta_info.json").write_text(json.dumps(exporter.meta, indent=2) + "\n")
    metadata = read_metadata(exporter.root)
    (exporter.root / "export_meta_info.json").write_text(json.dumps(export_entries(metadata), indent=2) + "\n")
    (exporter.root / "build_progress.json").unlink()
    return metadata
