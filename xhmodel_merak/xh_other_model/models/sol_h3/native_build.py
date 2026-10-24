# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Build native H3 and independent keep-LoRA views with shared encoders/codecs."""

import json
from copy import deepcopy
from pathlib import Path

from .build import BundleExporter
from .capacity import MODES
from .native_assets import NativeSourceAssets, read_native_adapters
from .native_schedule import native_schedule


def validate_native_config(config):
    cfg = deepcopy(dict(config))
    mode = cfg.get("mode")
    if mode not in (*MODES, "all"):
        raise ValueError("native H3 mode must name a supported input mode or all")
    scheme = cfg.get("model", {}).get("quant_scheme", {})
    if cfg.get("target_device") != "XH2a" or scheme.get("target_device") != "XH2a":
        raise ValueError("native H3 requires XH2a")
    if scheme.get("quant_type") not in ("w8a8_sefp", "w16a16_sefp"):
        raise ValueError("native H3 requires W8A8/W16A16 SEFP base weights")
    if cfg.get("conditioner_quant_type") not in (None, "w4a8h0_ssfp", "w8a8_sefp", "w16a16_sefp"):
        raise ValueError("unsupported native H3 conditioner quantization")
    if cfg.get("draft_frames") != 124 or cfg.get("fps") != 24 or cfg.get("draft_attention") != "dense":
        raise ValueError("native H3 currently exports dense 124-frame, 24-fps graphs")
    size = cfg.get("draft_size")
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or any(type(n) is not int or n <= 0 or n % 32 for n in size)
    ):
        raise ValueError("native H3 draft_size must contain two positive multiples of 32")
    profile = cfg.get("profile", "")
    if (
        not isinstance(profile, str)
        or not profile.startswith(f"{mode}-native-{size[0]}p-")
        or not profile.endswith("-5s")
    ):
        raise ValueError("native H3 profile must match its declared mode and canvas")
    capacity = cfg.get("prompt", {}).get("max_tokens")
    if type(capacity) is not int or capacity <= 0 or cfg.get("text_capacity", capacity) != capacity:
        raise ValueError("native H3 requires a positive prompt.max_tokens consistent with text_capacity")
    cfg["text_capacity"] = capacity
    layers, group = cfg.get("text_layers"), cfg.get("text_layers_per_graph")
    if type(layers) is not int or not 0 < layers <= 50 or type(group) is not int or group <= 0 or layers % group:
        raise ValueError("text_layers must be at most 50 and divisible by text_layers_per_graph")
    chunk = cfg.get("qwen3_vl", {}).get("prefill_chunk_length", 256)
    if type(chunk) is not int or chunk <= 0:
        raise ValueError("prefill_chunk_length must be a positive integer")
    if not isinstance(cfg.get("calibration_prompt"), str) or not cfg["calibration_prompt"].strip():
        raise ValueError("native H3 requires a real calibration_prompt")
    if type(cfg.get("calibration_seed")) is not int:
        raise ValueError("native H3 requires an integer calibration_seed")
    cfg.setdefault("native_sampling", dict(steps=50, video_shift=12.0, audio_shift=3.0))
    native_schedule(**cfg["native_sampling"])
    cfg.setdefault("video_tile", [256, 256])
    cfg.setdefault("video_overlap", [64, 64])
    tile, overlap = cfg["video_tile"], cfg["video_overlap"]
    if (
        not isinstance(tile, (list, tuple))
        or not isinstance(overlap, (list, tuple))
        or len(tile) != 2
        or len(overlap) != 2
        or any(type(n) is not int or n <= 0 or n % 16 for n in tile)
        or any(type(n) is not int or n < 0 or n % 16 for n in overlap)
        or any(o >= t for o, t in zip(overlap, tile, strict=True))
    ):
        raise ValueError("native video tiles/overlaps must be aligned to 16 with positive stride")
    from .precision import H3PrecisionPolicy

    cfg["h3_precision"] = H3PrecisionPolicy(**cfg.get("h3_precision", {})).to_dict()
    return cfg


class NativeBundleExporter(BundleExporter):
    """Reuse graph export and conditioning without importing Sol/LTX sources."""

    def __init__(self, assets, root, config, device, conditioner_checkpoint=None):
        import torch

        from .precision import ATTENTION_VALUE_DIVISOR, H3PrecisionPolicy

        self.assets, self.root, self.cfg = assets, Path(root), config
        self.device, self.conditioner_checkpoint = torch.device(device), conditioner_checkpoint
        if self.device.type != "cuda":
            raise ValueError("native H3 SEFP weight conversion requires CUDA; graph capture remains on CPU")
        torch.cuda.set_device(self.device)
        self.sessions, self.branch, self.variant = {}, None, None
        self.root.mkdir(parents=True, exist_ok=False)
        self.meta = dict(
            schema_version=2,
            kind="native_h3",
            complete=False,
            mode=config["mode"],
            target_device="XH2a",
            input_dtype="float16",
            quant_type=config["model"]["quant_scheme"]["quant_type"],
            attention_value_divisor=ATTENTION_VALUE_DIVISOR,
            scaled_rms_policy=H3PrecisionPolicy(**config["h3_precision"]).scaled_rms_policy,
            h3_precision=config["h3_precision"],
            graphs={},
            resources={},
            tokenizer_files=[],
            feature_graphs=[],
            visual_graphs=[],
            video_decoder_graphs=[],
            native_variants={},
            mode_profiles={},
            text_feature_divisor=16.0,
            text_refiner_input_divisor=16.0,
            text_rms_policy="plain_wide_reduction",
        )
        for key in ("profile", "draft_size", "draft_frames", "fps", "text_capacity", "text_layers"):
            self.meta[key] = config[key]

    def graph_name(self, name):
        if name in ("h3_text_refiner", "h3_denoiser") and self.branch is not None:
            return f"{name}_{self.variant}_{self.branch}"
        return name

    def draft(self, schedule, adapter=None):
        import torch
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

        from .conditioning import conditioned_tables, fold_h3_conditioning

        self.progress(f"h3_source_{self.variant}_{self.branch}")
        source = (
            MiniMaxH3Transformer3DModel.from_pretrained(
                self.assets["h3_model"] / ("transformer_ref" if self.branch == "ref" else "transformer"),
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
        )
        folded = fold_h3_conditioning(source, schedule, compute_device=self.device)
        projection_parameters = folded.original_projection_parameters
        tables, output_table = conditioned_tables(folded, schedule)
        result = self._export_h3_draft(
            source, tables, output_table, schedule=schedule, adapter=adapter, variant=self.variant
        )
        result["folded_h3_projection_parameters"] = projection_parameters
        return result

    def video_vae(self):
        import torch

        from .native_video import NativeVideoDecodeAdapter, NativeVideoDecodeGraph, load_native_video_vae

        source, config = load_native_video_vae(self.assets["h3_model"])
        self.resource("h3_video_config", config)
        policy = dict(tile_size=self.cfg["video_tile"], overlap=self.cfg["video_overlap"])
        tiles = {}

        def invoke(shape, value):
            if value.dtype != torch.float32 or value.device.type != "cpu":
                raise ValueError("native video graph calibration must use CPU FP32 latents")
            key = "x".join(map(str, shape))
            if key not in tiles:
                self.close()
                name = f"h3_video_decode_t{shape[0]}_h{shape[1]}_w{shape[2]}"
                graph = NativeVideoDecodeGraph(source, shape, sdk_ops=True)
                self.graph(
                    name,
                    f"vae_decode_h3_video_t{shape[0]}_h{shape[1]}_w{shape[2]}",
                    graph,
                    (value,),
                    ["latents"],
                    ["pixels"],
                    dynamo=True,
                )
                tiles[key] = name
                self.meta["video_decoder_graphs"].append(name)
                del graph
            result = self.infer(tiles[key], (value,))[0]
            return result

        decoder = NativeVideoDecodeAdapter(config, invoke, dtype=torch.float32, **policy)
        expected = decoder.tile_shapes([tuple(self.h3_latent.shape[2:])])
        pixels = decoder(self.h3_latent)
        if tuple(pixels.shape) != (1, 3, self.cfg["draft_frames"], *self.cfg["draft_size"]):
            raise ValueError("native H3 video decoder produced the wrong output canvas")
        if set(tiles) != {"x".join(map(str, shape)) for shape in expected} or not bool(torch.isfinite(pixels).all()):
            raise ValueError("native H3 decoder calibration did not cover every finite tile gear")
        self.meta["native_video_decoder"] = dict(tiles=tiles, config="h3_video_config", **policy)


def build_native_bundle(model_dir, output_dir, config, *, device="cuda:0", conditioner_checkpoint=None):
    import torch
    from transformers import AutoProcessor, AutoTokenizer

    from .build import calibration_inputs
    from .capacity import configure_processor, mode_configs, plan_capacities, plan_unified_capacities
    from .native_artifacts import export_entries, read_metadata
    from .reference_vae import ReferenceVAEAdapter, invoke_reference_tile
    from .request import prepare_configured_request, prepare_encoder_calibration
    from .unified_build import _replay_conditioning

    cfg, assets = validate_native_config(config), NativeSourceAssets.read(model_dir)
    modes = MODES if cfg["mode"] == "all" else (cfg["mode"],)
    branches = {"ref" if mode == "ref2va" else "base" for mode in modes}
    adapters = read_native_adapters(cfg.get("model", {}).get("lora", {}).get("path", []))
    for name, adapter in adapters.items():
        adapter["branches"] = {key: value for key, value in adapter["branches"].items() if key in branches}
        if not adapter["branches"]:
            raise ValueError(f"Native H3 adapter {name!r} has no weights for the selected mode")
    variants = {
        "standard": dict(
            sampling=cfg["native_sampling"],
            schedule=native_schedule(**cfg["native_sampling"]),
            branches={branch: None for branch in sorted(branches)},
        ),
        **adapters,
    }
    actual = "w4a8h0_ssfp" if conditioner_checkpoint is not None else cfg["model"]["quant_scheme"]["quant_type"]
    if cfg.get("conditioner_quant_type") not in (None, actual):
        raise ValueError(f"conditioner_quant_type requires matching weights; got {actual}")
    tokenizer = AutoTokenizer.from_pretrained(assets["h3_model"] / "tokenizer", local_files_only=True)
    processor = None
    if modes != ("t2va",):
        processor = configure_processor(
            AutoProcessor.from_pretrained(assets["h3_model"] / "processor", local_files_only=True),
            cfg.get("qwen3_vl", {}),
        )
    text_config = json.loads((assets["h3_model"] / "text_encoder/config.json").read_text())["text_config"]
    components = dict(
        tokenizer=tokenizer, processor=processor, max_position_embeddings=text_config["max_position_embeddings"]
    )
    if cfg["mode"] == "all":
        plans = plan_unified_capacities(cfg, **components)
        configs = mode_configs(cfg)
        for mode, mode_cfg in configs.items():
            mode_cfg["profile"] = f"{mode}-native-{cfg['draft_size'][0]}p-5s"
        calibration_modes = ("ref2va", "fl2va")
    else:
        plan = plan_capacities(cfg, **components)
        configs = {cfg["mode"]: deepcopy(cfg)}
        plans = dict(
            modes={cfg["mode"]: plan},
            branches={next(iter(branches)): plan},
            encoder={
                key: plan[key]
                for key in ("text_capacity", "kv_capacity", "qwen_visual_max_patches", "qwen_visual_patch_gears")
            },
        )
        calibration_modes = (cfg["mode"],)
    requests = {
        mode: prepare_configured_request(
            cfg["calibration_prompt"],
            config=configs[mode],
            tokenizer=tokenizer,
            **calibration_inputs(model_dir, configs[mode]),
        )
        for mode in calibration_modes
        if mode != "t2va"
    }
    calibrations = {
        mode: prepare_encoder_calibration(request, processor, configs[mode]) for mode, request in requests.items()
    }
    common = deepcopy(plans["modes"][calibration_modes[0]])
    common.update(mode=cfg["mode"], **plans["encoder"])
    for field in ("video_capacity", "audio_capacity"):
        common[field] = max(branch[field] for branch in plans["branches"].values())
    common["dit_capacity"] = sum(common[key] for key in ("text_capacity", "video_capacity", "audio_capacity"))
    cfg["text_capacity"] = common["text_capacity"]
    exporter = NativeBundleExporter(assets, output_dir, cfg, device, conditioner_checkpoint)
    exporter.meta.update(capacity_plan=common, prompt_max_tokens=common["prompt_max_tokens"])
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
    for name, variant in variants.items():
        schedule = variant["schedule"]
        exporter.meta["native_variants"][name] = dict(
            steps=schedule.steps,
            sampling=variant["sampling"],
            h3_sigmas=list(schedule.video_sigmas),
            audio_sigmas=list(schedule.audio_sigmas),
            branches={},
            adapter_resource=None,
        )
        if name != "standard":
            resource = f"adapter_{name}"
            exporter.resource(
                resource,
                dict(
                    name=name,
                    sampling=variant["sampling"],
                    branches={branch: spec.to_dict() for branch, spec in variant["branches"].items()},
                ),
            )
            exporter.meta["native_variants"][name]["adapter_resource"] = resource
    if processor is not None:
        processor.save_pretrained(exporter.root / "processor")
        exporter.meta["processor_files"] = [
            p.relative_to(exporter.root).as_posix()
            for p in sorted((exporter.root / "processor").iterdir())
            if p.is_file()
        ]
    try:
        vae_config, audio_vae = None, None
        if calibrations:
            calibration = calibrations[calibration_modes[0]]
            samples = [sample for value in calibrations.values() for sample in value["visual_calibrations"]]
            exporter.vision(*samples[0], additional_calibrations=samples[1:])
            ref_calibration = calibrations.get("ref2va", {})
            exporter.reference_vae(
                calibration["pixels"],
                base_image_shape=tuple(cfg["draft_size"])
                if any(mode in ("i2va", "l2va", "fl2va") for mode in modes)
                else None,
                image_shapes=ref_calibration.get("image_shapes", ()),
                video_shapes=ref_calibration.get("video_shapes", ()),
            )
            vae_config = json.loads((exporter.root / exporter.meta["resources"]["reference_vae_config"]).read_text())
            if "ref2va" in calibrations:
                # Native Ref2VA expects an audio adapter even for image-only
                # requests. The shared helper exports no graph for zero capacity.
                audio_vae = exporter.reference_audio(ref_calibration["waveform"])

        def encode_tile(shape, value):
            return invoke_reference_tile(
                exporter.meta,
                shape,
                value,
                lambda name, *args: exporter.infer(name, args)[0],
                mode="fl2va" if exporter.branch == "base" else "ref2va",
            )

        for index, mode in enumerate(calibration_modes):
            branch = "ref" if mode == "ref2va" else "base"
            exporter.branch = branch
            vae = (
                ReferenceVAEAdapter(
                    vae_config, encode_tile, untiled=branch == "base", **exporter.meta["reference_vae_policy"]
                )
                if vae_config is not None
                else None
            )
            if index == 0:
                exporter.text(requests.get(mode), processor=processor, vae=vae, audio_vae=audio_vae)
                exporter.close()
            exporter.cfg = {**configs[mode], "text_capacity": plans["modes"][mode]["text_capacity"]}
            exporter.meta.update(text_capacity=exporter.cfg["text_capacity"], capacity_plan=plans["modes"][mode])
            if index:
                _replay_conditioning(exporter, requests[mode], processor, vae, audio_vae if branch == "ref" else None)
            else:
                exporter.text_features = torch.nn.functional.pad(
                    exporter.text_features[:, : exporter.text_length],
                    (0, 0, 0, exporter.cfg["text_capacity"] - exporter.text_length),
                )
            for name, variant in variants.items():
                if branch not in variant["branches"]:
                    continue
                exporter.variant = name
                exporter.meta["native_variants"][name]["branches"][branch] = exporter.draft(
                    variant["schedule"], variant["branches"][branch]
                )
        exporter.branch = exporter.variant = None
        exporter.cfg = cfg
        for stage in (exporter.h3_audio, exporter.video_vae):
            try:
                stage()
            finally:
                exporter.close()
    finally:
        exporter.close()
    exporter.meta.update(text_capacity=common["text_capacity"], capacity_plan=common, complete=True)
    (exporter.root / "runtime_meta_info.json").write_text(json.dumps(exporter.meta, indent=2) + "\n")
    metadata = read_metadata(exporter.root)
    (exporter.root / "export_meta_info.json").write_text(json.dumps(export_entries(metadata), indent=2) + "\n")
    (exporter.root / "build_progress.json").unlink()
    return metadata
