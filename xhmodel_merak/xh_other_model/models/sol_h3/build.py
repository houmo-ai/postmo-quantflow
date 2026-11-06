# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Export the complete local Sol-H3 asset inventory into independently owned graphs."""

from __future__ import annotations

import gc
import json
from copy import deepcopy
from pathlib import Path

from .assets import SourceAssets


def _example_latent(shape, *, seed):
    """Deterministic bounded CPU export input, without evaluating a component."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.empty(shape, dtype=torch.float32, device="cpu").uniform_(-1, 1, generator=generator)


def _example_vision_features(pixels, config, *, seed):
    merged = pixels.shape[1] // config["spatial_merge_size"] ** 2
    shape = (pixels.shape[0], merged, config["out_hidden_size"])
    return tuple(_example_latent(shape, seed=seed + index) for index in range(4))


def _example_reference_moments(shape, geometry, *, seed):
    temporal, spatial, channels = geometry
    frames, height, width = shape
    return _example_latent(
        (1, 2 * channels, (frames - 1) // temporal + 1, height // spatial, width // spatial), seed=seed
    )


def validate_build_config(config):
    cfg = deepcopy(dict(config))
    from .capacity import MODES, matches_dense_profile
    from .duration import duration_gear

    cfg.setdefault("norm_fp16", True)
    if type(cfg["norm_fp16"]) is not bool:
        raise ValueError("norm_fp16 must be a boolean")
    cfg.setdefault("upscaler_conv3d_mode", "slice")
    if cfg["upscaler_conv3d_mode"] not in ("slice", "gather"):
        raise ValueError("upscaler_conv3d_mode must be 'slice' or 'gather'")
    cfg.setdefault("upscaler_conv3d_fastpath", False)
    if type(cfg["upscaler_conv3d_fastpath"]) is not bool:
        raise ValueError("upscaler_conv3d_fastpath must be a boolean")
    reference = cfg.get("ltx_reference_conditioning", {})
    if not isinstance(reference, dict) or set(reference) - {"enabled", "encoder_norm_eps", "require_native_resolution"}:
        raise ValueError("ltx_reference_conditioning accepts enabled, encoder_norm_eps and require_native_resolution")
    reference.setdefault("enabled", True)
    if type(reference["enabled"]) is not bool:
        raise ValueError("ltx_reference_conditioning.enabled must be a boolean")
    require_native = reference.get("require_native_resolution", False)
    if type(require_native) is not bool or (require_native and not reference["enabled"]):
        raise ValueError("require_native_resolution must be a boolean and requires enabled LTX image conditioning")
    epsilon = reference.setdefault("encoder_norm_eps", 2**-24)
    if epsilon is not None:
        import math

        if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon < 2**-24:
            raise ValueError("encoder_norm_eps must be None or a finite number >= FP16 minimum positive value")
    cfg["ltx_reference_conditioning"] = reference
    if cfg.get("mode", "t2va") not in (*MODES, "all"):
        raise ValueError("unsupported Sol-H3 mode")
    if "prompt" in cfg:
        maximum = cfg["prompt"].get("max_tokens")
        if "text_capacity" in cfg and cfg["text_capacity"] != maximum:
            raise ValueError("prompt.max_tokens conflicts with legacy text_capacity")
        cfg["text_capacity"] = maximum
    scheme = cfg.get("model", {}).get("quant_scheme", {})
    if cfg.get("conditioner_quant_type") not in (None, "w4a8h0_ssfp", "w8a8_sefp", "w16a16_sefp"):
        raise ValueError("unsupported Sol-H3 conditioner_quant_type")
    if (
        cfg.get("target_device") != "XH2a"
        or scheme.get("target_device") != "XH2a"
        or scheme.get("quant_type") not in ("w8a8_sefp", "w16a16_sefp")
    ):
        raise ValueError("Sol-H3 requires XH2a W8A8/W16A16 SEFP")
    expected = {
        "fps": 24,
        "draft_attention": "dense",
        "refiner_attention": "dense",
    }
    for name, value in expected.items():
        if cfg.get(name) != value:
            raise ValueError(f"this Sol-H3 gear requires {name}={value}")
    gear = duration_gear(cfg.get("draft_frames"), cfg.get("refiner_frames"), cfg.get("fps"))
    size = cfg.get("draft_size")
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or any(type(n) is not int or n <= 0 or n % 32 for n in size)
        or cfg.get("refiner_size") != [n * 2 for n in size]
        or not matches_dense_profile(cfg.get("profile"), size, gear.seconds, cfg.get("mode", "t2va"))
    ):
        raise ValueError("Sol-H3 static canvases must be aligned to 32 with a fixed x2 refiner")
    layers, capacity = cfg.get("text_layers"), cfg.get("text_capacity")
    if type(layers) is not int or not 0 < layers <= 50:
        raise ValueError("text_layers must select a used prefix of at most 50 layers")
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("prompt.max_tokens must be a positive integer")
    chunk = cfg.get("qwen3_vl", {}).get("prefill_chunk_length")
    if chunk is not None and (type(chunk) is not int or chunk <= 0):
        raise ValueError("qwen3_vl.prefill_chunk_length must be a positive integer")
    group = cfg.get("text_layers_per_graph")
    if type(group) is not int or group <= 0 or layers % group:
        raise ValueError("text_layers_per_graph must divide the used language prefix")
    if not isinstance(cfg.get("calibration_prompt"), str) or not cfg["calibration_prompt"].strip():
        raise ValueError("Sol-H3 export requires a real calibration prompt")
    from .precision import H3PrecisionPolicy

    cfg["h3_precision"] = H3PrecisionPolicy(**cfg.get("h3_precision", {})).to_dict()
    return cfg


class BundleExporter:
    def __init__(self, assets, root, config, device, conditioner_checkpoint=None):
        self.conditioner_checkpoint = conditioner_checkpoint
        import torch

        from .conditioning import AUDIO_SIGMAS, REFINER_SIGMAS, VIDEO_SIGMAS
        from .precision import ATTENTION_VALUE_DIVISOR, H3PrecisionPolicy

        self.assets, self.root, self.cfg, self.device = assets, Path(root), config, torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Sol-H3 SEFP weight conversion requires a CUDA device")
        torch.cuda.set_device(self.device)
        self.sessions = {}
        self.branch = None
        self.root.mkdir(parents=True, exist_ok=False)
        self.meta = dict(
            schema_version=9,
            kind="sol_h3",
            complete=False,
            target_device="XH2a",
            input_dtype="float16",
            attention_value_divisor=ATTENTION_VALUE_DIVISOR,
            scaled_rms_policy=H3PrecisionPolicy(**config.get("h3_precision", {})).scaled_rms_policy,
            quant_type=config["model"]["quant_scheme"]["quant_type"],
            draft_steps=4,
            refiner_steps=3,
            graphs={},
            resources={},
            tokenizer_files=[],
            feature_graphs=[],
            visual_graphs=[],
            upscaler_graphs=[],
            video_decoder_graphs=[],
            h3_sigmas=VIDEO_SIGMAS,
            audio_sigmas=AUDIO_SIGMAS,
            refiner_sigmas=REFINER_SIGMAS,
            text_feature_divisor=16.0,
            text_refiner_input_divisor=16.0,
            text_rms_policy="plain_wide_reduction",
            refiner_cache_policy="per_sigma_exact_fixed_prompt",
            ltx_audio_policy=dict(
                input_name="audio_mode", dtype="int32", shape=[1], modes=dict(joint=0, frozen=1), default="frozen"
            ),
            attention_contract=dict(mask_input_index=4, min_flash_kv_length=512),
            norm_fp16=config.get("norm_fp16", False),
            ltx_reference_conditioning=deepcopy(config.get("ltx_reference_conditioning", {"enabled": False})),
        )
        self.meta.update(
            {
                name: config[name]
                for name in (
                    "profile",
                    "draft_size",
                    "draft_frames",
                    "refiner_size",
                    "refiner_frames",
                    "fps",
                    "text_capacity",
                    "text_layers",
                )
            }
        )

    def progress(self, stage):
        (self.root / "build_progress.json").write_text(
            json.dumps(dict(stage=stage, graphs=list(self.meta["graphs"])), indent=2)
        )

    def resource(self, name, value, *, extension="json"):
        import numpy as np

        directory = self.root / "hf_config"
        directory.mkdir(exist_ok=True)
        path = directory / f"{name}.{extension}"
        if extension == "json":
            path.write_text(json.dumps(value, indent=2) + "\n")
        elif extension == "npy":
            np.save(path, value, allow_pickle=False)
        elif extension == "npz":
            np.savez(path, **value)
        else:
            raise ValueError("unsupported Sol-H3 runtime resource kind")
        self.meta["resources"][name] = path.relative_to(self.root).as_posix()

    def graph(
        self,
        name,
        directory,
        module,
        inputs,
        names,
        outputs,
        *,
        dynamo=False,
        quant_scheme=None,
        conv3d_mode=None,
        conv3d_fastpath=None,
    ):
        import torch
        from torch.utils._pytree import tree_map

        from .export import export_graph

        norm_fp16 = self.cfg.get("norm_fp16", False) and name in ("h3_denoiser", "ltx_denoiser")
        physical = self.graph_name(name)
        if physical != name:
            directory += "_" + self.branch
        name = physical
        if name in self.meta["graphs"]:
            raise ValueError(f"Sol-H3 component {name} was already exported")
        path = self.root / directory / f"{name}.onnx"
        export_graph(
            module.cpu().float(),
            tree_map(lambda value: value.detach().cpu(), inputs),
            path,
            names,
            outputs,
            quant_scheme=quant_scheme or self.cfg["model"]["quant_scheme"],
            dynamo=dynamo,
            norm_fp16=norm_fp16,
            conv3d_mode=conv3d_mode,
            conv3d_fastpath=conv3d_fastpath,
        )
        self.meta["graphs"][name] = dict(
            path=path.relative_to(self.root).as_posix(),
            input_names=names,
            output_names=outputs,
            quant_type=(quant_scheme or self.cfg["model"]["quant_scheme"])["quant_type"],
            norm_fp16=norm_fp16,
        )
        if conv3d_mode is not None:
            self.meta["graphs"][name]["conv3d_to_conv2d_mode"] = conv3d_mode
        if conv3d_fastpath is not None:
            self.meta["graphs"][name]["conv3d_to_conv2d_fastpath"] = conv3d_fastpath
        self.progress(name)
        torch.cuda.empty_cache()
        return path

    def graph_name(self, name):
        from .bundle import BRANCH_GRAPHS

        return f"{name}_{self.branch}" if self.branch is not None and name in BRANCH_GRAPHS else name

    def infer(self, name, values):
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

        from .precision import require_finite

        name = self.graph_name(name)
        path = self.root / self.meta["graphs"][name]["path"]
        if name not in self.sessions:
            self.sessions[name] = HMONNXInferenceV2(
                str(path), session_config=HMONNXInferenceConfig(exec_devices=[str(self.device)])
            )
        session = self.sessions[name]
        inputs = {
            key: value.to(device=self.device, dtype=session.get_input(key).dtype)
            for key, value in zip(session.get_input_names(), values, strict=True)
        }
        require_finite(tuple(inputs.values()), name=f"{name} FP16 calibration inputs")
        output = session.run(inputs)
        result = output if isinstance(output, (list, tuple)) else (output,)
        require_finite(result, name=name)
        return tuple(value.detach().cpu() for value in result)

    def close(self):
        import torch

        self.sessions.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def reference_vae(self, calibration_pixels, *, base_image_shape=None, image_shapes=(), video_shapes=()):
        """Export every native reference tile gear from real normalized media."""
        from torch.nn import functional as F

        from .reference_geometry import image_gears, video_gear
        from .reference_vae import ReferenceVAEAdapter, load_reference_vae

        if (
            calibration_pixels.ndim != 5
            or calibration_pixels.shape[:2] != (1, 3)
            or any(n <= 0 for n in calibration_pixels.shape[2:])
        ):
            raise ValueError("reference VAE calibration requires normalized [1, 3, T, H, W] media")
        source, config = load_reference_vae(self.assets["h3_model"])
        if self.meta["kind"] == "sol_h3":
            self._reference_vae_geometry = (
                source.temporal_compression_ratio,
                source.spatial_compression_ratio,
                source.config.latent_channels,
            )
        options = self.cfg.get("reference_vae", {})
        tile_size, overlap = options.get("tile_size", [768, 768]), options.get("overlap", [64, 64])
        planner = ReferenceVAEAdapter(config, None, tile_size=tile_size, overlap=overlap)
        refs = self.cfg.get("references", {})
        fixed_images = image_gears(refs.get("image", {}))
        fixed_video = video_gear(refs.get("video", {}))
        fixed = fixed_images is not None or fixed_video is not None
        if fixed:
            if (image_shapes and fixed_images is None) or (video_shapes and fixed_video is None):
                raise ValueError("fixed reference VAE profiles require explicit gears for every media kind")
            if any(tuple(shape) not in (fixed_images or ()) for shape in image_shapes) or any(
                tuple(shape) != fixed_video for shape in video_shapes
            ):
                raise ValueError("reference VAE calibration geometries must match configured fixed gears")
            image_tiles = planner.tile_shapes(image_shapes)
            video_tiles = planner.tile_shapes(video_shapes, video=True)
            expected_images = {(1, *shape) for shape in image_shapes}
            expected_videos = {(source.config.clip_length, *shape) for shape in video_shapes}
            if set(image_tiles) != expected_images or set(video_tiles) != expected_videos:
                raise ValueError("fixed reference VAE geometries must fit without spatial tiling")
            gears = sorted(expected_images | expected_videos)
        else:
            gears = sorted(set(planner.tile_shapes(image_shapes) + planner.tile_shapes(video_shapes, video=True)))
        if not gears and base_image_shape is None:
            raise ValueError("reference VAE export requires at least one image or video geometry")
        self.resource("reference_vae_config", config)
        self.meta["reference_vae_policy"] = dict(tile_size=list(tile_size), overlap=list(overlap))
        self.meta["reference_vae_layout"] = "fixed_v1" if fixed else "split_v1"
        self.meta["reference_vae_tiles"] = {}
        if base_image_shape is not None:
            from copy import deepcopy

            from .h3_vae_wrappers import MiniMaxH3VAEEncoderTileWrapper, wrap_vae_encoder

            shape = (1, *base_image_shape)
            planner.tile_shapes([base_image_shape])  # Validate native latent alignment.
            plain = deepcopy(source)
            plain = wrap_vae_encoder(plain)
            module = MiniMaxH3VAEEncoderTileWrapper(plain.encoder, plain.quant_conv)
            name = f"h3_reference_encode_base_t1_h{shape[1]}_w{shape[2]}"
            sample = F.interpolate(
                calibration_pixels[:, :, :1].detach().cpu().float(), size=shape, mode="trilinear", align_corners=False
            )
            self.graph(
                name,
                name.replace("h3_reference_encode", "vae_encode_h3_reference"),
                module,
                (sample,),
                ["pixels"],
                ["moments"],
                dynamo=True,
            )
            self.meta["reference_vae_base"] = dict(graph=name, shape=shape, spatial_padding_mode="reflect")
            self.close()
        if fixed:
            from copy import deepcopy

            from .h3_vae_wrappers import MiniMaxH3VAEEncoderTileWrapper, wrap_vae_encoder

            for shape in gears:
                base = self.meta.get("reference_vae_base")
                if base is not None and tuple(base["shape"]) == shape:
                    name = base["graph"]
                else:
                    frames, height, width = shape
                    plain = wrap_vae_encoder(deepcopy(source))
                    module = MiniMaxH3VAEEncoderTileWrapper(plain.encoder, plain.quant_conv)
                    sample = calibration_pixels.detach().cpu().float()
                    if frames == 1:
                        sample = sample[:, :, :1]
                    sample = F.interpolate(sample, size=shape, mode="trilinear", align_corners=False)
                    suffix = f"t{frames}_h{height}_w{width}"
                    name = f"h3_reference_encode_ref_{suffix}"
                    self.graph(
                        name, f"vae_encode_h3_reference_ref_{suffix}", module, (sample,), ["pixels"], ["moments"],
                        dynamo=True,
                    )
                    self.close()
                self.meta["reference_vae_tiles"]["x".join(map(str, shape))] = name
            return
        from .masked_vae import MaskedReferenceEncodeGraph, reference_mask_inputs, reference_mask_names

        # Ref images and videos each share a masked spatial canvas. Ordinary
        # first/last-frame images use the separate full-frame graph above.
        canvases = [
            tuple(max(shape[axis] for shape in gears if shape[0] == frames) for axis in range(3))
            for frames in sorted({shape[0] for shape in gears})
        ]
        self.meta["reference_vae_masks"] = {}
        for maximum in canvases:
            frames, height, width = maximum
            module = MaskedReferenceEncodeGraph(source)
            sample = calibration_pixels.detach().cpu().float()
            if frames == 1:
                sample = sample[:, :, :1]
            sample = F.interpolate(sample, size=maximum, mode="trilinear", align_corners=False)
            policy = dict(shape=maximum, levels=module.levels, spatial_padding_mode=module.spatial_padding_mode)
            suffix = f"t{frames}_h{height}_w{width}" if frames == 1 else f"t{frames}"
            name = f"h3_reference_encode_ref_{suffix}"
            self.graph(
                name,
                f"vae_encode_h3_reference_ref_{suffix}",
                module,
                reference_mask_inputs(sample, **policy),
                reference_mask_names(module.levels, module.spatial_padding_mode),
                ["moments"],
                dynamo=True,
            )
            self.meta["reference_vae_masks"][name] = policy
            self.meta["reference_vae_tiles"].update(
                {
                    "x".join(map(str, shape)): name
                    for shape in gears
                    if shape[0] == frames
                }
            )
            self.close()

    def vision(self, pixel_values, grid_thw, *, additional_calibrations=()):
        """Export processor-derived vision gears using real fitting media groups."""
        import torch
        from torch.nn import functional as F

        from ..qwen_image21.vision_graph import KEY_MASK_VISION_INPUT_NAMES, KeyMaskedVision, geometry_inputs
        from .vision import load_vision_source

        capacities = self.meta["capacity_plan"]["qwen_visual_patch_gears"]
        if not capacities or capacities != sorted(set(capacities)):
            raise ValueError("vision export requires increasing patch capacity gears")
        samples = []
        for pixels, grids in ((pixel_values, grid_thw), *additional_calibrations):
            if grids.ndim != 2 or grids.shape[1] != 3 or grids.shape[0] == 0 or grids.is_floating_point():
                raise ValueError("vision calibration requires native [media, 3] grids")
            if any(
                t <= 0 or h <= 0 or w <= 0 or h % 2 or w % 2 or h * w > capacities[-1]
                for t, h, w in grids.detach().cpu().tolist()
            ):
                raise ValueError("vision calibration grids must fit the exported patch capacity")
            if pixels.shape != (int(grids.prod(-1).sum()), 1536):
                raise ValueError("vision calibration pixels do not match their native grids")
            offset = 0
            for temporal, height, width in grids.detach().cpu().tolist():
                count = height * width
                samples.append((height, width, pixels[offset : offset + count]))
                offset += temporal * count
        source, config = load_vision_source(self.assets["h3_model"])
        self.resource("qwen_vision_config", config)
        self.meta["visual_graphs"] = []
        for capacity in capacities:
            fitting = [sample for sample in samples if sample[0] * sample[1] <= capacity]
            if not fitting:
                raise ValueError(f"vision gear {capacity} needs real calibration media that fits its patch capacity")
            height, width, pixels = max(fitting, key=lambda sample: sample[0] * sample[1])
            geometry = geometry_inputs(
                (height, width),
                capacity,
                source.num_grid_per_side,
                config["hidden_size"] // config["num_heads"],
                device="cpu",
                dtype=torch.float32,
                key_mask_only=True,
            )
            pixels = F.pad(pixels.detach().cpu().float(), (0, 0, 0, capacity - pixels.shape[0]))[None]
            name = f"h3_visual_p{capacity}"
            self.graph(
                name,
                f"visual_qwen3_vl_p{capacity}",
                KeyMaskedVision(source, capacity),
                (pixels, *geometry),
                KEY_MASK_VISION_INPUT_NAMES,
                ["image_features", "deepstack_0", "deepstack_1", "deepstack_2"],
            )
            self.meta["visual_graphs"].append(dict(graph=name, capacity=capacity))
            self.close()

    def reference_audio(self, waveform=None):
        import math

        import torch
        from torch.nn import functional as F

        from .reference_audio import ReferenceAudioAdapter, ReferenceAudioGraph, load_reference_audio

        config = json.loads((self.assets["h3_model"] / "audio_vae/config.json").read_text())
        hop = math.prod(config["encoder_rates"])
        if hop != 800 or config["sampling_rate"] != 32000:
            raise ValueError("Sol-H3 reference audio requires the native 32kHz / 800-sample encoder")
        seconds = self.cfg.get("references", {}).get("audio", {}).get("max_duration_seconds", 5)
        if not self.cfg.get("references", {}).get("independent_duration", False):
            seconds = min(seconds, self.cfg["draft_frames"] / 24)
        frames = math.ceil(seconds * 40)
        self.resource("reference_audio_config", config)
        self.meta["reference_audio_frames"] = frames
        if self.meta["capacity_plan"]["reference_audio_tokens"]:
            if waveform is None or waveform.ndim != 3 or waveform.shape[:2] != (2, 1):
                raise ValueError("reference audio capacity requires a real stereo calibration soundtrack")
            count = math.ceil(waveform.shape[-1] / hop)
            if not 0 < count <= frames:
                raise ValueError("reference audio calibration exceeds configured capacity")
            sample = F.pad(waveform.detach().cpu().float(), (0, frames * hop - waveform.shape[-1]))
            mask = (torch.arange(frames) < count).float().reshape(1, 1, frames)
            source, _ = load_reference_audio(self.assets["h3_model"])
            self.graph(
                "h3_reference_audio_encode",
                f"vae_encode_h3_reference_audio_f{frames}",
                ReferenceAudioGraph(source, frames),
                (sample, mask),
                ["waveform", "frame_mask"],
                ["mean"],
                dynamo=True,
            )
            self.close()

        def encode(waveform, mask):
            if self.meta["kind"] == "sol_h3":
                return (
                    _example_latent((2, len(config["latents_mean"]), frames), seed=self.cfg["calibration_seed"]) * mask
                )
            return self.infer("h3_reference_audio_encode", (waveform, mask))[0]

        return ReferenceAudioAdapter(config, encode, max_frames=frames)

    def text(self, request=None, *, processor=None, vae=None, audio_vae=None):
        import torch
        from transformers import AutoTokenizer

        from .feature_graph import TextFeatureStack, load_text_source
        from .request import tokenize_prompt

        divisor = self.meta["text_feature_divisor"]

        self.progress("text_source")
        source, config = load_text_source(
            self.assets["h3_model"], layers=self.cfg["text_layers"], quantized_dir=self.conditioner_checkpoint
        )
        feature_scheme = None
        if self.conditioner_checkpoint is not None:
            feature_scheme = dict(target_device="XH2a", quant_type="w4a8h0_ssfp")
            self.resource(
                "conditioner_quantization",
                json.loads((Path(self.conditioner_checkpoint) / "qwen3_vl_feature_quantization.json").read_text()),
            )
        self.resource("text_config", config)
        self.resource(
            "text_embedding",
            (source.embed_tokens.weight.detach().float() / divisor).half().numpy(),
            extension="npy",
        )
        self.resource("text_rope_inv_freq", source.rotary_emb.inv_freq.detach().float().numpy(), extension="npy")
        tokenizer = AutoTokenizer.from_pretrained(self.assets["h3_model"] / "tokenizer", local_files_only=True)
        directory = self.root / "tokenizer"
        tokenizer.save_pretrained(directory)
        self.meta["tokenizer_files"] = [
            path.relative_to(self.root).as_posix() for path in sorted(directory.iterdir()) if path.is_file()
        ]
        capacity = self.cfg["text_capacity"]
        if request is not None:
            if processor is None or not self.meta["visual_graphs"]:
                raise ValueError("conditioned text export requires its native processor and exported vision graph")
            from .conditioner import H3FeatureEncoder
            from .request import encode_request
            from .vision import VisionFeatureAdapter

            chunk = self.cfg.get("qwen3_vl", {}).get("prefill_chunk_length", 256)
            full_config = json.loads((self.assets["h3_model"] / "text_encoder/config.json").read_text())
            self.resource("qwen_config", full_config)
            vision = VisionFeatureAdapter(
                (
                    (
                        lambda name, *values: _example_vision_features(
                            values[0], full_config["vision_config"], seed=self.cfg["calibration_seed"]
                        )
                    )
                    if self.meta["kind"] == "sol_h3"
                    else lambda name, *values: self.infer(name, values)
                ),
                full_config["vision_config"],
                gears=self.meta["visual_graphs"],
                device="cpu" if self.meta["kind"] == "sol_h3" else self.device,
            )

            def run_features(hidden, cos, sin, additions, source=source):
                return self._cached_text(
                    source,
                    config,
                    hidden,
                    cos,
                    sin,
                    additions=additions,
                    chunk=chunk,
                    capacity=capacity,
                    divisor=divisor,
                    quant_scheme=feature_scheme,
                )

            encoder = H3FeatureEncoder(
                full_config,
                source.embed_tokens.weight,
                vision,
                run_features,
                feature_layer=self.cfg["text_layers"],
                divisor=divisor,
            )
            features, _ = encode_request(
                request,
                text_encoder=encoder,
                tokenizer=tokenizer,
                processor=processor,
                vae=vae,
                audio_vae=audio_vae,
                text_encoder_layer=self.cfg["text_layers"],
            )
            self.text_length = features.shape[1]
            self.text_features = torch.nn.functional.pad(
                features.float() / divisor, (0, 0, 0, capacity - self.text_length)
            )
            self.request = request
            self.close()
            return
        ids = tokenize_prompt(
            self.cfg["calibration_prompt"], tokenizer, self.cfg.get("prompt", {}).get("max_tokens", capacity)
        )
        hidden = torch.zeros(1, capacity, config["hidden_size"])
        hidden[:, : len(ids)] = source.embed_tokens(torch.tensor([ids])).float() / divisor
        positions = torch.ones(capacity, dtype=torch.float32)
        positions[: len(ids)] = torch.arange(len(ids), dtype=torch.float32)
        phase = positions[:, None] * source.rotary_emb.inv_freq.detach().float()[None]
        phase = torch.cat((phase, phase), -1)[None]
        cos, sin = phase.cos(), phase.sin()
        valid = torch.tensor([len(ids)], dtype=torch.int32)
        chunk = self.cfg.get("qwen3_vl", {}).get("prefill_chunk_length")
        if chunk is not None:
            self._cached_text(
                source,
                config,
                hidden[:, : len(ids)],
                cos[:, : len(ids)],
                sin[:, : len(ids)],
                chunk=chunk,
                capacity=capacity,
                divisor=divisor,
                quant_scheme=feature_scheme,
            )
            return
        group = self.cfg["text_layers_per_graph"]
        for start in range(0, self.cfg["text_layers"], group):
            name = f"h3_feature_{start:02d}_{start + group:02d}"
            module = TextFeatureStack(source.layers[start : start + group], sdk_ops=True, divisor=divisor)
            self.graph(
                name,
                f"text_encode_h3_s{capacity}",
                module,
                (hidden, cos, sin, valid),
                ["hidden", "cos", "sin", "valid_length"],
                ["features"],
                quant_scheme=feature_scheme,
            )
            self.meta["feature_graphs"].append(name)
            hidden = (
                _example_latent(hidden.shape, seed=self.cfg["calibration_seed"] + start) / divisor
                if self.meta["kind"] == "sol_h3"
                else self.infer(name, (hidden, cos, sin, valid))[0].float()
            )
            self.close()
            del module
        del source
        gc.collect()
        self.text_features, self.text_length = hidden, len(ids)

    def _cached_text(self, source, config, hidden, cos, sin, *, chunk, capacity, divisor, quant_scheme, additions=None):
        import torch
        from torch.nn import functional as F

        from xhmodel_merak.exec_backends import CacheTensor

        from .feature_graph import CachedMultimodalTextFeatureStack
        from .prefill import prefill_feature_group

        head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        cache_capacity = self.meta["capacity_plan"]["kv_capacity"]
        cache_shape = [1, config["num_key_value_heads"], cache_capacity, head_dim]
        self.meta["text_prefill_chunk_length"] = chunk
        self.meta["text_cache_shape"] = cache_shape
        group = self.cfg["text_layers_per_graph"]
        length = hidden.shape[1]
        if not 0 < length <= capacity:
            raise ValueError("Qwen presentation exceeds the exported text capacity")
        if additions is None:
            additions = [torch.zeros_like(hidden) for _ in range(3)]
        if len(additions) != 3 or any(value.shape != hidden.shape for value in additions):
            raise ValueError("Qwen export requires three dense DeepStack increments")
        for start in range(0, self.cfg["text_layers"], group):
            name = f"h3_feature_{start:02d}_{start + group:02d}"
            module = CachedMultimodalTextFeatureStack(
                source.layers[start : start + group], start_layer=start, divisor=divisor
            )
            count = min(chunk, length)

            def first(value, count=count):
                return F.pad(value[:, :count], (0, 0, 0, chunk - count))

            caches = [[CacheTensor(torch.zeros(cache_shape)) for _ in range(group)] for _ in range(2)]
            inputs = (
                first(hidden),
                first(cos),
                first(sin),
                torch.tensor([count], dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                *(first(value) for value in additions),
                *caches,
            )
            names = [
                "hidden",
                "cos",
                "sin",
                "valid_length",
                "past_length",
                "deepstack_0",
                "deepstack_1",
                "deepstack_2",
                *[f"key_cache_{i}" for i in range(group)],
                *[f"value_cache_{i}" for i in range(group)],
            ]
            self.graph(
                name,
                f"text_encode_h3_c{chunk}_s{capacity}",
                module,
                inputs,
                names,
                ["features"],
                quant_scheme=quant_scheme,
            )
            self.meta["graphs"][name]["text_layers"] = [start, start + group]
            self.meta["feature_graphs"].append(name)
            if self.meta["kind"] == "sol_h3":
                hidden = _example_latent(hidden.shape, seed=self.cfg["calibration_seed"] + start) / divisor
            else:
                hidden = prefill_feature_group(
                    lambda *values, graph=name: self.infer(graph, values)[0],
                    hidden,
                    cos,
                    sin,
                    additions,
                    chunk_length=chunk,
                    cache_shape=cache_shape,
                    num_layers=group,
                    device=self.device,
                ).float()
            self.close()
            del module, caches, inputs
        self.text_features = F.pad(hidden, (0, 0, 0, capacity - length))
        self.text_length = length
        return hidden

    def draft(self):
        import torch
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

        from .conditioning import ConditionedH3Schedule, conditioned_tables, fold_h3_conditioning
        from .hybrid import merge_dense_adapter, merge_reference_adapter

        self.progress("h3_source")
        source = (
            MiniMaxH3Transformer3DModel.from_pretrained(
                self.assets["h3_model"] / ("transformer_ref" if self.cfg.get("mode") == "ref2va" else "transformer"),
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
        )
        if self.cfg.get("mode") == "ref2va":
            self.meta["h3_adapter"] = merge_reference_adapter(source, self.assets["ref2va_lora"])
        else:
            self.meta["h3_adapter"] = merge_dense_adapter(source, self.assets["dense_lora"], compute_device=self.device)
        # Every Sol-H3 mode uses the same five-role ABI. T2VA leaves reference
        # spans empty; its live rows select exactly the same native time rows.
        schedule = ConditionedH3Schedule()
        folded = fold_h3_conditioning(source, schedule, compute_device=self.device)
        tables, output_table = conditioned_tables(folded, schedule)
        self.meta["folded_h3_projection_parameters"] = folded.original_projection_parameters
        self._export_h3_draft(source, tables, output_table, schedule=schedule)

    def _export_h3_draft(self, source, tables, output_table, *, schedule=None, adapter=None, variant=None):
        """Shared H3 graphs and calibration; an optional adapter keeps A/B separate."""
        from copy import copy, deepcopy
        from dataclasses import replace

        import torch
        from torch import nn

        from .duration import gear_for_draft
        from .h3_graph import H3TextGraph, PackedConditionedH3DenoiserGraph, RolePartitionedH3DenoiserGraph
        from .host import ConditionedDraft, RolePartitionedDraft, unpatch_h3
        from .precision import H3PrecisionPolicy

        gear = gear_for_draft(self.cfg["draft_frames"])

        for block in source.transformer_blocks:
            block.adaln_proj = nn.Identity()
        source.norm_out.linear = nn.Identity()
        policy = H3PrecisionPolicy(**self.cfg["h3_precision"])
        self.meta["h3_precision"] = policy.to_dict()
        bindings = []

        def bind(graph, component):
            graph.cpu().float()
            if adapter is not None:
                from .native_lora import bind_native_lora

                bindings.append(bind_native_lora(graph, adapter, component=component, policy=policy))
            return graph

        suffix = "" if variant is None else "_" + variant
        text = bind(H3TextGraph(source, policy, sdk_ops=True, input_divisor=self.meta["text_feature_divisor"]), "text")
        key_mask = torch.full((1, 1, 1, self.cfg["text_capacity"]), -65504.0)
        key_mask[..., : self.text_length] = 0
        text_inputs = (self.text_features, key_mask)
        self.graph(
            "h3_text_refiner",
            f"text_encode_h3_refine_s{self.cfg['text_capacity']}{suffix}",
            text,
            text_inputs,
            ["text_features", "key_mask"],
            ["refined_text"],
        )
        fixed_roles = self.meta["kind"] == "sol_h3"
        if fixed_roles:
            refined = _example_latent(
                (1, self.text_features.shape[1], source.context_embedder.out_features),
                seed=self.cfg["calibration_seed"],
            )
            refined[:, self.text_length :] = 0
        else:
            refined = self.infer("h3_text_refiner", text_inputs)[0]
        self.close()
        del text
        plan = self.meta["capacity_plan"]
        rope = dict(rope_freq_dim=source.config.rope_freq_dim, rope_theta=source.config.rope_theta)
        self.meta["h3_rope_config"], self.meta["h3_packed_inputs"] = rope, True
        options = dict(seed=self.cfg["calibration_seed"], capacities=plan, rope_config=rope)
        draft_type = RolePartitionedDraft if fixed_roles else ConditionedDraft
        graph_type = RolePartitionedH3DenoiserGraph if fixed_roles else PackedConditionedH3DenoiserGraph
        graph_options = {}
        if fixed_roles:
            from .role_layout import role_capacities_from_plan

            role_capacities = getattr(self, "role_capacities", role_capacities_from_plan(plan))
            options.update(role_capacities=role_capacities, attention_heads=source.config.num_attention_heads)
            graph_options["role_capacities"] = role_capacities
        if hasattr(self, "request"):
            # Native preparation mutates reference latents. Each variant needs
            # fresh state, while encoder modules retain their shared weights.
            request = replace(
                self.request, state=deepcopy(self.request.state), components=copy(self.request.components)
            )
            draft = draft_type(request, **options)
        else:
            draft = draft_type.from_text(
                size=self.cfg["draft_size"], text_length=self.text_length,
                video_frames=gear.h3_video_frames, audio_frames=gear.h3_audio_frames, **options
            )
        if fixed_roles:
            self.meta["h3_role_layout"] = draft.layout.metadata()
        graph = bind(
            graph_type(
                source,
                tables,
                output_table,
                plan["video_capacity"],
                plan["audio_capacity"],
                policy,
                sdk_ops=True,
                **graph_options,
            ),
            "dit",
        )
        if adapter is not None and sum(item["target_count"] for item in bindings) != len(adapter.pairs):
            raise ValueError("H3 adapter was not bound completely across text and DiT graphs")
        size = self.cfg["draft_size"]
        self.graph(
            "h3_denoiser",
            f"denoise_h3_{size[1]}x{size[0]}x{gear.draft_frames}{suffix}",
            graph,
            draft.inputs(refined),
            (
                [
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
                if fixed_roles
                else [
                    "video_tokens",
                    "audio_tokens",
                    "text_features",
                    "step",
                    "cos",
                    "sin",
                    "valid_length",
                    "row_roles",
                    "pack",
                    "unpack",
                ]
            ),
            ["video_velocity", "audio_velocity"],
        )
        del graph
        gc.collect()
        if fixed_roles:
            # SEFP RTN derives weight exponents from the weights themselves.
            # Downstream bridge/codecs need finite export examples, while real
            # denoising belongs to the separate published-package validation.
            seed = self.cfg["calibration_seed"]
            self.h3_latent = _example_latent((1, 24, gear.h3_video_frames, size[0] // 16, size[1] // 16), seed=seed)
            self.audio_latent = _example_latent((2, 32, gear.h3_audio_frames), seed=seed + 1)
        else:
            # Native H3 and its switchable LoRA variants keep their own flow.
            video, audio = draft.run(lambda *values: self.infer("h3_denoiser", values), refined, schedule=schedule)
            self.h3_latent = unpatch_h3(video, size=size, frames=gear.h3_video_frames)
            self.audio_latent = audio.reshape(2, gear.h3_audio_frames, 32).permute(0, 2, 1).contiguous()
        self.close()
        return dict(
            graphs={name: self.graph_name(name) for name in ("h3_text_refiner", "h3_denoiser")},
            h3_rope_config=rope,
            h3_packed_inputs=True,
            draft_steps=4 if schedule is None else schedule.steps,
            lora=None if adapter is None else dict(**adapter.to_dict(), bindings=bindings),
        )

    def bridge(self):
        from . import source_io
        from .bridge_graph import AdapterGraph, UpscalerSegment, upscaler_segments
        from .duration import duration_gear

        gear = duration_gear(self.cfg["draft_frames"], self.cfg["refiner_frames"], self.cfg["fps"])

        self.progress("bridge_source")
        upscaler, receipt = source_io.upscaler(self.assets, self.device)
        self.resource(
            "upscaler_statistics",
            dict(
                mean=upscaler.comfy_latents_mean.flatten().float().cpu().tolist(),
                std=upscaler.comfy_latents_std.flatten().float().cpu().tolist(),
            ),
        )
        upscaler.cpu()
        segments = upscaler_segments(gear.h3_video_frames)
        module = UpscalerSegment(upscaler).eval()
        for frames in sorted({plan["input_end"] - plan["input_start"] for plan in segments}):
            name = f"h3_upscale_t{frames}"
            self.graph(
                name,
                f"visual_h3_upscale_t{frames}",
                module,
                (
                    _example_latent(
                        (1, 24, frames, self.cfg["draft_size"][0] // 16, self.cfg["draft_size"][1] // 16),
                        seed=self.cfg["calibration_seed"],
                    ),
                ),
                ["latent"],
                ["upscaled"],
                conv3d_mode=self.cfg["upscaler_conv3d_mode"],
                conv3d_fastpath=self.cfg["upscaler_conv3d_fastpath"],
            )
            self.meta["upscaler_graphs"].append(name)
        self.meta["upscaler_segments"] = segments
        del module, upscaler
        gc.collect()
        size = self.cfg["refiner_size"]
        highres = _example_latent(
            (1, 24, gear.h3_video_frames, size[0] // 16, size[1] // 16), seed=self.cfg["calibration_seed"]
        )
        adapter = source_io.adapter(self.assets)
        module = AdapterGraph(
            adapter, pixel_frames=gear.draft_frames, target_frames=gear.ltx_video_frames, sdk_ops=True
        )
        self.graph("h3_to_ltx", "visual_h3_to_ltx", module, (highres.float(),), ["upscaled_h3"], ["ltx_latent"])
        del module, adapter
        gc.collect()
        self.ltx_latent = _example_latent(
            (1, 128, gear.ltx_video_frames, size[0] // 32, size[1] // 32), seed=self.cfg["calibration_seed"]
        )

    def h3_audio(self):
        """Export the shared native audio decoder; Native retains its replay."""
        import torch
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio

        from ..minimax_h3.audio_vae_decoder_wrapper import MiniMaxH3AudioVAEDecoderWrapper

        self.progress("audio_source")
        vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
            self.assets["h3_model"] / "audio_vae", torch_dtype=torch.bfloat16, local_files_only=True
        )
        config = {name: list(vae.config[name]) for name in ("latents_mean", "latents_std")}
        self.resource("h3_audio_config", config)
        raw = (
            self.audio_latent.float() * torch.tensor(config["latents_std"])[None, :, None]
            + torch.tensor(config["latents_mean"])[None, :, None]
        )
        module = MiniMaxH3AudioVAEDecoderWrapper(vae)
        pcm_shape = (raw.shape[0], 1, raw.shape[-1] * vae.hop_length)
        self.graph(
            "h3_audio_decode", f"vae_decode_h3_audio_f{raw.shape[-1]}", module, (raw,), ["latents"], ["pcm"],
            dynamo=True,
        )
        del module, vae
        gc.collect()
        self.audio_pcm = (
            _example_latent(pcm_shape, seed=self.cfg["calibration_seed"])
            if self.meta["kind"] == "sol_h3"
            else self.infer("h3_audio_decode", (raw,))[0]
        )

    def audio(self):
        from . import source_io
        from .audio_graph import AudioEncodeGraph
        from .duration import duration_gear
        from .host import audio_mel

        self.h3_audio()
        encoder = source_io.audio_encoder(self.assets)
        settings = {
            "sample_rate": encoder.sample_rate,
            "mel_bins": encoder.mel_bins,
            "mel_hop_length": encoder.mel_hop_length,
            "n_fft": encoder.n_fft,
        }
        self.resource("ltx_audio_frontend", settings)
        mel = audio_mel(self.audio_pcm, source_rate=32000, config=settings)
        self.graph(
            "ltx_audio_encode",
            f"vae_encode_ltx_audio_t{mel.shape[2]}",
            AudioEncodeGraph(encoder),
            (mel.float(),),
            ["spectrogram"],
            ["audio_latent"],
            dynamo=True,
        )
        del encoder
        gc.collect()
        gear = duration_gear(self.cfg["draft_frames"], self.cfg["refiner_frames"], self.cfg["fps"])
        self.ltx_audio = _example_latent((1, 8, gear.ltx_audio_frames, 16), seed=self.cfg["calibration_seed"])

    def ltx_image_encoder(self):
        """One FP16 T=1 encoder, released before loading the large refiner."""
        import torch

        from . import source_io
        from .ltx_keyframes import prepare_keyframe_pixels
        from .vae_graph import VideoEncodeGraph

        size = self.cfg["refiner_size"]
        encoder, crf = source_io.video_encoder(self.assets)
        self.meta["ltx_reference_conditioning"]["image_crf"] = crf
        request = getattr(self, "request", None)
        media = (
            calibration_inputs(self.assets.root, self.cfg)
            if request is not None and request.mode in ("i2va", "l2va", "fl2va")
            else {}
        )
        images = []
        if request is not None and request.mode in ("i2va", "l2va", "fl2va"):
            for name, index in (("image", 0), ("last_image", self.cfg["refiner_frames"] - 1)):
                image = media.get(name)
                if image is None:
                    image = request.state.get(name)
                if image is not None:
                    images.append((index, prepare_keyframe_pixels(image, size=size, crf=crf)))
        # W16A16 has no activation calibration scales. A request without images
        # still exports the shared encoder using a bounded, nonconstant canvas.
        pixels = (
            images[0][1]
            if images
            else torch.linspace(-1, 1, size[1]).reshape(1, 1, 1, 1, -1).expand(1, 3, 1, *size).contiguous()
        )
        module = VideoEncodeGraph(
            encoder,
            residual_scale=32.0,
            normalization_epsilon=self.meta["ltx_reference_conditioning"]["encoder_norm_eps"],
        )
        scheme = {**self.cfg["model"]["quant_scheme"], "quant_type": "w16a16_sefp"}
        self.graph(
            "ltx_image_encode",
            f"vae_encode_ltx_image_t1_h{size[0]}_w{size[1]}",
            module,
            (pixels,),
            ["pixels"],
            ["latent"],
            dynamo=True,
            quant_scheme=scheme,
        )
        del module, encoder
        gc.collect()
        latents = [
            (
                index,
                _example_latent((1, 128, 1, size[0] // 32, size[1] // 32), seed=self.cfg["calibration_seed"] + index),
            )
            for index, _ in images
        ]
        self.close()
        return latents

    def refiner(self):
        import torch

        from . import source_io
        from .duration import duration_gear
        from .ltx_graph import LTXDenoiserGraph, LTXInputRoPEGraph, fold_ltx_block

        size = self.cfg["refiner_size"]
        gear = duration_gear(self.cfg["draft_frames"], self.cfg["refiner_frames"], self.cfg["fps"])
        enabled = self.cfg.get("ltx_reference_conditioning", {}).get("enabled", True)
        frame_tokens = (size[0] // 32) * (size[1] // 32)
        self.meta["ltx_reference_conditioning"] = dict(
            enabled=enabled,
            require_native_resolution=self.cfg.get("ltx_reference_conditioning", {}).get(
                "require_native_resolution", False
            ),
            graph_mode="static_maximum_guide" if enabled else "target_only",
            first_frame_tokens=frame_tokens,
            target_tokens=gear.ltx_video_frames * frame_tokens,
            video_tokens=(gear.ltx_video_frames + int(enabled)) * frame_tokens,
            end_pixel_frame=gear.refiner_frames - 1,
            encoder_residual_scale=32.0,
            encoder_norm_eps=self.cfg.get("ltx_reference_conditioning", {}).get("encoder_norm_eps", 2**-24),
        )
        keyframes = self.ltx_image_encoder() if enabled else []

        self.progress("ltx_source")
        (
            source,
            cache,
            video_args,
            audio_args,
            rope,
            keyframes_mask,
            joint_video_args,
            joint_audio_args,
        ) = source_io.refiner(self.assets, self.cfg, self.ltx_latent, self.ltx_audio)
        self.meta["refiner_prompt_fingerprint"] = cache["fingerprint"]
        self.resource(
            "refiner_prompt_recipe", {key: cache[key] for key in ("prompt", "recipe", "fingerprint", "prompt_stats")}
        )
        self.meta["refiner_lora_strength"] = 0.8
        # Keep RoPE as shared inputs: constant folding otherwise duplicates
        # the expanded positional grids in every layer's Q/K rotation.
        self.meta["ltx_rope_mode"] = "inputs"
        rope = tuple(value.detach().cpu() for value in rope)
        self.resource("ltx_rope", {f"rope_{i}": value.numpy() for i, value in enumerate(rope)}, extension="npz")
        tables = [
            fold_ltx_block(
                block, video_args, audio_args, joint_video_args=joint_video_args, joint_audio_args=joint_audio_args
            )
            for block in source.transformer_blocks
        ]
        # Native LTX marks the target's first frame even without a supplied
        # image. Keep that learned positional marker independent of time masks.
        graph = LTXDenoiserGraph(
            source,
            tables,
            video_args,
            audio_args,
            sdk_ops=True,
            keyframes_mask=keyframes_mask,
            first_frame_tokens=frame_tokens if enabled else None,
            joint_audio_args=joint_audio_args,
        ).cpu()
        if not enabled:
            graph = LTXInputRoPEGraph(graph)
        del source, tables, video_args, audio_args, joint_video_args, joint_audio_args
        gc.collect()
        # These are trace inputs, not a denoising replay. Keep the maximum
        # canvas and dynamic controls, with an unused guide represented by zero.
        target_tokens = gear.ltx_video_frames * frame_tokens
        video = _example_latent(
            (1, (gear.ltx_video_frames + int(enabled)) * frame_tokens, self.ltx_latent.shape[1]),
            seed=self.cfg["calibration_seed"],
        )
        audio = _example_latent(
            (1, self.ltx_audio.shape[2], self.ltx_audio.shape[1] * self.ltx_audio.shape[3]),
            seed=self.cfg["calibration_seed"] + 1,
        )
        extra = ()
        if enabled:
            supplied = {index for index, _ in keyframes}
            if gear.refiner_frames - 1 not in supplied:
                video[:, target_tokens:] = 0
            extra = (
                torch.tensor([int(0 in supplied)], dtype=torch.int32),
                torch.tensor(
                    [target_tokens + (frame_tokens if gear.refiner_frames - 1 in supplied else 0)], dtype=torch.int32
                ),
            )
        inputs = (
            video, audio, torch.tensor([0], dtype=torch.int32), *rope, *extra, torch.tensor([1], dtype=torch.int32)
        )
        self.graph(
            "ltx_denoiser",
            f"denoise_ltx_{size[1]}x{size[0]}_v{(gear.ltx_video_frames + int(enabled)) * frame_tokens}",
            graph,
            inputs,
            ["video", "audio", "step", "vcos", "vsin", "acos", "asin", "vccos", "vcsin", "accos", "acsin"]
            + (["first_role", "video_valid_length"] if enabled else [])
            + ["audio_mode"],
            ["video_velocity", "audio_velocity"],
        )
        del graph
        gc.collect()
        # The decoder consumes normalized target latents, never the appended
        # reference guide. Its RTN export does not require a generated video.
        self.refined_latent = _example_latent(
            (1, 128, gear.ltx_video_frames, size[0] // 32, size[1] // 32), seed=self.cfg["calibration_seed"]
        )
        self.close()

    def video_vae(self):
        from . import source_io
        from .vae_graph import MaskedVideoDecodeGraph, VideoDecodeGraph, decoder_mask_inputs

        self.progress("video_vae_source")
        decoder, native, complementary = source_io.video_decoder(self.assets, self.refined_latent, self.cfg)
        self.meta["decoder_complementary_masks"] = complementary
        plans, masks = [], {}
        shared = self.meta.get("mode") == "all"
        module = MaskedVideoDecodeGraph(decoder) if shared else VideoDecodeGraph(decoder)
        if shared:
            shapes = [tuple(self.refined_latent[tile.in_coords].shape[2:]) for tile in native]
            temporal_gears = sorted({shape[0] for shape in shapes})
            maxima = {
                frames: tuple(max(shape[axis] for shape in shapes if shape[0] == frames) for axis in range(3))
                for frames in temporal_gears
            }
            if len(temporal_gears) == 1:
                self.meta["video_decoder_mask"] = dict(shape=maxima[temporal_gears[0]], scales=module.mask_scales)
            else:
                self.meta["video_decoder_masks"] = {
                    f"ltx_video_decode_t{frames}": dict(shape=maxima[frames], scales=module.mask_scales)
                    for frames in temporal_gears
                }
        for i, tile in enumerate(native):
            sample = self.refined_latent[tile.in_coords].float()
            shape = tuple(sample.shape)
            name = (
                ("ltx_video_decode" if len(temporal_gears) == 1 else f"ltx_video_decode_t{shape[2]}")
                if shared else "ltx_video_decode_" + "x".join(map(str, shape[2:]))
            )
            if name not in self.meta["graphs"]:
                args = decoder_mask_inputs(sample, maxima[shape[2]], module.mask_scales) if shared else (sample,)
                inputs = ["latent", *[f"spatial_mask_{j}" for j in range(len(args) - 1)]]
                self.graph(
                    name,
                    (
                        ("vae_decode_ltx_video" if len(temporal_gears) == 1 else f"vae_decode_ltx_video_t{shape[2]}")
                        if shared else "vae_decode_ltx_video_" + "x".join(map(str, shape[2:]))
                    ),
                    module,
                    args,
                    inputs,
                    ["pixels"],
                    dynamo=True,
                )
                self.meta["video_decoder_graphs"].append(name)

            def coords(values):
                return [[value.start, value.stop, value.step] for value in values]

            plans.append(dict(graph=name, input=coords(tile.in_coords), output=coords(tile.out_coords)))
            for j, mask in enumerate(tile.masks_1d):
                masks[f"tile_{i}_axis_{j}"] = mask.cpu().float().numpy()
        self.meta["decoder_tiles"] = plans
        self.resource("decoder_masks", masks, extension="npz")
        del module, decoder
        gc.collect()


def calibration_inputs(model_dir, config):
    """Explicit YAML media wins; public defaults read the mode's local calibration media."""
    if "calibration_inputs" in config:
        if not isinstance(config["calibration_inputs"], dict):
            raise ValueError("calibration_inputs must be an object containing the mode's media")
        return config["calibration_inputs"]
    path = Path(model_dir) / "calibration_inputs.json"
    mode = config.get("mode", "t2va")
    if not path.is_file():
        raise ValueError(f"{mode} export requires calibration_inputs in YAML or {path}")
    inventory = json.loads(path.read_text())
    inputs = inventory.get(mode) if isinstance(inventory, dict) else None
    if not isinstance(inputs, dict):
        raise ValueError(f"{path} must contain an object for {mode}")
    # Media paths in this inventory are relative to the inventory, not the caller's cwd.
    inputs = deepcopy(inputs)
    for key in ("image", "last_image"):
        if key in inputs:
            inputs[key] = str(path.parent / inputs[key])
    for reference in inputs.get("references", []):
        reference["path"] = str(path.parent / reference["path"])
    return inputs


def build_bundle(model_dir, output_dir, config, *, device="cuda:0", conditioner_checkpoint=None):
    from .export import validate_conv3d_mode

    validate_conv3d_mode(
        config.get("upscaler_conv3d_mode", "slice"), fastpath=config.get("upscaler_conv3d_fastpath", False)
    )
    if config.get("mode") == "all":
        from .unified_build import build_unified_bundle

        return build_unified_bundle(
            model_dir, output_dir, config, device=device, conditioner_checkpoint=conditioner_checkpoint
        )
    from .artifacts import export_entries, read_metadata

    assets, cfg = SourceAssets.read(model_dir), validate_build_config(config)
    requested = cfg.get("conditioner_quant_type")
    actual = "w4a8h0_ssfp" if conditioner_checkpoint is not None else cfg["model"]["quant_scheme"]["quant_type"]
    if requested is not None and requested != actual:
        raise ValueError(f"conditioner_quant_type={requested} requires matching weights; got {actual}")
    from transformers import AutoProcessor, AutoTokenizer

    from .capacity import configure_processor, plan_capacities
    from .request import prepare_configured_request, prepare_encoder_calibration

    if cfg.get("mode") == "ref2va":
        if (
            "ref2va_lora" not in assets.paths
            or not assets["ref2va_lora"].is_file()
            or not (assets["h3_model"] / "transformer_ref/config.json").is_file()
        ):
            raise ValueError("Ref2VA requires transformer_ref weights and ref2va_lora in the source inventory")

    text_config = json.loads((assets["h3_model"] / "text_encoder/config.json").read_text())["text_config"]
    tokenizer = AutoTokenizer.from_pretrained(assets["h3_model"] / "tokenizer", local_files_only=True)
    processor, request = None, None
    if cfg.get("mode", "t2va") != "t2va":
        processor = configure_processor(
            AutoProcessor.from_pretrained(assets["h3_model"] / "processor", local_files_only=True),
            cfg.get("qwen3_vl", {}),
        )
        request = prepare_configured_request(
            cfg["calibration_prompt"], config=cfg, tokenizer=tokenizer, **calibration_inputs(model_dir, cfg)
        )
    capacity_plan = plan_capacities(
        cfg, tokenizer=tokenizer, processor=processor, max_position_embeddings=text_config["max_position_embeddings"]
    )
    cfg["text_capacity"] = capacity_plan["text_capacity"]
    calibration = prepare_encoder_calibration(request, processor, cfg) if request is not None else None
    assets.activate_build_sources()
    exporter = BundleExporter(assets, output_dir, cfg, device, conditioner_checkpoint)
    exporter.meta["capacity_plan"] = capacity_plan
    exporter.meta["prompt_max_tokens"] = capacity_plan["prompt_max_tokens"]
    exporter.meta["mode"] = cfg.get("mode", "t2va")
    if request is not None:
        from .reference_vae import ReferenceVAEAdapter

        exporter.resource(
            "input_config",
            {
                key: cfg[key]
                for key in ("mode", "prompt", "draft_size", "draft_frames", "references", "qwen3_vl")
                if key in cfg
            },
        )
        processor.save_pretrained(exporter.root / "processor")
        exporter.meta["processor_files"] = [
            path.relative_to(exporter.root).as_posix()
            for path in sorted((exporter.root / "processor").iterdir())
            if path.is_file()
        ]
        exporter.vision(
            calibration["visual_pixels"],
            calibration["grids"],
            additional_calibrations=calibration["visual_calibrations"][1:],
        )
        exporter.reference_vae(
            calibration["pixels"],
            base_image_shape=tuple(cfg["draft_size"]) if request.mode != "ref2va" else None,
            image_shapes=calibration["image_shapes"] if request.mode == "ref2va" else (),
            video_shapes=calibration["video_shapes"] if request.mode == "ref2va" else (),
        )
        vae_config = json.loads((exporter.root / exporter.meta["resources"]["reference_vae_config"]).read_text())

        def encode_tile(shape, value):
            from .reference_vae import invoke_reference_tile

            return invoke_reference_tile(
                exporter.meta,
                shape,
                value,
                lambda name, *args: _example_reference_moments(
                    args[0].shape[2:], exporter._reference_vae_geometry, seed=cfg["calibration_seed"]
                ),
            )

        vae = ReferenceVAEAdapter(
            vae_config, encode_tile, untiled=request.mode != "ref2va", **exporter.meta["reference_vae_policy"]
        )
        audio_vae = exporter.reference_audio(calibration["waveform"]) if request.mode == "ref2va" else None

        def text_stage():
            return exporter.text(request, processor=processor, vae=vae, audio_vae=audio_vae)

    else:
        text_stage = exporter.text
    for stage in (text_stage, exporter.draft, exporter.bridge, exporter.audio, exporter.refiner, exporter.video_vae):
        try:
            stage()
        finally:
            exporter.close()
    exporter.meta["complete"] = True
    (exporter.root / "runtime_meta_info.json").write_text(json.dumps(exporter.meta, indent=2) + "\n")
    meta = read_metadata(exporter.root)
    (exporter.root / "export_meta_info.json").write_text(json.dumps(export_entries(meta), indent=2) + "\n")
    exporter.progress("complete")
    (exporter.root / "build_progress.json").unlink()
    return meta
