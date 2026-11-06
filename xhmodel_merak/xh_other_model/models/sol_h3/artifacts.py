# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Complete-bundle inventory, shared by direct workflow and public publication."""

from __future__ import annotations

import json
from pathlib import Path


def _validate_role_layout(meta):
    from .role_layout import ROLE_LAYOUT_FORMAT, ROLE_NAMES, role_capacities_from_plan, role_spans

    layout = meta.get("h3_role_layout")
    if layout is None:
        return None
    if not isinstance(layout, dict) or layout.get("format") != ROLE_LAYOUT_FORMAT:
        raise ValueError("unknown H3 role layout format")
    caps, block = layout.get("role_capacities"), layout.get("attention_block_size")
    if not isinstance(caps, list) or block not in (64, 128, 256, 512):
        raise ValueError("invalid H3 role capacity or attention block size")
    spans = role_spans(caps)
    if (
        any(n % block for n in caps)
        or layout.get("role_names") != list(ROLE_NAMES)
        or layout.get("role_spans") != [list(x) for x in spans]
        or layout.get("sequence_capacity") != sum(caps)
        or type(layout.get("attention_heads")) is not int
        or layout["attention_heads"] <= 0
        or layout.get("attention_mode") != "broadcast_key_mask"
    ):
        raise ValueError("H3 fixed role spans, capacity and attention metadata disagree")
    minimum = role_capacities_from_plan(meta["capacity_plan"], block_size=block)
    if any(actual < required for actual, required in zip(caps, minimum, strict=True)):
        raise ValueError("H3 role capacity does not cover the mode's native token budget")
    return layout


def _validate_ltx_reference_policy(meta):
    policy = meta.get("ltx_reference_conditioning")
    if policy is None:
        # Legacy component fixtures remain useful independent of new build defaults.
        return None
    if not isinstance(policy, dict) or type(policy.get("enabled")) is not bool:
        raise ValueError("LTX reference conditioning requires an explicit enabled boolean")
    require_native = policy.get("require_native_resolution", False)
    if type(require_native) is not bool or (require_native and not policy["enabled"]):
        raise ValueError("require_native_resolution must be a boolean and requires enabled LTX image conditioning")
    size = meta["refiner_size"]
    frames = (size[0] // 32) * (size[1] // 32)
    target_frames = (meta.get("refiner_frames", 121) - 1) // 8 + 1
    maximum = policy.get("graph_mode") == "static_maximum_guide"
    if (
        policy.get("graph_mode") not in ("static_maximum_guide", "target_only")
        or policy["enabled"] != maximum
        or policy.get("first_frame_tokens") != frames
        or policy.get("target_tokens") != target_frames * frames
        or policy.get("video_tokens") != (target_frames + int(maximum)) * frames
        or policy.get("end_pixel_frame") != meta.get("refiner_frames", 121) - 1
        or policy.get("encoder_residual_scale") != 32.0
    ):
        raise ValueError("LTX reference policy disagrees with the fixed target/guide geometry")
    if maximum and (type(policy.get("image_crf")) is not int or not 0 <= policy["image_crf"] <= 51):
        raise ValueError("LTX reference encoder must publish its checkpoint image CRF")
    epsilon = policy.get("encoder_norm_eps")
    if epsilon is not None:
        import math

        if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon < 2**-24:
            raise ValueError("LTX encoder_norm_eps must be None or finite and >= 2^-24")
    return policy


def _npz_shapes(path):
    """Read NPY headers without allocating the large per-head rotary grids."""
    import zipfile

    import numpy as np

    shapes = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                raise ValueError("LTX RoPE resource must contain only NPY arrays")
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                else:
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            if dtype not in (np.dtype("float32"), np.dtype("float16")):
                raise ValueError("LTX rotary factors must be floating point")
            shapes[name[:-4]] = list(shape)
    return shapes


def artifact_path(root, value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Sol-H3 resources require nonempty POSIX relative paths")
    relative = Path(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("Sol-H3 resource path escapes the bundle")
    path = Path(root)
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Sol-H3 published resources cannot be symlinks")
    if not path.is_file():
        raise ValueError(f"missing Sol-H3 resource: {value}")
    return path


def export_entries(meta):
    entries = {name: {"path": value["path"], "type": "hmonnx"} for name, value in meta["graphs"].items()}
    for name, value in meta["resources"].items():
        if value.endswith(".json"):
            entries[name] = {"path": value, "type": "json"}
        elif name == "text_embedding":
            entries[name] = {"path": value, "type": "embedding"}
    entries["tokenizer"] = {"path": "tokenizer", "type": "tokenizer"}
    if meta.get("processor_files"):
        entries["processor"] = {"path": "processor", "type": "tokenizer"}
        for i, path in enumerate(meta["processor_files"]):
            if path.endswith(".json"):
                entries[f"processor_json_{i}"] = {"path": path, "type": "json"}
    for i, path in enumerate(meta["tokenizer_files"]):
        if path.endswith(".json"):
            entries[f"tokenizer_json_{i}"] = {"path": path, "type": "json"}
    entries["runtime_meta_info"] = {"path": "runtime_meta_info.json", "type": "json"}
    return entries


def read_metadata(root):
    meta = json.loads(artifact_path(root, "runtime_meta_info.json").read_text())
    if meta.get("schema_version") == 5 or (meta.get("schema_version") in (8, 9) and meta.get("mode") == "all"):
        from .bundle import mode_metadata
        from .capacity import MODES, matches_dense_profile
        from .duration import duration_gear

        size = meta.get("draft_size", [])
        gear = duration_gear(meta.get("draft_frames"), meta.get("refiner_frames"), meta.get("fps"))
        if (
            meta.get("mode") != "all"
            or len(size) != 2
            or not matches_dense_profile(meta.get("profile"), size, gear.seconds, "all", require_mode=True)
        ):
            raise ValueError("unified Sol-H3 requires a five-mode profile")
        audits = {}
        views = [mode_metadata(meta, mode) for mode in MODES]
        for field in ("text_capacity", "video_capacity", "audio_capacity", "kv_capacity", "qwen_visual_max_patches"):
            if meta.get("capacity_plan", {}).get(field) != max(view["capacity_plan"][field] for view in views):
                raise ValueError("unified Sol-H3 root capacities must cover the mode profiles exactly")
        if meta.get("text_capacity") != meta["capacity_plan"]["text_capacity"] or meta["capacity_plan"].get(
            "dit_capacity"
        ) != sum(meta["capacity_plan"][key] for key in ("text_capacity", "video_capacity", "audio_capacity")):
            raise ValueError("unified Sol-H3 root dimensions disagree with its capacity plan")
        for view in views:
            for field in ("kv_capacity", "qwen_visual_max_patches", "qwen_visual_patch_gears"):
                if view["capacity_plan"].get(field) != meta["capacity_plan"].get(field):
                    raise ValueError("unified Sol-H3 modes must share their encoder capacities")
            _validate_metadata(root, view, audits=audits)
        return meta
    return _validate_metadata(root, meta)


def _validate_ltx_audio_policy(meta):
    if meta.get("schema_version") == 9:
        policy = meta.get("ltx_audio_policy")
        if (
            not isinstance(policy, dict)
            or set(policy) != {"input_name", "dtype", "shape", "modes", "default"}
            or policy.get("input_name") != "audio_mode"
            or policy.get("dtype") != "int32"
            or policy.get("shape") != [1]
            or type(policy["shape"][0]) is not int
            or not isinstance(policy.get("modes"), dict)
            or set(policy["modes"]) != {"joint", "frozen"}
            or type(policy["modes"]["joint"]) is not int
            or type(policy["modes"]["frozen"]) is not int
            or policy["modes"] != {"joint": 0, "frozen": 1}
            or policy.get("default") != "frozen"
            or "ltx_audio_frozen" in meta
        ):
            raise ValueError("schema-9 requires the dynamic INT32 LTX audio_mode contract; re-export this bundle")
        return
    if "ltx_audio_policy" in meta:
        raise ValueError("dynamic LTX audio policy requires a matching schema-9 graph; re-export this bundle")
    frozen = meta.get("ltx_audio_frozen", False)
    if type(frozen) is not bool or frozen != (meta.get("schema_version") in (6, 7, 8)):
        raise ValueError("frozen LTX audio requires matching schema-6/schema-7/schema-8 export and runtime semantics")
    return frozen


def resolve_ltx_audio_mode(meta, mode=None):
    """Resolve a request against the graph's declared, or legacy fixed, audio policy."""
    _validate_ltx_audio_policy(meta)
    if mode is not None and (not isinstance(mode, str) or mode not in ("joint", "frozen")):
        raise ValueError("ltx_audio_mode must be 'joint' or 'frozen'")
    policy = meta.get("ltx_audio_policy")
    if policy is not None:
        return policy["default"] if mode is None else mode
    fixed = "frozen" if meta.get("ltx_audio_frozen", False) else "joint"
    if mode is not None and mode != fixed:
        raise ValueError(f"this Sol-H3 bundle supports only LTX audio mode {fixed!r}; re-export for {mode!r}")
    return fixed


def _validate_metadata(root, meta, *, audits=None):
    from .duration import duration_gear
    from .precision import ATTENTION_VALUE_DIVISOR, H3PrecisionPolicy

    if (
        meta.get("schema_version") not in (3, 4, 8, 9)
        or meta.get("kind") != "sol_h3"
        or meta.get("complete") is not True
    ):
        raise ValueError(
            "Sol-H3 requires a completed compatible bundle; re-export incompatible bundles"
        )
    _validate_ltx_audio_policy(meta)
    if meta["schema_version"] in (8, 9) and meta.get("attention_contract") != dict(
        mask_input_index=4, min_flash_kv_length=512
    ):
        raise ValueError("schema-8/9 requires compiler mask-slot-4 and minimum KV 512; re-export this bundle")
    if meta["schema_version"] in (4, 8, 9) and (
        meta.get("capacity_plan", {}).get("token_alignment") != 256 or meta.get("h3_packed_inputs") is not True
    ):
        raise ValueError(
            f"schema-{meta['schema_version']} Sol-H3 requires aligned capacities and packed attention inputs"
        )
    if meta.get("attention_value_divisor") != ATTENTION_VALUE_DIVISOR:
        raise ValueError("Sol-H3 requires compensated long-sequence PV accumulation")
    policy = H3PrecisionPolicy(**meta["h3_precision"])
    if meta.get("scaled_rms_policy") != policy.scaled_rms_policy:
        raise ValueError("Sol-H3 scaled RMS policy disagrees with h3_precision")
    if meta.get("target_device") != "XH2a" or meta.get("input_dtype") != "float16":
        raise ValueError("Sol-H3 requires XH2a static FP16 graph interfaces")
    if meta.get("quant_type") not in ("w8a8_sefp", "w16a16_sefp"):
        raise ValueError("Sol-H3 requires an explicitly published SEFP precision")
    if meta.get("draft_steps") != 4 or meta.get("refiner_steps") != 3:
        raise ValueError("Sol-H3 requires four draft updates and three refiner updates")
    divisor = meta.get("text_feature_divisor", 256.0)
    if divisor not in (16.0, 256.0) or meta.get("text_refiner_input_divisor", divisor) != divisor:
        raise ValueError("Sol-H3 text feature and refiner input divisors must match")
    if meta.get("draft_attention", "dense") != "dense" or meta.get("refiner_attention", "dense") != "dense":
        raise ValueError("this release supports only Dense Sol-H3")
    gear = duration_gear(meta.get("draft_frames"), meta.get("refiner_frames"), meta.get("fps"))
    from .capacity import matches_dense_profile

    size = meta.get("draft_size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(n) is not int or n <= 0 or n % 32 for n in size)
        or meta.get("refiner_size") != [n * 2 for n in size]
        or not matches_dense_profile(meta.get("profile"), size, gear.seconds, meta.get("mode", "t2va"))
    ):
        raise ValueError("unexpected Sol-H3 deployment canvas")
    role_layout = _validate_role_layout(meta)
    ltx_policy = _validate_ltx_reference_policy(meta)
    if meta["schema_version"] == 9 and ltx_policy is None:
        raise ValueError("schema-9 requires the LTX reference and audio graph contracts")
    from .conditioning import AUDIO_SIGMAS, REFINER_SIGMAS, VIDEO_SIGMAS

    for field, expected in (
        ("h3_sigmas", VIDEO_SIGMAS),
        ("audio_sigmas", AUDIO_SIGMAS),
        ("refiner_sigmas", REFINER_SIGMAS),
    ):
        if meta.get(field) != list(expected):
            raise ValueError(f"Sol-H3 {field} disagrees with its folded step tables")
    if type(meta.get("text_capacity")) is not int or meta["text_capacity"] <= 0:
        raise ValueError("Sol-H3 text capacity is invalid")
    if "capacity_plan" in meta:
        plan = meta["capacity_plan"]
        if (
            plan.get("text_capacity") != meta["text_capacity"]
            or plan.get("prompt_max_tokens") != meta.get("prompt_max_tokens")
            or plan.get("dit_capacity")
            != sum(plan.get(key, 0) for key in ("text_capacity", "video_capacity", "audio_capacity"))
        ):
            raise ValueError("Sol-H3 capacity plan disagrees with published dimensions")
        if "token_alignment" in plan:
            if plan["token_alignment"] != 256 or any(
                type(plan.get(key)) is not int or plan[key] % 256
                for key in (
                    "text_capacity",
                    "kv_capacity",
                    "video_capacity",
                    "audio_capacity",
                    "dit_capacity",
                    "qwen_visual_max_patches",
                )
            ):
                raise ValueError("Sol-H3 exported capacities must be aligned to 256")
            if not meta.get("h3_packed_inputs"):
                raise ValueError("aligned Sol-H3 capacities require packed attention inputs")
    chunk = meta.get("text_prefill_chunk_length")
    if chunk is not None:
        shape = meta.get("text_cache_shape")
        if (
            type(chunk) is not int
            or chunk <= 0
            or not isinstance(shape, list)
            or len(shape) != 4
            or any(type(n) is not int or n <= 0 for n in shape)
            or shape[0] != 1
            or shape[2]
            != meta.get("capacity_plan", {}).get("kv_capacity", ((meta["text_capacity"] + chunk - 1) // chunk) * chunk)
            or shape[2] < meta["text_capacity"]
            or shape[2] % chunk
        ):
            raise ValueError("Sol-H3 cached text prefill has invalid chunk/cache dimensions")
        cursor = 0
        for name in meta.get("feature_graphs", []):
            span = meta.get("graphs", {}).get(name, {}).get("text_layers")
            if (
                not isinstance(span, list)
                or len(span) != 2
                or any(type(n) is not int for n in span)
                or span[0] != cursor
                or span[1] <= cursor
            ):
                raise ValueError("Sol-H3 feature graphs must cover a contiguous language prefix")
            cursor = span[1]
        if cursor != meta.get("text_layers"):
            raise ValueError("Sol-H3 feature graph prefix differs from text_layers")
    required_resources = {
        "text_config",
        "text_embedding",
        "text_rope_inv_freq",
        "h3_audio_config",
        "ltx_audio_frontend",
        "upscaler_statistics",
        "refiner_prompt_recipe",
        "decoder_masks",
    }
    rope_mode = meta.get("ltx_rope_mode", "inputs")
    if rope_mode not in ("inputs", "constant"):
        raise ValueError("unknown LTX RoPE mode")
    if rope_mode == "inputs":
        required_resources.add("ltx_rope")
    if not required_resources.issubset(meta.get("resources", {})):
        raise ValueError("Sol-H3 bundle is missing runtime preprocessing resources")
    required = {"h3_text_refiner", "h3_denoiser", "h3_to_ltx", "h3_audio_decode", "ltx_audio_encode", "ltx_denoiser"}
    if ltx_policy is not None and ltx_policy["enabled"]:
        required.add("ltx_image_encode")
    if (
        not required.issubset(meta["graphs"])
        or not meta.get("feature_graphs")
        or not meta.get("upscaler_graphs")
        or not meta.get("video_decoder_graphs")
    ):
        raise ValueError("Sol-H3 bundle is missing learned pipeline components")
    if rope_mode == "constant":
        expected_ltx_names = ["video", "audio", "step"]
        if meta["schema_version"] == 9:
            expected_ltx_names.append("audio_mode")
        if meta["graphs"]["ltx_denoiser"]["input_names"] != expected_ltx_names:
            raise ValueError("constant LTX RoPE graph inputs disagree with its audio policy")
    gears = meta.get("visual_graphs", [])
    if not isinstance(gears, list) or any(
        not isinstance(gear, dict)
        or set(gear) != {"graph", "capacity"}
        or not isinstance(gear["graph"], str)
        or not gear["graph"]
        or type(gear["capacity"]) is not int
        or gear["capacity"] <= 0
        or gear["capacity"] % 4
        for gear in gears
    ):
        raise ValueError("Sol-H3 visual graph gears require graph names and positive merged patch capacities")
    capacities = [gear["capacity"] for gear in gears]
    visual_capacities = {gear["graph"]: gear["capacity"] for gear in gears}
    plan = meta.get("capacity_plan", {})
    if (
        capacities != sorted(set(capacities))
        or len(visual_capacities) != len(gears)
        or capacities != plan.get("qwen_visual_patch_gears", [])
        or (capacities and capacities[-1] != plan.get("qwen_visual_max_patches"))
        or (plan.get("token_alignment") == 256 and any(n % 256 for n in capacities))
        or "h3_visual" in meta["graphs"]
        or set(visual_capacities) != {name for name in meta["graphs"] if name.startswith("h3_visual_")}
    ):
        raise ValueError("Sol-H3 visual graph gears disagree with the capacity plan; re-export")
    if gears and "qwen_vision_config" not in meta["resources"]:
        raise ValueError("Sol-H3 vision gears require their native vision configuration")
    vision_config = (
        json.loads(artifact_path(root, meta["resources"]["qwen_vision_config"]).read_text()) if gears else None
    )
    mode = meta.get("mode", "t2va")
    if mode not in ("t2va", "i2va", "l2va", "fl2va", "ref2va"):
        raise ValueError("unsupported Sol-H3 artifact mode")
    if mode != "t2va":
        if (
            not {"qwen_config", "qwen_vision_config", "reference_vae_config", "input_config"}.issubset(
                meta["resources"]
            )
            or not gears
            or not meta.get("processor_files")
            or not (meta.get("reference_vae_tiles") if mode == "ref2va" else meta.get("reference_vae_base"))
            or not meta.get("h3_rope_config")
            or chunk is None
            or "capacity_plan" not in meta
        ):
            raise ValueError("conditioned Sol-H3 artifact is missing encoder resources or layout configuration")
        inputs = json.loads(artifact_path(root, meta["resources"]["input_config"]).read_text())
        if inputs.get("mode") != mode or meta["capacity_plan"].get("mode") != mode:
            raise ValueError("conditioned Sol-H3 input mode disagrees with its capacity plan")
    if mode != "t2va" or meta.get("h3_packed_inputs", False):
        if not meta.get("h3_rope_config") or "capacity_plan" not in meta:
            raise ValueError("packed Sol-H3 requires its RoPE and capacity configuration")
        names = [
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
        if role_layout is not None:
            names = [
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
        if meta["graphs"]["h3_denoiser"]["input_names"] != names:
            raise ValueError("Sol-H3 requires packed attention inputs")
    if mode == "ref2va":
        frames = meta.get("reference_audio_frames")
        if (
            "reference_audio_config" not in meta["resources"]
            or type(frames) is not int
            or frames <= 0
            or meta.get("h3_adapter", {}).get("format") != "peft"
        ):
            raise ValueError("Ref2VA requires its reference audio configuration and fused adapter identity")
        if meta["capacity_plan"]["reference_audio_tokens"] and "h3_reference_audio_encode" not in meta["graphs"]:
            raise ValueError("Ref2VA audio capacity requires an exported reference audio encoder")
    for names in (meta["feature_graphs"], meta["upscaler_graphs"], meta["video_decoder_graphs"]):
        if any(name not in meta["graphs"] for name in names):
            raise ValueError("Sol-H3 stage references an absent graph")
    reference_shapes = {}
    reference_masks = meta.get("reference_vae_masks")
    if "reference_vae_mask" in meta:
        raise ValueError("Sol-H3 legacy reference VAE bundles are unsupported; re-export with per-graph masks")
    if any(
        key in meta
        for key in ("reference_vae_layout", "reference_vae_tiles", "reference_vae_base", "reference_vae_masks")
    ):
        layout = meta.get("reference_vae_layout")
        if (
            layout not in ("split_v1", "fixed_v1")
            or (layout == "split_v1" and not isinstance(reference_masks, dict))
            or (layout == "fixed_v1" and "reference_vae_masks" in meta)
            or not isinstance(meta.get("reference_vae_tiles"), dict)
            or ("reference_vae_base" in meta and not isinstance(meta["reference_vae_base"], dict))
        ):
            raise ValueError("Sol-H3 reference VAE requires a supported tile layout; re-export the bundle")
        if "reference_vae_config" not in meta["resources"] or not (
            meta.get("reference_vae_tiles") or meta.get("reference_vae_base")
        ):
            raise ValueError("Sol-H3 reference VAE requires its config and tile graphs")
        policy = meta.get("reference_vae_policy", {})
        tile, overlap = policy.get("tile_size", []), policy.get("overlap", [])
        if (
            len(tile) != 2
            or len(overlap) != 2
            or any(type(n) is not int or n <= 0 for n in tile)
            or any(type(n) is not int or n < 0 for n in overlap)
            or any(o >= t for o, t in zip(overlap, tile, strict=True))
        ):
            raise ValueError("Sol-H3 reference VAE tile policy is invalid")
        for shape, name in meta["reference_vae_tiles"].items():
            parts = shape.split("x")
            if len(parts) != 3 or any(not n.isdecimal() or int(n) <= 0 for n in parts) or name not in meta["graphs"]:
                raise ValueError("Sol-H3 reference VAE tile mapping is invalid")
            geometry = [1, 3, *map(int, parts)]
            if layout == "fixed_v1" and name in reference_shapes and reference_shapes[name] != geometry:
                raise ValueError("Sol-H3 fixed reference VAE graph cannot represent multiple geometries")
            reference_shapes[name] = geometry
        if reference_masks is not None:
            if set(reference_masks) != set(reference_shapes):
                raise ValueError("Sol-H3 reference VAE mask policies must cover the tile graphs exactly")
            for name, mask in reference_masks.items():
                if not isinstance(mask, dict):
                    raise ValueError("Sol-H3 shared reference VAE mask policy is invalid")
                shape, levels = mask.get("shape", []), mask.get("levels", [])
                if (
                    len(shape) != 3
                    or any(type(n) is not int or n <= 0 for n in shape)
                    or not levels
                    or levels[0] != [1, 1]
                    or levels[-1] != [4, 16]
                    or mask.get("spatial_padding_mode", "reflect") not in ("reflect", "constant")
                    or any(len(level) != 2 or any(type(n) is not int or n <= 0 for n in level) for level in levels)
                    or any(
                        any(a > b for a, b in zip(map(int, key.split("x")), shape, strict=True))
                        or int(key.split("x")[0]) != shape[0]
                        for key, graph in meta["reference_vae_tiles"].items()
                        if graph == name
                    )
                ):
                    raise ValueError("Sol-H3 shared reference VAE mask policy is invalid")
                reference_shapes[name] = [1, 3, *shape]
    base_reference = meta.get("reference_vae_base")
    if base_reference is not None:
        shape = base_reference.get("shape", [])
        name = base_reference.get("graph")
        if (
            len(shape) != 3
            or shape[0] != 1
            or any(type(n) is not int or n <= 0 for n in shape)
            or shape[1:] != list(meta["draft_size"])
            or name not in meta["graphs"]
            or (
                name in reference_shapes
                and (meta.get("reference_vae_layout") != "fixed_v1" or reference_shapes[name] != [1, 3, *shape])
            )
            or base_reference.get("spatial_padding_mode") not in ("reflect", "constant")
        ):
            raise ValueError("Sol-H3 base reference VAE requires its exact full-frame geometry")
        reference_shapes[name] = [1, 3, *shape]
    reference_masks = reference_masks or {}
    for name, graph in meta["graphs"].items():
        path = artifact_path(root, graph["path"])
        if "norm_fp16" in meta:
            expected_norm = meta["norm_fp16"] if name in ("h3_denoiser", "ltx_denoiser") else False
            if type(meta["norm_fp16"]) is not bool or graph.get("norm_fp16", False) != expected_norm:
                raise ValueError("Sol-H3 component normalization precision disagrees with its root policy")
        if not graph.get("input_names") or not graph.get("output_names"):
            raise ValueError("Sol-H3 graph must publish input/output names")
        from .graph_audit import audit_graph

        audit_key = (
            str(path),
            graph.get("quant_type", meta["quant_type"]),
            tuple(graph["input_names"]),
            tuple(graph["output_names"]),
            graph.get("norm_fp16", False),
        )
        if audits is not None and audit_key in audits:
            audit = audits[audit_key]
        else:
            audit = audit_graph(
                path,
                quant_type=audit_key[1],
                inputs=graph["input_names"],
                outputs=graph["output_names"],
                norm_fp16=graph.get("norm_fp16", False),
            )
            if audits is not None:
                audits[audit_key] = audit
        if name in visual_capacities:
            from ..qwen_image21.vision_graph import KEY_MASK_VISION_INPUT_NAMES
            from .graph_audit import validate_external_key_mask

            capacity = visual_capacities[name]
            head_dim = vision_config["hidden_size"] // vision_config["num_heads"]
            expected_inputs = dict(
                zip(
                    KEY_MASK_VISION_INPUT_NAMES,
                    (
                        [1, capacity, 1536],
                        [4, capacity],
                        [4, capacity],
                        [1, 1, capacity, head_dim],
                        [1, 1, capacity, head_dim],
                        [1, 1, 1, capacity],
                    ),
                    strict=True,
                )
            )
            output_names = ["image_features", "deepstack_0", "deepstack_1", "deepstack_2"]
            if (
                graph["input_names"] != KEY_MASK_VISION_INPUT_NAMES
                or graph["output_names"] != output_names
                or audit["input_shapes"] != expected_inputs
                or audit["output_shapes"]
                != {output: [1, capacity // 4, vision_config["out_hidden_size"]] for output in output_names}
                or audit["operators"].get("FlashAttention", 0) != vision_config["depth"] * int(capacity >= 512)
                or audit["operators"].get("MatMul", 0) != 2 * vision_config["depth"] * int(capacity < 512)
                or audit["operators"].get("Softmax", 0) != vision_config["depth"] * int(capacity < 512)
                or audit["operators"].get("MaskedSoftmax", 0)
            ):
                raise ValueError("Sol-H3 vision gear graph disagrees with its capacity or FlashAttention contract")
            validate_external_key_mask(path, audit, capacity)
            import onnx

            for node in onnx.load(path, load_external_data=False).graph.node:
                if node.op_type == "FlashAttention":
                    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                    if (
                        attrs.get("is_causal") != 0
                        or len(node.input) != 5
                        or not node.input[4]
                    ):
                        raise ValueError(
                            "Sol-H3 vision requires noncausal FlashAttention with a broadcast key mask at input 4"
                        )
        if name == "h3_text_refiner" and meta["schema_version"] in (8, 9):
            from .graph_audit import validate_external_key_mask

            capacity = meta["text_capacity"]
            if (
                graph["input_names"] != ["text_features", "key_mask"]
                or graph["output_names"] != ["refined_text"]
                or audit["input_shapes"].get("text_features", [])[:2] != [1, capacity]
                or audit["output_shapes"].get("refined_text", [])[:2] != [1, capacity]
            ):
                raise ValueError("H3 text refiner requires fixed features and an external key mask")
            validate_external_key_mask(path, audit, capacity)
        if name in reference_shapes:
            inputs = ["pixels"]
            if name in reference_masks:
                from .masked_vae import reference_mask_names

                reference_mask = reference_masks[name]
                padding_mode = reference_mask.get("spatial_padding_mode", "reflect")
                inputs = reference_mask_names(reference_mask["levels"], padding_mode)
                frames, height, width = reference_mask["shape"]
                for i, (temporal, spatial) in enumerate(reference_mask["levels"]):
                    expected = {
                        f"valid_mask_{i}": [1, 1, (frames - 1) // temporal + 1, height // spatial, width // spatial],
                    }
                    if padding_mode == "reflect":
                        expected.update(
                            {
                                f"height_indices_{i}": [height // spatial + 2],
                                f"width_indices_{i}": [width // spatial + 2],
                            }
                        )
                    if any(audit["input_shapes"].get(key) != value for key, value in expected.items()):
                        raise ValueError("Sol-H3 reference mask graph disagrees with its valid domains")
            config = json.loads(artifact_path(root, meta["resources"]["reference_vae_config"]).read_text())
            frames, height, width = reference_shapes[name][2:]
            expected_moments = [1, 2 * config["latent_channels"], (frames - 1) // 4 + 1, height // 16, width // 16]
            if audit["output_shapes"].get("moments") != expected_moments:
                raise ValueError("Sol-H3 reference graph disagrees with its posterior shape")
            if (
                graph["input_names"] != inputs
                or graph["output_names"] != ["moments"]
                or audit["input_shapes"].get("pixels") != reference_shapes[name]
            ):
                raise ValueError("Sol-H3 reference VAE graph disagrees with its tile mapping")
        if name in meta["video_decoder_graphs"] and ("video_decoder_mask" in meta or "video_decoder_masks" in meta):
            policies = meta.get("video_decoder_masks")
            if policies is not None:
                if (
                    "video_decoder_mask" in meta
                    or not isinstance(policies, dict)
                    or set(policies) != set(meta["video_decoder_graphs"])
                ):
                    raise ValueError("Sol-H3 temporal decoder masks must cover each decoder graph")
                policy = policies[name]
            else:
                if len(meta["video_decoder_graphs"]) != 1:
                    raise ValueError("Sol-H3 single decoder mask requires exactly one decoder graph")
                policy = meta["video_decoder_mask"]
            frames, height, width = policy["shape"]
            names = ["latent", *[f"spatial_mask_{i}" for i in range(len(policy["scales"]))]]
            if (
                graph["input_names"] != names
                or audit["input_shapes"].get("latent") != [1, 128, frames, height, width]
                or any(
                    audit["input_shapes"].get(f"spatial_mask_{i}") != [1, 1, 1, height * sy, width * sx]
                    for i, (sy, sx) in enumerate(policy["scales"])
                )
            ):
                raise ValueError("Sol-H3 shared video decoder graph disagrees with its mask policy")
        if name == "h3_reference_audio_encode":
            frames = meta.get("reference_audio_frames")
            if (
                graph["input_names"] != ["waveform", "frame_mask"]
                or graph["output_names"] != ["mean"]
                or type(frames) is not int
                or frames <= 0
                or audit["input_shapes"].get("waveform") != [2, 1, frames * 800]
                or audit["input_shapes"].get("frame_mask") != [1, 1, frames]
                or audit["output_shapes"].get("mean") != [2, 32, frames]
            ):
                raise ValueError("Sol-H3 reference audio graph disagrees with its frame capacity")
        if name == "h3_denoiser" and (mode != "t2va" or meta.get("h3_packed_inputs", False)):
            for field, capacity in (
                ("video_tokens", "video_capacity"),
                ("audio_tokens", "audio_capacity"),
                ("text_features", "text_capacity"),
            ):
                if audit["input_shapes"].get(field, [])[:2] != [1, meta["capacity_plan"][capacity]]:
                    raise ValueError("Sol-H3 graph dimensions disagree with its capacity plan")
            if role_layout is not None:
                count = role_layout["sequence_capacity"]
                width = 6 * meta["h3_rope_config"]["rope_freq_dim"]
                shapes = {
                    "step": [1],
                    "cos": [1, 1, count, width],
                    "sin": [1, 1, count, width],
                    "role_pack": [count],
                    "head_unpack": [meta["capacity_plan"]["video_capacity"] + meta["capacity_plan"]["audio_capacity"]],
                    "key_mask": [1, 1, 1, count],
                }
                if any(audit["input_shapes"].get(k) != v for k, v in shapes.items()):
                    raise ValueError("H3 fixed-role graph ABI disagrees with its role layout")
                import onnx

                types = {
                    v.name: v.type.tensor_type.elem_type for v in onnx.load(path, load_external_data=False).graph.input
                }
                if (
                    any(types.get(k) != onnx.TensorProto.INT32 for k in ("step", "role_pack", "head_unpack"))
                    or types.get("key_mask") != onnx.TensorProto.FLOAT16
                ):
                    raise ValueError("H3 role indices require INT32 and its broadcast key mask requires FP16")
        if name == "ltx_image_encode":
            height, width = meta["refiner_size"]
            if (
                ltx_policy is None
                or not ltx_policy["enabled"]
                or graph.get("quant_type") != "w16a16_sefp"
                or graph.get("norm_fp16", False)
                or graph["input_names"] != ["pixels"]
                or graph["output_names"] != ["latent"]
                or audit["input_shapes"] != {"pixels": [1, 3, 1, height, width]}
                or audit["output_shapes"] != {"latent": [1, 128, 1, height // 32, width // 32]}
            ):
                raise ValueError("LTX image encoder must be the shared T1 FP16 residual32 graph")
        if name == "ltx_denoiser" and ltx_policy is not None:
            names = ["video", "audio", "step"]
            rope_names = ["vcos", "vsin", "acos", "asin", "vccos", "vcsin", "accos", "acsin"]
            if rope_mode == "inputs":
                names += rope_names
            if ltx_policy["enabled"]:
                if rope_mode != "inputs":
                    raise ValueError("static LTX guides require the published shared rotary inputs")
                names += ["first_role", "video_valid_length"]
            if meta["schema_version"] == 9:
                names.append("audio_mode")
            expected = {
                "video": [1, ltx_policy["video_tokens"], 128],
                "audio": [1, gear.ltx_audio_frames, 128],
                "step": [1],
            }
            if ltx_policy["enabled"]:
                expected.update(first_role=[1], video_valid_length=[1])
            if meta["schema_version"] == 9:
                expected["audio_mode"] = [1]
            if (
                graph["input_names"] != names
                or graph["output_names"] != ["video_velocity", "audio_velocity"]
                or any(audit["input_shapes"].get(k) != v for k, v in expected.items())
                or audit["output_shapes"] != {"video_velocity": expected["video"], "audio_velocity": expected["audio"]}
            ):
                raise ValueError("LTX DiT inputs/outputs disagree with its fixed guide policy")
            if ltx_policy["enabled"] or meta["schema_version"] == 9:
                import onnx

                model = onnx.load(path, load_external_data=False)
                types = {v.name: v.type.tensor_type.elem_type for v in model.graph.input}
                controls = ["step"]
                if ltx_policy["enabled"]:
                    controls += ["first_role", "video_valid_length"]
                if meta["schema_version"] == 9:
                    controls.append("audio_mode")
                if any(types.get(k) != onnx.TensorProto.INT32 for k in controls):
                    raise ValueError("LTX step, role, effective length and audio mode controls must be INT32")
            if ltx_policy["enabled"]:
                attentions = [n for n in model.graph.node if n.op_type == "FlashAttention"]
                masked = [n for n in attentions if len(n.input) == 5 and n.input[4]]
                matmuls = audit["operators"].get("MatMul", 0)
                cores = len(attentions) + matmuls // 2
                if (
                    not cores
                    or matmuls % 2
                    or cores % 6
                    or len(masked) != 2 * (cores // 6) * int(expected["video"][1] >= 512)
                    or any((len(n.input) > 5 and n.input[5]) or (len(n.input) > 9 and n.input[9]) for n in attentions)
                ):
                    raise ValueError(
                        "LTX guide KV must use the compiler broadcast key-mask interface in both video-KV attentions"
                    )
            if rope_mode == "inputs":
                rope_path = artifact_path(root, meta["resources"]["ltx_rope"])
                key = (str(rope_path), "rope_shapes")
                rope_shapes = audits.get(key) if audits is not None else None
                if rope_shapes is None:
                    rope_shapes = _npz_shapes(rope_path)
                    if audits is not None:
                        audits[key] = rope_shapes
                if (
                    set(rope_shapes) != {f"rope_{i}" for i in range(8)}
                    or any(
                        audit["input_shapes"].get(name) != rope_shapes[f"rope_{i}"] for i, name in enumerate(rope_names)
                    )
                    or any(
                        rope_shapes[f"rope_{i}"][2]
                        != (ltx_policy["video_tokens"] if i in (0, 1, 4, 5) else gear.ltx_audio_frames)
                        for i in range(8)
                    )
                ):
                    raise ValueError("LTX rotary resource and graph disagree on target/guide positions")
        if chunk is not None and "text_layers" in graph:
            start, end = graph["text_layers"]
            for kind in ("key", "value"):
                for index in range(end - start):
                    if audit["input_shapes"].get(f"{kind}_cache_{index}") != meta["text_cache_shape"]:
                        raise ValueError("Sol-H3 text cache graph shape disagrees with metadata")
            if audit["input_shapes"].get("hidden", [0, 0])[1] != chunk:
                raise ValueError("Sol-H3 feature graph chunk length disagrees with metadata")
        if any(kind.startswith("SparseFlashAttention") for kind in audit["operators"]):
            raise ValueError("Dense Sol-H3 cannot contain sparse attention operators")
    for value in meta["resources"].values():
        artifact_path(root, value)
    for value in meta["tokenizer_files"]:
        artifact_path(root, value)
    for value in meta.get("processor_files", []):
        artifact_path(root, value)
    if meta.get("refiner_cache_policy") != "per_sigma_exact_fixed_prompt" or not meta.get("refiner_prompt_fingerprint"):
        raise ValueError("LTX prompt K/V must identify all three sigma-specific caches")
    return meta
