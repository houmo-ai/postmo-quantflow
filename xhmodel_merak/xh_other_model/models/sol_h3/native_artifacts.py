# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native H3 package and adapter views, independent of original source weights."""

import json
import re
from copy import deepcopy

from .artifacts import artifact_path
from .artifacts import export_entries as shared_entries
from .capacity import MODES
from .conditioning import ConditionedH3Schedule
from .precision import H3PrecisionPolicy


def mode_metadata(meta, mode, lora_adapter=None):
    name = "standard" if lora_adapter is None else lora_adapter
    if mode not in MODES or name not in meta["native_variants"]:
        raise ValueError("Unknown native H3 mode or packaged LoRA adapter")
    if mode not in meta["mode_profiles"]:
        raise ValueError(f"Native H3 mode {mode!r} was not exported; available modes: {sorted(meta['mode_profiles'])}")
    selected = deepcopy(meta["mode_profiles"][mode])
    variant = deepcopy(meta["native_variants"][name])
    branch = variant["branches"].get(selected["branch"])
    if branch is None:
        raise ValueError(f"H3 adapter {name!r} does not support {mode}")
    view = deepcopy(meta)
    view.update(
        mode=mode,
        active_adapter=lora_adapter,
        draft_steps=variant["steps"],
        h3_sigmas=variant["h3_sigmas"],
        audio_sigmas=variant["audio_sigmas"],
        capacity_plan=selected["capacity_plan"],
        h3_rope_config=branch["h3_rope_config"],
    )
    view["text_capacity"] = view["capacity_plan"]["text_capacity"]
    view["resources"]["input_config"] = meta["resources"][selected["input_config"]]
    for alias, physical in branch["graphs"].items():
        view["graphs"][alias] = deepcopy(meta["graphs"][physical])
    return view


def export_entries(meta):
    entries = shared_entries(meta)
    adapter_graphs = {
        graph
        for name, variant in meta["native_variants"].items()
        if name != "standard"
        for branch in variant["branches"].values()
        for graph in branch["graphs"].values()
    }
    standard_graphs = {
        graph
        for branch in meta["native_variants"]["standard"]["branches"].values()
        for graph in branch["graphs"].values()
    }
    adapter_resources = {
        variant["adapter_resource"] for name, variant in meta["native_variants"].items() if name != "standard"
    }
    shared = {
        key: value for key, value in entries.items() if key not in adapter_graphs | standard_graphs | adapter_resources
    }
    groups = {}
    for name, variant in meta["native_variants"].items():
        if name == "standard":
            continue
        physical = {value for branch in variant["branches"].values() for value in branch["graphs"].values()}
        group = deepcopy(shared)
        group.update({key: entries.pop(key) for key in sorted(physical)})
        resource = variant["adapter_resource"]
        group["adapter_config"] = entries.pop(resource)
        groups[name] = group
    if groups:
        entries["lora"] = groups
    return entries


def read_metadata(root):
    from .graph_audit import audit_graph
    from .native_schedule import native_schedule

    meta = json.loads(artifact_path(root, "runtime_meta_info.json").read_text())
    if (meta.get("kind"), meta.get("schema_version"), meta.get("complete")) != ("native_h3", 2, True):
        raise ValueError("Native H3 requires a complete schema-2 package")
    mode = meta.get("mode")
    if mode not in (*MODES, "all"):
        raise ValueError("Native H3 declares an invalid package mode")
    modes = set(MODES) if mode == "all" else {mode}
    if set(meta.get("mode_profiles", {})) != modes:
        raise ValueError("Native H3 mode profiles must match the exported package mode")
    required_branches = {"ref" if item == "ref2va" else "base" for item in modes}
    if meta.get("draft_frames") != 124 or meta.get("fps") != 24:
        raise ValueError("Unsupported native H3 temporal gear")
    size = meta.get("draft_size", [])
    if len(size) != 2 or any(type(n) is not int or n <= 0 or n % 32 for n in size):
        raise ValueError("Native H3 image dimensions must align to 32")
    policy = H3PrecisionPolicy(**meta["h3_precision"])
    if meta.get("scaled_rms_policy") != policy.scaled_rms_policy:
        raise ValueError("Native H3 RMS policy disagrees with exported precision")
    variants = meta.get("native_variants", {})
    if "standard" not in variants or set(variants["standard"]["branches"]) != required_branches:
        raise ValueError("Native H3 standard weight branches must match exported modes")
    for name, variant in variants.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
            raise ValueError("Native H3 adapter names must be safe artifact identifiers")
        expected = native_schedule(**variant["sampling"])
        schedule = ConditionedH3Schedule(tuple(variant["h3_sigmas"]), tuple(variant["audio_sigmas"]))
        if schedule != expected or schedule.steps != variant["steps"]:
            raise ValueError("Native H3 sigma tables disagree with adapter sampling")
        if not variant["branches"] or set(variant["branches"]) - required_branches:
            raise ValueError("Native H3 adapter declares invalid weight branches")
        if name != "standard" and variant.get("adapter_resource") not in meta["resources"]:
            raise ValueError("Native H3 adapter configuration is missing")
        _validate_variant(root, meta, name, variant)
        for branch in variant["branches"].values():
            if set(branch["graphs"]) != {"h3_text_refiner", "h3_denoiser"}:
                raise ValueError("Native H3 needs matching text and denoiser adapter graphs")
            if not set(branch["graphs"].values()) <= meta["graphs"].keys():
                raise ValueError("Native H3 adapter graph is missing")
    input_configs = {}
    for mode, profile in meta["mode_profiles"].items():
        if profile["branch"] != ("ref" if mode == "ref2va" else "base"):
            raise ValueError("Native H3 mode refers to the wrong weight branch")
        plan = profile["capacity_plan"]
        if plan.get("token_alignment") != 256 or any(
            plan[k] % 256 for k in ("text_capacity", "video_capacity", "audio_capacity")
        ):
            raise ValueError("Native H3 capacities must align to 256")
        if plan["dit_capacity"] != sum(plan[k] for k in ("text_capacity", "video_capacity", "audio_capacity")):
            raise ValueError("Native H3 packed capacity is inconsistent")
        inputs = json.loads(artifact_path(root, meta["resources"][profile["input_config"]]).read_text())
        input_configs[mode] = inputs
        if inputs.get("mode") != mode or plan.get("mode") != mode:
            raise ValueError("Native H3 input configuration mode disagrees with capacity plan")
    if not meta.get("feature_graphs") or "h3_audio_decode" not in meta["graphs"]:
        raise ValueError("Native H3 is missing its shared text/audio components")
    decoder = meta["native_video_decoder"]
    if not decoder["tiles"] or not set(decoder["tiles"].values()) <= meta["graphs"].keys():
        raise ValueError("Native H3 video decoder tiles are missing")
    if modes != {"t2va"} and meta.get("reference_vae_layout") != "split_v1":
        raise ValueError("Native H3 requires split reference encoder gears")
    for value in meta["resources"].values():
        artifact_path(root, value)
    for value in (*meta["tokenizer_files"], *meta.get("processor_files", [])):
        artifact_path(root, value)
    _validate_encoders(meta, input_configs)
    paths = [graph["path"] for graph in meta["graphs"].values()]
    if len(set(paths)) != len(paths):
        raise ValueError("Native H3 physical graphs must not alias another graph path")
    for graph in meta["graphs"].values():
        audit_graph(
            artifact_path(root, graph["path"]),
            quant_type=graph.get("quant_type", meta["quant_type"]),
            inputs=graph["input_names"],
            outputs=graph["output_names"],
            norm_fp16=graph.get("norm_fp16", False),
        )
    return meta


def _validate_encoders(meta, input_configs=None):
    """Validate shared host-side dimensions before opening simulator sessions."""
    required = {
        "text_embedding",
        "text_rope_inv_freq",
        "text_config",
        "h3_audio_config",
    }
    modes = set(meta["mode_profiles"])
    visual = modes != {"t2va"}
    base_images = bool(modes & {"i2va", "l2va", "fl2va"})
    reference = "ref2va" in modes
    reference_audio = any(
        p["capacity_plan"].get("reference_audio_tokens", 0) > 0 for p in meta["mode_profiles"].values()
    )
    for mode, profile in meta["mode_profiles"].items():
        plan = profile["capacity_plan"]
        if mode != "ref2va" and plan.get("reference_audio_tokens", 0):
            raise ValueError("Native H3 ordinary modes must not reserve reference audio")
        if mode == "t2va" and plan.get("reference_video_tokens", 0):
            raise ValueError("Native H3 text-only mode must not reserve reference video")
    if reference and input_configs is not None:
        limits = input_configs["ref2va"].get("references", {})
        has_audio = any(limits.get(kind, {}).get("max_count", 0) > 0 for kind in ("audio", "video"))
        if has_audio != reference_audio:
            raise ValueError("Native H3 reference audio capacity disagrees with input limits")
    if visual:
        required.update(("qwen_config", "qwen_vision_config", "reference_vae_config"))
    if reference:
        required.add("reference_audio_config")
    if (
        not required <= meta["resources"].keys()
        or not meta.get("tokenizer_files")
        or (visual and not meta.get("processor_files"))
    ):
        raise ValueError("Native H3 shared encoder resources are incomplete")
    if meta.get("text_feature_divisor") != 16 or meta.get("text_refiner_input_divisor") != 16:
        raise ValueError("Native H3 text feature divisors disagree with exported graphs")
    plan = meta["capacity_plan"]
    fields = ("text_capacity", "video_capacity", "audio_capacity", "kv_capacity")
    if any(type(plan.get(k)) is not int or plan[k] <= 0 or plan[k] % 256 for k in fields):
        raise ValueError("Native H3 root capacities must be positive aligned integers")
    patches = plan.get("qwen_visual_max_patches")
    if type(patches) is not int or patches % 256 or (patches <= 0 if visual else patches != 0):
        raise ValueError("Native H3 visual capacity disagrees with exported modes")
    fields = (*fields, "qwen_visual_max_patches")
    for key in fields:
        if plan[key] != max(p["capacity_plan"][key] for p in meta["mode_profiles"].values()):
            raise ValueError("Native H3 root capacity does not cover mode profiles exactly")
    if meta.get("text_capacity") != plan["text_capacity"] or meta.get("prompt_max_tokens") != plan["prompt_max_tokens"]:
        raise ValueError("Native H3 text capacity disagrees with its plan")
    for profile in meta["mode_profiles"].values():
        if any(
            profile["capacity_plan"][k] != plan[k]
            for k in ("kv_capacity", "qwen_visual_max_patches", "qwen_visual_patch_gears")
        ):
            raise ValueError("Native H3 modes must share encoder capacities")
    chunk, shape = meta.get("text_prefill_chunk_length"), meta.get("text_cache_shape", [])
    if (
        type(chunk) is not int
        or chunk <= 0
        or len(shape) != 4
        or any(type(n) is not int or n <= 0 for n in shape)
        or shape[0] != 1
        or shape[2] != plan["kv_capacity"]
        or shape[2] < plan["text_capacity"]
        or shape[2] % chunk
    ):
        raise ValueError("Native H3 text KV cache dimensions are invalid")
    cursor = 0
    for name in meta["feature_graphs"]:
        span = meta["graphs"].get(name, {}).get("text_layers", [])
        if len(span) != 2 or span[0] != cursor or type(span[1]) is not int or span[1] <= cursor:
            raise ValueError("Native H3 text feature layers must be contiguous")
        cursor = span[1]
    if cursor != meta.get("text_layers"):
        raise ValueError("Native H3 text feature layers do not cover the requested prefix")
    gears = meta.get("visual_graphs", [])
    capacities = [g.get("capacity") for g in gears]
    names = [g.get("graph") for g in gears]
    if (
        (not capacities if visual else bool(capacities))
        or any(type(n) is not int or n <= 0 or n % 256 for n in capacities)
        or capacities != sorted(set(capacities))
        or capacities != plan["qwen_visual_patch_gears"]
        or (capacities[-1] if capacities else 0) != plan["qwen_visual_max_patches"]
        or len(set(names)) != len(names)
        or set(names) != {n for n in meta["graphs"] if n.startswith("h3_visual_")}
    ):
        raise ValueError("Native H3 visual gears disagree with encoder capacities")
    base = meta.get("reference_vae_base", {})
    tiles, masks = meta.get("reference_vae_tiles", {}), meta.get("reference_vae_masks", {})
    if bool(base) != base_images or bool(tiles) != reference:
        raise ValueError("Native H3 reference VAE components disagree with exported modes")
    if not visual:
        if masks or any(name.startswith("h3_reference_encode") for name in meta["graphs"]):
            raise ValueError("Native H3 text-only package must not contain reference VAE graphs")
        _validate_reference_audio(meta, reference_audio, reference)
        return
    if (
        (base_images and (base.get("shape") != [1, *meta["draft_size"]] or base.get("graph") not in meta["graphs"]))
        or set(masks) != set(tiles.values())
        or base.get("graph") in masks
        or not set(tiles.values()) <= meta["graphs"].keys()
    ):
        raise ValueError("Native H3 reference VAE gears or masks are incomplete")
    declared = set(tiles.values()) | ({base["graph"]} if base_images else set())
    if {name for name in meta["graphs"] if name.startswith("h3_reference_encode")} - declared:
        raise ValueError("Native H3 contains an unused reference VAE graph")
    if reference and input_configs is not None:
        limits = input_configs["ref2va"].get("references", {})
        expected = ({1} if limits.get("image", {}).get("max_count", 0) else set()) | (
            {17} if limits.get("video", {}).get("max_count", 0) else set()
        )
        if {policy.get("shape", [None])[0] for policy in masks.values()} != expected:
            raise ValueError("Native H3 reference VAE temporal gears disagree with input limits")
    for key, graph in tiles.items():
        parts = key.split("x")
        mask = masks[graph]
        shape, levels = mask.get("shape", []), mask.get("levels", [])
        if (
            len(parts) != 3
            or any(not n.isdecimal() or int(n) <= 0 for n in parts)
            or len(shape) != 3
            or any(type(n) is not int or n <= 0 for n in shape)
            or not levels
            or levels[0] != [1, 1]
            or levels[-1] != [4, 16]
            or any(len(level) != 2 or any(type(n) is not int or n <= 0 for n in level) for level in levels)
            or int(parts[0]) != shape[0]
            or any(int(n) > cap for n, cap in zip(parts, shape, strict=True))
            or mask.get("spatial_padding_mode", "reflect") not in ("reflect", "constant")
        ):
            raise ValueError("Native H3 reference VAE mask geometry is invalid")
    policy = meta.get("reference_vae_policy", {})
    tile, overlap = policy.get("tile_size", []), policy.get("overlap", [])
    if (
        len(tile) != 2
        or len(overlap) != 2
        or any(type(t) is not int or type(o) is not int or not 0 <= o < t for t, o in zip(tile, overlap, strict=True))
    ):
        raise ValueError("Native H3 reference VAE tile policy is invalid")
    _validate_reference_audio(meta, reference_audio, reference)


def _validate_reference_audio(meta, required, reference):
    if ("h3_reference_audio_encode" in meta["graphs"]) != required:
        raise ValueError("Native H3 reference audio encoder disagrees with input capacity")
    if reference and (type(meta.get("reference_audio_frames")) is not int or meta["reference_audio_frames"] <= 0):
        raise ValueError("Native H3 reference audio capacity is invalid")


def _validate_variant(root, meta, name, variant):
    config = None
    if name != "standard":
        config = json.loads(artifact_path(root, meta["resources"][variant["adapter_resource"]]).read_text())
        if (
            config.get("name") != name
            or config.get("sampling") != variant["sampling"]
            or set(config.get("branches", {})) != set(variant["branches"])
        ):
            raise ValueError("Native H3 adapter resource disagrees with variant")
    for branch_name, branch in variant["branches"].items():
        expected = {role: f"{role}_{name}_{branch_name}" for role in ("h3_text_refiner", "h3_denoiser")}
        if branch.get("graphs") != expected:
            raise ValueError("Native H3 branch graphs are bound to the wrong adapter or weight branch")
        if branch.get("draft_steps") != variant["steps"] or branch.get("h3_packed_inputs") is not True:
            raise ValueError("Native H3 branch sampling or packed ABI disagrees with variant")
        lora = branch.get("lora")
        if name == "standard":
            if lora is not None:
                raise ValueError("Native H3 standard branch must not contain LoRA")
            continue
        spec = config["branches"][branch_name]
        if not isinstance(lora, dict) or {k: v for k, v in lora.items() if k != "bindings"} != spec:
            raise ValueError("Native H3 adapter resource and bound checkpoint disagree")
        if not re.fullmatch(r"[0-9a-f]{64}", str(spec.get("checkpoint_sha256", ""))):
            raise ValueError("Native H3 adapter requires checkpoint SHA256")
        bindings = lora.get("bindings", [])
        counts = spec.get("component_target_counts", {})
        if len(bindings) != 2 or {b.get("component") for b in bindings} != {"text", "dit"}:
            raise ValueError("Native H3 adapter requires both text and DiT bindings")
        targets = []
        for binding in bindings:
            component = binding["component"]
            names = binding.get("targets", [])
            if (
                not names
                or binding.get("target_count") != len(names)
                or counts.get(component) != len(names)
                or any(n.startswith("token_refiner.") != (component == "text") for n in names)
            ):
                raise ValueError("Native H3 adapter binding counts or component targets disagree")
            targets.extend(names)
        if len(set(targets)) != len(targets) or len(targets) != spec.get("target_count"):
            raise ValueError("Native H3 adapter targets were not consumed exactly once")
