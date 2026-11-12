"""Qwen4's component vision ABI, shared by publication and loading."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path


def _file(root: Path, value: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"Invalid Qwen4 vision resource path: {value!r}")
    path = root / relative
    if (not path.is_file() or not path.resolve().is_relative_to(root)
            or any(parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent)):
        raise ValueError(f"Missing or unconfined Qwen4 vision resource: {value}")
    return path


def validate_qwen4_exp_vision(
    visual: Mapping,
    root: str | Path,
    *,
    image_token_gears: Sequence[int] | None = None,
    vision_config: Mapping | None = None,
) -> dict:
    """Validate every executable component, including its public tensor ABI."""
    import onnx
    from google.protobuf.message import DecodeError

    from ..qwen3_5.visual_token_gears import (
        VISUAL_ATTENTION_MASK_FORMAT,
        VISUAL_ATTENTION_MASK_OPERATOR,
        VISUAL_ATTENTION_MASK_SHAPE,
        VISUAL_ROTARY_POSITION_FORMAT,
    )

    root = Path(root).resolve()
    if not isinstance(visual, Mapping) or (
        visual.get("abi_version") != 2 or isinstance(visual.get("abi_version"), bool)
        or visual.get("layout") != "components" or visual.get("dtype") != "float16"
        or visual.get("visual_input_mode") != "patches" or visual.get("routing_policy") != "smallest_fit"
    ):
        raise ValueError("Unsupported Qwen4 component vision ABI")
    config = visual.get("vision_config")
    if not isinstance(config, Mapping):
        raise ValueError("Qwen4 vision metadata requires its native configuration")
    fields = ("depth", "hidden_size", "num_heads", "out_hidden_size", "patch_size",
              "spatial_merge_size", "temporal_patch_size", "in_channels", "num_position_embeddings")
    for field in fields:
        value = config.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Invalid Qwen4 vision dimension: {field}")
        if vision_config is not None and value != vision_config.get(field):
            raise ValueError(f"Qwen4 vision source configuration mismatch: {field}")
    if config.get("deepstack_visual_indexes") or (vision_config or {}).get("deepstack_visual_indexes"):
        raise ValueError("Qwen4 component vision does not support DeepStack features")
    for field in ("patch_size", "spatial_merge_size", "temporal_patch_size", "hidden_size",
                  "num_heads", "in_channels", "num_position_embeddings"):
        if type(visual.get(field)) is not int or visual[field] != config[field]:
            raise ValueError(f"Qwen4 vision geometry mismatch: {field}")
    formats = {
        "token_unit": "post_spatial_merge", "overflow_policy": "reject",
        "attention_mask_format": VISUAL_ATTENTION_MASK_FORMAT,
        "attention_mask_operator": VISUAL_ATTENTION_MASK_OPERATOR,
        "attention_mask_shape": VISUAL_ATTENTION_MASK_SHAPE,
        "rotary_position_format": VISUAL_ROTARY_POSITION_FORMAT,
        "input_names": ["pixel_values", "position_ids", "position_weights",
                        "rotary_position_ids", "attention_mask"],
    }
    if any(visual.get(key) != expected for key, expected in formats.items()):
        raise ValueError("Qwen4 vision host geometry formats are inconsistent")
    gears = visual.get("image_token_gears")
    if (not isinstance(gears, list) or not gears or any(type(g) is not int or g <= 0 for g in gears)
            or gears != sorted(set(gears)) or visual.get("image_token_capacity") != gears[-1]
            or (image_token_gears is not None and gears != list(image_token_gears))):
        raise ValueError("Qwen4 vision token gears disagree with the declared profile")
    if visual.get("patch_token_capacity") != gears[-1] * config["spatial_merge_size"] ** 2:
        raise ValueError("Qwen4 vision maximum patch capacity is inconsistent")
    if (type(visual.get("visual_rope_cache_length")) is not int
            or visual["visual_rope_cache_length"] < gears[-1] * config["spatial_merge_size"]):
        raise ValueError("Qwen4 vision rotary cache does not cover its gears")
    entries = visual.get("gears")
    if not isinstance(entries, list) or len(entries) != len(gears):
        raise ValueError("Qwen4 vision metadata must describe every token gear")
    manifest_name = visual.get("gear_manifest")
    if manifest_name != "visual_gears.json":
        raise ValueError("Qwen4 vision requires a root-level gear manifest")
    manifest = json.loads(_file(root, manifest_name).read_text())
    if (not isinstance(manifest, Mapping) or manifest.get("schema_version") != 2
            or any(manifest.get(key) != value for key, value in visual.items()
                   if key not in {"image_size_w", "image_size_h", "onnx"})):
        raise ValueError("Qwen4 vision gear manifest disagrees with runtime metadata")
    width, heads = config["hidden_size"], config["num_heads"]
    if width % heads:
        raise ValueError("Qwen4 vision head geometry is invalid")
    padded_head = math.ceil(width / heads / 64) * 64
    patch_dim = config["in_channels"] * config["temporal_patch_size"] * config["patch_size"] ** 2
    expected_names = ["stem", *[f"block_{i:02d}" for i in range(config["depth"])], "merger"]
    exports = {}

    def tensor(value):
        shape = [dim.dim_value if dim.HasField("dim_value") else None
                 for dim in value.type.tensor_type.shape.dim]
        return value.name, value.type.tensor_type.elem_type, shape

    for capacity, gear in zip(gears, entries, strict=True):
        patches = capacity * config["spatial_merge_size"] ** 2
        if (not isinstance(gear, Mapping) or gear.get("image_token_capacity") != capacity
                or gear.get("patch_token_capacity") != patches):
            raise ValueError("Qwen4 vision gear capacity mismatch")
        components = gear.get("components")
        if (not isinstance(components, list) or not all(isinstance(part, Mapping) for part in components)
                or [part.get("name") for part in components] != expected_names):
            raise ValueError("Qwen4 vision components are incomplete or unordered")
        hidden = [1, patches, width]
        fp16, int32 = onnx.TensorProto.FLOAT16, onnx.TensorProto.INT32
        for component in components:
            name = component["name"]
            relative = component.get("hmonnx")
            if relative != f"visual_m{capacity}/{name}.onnx":
                raise ValueError("Qwen4 vision component path disagrees with its gear and role")
            path = _file(root, relative)
            if hashlib.md5(path.read_bytes()).hexdigest() != component.get("md5"):
                raise ValueError(f"Qwen4 vision component checksum mismatch: {relative}")
            if name == "stem":
                inputs = [("pixel_values", fp16, [1, patches, patch_dim]),
                          ("position_ids", int32, [4, patches]),
                          ("position_weights", fp16, [4, patches]),
                          ("rotary_position_ids", int32, [2, patches])]
                outputs = [("hidden_states", fp16, hidden),
                           ("rotary_cos", fp16, [patches, 1, padded_head]),
                           ("rotary_sin", fp16, [patches, 1, padded_head])]
            elif name == "merger":
                inputs = [("hidden_states", fp16, hidden)]
                outputs = [("image_embeds", fp16, [1, capacity, config["out_hidden_size"]])]
            else:
                inputs = [("hidden_states", fp16, hidden),
                          ("rotary_cos", fp16, [patches, 1, padded_head]),
                          ("rotary_sin", fp16, [patches, 1, padded_head]),
                          ("attention_mask", fp16, [1, 1, 1, patches])]
                outputs = [("hidden_states_out", fp16, hidden)]
            try:
                graph = onnx.load(path, load_external_data=False).graph
                onnx.checker.check_model(str(path))
            except (onnx.checker.ValidationError, DecodeError) as exc:
                raise ValueError(f"Invalid Qwen4 vision component graph: {relative}: {exc}") from exc
            if ([tensor(v) for v in graph.input] != inputs or [tensor(v) for v in graph.output] != outputs
                    or component.get("input_names") != [v[0] for v in inputs]
                    or component.get("output_names") != [v[0] for v in outputs]):
                raise ValueError(f"Qwen4 vision component tensor ABI mismatch: {relative}")
            exports[f"visual_m{capacity}_{name}"] = {"path": relative, "type": "hmonnx"}
    return {"token_unit": "post_spatial_merge", "routing_policy": "smallest_fit",
            "gear_manifest": manifest_name, "gears": entries, "export_entries": exports}
