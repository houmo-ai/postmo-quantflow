"""Checkpoint views for small Qwen4-Exp language-only export experiments."""

from __future__ import annotations

import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any


_INDEX_NAME = "model.safetensors.index.json"
_LAYER_PREFIX = "model.language_model.layers."
_TEXT_ROOT_PREFIXES = (
    "lm_head.",
    "model.language_model.embed_tokens.",
    "model.language_model.hyper_connection_mixer.",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_or_link(source: Path, target: Path, *, link_mode: str) -> None:
    if link_mode == "symlink":
        os.symlink(source, target)
        return
    if link_mode == "hardlink":
        os.link(source, target)
        return
    if link_mode == "copy":
        shutil.copy2(source, target)
        return
    raise ValueError(f"unsupported link_mode={link_mode!r}")


def _layer_index(tensor_name: str) -> int | None:
    if not tensor_name.startswith(_LAYER_PREFIX):
        return None
    suffix = tensor_name[len(_LAYER_PREFIX) :]
    layer, _, _ = suffix.partition(".")
    if not layer.isdecimal():
        return None
    return int(layer)


def _is_retained_tensor(tensor_name: str, *, num_layers: int) -> bool:
    if tensor_name.startswith(_TEXT_ROOT_PREFIXES):
        return True
    layer_idx = _layer_index(tensor_name)
    return layer_idx is not None and layer_idx < num_layers


def _truncate_text_config(config: dict[str, Any], *, num_layers: int) -> tuple[list[str], list[int]]:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("Qwen4-Exp config.json must contain a dict text_config")

    original_layers = int(text_config.get("num_hidden_layers", 0))
    if num_layers <= 0 or num_layers > original_layers:
        raise ValueError(f"num_layers must be in [1,{original_layers}], got {num_layers}")

    layer_types = text_config.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) < num_layers:
        raise ValueError("text_config.layer_types is missing or shorter than the requested layer count")
    retained_layer_types = [str(item) for item in layer_types[:num_layers]]

    ple_layer_ids = text_config.get("ple_layer_ids") or []
    retained_ple_layer_ids = [int(layer_id) for layer_id in ple_layer_ids if 1 <= int(layer_id) <= num_layers]

    text_config["num_hidden_layers"] = num_layers
    text_config["layer_types"] = retained_layer_types
    text_config["ple_layer_ids"] = retained_ple_layer_ids

    mtp_config = text_config.get("mtp")
    if isinstance(mtp_config, dict):
        mtp_config["num_hidden_layers"] = 0
        mtp_config["layer_types"] = []
    text_config["mtp_num_hidden_layers"] = 0
    return retained_layer_types, retained_ple_layer_ids


def prepare_qwen4_exp_partial_checkpoint(
    source: str | Path,
    output: str | Path,
    *,
    num_layers: int = 4,
    link_mode: str = "symlink",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a filtered, linked checkpoint view for the first text layers.

    The view writes a truncated ``config.json`` and a filtered safetensors
    index.  Referenced shards are linked without copying; unreferenced visual,
    MTP and later decoder tensors are omitted from the index so loaders do not
    account for them as part of this language-only experiment.
    """

    source_path = Path(source).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().resolve()
    if output_path == source_path or output_path in source_path.parents or source_path in output_path.parents:
        raise ValueError("checkpoint output must not replace, contain, or be contained by the source")
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    config = json.loads((source_path / "config.json").read_text(encoding="utf-8"))
    layer_types, ple_layer_ids = _truncate_text_config(config, num_layers=num_layers)

    index = json.loads((source_path / _INDEX_NAME).read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    retained = {name: shard for name, shard in weight_map.items() if _is_retained_tensor(name, num_layers=num_layers)}
    removed = sorted(set(weight_map).difference(retained))
    if not retained:
        raise ValueError(f"no tensors retained from {source_path}")

    index["weight_map"] = retained
    retained_shards = set(retained.values())
    total_size = 0
    for shard in retained_shards:
        with (source_path / shard).open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
        for name in retained:
            if retained[name] == shard:
                start, end = header[name]["data_offsets"]
                total_size += end - start
    index.setdefault("metadata", {})["total_size"] = total_size

    _write_json(output_path / "config.json", config)
    _write_json(output_path / _INDEX_NAME, index)

    for item in source_path.iterdir():
        if item.name in {"config.json", _INDEX_NAME}:
            continue
        if item.suffix == ".safetensors" and item.name not in retained_shards:
            continue
        _copy_or_link(item, output_path / item.name, link_mode=link_mode)

    manifest = {
        "source": str(source_path),
        "output": str(output_path),
        "num_hidden_layers": int(num_layers),
        "layer_types": layer_types,
        "ple_layer_ids": ple_layer_ids,
        "visual_weights_included": False,
        "mtp_weights_included": False,
        "retained_tensor_count": len(retained),
        "retained_tensor_bytes": total_size,
        "removed_tensor_count": len(removed),
        "retained_shard_count": len(retained_shards),
        "link_mode": link_mode,
    }
    _write_json(output_path / "partial_checkpoint_manifest.json", manifest)
    return manifest


__all__ = ["prepare_qwen4_exp_partial_checkpoint"]
