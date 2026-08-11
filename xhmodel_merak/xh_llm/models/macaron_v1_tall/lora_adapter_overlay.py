"""Build small HMONNX test overlays for Macaron LoRA selection."""

from __future__ import annotations

import json
import os
from pathlib import Path

import onnx
from onnx import TensorProto

from xhmodel_merak.utils import calculate_file_md5

from .lora_constant_bundle import read_manifest
from .lora_weights import ADAPTER_NAMES


_PHASES = ("prefill", "decode")


def _adapter_initializers(adapter_onnx: Path) -> dict[str, TensorProto]:
    model = onnx.load(adapter_onnx, load_external_data=False)
    values = {item.name: item for item in model.graph.initializer}
    if not values:
        raise ValueError(f"No initializers in LoRA pack: {adapter_onnx}")
    for name, value in values.items():
        keys = {item.key for item in value.external_data}
        if value.data_location != TensorProto.EXTERNAL or not {
            "offset",
            "length",
        } <= keys:
            raise ValueError(f"{adapter_onnx}: {name} is not external")
    return values


def _point_to_adapter(value: TensorProto, adapter: str) -> None:
    location = f"../runtime_lora/{adapter}.onnx.data"
    for entry in value.external_data:
        if entry.key == "location":
            entry.value = location
            return
    value.external_data.add(key="location", value=location)


def _bind_graph(
    source: Path,
    destination: Path,
    pack: dict[str, TensorProto],
    initializer_order: list[str],
    adapter: str,
) -> None:
    model = onnx.load(source, load_external_data=False)
    graph_values = {item.name: item for item in model.graph.initializer}
    missing = [name for name in initializer_order if name not in graph_values]
    if missing:
        raise ValueError(f"{source}: missing LoRA initializers: {missing[:3]}")
    missing = [name for name in initializer_order if name not in pack]
    if missing:
        raise ValueError(f"{adapter}: pack misses initializers: {missing[:3]}")
    for name in initializer_order:
        graph_value = graph_values[name]
        adapter_value = pack[name]
        if (
            graph_value.data_type != adapter_value.data_type
            or tuple(graph_value.dims) != tuple(adapter_value.dims)
        ):
            raise ValueError(f"{source}: schema mismatch for {name}")
        graph_value.CopyFrom(adapter_value)
        _point_to_adapter(graph_value, adapter)
    onnx.save(model, destination)


def _link(source: Path, destination: Path) -> None:
    os.symlink(source.resolve(), destination, target_is_directory=source.is_dir())


def _check_existing_overlay(
    output_dir: Path,
    export_dir: Path,
    adapter: str,
) -> Path | None:
    metadata_path = output_dir / "golden_meta_info.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("python_validation_adapter") == adapter
        and metadata.get("python_validation_export_dir") == str(export_dir)
    ):
        return output_dir
    raise FileExistsError(f"Refusing to overwrite incompatible overlay: {output_dir}")


def _phase_graph_path(
    export_dir: Path,
    metadata: dict,
    phase: str,
) -> Path:
    graph_rel = metadata.get(f"{phase}_hmonnx")
    if graph_rel:
        source = export_dir / graph_rel
        if source.parent.name == phase and source.is_file():
            return source
    candidates = sorted((export_dir / phase).glob("*_with_act.onnx"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Cannot resolve {phase} HMONNX graph")
    return candidates[0]


def create_macaron_lora_overlay(
    export_dir: str | Path,
    adapter: str,
    output_dir: str | Path,
) -> Path:
    """Bind one quantized LoRA pack into a test-only HMONNX overlay.

    The output contains modified Prefill/Decode protobufs and symlinks to all
    immutable source artifacts. It does not duplicate or modify base weights.
    Both Linear and 28-input MoeBlock LoRA consume these same initializers.
    """

    if adapter not in ADAPTER_NAMES:
        raise ValueError(f"Unsupported Macaron adapter: {adapter}")
    export_dir = Path(export_dir).resolve()
    output_dir = Path(output_dir).resolve()
    metadata_path = export_dir / "golden_meta_info.json"
    manifest_path = export_dir / "runtime_lora" / "manifest.json"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Export lacks golden metadata or runtime_lora")
    if output_dir.exists():
        existing = _check_existing_overlay(output_dir, export_dir, adapter)
        if existing is not None:
            return existing
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")

    manifest = read_manifest(manifest_path)
    initializer_order = manifest["input_order"]
    adapter_onnx = export_dir / "runtime_lora" / manifest["adapters"][adapter]["onnx"]
    pack = _adapter_initializers(adapter_onnx)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True)
    for entry in export_dir.iterdir():
        if entry.name not in {"golden_meta_info.json", *_PHASES}:
            _link(entry, output_dir / entry.name)

    for phase in _PHASES:
        source = _phase_graph_path(export_dir, metadata, phase)
        destination_dir = output_dir / phase
        destination_dir.mkdir()
        for entry in source.parent.iterdir():
            if entry.name != source.name:
                _link(entry, destination_dir / entry.name)
        destination = destination_dir / source.name
        _bind_graph(source, destination, pack, initializer_order, adapter)
        graph_rel = str(destination.relative_to(output_dir))
        metadata[f"{phase}_hmonnx"] = graph_rel
        if f"{phase}_onnx" in metadata:
            metadata[f"{phase}_onnx"] = graph_rel
        metadata[f"{phase}_hmonnx_md5"] = calculate_file_md5(destination)

    metadata["python_validation_adapter"] = adapter
    metadata["python_validation_export_dir"] = str(export_dir)
    metadata["python_validation_binding"] = "initializer_external_data"
    (output_dir / "golden_meta_info.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


__all__ = ["create_macaron_lora_overlay"]
