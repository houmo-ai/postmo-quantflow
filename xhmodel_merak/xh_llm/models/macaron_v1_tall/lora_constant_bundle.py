"""Write and validate compiler-facing LoRA initializer ONNX bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .lora_weights import ADAPTER_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_hash(schema: Iterable[dict]) -> str:
    encoded = json.dumps(list(schema), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ConstantONNXArtifact:
    """Files produced for one adapter's initializer-only ONNX."""

    onnx: str
    sha256: str
    external_data: str
    external_data_sha256: str

    def manifest_entry(self) -> dict[str, str]:
        return {
            "onnx": self.onnx,
            "sha256": self.sha256,
            "external_data": self.external_data,
            "external_data_sha256": self.external_data_sha256,
        }


@dataclass(frozen=True)
class MacaronLoRAConstantBundleArtifact:
    """Published bundle metadata used by graph binding and validation."""

    bundle_dir: Path
    manifest_path: Path
    input_order: tuple[str, ...]
    schema: tuple[dict, ...]
    adapters: tuple[tuple[str, ConstantONNXArtifact], ...]

    def adapter_onnx(self, adapter_name: str) -> Path:
        for name, artifact in self.adapters:
            if name == adapter_name:
                return self.bundle_dir / artifact.onnx
        raise KeyError(f"Unknown Macaron LoRA adapter: {adapter_name}")


class ConstantONNXWriter:
    """Stream initializer bytes to external data and keep only TensorProto metadata."""

    def __init__(self, onnx_path: str | Path):
        self.onnx_path = Path(onnx_path)
        self.onnx_path.parent.mkdir(parents=True, exist_ok=True)
        self.external_path = self.onnx_path.with_suffix(self.onnx_path.suffix + ".data")
        self._temporary_onnx_path = self.onnx_path.with_suffix(self.onnx_path.suffix + ".tmp")
        self._temporary_external_path = self.external_path.with_suffix(self.external_path.suffix + ".tmp")
        self._stream = self._temporary_external_path.open("wb")
        self._offset = 0
        self._initializers = []
        self._outputs = []
        self._schema: list[dict] = []
        self._names: set[str] = set()
        self._closed = False

    @property
    def schema(self) -> tuple[dict, ...]:
        return tuple(self._schema)

    def add(self, name: str, value: np.ndarray) -> TensorProto:
        if self._closed:
            raise RuntimeError("Cannot add an initializer after the writer is closed")
        if name in self._names:
            raise ValueError(f"Duplicate LoRA initializer name: {name}")
        value = np.ascontiguousarray(value)
        initializer = numpy_helper.from_array(value, name=name)
        data = initializer.raw_data
        self._stream.write(data)
        initializer.ClearField("raw_data")
        initializer.data_location = TensorProto.EXTERNAL
        initializer.external_data.add(key="location", value=self.external_path.name)
        initializer.external_data.add(key="offset", value=str(self._offset))
        initializer.external_data.add(key="length", value=str(len(data)))
        self._offset += len(data)

        descriptor = {
            "name": name,
            "dtype": int(initializer.data_type),
            "shape": [int(dimension) for dimension in initializer.dims],
        }
        self._initializers.append(initializer)
        self._outputs.append(helper.make_tensor_value_info(name, initializer.data_type, list(initializer.dims)))
        self._schema.append(descriptor)
        self._names.add(name)
        return initializer

    def close(self) -> ConstantONNXArtifact:
        if self._closed:
            raise RuntimeError("LoRA constant ONNX writer is already closed")
        self._stream.close()
        graph = helper.make_graph(
            [],
            f"{self.onnx_path.stem}_lora_constants",
            [],
            self._outputs,
            initializer=self._initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.metadata_props.add(key="lora_schema_hash", value=schema_hash(self._schema))
        onnx.save(model, self._temporary_onnx_path)
        os.replace(self._temporary_external_path, self.external_path)
        os.replace(self._temporary_onnx_path, self.onnx_path)
        self._closed = True
        return ConstantONNXArtifact(
            onnx=self.onnx_path.name,
            sha256=_sha256(self.onnx_path),
            external_data=self.external_path.name,
            external_data_sha256=_sha256(self.external_path),
        )

    def abort(self) -> None:
        """Close and remove this writer's uncommitted temporary files."""

        if not self._closed:
            self._stream.close()
            self._closed = True
        for path in (
            self._temporary_external_path,
            self._temporary_onnx_path,
        ):
            if path.exists():
                path.unlink()

    def __enter__(self) -> "ConstantONNXWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None and not self._closed:
            self.abort()


class MacaronLoRAConstantBundle:
    """Stream four adapter packs into one atomically published bundle."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.bundle_dir = self.output_dir / "runtime_lora"
        self.temporary_bundle_dir = self.output_dir / ".runtime_lora.tmp"
        if self.bundle_dir.exists() or self.temporary_bundle_dir.exists():
            raise FileExistsError("Macaron LoRA bundle destination already exists: " f"{self.bundle_dir}")
        self.temporary_bundle_dir.mkdir(parents=True)
        self._writers = {
            adapter_name: ConstantONNXWriter(self.temporary_bundle_dir / f"{adapter_name}.onnx")
            for adapter_name in ADAPTER_NAMES
        }
        self._input_order: list[str] = []
        self._names: set[str] = set()
        self._closed = False

    @property
    def input_order(self) -> tuple[str, ...]:
        return tuple(self._input_order)

    def has(self, name: str) -> bool:
        return name in self._names

    def add(
        self,
        name: str,
        values_by_adapter: Mapping[str, np.ndarray],
    ) -> None:
        """Append one identically shaped initializer for all four adapters."""

        if self._closed:
            raise RuntimeError("Cannot append to a closed LoRA constant bundle")
        if name in self._names:
            raise ValueError(f"Duplicate LoRA initializer name: {name}")
        if set(values_by_adapter) != set(ADAPTER_NAMES):
            raise ValueError("LoRA constant values must contain exactly " f"{list(ADAPTER_NAMES)}: {name}")
        template = values_by_adapter[ADAPTER_NAMES[0]]
        for adapter_name in ADAPTER_NAMES[1:]:
            value = values_by_adapter[adapter_name]
            if value.shape != template.shape or value.dtype != template.dtype:
                raise ValueError(
                    f"LoRA initializer {name} differs across adapters: "
                    f"{adapter_name} has shape={value.shape}, dtype={value.dtype}; "
                    f"expected shape={template.shape}, dtype={template.dtype}"
                )
        for adapter_name in ADAPTER_NAMES:
            self._writers[adapter_name].add(name, values_by_adapter[adapter_name])
        self._names.add(name)
        self._input_order.append(name)

    def close(
        self,
        *,
        macaron_lora: Mapping[str, object] | None = None,
    ) -> MacaronLoRAConstantBundleArtifact:
        """Finalize the four packs and atomically publish runtime_lora."""

        if self._closed:
            raise RuntimeError("LoRA constant bundle is already closed")
        common_schema = self._writers[ADAPTER_NAMES[0]].schema
        if not common_schema:
            raise ValueError("Macaron LoRA constant bundle cannot be empty")
        for adapter_name in ADAPTER_NAMES[1:]:
            if self._writers[adapter_name].schema != common_schema:
                raise ValueError(f"Combined LoRA ONNX schema differs for {adapter_name}")
        adapter_artifacts = {adapter_name: self._writers[adapter_name].close() for adapter_name in ADAPTER_NAMES}
        manifest_path = write_manifest(
            self.temporary_bundle_dir,
            input_order=self._input_order,
            schema=common_schema,
            adapters={name: artifact.manifest_entry() for name, artifact in adapter_artifacts.items()},
            macaron_lora=macaron_lora,
        )
        self.temporary_bundle_dir.replace(self.bundle_dir)
        self._closed = True
        return MacaronLoRAConstantBundleArtifact(
            bundle_dir=self.bundle_dir,
            manifest_path=self.bundle_dir / manifest_path.name,
            input_order=tuple(self._input_order),
            schema=tuple(common_schema),
            adapters=tuple(adapter_artifacts.items()),
        )

    def abort(self) -> None:
        """Discard the unfinished bundle and its temporary external data."""

        if self._closed:
            return
        for writer in self._writers.values():
            writer.abort()
        shutil.rmtree(self.temporary_bundle_dir, ignore_errors=True)
        self._closed = True

    def __enter__(self) -> "MacaronLoRAConstantBundle":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._closed:
            self.abort()


def load_initializer_references(
    onnx_path: str | Path,
    *,
    relative_data_location: str,
    offset_delta: int = 0,
) -> dict[str, TensorProto]:
    """Return external-data initializer metadata repointed at a main graph.

    The tensors remain ONNX initializers, but no weight bytes are embedded in
    Prefill/Decode protobufs. This preserves the initializer-name contract while
    avoiding ONNX's practical message-size limit for the full expert pack.
    """
    if offset_delta < 0:
        raise ValueError(f"External-data offset delta must be non-negative, got {offset_delta}")

    model = onnx.load(str(onnx_path), load_external_data=False)
    references: dict[str, TensorProto] = {}
    for initializer in model.graph.initializer:
        metadata = {entry.key: entry.value for entry in initializer.external_data}
        if initializer.data_location != TensorProto.EXTERNAL or not {
            "offset",
            "length",
        } <= set(metadata):
            raise ValueError(f"LoRA bundle initializer is not external-data backed: {initializer.name}")
        reference = TensorProto()
        reference.CopyFrom(initializer)
        for entry in reference.external_data:
            if entry.key == "location":
                entry.value = relative_data_location
            elif entry.key == "offset":
                entry.value = str(int(entry.value) + offset_delta)
        if not any(entry.key == "location" for entry in reference.external_data):
            reference.external_data.add(key="location", value=relative_data_location)
        references[reference.name] = reference
    return references


def append_canonical_lora_to_stage(
    output_dir: str | Path,
    phase: str,
    canonical_onnx: str | Path,
) -> dict[str, TensorProto]:
    """Append canonical L0 bytes to a Qwen stage external-data file."""

    output_dir = Path(output_dir)
    canonical_onnx = Path(canonical_onnx)
    stage_dir = output_dir / phase
    stage_external_data = stage_dir / f"{output_dir.name}_{phase}_external_data"
    if not stage_external_data.is_file():
        candidates = sorted(stage_dir.glob("*_external_data"))
        raise FileNotFoundError(
            f"{phase}: expected {stage_external_data.name}; " f"found {[path.name for path in candidates]}"
        )

    canonical_data = canonical_onnx.with_suffix(canonical_onnx.suffix + ".data")
    if not canonical_data.is_file():
        raise FileNotFoundError(f"Missing canonical LoRA external data: {canonical_data}")

    offset_delta = stage_external_data.stat().st_size
    with (
        canonical_data.open("rb") as source,
        stage_external_data.open("ab") as destination,
    ):
        shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)

    return load_initializer_references(
        canonical_onnx,
        relative_data_location=stage_external_data.name,
        offset_delta=offset_delta,
    )


def write_manifest(
    output_dir: str | Path,
    *,
    input_order: Iterable[str],
    schema: Iterable[dict],
    adapters: dict[str, dict],
    macaron_lora: Mapping[str, object] | None = None,
) -> Path:
    input_order = list(input_order)
    non_lowercase = [name for name in input_order if name != name.lower()]
    if non_lowercase:
        raise ValueError(f"LoRA initializer names must be lowercase for HMONNX V2: {non_lowercase[:3]}")
    output_dir = Path(output_dir)
    schema = list(schema)
    if not input_order or len(input_order) != len(set(input_order)):
        raise ValueError("LoRA constant input_order must be non-empty and unique")
    if [entry["name"] for entry in schema] != input_order:
        raise ValueError("LoRA constant schema order does not match input_order")
    if set(adapters) != set(ADAPTER_NAMES):
        raise ValueError(f"LoRA constant bundle must contain exactly {list(ADAPTER_NAMES)}")

    manifest = {
        "format": "onnx_initializer_bundle",
        "input_order": input_order,
        "schema": schema,
        "schema_hash": schema_hash(schema),
        "adapters": adapters,
    }
    if macaron_lora is not None:
        manifest["macaron_lora"] = dict(macaron_lora)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def read_manifest(manifest_path: str | Path) -> dict:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "onnx_initializer_bundle":
        raise ValueError(f"Unsupported LoRA constant bundle format: {manifest.get('format')!r}")
    input_order = manifest.get("input_order")
    schema = manifest.get("schema")
    adapters = manifest.get("adapters")
    if not isinstance(input_order, list) or not input_order or len(input_order) != len(set(input_order)):
        raise ValueError("LoRA constant manifest input_order must be non-empty and unique")
    if any(name != name.lower() for name in input_order):
        raise ValueError("LoRA constant manifest initializer names must be lowercase for HMONNX V2")
    if (
        not isinstance(schema, list)
        or not all(isinstance(entry, dict) for entry in schema)
        or [entry.get("name") for entry in schema] != input_order
    ):
        raise ValueError("LoRA constant manifest schema does not match input_order")
    if manifest.get("schema_hash") != schema_hash(schema):
        raise ValueError("LoRA constant manifest schema_hash mismatch")
    if not isinstance(adapters, dict) or set(adapters) != set(ADAPTER_NAMES):
        raise ValueError(f"LoRA constant manifest must contain exactly {list(ADAPTER_NAMES)}")
    return manifest
