"""Publish Qwen4 target graphs with shared external weights and Host PLE tables."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import onnx

from xhmodel_merak.model_artifact.files import confined_file, file_sha256
from xhmodel_merak.model_build.packaging import _model_tensors, validate_onnx_external_locations
from xhmodel_merak.model_integrations.shared_weights import share_language_weights

from .ple_export import _safe_asset, ple_resource_manifest, read_qwen4_exp_ple_specs


def _write_json(path, value):
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    if path.is_file() and path.read_bytes() == encoded:
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resources(root, graph, model=None):
    model = model if model is not None else onnx.load(graph, load_external_data=False)
    paths = {graph}
    shared_blob = graph.parent / "weights.bin"
    if shared_blob.is_file():
        paths.add(shared_blob)
    for tensor in _model_tensors(model):
        entries = {entry.key: entry.value for entry in tensor.external_data}
        if "location" in entries:
            paths.add(confined_file(graph.parent, entries["location"], "external_data.location"))
    owner, _ = ple_resource_manifest(graph)
    specs = read_qwen4_exp_ple_specs(graph)
    for spec in specs:
        paths.update(_safe_asset(owner, spec[kind]) for kind in ("packed", "exponent"))
    if specs:
        manifest = owner / "ple" / "manifest.json"
        if not manifest.is_file():
            manifest = owner / "ple.json"
        if manifest.is_file():
            paths.add(manifest)
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("Qwen4 graph resources must remain inside the export root")
    return paths


def _metadata_files(root, value):
    if isinstance(value, dict):
        return set().union(*(_metadata_files(root, item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_metadata_files(root, item) for item in value), set())
    if isinstance(value, str) and value and not Path(value).is_absolute() and ".." not in Path(value).parts:
        path = root / value
        if path.is_file():
            return {confined_file(root, value, "runtime.resource")}
    return set()


def _retire_target_temporaries(root, graphs, metadata, candidates):
    """Retire only superseded external blobs; component cleanup owns its stages."""
    keep = _metadata_files(root, metadata) | set(graphs.values())
    for graph in root.rglob("*.onnx"):
        keep.update(_resources(root, graph.resolve()))
    for path in candidates - keep:
        if path.is_file() and path.is_relative_to(root):
            path.unlink()


def publish_qwen4_exp_shared_weights(output_dir):
    """Keep graph names and tensor identities; share only their external storage.

    The common DeepSeek/GLM writer handles tensor-content interning. Qwen4 adds
    PLE hardlinks and refreshes its target completion records without changing
    the graph ABI, draft resources or private fixed-head checkpoint.
    """
    root = Path(output_dir).resolve()
    metadata_path = root / "golden_meta_info.json"
    metadata = json.loads(metadata_path.read_text())
    graphs = {stage: confined_file(root, metadata[stage + "_hmonnx"], stage) for stage in ("prefill", "decode")}
    models = {stage: onnx.load(path, load_external_data=False) for stage, path in graphs.items()}
    weights = {stage: root / stage / "weights.bin" for stage in graphs}
    already_shared = (
        all(path.is_file() and not path.is_symlink() for path in weights.values())
        and weights["prefill"].samefile(weights["decode"])
        and all(
            dict((entry.key, entry.value) for entry in tensor.external_data).get("location") == "weights.bin"
            for model in models.values()
            for tensor in _model_tensors(model)
            if tensor.data_location == onnx.TensorProto.EXTERNAL
        )
    )
    manifest_path = root / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if manifest is not None and manifest.get("status") != "complete":
        raise ValueError("Shared weights require a completed Qwen4 export")

    # The same inode can appear under several phase-relative resource names.
    digests = {}

    def digest(path):
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity not in digests:
            digests[identity] = file_sha256(path)
            after = path.stat()
            if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError(f"Qwen4 resource changed during publication: {path}")
        return digests[identity]

    ple_groups, ple_links = {}, []
    for stage in models:
        owner, _ = ple_resource_manifest(graphs[stage])
        for spec in read_qwen4_exp_ple_specs(graphs[stage]):
            for kind in ("packed", "exponent"):
                declaration = spec[kind]
                path = _safe_asset(owner, declaration)
                key = (kind, declaration["dtype"], tuple(declaration["shape"]), path.stat().st_size, digest(path))
                canonical = ple_groups.setdefault(key, path)
                if not canonical.samefile(path):
                    ple_links.append((canonical, path))

    old_metadata = metadata_path.read_bytes()
    old_manifest = manifest_path.read_bytes() if manifest is not None else None
    backups, created, candidates = [], [], set()

    def backup(path):
        saved = path.with_name(f".{path.name}.{uuid.uuid4().hex}.bak")
        os.link(path, saved)
        backups.append((path, saved))

    try:
        report = None
        if not already_shared:
            report = share_language_weights(
                root, {stage: path.relative_to(root).as_posix() for stage, path in graphs.items()}
            )
            created.extend(root / relative for relative in report["graphs"].values())
            created.extend(weights.values())
            candidates.update(root / relative for relative in report["source_external_files"])
            for stage, graph in graphs.items():
                backup(graph)
                os.replace(root / report["graphs"][stage], graph)
        for canonical, path in ple_links:
            backup(path)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.link")
            try:
                os.link(canonical, temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

        shared = metadata.get("shared_external_data") or (manifest or {}).get("shared_external_data")
        if shared is None:
            shared = {
                "files": {stage: path.relative_to(root).as_posix() for stage, path in weights.items()},
                "bytes": weights["prefill"].stat().st_size,
                "source_file_bytes": report["source_file_bytes"] if report else weights["prefill"].stat().st_size,
                "reused_tensors": report["reused_tensors"] if report else 0,
                "ple_shared_files": len(ple_links),
            }
        metadata["shared_external_data"] = shared
        graph_md5 = {path: hashlib.md5(path.read_bytes()).hexdigest() for path in graphs.values()}
        for key, value in list(metadata.items()):
            if key.endswith("_hmonnx") and isinstance(value, str):
                path = (root / value).resolve()
                if path in graph_md5:
                    metadata[key + "_md5"] = graph_md5[path]
        if manifest is not None:
            manifest["shared_external_data"] = shared
            for record in manifest.get("graphs", {}).values():
                graph = Path(record["hmonnx"])
                graph = (graph if graph.is_absolute() else root / graph).resolve()
                if graph in graph_md5:
                    record["resources"] = {
                        path.relative_to(root).as_posix(): digest(path) for path in sorted(_resources(root, graph))
                    }
            _write_json(manifest_path, manifest)
        validate_onnx_external_locations(root)
        _write_json(metadata_path, metadata)
    except BaseException:
        for path, saved in reversed(backups):
            os.replace(saved, path)
        if metadata_path.read_bytes() != old_metadata:
            metadata_path.write_bytes(old_metadata)
        if old_manifest is not None and manifest_path.read_bytes() != old_manifest:
            manifest_path.write_bytes(old_manifest)
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        for _, saved in backups:
            saved.unlink(missing_ok=True)
    _retire_target_temporaries(root, graphs, metadata, candidates)
    return shared
