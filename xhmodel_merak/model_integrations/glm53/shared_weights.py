"""Stream identical external tensors into one portable language weight file."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import onnx

from xhmodel_merak.model_artifact.files import confined_file
from xhmodel_merak.model_build.packaging import _model_tensors


def share_language_weights(root, graphs, *, directory="language", chunk_size=8 << 20):
    """Write new graphs without modifying sources; caller publishes/removes old files.

    Equality requires dtype, shape, byte length and SHA256, not tensor names.
    Host memory is bounded by the graph metadata and one I/O chunk. Both graphs
    live beside weights.bin, avoiding parent-relative external-data locations.
    """
    root = Path(root).resolve()
    if directory != Path(directory).name or directory in {"", ".", ".."} or chunk_size <= 0:
        raise ValueError("Invalid shared weight directory or chunk size")
    if set(graphs) != {"prefill", "decode"}:
        raise ValueError("Expected prefill and decode graphs")
    paths = {name: confined_file(root, str(value), name) for name, value in graphs.items()}
    models = {name: onnx.load(path, load_external_data=False) for name, path in paths.items()}
    destination = root / directory
    destination.mkdir()  # Never overwrite a previous conversion, including a failed one.
    output_paths = {}
    sources = {}
    spans = {}
    total_bytes = duplicates = tensors = 0
    with ExitStack() as stack:
        output = stack.enter_context((destination / "weights.bin").open("xb"))
        readers = {}

        def chunks(reader, offset, length):
            reader.seek(offset)
            remaining = length
            while remaining:
                data = reader.read(min(chunk_size, remaining))
                if not data:
                    raise ValueError("Truncated ONNX external tensor")
                remaining -= len(data)
                yield data

        for stage, model in models.items():
            for tensor in _model_tensors(model):
                if tensor.data_location != onnx.TensorProto.EXTERNAL:
                    continue
                if tensor.HasField("raw_data"):
                    raise ValueError("External tensor also contains inline raw_data")
                entries = {entry.key: entry.value for entry in tensor.external_data}
                location = entries.get("location", "")
                # Validate the location before joining, so '..' cannot be normalized away.
                source = confined_file(paths[stage].parent, location, "external_data.location")
                if not source.is_relative_to(root):
                    raise ValueError("External tensor escapes the artifact")
                stat = source.stat()
                identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                if source not in readers:
                    readers[source] = stack.enter_context(source.open("rb"))
                    sources[source] = identity
                if sources[source] != identity:
                    raise ValueError(f"External data changed during conversion: {source}")
                offset = int(entries.get("offset", 0))
                length = int(entries.get("length", stat.st_size - offset))
                if offset < 0 or length < 0 or offset + length > stat.st_size:
                    raise ValueError(f"Invalid external tensor range: {tensor.name}")
                digest = hashlib.sha256()
                for data in chunks(readers[source], offset, length):
                    digest.update(data)
                key = (tensor.data_type, tuple(tensor.dims), length, digest.digest())
                shared_offset = spans.get(key)
                if shared_offset is None:
                    padding = (-output.tell()) % 64
                    output.write(b"\0" * padding)
                    shared_offset = output.tell()
                    copied = hashlib.sha256()
                    for data in chunks(readers[source], offset, length):
                        output.write(data)
                        copied.update(data)
                    if copied.digest() != digest.digest():
                        raise ValueError(f"External tensor changed while copying: {tensor.name}")
                    spans[key] = shared_offset
                else:
                    duplicates += 1
                total_bytes += length
                tensors += 1
                tensor.ClearField("external_data")
                for key_name, value in (("location", "weights.bin"), ("offset", shared_offset), ("length", length)):
                    entry = tensor.external_data.add()
                    entry.key, entry.value = key_name, str(value)
                tensor.ClearField("raw_data")
            path = destination / (stage + ".onnx")
            # No tensor payload is materialized in the protobuf.
            path.write_bytes(model.SerializeToString())
            output_paths[stage] = path.relative_to(root).as_posix()
            print(f"Shared weights: {stage}, {tensors} tensors, {output.tell()} bytes", flush=True)
        size = output.tell()
        for source, identity in sources.items():
            stat = source.stat()
            if identity != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns):
                raise ValueError(f"External data changed during conversion: {source}")
    return {
        "graphs": output_paths,
        "source_graphs": [str(path.relative_to(root)) for path in paths.values()],
        "source_external_files": [str(path.relative_to(root)) for path in sources],
        "external_tensor_bytes_before": total_bytes,
        "source_file_bytes": sum(identity[2] for identity in sources.values()),
        "shared_file_bytes": size,
        "external_tensors": tensors,
        "reused_tensors": duplicates,
    }


def compact_glm53_language_weights(root):
    """Finalize an unpublished GLM bundle; refresh graph checksums before cleanup."""
    from xhmodel_merak.model_build.packaging import validate_onnx_external_locations

    root = Path(root).resolve()
    metadata_path = root / "golden_meta_info.json"
    metadata = json.loads(metadata_path.read_text())
    report = share_language_weights(root, {stage: metadata[stage + "_hmonnx"] for stage in ("prefill", "decode")})
    validate_onnx_external_locations(root)
    for stage, relative in report["graphs"].items():
        metadata[stage + "_hmonnx"] = relative
        metadata[stage + "_hmonnx_md5"] = hashlib.md5((root / relative).read_bytes()).hexdigest()
    metadata["glm53_shared_external_data"] = {
        "file": "language/weights.bin",
        "bytes": report["shared_file_bytes"],
        "source_file_bytes": report["source_file_bytes"],
        "reused_tensors": report["reused_tensors"],
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(metadata_path)
    # Do not delete an old data file if another graph still references it.
    old_graphs = set(report["source_graphs"])
    referenced = set()
    for path in root.rglob("*.onnx"):
        if path.relative_to(root).as_posix() in old_graphs:
            continue
        for tensor in _model_tensors(onnx.load(path, load_external_data=False)):
            values = {entry.key: entry.value for entry in tensor.external_data}
            if "location" in values:
                referenced.add((path.parent / values["location"]).resolve())
    for relative in report["source_graphs"] + report["source_external_files"]:
        path = root / relative
        if path not in referenced:
            path.unlink()
    return report
