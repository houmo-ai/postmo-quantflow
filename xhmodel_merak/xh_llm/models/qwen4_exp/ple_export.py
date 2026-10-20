"""Stream PLE deployment weights into a compact, portable SEFP4 HMONNX."""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, external_data_helper, helper, numpy_helper


_DOMAIN = "ai.houmo.xh2a"
_OP = "Qwen4PLESEFP4Gather"
_COPY_BYTES = 16 * 1024 * 1024


def _external_range(tensor, root: Path) -> tuple[Path, int, int]:
    info = {item.key: item.value for item in tensor.external_data}
    location = info.get("location", "")
    path = (root / location).resolve()
    if not location or Path(location).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe external data location for {tensor.name}: {location!r}")
    offset = int(info.get("offset", 0))
    length = int(info.get("length", path.stat().st_size - offset))
    dtype = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.data_type))
    expected = int(np.prod(tensor.dims, dtype=object)) * dtype.itemsize
    if offset < 0 or length != expected or offset + length > path.stat().st_size:
        raise ValueError(f"Invalid external tensor range for {tensor.name}")
    return path, offset, length


def _set_external(tensor, location: str, offset: int, length: int) -> None:
    tensor.ClearField("raw_data")
    for field in ("float_data", "int32_data", "int64_data", "double_data", "uint64_data"):
        tensor.ClearField(field)
    tensor.ClearField("external_data")
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in (("location", location), ("offset", offset), ("length", length)):
        entry = tensor.external_data.add()
        entry.key, entry.value = key, str(value)


def _copy_tensor(tensor, root: Path, stream, location: str) -> int:
    offset = stream.tell()
    if tensor.data_location == TensorProto.EXTERNAL:
        path, start, length = _external_range(tensor, root)
        with path.open("rb") as source:
            source.seek(start)
            remaining = length
            while remaining:
                block = source.read(min(remaining, _COPY_BYTES))
                if not block:
                    raise ValueError(f"Truncated external tensor: {tensor.name}")
                stream.write(block)
                remaining -= len(block)
    elif tensor.data_type != TensorProto.STRING:
        data = tensor.raw_data or numpy_helper.to_array(tensor).tobytes()
        stream.write(data)
        length = len(data)
    else:
        return 0
    _set_external(tensor, location, offset, length)
    return length


def _rows(tensor, root: Path, chunk_rows: int):
    import torch

    rows, width = tensor.dims
    if tensor.data_location == TensorProto.EXTERNAL:
        path, offset, _ = _external_range(tensor, root)
        array = np.memmap(path, mode="r", offset=offset, dtype="<u2", shape=(rows, width))
    else:
        array = np.frombuffer(tensor.raw_data, dtype="<u2").reshape(rows, width) if tensor.raw_data else (
            numpy_helper.to_array(tensor).view(np.uint16).reshape(rows, width)
        )
    dtype = torch.float16 if tensor.data_type == TensorProto.FLOAT16 else torch.bfloat16
    for start in range(0, rows, chunk_rows):
        # Copy a bounded chunk: torch must never mutate a read-only mmap view.
        yield torch.from_numpy(np.array(array[start:start + chunk_rows], copy=True)).view(dtype)


def extract_qwen4_exp_ple_specs(native_model) -> list[dict]:
    """Read the native NGram hashing contract before the export rewrites its graph."""
    specs = []
    for module in native_model.modules():
        if not all(hasattr(module, key) for key in (
            "layer_idx", "layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets", "ngram_size",
            "heads_per_ngram", "eos_token_id", "ngram_embedding",
        )):
            continue
        specs.append({
            "layer_idx": int(module.layer_idx), "ngram_size": int(module.ngram_size),
            "heads_per_ngram": int(module.heads_per_ngram), "eos_token_id": int(module.eos_token_id),
            "multipliers": [int(v) for v in module.layer_multipliers.detach().cpu().tolist()],
            "vocab_sizes": [int(v) for v in module.ngram_heads_vocab_sizes.detach().cpu().tolist()],
            "offsets": [int(v) for v in module.ngram_heads_offsets.detach().cpu().tolist()],
        })
    if len({s["layer_idx"] for s in specs}) != len(specs):
        raise ValueError("Duplicate native PLE layer indices")
    return specs


def _safe_asset(root: Path, value: dict) -> Path:
    relative = Path(value["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe PLE asset path: {relative}")
    shape = value["shape"]
    if not shape or any(not isinstance(v, int) or v < 1 for v in shape):
        raise ValueError("Invalid PLE asset shape")
    dtype = np.dtype(value["dtype"])
    if path.stat().st_size != int(np.prod(shape, dtype=object)) * dtype.itemsize:
        raise ValueError(f"Invalid PLE asset byte length: {path}")
    return path


def read_qwen4_exp_ple_specs(onnx_path: str | Path) -> list[dict]:
    path = Path(onnx_path).resolve()
    return _metadata_specs(onnx.load(path, load_external_data=False), path.parent)


def _metadata_specs(model, root: Path) -> list[dict]:
    values = [entry.value for entry in model.metadata_props if entry.key == "qwen4_exp_ple"]
    if not values:
        return []
    if len(values) != 1:
        raise ValueError("Duplicate qwen4_exp_ple metadata")
    metadata = json.loads(values[0])
    if metadata.get("version") != 1 or not isinstance(metadata.get("layers"), list):
        raise ValueError("Invalid qwen4_exp_ple metadata")
    inputs = {value.name: value for value in model.graph.input}
    seen = set()
    for spec in metadata["layers"]:
        width = int(spec["embedding_dim"])
        dtype = {"float16": TensorProto.FLOAT16, "bfloat16": TensorProto.BFLOAT16}.get(spec["output_dtype"])
        name = spec["input_name"]
        if name in seen or name != f"ple_embeddings_{spec['layer_idx']}" or name not in inputs or dtype is None:
            raise ValueError("Invalid external PLE input contract")
        seen.add(name)
        shape = inputs[name].type.tensor_type.shape.dim
        if (inputs[name].type.tensor_type.elem_type != dtype or len(shape) != 4 or shape[-1].dim_value != width
                or shape[-2].dim_value != (int(spec["ngram_size"]) - 1) * int(spec["heads_per_ngram"])):
            raise ValueError("External PLE input shape/dtype mismatch")
        _validate_hash_spec(spec, spec["packed"]["shape"][0])
        groups = (width + 63) // 64
        rows = spec["packed"]["shape"][0]
        if (spec["packed"]["dtype"] != "int8" or spec["exponent"]["dtype"] != "int16"
                or spec["packed"]["shape"] != [rows, groups, 32]
                or spec["exponent"]["shape"] != [rows, groups, 1]):
            raise ValueError("Invalid external PLE packed layout")
        for key in ("packed", "exponent"):
            _safe_asset(root, spec[key])
    if metadata["layers"] and [value.name for value in model.graph.input][-len(metadata["layers"]):] != [
        spec["input_name"] for spec in metadata["layers"]
    ]:
        raise ValueError("External PLE inputs must be appended after original inputs")
    return metadata["layers"]


def _validate_hash_spec(spec: dict, rows: int) -> None:
    layer = int(spec["layer_idx"])
    ngram = int(spec["ngram_size"])
    heads = int(spec["heads_per_ngram"])
    sizes, offsets = spec["vocab_sizes"], spec["offsets"]
    if (layer < 0 or ngram < 2 or heads < 1 or len(spec["multipliers"]) != ngram
            or len(sizes) != (ngram - 1) * heads or len(offsets) != len(sizes)
            or any(int(size) <= 0 for size in sizes)
            or any(int(offset) < 0 or int(offset) + int(size) > rows
                   for offset, size in zip(offsets, sizes, strict=True))):
        raise ValueError("Invalid native PLE hash contract")


def _set_metadata(model, specs) -> None:
    entries = [entry for entry in model.metadata_props if entry.key != "qwen4_exp_ple"]
    del model.metadata_props[:]
    model.metadata_props.extend(entries)
    if specs:
        entry = model.metadata_props.add()
        entry.key, entry.value = "qwen4_exp_ple", json.dumps({"version": 1, "layers": specs})


def _prune_hash_chain(model, removed_nodes, roots, removed_weights) -> None:
    producers = {output: node for node in model.graph.node for output in node.output}
    ancestors = set()
    pending = list(roots)
    while pending:
        value = pending.pop()
        node = producers.get(value)
        if node is not None and id(node) not in ancestors:
            ancestors.add(id(node))
            pending.extend(node.input)
    # Identity of protobuf wrappers is stable only while references are retained.
    nodes = [node for node in model.graph.node if node not in removed_nodes]
    protected = {value.name for value in model.graph.output}
    removed_inputs = set(removed_weights)
    while True:
        used = protected | {name for node in nodes for name in node.input}
        dead = [node for node in nodes if id(node) in ancestors
                and "cache" not in node.op_type.lower() and not node.op_type.startswith("GDR")
                and node.op_type not in {"GatedDeltaNet", "GatedDeltaRule"}
                and not any(name in used for name in node.output)]
        if not dead:
            break
        for node in dead:
            removed_inputs.update(node.input)
            nodes.remove(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = protected | {value.name for value in model.graph.input} | {name for node in nodes for name in node.input}
    retained = [t for t in model.graph.initializer if t.name not in removed_weights
                and not (t.name in removed_inputs and t.name not in used)]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    available = used | {name for node in nodes for name in node.output} | {t.name for t in retained}
    values = [v for v in model.graph.value_info if v.name in available]
    del model.graph.value_info[:]
    model.graph.value_info.extend(values)


def export_ple_sefp4_hmonnx(
    source_onnx: str | Path,
    output_onnx: str | Path,
    *,
    ple_specs: list[dict] | None = None,
    chunk_rows: int = 16384,
    quant_device: str = "cpu",
) -> dict:
    """Move PLE lookup to a host input and stream raw SEFP4 sidecar tables.

    Accepts native FP16/BF16 Gather graphs and previously packed custom-op graphs.
    No custom PLE operation or table remains in the resulting executable graph.
    Source assets are never modified; graph publication is the final atomic step.
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    started = time.monotonic()
    source, output = Path(source_onnx).resolve(), Path(output_onnx).resolve()
    model = onnx.load(source, load_external_data=False)
    for tensor in external_data_helper._get_all_tensors(model):
        if tensor.data_location == TensorProto.EXTERNAL:
            _external_range(tensor, source.parent)
    current_specs = _metadata_specs(model, source.parent)
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    candidates = []
    for node in model.graph.node:
        if node.op_type == _OP:
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if node.domain != _DOMAIN or len(node.input) != 3 or any(
                attrs.get(key) != value for key, value in (("nshare", 64), ("manbit", 3), ("keep_bit_for_exp", 1))
            ):
                raise ValueError("Invalid legacy PLE packed operator")
            packed, exp = tensors.get(node.input[0]), tensors.get(node.input[1])
            width = int(attrs.get("embedding_dim", 0))
            dtype = int(attrs.get("output_dtype", TensorProto.FLOAT16))
            if packed is None or exp is None or width < 1 or dtype not in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
                raise ValueError("Invalid legacy PLE packed weights")
            rows = packed.dims[0] if len(packed.dims) == 3 else 0
            groups = (width + 63) // 64
            if (packed.data_type != TensorProto.INT8 or exp.data_type != TensorProto.INT16
                    or list(packed.dims) != [rows, groups, 32] or list(exp.dims) != [rows, groups, 1]):
                raise ValueError("Invalid legacy PLE packed shapes")
            if any(len([user for user in model.graph.node if tensor.name in user.input]) != 1
                   for tensor in (packed, exp)):
                raise ValueError("Legacy PLE weights have multiple consumers")
            candidates.append((node, packed, exp, rows, width, dtype, node.input[2]))
    legacy_weights = {entry[1].name for entry in candidates}
    for tensor in model.graph.initializer:
        if ("ple" not in tensor.name or "ngram_embedding" not in tensor.name or tensor.name in legacy_weights
                or tensor.data_type not in (TensorProto.FLOAT16, TensorProto.BFLOAT16) or len(tensor.dims) != 2):
            continue
        users = [node for node in model.graph.node if tensor.name in node.input]
        if len(users) != 1 or users[0].op_type != "Gather" or users[0].domain not in ("", "ai.onnx", _DOMAIN):
            raise ValueError(f"PLE table must have exactly one ordinary Gather consumer: {tensor.name}")
        node = users[0]
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if len(node.input) != 2 or node.input[0] != tensor.name or attrs.get("axis", 0) != 0:
            raise ValueError("PLE Gather must use axis 0")
        candidates.append((node, tensor, None, *tensor.dims, tensor.data_type, node.input[1]))
    if not candidates and source == output:
        return {"status": "already_external_ple" if current_specs else "no_ple", "layers": current_specs,
                "elapsed_seconds": time.monotonic() - started}
    output.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    data_name = f"{output.stem}.ple_external.{token}.data"
    paths = []
    temporary_graph = None
    published = False
    report = {"status": "externalized" if candidates else "copied", "source": str(source), "output": str(output),
              "tables": [], "layers": [], "chunk_rows": chunk_rows, "quant_device": str(quant_device)}
    try:
        specs_by_layer = {int(s["layer_idx"]): dict(s) for s in (ple_specs or [])}
        removed, roots, removed_weights = [], [], set()
        values = {value.name: value for value in list(model.graph.value_info) + list(model.graph.output)}
        original_inputs = {value.name for value in model.graph.input}
        for node, weight, exp, rows, width, dtype, ids in candidates:
            match = re.search(r"layers[_\.](\d+)[_\.]ple", weight.name)
            if match is None:
                match = re.search(r"layers.*?modules[_\.]+(\d+)[_\.]+ple", weight.name)
            if match is None or int(match.group(1)) not in specs_by_layer:
                raise ValueError(f"Native PLE hash spec is required for {weight.name}")
            layer = int(match.group(1))
            spec = specs_by_layer[layer]
            _validate_hash_spec(spec, rows)
            input_name = f"ple_embeddings_{layer}"
            if input_name in original_inputs or weight.name in original_inputs:
                raise ValueError(f"Conflicting PLE graph input: {input_name}")
            name_prefix = f"{output.stem}.ple{layer}.{token}"
            packed_path = output.parent / f"{name_prefix}.packed.bin"
            exp_path = output.parent / f"{name_prefix}.exponent.bin"
            paths.extend((packed_path, exp_path))
            with packed_path.open("xb") as packed_file, exp_path.open("xb") as exp_file:
                if exp is not None:
                    _copy_tensor(weight, source.parent, packed_file, packed_path.name)
                    _copy_tensor(exp, source.parent, exp_file, exp_path.name)
                else:
                    from .ple_sefp4 import quantize_ple_sefp4_rows
                    for chunk in _rows(weight, source.parent, chunk_rows):
                        packed, exponent = quantize_ple_sefp4_rows(chunk.to(quant_device))
                        packed_file.write(packed.detach().cpu().numpy().tobytes())
                        exp_file.write(exponent.detach().cpu().numpy().astype("<i2", copy=False).tobytes())
                for stream in (packed_file, exp_file):
                    stream.flush()
                    os.fsync(stream.fileno())
            groups = (width + 63) // 64
            spec.update(input_name=input_name, embedding_dim=width,
                        output_dtype="float16" if dtype == TensorProto.FLOAT16 else "bfloat16",
                        packed={"path": packed_path.name, "shape": [rows, groups, 32], "dtype": "int8"},
                        exponent={"path": exp_path.name, "shape": [rows, groups, 1], "dtype": "int16"})
            value = values.get(node.output[0])
            if value is None:
                raise ValueError(f"Missing PLE Gather output shape: {node.name}")
            new_input = onnx.ValueInfoProto()
            new_input.CopyFrom(value)
            new_input.name = input_name
            dims = new_input.type.tensor_type.shape.dim
            if (new_input.type.tensor_type.elem_type != dtype or len(dims) != 4
                    or dims[-1].dim_value != width
                    or dims[-2].dim_value != (int(spec["ngram_size"]) - 1) * int(spec["heads_per_ngram"])):
                raise ValueError("PLE Gather output shape/dtype mismatch")
            model.graph.input.append(new_input)
            old_output = node.output[0]
            for user in model.graph.node:
                if user is not node:
                    for i, name in enumerate(user.input):
                        if name == old_output:
                            user.input[i] = input_name
            if any(value.name == old_output for value in model.graph.output):
                model.graph.node.append(helper.make_node("Identity", [input_name], [old_output],
                                                        name=f"ple_host_output_{layer}"))
            removed.append(node)
            roots.append(ids)
            removed_weights.add(weight.name)
            if exp is not None:
                removed_weights.add(exp.name)
            report["layers"].append(spec)
            report["tables"].append({"name": weight.name, "shape": [rows, width], "source_bytes": rows * width * 2,
                                     "sefp4_bytes": rows * groups * 34, "reused_packed_bytes": exp is not None})
        _prune_hash_chain(model, removed, roots, removed_weights)
        for spec in current_specs:
            spec = json.loads(json.dumps(spec))
            for key in ("packed", "exponent"):
                original = _safe_asset(source.parent, spec[key])
                copied = output.parent / f"{output.stem}.ple{spec['layer_idx']}.{token}.{key}.bin"
                paths.append(copied)
                with original.open("rb") as src, copied.open("xb") as dst:
                    while block := src.read(_COPY_BYTES):
                        dst.write(block)
                spec[key]["path"] = copied.name
            report["layers"].append(spec)
        _set_metadata(model, report["layers"])
        data_path = output.parent / data_name
        paths.append(data_path)
        with data_path.open("xb") as stream:
            for tensor in external_data_helper._get_all_tensors(model):
                _copy_tensor(tensor, source.parent, stream, data_name)
            stream.flush()
            os.fsync(stream.fileno())
        _metadata_specs(model, output.parent)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as handle:
            temporary_graph = Path(handle.name)
            handle.write(model.SerializeToString())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_graph, output)
        published = True
        report.update(external_data=data_name, external_bytes=data_path.stat().st_size,
                      elapsed_seconds=time.monotonic() - started)
        return report
    except BaseException:
        if not published:
            for path in paths:
                path.unlink(missing_ok=True)
        if temporary_graph is not None:
            temporary_graph.unlink(missing_ok=True)
        raise
