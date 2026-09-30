#!/usr/bin/env python3
"""Materialize HMONNX V2 parallel-Linear fusion in an ONNX graph.

The HMONNX aligned Linear layout stores the output channel in the last axis:

    qweight:      [in_groups, nshare, out_features]
    scale_or_exp: [in_groups,      1, out_features]

This tool fuses Linears only when they consume the same input and use the same
input SEFP conversion format. It concatenates their quantized weights without
dequantizing them, and replaces each group with:

    Linear -> Split -> TupleGetItem(s)

Unchanged initializers keep referencing the source model's external-data file.
Only the newly fused tensors are written to a separate external-data file.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterable, cast

import onnx
from onnx import TensorProto, helper, numpy_helper

_INPUT_SEFP_ATTRS = (
    "mode",
    "hmfp_act_exp_bit",
    "hmfp_act_man_bit",
    "hmfp_act_nshare",
    "hmfp_act_rounding",
    "hmfp_act_hidden_bit",
    "hmfp_act_max_exp_boost",
)

_LINEAR_COMPATIBILITY_ATTRS = (
    "have_bias",
    "in_features",
    "hmfp_weight_exp_bit",
    "hmfp_weight_man_bit",
    "hmfp_weight_nshare",
    "hmfp_weight_rounding",
    "hmfp_weight_hidden_bit",
    "hmfp_psum_dtype",
    "hmfp_psum_round_mode",
    "output_dtype",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source HMONNX/ONNX graph")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output fused ONNX graph; defaults to <input_stem>_linear_fused.onnx",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output graph and fused external-data file",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run source and fused graphs with HMONNXInferenceV2 and compare outputs",
    )
    parser.add_argument("--verify-seed", type=int, default=0, help="Random verification seed")
    parser.add_argument("--verify-atol", type=float, default=0.0, help="Verification absolute tolerance")
    parser.add_argument("--verify-rtol", type=float, default=0.0, help="Verification relative tolerance")
    return parser.parse_args()


def _attribute_map(node: onnx.NodeProto) -> dict[str, object]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _initializer_map(graph: onnx.GraphProto) -> dict[str, onnx.TensorProto]:
    return {initializer.name: initializer for initializer in graph.initializer}


def _input_sefp_format(node: onnx.NodeProto) -> tuple[object, ...]:
    attributes = _attribute_map(node)
    return tuple(attributes.get(name, 0 if name == "hmfp_act_max_exp_boost" else None) for name in _INPUT_SEFP_ATTRS)


def _linear_fusion_key(
    node: onnx.NodeProto,
    initializers: dict[str, onnx.TensorProto],
) -> tuple[object, ...] | None:
    if node.domain != "ai.houmo.xh2a" or node.op_type != "Linear" or len(node.input) < 3:
        return None
    qweight = initializers.get(node.input[1])
    scale_or_exp = initializers.get(node.input[2])
    if qweight is None or scale_or_exp is None:
        return None

    attributes = _attribute_map(node)
    compatibility_values = tuple(attributes.get(name) for name in _LINEAR_COMPATIBILITY_ATTRS)

    weight = initializers.get(node.input[3]) if len(node.input) > 3 else None
    have_bias = bool(attributes.get("have_bias", 0))
    bias = initializers.get(node.input[4]) if have_bias and len(node.input) > 4 else None
    return (
        node.domain,
        node.op_type,
        node.input[0],
        _input_sefp_format(node),
        *compatibility_values,
        qweight.data_type,
        scale_or_exp.data_type,
        weight is not None,
        weight.data_type if weight is not None else None,
        bias.data_type if bias is not None else None,
    )


def _find_fusion_groups(
    graph: onnx.GraphProto,
    initializers: dict[str, onnx.TensorProto],
) -> list[list[onnx.NodeProto]]:
    groups: dict[tuple[object, ...], list[onnx.NodeProto]] = defaultdict(list)
    for node in graph.node:
        key = _linear_fusion_key(node, initializers)
        if key is not None:
            groups[key].append(node)
    return [group for group in groups.values() if len(group) >= 2]


def _element_size(data_type: int) -> int:
    sizes: dict[int, int] = {
        TensorProto.FLOAT: 4,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.BOOL: 1,
        TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.UINT32: 4,
        TensorProto.UINT64: 8,
        TensorProto.COMPLEX64: 8,
        TensorProto.COMPLEX128: 16,
        TensorProto.BFLOAT16: 2,
    }
    if data_type not in sizes:
        raise ValueError(f"Unsupported tensor data type for streaming fusion: {data_type}")
    return sizes[data_type]


def _external_fields(tensor: onnx.TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


class _ExternalTensorReader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._handles: dict[Path, BinaryIO] = {}
        self._embedded_payloads: dict[str, bytes] = {}

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._embedded_payloads.clear()

    def _storage(self, tensor: onnx.TensorProto) -> tuple[BinaryIO, int, int]:
        fields = _external_fields(tensor)
        location = fields.get("location")
        if tensor.data_location != TensorProto.EXTERNAL or not location:
            raise ValueError(
                f"Fusion tensor {tensor.name!r} is not stored as external data; "
                "this tool intentionally avoids loading large embedded tensors"
            )
        path = (self.model_dir / location).resolve()
        handle = self._handles.get(path)
        if handle is None:
            handle = path.open("rb")
            self._handles[path] = handle
        offset = int(fields.get("offset", "0"))
        expected_length = math.prod(tensor.dims) * _element_size(tensor.data_type)
        length = int(fields.get("length", str(expected_length)))
        if length != expected_length:
            raise ValueError(f"External tensor {tensor.name!r} has length={length}, expected={expected_length}")
        return handle, offset, length

    def read_row(self, tensor: onnx.TensorProto, row: int, row_bytes: int) -> bytes:
        fields = _external_fields(tensor)
        if tensor.data_location != TensorProto.EXTERNAL or not fields.get("location"):
            payload = self._embedded_payloads.get(tensor.name)
            if payload is None:
                payload = numpy_helper.to_array(tensor).tobytes(order="C")
                expected_length = math.prod(tensor.dims) * _element_size(tensor.data_type)
                if len(payload) != expected_length:
                    raise ValueError(
                        f"Embedded tensor {tensor.name!r} has {len(payload)} bytes, expected {expected_length}"
                    )
                self._embedded_payloads[tensor.name] = payload
            row_offset = row * row_bytes
            row_payload = payload[row_offset : row_offset + row_bytes]
            if len(row_payload) != row_bytes:
                raise ValueError(f"Row {row} exceeds embedded tensor {tensor.name!r}")
            return row_payload

        handle, offset, length = self._storage(tensor)
        row_offset = row * row_bytes
        if row_offset + row_bytes > length:
            raise ValueError(f"Row {row} exceeds external tensor {tensor.name!r}")
        handle.seek(offset + row_offset)
        payload = handle.read(row_bytes)
        if len(payload) != row_bytes:
            raise EOFError(f"Short read for {tensor.name!r}: got {len(payload)} bytes, expected {row_bytes}")
        return payload


def _align_output(handle: BinaryIO, alignment: int = 64) -> int:
    padding = (-handle.tell()) % alignment
    if padding:
        handle.write(b"\0" * padding)
    return handle.tell()


def _stream_concat_last_axis(
    tensors: list[onnx.TensorProto],
    reader: _ExternalTensorReader,
    output_handle: BinaryIO,
    *,
    output_name: str,
    external_location: str,
) -> tuple[onnx.TensorProto, int]:
    first = tensors[0]
    if not first.dims:
        raise ValueError(f"Cannot concatenate scalar tensor {first.name!r}")
    prefix = tuple(first.dims[:-1])
    data_type = first.data_type
    for tensor in tensors[1:]:
        if tuple(tensor.dims[:-1]) != prefix:
            raise ValueError(
                f"Cannot fuse {first.name!r} and {tensor.name!r}: "
                f"non-output dimensions differ ({prefix} vs {tuple(tensor.dims[:-1])})"
            )
        if tensor.data_type != data_type:
            raise ValueError(f"Cannot fuse {first.name!r} and {tensor.name!r}: data types differ")

    element_size = _element_size(data_type)
    row_count = math.prod(prefix)
    row_bytes = [tensor.dims[-1] * element_size for tensor in tensors]
    output_dims = [*prefix, sum(tensor.dims[-1] for tensor in tensors)]
    output_length = math.prod(output_dims) * element_size
    output_offset = _align_output(output_handle)

    for row in range(row_count):
        for tensor, tensor_row_bytes in zip(tensors, row_bytes, strict=True):
            output_handle.write(reader.read_row(tensor, row, tensor_row_bytes))

    if output_handle.tell() - output_offset != output_length:
        raise RuntimeError(f"Fused tensor {output_name!r} was written with an unexpected length")

    fused = onnx.TensorProto()
    fused.name = output_name
    fused.data_type = data_type
    fused.dims.extend(output_dims)
    fused.data_location = TensorProto.EXTERNAL
    for key, value in (
        ("location", external_location),
        ("offset", str(output_offset)),
        ("length", str(output_length)),
    ):
        entry = fused.external_data.add()
        entry.key = key
        entry.value = value
    return fused, output_length


def _validate_concatenated_tensor(
    sources: list[onnx.TensorProto],
    fused: onnx.TensorProto,
    reader: _ExternalTensorReader,
) -> None:
    element_size = _element_size(fused.data_type)
    row_count = math.prod(fused.dims[:-1])
    source_row_bytes = [tensor.dims[-1] * element_size for tensor in sources]
    fused_row_bytes = fused.dims[-1] * element_size
    for row in range(row_count):
        expected = b"".join(
            reader.read_row(tensor, row, row_bytes) for tensor, row_bytes in zip(sources, source_row_bytes, strict=True)
        )
        actual = reader.read_row(fused, row, fused_row_bytes)
        if actual != expected:
            raise ValueError(f"Fused tensor {fused.name!r} differs from its sources at row {row}")


def _validate_fused_tensor_equivalence(
    groups: list[list[onnx.NodeProto]],
    initializers: dict[str, onnx.TensorProto],
    fused_initializers: list[tuple[onnx.TensorProto, onnx.TensorProto]],
    model_dir: Path,
) -> None:
    reader = _ExternalTensorReader(model_dir)
    try:
        for group, (fused_qweight, fused_scale_or_exp) in zip(groups, fused_initializers, strict=True):
            _validate_concatenated_tensor(
                [initializers[node.input[1]] for node in group],
                fused_qweight,
                reader,
            )
            _validate_concatenated_tensor(
                [initializers[node.input[2]] for node in group],
                fused_scale_or_exp,
                reader,
            )
    finally:
        reader.close()


def _random_verification_input(input_info: Any, generator: Any) -> Any:
    import torch

    shape = tuple(
        int(dimension) if isinstance(dimension, int) and dimension > 0 else 1 for dimension in input_info.shape
    )
    dtype = input_info.dtype
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=generator, dtype=torch.int64).to(dtype)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=generator, dtype=dtype)
    return torch.randint(0, 8, shape, generator=generator, dtype=torch.int64).to(dtype)


def _as_output_tuple(outputs: Any) -> tuple[Any, ...]:
    return outputs if isinstance(outputs, tuple) else (outputs,)


def _runtime_input_feed(session: Any, base_input_feed: dict[str, Any], device: Any) -> dict[str, Any]:
    from xhquant.core import CacheTensor

    cache_input_names = {
        module.input_names[3]
        for module in session.node_modules
        if type(module).__name__ in {"LLMCache", "KVcache"} and len(module.input_names) > 3
    }
    return {
        name: CacheTensor(value.to(device).clone()) if name in cache_input_names else value.to(device).clone()
        for name, value in base_input_feed.items()
    }


def _verify_runtime_outputs(
    input_path: Path,
    output_path: Path,
    *,
    seed: int,
    atol: float,
    rtol: float,
) -> None:
    import gc

    import torch
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import (
        HMONNXInferenceConfig,
        HMONNXInferenceV2,
    )

    device: str | int = 0 if torch.cuda.is_available() else "cpu"
    execution_device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    source_session = HMONNXInferenceV2(
        str(input_path),
        session_config=HMONNXInferenceConfig(exec_devices=[device]),
    )
    assert source_session.graph_module is not None
    source_session.graph_module.to(execution_device)
    source_contract = [(info.name, tuple(info.shape), info.dtype) for info in source_session.inputs]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base_input_feed = {info.name: _random_verification_input(info, generator) for info in source_session.inputs}
    source_input_feed = _runtime_input_feed(source_session, base_input_feed, execution_device)
    source_outputs = tuple(output.detach().cpu() for output in _as_output_tuple(source_session.run(source_input_feed)))
    del source_input_feed, source_session
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fused_session = HMONNXInferenceV2(
        str(output_path),
        session_config=HMONNXInferenceConfig(exec_devices=[device]),
    )
    assert fused_session.graph_module is not None
    fused_session.graph_module.to(execution_device)
    fused_contract = [(info.name, tuple(info.shape), info.dtype) for info in fused_session.inputs]
    if fused_contract != source_contract:
        raise ValueError(
            "Fused graph input contract differs from the source graph: "
            f"source={source_contract}, fused={fused_contract}"
        )
    fused_input_feed = _runtime_input_feed(fused_session, base_input_feed, execution_device)
    fused_outputs = tuple(output.detach().cpu() for output in _as_output_tuple(fused_session.run(fused_input_feed)))
    if len(fused_outputs) != len(source_outputs):
        raise ValueError(f"Output count differs: source={len(source_outputs)}, fused={len(fused_outputs)}")

    for index, (source, fused) in enumerate(zip(source_outputs, fused_outputs, strict=True)):
        if source.shape != fused.shape or source.dtype != fused.dtype:
            raise ValueError(
                f"Output {index} contract differs: source=({tuple(source.shape)}, {source.dtype}), "
                f"fused=({tuple(fused.shape)}, {fused.dtype})"
            )
        try:
            torch.testing.assert_close(fused, source, atol=atol, rtol=rtol)
        except AssertionError as error:
            source_float = source.to(torch.float32)
            fused_float = fused.to(torch.float32)
            max_abs_error = (fused_float - source_float).abs().max().item()
            raise ValueError(
                f"Output {index} is not equivalent: max_abs_error={max_abs_error:.6e}, atol={atol}, rtol={rtol}"
            ) from error
    print(f"runtime_equivalence=passed outputs={len(source_outputs)} seed={seed} atol={atol} rtol={rtol}")


def _set_int_attribute(node: onnx.NodeProto, name: str, value: int) -> None:
    for attribute in node.attribute:
        if attribute.name == name:
            attribute.CopyFrom(helper.make_attribute(name, value))
            return
    node.attribute.append(helper.make_attribute(name, value))


def _unique_name(base: str, used: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _replace_nodes(
    model: onnx.ModelProto,
    groups: list[list[onnx.NodeProto]],
    fused_initializers: list[tuple[onnx.TensorProto, onnx.TensorProto]],
) -> set[str]:
    graph = model.graph
    used_node_names = {node.name for node in graph.node if node.name}
    used_tensor_names = {name for node in graph.node for name in (*node.input, *node.output) if name}
    used_tensor_names.update(initializer.name for initializer in graph.initializer)

    group_by_first = {id(group[0]): (index, group) for index, group in enumerate(groups, 1)}
    removed_node_ids = {id(node) for group in groups for node in group}
    removable_initializers: set[str] = set()
    replacement_nodes: list[onnx.NodeProto] = []

    for node in graph.node:
        first_group = group_by_first.get(id(node))
        if first_group is None:
            if id(node) not in removed_node_ids:
                replacement_nodes.append(node)
            continue

        group_index, group = first_group
        qweight, scale_or_exp = fused_initializers[group_index - 1]
        output_sizes = [cast(int, _attribute_map(member)["out_features"]) for member in group]
        original_outputs = [member.output[0] for member in group]
        removable_initializers.update(member.input[1] for member in group)
        removable_initializers.update(member.input[2] for member in group)

        tag = f"parallel_fused_{group_index:03d}"
        linear_output = _unique_name(f"{node.output[0]}__{tag}", used_tensor_names)
        split_output = _unique_name(f"{linear_output}__tuple", used_tensor_names)

        fused_linear = onnx.NodeProto()
        fused_linear.CopyFrom(node)
        fused_linear.name = _unique_name(f"{node.name}__{tag}", used_node_names)
        del fused_linear.input[:]
        fused_linear.input.extend([node.input[0], qweight.name, scale_or_exp.name])
        del fused_linear.output[:]
        fused_linear.output.append(linear_output)
        _set_int_attribute(fused_linear, "out_features", sum(output_sizes))
        replacement_nodes.append(fused_linear)

        split_node = helper.make_node(
            "Split",
            [linear_output],
            [split_output],
            name=_unique_name(f"{fused_linear.name}__split", used_node_names),
            domain="ai.houmo.xh2a",
            axis=-1,
            indices_or_sections=output_sizes,
        )
        replacement_nodes.append(split_node)

        for output_index, output_name in enumerate(original_outputs):
            getitem = helper.make_node(
                "TupleGetItem",
                [split_output],
                [output_name],
                name=_unique_name(f"{fused_linear.name}__getitem_{output_index}", used_node_names),
                domain="ai.houmo.xh2a",
                index=output_index,
            )
            replacement_nodes.append(getitem)

    del graph.node[:]
    graph.node.extend(replacement_nodes)

    used_initializer_names = {input_name for node in graph.node for input_name in node.input if input_name}
    retained_initializers = [
        initializer
        for initializer in graph.initializer
        if initializer.name not in removable_initializers or initializer.name in used_initializer_names
    ]
    del graph.initializer[:]
    graph.initializer.extend(retained_initializers)
    for qweight, scale_or_exp in fused_initializers:
        graph.initializer.extend([qweight, scale_or_exp])
    return removable_initializers


def _validate_topology(model: onnx.ModelProto) -> None:
    graph = model.graph
    available = {value.name for value in graph.input}
    available.update(initializer.name for initializer in graph.initializer)
    available.update(initializer.values.name for initializer in graph.sparse_initializer)
    produced: set[str] = set()
    for node_index, node in enumerate(graph.node):
        missing = [name for name in node.input if name and name not in available]
        if missing:
            raise ValueError(f"Node {node_index} {node.name!r} is not topologically sorted; missing inputs: {missing}")
        for output in node.output:
            if not output:
                continue
            if output in produced:
                raise ValueError(f"Tensor {output!r} is produced more than once")
            produced.add(output)
            available.add(output)
    missing_outputs = [value.name for value in graph.output if value.name not in available]
    if missing_outputs:
        raise ValueError(f"Graph outputs are not produced: {missing_outputs}")


def _validate_external_ranges(model: onnx.ModelProto, model_path: Path) -> None:
    sizes: dict[Path, int] = {}
    for tensor in model.graph.initializer:
        fields = _external_fields(tensor)
        location = fields.get("location")
        if tensor.data_location != TensorProto.EXTERNAL or not location:
            continue
        path = (model_path.parent / location).resolve()
        size = sizes.setdefault(path, path.stat().st_size)
        offset = int(fields.get("offset", "0"))
        expected = math.prod(tensor.dims) * _element_size(tensor.data_type)
        length = int(fields.get("length", str(expected)))
        if length != expected:
            raise ValueError(f"Initializer {tensor.name!r} external length={length}, expected={expected}")
        if offset < 0 or offset + length > size:
            raise ValueError(
                f"Initializer {tensor.name!r} range [{offset}, {offset + length}) exceeds {path} size {size}"
            )


def _set_metadata(model: onnx.ModelProto, values: dict[str, str]) -> None:
    properties = {entry.key: entry.value for entry in model.metadata_props}
    properties.update(values)
    del model.metadata_props[:]
    for key, value in sorted(properties.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_outputs_absent(paths: Iterable[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {rendered}; pass --force to replace")


def main() -> None:
    args = _parse_args()
    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_linear_fused.onnx")
    )
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path == output_path:
        raise ValueError("Input and output ONNX paths must differ")
    if input_path.parent != output_path.parent:
        raise ValueError(
            "Output ONNX must be beside the input so unchanged initializers can keep "
            "referencing the source external-data file"
        )

    external_path = output_path.with_name(f"{output_path.stem}_external_data")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_outputs_absent((output_path, external_path), args.force)

    output_tmp = output_path.with_name(f".{output_path.name}.tmp")
    external_tmp = external_path.with_name(f".{external_path.name}.tmp")
    for temporary in (output_tmp, external_tmp):
        if temporary.exists():
            temporary.unlink()

    model = onnx.load(input_path, load_external_data=False)
    initializers = _initializer_map(model.graph)
    groups = _find_fusion_groups(model.graph, initializers)
    if not groups:
        raise ValueError("No compatible parallel Linear groups were found")

    fused_initializers: list[tuple[onnx.TensorProto, onnx.TensorProto]] = []
    reader = _ExternalTensorReader(input_path.parent)
    total_fused_bytes = 0
    try:
        with external_tmp.open("wb") as output_handle:
            for group_index, group in enumerate(groups, 1):
                for node in group:
                    attributes = _attribute_map(node)
                    if len(node.input) != 3 or bool(attributes.get("have_bias", 0)):
                        raise ValueError(
                            "This graph-materialization tool currently requires bias-free, "
                            f"three-input Linears; group {group_index} contains {node.name!r}"
                        )
                qweights = [initializers[node.input[1]] for node in group]
                scales_or_exps = [initializers[node.input[2]] for node in group]
                base_name = f"parallel_linear_fusion.{group_index:03d}"
                fused_qweight, qweight_bytes = _stream_concat_last_axis(
                    qweights,
                    reader,
                    output_handle,
                    output_name=f"{base_name}.qweight",
                    external_location=external_path.name,
                )
                fused_scale_or_exp, scale_bytes = _stream_concat_last_axis(
                    scales_or_exps,
                    reader,
                    output_handle,
                    output_name=f"{base_name}.scale_or_exp",
                    external_location=external_path.name,
                )
                total_fused_bytes += qweight_bytes + scale_bytes
                fused_initializers.append((fused_qweight, fused_scale_or_exp))
            output_handle.flush()
            os.fsync(output_handle.fileno())
    except Exception:
        if external_tmp.exists():
            external_tmp.unlink()
        raise
    finally:
        reader.close()

    original_linear_count = sum(
        node.domain == "ai.houmo.xh2a" and node.op_type == "Linear" for node in model.graph.node
    )
    fused_source_linear_count = sum(len(group) for group in groups)
    _replace_nodes(model, groups, fused_initializers)
    final_linear_count = sum(node.domain == "ai.houmo.xh2a" and node.op_type == "Linear" for node in model.graph.node)
    expected_final_linear_count = original_linear_count - fused_source_linear_count + len(groups)
    if final_linear_count != expected_final_linear_count:
        raise RuntimeError(f"Linear count mismatch: got {final_linear_count}, expected {expected_final_linear_count}")

    _set_metadata(
        model,
        {
            "hmonnx.parallel_linear_fusion": "materialized",
            "hmonnx.parallel_linear_fusion.groups": str(len(groups)),
            "hmonnx.parallel_linear_fusion.source_linears": str(fused_source_linear_count),
            "hmonnx.parallel_linear_fusion.removed_linears": str(fused_source_linear_count - len(groups)),
        },
    )
    _validate_topology(model)

    try:
        onnx.save_model(model, output_tmp)
        os.replace(external_tmp, external_path)
        _validate_fused_tensor_equivalence(
            groups,
            initializers,
            fused_initializers,
            output_path.parent,
        )
        _validate_external_ranges(model, output_tmp)
        onnx.checker.check_model(str(output_tmp), full_check=False)
        reloaded = onnx.load(output_tmp, load_external_data=False)
        _validate_topology(reloaded)
        os.replace(output_tmp, output_path)
        if args.verify:
            _verify_runtime_outputs(
                input_path,
                output_path,
                seed=args.verify_seed,
                atol=args.verify_atol,
                rtol=args.verify_rtol,
            )
    except Exception:
        if output_tmp.exists():
            output_tmp.unlink()
        if external_tmp.exists():
            external_tmp.unlink()
        if output_path.exists():
            output_path.unlink()
        if external_path.exists():
            external_path.unlink()
        raise

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"external_data={external_path}")
    print(f"fusion_groups={len(groups)}")
    print(f"source_linears={fused_source_linear_count}")
    print(f"removed_linears={fused_source_linear_count - len(groups)}")
    print(f"linear_count={original_linear_count}->{final_linear_count}")
    print(f"fused_tensor_bytes={total_fused_bytes}")
    print(f"external_file_bytes={external_path.stat().st_size}")
    print(f"external_file_sha256={_sha256(external_path)}")


if __name__ == "__main__":
    main()
