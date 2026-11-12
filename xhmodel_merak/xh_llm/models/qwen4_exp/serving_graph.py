"""Publish Qwen4's explicit cache ABI and package-local external resources."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any


def convert_qwen4_exp_kv_to_llm_cache(
    input_path: str | Path,
    output_path: str | Path,
    *,
    context_length: int,
    num_kv_heads: int,
    head_dim: int,
) -> str:
    """Publish SDK-standard KVcache nodes with explicit in-place K/V storage.

    The runtime class and existing ABI remain named LLMCache; XH2 HMONNX
    uses KVcache, matching the SDK exporter and DeepSeek graphs.
    In-place updates are enabled on the SDK runtime, not by an ONNX attribute.
    A negative attention limit selects full storage rather than a sliding view.
    """
    import re

    import onnx
    from onnx import helper

    source, destination = Path(input_path), Path(output_path)
    model = onnx.load(str(source), load_external_data=False)
    if any(node.op_type == "PageKVCache" for node in model.graph.node):
        raise ValueError("explicit LLMCache conversion requires the raw graph, not PageKVCache")
    pattern = re.compile(r"^past_(key|value)_cache_(\d+)$")
    indexer_pattern = re.compile(r"^qsa_indexer_(?:key|pool)_cache_\d+$")
    graph_inputs = {value.name: value for value in model.graph.input}
    producers = {name: node for node in model.graph.node for name in node.output}
    writers: dict[tuple[str, int], Any] = {}
    indexer_writers: set[str] = set()
    for node in model.graph.node:
        if node.op_type not in {"KVcache", "LLMCache"} or len(node.input) < 4:
            continue
        name = node.input[3]
        visited: set[str] = set()
        while name not in graph_inputs and name not in visited:
            visited.add(name)
            alias = producers.get(name)
            if alias is None or alias.op_type not in {"Identity", "HMIdentity"}:
                break
            name = alias.input[0]
        if indexer_pattern.fullmatch(name):
            attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            if name not in graph_inputs or attrs.get("axis", 2) != 2 or attrs.get("only_handle_old_cache", 0):
                raise ValueError(f"Qwen4 indexer cache {name} must have an axis-2 cache writer")
            if name in indexer_writers:
                raise ValueError(f"duplicate Qwen4 indexer cache writer {name}")
            node.input[3] = name
            indexer_writers.add(name)
            continue
        match = pattern.fullmatch(name)
        if match is None:
            continue
        kind, ordinal = match.groups()
        key = (kind, int(ordinal))
        if key in writers:
            raise ValueError(f"duplicate Qwen4 {kind} cache writer {ordinal}")
        value = graph_inputs.get(name)
        shape = [] if value is None else [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        if shape != [1, num_kv_heads, context_length, head_dim]:
            raise ValueError(f"Qwen4 cache {name} has incompatible shape {shape}")
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if attrs.get("axis", 2) != 2 or attrs.get("only_handle_old_cache", 0):
            raise ValueError(f"Qwen4 cache {name} must be an axis-2 K/V writer")
        node.op_type = "KVcache"
        node.input[3] = name
        del node.attribute[:]
        node.attribute.extend([helper.make_attribute("axis", 2), helper.make_attribute("attention_max_length", -1)])
        writers[key] = node
    keys = {ordinal for kind, ordinal in writers if kind == "key"}
    values = {ordinal for kind, ordinal in writers if kind == "value"}
    expected = {name for name in graph_inputs if pattern.fullmatch(name)}
    if not keys or keys != values or len(writers) != len(expected):
        raise ValueError("Qwen4 explicit LLMCache conversion requires all paired K/V writers")
    if indexer_writers != {name for name in graph_inputs if indexer_pattern.fullmatch(name)}:
        raise ValueError("Qwen4 explicit LLMCache conversion requires all indexer cache writers")
    for ordinal in keys:
        if list(writers[("key", ordinal)].input[1:3]) != list(writers[("value", ordinal)].input[1:3]):
            raise ValueError(f"Qwen4 K/V cache lengths differ for ordinal {ordinal}")
    # Normalize legacy indexer/state cache aliases too, preserving their ports
    # and attributes. The SDK parses both names with the same cache semantics.
    for node in model.graph.node:
        if node.domain == "XH2a" and node.op_type == "LLMCache":
            node.op_type = "KVcache"
    # K/V and indexer storage belongs to the caller and is updated in place.
    # Keep writer outputs inside the attention graph, but do not expose redundant
    # functional-cache ports at the serving boundary.
    removed_outputs = {f"present_{name}" for name in expected | indexer_writers}
    # The in-place serving ABI removes cache aliases from the public graph
    # boundary, but cache writers can still feed downstream attention nodes.
    # Preserve their type information as internal value_info so ONNX frontends
    # can build single-op diagnostics and lower KVcache/SelectKV together.
    known_value_info = {value.name for value in (*model.graph.input, *model.graph.value_info)}
    for value in model.graph.output:
        if value.name in removed_outputs and value.name not in known_value_info and value.type.ByteSize() > 0:
            model.graph.value_info.append(copy.deepcopy(value))
            known_value_info.add(value.name)
    outputs = [value for value in model.graph.output if value.name not in removed_outputs]
    del model.graph.output[:]
    model.graph.output.extend(outputs)
    # Drop only now-unused aliases of those ports. Never run general DCE here:
    # cache writers have observable mutations even without a public output.
    pending = set(removed_outputs)
    while pending:
        name = pending.pop()
        alias = producers.get(name)
        if (
            alias is None
            or alias.op_type not in {"Identity", "HMIdentity"}
            or alias.domain not in {"", "XH2a", "ai.houmo.xh2a"}
            or len(alias.input) != 1
            or list(alias.output) != [name]
            or any(name in node.input for node in model.graph.node)
            or any(value.name == name for value in model.graph.output)
            or any(
                attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
                for node in model.graph.node
                for attr in node.attribute
            )
        ):
            continue
        source_name = alias.input[0]
        model.graph.node.remove(alias)
        pending.add(source_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(destination), save_as_external_data=False)
    _link_qwen4_exp_external_data(model, source=source, destination=destination)
    onnx.checker.check_model(str(destination))
    return str(destination)


def convert_qwen4_exp_kv_to_page_cache(
    input_path: str | Path,
    output_path: str | Path,
    *,
    context_length: int,
    num_kv_heads: int,
    head_dim: int,
) -> str:
    """Replace Qwen4 K/V writers with PageKVCache while retaining QSA computation."""

    import re

    import onnx
    from onnx import helper

    source = Path(input_path)
    destination = Path(output_path)
    model = onnx.load(str(source), load_external_data=False)
    cache_pattern = re.compile(r"^past_(key|value)_cache_(\d+)$")
    producers = {output: node for node in model.graph.node for output in node.output if output}

    def resolve_transparent_source(value_name: str) -> str:
        visited: set[str] = set()
        while value_name not in visited:
            visited.add(value_name)
            producer = producers.get(value_name)
            if (
                producer is None
                or producer.op_type not in {"Cast", "HMIdentity", "Identity"}
                or len(producer.input) != 1
            ):
                break
            value_name = producer.input[0]
        return value_name

    key_nodes: dict[int, tuple[int, Any]] = {}
    value_nodes: dict[int, tuple[int, Any]] = {}
    for node_index, node in enumerate(model.graph.node):
        if node.op_type not in {"KVcache", "LLMCache"} or len(node.input) < 4:
            continue
        match = cache_pattern.fullmatch(resolve_transparent_source(node.input[3]))
        if match is None:
            continue
        kind, ordinal_text = match.groups()
        ordinal = int(ordinal_text)
        target = key_nodes if kind == "key" else value_nodes
        if ordinal in target:
            raise ValueError(f"duplicate Qwen4 {kind} cache writer {ordinal}")
        target[ordinal] = (node_index, node)
    if set(key_nodes) != set(value_nodes) or not key_nodes:
        raise ValueError(
            "Qwen4 PageKVCache conversion requires paired K/V writers; "
            f"key={sorted(key_nodes)}, value={sorted(value_nodes)}"
        )

    replacement_by_index: dict[int, Any] = {}
    removed_indices: set[int] = set()
    for ordinal in sorted(key_nodes):
        key_index, key_node = key_nodes[ordinal]
        value_index, value_node = value_nodes[ordinal]
        if list(key_node.input[1:3]) != list(value_node.input[1:3]):
            raise ValueError(f"Qwen4 K/V cache lengths differ for ordinal {ordinal}")
        insertion_index = max(key_index, value_index)
        replacement_by_index[insertion_index] = helper.make_node(
            "PageKVCache",
            inputs=[
                key_node.input[0],
                value_node.input[0],
                key_node.input[1],
                key_node.input[2],
            ],
            outputs=[key_node.output[0], value_node.output[0]],
            name=f"page_kv_cache_{ordinal}",
            domain=key_node.domain or "ai.houmo.xh2a",
            layer_idx=ordinal,
            cache_index=ordinal,
            attention_kind="sliding_attention",
            sliding_window=int(context_length),
            output_sequence_length=int(context_length),
            num_kv_heads=int(num_kv_heads),
            head_size=int(head_dim),
        )
        removed_indices.update((key_index, value_index))

    nodes = []
    for node_index, node in enumerate(model.graph.node):
        if node_index in removed_indices:
            replacement = replacement_by_index.get(node_index)
            if replacement is not None:
                nodes.append(replacement)
            continue
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    cache_outputs = re.compile(r"^present_past_(?:key|value)_cache_\d+$")
    outputs = [value for value in model.graph.output if cache_outputs.fullmatch(value.name) is None]
    del model.graph.output[:]
    model.graph.output.extend(outputs)

    required_values = {value.name for value in model.graph.output}
    retained_nodes = []
    for node in reversed(model.graph.node):
        if not required_values.intersection(node.output):
            continue
        retained_nodes.append(node)
        required_values.update(name for name in node.input if name)
    retained_nodes.reverse()
    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)
    graph_inputs = [
        value for value in model.graph.input if value.name in required_values or not cache_pattern.fullmatch(value.name)
    ]
    del model.graph.input[:]
    model.graph.input.extend(graph_inputs)

    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(destination), save_as_external_data=False)
    _link_qwen4_exp_external_data(model, source=source, destination=destination)
    onnx.checker.check_model(str(destination))
    return str(destination)


def _link_qwen4_exp_external_data(model: Any, *, source: Path, destination: Path) -> None:
    """Keep large HMONNX weights zero-copy when writing a sibling graph."""

    from onnx import TensorProto, external_data_helper

    locations: set[str] = set()
    for tensor in external_data_helper._get_all_tensors(model):
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        location = next((item.value for item in tensor.external_data if item.key == "location"), None)
        if location:
            locations.add(location)
    from .ple_export import _metadata_specs

    for spec in _metadata_specs(model, source.parent):
        for key in ("packed", "exponent"):
            locations.add(spec[key]["path"])
    for location in locations:
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe Qwen4 HMONNX external-data location: {location!r}")
        source_data = source.parent / relative
        destination_data = destination.parent / relative
        if not source_data.is_file():
            raise FileNotFoundError(f"Qwen4 HMONNX external data is missing: {source_data}")
        destination_data.parent.mkdir(parents=True, exist_ok=True)
        if destination_data.exists():
            if not os.path.samefile(source_data, destination_data):
                raise FileExistsError(f"refusing to replace external data at {destination_data}")
            continue
        os.link(source_data, destination_data)


def publish_qwen4_exp_qsa_layout_resources(
    path: str | Path,
    *,
    batch_size: int,
    input_sequence_length: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> dict[str, int]:
    """Publish QSA's static layout plan after the standard SDK export.

    Layout fusion can leave a view's dimension tensor unreferenced. Preserve
    the complete selected-KV layout plan, including its two degenerate views,
    as part of Qwen4's published graph contract. Existing computation, tensor
    storage and SDK diagnostics stay intact. Only small INT64 shape ranges
    are read; packed model weights are never materialized.
    """
    import tempfile

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    geometry = (batch_size, input_sequence_length, num_attention_heads, num_key_value_heads, head_dim)
    if any(type(value) is not int or value < 1 for value in geometry):
        raise ValueError("Qwen4 QSA layout geometry must contain positive integers")
    if batch_size != 1 or num_attention_heads % num_key_value_heads:
        raise ValueError("Qwen4 QSA layouts require batch one and divisible query/KV heads")
    path = Path(path)
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    nodes = list(graph.node)
    selectors = [node for node in nodes if node.op_type == "SparseAttentionSelectKV"]
    counts = {"qsa_count": len(selectors), "layout_constants_added": 0, "layout_views_added": 0}
    if not selectors:
        return counts
    if graph.sparse_initializer or any(
        attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
        for node in nodes
        for attr in node.attribute
    ):
        raise ValueError("Qwen4 QSA layout publication requires a flat dense graph")
    producers = {value: node for node in nodes for value in node.output}
    users: dict[str, list[tuple[Any, int]]] = {}
    for node in nodes:
        for port, value in enumerate(node.input):
            users.setdefault(value, []).append((node, port))
    reserved = {value for node in nodes for value in (*node.input, *node.output)}
    reserved.update(tensor.name for tensor in graph.initializer)
    reserved.update(node.name for node in nodes)
    reserved.update(value.name for value in (*graph.input, *graph.output, *graph.value_info))
    value_types = {value.name: value.type.tensor_type for value in (*graph.input, *graph.output, *graph.value_info)}
    shape_values: dict[str, tuple[int, ...]] = {}
    shapes: dict[tuple[str, tuple[int, ...]], list[str]] = {}
    for tensor in graph.initializer:
        if tensor.data_type != onnx.TensorProto.INT64 or len(tensor.dims) != 1 or not 1 <= tensor.dims[0] <= 4:
            continue
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            fields = {entry.key: entry.value for entry in tensor.external_data}
            location = Path(fields.get("location", ""))
            offset, length = int(fields.get("offset", "0")), int(fields.get("length", "0"))
            if location.is_absolute() or ".." in location.parts or offset < 0 or length != tensor.dims[0] * 8:
                raise ValueError(f"Invalid Qwen4 shape tensor external range: {tensor.name}")
            with (path.parent / location).open("rb") as stream:
                stream.seek(offset)
                data = stream.read(length)
            if len(data) != length:
                raise ValueError(f"Truncated Qwen4 shape tensor: {tensor.name}")
            values = tuple(int(value) for value in np.frombuffer(data, dtype="<i8"))
        else:
            values = tuple(int(value) for value in numpy_helper.to_array(tensor))
        shape_values[tensor.name] = values
        namespace = tensor.name.rpartition("/")[0]
        shapes.setdefault((namespace, values), []).append(tensor.name)

    def unique_name(stem: str) -> str:
        candidate, suffix = stem, 0
        while candidate in reserved:
            suffix += 1
            candidate = f"{stem}_{suffix}"
        reserved.add(candidate)
        return candidate

    def shape_tensor(namespace: str, values: tuple[int, ...]) -> str:
        matches = shapes.get((namespace, values), [])
        if len(matches) > 1:
            raise ValueError(f"Ambiguous Qwen4 QSA layout shape {values} in {namespace!r}")
        if matches:
            return matches[0]
        prefix = namespace + "/" if namespace else ""
        name = unique_name(prefix + "qsa_layout_shape_" + "_".join(map(str, values)))
        graph.initializer.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name))
        shapes[(namespace, values)] = [name]
        counts["layout_constants_added"] += 1
        return name

    def boundary(value: str) -> Any:
        found, pending, visited = [], [value], set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            for node, port in users.get(current, []):
                if node.op_type == "MatMul" and port == 1:
                    found.append(node)
                elif node.op_type in {"Concat", "Reshape", "Transpose"}:
                    pending.extend(node.output)
        if len(found) != 1:
            raise ValueError("Qwen4 SelectKV must have unique key/value MatMul boundaries")
        return found[0]

    for selector in selectors:
        if selector.domain != "ai.houmo.xh2a" or len(selector.output) != 2:
            raise ValueError("Qwen4 SelectKV must expose ordered key/value outputs")
        score, output = (boundary(value) for value in selector.output)
        softmax = producers.get(output.input[0])
        if softmax is None or softmax.op_type != "Softmax":
            raise ValueError("Qwen4 QSA value MatMul must be fed by Softmax")
        pending, ancestors = list(softmax.input), set()
        while pending:
            value = pending.pop()
            if value in ancestors:
                continue
            ancestors.add(value)
            node = producers.get(value)
            if node is not None and node.op_type in {"Add", "Mul", "Cast"}:
                pending.extend(node.input)
        if score.output[0] not in ancestors:
            raise ValueError("Qwen4 SelectKV key/value paths must form the same attention")
        key_transpose = producers.get(score.input[1])
        key_layout = None if key_transpose is None else producers.get(key_transpose.input[0])
        if (
            key_transpose is None
            or key_transpose.op_type != "Transpose"
            or key_layout is None
            or key_layout.op_type != "Reshape"
            or len(key_layout.input) != 2
        ):
            raise ValueError("Qwen4 selected keys must use the configured packed head layout")
        # Placeholder assembly owns independent initializer namespaces. Reuse
        # shapes inside this actual key-layout owner, never merge block pools.
        namespace = key_layout.input[1].rpartition("/")[0]
        attributes = {attr.name: helper.get_attribute_value(attr) for attr in selector.attribute}
        width = int(attributes["candidate_topk"]) * 4 + 3
        packed = (batch_size * input_sequence_length, num_attention_heads, 1, head_dim)
        outer = (batch_size, input_sequence_length, num_attention_heads, head_dim)
        grouped = (
            batch_size * input_sequence_length,
            num_key_value_heads,
            num_attention_heads // num_key_value_heads,
            head_dim,
        )
        key_shape = (batch_size * input_sequence_length, num_key_value_heads, width, head_dim)
        if key_layout.input[1] not in shapes.get((namespace, key_shape), []):
            raise ValueError("Qwen4 selected-key shape disagrees with the model geometry")
        query = producers.get(score.input[0])
        tensor_type = value_types.get(score.input[0])
        query_shape = None if tensor_type is None else tuple(dim.dim_value for dim in tensor_type.shape.dim)
        if query is not None and query.op_type == "Reshape":
            query_shape = shape_values.get(query.input[1])
        if query_shape is not None:
            if query_shape != grouped:
                raise ValueError("Qwen4 query shape disagrees with the configured query heads")
        else:
            # A single-query/single-KV-head wrapper omits its identity view.
            # Check the real projected query width, or its typed graph input.
            pending, seen, widths = [score.input[0]], set(), set()
            while pending:
                value = pending.pop()
                if value in seen:
                    continue
                seen.add(value)
                node = producers.get(value)
                if node is None:
                    tensor_type = value_types.get(value)
                    if tensor_type is not None:
                        dimensions = tuple(dim.dim_value for dim in tensor_type.shape.dim)
                        if dimensions == (batch_size, num_attention_heads, input_sequence_length, head_dim):
                            widths.add(2 * num_attention_heads * head_dim)
                elif node.op_type == "Linear":
                    attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
                    widths.add(int(attrs.get("out_features", -1)))
                elif node.op_type == "Concat":
                    pending.extend(node.input)
                elif node.op_type in {"Rope", "RMSNorm", "Reshape", "Transpose", "Slice", "Cast"}:
                    pending.append(node.input[0])
            if widths != {2 * num_attention_heads * head_dim}:
                raise ValueError("Qwen4 query projection disagrees with the configured query heads")
        pending, seen, output_widths = [output.output[0]], set(), set()
        while pending:
            value = pending.pop()
            if value in seen:
                continue
            seen.add(value)
            tensor_type = value_types.get(value)
            if tensor_type is not None:
                dimensions = tuple(dim.dim_value for dim in tensor_type.shape.dim)
                if len(dimensions) == 3 and dimensions[:2] == (batch_size, input_sequence_length):
                    output_widths.add(dimensions[2])
            for node, _ in users.get(value, []):
                if node.op_type == "Linear":
                    attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
                    output_widths.add(int(attrs.get("in_features", -1)))
                elif node.op_type in {"Reshape", "Transpose", "Mul", "Cast"}:
                    pending.extend(node.output)
        if output_widths != {num_attention_heads * head_dim}:
            raise ValueError("Qwen4 attention output width disagrees with the configured query heads")
        for dimensions in (packed, outer, grouped, key_shape, (batch_size * input_sequence_length, 1, 1, width)):
            shape_tensor(namespace, dimensions)
        if outer != grouped or packed == grouped:
            continue
        if (
            query is None
            or query.op_type != "Transpose"
            or {attr.name: helper.get_attribute_value(attr) for attr in query.attribute} != {"perm": [0, 2, 1, 3]}
        ):
            raise ValueError("Qwen4 degenerate grouped-query identity must be lowered in the wrapper")
        target = shape_tensor(namespace, packed)
        for source, domain, allowzero, role in (
            (score.input[0], "ai.houmo.xh2a", 0, "packed_query"),
            (output.output[0], "", 1, "packed_output"),
        ):
            existing = [
                node
                for node in nodes
                if node.op_type == "Reshape"
                and node.domain == domain
                and list(node.input) == [source, target]
                and {attr.name: helper.get_attribute_value(attr) for attr in node.attribute} == {"allowzero": allowzero}
            ]
            if len(existing) > 1:
                raise ValueError(f"Ambiguous Qwen4 QSA {role} publication view")
            if not existing:
                prefix = namespace + "/" if namespace else ""
                node = helper.make_node(
                    "Reshape",
                    [source, target],
                    [unique_name(prefix + "qsa_layout_" + role + "_value")],
                    name=unique_name(prefix + "qsa_layout_" + role),
                    domain=domain,
                    allowzero=allowzero,
                )
                nodes.insert(nodes.index(producers[source]) + 1, node)
                counts["layout_views_added"] += 1
    if counts["layout_constants_added"] or counts["layout_views_added"]:
        del graph.node[:]
        graph.node.extend(nodes)
        handle, temporary = tempfile.mkstemp(prefix=".qwen4-qsa-layout-", suffix=".onnx", dir=path.parent)
        os.close(handle)
        try:
            onnx.save_model(model, temporary, save_as_external_data=False)
            onnx.checker.check_model(temporary)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return counts
