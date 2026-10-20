"""Adapt existing Qwen4 text graphs without re-exporting their weight tensors."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def rewrite_gather_index_normalization(model, *, external_data_dir=None) -> int:
    """Keep SDK GatherElements negative-index normalization in its INT32 ABI.

    Match the complete closed SDK decomposition, not arbitrary INT64 arithmetic.
    Negative INT32 indices plus a positive INT32 axis size cannot overflow. The
    positive branch's unused addition may overflow, but only the original index
    reaches the gather in that branch. Invalid negative indices retain the same
    out-of-bounds sentinel (axis size).
    """
    import numpy as np
    import onnx

    if any(a.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
           for n in model.graph.node for a in n.attribute):
        return 0
    nodes = list(model.graph.node)
    producers = {o: n for n in nodes for o in n.output}
    users = {}
    for node in nodes:
        for index, name in enumerate(node.input):
            users.setdefault(name, []).append((node, index))
    infos = {v.name: v for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
    tensors = {t.name: t for t in model.graph.initializer}
    graph_inputs = {v.name for v in model.graph.input}
    public = {v.name for v in (*model.graph.input, *model.graph.output)}
    reserved = set(producers) | set(infos) | set(tensors)
    removals = set()
    constants = {}
    rewritten = 0

    def op(name, kind, arity):
        node = producers.get(name)
        if (node is None or node.op_type != kind or node.domain not in {"", "ai.onnx"}
                or len(node.input) != arity or len(node.output) != 1):
            return None
        return node

    def scalar(name):
        tensor = tensors.get(name)
        if name in graph_inputs or tensor is None or tensor.data_type != onnx.TensorProto.INT64 or tensor.dims:
            return None
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            fields = {f.key: f.value for f in tensor.external_data}
            if external_data_dir is None or not {"location", "offset", "length"} <= fields.keys():
                return None
            base = Path(external_data_dir).resolve()
            path = (base / fields["location"]).resolve()
            offset, length = int(fields["offset"]), int(fields["length"])
            if not path.is_relative_to(base) or offset < 0 or length != 8:
                return None
            with path.open("rb") as stream:
                stream.seek(offset)
                data = stream.read(length)
            return int(np.frombuffer(data, dtype="<i8")[0]) if len(data) == 8 else None
        return int(onnx.numpy_helper.to_array(tensor))

    def shape(name):
        value = infos.get(name)
        if value is None:
            tensor = tensors.get(name)
            if tensor is not None and name not in graph_inputs and all(d > 0 for d in tensor.dims):
                return tuple(tensor.dims), tensor.data_type
            return None
        if not value.type.HasField("tensor_type"):
            return None
        tensor = value.type.tensor_type
        if not tensor.HasField("shape") or any(not d.HasField("dim_value") or d.dim_value <= 0
                                               for d in tensor.shape.dim):
            return None
        return tuple(d.dim_value for d in tensor.shape.dim), tensor.elem_type

    for final in nodes:
        if not final.output or op(final.output[0], "Where", 3) is not final:
            continue
        negative = op(final.input[0], "Less", 2)
        first = op(final.input[2], "Where", 3)
        size = scalar(final.input[1])
        if negative is None or first is None or size is None or not 0 < size <= 2**31 - 1:
            continue
        if list(negative.input) != [first.output[0], negative.input[1]] or scalar(negative.input[1]) != 0:
            continue
        less = op(first.input[0], "Less", 2)
        add = op(first.input[1], "Add", 2)
        cast = op(first.input[2], "Cast", 1)
        if less is None or add is None or cast is None:
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in cast.attribute}
        if attrs != {"to": onnx.TensorProto.INT64}:
            continue
        source = shape(cast.input[0])
        if source is None or source[1] != onnx.TensorProto.INT32:
            continue
        value = cast.output[0]
        if (list(less.input) != [value, less.input[1]] or scalar(less.input[1]) != 0
                or list(add.input) != [value, add.input[1]] or scalar(add.input[1]) != size):
            continue
        chain = [cast, less, add, first, negative, final]
        if any(n.attribute for n in chain[1:]) or any(n.output[0] in public for n in chain):
            continue
        expected = {
            value: [(less, 0), (add, 0), (first, 2)],
            less.output[0]: [(first, 0)], add.output[0]: [(first, 1)],
            first.output[0]: [(negative, 0), (final, 2)], negative.output[0]: [(final, 0)],
        }
        if any({(id(n), i) for n, i in users.get(name, [])} != {(id(n), i) for n, i in uses}
               for name, uses in expected.items()):
            continue
        gathers = users.get(final.output[0], [])
        if not gathers:
            continue
        valid = True
        for gather, slot in gathers:
            data = shape(gather.input[0])
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in gather.attribute}
            axis = attrs.get("axis", 0)
            if (gather.op_type != "GatherElements" or gather.domain not in {"", "ai.onnx"}
                    or slot != 1 or len(gather.input) != 2 or set(attrs) - {"axis"}
                    or data is None or not -len(data[0]) <= axis < len(data[0])
                    or data[0][axis] != size):
                valid = False
                break
        if not valid:
            continue
        for old, number in [(less.input[1], 0), (negative.input[1], 0),
                            (add.input[1], size), (final.input[1], size)]:
            if old in constants:
                continue
            name = old + "_gather_int32"
            suffix = 0
            while name in reserved:
                suffix += 1
                name = old + f"_gather_int32_{suffix}"
            reserved.add(name)
            constants[old] = name
            model.graph.initializer.append(onnx.numpy_helper.from_array(np.array(number, dtype=np.int32), name))
        for node in chain[1:]:
            for index, name in enumerate(node.input):
                if name == value:
                    node.input[index] = cast.input[0]
                elif name in constants:
                    node.input[index] = constants[name]
        for node in [add, first, final]:
            if node.output[0] in infos:
                infos[node.output[0]].type.tensor_type.elem_type = onnx.TensorProto.INT32
        removals.add(id(cast))
        rewritten += 1
    if removals:
        removed_outputs = {o for n in nodes if id(n) in removals for o in n.output}
        kept_info = [v for v in model.graph.value_info if v.name not in removed_outputs]
        del model.graph.node[:]
        model.graph.node.extend(n for n in nodes if id(n) not in removals)
        del model.graph.value_info[:]
        model.graph.value_info.extend(kept_info)
    return rewritten


def normalize_gather_index_graph(path) -> int:
    """Atomically lower only proven closed INT32-origin gather index chains."""
    import onnx

    path = Path(path)
    model = onnx.load(path, load_external_data=False)
    count = rewrite_gather_index_normalization(model, external_data_dir=path.parent)
    if count:
        handle, temporary = tempfile.mkstemp(prefix=".qwen4-gather-", suffix=".onnx", dir=path.parent)
        os.close(handle)
        try:
            onnx.save_model(model, temporary)
            onnx.checker.check_model(temporary)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return count


def simplify_qwen4_static_layouts(model, *, external_data_dir=None) -> tuple[str, ...]:
    """Remove proven static view identities and reuse exact boolean mask casts.

    Work on the complete exported graph before serving slices are extracted.
    Public names, subgraph captures and cache aliases are deliberately excluded.
    """
    import math

    import numpy as np
    import onnx

    if any(attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
           for node in model.graph.node for attr in node.attribute):
        return ()
    infos = {v.name: v for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
    tensors = {v.name: v for v in model.graph.initializer}
    public = {v.name for v in (*model.graph.input, *model.graph.output)}
    nodes = list(model.graph.node)
    removed = []
    orphan_candidates = set()

    def info(name):
        value = infos.get(name)
        if value is None or not value.type.HasField("tensor_type"):
            return None
        tensor = value.type.tensor_type
        if not tensor.HasField("shape") or any(not d.HasField("dim_value") or d.dim_value <= 0
                                               for d in tensor.shape.dim):
            return None
        return tuple(d.dim_value for d in tensor.shape.dim), tensor.elem_type

    def attributes(node, allowed):
        domains = {"", "ai.onnx"}
        if node.op_type in {"Reshape", "Cast"}:
            domains.add("ai.houmo.xh2a")
        if node.domain not in domains or set(a.name for a in node.attribute) - allowed:
            return None
        return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

    def integers(name):
        tensor = tensors.get(name)
        if (name in public or tensor is None or tensor.data_type != onnx.TensorProto.INT64
                or len(tensor.dims) != 1):
            return None
        count = tensor.dims[0]
        if not 1 <= count <= 8:
            return None
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            fields = {f.key: f.value for f in tensor.external_data}
            if external_data_dir is None or not {"location", "offset", "length"} <= fields.keys():
                return None
            base = Path(external_data_dir).resolve()
            location = (base / fields["location"]).resolve()
            offset, length = int(fields["offset"]), int(fields["length"])
            if not location.is_relative_to(base) or offset < 0 or length != count * 8:
                return None
            with location.open("rb") as stream:
                stream.seek(offset)
                raw = stream.read(length)
            if len(raw) != length:
                return None
            return tuple(int(x) for x in np.frombuffer(raw, dtype="<i8"))
        return tuple(int(x) for x in onnx.numpy_helper.to_array(tensor))

    def reshape(node):
        attrs = attributes(node, {"allowzero"})
        if (attrs is None or attrs.get("allowzero", 0) not in {0, 1} or node.op_type != "Reshape"
                or len(node.input) != 2 or len(node.output) != 1):
            return False
        source, result, shape = info(node.input[0]), info(node.output[0]), integers(node.input[1])
        if source is None or result is None or source[1] != result[1] or shape is None:
            return False
        resolved = []
        for axis, size in enumerate(shape):
            if size == 0 and attrs.get("allowzero", 0) == 0:
                if axis >= len(source[0]):
                    return False
                size = source[0][axis]
            resolved.append(size)
        if any(size < -1 or size == 0 for size in resolved) or resolved.count(-1) > 1:
            return False
        elements = math.prod(source[0])
        if -1 in resolved:
            known = -math.prod(resolved)
            if elements % known:
                return False
            resolved[resolved.index(-1)] = elements // known
        return tuple(resolved) == result[0] and math.prod(resolved) == elements

    def protected(node):
        return (len(node.output) != 1 or node.output[0] in public
                or any(other.op_type.lower() in {"kvcache", "llmcache", "pagekvcache",
                                                 "gdrrecurrentscan", "gdrchunkscan"}
                       for other in nodes if node.output[0] in other.input))

    def replace(node, replacement):
        if protected(node):
            return False
        output = node.output[0]
        users = [other for other in nodes if output in other.input]
        for other in users:
            for index, name in enumerate(other.input):
                if name == output:
                    other.input[index] = replacement
        orphan_candidates.update(node.input)
        nodes.remove(node)
        removed.append(node.name)
        return True

    for node in list(nodes):
        if protected(node):
            continue
        producers = {output: other for other in nodes for output in other.output}
        if reshape(node):
            source = node.input[0]
            previous = producers.get(source)
            if info(source) == info(node.output[0]):
                replace(node, source)
            elif previous is not None and reshape(previous):
                # The original view may have other users; only bypass it on
                # this branch and remove it later if it becomes unused.
                if (info(previous.input[0]) != info(node.output[0])
                        and (previous.output[0] in public
                             or sum(previous.output[0] in other.input for other in nodes) != 1)):
                    continue
                original = node.input[0]
                node.input[0] = previous.input[0]
                if not reshape(node):
                    node.input[0] = original
                elif info(node.input[0]) == info(node.output[0]):
                    replace(node, node.input[0])
                else:
                    orphan_candidates.add(previous.output[0])
        elif node.op_type == "Transpose" and len(node.input) == len(node.output) == 1:
            attrs = attributes(node, {"perm"})
            previous = producers.get(node.input[0])
            old_attrs = attributes(previous, {"perm"}) if previous is not None else None
            if (attrs is not None and previous is not None and previous.op_type == "Transpose"
                    and old_attrs is not None and len(previous.input) == len(previous.output) == 1):
                source = info(previous.input[0])
                rank = len(source[0]) if source is not None else 0
                first, second = old_attrs.get("perm", ()), attrs.get("perm", ())
                if (rank and sorted(first) == sorted(second) == list(range(rank))
                        and [first[i] for i in second] == list(range(rank))
                        and info(node.output[0]) == source):
                    replace(node, previous.input[0])
        elif node.op_type == "Cast" and len(node.input) == len(node.output) == 1:
            attrs = attributes(node, {"to", "round_mode"})
            unsqueeze = producers.get(node.input[0])
            if (attrs is None or attrs.get("to") != onnx.TensorProto.FLOAT16
                    or attrs.get("round_mode", b"NONE") not in {b"NONE", b"RNE", b"RTZ", b"RTP", b"RTN"}
                    or unsqueeze is None
                    or unsqueeze.op_type != "Unsqueeze" or attributes(unsqueeze, set()) is None
                    or len(unsqueeze.input) != 2 or len(unsqueeze.output) != 1):
                continue
            mask_info = info(unsqueeze.input[0])
            axes = integers(unsqueeze.input[1])
            if mask_info is None or mask_info[1] != onnx.TensorProto.BOOL or axes is None:
                continue
            for earlier in nodes[:nodes.index(node)]:
                if earlier.op_type != "Unsqueeze" or len(earlier.input) != 2 or len(earlier.output) != 1:
                    continue
                cast = producers.get(earlier.input[0])
                cast_attrs = attributes(cast, {"to", "round_mode"}) if cast is not None else None
                if (attributes(earlier, set()) is not None and integers(earlier.input[1]) == axes
                        and cast is not None and cast.op_type == "Cast" and cast_attrs is not None
                        and cast_attrs.get("round_mode", b"NONE") in {b"NONE", b"RNE", b"RTZ", b"RTP", b"RTN"}
                        and cast_attrs.get("to") == onnx.TensorProto.FLOAT16
                        and list(cast.input) == [unsqueeze.input[0]]
                        and info(earlier.output[0]) == info(node.output[0])):
                    replace(node, earlier.output[0])
                    break
    # Only remove newly orphaned layout nodes; cache/state operators are never
    # subject to generic dead-code elimination.
    while True:
        used = public | {name for node in nodes for name in node.input}
        dead = [node for node in nodes if node.op_type in {"Reshape", "Transpose", "Unsqueeze"}
                and attributes(node, {"allowzero", "perm"}) is not None
                and len(node.output) == 1 and node.output[0] in orphan_candidates and node.output[0] not in used]
        if not dead:
            break
        for node in dead:
            orphan_candidates.update(node.input)
            nodes.remove(node)
            removed.append(node.name)
    if removed:
        retained = {name for node in nodes for name in (*node.input, *node.output)} | public
        values = [value for value in model.graph.value_info if value.name in retained]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        del model.graph.value_info[:]
        model.graph.value_info.extend(values)
    return tuple(removed)


def normalize_qwen4_static_layout_graph(path) -> tuple[str, ...]:
    """Apply the model's bounded layout cleanup atomically, retaining sidecars."""
    import onnx

    path = Path(path)
    model = onnx.load(path, load_external_data=False)
    removed = ()
    while changed := simplify_qwen4_static_layouts(model, external_data_dir=path.parent):
        removed += changed
    if removed:
        handle, temporary = tempfile.mkstemp(prefix=".qwen4-layout-", suffix=".onnx", dir=path.parent)
        os.close(handle)
        try:
            onnx.save_model(model, temporary)
            onnx.checker.check_model(temporary)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return removed
