"""Exact port-labelled HMONNX DAG comparison; no graph rewriting.

The structural view excludes identifier spelling and diagnostic protobuf fields.
It preserves all operator attributes, tensor dtype/shape/payload, graph boundary
order/name/type, port positions, fanout/sharing, unused operators/initializers.
Raw file SHA and protobufs are reported separately, never claimed equal by this view.
"""

import base64
import collections
import copy
import hashlib
import json
import sys
from pathlib import Path

import onnx
import pytest
from google.protobuf.json_format import MessageToDict
from onnx import numpy_helper


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_record(tensor, base):
    if tensor.data_location == onnx.TensorProto.EXTERNAL:
        fields = {x.key: x.value for x in tensor.external_data}
        location = Path(base) / fields["location"]
        start = int(fields.get("offset", "0"))
        size = int(fields.get("length", location.stat().st_size - start))
        h = hashlib.sha256()
        with location.open("rb") as stream:
            stream.seek(start)
            remaining = size
            while remaining:
                block = stream.read(min(remaining, 8 * 1024 * 1024))
                if not block:
                    raise EOFError((str(location), start, size))
                h.update(block)
                remaining -= len(block)
        payload = h.hexdigest()
    else:
        payload = hashlib.sha256(numpy_helper.to_array(tensor).tobytes()).hexdigest()
    result = {"dtype": tensor.data_type, "shape": list(tensor.dims), "payload_sha256": payload}
    if tensor.HasField("segment"):
        result["segment"] = MessageToDict(tensor.segment, preserving_proto_field_name=True)
    return result


def attribute_record(attribute, base):
    result = MessageToDict(attribute, preserving_proto_field_name=True)
    # These fields contain TensorProto identifiers/storage paths, not operator semantics.
    # The full original bytes remain in raw_proto_sha256 and the original artifacts.
    result.pop("doc_string", None)
    if attribute.HasField("t"):
        result["t"] = tensor_record(attribute.t, base)
    if attribute.tensors:
        result["tensors"] = [tensor_record(t, base) for t in attribute.tensors]
    if attribute.HasField("g") or attribute.graphs:
        raise ValueError("Graph-valued operator attributes require explicit recursive comparison")
    return result


def graph_manifest(path):
    path = Path(path)
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    if graph.sparse_initializer or graph.quantization_annotation:
        raise ValueError("Sparse initializers and quantization annotations need explicit comparison")
    definitions = {}
    initializers = []
    for index, tensor in enumerate(graph.initializer):
        definitions[tensor.name] = ["initializer", index]
        initializers.append(tensor_record(tensor, path.parent))
    inputs = []
    for index, value in enumerate(graph.input):
        inputs.append({"name": value.name, "type": MessageToDict(value.type, preserving_proto_field_name=True)})
        # Legacy ONNX permits initializer/input overlap. Keep the overlap explicit.
        if value.name not in definitions:
            definitions[value.name] = ["input", index]
    for index, node in enumerate(graph.node):
        for port, name in enumerate(node.output):
            if name:
                if name in definitions:
                    raise ValueError(f"Multiple definitions for {name}")
                definitions[name] = ["node", index, port]
    nodes = []
    for node in graph.node:
        nodes.append(
            {
                "domain": node.domain,
                "op": node.op_type,
                "attributes": sorted(
                    [attribute_record(a, path.parent) for a in node.attribute], key=lambda a: a["name"]
                ),
                "inputs": [definitions[name] if name else ["empty"] for name in node.input],
                "output_count": len(node.output),
                "empty_output_ports": [i for i, name in enumerate(node.output) if not name],
            }
        )
    outputs = [
        {"name": v.name, "type": MessageToDict(v.type, preserving_proto_field_name=True), "source": definitions[v.name]}
        for v in graph.output
    ]
    return {
        "format_version": 1,
        "inputs": inputs,
        "outputs": outputs,
        "nodes": nodes,
        "initializers": initializers,
        "initializer_input_overlap": [
            [i, definitions[v.name]] for i, v in enumerate(graph.input) if definitions[v.name][0] == "initializer"
        ],
        "model_contract": {
            "ir_version": model.ir_version,
            "opset_import": sorted([[x.domain, x.version] for x in model.opset_import]),
            "functions_sha256": [
                hashlib.sha256(f.SerializeToString(deterministic=True)).hexdigest() for f in model.functions
            ],
        },
        "raw": {
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "node_names": [n.name for n in graph.node],
            "node_inputs": [list(n.input) for n in graph.node],
            "node_outputs": [list(n.output) for n in graph.node],
            "initializer_names": [t.name for t in graph.initializer],
            "initializer_proto_sha256": [
                hashlib.sha256(t.SerializeToString(deterministic=True)).hexdigest() for t in graph.initializer
            ],
            "node_proto_sha256": [
                hashlib.sha256(n.SerializeToString(deterministic=True)).hexdigest() for n in graph.node
            ],
        },
    }


def node_label(node):
    return {k: v for k, v in node.items() if k != "inputs"}


def forward_hashes(manifest):
    node_hashes = []

    def value_hash(source):
        kind = source[0]
        if kind == "input":
            return digest(["input", source[1], manifest["inputs"][source[1]]])
        if kind == "initializer":
            return digest(["initializer", manifest["initializers"][source[1]]])
        if kind == "empty":
            return digest(["empty"])
        return digest(["node_output", node_hashes[source[1]], source[2]])

    for node in manifest["nodes"]:
        node_hashes.append(digest([node_label(node), [value_hash(s) for s in node["inputs"]]]))
    return node_hashes, [value_hash(x["source"]) for x in manifest["outputs"]]


def compare_manifests(old, new):
    """Complete bijection, preserving sharing; forward hashes only prune candidates.

    An unfolded-expression hash alone cannot detect changing two consumers from
    a shared producer to two identical producers. Forward *and inverse* mappings
    below make that difference fail, including disconnected/dead components.
    """
    old_hashes, old_out = forward_hashes(old)
    new_hashes, new_out = forward_hashes(new)
    boundary_equal = old["inputs"] == new["inputs"] and [
        {k: v for k, v in x.items() if k != "source"} for x in old["outputs"]
    ] == [{k: v for k, v in x.items() if k != "source"} for x in new["outputs"]]
    result = {
        "raw_file_equal": bool(old.get("raw", {}).get("file_sha256"))
        and old.get("raw", {}).get("file_sha256") == new.get("raw", {}).get("file_sha256"),
        "input_output_contract_equal": boundary_equal,
        "model_contract_equal": old["model_contract"] == new["model_contract"],
        "initializer_input_overlap_count_equal": len(old["initializer_input_overlap"])
        == len(new["initializer_input_overlap"]),
        "node_count": [len(old["nodes"]), len(new["nodes"])],
        "initializer_count": [len(old["initializers"]), len(new["initializers"])],
        "all_node_labels_multiset_equal": collections.Counter(digest(node_label(n)) for n in old["nodes"])
        == collections.Counter(digest(node_label(n)) for n in new["nodes"]),
        "initializer_payload_multiset_equal": collections.Counter(map(digest, old["initializers"]))
        == collections.Counter(map(digest, new["initializers"])),
        "forward_node_fingerprints_equal": collections.Counter(old_hashes) == collections.Counter(new_hashes),
        "ordered_output_fingerprints_equal": old_out == new_out,
        "port_labelled_dag_isomorphic": False,
    }
    prerequisites = [
        boundary_equal,
        result["model_contract_equal"],
        result["all_node_labels_multiset_equal"],
        result["initializer_payload_multiset_equal"],
        result["forward_node_fingerprints_equal"],
        result["ordered_output_fingerprints_equal"],
        result["initializer_input_overlap_count_equal"],
    ]
    if not all(prerequisites):
        return result
    mapping, inverse = {}, {}

    def bind(left, right):
        left, right = tuple(left), tuple(right)
        if left in mapping:
            return mapping[left] == right
        if right in inverse:
            return False
        if left[0] != right[0]:
            return False
        mapping[left], inverse[right] = right, left
        kind = left[0]
        if kind in ("input", "empty"):
            return left == right
        if kind == "initializer":
            return old["initializers"][left[1]] == new["initializers"][right[1]]
        # 'node' references include output port, 'operation' references bind all outputs.
        if kind == "node":
            return left[2] == right[2] and bind(["operation", left[1]], ["operation", right[1]])
        a, b = old["nodes"][left[1]], new["nodes"][right[1]]
        return (
            node_label(a) == node_label(b)
            and len(a["inputs"]) == len(b["inputs"])
            and all(bind(x, y) for x, y in zip(a["inputs"], b["inputs"], strict=True))
            and all(bind(["node", left[1], p], ["node", right[1], p]) for p in range(a["output_count"]))
        )

    # Find a bijection for every remaining component, including unused tensors.
    entities = [("operation", i) for i in range(len(old["nodes"]))] + [
        ("initializer", i) for i in range(len(old["initializers"]))
    ]
    candidates = collections.defaultdict(list)
    for i, h in enumerate(new_hashes):
        candidates[("operation", h)].append(("operation", i))
    for i, t in enumerate(new["initializers"]):
        candidates[("initializer", digest(t))].append(("initializer", i))

    def finish(offset=0):
        while offset < len(entities) and entities[offset] in mapping:
            offset += 1
        if offset == len(entities):
            return True
        left = entities[offset]
        label = old_hashes[left[1]] if left[0] == "operation" else digest(old["initializers"][left[1]])
        for right in candidates[(left[0], label)]:
            if right in inverse:
                continue
            before_map, before_inverse = dict(mapping), dict(inverse)
            if bind(left, right) and finish(offset + 1):
                return True
            mapping.clear()
            mapping.update(before_map)
            inverse.clear()
            inverse.update(before_inverse)
        return False

    saved_recursion_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(saved_recursion_limit, 20 * len(entities) + 1000))
        overlap_old = dict(old["initializer_input_overlap"])
        overlap_new = dict(new["initializer_input_overlap"])
        overlap_equal = overlap_old.keys() == overlap_new.keys() and all(
            bind(overlap_old[i], overlap_new[i]) for i in overlap_old
        )
        outputs_equal = overlap_equal and all(
            bind(a["source"], b["source"]) for a, b in zip(old["outputs"], new["outputs"], strict=True)
        )
        result["port_labelled_dag_isomorphic"] = outputs_equal and finish()
    finally:
        sys.setrecursionlimit(saved_recursion_limit)
    if result["port_labelled_dag_isomorphic"]:
        result["node_bijection"] = {str(k[1]): v[1] for k, v in mapping.items() if k[0] == "operation"}
    return result


FIXTURE = Path(__file__).with_name("qwen4_exp_dynamo_tiny_structure.json")


def unpack_frozen_manifest(packed):
    result = {k: v for k, v in packed.items() if k not in ("nodes", "labels")}
    result["nodes"] = [{**packed["labels"][label], "inputs": inputs} for label, inputs in packed["nodes"]]
    return result


def _mini_graph(path):
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["input"], ["left"], name="left_node"),
            helper.make_node("Identity", ["input"], ["right"], name="right_node"),
            helper.make_node("Concat", ["left", "left"], ["joined"], axis=0),
            helper.make_node("Add", ["joined", "bias"], ["output"]),
        ],
        "sharing",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
        [numpy_helper.from_array(np.array([1, 2], dtype=np.float32), name="bias")],
    )
    model = helper.make_model(graph)
    onnx.save(model, path)
    return model


@pytest.mark.parametrize("mutation", ["edge", "attribute", "domain", "constant", "sharing", "unused", "io"])
def test_structure_comparison_rejects_real_differences(tmp_path, mutation):
    from onnx import helper

    old_path, new_path = tmp_path / "old.onnx", tmp_path / "new.onnx"
    model = _mini_graph(old_path)
    if mutation == "edge":
        model.graph.node[-1].input.reverse()
    elif mutation == "attribute":
        model.graph.node[2].attribute[0].i = -1
    elif mutation == "domain":
        model.graph.node[2].domain = "ai.houmo.xh2a"
    elif mutation == "constant":
        model.graph.initializer[0].raw_data = b"\x00\x00\x00\x00" * 2
    elif mutation == "sharing":
        # Same node count, labels and unfolded-expression hashes; fanout differs.
        model.graph.node[2].input[1] = "right"
    elif mutation == "unused":
        model.graph.node.insert(2, helper.make_node("Identity", ["input"], ["unused"]))
    else:
        model.graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    onnx.save(model, new_path)
    comparison = compare_manifests(graph_manifest(old_path), graph_manifest(new_path))
    assert not comparison["port_labelled_dag_isomorphic"], comparison
    if mutation == "sharing":
        assert comparison["forward_node_fingerprints_equal"]
        assert comparison["ordered_output_fingerprints_equal"]


def test_structure_comparison_allows_identifier_and_topological_order_changes(tmp_path):
    old_path, new_path = tmp_path / "old.onnx", tmp_path / "new.onnx"
    model = _mini_graph(old_path)
    first, second = onnx.NodeProto(), onnx.NodeProto()
    first.CopyFrom(model.graph.node[0])
    second.CopyFrom(model.graph.node[1])
    model.graph.node[0].CopyFrom(second)
    model.graph.node[1].CopyFrom(first)
    rename = {"left": "renamed_left", "right": "renamed_right", "joined": "renamed_joined", "bias": "renamed_bias"}
    for node in model.graph.node:
        node.name += "_new_process"
        for ports in (node.input, node.output):
            for i, name in enumerate(ports):
                ports[i] = rename.get(name, name)
    model.graph.initializer[0].name = rename["bias"]
    model.graph.doc_string = "Different diagnostics"
    onnx.save(model, new_path)
    comparison = compare_manifests(graph_manifest(old_path), graph_manifest(new_path))
    assert not comparison["raw_file_equal"]
    assert comparison["port_labelled_dag_isomorphic"], comparison


def test_qwen4_exp_two_phase_structure_matches_frozen_dynamo(
    tmp_path,
    monkeypatch,
    restore_model_initialization_state,
):
    """Actual public wrap/PTQ/export against old exported data, never the old capture code.

    Fixture provenance and controlled regeneration instructions live inside the JSON.
    This is only the two-layer W16 configuration; it does not certify packed W4,
    full-size/256K, MTP, or the distinct GDR key/value head-repeat branch.
    """
    import torch
    from transformers import Qwen4ExpForCausalLM

    from xhmodel_merak.xh_llm.models.qwen4_exp.configuration_qwen4_exp import XHQwen4ExpModelConfig
    from xhmodel_merak.xh_llm.models.qwen4_exp.qwen4_exp_model import XHQwen4ExpModel

    frozen = json.loads(FIXTURE.read_text())
    monkeypatch.setenv("HUGE_MODEL_EXPORT_ENABLED", "0")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    # Immutable native INPUT weights, never a saved ONNX graph or candidate output.
    for name, record in frozen["checkpoint_files"].items():
        payload = base64.b64decode(record["data"], validate=True)
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        (checkpoint / name).write_bytes(payload)
    from safetensors import safe_open

    with safe_open(checkpoint / "model.safetensors", framework="numpy") as source:
        source_parameters = {
            name: {
                "dtype": "torch." + str(source.get_tensor(name).dtype),
                "shape": list(source.get_tensor(name).shape),
                "sha256": hashlib.sha256(source.get_tensor(name).tobytes()).hexdigest(),
            }
            for name in source.keys()
        }
    assert source_parameters == frozen["checkpoint_parameters"]
    with restore_model_initialization_state():
        native = Qwen4ExpForCausalLM.from_pretrained(checkpoint, dtype=torch.float16).eval()
    actual_parameters = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest(),
        }
        for name, value in native.state_dict().items()
    }
    assert actual_parameters == frozen["loaded_parameters"], "Controlled checkpoint load changed parameters"
    config = {**frozen["export_config"], "hf_model": str(checkpoint)}
    declared_quant = copy.deepcopy(config["quant_scheme"])
    model = XHQwen4ExpModel(XHQwen4ExpModelConfig(**config))
    model.to_wrap(native)
    package = tmp_path / "package"
    metadata = model.export_hmonnx(str(package))
    saved = json.loads((package / "golden_meta_info.json").read_text())
    assert saved["model_config"]["quant_scheme"] == declared_quant
    assert "spec_decode" not in saved
    assert set(saved["qwen4_exp_ple"]) == {"prefill", "decode"}
    results = {}
    for phase in ("prefill", "decode"):
        relative = saved[phase + "_hmonnx"]
        assert relative == getattr(metadata, phase + "_hmonnx")
        assert not Path(relative).is_absolute() and Path(relative).parent == Path(phase)
        path = package / relative
        onnx.checker.check_model(str(path))
        graph = onnx.load(path, load_external_data=False).graph
        assert all(node.op_type != "PlaceHolder" for node in graph.node)
        assert {layer["input_name"] for layer in saved["qwen4_exp_ple"][phase]["layers"]} == {
            value.name for value in graph.input if value.name.startswith("ple_embeddings_")
        }
        assert all(not value.name.startswith(("present_past_", "_streamed_input_")) for value in graph.output)
        phase_files = [item for item in path.parent.rglob("*") if item.is_file()]
        assert all(item.suffix != ".json" for item in phase_files)
        assert [item.relative_to(path.parent) for item in phase_files if item.suffix == ".onnx"] == [Path(path.name)]
        assert not any(
            "qwen4_exp_text_external_data" in str(item.relative_to(path.parent))
            for item in phase_files if item.parent != path.parent
        )
        expected = unpack_frozen_manifest(frozen["phases"][phase])
        actual = graph_manifest(path)
        results[phase] = compare_manifests(expected, actual)
    assert all(result["port_labelled_dag_isomorphic"] for result in results.values()), {
        phase: {key: value for key, value in result.items() if key != "node_bijection"}
        for phase, result in results.items()
    }
    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import publish_qwen4_exp_qsa_layout_resources

    for phase in ("prefill", "decode"):
        path = package / getattr(metadata, phase + "_hmonnx")
        published_bytes = path.read_bytes()
        repeated = publish_qwen4_exp_qsa_layout_resources(
            path,
            batch_size=1,
            input_sequence_length=4 if phase == "prefill" else 1,
            num_attention_heads=native.config.num_attention_heads,
            num_key_value_heads=native.config.num_key_value_heads,
            head_dim=native.config.head_dim,
        )
        assert repeated["layout_constants_added"] == repeated["layout_views_added"] == 0
        assert path.read_bytes() == published_bytes


@pytest.mark.parametrize("unsupported", ["sparse_initializer", "quantization_annotation"])
def test_structure_comparison_rejects_unsupported_graph_contracts(tmp_path, unsupported):
    path = tmp_path / "unsupported.onnx"
    model = _mini_graph(path)
    getattr(model.graph, unsupported).add()
    onnx.save(model, path)
    with pytest.raises(ValueError, match="Sparse initializers and quantization annotations"):
        graph_manifest(path)


def _qsa_publication_graph(heads):
    """The typed selected-key/value and grouped-attention publication boundary."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    from xhquant.backend.xh2a import generate_relu_lut_table
    from xhquant.export.onnx.onnx_opset import xh2a_opset1

    # Load the same registered schemas as actual SDK serialization. Standalone
    # fast tests must check this fixture just as strictly as a post-export run.
    schema = onnx.defs.get_schema("SparseAttentionSelectKV", 1, xh2a_opset1.domain)
    lut_cut_points, lut_table, lut_scale = generate_relu_lut_table()
    select_attributes = dict(
        candidate_topk=1,
        score_scale=0.5,
        compression_ratio=4,
        pool_size=4,
        q_bit=16,
        k_bit=16,
        lut_cut_points=lut_cut_points,
        lut_table=lut_table,
        lut_scale=lut_scale,
    )
    assert {name for name, attribute in schema.attributes.items() if attribute.required} <= select_attributes.keys()
    matmul_attributes = dict(
        mode="sefp",
        hmfp_act_exp_bit=5,
        hmfp_act_man_bit=16,
        hmfp_act_nshare=64,
        hmfp_act_rounding="RNE",
        hmfp_act_hidden_bit=1,
        hmfp_weight_exp_bit=5,
        hmfp_weight_man_bit=16,
        hmfp_weight_nshare=64,
        hmfp_weight_rounding="RNE",
        hmfp_weight_hidden_bit=1,
        hmfp_psum_dtype="fp24",
        hmfp_psum_round_mode="trunc",
        output_dtype="float16",
    )
    dim, width = 4, 7
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape)
        for name, shape in (
            ("query", [1, heads, 1, dim]),
            ("index_query", [1, 1, 1, dim]),
            ("index_keys", [1, 1, 2, dim]),
            ("key_cache", [1, 1, 8, dim]),
            ("value_cache", [1, 1, 8, dim]),
            ("key_tail", [1, 1, 1, 3, dim]),
            ("value_tail", [1, 1, 1, 3, dim]),
            ("mask", [1, 1, 1, width]),
            # An unused graph input still owns its identifier.
            ("qsa_layout_packed_output_value", [1]),
        )
    ]
    inputs.extend(helper.make_tensor_value_info(name, TensorProto.INT32, [1]) for name in ("past", "current"))
    nodes = [
        helper.make_node(
            "SparseAttentionSelectKV",
            ["index_query", "index_keys", "key_cache", "past", "", "current", "value_cache"],
            ["selected_key", "selected_value"],
            domain="ai.houmo.xh2a",
            **select_attributes,
        ),
        helper.make_node("Concat", ["selected_key", "key_tail"], ["key_segments"], axis=-2),
        helper.make_node("Concat", ["selected_value", "value_tail"], ["value_segments"], axis=-2),
        helper.make_node("Reshape", ["key_segments", "key_shape"], ["packed_key"]),
        helper.make_node("Reshape", ["value_segments", "key_shape"], ["packed_value"]),
        helper.make_node("Transpose", ["packed_key"], ["transposed_key"], perm=[0, 1, 3, 2]),
        helper.make_node("Transpose", ["query"], ["grouped_query"], perm=[0, 2, 1, 3]),
        helper.make_node(
            "MatMul", ["grouped_query", "transposed_key"], ["scores"], domain="ai.houmo.xh2a", **matmul_attributes
        ),
        helper.make_node("Mul", ["scores", "scale"], ["scaled"]),
        helper.make_node("Add", ["scaled", "mask"], ["masked"]),
        helper.make_node("Softmax", ["masked"], ["probabilities"], axis=-1),
        helper.make_node(
            "MatMul", ["probabilities", "packed_value"], ["attended"], domain="ai.houmo.xh2a", **matmul_attributes
        ),
        helper.make_node("Reshape", ["attended", "hidden_shape"], ["output"]),
        # An unrelated, unused node's name must also remain unique.
        helper.make_node("Identity", ["query"], ["unused_debug"], name="qsa_layout_packed_query"),
    ]
    for index, node in enumerate(nodes):
        if not node.name:
            node.name = f"boundary_{index}"
        entry = node.metadata_props.add()
        entry.key, entry.value = "source_diagnostic", f"actual_boundary_{index}"
    initializers = [
        numpy_helper.from_array(np.array([1, 1, width, dim], dtype=np.int64), name="key_shape"),
        numpy_helper.from_array(np.array([1, 1, -1], dtype=np.int64), name="hidden_shape"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float16), name="scale"),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "qsa_publication",
            inputs,
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 1, heads * dim])],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("ai.houmo.xh2a", 1)],
    )
    helper.set_model_props(model, {"original_metadata": "preserve exactly"})
    return model


@pytest.mark.parametrize("heads,expected_views", [(1, 0), (2, 2)])
def test_qsa_publication_preserves_geometry_storage_metadata_and_identifier_ownership(
    tmp_path,
    monkeypatch,
    heads,
    expected_views,
):
    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import publish_qwen4_exp_qsa_layout_resources

    path = tmp_path / "qsa.onnx"
    original = _qsa_publication_graph(heads)
    onnx.save_model(
        original,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    storage = (tmp_path / "weights.bin").read_bytes()
    original_nodes = {node.name: node.SerializeToString(deterministic=True) for node in original.graph.node}
    original_load = onnx.load

    def load_without_weights(*args, **kwargs):
        assert kwargs.get("load_external_data") is False
        return original_load(*args, **kwargs)

    monkeypatch.setattr(onnx, "load", load_without_weights)
    geometry = dict(batch_size=1, input_sequence_length=1, num_attention_heads=heads, num_key_value_heads=1, head_dim=4)
    result = publish_qwen4_exp_qsa_layout_resources(path, **geometry)
    assert result["layout_views_added"] == expected_views
    assert (tmp_path / "weights.bin").read_bytes() == storage
    published = original_load(path, load_external_data=False)
    current_nodes = {node.name: node.SerializeToString(deterministic=True) for node in published.graph.node}
    assert all(current_nodes[name] == value for name, value in original_nodes.items())
    assert published.metadata_props == original.metadata_props
    assert len(current_nodes) == len(published.graph.node)
    all_values = [value for node in published.graph.node for value in node.output]
    assert "qsa_layout_packed_output_value" not in all_values
    added = [node for node in published.graph.node if node.name not in original_nodes]
    if heads == 2:
        assert len({node.input[1] for node in added}) == 1
        assert {(node.domain, tuple((a.name, a.i) for a in node.attribute)) for node in added} == {
            ("ai.houmo.xh2a", (("allowzero", 0),)),
            ("", (("allowzero", 1),)),
        }
    first = path.read_bytes()
    repeated = publish_qwen4_exp_qsa_layout_resources(path, **geometry)
    assert repeated["layout_constants_added"] == repeated["layout_views_added"] == 0
    assert path.read_bytes() == first
    onnx.checker.check_model(str(path))


@pytest.mark.parametrize("wrong_geometry", ["query_heads", "output_width"])
def test_qsa_publication_rejects_inconsistent_actual_shapes_atomically(tmp_path, wrong_geometry):
    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import publish_qwen4_exp_qsa_layout_resources

    path = tmp_path / "qsa.onnx"
    model = _qsa_publication_graph(2)
    if wrong_geometry == "output_width":
        model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value = 12
    onnx.save(model, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Qwen4 (query|attention output)"):
        publish_qwen4_exp_qsa_layout_resources(
            path,
            batch_size=1,
            input_sequence_length=1,
            num_attention_heads=3 if wrong_geometry == "query_heads" else 2,
            num_key_value_heads=1,
            head_dim=4,
        )
    assert path.read_bytes() == before


def test_qsa_publication_keeps_equal_shapes_in_independent_block_pools(tmp_path):
    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import publish_qwen4_exp_qsa_layout_resources

    model = _qsa_publication_graph(2)
    del model.graph.node[:]
    del model.graph.input[:]
    del model.graph.output[:]
    del model.graph.initializer[:]
    for owner in ("block_a", "block_b"):
        block = _qsa_publication_graph(2).graph
        for node in block.node:
            node.name = f"{owner}/{node.name}"
            for ports in (node.input, node.output):
                for index, value in enumerate(ports):
                    if value:
                        ports[index] = f"{owner}/{value}"
        for value in (*block.input, *block.output, *block.initializer):
            value.name = f"{owner}/{value.name}"
        model.graph.node.extend(block.node)
        model.graph.input.extend(block.input)
        model.graph.output.extend(block.output)
        model.graph.initializer.extend(block.initializer)
    path = tmp_path / "two_blocks.onnx"
    onnx.save(model, path)
    original_names = {node.name for node in model.graph.node}
    result = publish_qwen4_exp_qsa_layout_resources(
        path,
        batch_size=1,
        input_sequence_length=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    assert result == {"qsa_count": 2, "layout_constants_added": 6, "layout_views_added": 4}
    published = onnx.load(path)
    tensors = {tensor.name: tensor for tensor in published.graph.initializer}
    targets = {}
    for owner in ("block_a", "block_b"):
        views = [
            node
            for node in published.graph.node
            if node.name not in original_names and node.name.startswith(owner + "/")
        ]
        assert len(views) == 2
        (target,) = {node.input[1] for node in views}
        assert target.startswith(owner + "/")
        assert all(node.input[0].startswith(owner + "/") for node in views)
        targets[owner] = target
        owned_shapes = [tensor for name, tensor in tensors.items() if name.startswith(owner + "/qsa_layout_shape_")]
        assert len(owned_shapes) == 3
        assert {tuple(numpy_helper.to_array(tensor)) for tensor in owned_shapes} == {
            (1, 2, 1, 4),
            (1, 1, 2, 4),
            (1, 1, 1, 7),
        }
    assert targets["block_a"] != targets["block_b"]
    assert tensor_record(tensors[targets["block_a"]], tmp_path) == tensor_record(tensors[targets["block_b"]], tmp_path)
    published_bytes = path.read_bytes()
    repeated = publish_qwen4_exp_qsa_layout_resources(
        path,
        batch_size=1,
        input_sequence_length=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    assert repeated["layout_constants_added"] == repeated["layout_views_added"] == 0
    assert path.read_bytes() == published_bytes
    onnx.checker.check_model(str(path))


@pytest.mark.parametrize("failure_boundary", ["save", "checker"])
def test_qsa_publication_write_failures_preserve_original_graph_and_sidecar(tmp_path, monkeypatch, failure_boundary):
    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import publish_qwen4_exp_qsa_layout_resources

    path = tmp_path / "qsa.onnx"
    onnx.save_model(
        _qsa_publication_graph(2),
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    onnx.checker.check_model(str(path))
    sidecar = tmp_path / "weights.bin"
    before = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in (path, sidecar)}
    attempts = []
    original_save, original_check = onnx.save_model, onnx.checker.check_model

    if failure_boundary == "save":

        def fail_after_save(model, destination, *args, **kwargs):
            # Exercise the real transformation and temporary-file serialization.
            assert len(model.graph.node) == len(_qsa_publication_graph(2).graph.node) + 2
            original_save(model, destination, *args, **kwargs)
            attempts.append(Path(destination))
            raise OSError("injected save completion failure")

        monkeypatch.setattr(onnx, "save_model", fail_after_save)
    else:

        def fail_after_check(candidate, *args, **kwargs):
            # Validate the actual saved graph before injecting a checker failure.
            original_check(candidate, *args, **kwargs)
            attempts.append(Path(candidate))
            raise OSError("injected checker completion failure")

        monkeypatch.setattr(onnx.checker, "check_model", fail_after_check)
    with pytest.raises(OSError, match=f"injected {failure_boundary} completion failure"):
        publish_qwen4_exp_qsa_layout_resources(
            path,
            batch_size=1,
            input_sequence_length=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
    assert len(attempts) == 1
    assert attempts[0] != path and attempts[0].parent == tmp_path
    assert not attempts[0].exists()
    assert {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in (path, sidecar)} == before
    assert {file.name for file in tmp_path.iterdir()} == {path.name, sidecar.name}
