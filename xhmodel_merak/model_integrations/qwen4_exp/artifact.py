"""Check the Qwen4 state ABI at publication and loading."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import artifact_file, confined_file
from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_integrations.autoregressive.artifact import (
    validate_artifact as validate_autoregressive_artifact,
)


def validate_component_vision(
    root: Path, metadata: Mapping[str, Any], *, image_token_gears=(),
) -> dict[str, Any] | None:
    """Own Qwen4's streamed vision contract alongside its text/cache ABI."""
    if "qwen4_exp_vision" not in metadata:
        return None
    try:
        from xhmodel_merak.xh_llm.models.qwen4_exp.vision_contract import validate_qwen4_exp_vision

        if metadata.get("visual_config"):
            raise ValueError("Qwen4 must declare one vision ABI, not both component and legacy graphs")
        config_path = confined_file(root.resolve(), metadata["hf_config"] + "/config.json", "hf_config")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return validate_qwen4_exp_vision(
            metadata["qwen4_exp_vision"], root,
            image_token_gears=image_token_gears or None, vision_config=config["vision_config"],
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, f"Invalid Qwen4 component vision: {exc}") from exc


def validate_serving_contract(root: Path, metadata: Mapping[str, Any]) -> None:
    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import (
        QWEN4_EXP_LLM_CACHE_ABI,
        resolve_qwen4_exp_cache_abi,
    )

    validate_component_vision(root, metadata)
    try:
        validate_external_ple_contract(root, metadata)
        config_path = confined_file(root.resolve(), metadata["hf_config"] + "/config.json", "hf_config")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected = _declared_cache_contract(config["text_config"], metadata.get("qwen4_exp_cache_contract"))
        serving = metadata.get("qwen4_exp_serving", {})
        cache_abi = resolve_qwen4_exp_cache_abi(metadata.get("qwen4_exp_cache_abi"))
        explicit_cache = cache_abi == QWEN4_EXP_LLM_CACHE_ABI
        if (
            not isinstance(serving, Mapping)
            or serving.get("enable_page_attention") is not (not explicit_cache)
            or type(serving.get("max_num_seqs")) is not int
            or serving["max_num_seqs"] != 1
            or serving.get("qsa_attention") != "selected_kv_matmul_attention"
        ):
            raise ValueError("Qwen4 unified inference requires a consistent single-sequence cache ABI")
        model = metadata["model_config"]
        if (
            not isinstance(model, Mapping)
            or model.get("model_type") != "Qwen4ExpForConditionalGeneration"
            or model.get("qwen4_exp_internal_state") is not True
        ):
            raise ValueError("Qwen4 runtime must own its internal recurrent and PLE state")
        if metadata.get("spec_decode") is not None and model.get("return_hidden_states") is not True:
            raise ValueError("Qwen4 MTP requires pre-mixer target HC output")
        attention_layers = [
            layer.layer_idx
            for layer in expected.layers
            if layer.layer_type in {"full_attention", "qwen_sparse_attention"}
        ]
        if model.get("qwen4_exp_attention_layers") != attention_layers:
            raise ValueError("Qwen4 attention layer mapping disagrees with its cache contract")
        kv_cache = metadata.get("kv_cache", {})
        expected_shape = [1, expected.num_key_value_heads, model["context_max_length"], expected.head_dim]
        if (
            type(model.get("batch_size")) is not int
            or model["batch_size"] != 1
            or not isinstance(kv_cache, Mapping)
            or kv_cache.get("num_layers") != len(attention_layers)
            or kv_cache.get("kv_cache_shape") != [expected_shape, expected_shape]
            or kv_cache.get("cache_axis") != 2
            or kv_cache.get("cache_dtype") != "float16"
            or kv_cache.get("batch_size") != 1
            or kv_cache.get("use_cache") is not True
        ):
            raise ValueError("Qwen4 KV cache layout disagrees with its cache contract")
        import onnx

        for phase, length in (("prefill", model["prefill_chunk_length"]), ("decode", 1)):
            graph_path = confined_file(root.resolve(), metadata[f"{phase}_hmonnx"], f"{phase}_hmonnx")
            graph = onnx.load(str(graph_path), load_external_data=False).graph
            _validate_mrope_inputs(graph, length, phase)
            if explicit_cache:
                _validate_explicit_kv_cache_graph(graph, expected, model["context_max_length"], phase)
        validate_mtp_v3_contract(root, metadata, config["text_config"], expected)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            f"Invalid Qwen4 serving contract: {exc}",
        ) from exc


def _declared_cache_contract(text_config, declared):
    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import Qwen4ExpCacheContract

    expected = Qwen4ExpCacheContract.from_text_config(text_config)
    if declared != expected.to_dict():
        raise ValueError("Qwen4 cache contract does not match the packaged text configuration")
    return expected


def _validate_mrope_inputs(graph, length, role):
    import onnx

    from xhmodel_merak.xh_llm.mrope import MROPE_CURRENT_POSITION_NAMES, MROPE_POOL_POSITION_NAMES

    inputs = {value.name: value for value in graph.input}
    outputs = {value.name for value in graph.output}
    positions = ((name, length) for name in MROPE_CURRENT_POSITION_NAMES)
    groups = ((name, (length + 3) // 4) for name in MROPE_POOL_POSITION_NAMES)
    for name, width in (*positions, *groups):
        value = inputs.get(name)
        if (value is None or value.type.tensor_type.elem_type != onnx.TensorProto.INT32
                or [dim.dim_value for dim in value.type.tensor_type.shape.dim] != [1, width]):
            raise ValueError(f"Qwen4 {role} tensor ABI mismatch: {name}")
    if "past_position_ids" in inputs or "present_past_position_ids" in outputs:
        raise ValueError(f"Qwen4 {role} must use compact T/H/W position inputs")


def _kv_input_names(contract):
    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import qwen4_exp_cache_input_specs

    return {
        tensor.name: name for name, tensor in qwen4_exp_cache_input_specs(contract).items()
        if tensor.state_kind in {"kv_key", "kv_value"}
    }


def _validate_explicit_kv_cache_graph(graph, contract, context_length, role):
    """Validate actual cache ports and writers, not just a metadata opt-in."""
    import onnx

    names = _kv_input_names(contract)
    inputs = {value.name: value for value in graph.input}
    producers = {name: node for node in graph.node for name in node.output}

    def source(name):
        seen = set()
        while name not in seen:
            seen.add(name)
            node = producers.get(name)
            if node is None or node.op_type not in {"Cast", "Identity", "HMIdentity"} or len(node.input) != 1:
                break
            name = node.input[0]
        return name

    if any(node.op_type.startswith(("PageKVCache", "PageAttention")) for node in graph.node):
        raise ValueError(f"Qwen4 {role} explicit cache graph still contains paged cache operators")
    for tensor in contract.tensors:
        if tensor.name not in names:
            continue
        name = names[tensor.name]
        value = inputs.get(name)
        shape = [{"batch": 1, "cache_length": context_length}.get(dim, dim) for dim in tensor.shape]
        if (value is None or value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT16
                or [dim.dim_value for dim in value.type.tensor_type.shape.dim] != shape):
            raise ValueError(f"Qwen4 {role} explicit cache input mismatch: {name}")
        writers = [node for node in graph.node if node.op_type in {"KVcache", "LLMCache"}
                   and len(node.input) >= 4 and source(node.input[3]) == name]
        if len(writers) != 1:
            raise ValueError(f"Qwen4 {role} requires exactly one LLMCache writer for {name}")
        writer = writers[0]
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in writer.attribute}
        if (attrs.get("axis") != 2 or attrs.get("attention_max_length") != -1
                or attrs.get("only_handle_old_cache", 0) != 0
                or [source(name) for name in writer.input[1:3]] != ["past_seq_length", "current_input_length"]):
            raise ValueError(f"Qwen4 {role} LLMCache writer contract mismatch: {name}")


def validate_artifact(
    root: Path,
    manifest: Mapping[str, Any],
    meta_file: Path,
    recipe: ModelRecipe,
) -> None:
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    component_vision = validate_component_vision(root, metadata, image_token_gears=recipe.vision_gears)
    # Component vision has its own tensor ABI; the shared vision validator
    # describes Qwen3.5's single-graph gears. Keep every shared text, source,
    # resource and manifest check, and validate all vision components above.
    text_recipe = replace(recipe, vision_gears=(), vision_profile=None) if component_vision else recipe
    validate_autoregressive_artifact(root, manifest, meta_file, text_recipe)
    validate_serving_contract(root, metadata)
    for entry in {**validate_external_ple_contract(root, metadata), **mtp_v3_graph_entries(metadata)}.values():
        artifact_file(root.resolve(), manifest, entry["path"], "qwen4_exp_ple.resource")


def validate_external_ple_contract(root: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate external host PLE resources against each actual graph input ABI."""
    import onnx

    root = root.resolve()
    declared = metadata.get("qwen4_exp_ple")
    phases = {"prefill": metadata["prefill_hmonnx"], "decode": metadata["decode_hmonnx"]}
    spec = metadata.get("spec_decode")
    if isinstance(spec, Mapping) and spec.get("runtime_contract_version") in (3, 4):
        verify_role = spec.get("target_verify_graph", "decode_m2")
        phases[verify_role] = metadata.get(f"{verify_role}_hmonnx")
    if declared is not None and (not isinstance(declared, Mapping) or set(declared) != set(phases)):
        raise ValueError("External PLE metadata must describe every target phase")
    entries = {}
    for phase, graph_value in phases.items():
        graph_path = confined_file(root, graph_value, f"{phase}_hmonnx")
        try:
            graph = onnx.load(str(graph_path), load_external_data=False)
        except Exception as exc:
            raise ValueError(f"Cannot read {phase} PLE graph: {exc}") from exc
        values = [item.value for item in graph.metadata_props if item.key == "qwen4_exp_ple"]
        if len(values) > 1:
            raise ValueError("Graph has duplicate PLE metadata")
        graph_manifest = json.loads(values[0]) if values else None
        phase_manifest = declared[phase] if declared is not None else None
        if phase_manifest is None and graph_manifest is None:
            if any(item.name.startswith("ple_embeddings_") for item in graph.graph.input):
                raise ValueError("External PLE graph inputs require a manifest")
            continue  # Existing artifacts keep their original in-graph PLE ABI.
        if phase_manifest is None or graph_manifest is None:
            raise ValueError("Root and graph PLE metadata must both be present")
        if (not isinstance(graph_manifest, Mapping) or type(graph_manifest.get("version")) is not int
                or graph_manifest.get("version") != 1):
            raise ValueError("Unsupported external PLE graph manifest")
        layers = graph_manifest.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("External PLE manifest requires layers")
        if any(node.op_type == "Qwen4PLESEFP4Gather" for node in graph.graph.node):
            raise ValueError("External PLE graph still contains Qwen4PLESEFP4Gather")
        if any("ple" in item.name and "ngram_embedding" in item.name and len(item.dims) >= 2
               for item in graph.graph.initializer):
            raise ValueError("External PLE graph still embeds its lookup table")
        rebased = json.loads(json.dumps(graph_manifest))
        inputs = {item.name: item for item in graph.graph.input}
        layer_ids, names = set(), set()
        for spec in rebased["layers"]:
            if not isinstance(spec, dict):
                raise ValueError("Invalid external PLE layer specification")
            layer = spec.get("layer_idx")
            width = spec.get("embedding_dim")
            ngram = spec.get("ngram_size")
            heads = spec.get("heads_per_ngram")
            if (type(layer) is not int or layer < 0 or layer in layer_ids
                    or any(type(value) is not int or value < 1 for value in (width, heads))
                    or type(ngram) is not int or ngram < 2):
                raise ValueError("Invalid external PLE layer dimensions")
            layer_ids.add(layer)
            name = spec.get("input_name")
            if name != f"ple_embeddings_{layer}" or name in names or name not in inputs:
                raise ValueError("External PLE input name is missing or inconsistent")
            names.add(name)
            dtype = {"float16": onnx.TensorProto.FLOAT16, "bfloat16": onnx.TensorProto.BFLOAT16}.get(
                spec.get("output_dtype")
            )
            tensor = inputs[name].type.tensor_type
            if dtype is None or tensor.elem_type != dtype:
                raise ValueError("External PLE input dtype disagrees with metadata")
            dimensions = tensor.shape.dim
            # The selected table rows retain their ngram-head axis at the boundary.
            if (len(dimensions) != 4 or dimensions[-2].dim_value != (ngram - 1) * heads
                    or dimensions[-1].dim_value != width):
                raise ValueError("External PLE input shape disagrees with metadata")
            reference = inputs.get("ple_input_ids", inputs.get("input_ids"))
            if reference is not None and list(dimensions[:2]) != list(reference.type.tensor_type.shape.dim[:2]):
                raise ValueError("External PLE batch/sequence axes disagree with token inputs")
            if any(item.name == name for item in graph.graph.initializer):
                raise ValueError("External PLE inputs cannot have initializer defaults")
            multipliers, vocab, offsets = (spec.get(key) for key in ("multipliers", "vocab_sizes", "offsets"))
            if (type(spec.get("eos_token_id")) is not int
                    or not isinstance(multipliers, list) or len(multipliers) != ngram
                    or any(type(value) is not int for value in multipliers)
                    or not isinstance(vocab, list) or len(vocab) != (ngram - 1) * heads
                    or any(type(value) is not int or value < 1 for value in vocab)
                    or not isinstance(offsets, list) or len(offsets) != len(vocab)
                    or any(type(value) is not int or value < 0 for value in offsets)):
                raise ValueError("Invalid external PLE hashing contract")
            expected_offsets, total = [], 0
            for size in vocab:
                expected_offsets.append(total)
                total += size
            if offsets != expected_offsets:
                raise ValueError("External PLE vocabulary offsets are inconsistent")
            groups = (width + 63) // 64
            packed_resource = spec.get("packed")
            packed_shape = packed_resource.get("shape") if isinstance(packed_resource, Mapping) else None
            if (not isinstance(packed_shape, list) or len(packed_shape) != 3
                    or type(packed_shape[0]) is not int or packed_shape[0] < total):
                raise ValueError("External PLE table cannot contain every vocabulary range")
            # Native checkpoints can pad the final table beyond the last head.
            # Preserve all stored rows while requiring every hash range in bounds.
            rows = packed_shape[0]
            for kind, expected_dtype, tail, itemsize in (
                ("packed", "int8", [groups, 32], 1), ("exponent", "int16", [groups, 1], 2),
            ):
                resource = spec.get(kind)
                if (not isinstance(resource, dict) or resource.get("dtype") != expected_dtype
                        or not isinstance(resource.get("shape"), list)
                        or any(type(value) is not int for value in resource["shape"])
                        or resource["shape"] != [rows, *tail]):
                    raise ValueError(f"Invalid external PLE {kind} resource layout")
                path = confined_file(graph_path.parent, resource.get("path"), f"ple.{kind}")
                expected_size = rows * tail[0] * tail[1] * itemsize
                if path.stat().st_size != expected_size:
                    raise ValueError(f"External PLE {kind} resource size differs from its shape")
                relative = path.relative_to(root).as_posix()
                resource["path"] = relative
                entries[f"ple_{phase}_{layer}_{kind}"] = {"path": relative, "type": "embedding"}
        declared_order = [spec["input_name"] for spec in rebased["layers"]]
        if [item.name for item in graph.graph.input][-len(declared_order):] != declared_order:
            raise ValueError("External PLE inputs must follow the original graph inputs")
        if {name for name in inputs if name.startswith("ple_embeddings_")} != names:
            raise ValueError("External PLE graph has undeclared embedding inputs")
        if rebased != phase_manifest:
            raise ValueError("Root and graph PLE manifests disagree")
    return entries



def mtp_v3_graph_entries(metadata: Mapping[str, Any]) -> dict[str, Any]:
    spec = metadata.get("spec_decode")
    if not isinstance(spec, Mapping) or spec.get("runtime_contract_version") not in (3, 4):
        return {}
    return {"target_verify": {"path": spec["target"]["verify_hmonnx"], "type": "decode"}}


def validate_mtp_v3_contract(root, metadata, text, target_contract):
    """Require independent draft phases and the declared target verify width."""
    import onnx
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import (
        QWEN4_EXP_LLM_CACHE_ABI,
        resolve_qwen4_exp_cache_abi,
        resolve_qwen4_exp_cache_shape,
    )

    spec = metadata.get("spec_decode")
    if spec is None:
        return
    version = spec.get("runtime_contract_version") if isinstance(spec, Mapping) else None
    if version == 2:
        return  # The existing version-2 build checks retain their original contract.
    if version not in (3, 4) or spec.get("mode") != "mtp":
        raise ValueError("Unsupported Qwen4 speculative runtime contract")
    count = spec.get("num_draft_tokens")
    if (type(count) is not int or count < 1 or (version == 3 and count != 1)
            or type(spec.get("verify_length")) is not int or spec["verify_length"] < count + 1):
        raise ValueError("Qwen4 MTP requires positive draft count and verify length >= num_draft_tokens + 1")
    if version == 4:
        from xhmodel_merak.xh_llm.shared_head import resolve_shared_head_bits

        bits = resolve_shared_head_bits(metadata["model_config"].get("quant_scheme"))
        shared = spec.get("shared_head")
        if (not isinstance(shared, Mapping) or type(shared.get("version")) is not int
                or type(shared.get("logical_bits")) is not int
                or shared != {"version": 1, "owner": "target", "logical_bits": bits}
                or type(spec.get("draft_head_weight_bits")) is not int
                or spec["draft_head_weight_bits"] != bits):
            raise ValueError("Qwen4 MTP shared head must declare the target's logical precision")
    verify_length = spec["verify_length"]
    verify_role = f"decode_m{verify_length}"
    target, draft = spec.get("target"), spec.get("draft")
    if not isinstance(target, Mapping) or not isinstance(draft, Mapping):
        raise ValueError("Qwen4 MTP v3 requires target and draft contracts")
    cache_abi = resolve_qwen4_exp_cache_abi(metadata.get("qwen4_exp_cache_abi"))
    if resolve_qwen4_exp_cache_abi(draft.get("cache_abi")) != cache_abi:
        raise ValueError("Qwen4 target and draft cache ABI disagree")
    explicit_cache = cache_abi == QWEN4_EXP_LLM_CACHE_ABI
    prefill_length = metadata["model_config"]["prefill_chunk_length"]
    if verify_length > metadata["model_config"]["context_max_length"]:
        raise ValueError("Qwen4 target verify length exceeds cache capacity")
    if (any(type(value) is not int for value in (target.get("verify_chunk_length"),
                                                draft.get("prefill_chunk_length"), draft.get("decode_chunk_length")))
            or target.get("verify_chunk_length") != verify_length or spec.get("target_verify_graph") != verify_role
            or spec.get("target_verify_padding") != verify_length
            or draft.get("prefill_chunk_length") != prefill_length or draft.get("decode_chunk_length") != 1):
        raise ValueError("Qwen4 MTP v3 chunk lengths disagree with their graph roles")
    verify = target.get("verify_hmonnx")
    if metadata.get(f"{verify_role}_hmonnx") != verify:
        raise ValueError("Qwen4 target verify paths disagree")
    root = root.resolve()
    verify_path = confined_file(root, verify, "spec_decode.target.verify_hmonnx")
    if metadata.get(f"{verify_role}_hmonnx_md5") != hashlib.md5(verify_path.read_bytes()).hexdigest():
        raise ValueError("Qwen4 target verify graph digest disagrees with metadata")
    hidden, hc, vocab = text["hidden_size"], text["hc_count"], text["vocab_size"]
    if draft.get("hidden_size") != hidden or draft.get("hc_count") != hc:
        raise ValueError("Qwen4 draft HC dimensions disagree with the target")
    mtp = text.get("mtp", {})
    if (mtp.get("num_hidden_layers", text.get("mtp_num_hidden_layers")) != 1
            or mtp.get("layer_types") != ["full_attention"]):
        raise ValueError("Qwen4 MTP requires the native single full-attention assistant")
    draft_text = dict(text, num_hidden_layers=1, layer_types=["full_attention"], ple_layer_ids=[])
    expected_draft = _declared_cache_contract(Qwen4ExpTextConfig(**draft_text), draft.get("cache_contract"))
    if target_contract.ple_layer_indices:
        ple = metadata.get("qwen4_exp_ple", {})
        if set(ple) != {"prefill", "decode", verify_role} or any(
            {layer["layer_idx"] for layer in phase["layers"]} != set(target_contract.ple_layer_indices)
            for phase in ple.values()
        ):
            raise ValueError("Qwen4 MTP v3 requires all target PLE resources for every phase")
    paths = []
    for role, path, length, contract, embedding, output in (
        ("mtp_prefill", draft.get("prefill_hmonnx"), prefill_length, expected_draft,
         "next_token_embedding", "mtp_hidden_states"),
        ("mtp_decode", draft.get("decode_hmonnx"), 1, expected_draft,
         "next_token_embedding", "mtp_hidden_states"),
        (verify_role, verify, verify_length, target_contract, "inputs_embeds", "target_hidden_states"),
    ):
        graph_path = confined_file(root, path, f"spec_decode.{role}")
        if graph_path.relative_to(root).parts[0] != role:
            raise ValueError("Qwen4 MTP v3 graphs must use independent flat role directories")
        paths.append(graph_path)
        graph = onnx.load(str(graph_path), load_external_data=False).graph
        inputs = {value.name: value for value in graph.input}
        outputs = {value.name: value for value in graph.output}

        def require(values, name, shape, dtype, role=role):
            value = values.get(name)
            if (value is None or value.type.tensor_type.elem_type != dtype
                    or [dim.dim_value for dim in value.type.tensor_type.shape.dim] != shape):
                raise ValueError(f"Qwen4 {role} tensor ABI mismatch: {name}")

        require(inputs, embedding, [1, length, hidden], onnx.TensorProto.FLOAT16)
        if role != verify_role:
            require(inputs, "target_hidden", [1, length, hidden * hc], onnx.TensorProto.FLOAT16)
        else:
            require(inputs, "ple_input_ids", [1, length], onnx.TensorProto.INT32)
        for name in ("past_seq_length", "current_input_length"):
            require(inputs, name, [1], onnx.TensorProto.INT32)
        _validate_mrope_inputs(graph, length, role)
        allowed_inputs = {embedding, "past_seq_length", "current_input_length"}
        from xhmodel_merak.xh_llm.mrope import MROPE_CURRENT_POSITION_NAMES, MROPE_POOL_POSITION_NAMES

        allowed_inputs.update(MROPE_CURRENT_POSITION_NAMES + MROPE_POOL_POSITION_NAMES)
        allowed_inputs.update({"ple_input_ids"} if role == verify_role else {"target_hidden"})
        if role == verify_role:
            allowed_inputs.update(f"ple_embeddings_{layer}" for layer in contract.ple_layer_indices)
        cache_names = _kv_input_names(contract) if explicit_cache else {}
        for tensor in contract.tensors:
            if tensor.state_kind in {"kv_key", "kv_value"} and not explicit_cache:
                continue  # PageKVCache owns K/V outside the argument list.
            name = cache_names.get(tensor.name, tensor.name)
            allowed_inputs.add(name)
            shape = list(resolve_qwen4_exp_cache_shape(
                tensor, batch_size=1, context_length=metadata["model_config"]["context_max_length"],
            ))
            require(inputs, name, shape,
                    onnx.TensorProto.INT32 if tensor.dtype == "integer" else onnx.TensorProto.FLOAT16)
            if tensor.state_kind in {"qsa_indexer_tail", "qsa_key_tail", "qsa_value_tail"}:
                require(outputs, f"present_{name}", shape, onnx.TensorProto.FLOAT16)
        if any(layer.layer_type in {"full_attention", "qwen_sparse_attention"} for layer in contract.layers):
            context = metadata["model_config"]["context_max_length"]
            width = min(int(text["indexer_budget"]) // 4, context // 4) * 4 + 3
            controls = {
                "qsa_pool_offset": ([1], onnx.TensorProto.INT32),
                "qsa_pool_new_count": ([1], onnx.TensorProto.INT32),
                "qsa_pool_write_start": ([1], onnx.TensorProto.INT32),
                "qsa_query_valid": ([1, length], onnx.TensorProto.FLOAT16),
                "qsa_attention_mask": ([1, length, 1, width], onnx.TensorProto.FLOAT16),
            }
            for name, (shape, dtype) in controls.items():
                allowed_inputs.add(name)
                require(inputs, name, shape, dtype)
        if set(inputs) != allowed_inputs:
            raise ValueError(f"Qwen4 {role} graph input set disagrees with its cache contract")
        attention_count = sum(layer.layer_type in {"full_attention", "qwen_sparse_attention"}
                              for layer in contract.layers)
        if explicit_cache:
            _validate_explicit_kv_cache_graph(graph, contract, metadata["model_config"]["context_max_length"], role)
        elif sum(node.op_type == "PageKVCache" for node in graph.node) != attention_count:
            raise ValueError(f"Qwen4 {role} PageKVCache graph layout is incomplete")
        if role == verify_role:
            if any(node.op_type in {"GDRChunkScan", "GDRBlockTriInverse"} for node in graph.node):
                raise ValueError("Qwen4 verify must use recurrent decode arithmetic, not prefill GDR chunks")
            recurrent = [tensor for tensor in contract.tensors if tensor.state_kind == "delta_recurrent"]
            recurrent_scans = sum(node.op_type == "GDRRecurrentScan" for node in graph.node)
            if recurrent_scans or metadata["model_config"].get("use_fused_gdr") is True:
                if recurrent_scans != length * len(recurrent):
                    raise ValueError("Qwen4 verify requires one recurrent GDR update per token and layer")
            for tensor in recurrent:
                shape = [{"batch": 1}.get(dim, dim) for dim in tensor.shape]
                require(outputs, f"present_{tensor.name}", shape, onnx.TensorProto.FLOAT16)
        require(outputs, "logits", [1, length, vocab], onnx.TensorProto.FLOAT16)
        require(outputs, output, [1, length, hidden * hc], onnx.TensorProto.FLOAT16)
    if len(set(paths)) != 3 or verify_path in {
        confined_file(root, metadata[key], key) for key in ("prefill_hmonnx", "decode_hmonnx")
    }:
        raise ValueError("Qwen4 MTP v3 graph roles must not alias")
    for key, expected in (("draft_prefill_onnx", draft["prefill_hmonnx"]),
                          ("draft_decode_onnx", draft["decode_hmonnx"]),
                          ("draft_onnx", draft["decode_hmonnx"])):
        if spec.get(key) != expected:
            raise ValueError("Qwen4 MTP v3 draft aliases disagree")
