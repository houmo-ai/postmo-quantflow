"""GLM's frame component paths and hybrid state ABI, shared by build and load."""

import hashlib
import json
from pathlib import Path

from xhmodel_merak.errors import ErrorCode, ModelError


def validate_glm53_bundle(root, metadata):
    from xhmodel_merak.xh_llm.models.glm5_3_flash.contract import (
        Glm53CacheSpec,
        validate_state_abi,
        validate_state_graph,
    )

    root = Path(root).resolve()
    try:
        validate_state_abi(metadata.get("glm53_state_abi_version"))
    except ValueError as exc:
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, str(exc)) from exc
    if metadata.get("glm53_export_mode") != "streamed_ffn":
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Unsupported GLM hybrid state/export ABI")
    vision = metadata.get("glm53_vision")
    if not isinstance(vision, dict) or vision.get("abi_version") not in (1, 2) or vision.get("dtype") != "float16":
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "GLM multimodal artifact requires vision ABI 1 or 2")
    config = vision["vision_config"]
    names = ["stem", *[f"block_{i:02d}" for i in range(config["depth"])], "merger"]
    components = vision.get("components", [])
    capacity, merge = vision["max_patches_per_frame"], config["spatial_merge_size"]
    if capacity <= 0 or capacity % (merge * merge) or [x.get("name") for x in components] != names:
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Invalid GLM vision capacity or component order")
    for component in components:
        relative = Path(vision["directory"]) / component["hmonnx"]
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or not path.is_file()
        ):
            raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, f"Invalid GLM vision path: {relative}")
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != component["md5"]:
            raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, f"GLM vision graph checksum mismatch: {relative}")
    spec = metadata.get("glm53_cache_spec", {})
    if len(spec.get("layer_types", [])) != metadata["kv_cache"]["num_layers"]:
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "GLM state layer count mismatch")
    import onnx

    for stage in ("prefill", "decode"):
        graph = onnx.load(root / metadata[stage + "_hmonnx"], load_external_data=False)
        try:
            validate_state_graph(graph.graph, Glm53CacheSpec(**spec), stage)
        except ValueError as exc:
            raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, str(exc)) from exc


def validate_artifact(root, manifest, meta_file, recipe):
    from xhmodel_merak.model_integrations.autoregressive.artifact import validate_artifact as validate_text

    validate_text(root, manifest, meta_file, recipe)
    validate_glm53_bundle(root, json.loads(Path(meta_file).read_text()))


def audit_glm53_hmonnx(root, metadata):
    """Inspect actual operator precision; loading external weight data is unnecessary."""
    import onnx

    root = Path(root)
    config = json.loads((root / metadata["hf_config"] / "config.json").read_text())["text_config"]
    kinds = metadata["glm53_cache_spec"]["layer_types"]
    expected_moe = sum(x == "sparse" for x in config["mlp_layer_types"][: len(kinds)])
    expected_gates = 3 * kinds.count("linear_attention")
    expected_projections = 2 * kinds.count("deepseek_sparse_attention")
    expected_convolutions = kinds.count("linear_attention")
    expected_sinkhorn = 2 * len(kinds)
    report = {}
    for stage in ("prefill", "decode"):
        graph = onnx.load(root / metadata[stage + "_hmonnx"], load_external_data=False)
        initializers = {value.name for value in graph.graph.initializer}
        moe_count = gate_count = w8_count = 0
        projection_count = convolution_count = sinkhorn_count = select_count = scan_count = 0
        chunk_count = recurrent_count = a16_matmul_count = 0
        for node in graph.graph.node:
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if node.op_type == "MoeBlock":
                if (
                    attrs.get("hmfp_weight_man_bit"),
                    attrs.get("hmfp_weight_nshare"),
                    attrs.get("hmfp_act_man_bit"),
                ) != (4, 64, 16):
                    raise ValueError(f"{stage}: routed experts are not W4 G64 A16: {node.name}")
                if attrs.get("limit") != config["swiglu_limit"] or attrs.get("k") != config["num_experts_per_tok"]:
                    raise ValueError(f"{stage}: routed expert gate semantics changed: {node.name}")
                moe_count += 1
            elif node.op_type == "Linear":
                weight_name = node.input[1]
                gate = any(
                    f".attention.{name}." in weight_name
                    for name in ("forget_gate.f_a_proj", "forget_gate.f_b_proj", "b_proj")
                )
                expected_bits = 16 if gate else 8
                if attrs.get("hmfp_weight_man_bit") != expected_bits or attrs.get("hmfp_act_man_bit") != 16:
                    raise ValueError(f"{stage}: invalid W{expected_bits} A16 linear: {weight_name}")
                gate_count += gate
                w8_count += not gate
            elif node.op_type == "MatMul":
                constant_rhs = node.input[1] in initializers
                expected_bits = 8 if constant_rhs else 16
                if (
                    attrs.get("hmfp_weight_man_bit"),
                    attrs.get("hmfp_act_man_bit"),
                ) != (expected_bits, 16):
                    kind = "constant RHS" if constant_rhs else "activation"
                    raise ValueError(f"{stage}: invalid {kind} MatMul precision: {node.name}")
                projection_count += constant_rhs
                a16_matmul_count += not constant_rhs
            elif node.op_type == "Conv2d":
                if (attrs.get("hmfp_weight_man_bit"), attrs.get("hmfp_act_man_bit")) != (8, 16):
                    raise ValueError(f"{stage}: KDA convolution is not W8 A16: {node.name}")
                convolution_count += 1
            elif node.op_type == "MHCSinkhorn":
                if (
                    attrs.get("sinkhorn_iters") != config["hc_sinkhorn_iters"]
                    or abs(attrs.get("eps", float("inf")) - config["hc_eps"]) > 1e-10
                ):
                    raise ValueError(f"{stage}: invalid mHC Sinkhorn parameters: {node.name}")
                sinkhorn_count += 1
            elif node.op_type in ("GDRChunkScan", "GDRRecurrentScan"):
                expected_scan = "GDRChunkScan" if stage == "prefill" else "GDRRecurrentScan"
                if node.op_type != expected_scan or len(node.output) != 1:
                    raise ValueError(f"{stage}: invalid KDA scan contract: {node.name}")
                if stage == "prefill" and attrs.get("matmul_input_bit") != 16:
                    raise ValueError(f"{stage}: KDA scan MatMul is not A16: {node.name}")
                scan_count += 1
                chunk_count += node.op_type == "GDRChunkScan"
                recurrent_count += node.op_type == "GDRRecurrentScan"
            elif node.op_type == "SparseAttentionSelectKV":
                expected_k = min(config["index_topk"] // 4, metadata["glm53_cache_spec"]["max_context_length"] // 4)
                if (
                    attrs.get("score_mode") != b"vp"
                    or attrs.get("pool_size") != 4
                    or attrs.get("candidate_topk") != expected_k
                    or len(node.input) != 6
                    or len(node.output) != 1
                    or abs(attrs.get("score_scale", float("inf")) - config["index_head_dim"] ** -0.5) > 1e-7
                ):
                    raise ValueError(f"{stage}: invalid pooled DSA SelectKV contract: {node.name}")
                select_count += 1
        if (moe_count, gate_count) != (expected_moe, expected_gates):
            raise ValueError(
                f"{stage}: missing MoE/KDA gates: {(moe_count, gate_count)} != {(expected_moe, expected_gates)}"
            )
        if (projection_count, convolution_count) != (expected_projections, expected_convolutions):
            raise ValueError(
                f"{stage}: missing DSA projections/KDA convolutions: "
                f"{(projection_count, convolution_count)} != {(expected_projections, expected_convolutions)}"
            )
        if scan_count != kinds.count("linear_attention"):
            raise ValueError(f"{stage}: missing KDA scan operators: {scan_count}")
        if sinkhorn_count != expected_sinkhorn:
            raise ValueError(f"{stage}: missing mHC Sinkhorn operators: {sinkhorn_count} != {expected_sinkhorn}")
        if (select_count or metadata.get("glm53_dsa_selection") == "vp_pool") and select_count != kinds.count(
            "deepseek_sparse_attention"
        ):
            raise ValueError(f"{stage}: missing pooled DSA SelectKV operators: {select_count}")
        mask_input = next((value for value in graph.graph.input if value.name == "dsa_attention_mask"), None)
        if mask_input is None:
            raise ValueError(f"{stage}: missing Host DSA attention mask")
        shape = [dim.dim_value for dim in mask_input.type.tensor_type.shape.dim]
        query_length = graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value
        selected_pools = min(config["index_topk"] // 4, metadata["glm53_cache_spec"]["max_context_length"] // 4)
        if shape != [1, query_length, 1, selected_pools * 4 + 3]:
            raise ValueError(f"{stage}: invalid Host DSA attention mask shape: {shape}")
        masked_adds = [
            node for node in graph.graph.node if node.op_type == "Add" and "dsa_attention_mask" in node.input
        ]
        masked_outputs = {name for node in masked_adds for name in node.output}
        pairs = [node for node in masked_adds if any(name in masked_outputs for name in node.input)]
        if len(masked_adds) != 2 * select_count or len(pairs) != select_count:
            raise ValueError(f"{stage}: DSA mask must use two saturating MaskedAdd additions")
        if any(
            not any(attr.name == "saturate" and onnx.helper.get_attribute_value(attr) == 1 for attr in node.attribute)
            for node in masked_adds
        ):
            raise ValueError(f"{stage}: DSA MaskedAdd is not saturating")
        pair_outputs = {name for node in pairs for name in node.output}
        if (
            sum(node.op_type == "Softmax" and node.input[0] in pair_outputs for node in graph.graph.node)
            != select_count
        ):
            raise ValueError(f"{stage}: DSA MaskedAdd must directly precede Softmax")
        report[stage] = {
            "w4g64_moe_blocks": moe_count,
            "w8_linears": w8_count,
            "floating_gate_linears": gate_count,
            "w8_constant_matmuls": projection_count,
            "w8_convolutions": convolution_count,
            "mhc_sinkhorn_blocks": sinkhorn_count,
            "dsa_mask_add_pairs": len(pairs),
            "gdr_chunk_scan_blocks": chunk_count,
            "gdr_recurrent_scan_blocks": recurrent_count,
            "a16_w16_matmuls": a16_matmul_count,
            "a16_w16_chunk_scans": chunk_count,
        }
        if select_count:
            report[stage]["sparse_kv_select_blocks"] = select_count
    vision = metadata["glm53_vision"]
    for component in vision["components"]:
        graph = onnx.load(root / vision["directory"] / component["hmonnx"], load_external_data=False)
        for node in graph.graph.node:
            if node.op_type in ("Linear", "Conv2d"):
                attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                if (attrs.get("hmfp_weight_man_bit"), attrs.get("hmfp_act_man_bit")) != (8, 16):
                    raise ValueError(f"Vision component {component['name']} is not W8 A16")
    report["vision_components"] = len(vision["components"])
    return report
