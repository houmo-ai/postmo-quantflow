"""GLM's frame component paths and hybrid state ABI, shared by build and load."""

import hashlib
import json
from pathlib import Path

from xhmodel_merak.errors import ErrorCode, ModelError


def validate_glm53_bundle(root, metadata):
    root = Path(root).resolve()
    if metadata.get("glm53_state_abi_version") != 1 or metadata.get("glm53_export_mode") != "streamed_ffn":
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "Unsupported GLM hybrid state/export ABI")
    vision = metadata.get("glm53_vision")
    if not isinstance(vision, dict) or vision.get("abi_version") != 1 or vision.get("dtype") != "float16":
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "GLM multimodal artifact requires vision ABI 1")
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
        projection_count = convolution_count = sinkhorn_count = 0
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
        if (moe_count, gate_count) != (expected_moe, expected_gates):
            raise ValueError(
                f"{stage}: missing MoE/KDA gates: {(moe_count, gate_count)} != {(expected_moe, expected_gates)}"
            )
        if (projection_count, convolution_count) != (expected_projections, expected_convolutions):
            raise ValueError(
                f"{stage}: missing DSA projections/KDA convolutions: "
                f"{(projection_count, convolution_count)} != {(expected_projections, expected_convolutions)}"
            )
        if sinkhorn_count != expected_sinkhorn:
            raise ValueError(f"{stage}: missing mHC Sinkhorn operators: {sinkhorn_count} != {expected_sinkhorn}")
        report[stage] = {
            "w4g64_moe_blocks": moe_count,
            "w8_linears": w8_count,
            "floating_gate_linears": gate_count,
            "w8_constant_matmuls": projection_count,
            "w8_convolutions": convolution_count,
            "mhc_sinkhorn_blocks": sinkhorn_count,
        }
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
