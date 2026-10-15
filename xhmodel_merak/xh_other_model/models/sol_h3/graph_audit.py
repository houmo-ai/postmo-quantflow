# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Audit the serialized static ABI and actual SEFP/attention precision attributes."""

from __future__ import annotations

from collections import Counter


def audit_graph(path, *, quant_type, inputs, outputs, norm_fp16=False):
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    if [v.name for v in graph.input] != list(inputs) or [v.name for v in graph.output] != list(outputs):
        raise ValueError(f"Sol-H3 graph ABI differs from metadata: {path}")
    for value in (*graph.input, *graph.output):
        tensor = value.type.tensor_type
        if tensor.elem_type not in (TensorProto.FLOAT16, TensorProto.INT32):
            raise ValueError(f"Sol-H3 graph interface must be FP16/int32: {path}: {value.name}")
        if any(not d.HasField("dim_value") or d.dim_value <= 0 for d in tensor.shape.dim):
            raise ValueError(f"Sol-H3 graph interface has a dynamic/empty dimension: {path}: {value.name}")
    if any(v.data_type in (TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.BFLOAT16) for v in graph.initializer):
        raise ValueError(f"H3 static graph contains non-FP16 floating constants: {path}")
    precisions = {"w8a8_sefp": (8, 8, b"sefp"), "w16a16_sefp": (16, 16, b"sefp"), "w4a8h0_ssfp": (4, 8, b"ssfp")}
    if quant_type not in precisions:
        raise ValueError(f"unsupported Sol-H3 component precision: {quant_type}")
    weight_bits, bits, mode = precisions[quant_type]
    counts = Counter(v.op_type for v in graph.node)
    initializers = {value.name: value for value in graph.initializer}
    vp_convolutions = []
    for node in graph.node:
        attrs = {v.name: helper.get_attribute_value(v) for v in node.attribute}
        if (
            node.op_type in ("Conv1d", "Conv2d")
            and attrs.get("groups", 1) == attrs.get("in_channels")
            and attrs.get("groups", 1) > 0
            and attrs.get("out_channels", 0) > 0
            and attrs["out_channels"] % attrs.get("groups", 1) == 0
        ):
            # The SDK VP branch consumes raw FP16 coefficients, not the SEFP
            # payload described by the legacy TE attributes on the same node.
            weight = initializers.get(node.input[3]) if len(node.input) > 3 else None
            if (
                weight is None
                or weight.data_type != TensorProto.FLOAT16
                or len(weight.dims) not in (3, 4)
                or list(weight.dims[:2]) != [attrs["out_channels"], 1]
            ):
                raise ValueError(f"Sol-H3 VP convolution requires raw FP16 VP convolution weights: {path}: {node.name}")
            vp_convolutions.append(node.name)
        if node.op_type in ("Linear", "MatMul", "Conv2d", "ConvTranspose2d"):
            if (
                attrs.get("mode") != mode
                or attrs.get("hmfp_weight_man_bit") != weight_bits
                or attrs.get("hmfp_act_man_bit") != bits
            ):
                raise ValueError(f"H3 matrix/convolution operator disagrees with {quant_type}: {path}: {node.name}")
        if node.op_type in ("FlashAttention", "SparseFlashAttention", "SparseFlashAttentionStats"):
            # Current SDK scores are FP16; legacy s_bits is ignored by runtime
            # and is no longer emitted. Q/K/V/P remain the precision contract.
            if any(attrs.get(k) != bits for k in ("q_bits", "k_bits", "v_bits", "p_bits")):
                raise ValueError(f"Sol-H3 attention precision differs from deployment precision: {path}: {node.name}")
        if node.op_type in ("SparseFlashAttention", "SparseFlashAttentionStats") and attrs.get("block_size") not in (
            64,
            128,
            256,
            512,
        ):
            raise ValueError(f"unsupported sparse attention block size: {path}: {node.name}")
        if node.op_type in ("RMSNorm", "LayerNorm"):
            half = norm_fp16 and node.op_type == "RMSNorm"
            # RMS precision follows force_fp32 in both the SDK simulator and
            # HMCC; sum_dtype is consumed by LayerNorm, not RMSNorm.
            if attrs.get("force_fp32") != int(not half) or (
                node.op_type == "LayerNorm" and attrs.get("sum_dtype") != b"float32"
            ):
                raise ValueError(f"Sol-H3 normalization precision disagrees with norm_fp16={norm_fp16}: {path}: {node.name}")
            if half and attrs.get("keep_fp32_variance") != 0:
                raise ValueError(f"Sol-H3 FP16 RMSNorm must use the FP16 variance path: {path}: {node.name}")
        if node.op_type == "RMSNorm" and 0 < attrs.get("eps", 0) < 65504.0**-2:
            raise ValueError(f"Sol-H3 RMS reciprocal can overflow on zero padding: {path}: {node.name}")
    return dict(
        input_shapes={v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in graph.input},
        output_shapes={v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in graph.output},
        operators=dict(counts),
        vp_fp16_convolutions=vp_convolutions,
        sparse_block_sizes={
            kind: sorted(
                {
                    helper.get_attribute_value(attribute)
                    for node in graph.node
                    if node.op_type == kind
                    for attribute in node.attribute
                    if attribute.name == "block_size"
                }
            )
            for kind in ("SparseFlashAttention", "SparseFlashAttentionStats")
            if counts[kind]
        },
        initializer_types=dict(Counter(v.data_type for v in graph.initializer)),
    )
