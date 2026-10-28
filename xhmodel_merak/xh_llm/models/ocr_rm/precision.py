# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Enforce OCR RM's SEFP h1 requirement in configuration and exported graphs."""

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path


def validate_sefp_h1_config(model_config):
    from xhquant.api.quant_type import parse_quant_format

    def walk(value, path, mode="sefp"):
        if isinstance(value, str) and value.endswith(("_sefp", "_ssfp")):
            try:
                scheme = parse_quant_format(value)
            except ValueError:
                return  # A source pathname is not a quantization schema.
            if scheme.fp_mode == "sefp" and not scheme.hidden_bit:
                raise ValueError(f"OCR RM forbids SEFP h0; use SEFP h1: {path}={value!r}")
        elif isinstance(value, Mapping):
            mode = value.get("fp_mode", mode)
            quant_type = value.get("quant_type")
            if quant_type:
                mode = parse_quant_format(quant_type).fp_mode
            if mode == "sefp" and value.get("hidden_bit") is not None and value["hidden_bit"] not in (True, 1):
                raise ValueError(f"OCR RM forbids SEFP h0; use hidden_bit=true: {path}.hidden_bit")
            for key, child in value.items():
                walk(child, f"{path}.{key}", mode)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}.{index}", mode)

    walk(model_config, "export.model")


def check_sefp_h1_graph(model, graph_name):
    """Inspect the serialized attributes, including nodes inside ONNX subgraphs."""
    import onnx

    signatures = Counter()
    hidden_attributes = 0
    fused_gdr = Counter()

    def walk(graph):
        nonlocal hidden_attributes
        for node in graph.node:
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            mode = attrs.get("mode", attrs.get("fp_mode", b"sefp"))
            if isinstance(mode, bytes):
                mode = mode.decode()
            flags = {key: value for key, value in attrs.items() if "hidden_bit" in key}
            if mode == "sefp":
                for key, value in flags.items():
                    if value != 1:
                        raise ValueError(f"OCR RM forbids SEFP h0: {graph_name}/{node.name}/{key}={value!r}")
                hidden_attributes += len(flags)
                if node.op_type in {"Linear", "MatMul", "GroupMatMul", "Conv"}:
                    required = ("hmfp_act_hidden_bit", "hmfp_weight_hidden_bit")
                    if any(key not in flags for key in required):
                        raise ValueError(f"Cannot verify SEFP h1: {graph_name}/{node.name}")
                    signatures[(node.op_type, attrs["hmfp_act_man_bit"], attrs["hmfp_weight_man_bit"])] += 1
            if node.op_type.startswith("GDR"):
                fused_gdr[node.op_type] += 1
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    walk(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for child in attr.graphs:
                        walk(child)

    walk(model.graph)
    if not hidden_attributes:
        raise ValueError(f"Cannot verify SEFP h1: {graph_name} has no serialized hidden-bit attributes")
    return {
        "hidden_attributes_checked": hidden_attributes,
        "h0_attributes": 0,
        "operand_precision": [
            {"op_type": key[0], "act_man_bit": key[1], "weight_man_bit": key[2], "hidden_bit": 1, "nodes": count}
            for key, count in signatures.items()
        ],
        "fused_gdr_ops": dict(fused_gdr),
    }


def audit_sefp_h1(meta_file):
    import onnx

    meta_file = Path(meta_file).resolve()
    root = meta_file.parent
    meta = json.loads(meta_file.read_text())
    validate_sefp_h1_config(meta["model_config"])
    paths = {"prefill": meta["prefill_hmonnx"], "decode": meta["decode_hmonnx"]}
    visual = meta["visual_config"]
    for gear in visual["gears"]:
        paths[f"visual_m{gear['image_token_capacity']}"] = gear["hmonnx"]
    graphs = {}
    for name, relative in paths.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"OCR RM graph must remain inside its export directory: {relative}")
        graphs[name] = {"path": relative, **check_sefp_h1_graph(onnx.load(path, load_external_data=False), name)}
    sdk_gdr = None
    if any(graph["fused_gdr_ops"] for graph in graphs.values()):
        from xhquant.backend.xh2a.functions.gdr_common import GDR_DEFAULT_HMFP_CONFIG

        sdk_gdr = {key: value for key, value in GDR_DEFAULT_HMFP_CONFIG.items() if "hidden_bit" in key}
        if not sdk_gdr or any(value != 1 for value in sdk_gdr.values()):
            raise ValueError(f"OCR RM fused GDR must use SEFP h1: {sdk_gdr}")
    return {"model_type": meta["model_config"]["model_type"], "passed": True, "graphs": graphs, "sdk_gdr": sdk_gdr}
