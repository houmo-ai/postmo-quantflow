from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import onnx
import torch
from torch import Tensor, nn

from .assets import sha256
from .graph import FORBIDDEN_NPU_OPS, GraphArtifact, _value_contract
from .host import ATTENTION_MASK_MIN


def _export_partition(
    *,
    role: str,
    module: nn.Module,
    feed: dict[str, Tensor],
    expected: dict[str, Tensor],
    output_path: Path,
    opset: int,
    expected_lstm_nodes: int,
    npu_graph: bool,
) -> GraphArtifact:
    _validate_attention_mask_feed(feed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        tuple(feed.values()),
        output_path,
        input_names=list(feed),
        output_names=list(expected),
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    # The bucketed deployment deliberately keeps the legacy exporter output.
    # xhquant owns backend lowering; Kokoro must not mutate the ONNX graph with
    # onnxsim or model-specific post-export optimization passes.
    structural = {"post_export_optimizations": 0}
    graph = onnx.load(output_path, load_external_data=False)
    onnx.checker.check_model(graph)
    if "attention_mask" in feed:
        structural["double_mask_add_softmax_patterns"] = _validate_double_mask_add_before_softmax(graph)
    op_counts: dict[str, int] = {}
    for node in graph.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    lstm_nodes = op_counts.get("LSTM", 0)
    if lstm_nodes != expected_lstm_nodes:
        raise RuntimeError(f"{role} must preserve {expected_lstm_nodes} high-level LSTM nodes, got {lstm_nodes}")
    for node in graph.graph.node:
        if node.op_type != "LSTM":
            continue
        attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        if attributes.get("direction", b"forward") not in {b"", b"forward"}:
            raise RuntimeError(f"{role} LSTM {node.name!r} is not forward")
    if npu_graph:
        forbidden = sorted(FORBIDDEN_NPU_OPS.intersection(op_counts))
        if forbidden:
            raise RuntimeError(f"{role} contains forbidden NPU operators: {forbidden}")

    artifact = GraphArtifact(
        role=role,
        path=output_path,
        input_names=tuple(feed),
        output_names=tuple(expected),
        input_contracts=tuple(_value_contract(value) for value in graph.graph.input),
        output_contracts=tuple(_value_contract(value) for value in graph.graph.output),
        node_count=len(graph.graph.node),
        op_counts=dict(sorted(op_counts.items())),
        onnx_sha256=sha256(output_path),
        structural_rewrites=structural,
    )
    return artifact


def _validate_double_mask_add_before_softmax(
    graph: onnx.ModelProto,
    mask_name: str = "attention_mask",
) -> int:
    """Require every Softmax to consume a direct two-Add mask pattern."""

    producers = {output: node for node in graph.graph.node for output in node.output if output}
    softmax_nodes = [node for node in graph.graph.node if node.op_type == "Softmax"]
    if not softmax_nodes:
        raise RuntimeError("attention-mask graph contains no Softmax node")

    matched = 0
    for softmax in softmax_nodes:
        source = producers.get(softmax.input[0])
        if source is None or source.op_type != "Add" or mask_name not in source.input:
            raise RuntimeError(f"Softmax {softmax.name!r} is not preceded by the second mask Add")
        first_add_output = next(
            (name for name in source.input if name != mask_name),
            None,
        )
        first_add = producers.get(first_add_output or "")
        if first_add is None or first_add.op_type != "Add" or mask_name not in first_add.input:
            raise RuntimeError(f"Softmax {softmax.name!r} is not preceded by two mask Adds")
        matched += 1
    return matched


def _validate_attention_mask_feed(feed: Mapping[str, Tensor]) -> None:
    mask = feed.get("attention_mask")
    if mask is None:
        return
    if mask.ndim != 4 or mask.shape[0] != 1 or mask.shape[1] != 1 or mask.shape[2] != mask.shape[3]:
        raise ValueError(f"attention_mask must be [1,1,T,T], got {tuple(mask.shape)}")
    if not torch.isfinite(mask).all():
        raise ValueError("attention_mask must not contain NaN or infinity")
    allowed = (mask == 0) | (mask == ATTENTION_MASK_MIN)
    if not bool(allowed.all()):
        values = torch.unique(mask.detach().cpu()).tolist()
        raise ValueError(f"attention_mask values must be 0 or {ATTENTION_MASK_MIN}, got {values}")
