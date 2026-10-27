# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Optional LES branches in the MiniCPM decode frontend, before SDK quantization."""

from __future__ import annotations

from torch import nn

from .les import LES_REFERENCE, validate_prediction_k


def les_contract(num_layers, prediction_k, actual_k):
    if num_layers < 2:
        raise ValueError("LES requires at least two MoE body layers")
    outputs = []
    for layer in range(num_layers):
        outputs.append(dict(name=f"les_router_probs_{layer}", role="actual", layer=layer, k=actual_k))
        if layer + 1 < num_layers:
            outputs.append(
                dict(name=f"les_prediction_logits_{layer + 1}", role="prediction", layer=layer + 1, k=prediction_k)
            )
    return dict(
        schema_version=1,
        phase="single_token_decode",
        reference=LES_REFERENCE,
        prediction_k=prediction_k,
        actual_k=actual_k,
        outputs=outputs,
        routing="original_gate",
        scheduling="host_callbacks",
    )


def validate_les_contract(contract):
    if not isinstance(contract, dict):
        raise ValueError("LES metadata must be a mapping")
    outputs = contract.get("outputs", [])
    if not isinstance(outputs, list) or len(outputs) < 3 or len(outputs) % 2 != 1:
        raise ValueError("LES metadata must describe every actual and predicted router output")
    prediction_k, actual_k = contract.get("prediction_k"), contract.get("actual_k")
    if type(prediction_k) is not int or prediction_k <= 0 or type(actual_k) is not int or actual_k <= 0:
        raise ValueError("LES metadata requires positive integer Top-K values")
    expected = les_contract((len(outputs) + 1) // 2, prediction_k, actual_k)
    if contract != expected:
        raise ValueError("Invalid LES output/layer/routing contract")
    return contract


def configure_les_decode_graph(graph, prediction_k):
    """Add next-router projections without changing any existing computation.

    The projection has a separate activation observer, while its floating
    weight shares storage with the actual next-layer router. SDK owns its
    quantization and lowering. No expert weight or decoder implementation is
    duplicated. Predictions are executable before the current layer's MoE.
    """
    from xhquant.nn.modules.moeblock import MoeBlock

    gates, blocks, routing = [], [], []
    for node in graph.graph.nodes:
        if node.op != "call_module":
            continue
        block = graph.get_submodule(node.target)
        if not isinstance(block, MoeBlock):
            continue
        weights = node.args[1]
        gate = weights.args[0]
        if gate.op != "call_module" or not isinstance(graph.get_submodule(gate.target), nn.Linear):
            raise ValueError(f"LES requires the canonical Linear -> Softmax -> MoeBlock router: {node.target}")
        gates.append(gate)
        blocks.append(block)
        routing.append(weights)
    if len(gates) < 2:
        raise ValueError("LES requires at least two materialized MoE body layers")
    extras = []
    actual_k = blocks[0].k
    for index, gate in enumerate(gates):
        module = graph.get_submodule(gate.target)
        validate_prediction_k(prediction_k, module.out_features)
        if blocks[index].k != actual_k:
            raise ValueError("LES requires a consistent actual Top-K across body layers")
        # Observe the exact probabilities passed to MoeBlock, including any
        # quantized-softmax ties. Ranking unquantized logits could disagree.
        extras.append(routing[index])
        if index + 1 == len(gates):
            continue
        next_module = graph.get_submodule(gates[index + 1].target)
        projection = nn.Linear(
            next_module.in_features, next_module.out_features, bias=False, device="meta", dtype=next_module.weight.dtype
        )
        projection.weight = next_module.weight
        name = f"les_predict_layer_{index + 1}"
        graph.add_module(name, projection)
        with graph.graph.inserting_before(gate):
            predicted = graph.graph.call_module(name, (gate.args[0],))
        extras.append(predicted)
    output = next(node for node in graph.graph.nodes if node.op == "output")
    existing = output.args[0]
    if not isinstance(existing, (tuple, list)):
        existing = (existing,)
    output.args = (tuple(existing) + tuple(extras),)
    graph.graph.lint()
    graph.recompile()
    return les_contract(len(gates), prediction_k, actual_k)


class HMONNXLESScheduler:
    """Deliver LES signals during eager V2 graph execution, not after it.

    The callbacks implement device-specific expert I/O. ``require`` is called
    with all actual selected experts before that layer executes, including
    misses. This adapter never substitutes routing or changes summation order.
    Resident-weight SDK execution needs no I/O; accelerator offload backends
    must provide their own prefetch/require callbacks. CUDA graph capture is
    incompatible with Python I/O callbacks and is rejected explicitly.
    """

    def __init__(self, session, contract, *, prefetch=None, require=None):
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import DefaultGraphModuleInterpreter

        validate_les_contract(contract)
        if type(getattr(session, "interpreter", None)) is not DefaultGraphModuleInterpreter:
            raise ValueError("LES callbacks require eager HMONNXInferenceV2 (no CUDA graph capture/auto offload)")
        self.handles = []
        self.contract = contract
        self.pending = {}
        self.records = []
        self.prefetch, self.require = prefetch, require
        producers = {name: module for module in session.node_modules for name in module.output_names}
        missing = [entry["name"] for entry in contract["outputs"] if entry["name"] not in producers]
        if missing:
            raise ValueError(f"HMONNX LES outputs missing producers: {missing}")
        positions = {id(module): index for index, module in enumerate(session.node_modules)}
        moes = [module for module in session.node_modules if module.op_type == "MoeBlock"]
        actual = [entry for entry in contract["outputs"] if entry["role"] == "actual"]
        if len(moes) != len(actual):
            raise ValueError("LES requires one actual router per body MoeBlock")
        for layer, (entry, moe) in enumerate(zip(actual, moes, strict=True)):
            if moe.input_names[1] != entry["name"] or positions[id(producers[entry["name"]])] >= positions[id(moe)]:
                raise ValueError("LES actual router must be observed before its MoeBlock")
            if layer:
                predicted = producers[f"les_prediction_logits_{layer}"]
                if positions[id(predicted)] >= positions[id(moes[layer - 1])]:
                    raise ValueError("SDK reordered LES prediction after the current layer's MoeBlock")
        for entry in contract["outputs"]:
            module = producers[entry["name"]]
            self.handles.append(module.register_forward_hook(self._hook(entry, module)))

    def _hook(self, entry, module):
        def hook(_module, args, output):
            if len(module.output_names) != 1:
                output = output[module.output_names.index(entry["name"])]
            if output.numel() // output.shape[-1] != 1:
                raise ValueError("LES runtime only accepts single-token decode")
            ids = output.topk(entry["k"], dim=-1).indices.detach()
            layer = entry["layer"]
            if entry["role"] == "prediction":
                self.pending[layer] = ids
                if self.prefetch is not None:
                    self.prefetch(layer, ids.clone())
            else:
                if layer == 0:
                    self.records.clear()
                predicted = self.pending.pop(layer, None)
                if predicted is not None:
                    self.records.append((layer, predicted, ids))
                if self.require is not None:
                    self.require(layer, ids.clone())

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()
