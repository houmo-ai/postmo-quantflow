"""Checkpoint compatibility scoped to the explicit Qwen4 build process.

Importing this module installs nothing. The context reuses the installed shared
validator, changing only its terminal-weight predicate for Qwen4; container,
index, layer, embedding and shape checks remain the shared implementation.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
from threading import RLock


MODEL_ID = "qwen3.8-flash-next"
_LOCK = RLock()
_DEPTH = 0


def _autoround_markers(value):
    if isinstance(value, Mapping):
        if value.get("auto_round_enabled") is True:
            yield value
        for child in value.values():
            yield from _autoround_markers(child)
    elif isinstance(value, list):
        for child in value:
            yield from _autoround_markers(child)


def _producer_evidence(payload):
    markers = list(_autoround_markers(payload))
    if not markers:
        return ()
    direct_rtn = any(
        type(marker.get("iters")) is int and marker["iters"] == 0 and marker.get("disable_opt_rtn") is True
        for marker in markers
    )
    if not direct_rtn:
        return ("auto-round",)
    # Saved configs can repeat an outer engine flag without all algorithm
    # fields. That flag is not independent evidence of optimizer execution.
    optimized = any(
        type(marker.get("iters")) is int
        and (marker["iters"] > 0 or (marker["iters"] == 0 and marker.get("disable_opt_rtn") is False))
        for marker in markers
    )
    return ("rtn", "auto-round") if optimized else ("rtn",)


def _producer_metadata(path, payload):
    evidence = _producer_evidence(payload)
    if not evidence:
        return payload
    # Keep the evidence inside non-behavioral quant metadata for config.json,
    # so model-identity comparisons still see the unmodified HF configuration.
    result = dict(payload)
    quant = result.get("quantization_config")
    target = dict(quant) if path.name == "config.json" and isinstance(quant, Mapping) else result
    key = "_qwen4_producer_evidence"
    while key in target:
        key += "_"
    target[key] = [{"provider": method} for method in evidence]
    if target is not result:
        result["quantization_config"] = target
    return result


def _terminal_validator(original):
    """Adapt one verified predicate rather than copying the shared validator."""
    from xhmodel_merak.model_integrations.qwen4_exp.build import checkpoint_terminal_mismatches

    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    predicates = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Name) and node.operand.id == "has_final_norm"
    ]
    errors = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ModelError"
        and any(kw.arg == "has_final_norm" for kw in node.keywords)
    ]
    # Fail closed if upstream changes this private contract. In particular,
    # never silently adapt an already modified or structurally different check.
    function = tree.body[0]
    parameters = [arg.arg for arg in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)]
    if (parameters != ["path", "container_summaries", "index_assignments", "plan"]
            or len(predicates) != 1 or len(errors) != 1):
        raise RuntimeError("Qwen4 checkpoint patch is incompatible with the installed shared validator")

    class TerminalPredicate(ast.NodeTransformer):
        def visit_UnaryOp(self, node):
            if node is predicates[0]:
                call = ast.parse("_qwen4_terminal_check(tensor_shapes, plan)", mode="eval").body
                return ast.copy_location(call, node)
            return self.generic_visit(node)

    errors[0].keywords.append(ast.keyword(
        arg="terminal_mismatches", value=ast.parse("_qwen4_terminal_check(tensor_shapes, plan)", mode="eval").body,
    ))
    tree = ast.fix_missing_locations(TerminalPredicate().visit(tree))
    namespace = {**original.__globals__, "_qwen4_terminal_check": checkpoint_terminal_mismatches}
    exec(compile(tree, original.__code__.co_filename, "exec"), namespace)
    return namespace[original.__name__]


@contextmanager
def qwen4_exp_checkpoint_scope():
    """Install only for an explicit build; restore on success, failure or nesting."""
    from xhmodel_merak.model_build import checkpoint

    global _DEPTH
    with _LOCK:
        if _DEPTH:
            _DEPTH += 1
            try:
                yield
            finally:
                _DEPTH -= 1
            return
        reader = checkpoint._read_quant_metadata
        validator = checkpoint._verify_named_checkpoint_tensors
        method_resolver = checkpoint._checkpoint_method
        adapted = _terminal_validator(validator)

        @wraps(reader)
        def read_metadata(path, model):
            values = reader(path, model)
            if model != MODEL_ID:
                return values
            return [(metadata_path, _producer_metadata(metadata_path, payload)) for metadata_path, payload in values]

        @wraps(validator)
        def verify(path, container_summaries, index_assignments, plan):
            implementation = adapted if plan.model_id == MODEL_ID else validator
            return implementation(path, container_summaries, index_assignments, plan)

        @wraps(method_resolver)
        def resolve_method(providers, *, model, path):
            normalized = {checkpoint._normalize_quant_provider(value) for value in providers}
            if model != MODEL_ID or "rtn" not in normalized:
                return method_resolver(providers, model=model, path=path)
            if normalized & {"autoround", "auto-round", "qat"}:
                raise checkpoint.ModelError(
                    checkpoint.ErrorCode.QUANT_ARTIFACT_MISMATCH,
                    "Quant Artifact must declare one supported checkpoint producer",
                    model=model, path=str(path), providers=sorted(normalized),
                )
            # GPTQ/GPTQModel identifies the container packing, not an optimizer.
            return "rtn"

        checkpoint._read_quant_metadata = read_metadata
        checkpoint._verify_named_checkpoint_tensors = verify
        checkpoint._checkpoint_method = resolve_method
        _DEPTH = 1
        try:
            yield
        finally:
            checkpoint._read_quant_metadata = reader
            checkpoint._verify_named_checkpoint_tensors = validator
            checkpoint._checkpoint_method = method_resolver
            _DEPTH = 0


__all__ = ["qwen4_exp_checkpoint_scope"]
