"""Qwen4-Exp registrations against xhquant's public frontend extension points.

Qwen4's PLE hash and QSA tensor programs use model-scoped mappings to the
SDK's operator modules. These mappings are installed for this export process
only; generic operator implementations remain in xhquanttool.
"""

from __future__ import annotations

import operator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor
from torch.fx import Graph, Node

import xhquant.nn as xhnn
from xhquant.frontend.fx_transform.op_normalizers.op_normalizer_impl import OP_NORMALIZER
from xhquant.quantization.xh2a.builder import NONE_QUANTED_MODULES, register_none_quanted_module
from xhquant.xhonnxruntime.base_module import BaseModule
from xhquant.xhonnxruntime.builder import HMONNX_PARSERS


_REGISTERED = False


def register_qwen4_exp_xhquant_extensions() -> None:
    """Install idempotent, process-local operator mappings needed by Qwen4."""

    global _REGISTERED
    if _REGISTERED:
        return

    _register(_binary, operator.mod, torch.remainder, Tensor.remainder, norm_op=xhnn.Mod)
    _register(_binary, operator.and_, torch.logical_and, Tensor.logical_and, norm_op=xhnn.And)
    _register(_binary, operator.eq, torch.eq, Tensor.__eq__, norm_op=xhnn.Equal)
    _register(_binary, operator.lt, torch.lt, Tensor.__lt__, norm_op=xhnn.Less)
    _register(_binary, operator.le, torch.le, Tensor.__le__, norm_op=xhnn.LessOrEqual)
    _register(_binary, operator.gt, torch.gt, Tensor.__gt__, norm_op=xhnn.Greater)
    _register(_binary, operator.ge, torch.ge, Tensor.__ge__, norm_op=xhnn.GreaterOrEqual)
    _register(_unary, torch.logical_not, Tensor.logical_not, norm_op=xhnn.Not)
    _register(_unary, torch.sin, Tensor.sin, norm_op=xhnn.Sin)
    _register(_unary, torch.cos, Tensor.cos, norm_op=xhnn.Cos)
    _register(_unary, torch.sign, Tensor.sign, norm_op=xhnn.Sign)
    _register(_where, torch.where, norm_op=xhnn.Where)
    _register(_gather, torch.gather, Tensor.gather, norm_op=xhnn.GatherElements)
    _register(_scatter, torch.scatter, Tensor.scatter, norm_op=xhnn.ScatterElements)
    _register(_clamp, torch.clamp, Tensor.clamp, torch.clip, Tensor.clip, norm_op=xhnn.Clip)
    _register(_clamp, torch.clamp_min, Tensor.clamp_min, norm_op=xhnn.Clip)
    _register(_clamp, torch.clamp_max, Tensor.clamp_max, norm_op=xhnn.Clip)
    _register(_reshape, torch.reshape, Tensor.reshape, Tensor.view, norm_op=xhnn.Reshape)
    _register(_unflatten, torch.unflatten, Tensor.unflatten, norm_op=xhnn.Reshape)
    _register(_pow, torch.pow, Tensor.pow, norm_op=xhnn.Pow)
    _register(_square, torch.square, Tensor.square, norm_op=xhnn.Pow)
    _register(_cast, Tensor.long, dtype=torch.int64)
    _register(_cast, Tensor.int, dtype=torch.int32)
    _register(_cast, Tensor.float, dtype=torch.float32)
    _register(_cast, Tensor.half, dtype=torch.float16)
    _register(_cast, Tensor.bool, dtype=torch.bool)
    _register(_cast, Tensor.to, dtype=None)

    if hasattr(xhnn, "SparseAttentionSelectKV"):
        _register(
            _selected_kv,
            torch.ops.xh.SparseAttentionSelectKV,
            torch.ops.xh.SparseAttentionSelectKV.default,
            norm_op=xhnn.SparseAttentionSelectKV,
        )
        if hasattr(torch.ops.xh.SparseAttentionSelectKV, "gqa"):
            _register(
                _selected_kv_gqa,
                torch.ops.xh.SparseAttentionSelectKV.gqa,
                norm_op=xhnn.SparseAttentionSelectKV,
            )
    _register_gdr_frontend_ops()

    if xhnn.FloorDiv not in NONE_QUANTED_MODULES:
        register_none_quanted_module(xhnn.FloorDiv)
    from .ple_sefp4 import PLE_SEFP4_OP, PLESEFP4Gather

    _register_hmonnx_parser(f"XH2a::{PLE_SEFP4_OP}", PLESEFP4Gather)
    _REGISTERED = True


@contextmanager
def qwen4_exp_xhquant_extension_scope():
    """Install Qwen4 mappings temporarily and restore every shared registry."""

    global _REGISTERED
    normalizer_snapshot = dict(OP_NORMALIZER.D)
    none_quanted_snapshot = list(NONE_QUANTED_MODULES)
    parser_snapshot = dict(HMONNX_PARSERS.module_dict)
    registered_snapshot = _REGISTERED
    try:
        _REGISTERED = False
        register_qwen4_exp_xhquant_extensions()
        yield
    finally:
        OP_NORMALIZER.D.clear()
        OP_NORMALIZER.D.update(normalizer_snapshot)
        NONE_QUANTED_MODULES[:] = none_quanted_snapshot
        HMONNX_PARSERS.module_dict.clear()
        HMONNX_PARSERS.module_dict.update(parser_snapshot)
        _REGISTERED = registered_snapshot


def _register(converter: Any, *targets: Any, **metadata: Any) -> None:
    OP_NORMALIZER.add(*targets, **metadata)(converter)


def _register_hmonnx_parser(name: str, parser: type[BaseModule]) -> None:
    registered = HMONNX_PARSERS.get(name)
    if registered is None:
        HMONNX_PARSERS.register_module(name=name, module=parser)


def _register_gdr_frontend_ops() -> None:
    mappings = (
        ("GDRChunkScan", _gdr_chunk_scan),
        ("GDRRecurrentScan", _gdr_recurrent_scan),
        ("GDRBlockTriInverse", _gdr_block_tri_inverse),
    )
    for module_name, converter in mappings:
        norm_op = getattr(xhnn, module_name, None)
        torch_op = getattr(torch.ops.xh, module_name, None)
        default_op = getattr(torch_op, "default", None)
        if norm_op is None or torch_op is None or default_op is None:
            continue
        _register(converter, torch_op, default_op, norm_op=norm_op)


def _unary(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    return norm_op(), node.args, {}


def _binary(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    return norm_op(), node.args, {}


def _where(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    return norm_op(), node.args, node.kwargs


def _gather(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    tensor = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else node.kwargs["dim"]
    indices = node.args[2] if len(node.args) > 2 else node.kwargs["index"]
    return norm_op(axis=dim), (tensor, indices), {}


def _scatter(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    tensor = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else node.kwargs["dim"]
    indices = node.args[2] if len(node.args) > 2 else node.kwargs["index"]
    updates = node.args[3] if len(node.args) > 3 else node.kwargs["src"]
    return norm_op(axis=dim, reduction="none"), (tensor, indices, updates), {}


def _clamp(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    tensor = node.args[0]
    min_value = node.kwargs.get("min")
    max_value = node.kwargs.get("max")
    target_name = getattr(node.target, "__name__", "")
    if "clamp_min" in target_name or "clamp_min" in node.name:
        min_value = node.args[1] if len(node.args) > 1 else node.kwargs.get("min")
    elif "clamp_max" in target_name or "clamp_max" in node.name:
        max_value = node.args[1] if len(node.args) > 1 else node.kwargs.get("max")
    else:
        if len(node.args) > 1:
            min_value = node.args[1]
        if len(node.args) > 2:
            max_value = node.args[2]
    return norm_op(), (tensor, min_value, max_value), {}


def _reshape(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    tensor = node.args[0]
    shape = node.kwargs.get("shape", node.args[1] if len(node.args) == 2 else node.args[1:])
    return norm_op(), (tensor, shape), {}


def _unflatten(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    example = node.meta.get("example_value")
    if example is None:
        tensor_meta = node.meta.get("tensor_meta")
        shape = tuple(tensor_meta.shape)
    else:
        shape = tuple(example.shape)
    return norm_op(), (node.args[0], shape), {}


def _pow(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    exponent = node.args[1] if len(node.args) > 1 else node.kwargs["exponent"]
    return norm_op(exponent=exponent), (node.args[0], exponent), {}


def _square(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    return norm_op(exponent=2), (node.args[0], 2), {}


def _cast(_: Any, node: Node, __: Graph, dtype: torch.dtype | None):
    resolved_dtype = dtype
    if resolved_dtype is None:
        candidates = (*node.args[1:], node.kwargs.get("dtype"))
        resolved_dtype = next((value for value in candidates if isinstance(value, torch.dtype)), None)
    if resolved_dtype is None:
        return torch.nn.Identity(), node.args[:1], {}
    return xhnn.Cast(resolved_dtype), node.args[:1], {}


__all__ = ["qwen4_exp_xhquant_extension_scope", "register_qwen4_exp_xhquant_extensions"]


def _selected_kv(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    args = node.args
    module = norm_op(
        candidate_topk=int(args[5]),
        score_scale=float(args[6]),
        pool_size=int(args[7]) if len(args) > 7 else 1,
        compression_ratio=int(args[9]) if len(args) > 9 else int(node.kwargs.get("compression_ratio", 1)),
    )
    kwargs = {"current_length": args[8] if len(args) > 8 else node.kwargs.get("current_length")}
    return module, args[:5], kwargs


def _selected_kv_gqa(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    # Newer Dynamo preserves the ".gqa" overload name. It is not a dotted
    # GraphModule path for the SDK's generated module-name resolver.
    if not hasattr(node, "module_name") and "." in node.target.__name__:
        node.module_name = node.name
    args = node.args
    module = norm_op(
        candidate_topk=int(args[6]),
        score_scale=float(args[7]),
        pool_size=int(args[8]) if len(args) > 8 else 1,
        compression_ratio=int(args[10]) if len(args) > 10 else int(node.kwargs.get("compression_ratio", 1)),
    )
    kwargs = {
        "current_length": args[9] if len(args) > 9 else node.kwargs.get("current_length"),
        "value_cache": args[3],
    }
    return module, (args[0], args[1], args[2], args[4], args[5]), kwargs


def _gdr_chunk_scan(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    args = node.args
    module = norm_op(
        num_chunks=int(args[8]),
        num_heads=int(args[9]),
        k_head_dim=int(args[10]),
        v_head_dim=int(args[11]),
        chunk_size=int(args[12]),
        matmul_input_bit=int(args[13]) if len(args) > 13 else 8,
    )
    return module, args[:8], {}


def _gdr_recurrent_scan(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    args = node.args
    module = norm_op(sequence_length=int(args[6]), output_all_states=False)
    return module, args[:6], {}


def _gdr_block_tri_inverse(_: Any, node: Node, __: Graph, norm_op: type[torch.nn.Module]):
    args = node.args
    module = norm_op(chunk_size=int(args[2]), block_size=int(args[3]))
    return module, args[:2], {}
