"""Qwen4-Exp HMONNX export and Golden-runtime helpers."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.fx import GraphModule

import xhquant.nn as xhnn

from .cache_contract import Qwen4ExpCacheContract, resolve_qwen4_exp_cache_shape
from .modeling import (
    Qwen4ExpExportCapability,
    build_qwen4_exp_text_export_wrapper,
    extract_qwen4_exp_text_config,
    extract_qwen4_exp_text_model,
    inspect_qwen4_exp_export_capability,
)
from .precision import DEFAULT_PLE_QUANT_TYPE, resolve_ple_quant_type


# PyTorch 2.7 no longer honors the module-name trick used by xhquant's
# DynamoFX frontend to keep custom modules as leaf nodes.  Capturing a
# GPTQPackedLinear therefore inlines its validation path and fails on
# ``torch.equal(FakeTensor, ...)``.  Keep this workaround local to the Qwen4
# exporter: the packed modules are replaced only while Dynamo captures the
# graph, then restored as real call_module nodes before xhquant quantization.
@torch.library.custom_op("xh2_modelzoo::gptq_packed_linear_opaque", mutates_args=())
def _gptq_packed_linear_opaque(
    inputs: torch.Tensor,
    out_features: int,
    module_id: int,
) -> torch.Tensor:
    return torch.empty(
        (*inputs.shape[:-1], int(out_features)),
        dtype=inputs.dtype,
        device=inputs.device,
    )


@_gptq_packed_linear_opaque.register_fake
def _gptq_packed_linear_opaque_fake(
    inputs: torch.Tensor,
    out_features: int,
    module_id: int,
) -> torch.Tensor:
    del module_id
    return torch.empty(
        (*inputs.shape[:-1], int(out_features)),
        dtype=inputs.dtype,
        device=inputs.device,
    )


@torch.library.custom_op("xh2_modelzoo::gptq_packed_moe_opaque", mutates_args=())
def _gptq_packed_moe_opaque(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor | None,
    fast_mode: bool,
    module_id: int,
) -> torch.Tensor:
    del routing_weights, selected_experts, fast_mode, module_id
    return torch.empty_like(hidden_states)


@_gptq_packed_moe_opaque.register_fake
def _gptq_packed_moe_opaque_fake(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor | None,
    fast_mode: bool,
    module_id: int,
) -> torch.Tensor:
    del routing_weights, selected_experts, fast_mode, module_id
    return torch.empty_like(hidden_states)


_GPTQ_LINEAR_OP_TARGET = "xh2_modelzoo.gptq_packed_linear_opaque.default"
_GPTQ_MOE_OP_TARGET = "xh2_modelzoo.gptq_packed_moe_opaque.default"
_XH_DYNAMIC_SLICE_OP_TARGETS = (torch.ops.xh.DynamicSlice, torch.ops.xh.DynamicSlice.default)
_XH_LLM_CACHE_OP_TARGETS = (torch.ops.xh.LLMCache, torch.ops.xh.LLMCache.default)


def _normalize_frontend_gather_overloads(graph_module: GraphModule) -> tuple[str, ...]:
    """Reify Dynamo Gather calls for the SDK's module-based quantizer."""
    normalized = []
    for node in list(graph_module.graph.nodes):
        if node.op != "call_function" or node.target not in (torch.ops.xh.Gather, torch.ops.xh.Gather.default):
            continue
        if len(node.args) != 3 or not isinstance(node.args[1], int) or node.kwargs:
            raise RuntimeError(f"Qwen4 Gather {node.name!r} requires data, static axis, and indices")
        module_name = f"_xh2_frontend_gather_{node.name}"
        graph_module.add_module(module_name, xhnn.Gather(axis=node.args[1], normalize_negative_indices=True))
        with graph_module.graph.inserting_before(node):
            replacement = graph_module.graph.call_module(module_name, (node.args[0], node.args[2]))
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        graph_module.graph.erase_node(node)
        normalized.append(node.name)
    if normalized:
        graph_module.graph.lint()
        graph_module.recompile()
    return tuple(normalized)


def _static_int_tuple(value: Any, *, name: str) -> tuple[int, ...] | None:
    """Return a constant integer argument in tuple form, or ``None``."""

    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, (tuple, list)) and all(isinstance(item, int) for item in value):
        return tuple(int(item) for item in value)
    return None


def _graph_module_attribute(graph_module: GraphModule, target: str) -> Any:
    """Resolve a flattened or dotted ``get_attr`` target on a graph module."""

    owner_path, _, attribute_name = str(target).rpartition(".")
    owner = graph_module.get_submodule(owner_path) if owner_path else graph_module
    return getattr(owner, attribute_name)


def _normalize_frontend_rmsnorm_modules(graph_module: GraphModule) -> tuple[str, ...]:
    """Restore SDK norm leaves inlined by newer Dynamo versions."""
    normalized = []
    for node in tuple(graph_module.graph.nodes):
        if node.op != "call_function" or node.target not in (torch.ops.xh.RMSNorm, torch.ops.xh.RMSNorm.default):
            continue
        if len(node.args) != 3 or node.kwargs:
            raise ValueError(f"Qwen4 RMSNorm {node.name!r} requires input, static weight and epsilon")
        inputs, weight_node, epsilon = node.args
        if not isinstance(weight_node, torch.fx.Node) or weight_node.op != "get_attr":
            raise ValueError(f"Qwen4 RMSNorm {node.name!r} requires a static get_attr weight")
        weight = _graph_module_attribute(graph_module, str(weight_node.target))
        if not torch.is_tensor(weight) or weight.ndim != 1 or weight.numel() < 1:
            raise ValueError(f"Qwen4 RMSNorm {node.name!r} requires a one-dimensional norm weight")
        if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon) or epsilon < 0):
            raise ValueError(f"Qwen4 RMSNorm {node.name!r} requires a finite nonnegative epsilon")
        module = xhnn.RMSNorm(weight.numel(), eps=epsilon)
        del module.weight
        if isinstance(weight, torch.nn.Parameter):
            module.register_parameter("weight", weight)
        else:
            module.register_buffer("weight", weight)
        target = f"_xh2_frontend_rmsnorm_{node.name}"
        graph_module.add_module(target, module)
        node.op, node.target, node.args, node.kwargs = "call_module", target, (inputs,), {}
        normalized.append(node.name)
    if normalized:
        graph_module.graph.lint()
        graph_module.recompile()
    return tuple(normalized)


def _normalize_frontend_rope_modules(graph_module: GraphModule) -> tuple[str, ...]:
    """Keep inlined SDK rotary calls as the same fused SDK operator."""
    normalized = []
    for node in tuple(graph_module.graph.nodes):
        if node.op != "call_function" or node.target not in (torch.ops.xh.Rope, torch.ops.xh.Rope.default):
            continue
        if len(node.args) != 3 or node.kwargs:
            raise ValueError(f"Qwen4 Rope {node.name!r} requires input, cosine and sine")
        target = f"_xh2_frontend_rope_{node.name}"
        graph_module.add_module(target, xhnn.Rope())
        node.op, node.target = "call_module", target
        normalized.append(node.name)
    if normalized:
        graph_module.graph.lint()
        graph_module.recompile()
    return tuple(normalized)


def _normalize_frontend_placeholder_modules(graph_module: GraphModule) -> tuple[str, ...]:
    """Restore captured SDK placeholder boundaries before module quantization."""
    normalized = []
    for node in tuple(graph_module.graph.nodes):
        if (node.op != "call_function"
                or node.target not in (torch.ops.xh.PlaceHolder, torch.ops.xh.PlaceHolder.default)):
            continue
        if len(node.args) != 5 or node.kwargs or not isinstance(node.args[0], (tuple, list)):
            raise ValueError(f"Qwen4 placeholder {node.name!r} has an invalid flat boundary")
        inputs, shapes, count, content, dtypes = node.args
        if not isinstance(content, str) or not isinstance(count, int) or count < 1:
            raise ValueError(f"Qwen4 placeholder {node.name!r} requires static content and output count")
        module = xhnn.PlaceHolderModule(
            output_shapes=shapes, output_dtypes=dtypes, num_outputs=count, content=content,
        )
        target = f"_xh2_frontend_placeholder_{node.name}"
        graph_module.add_module(target, module)
        node.op, node.target, node.args = "call_module", target, tuple(inputs)
        normalized.append(node.name)
    if normalized:
        graph_module.graph.lint()
        graph_module.recompile()
    return tuple(normalized)


def _normalize_functional_linear_modules(graph_module: GraphModule) -> tuple[str, ...]:
    """Reify ordinary functional linears before xhquant's module-only pass.

    DynamoFX intentionally lowers ``nn.Linear`` calls to ``torch._C._nn.linear``
    with a ``get_attr`` weight.  xhquant's XH2a validator requires linear
    operations to be ``call_module`` nodes, and the ``lm_head`` is handled
    separately by :func:`sanitize_qwen4_exp_frontend_graph` so its node-level
    quantization rule remains effective.
    """

    linear_targets = {
        torch._C._nn.linear,
        torch.nn.functional.linear,
        torch.ops.aten.linear.default,
    }
    normalized: list[str] = []
    module_index = 0
    for node in list(graph_module.graph.nodes):
        if node.op != "call_function" or node.target not in linear_targets:
            continue
        if len(node.args) < 2 or not isinstance(node.args[1], torch.fx.Node):
            continue
        weight_node = node.args[1]
        if weight_node.op != "get_attr":
            continue
        weight_target = str(weight_node.target)
        # Keep the dedicated lm_head rewrite below so ``nodes_cfg.lm_head``
        # can still override its weight policy.
        if "lm_head" in weight_target:
            continue
        weight = _graph_module_attribute(graph_module, weight_target)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            continue

        bias_arg = node.args[2] if len(node.args) > 2 else None
        bias = None
        if bias_arg is not None:
            if not isinstance(bias_arg, torch.fx.Node) or bias_arg.op != "get_attr":
                continue
            bias = _graph_module_attribute(graph_module, str(bias_arg.target))
            if not torch.is_tensor(bias) or bias.ndim != 1:
                continue

        module_name = f"_xh2_frontend_linear_{module_index}_{node.name}"
        module_index += 1
        linear = xhnn.XHLinear(
            int(weight.shape[1]),
            int(weight.shape[0]),
            bias=bias is not None,
            device="meta",
            dtype=weight.dtype,
        )
        if isinstance(weight, torch.nn.Parameter):
            linear.weight = weight
        else:
            linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            if isinstance(bias, torch.nn.Parameter):
                linear.bias = bias
            else:
                linear.bias = torch.nn.Parameter(bias, requires_grad=False)
        graph_module.add_module(module_name, linear)
        with graph_module.graph.inserting_before(node):
            replacement = graph_module.graph.call_module(module_name, (node.args[0],))
        replacement.meta.update(node.meta)
        if isinstance(replacement.meta.get("scope"), dict):
            replacement.meta["scope"] = dict(replacement.meta["scope"])
            replacement.meta["scope"]["module_path"] = module_name
        node.replace_all_uses_with(replacement)
        graph_module.graph.erase_node(node)
        normalized.append(node.name)

    if normalized:
        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()
        graph_module.recompile()
    return tuple(normalized)


def _normalize_frontend_dynamic_slice_modules(graph_module: GraphModule) -> tuple[str, ...]:
    """Reify ``xh.DynamicSlice`` calls so xh2a can select ``QDynamicSlice``.

    DynamoFX lowers the leaf ``xhquant.nn.DynamicSlice`` module to its custom
    operator.  The xh2a validator only accepts supported operations as
    ``call_module`` nodes, even though its QDynamicSlice implementation is
    already registered.  Reintroducing the leaf module keeps that existing
    quantizer path intact without changing xhquant itself.
    """

    normalized: list[str] = []
    module_index = 0
    for node in list(graph_module.graph.nodes):
        if node.op != "call_function" or node.target not in _XH_DYNAMIC_SLICE_OP_TARGETS:
            continue
        if len(node.args) < 5:
            raise RuntimeError(f"Qwen4-Exp DynamicSlice node {node.name!r} has an incomplete signature")
        valid_length = _static_int_tuple(node.args[2], name="valid_length")
        axes = _static_int_tuple(node.args[3], name="axes")
        steps = _static_int_tuple(node.args[4], name="steps")
        if valid_length is None or axes is None or steps is None:
            raise RuntimeError(
                f"Qwen4-Exp DynamicSlice node {node.name!r} requires static valid_length/axes/steps, "
                f"got {node.args[2]!r}, {node.args[3]!r}, {node.args[4]!r}"
            )
        module_name = f"_xh2_frontend_dynamic_slice_{module_index}_{node.name}"
        module_index += 1
        graph_module.add_module(module_name, xhnn.DynamicSlice(list(valid_length), list(axes), list(steps)))
        with graph_module.graph.inserting_before(node):
            replacement = graph_module.graph.call_module(module_name, (node.args[0], node.args[1]))
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        graph_module.graph.erase_node(node)
        normalized.append(node.name)

    if normalized:
        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()
        graph_module.recompile()
    return tuple(normalized)


def _remove_gdr_qkv_projection_roundtrips(graph_module: GraphModule) -> tuple[str, ...]:
    """Remove only the private, ordered GDR pre-convolution split/reshape/cat identity."""

    def is_module(node, cls):
        return (isinstance(node, torch.fx.Node) and node.op == "call_module"
                and isinstance(graph_module.get_submodule(str(node.target)), cls))

    def tensor_info(node):
        value = node.meta.get("val")
        if not torch.is_tensor(value) or any(type(dim) is not int or dim < 1 for dim in value.shape):
            return None
        return tuple(value.shape), value.dtype

    removed: list[str] = []
    for concat in list(graph_module.graph.nodes):
        if not is_module(concat, xhnn.Concat) or len(concat.args) != 3 or concat.kwargs:
            continue
        branches = []
        for index, flat in enumerate(concat.args):
            if not is_module(flat, xhnn.Reshape) or len(flat.args) != 2 or flat.kwargs:
                break
            head = flat.args[0]
            if not is_module(head, xhnn.Reshape) or len(head.args) != 2 or head.kwargs:
                break
            item = head.args[0]
            if not is_module(item, xhnn.GetItem) or len(item.args) != 2 or item.kwargs or item.args[1] != index:
                break
            branches.append((item, head, flat))
        if len(branches) != 3:
            continue
        split = branches[0][0].args[0]
        if (not is_module(split, xhnn.Split) or len(split.args) != 1 or split.kwargs
                or any(item.args[0] is not split for item, _, _ in branches)):
            continue
        projection = split.args[0]
        projection_path = (projection.meta.get("qwen4_original_module_path", str(projection.target))
                           if isinstance(projection, torch.fx.Node) else "")
        if (not isinstance(projection, torch.fx.Node) or projection.op != "call_module"
                or not projection_path.endswith(("linear_attn.in_proj_qkv", "linear_attn__modules__in_proj_qkv",
                                                  "linear_attn_in_proj_qkv"))):
            continue
        source_info = tensor_info(projection)
        sections = graph_module.get_submodule(str(split.target)).split_size_or_sections
        if (source_info is None or len(source_info[0]) != 3 or tensor_info(concat) != source_info
                or not isinstance(sections, (tuple, list)) or len(sections) != 3
                or any(type(size) is not int or size < 1 for size in sections)
                or sum(sections) != source_info[0][-1]
                or any(graph_module.get_submodule(str(node.target)).dim % 3 != 2 for node in (split, concat))):
            continue
        valid_shapes = True
        for size, (item, head, flat) in zip(sections, branches, strict=True):
            lane_info = (*source_info[0][:-1], size), source_info[1]
            head_info = tensor_info(head)
            if (tensor_info(item) != lane_info or tensor_info(flat) != lane_info or head_info is None
                    or head_info[1] != source_info[1] or len(head_info[0]) != 4
                    or head_info[0][:2] != source_info[0][:2] or head_info[0][-2] * head_info[0][-1] != size):
                valid_shapes = False
                break
        private = {split, concat, *(node for branch in branches for node in branch)}
        if not valid_shapes or any(set(node.users) - private for node in private if node is not concat):
            continue
        concat.replace_all_uses_with(projection)
        for node in reversed(tuple(graph_module.graph.nodes)):
            if node in private:
                graph_module.graph.erase_node(node)
        removed.append(concat.name)
    if removed:
        graph_module.graph.lint()
        graph_module.recompile()
    return tuple(removed)


def _normalize_frontend_llm_cache_modules(graph_module: GraphModule) -> tuple[str, ...]:
    """Reify ``xh.LLMCache`` calls so xh2a can select its cache qmodule."""

    normalized: list[str] = []
    module_index = 0
    for node in list(graph_module.graph.nodes):
        if node.op != "call_function" or node.target not in _XH_LLM_CACHE_OP_TARGETS:
            continue
        if len(node.args) < 7:
            raise RuntimeError(f"Qwen4-Exp LLMCache node {node.name!r} has an incomplete signature")
        axis = node.args[4]
        attention_max_length = node.args[5]
        only_handle_old_cache = node.args[6]
        if (
            not isinstance(axis, int)
            or not isinstance(attention_max_length, int)
            or not isinstance(only_handle_old_cache, bool)
        ):
            raise RuntimeError(
                f"Qwen4-Exp LLMCache node {node.name!r} requires static axis/cache settings, "
                f"got {axis!r}, {attention_max_length!r}, {only_handle_old_cache!r}"
            )
        module_name = f"_xh2_frontend_llm_cache_{module_index}_{node.name}"
        module_index += 1
        graph_module.add_module(
            module_name,
            xhnn.LLMCacheV2(
                axis=int(axis),
                attention_max_length=int(attention_max_length),
                only_handle_old_cache=bool(only_handle_old_cache),
            ),
        )
        call_args = list(node.args[:4])
        if len(node.args) > 7 and node.args[7] is not None:
            call_args.append(node.args[7])
        with graph_module.graph.inserting_before(node):
            replacement = graph_module.graph.call_module(module_name, tuple(call_args))
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        graph_module.graph.erase_node(node)
        normalized.append(node.name)

    if normalized:
        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()
        graph_module.recompile()
    return tuple(normalized)


def _normalize_functional_conv1d_modules(graph_module: GraphModule) -> tuple[str, ...]:
    """Reify DynamoFX ``conv1d`` calls as modules understood by xh2a."""

    conv_targets = {
        torch.conv1d,
        torch.nn.functional.conv1d,
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv1d.padding,
    }
    normalized: list[str] = []
    module_index = 0
    for node in list(graph_module.graph.nodes):
        if node.op != "call_function" or node.target not in conv_targets:
            continue

        def argument(index: int, name: str, default: Any, node=node) -> Any:
            if len(node.args) > index:
                return node.args[index]
            return node.kwargs.get(name, default)

        if len(node.args) < 2 or not isinstance(node.args[1], torch.fx.Node):
            continue
        weight_node = node.args[1]
        if weight_node.op != "get_attr":
            continue
        weight = _graph_module_attribute(graph_module, str(weight_node.target))
        if not torch.is_tensor(weight) or weight.ndim != 3:
            continue

        bias_arg = argument(2, "bias", None)
        bias = None
        if bias_arg is not None:
            if not isinstance(bias_arg, torch.fx.Node) or bias_arg.op != "get_attr":
                continue
            bias = _graph_module_attribute(graph_module, str(bias_arg.target))
            if not torch.is_tensor(bias) or bias.ndim != 1:
                continue

        stride = _static_int_tuple(argument(3, "stride", 1), name="stride")
        padding = argument(4, "padding", 0)
        dilation = _static_int_tuple(argument(5, "dilation", 1), name="dilation")
        groups = argument(6, "groups", 1)
        if stride is None or dilation is None or not isinstance(groups, int):
            continue
        if len(stride) != 1 or len(dilation) != 1:
            continue
        if not isinstance(padding, (int, str, tuple, list)):
            continue
        if isinstance(padding, (tuple, list)) and not all(isinstance(item, int) for item in padding):
            continue
        padding_value: int | str | tuple[int, ...]
        if isinstance(padding, (tuple, list)):
            padding_value = tuple(int(item) for item in padding)
        else:
            padding_value = padding

        module_name = f"_xh2_frontend_conv1d_{module_index}_{node.name}"
        module_index += 1
        conv = torch.nn.Conv1d(
            in_channels=int(weight.shape[1]) * int(groups),
            out_channels=int(weight.shape[0]),
            kernel_size=int(weight.shape[2]),
            stride=int(stride[0]),
            padding=padding_value,
            dilation=int(dilation[0]),
            groups=int(groups),
            bias=bias is not None,
            device="meta",
            dtype=weight.dtype,
        )
        conv.weight = (
            weight if isinstance(weight, torch.nn.Parameter) else torch.nn.Parameter(weight, requires_grad=False)
        )
        if bias is not None:
            conv.bias = bias if isinstance(bias, torch.nn.Parameter) else torch.nn.Parameter(bias, requires_grad=False)
        graph_module.add_module(module_name, conv)
        with graph_module.graph.inserting_before(node):
            replacement = graph_module.graph.call_module(module_name, (node.args[0],))
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        graph_module.graph.erase_node(node)
        normalized.append(node.name)

    if normalized:
        graph_module.graph.eliminate_dead_code()
        graph_module.graph.lint()
        graph_module.recompile()
    return tuple(normalized)


class _DynamoOpaqueGPTQLinear(torch.nn.Module):
    def __init__(self, *, out_features: int, module_id: int, dtype_anchor: torch.Tensor | None) -> None:
        super().__init__()
        self.out_features = int(out_features)
        self.module_id = int(module_id)
        if torch.is_tensor(dtype_anchor):
            # Qwen4's shared GatedDeltaNet reads ``weight.dtype`` while
            # building its output.  Retain only that scalar metadata; the
            # actual packed module is reattached after capture.
            self.weight = dtype_anchor.new_empty(())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return _gptq_packed_linear_opaque(inputs, self.out_features, self.module_id)


class _DynamoOpaqueGPTQMoe(torch.nn.Module):
    def __init__(self, *, module_id: int, dtype_anchor: torch.Tensor | None) -> None:
        super().__init__()
        self.module_id = int(module_id)
        if torch.is_tensor(dtype_anchor):
            self.expert_gate_proj_weight = dtype_anchor.new_empty(())

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor | None = None,
        fast_mode: bool = True,
    ) -> torch.Tensor:
        return _gptq_packed_moe_opaque(
            hidden_states,
            routing_weights,
            selected_experts,
            fast_mode,
            self.module_id,
        )


@torch._dynamo.assume_constant_result
def _serialize_export_pragma(metadata: Mapping[str, Any]) -> str:
    return json.dumps(metadata)


def _dynamo_export_pragma(_: torch.Tensor, metadata: Mapping[str, Any]) -> None:
    torch.ops.xh._pragma.default(_serialize_export_pragma(metadata))


@contextmanager
def _dynamo_export_compatibility_scope():
    """Hide CUDA's eager-only stream-capture query from full-graph export."""

    from ..qwen3_5 import _delta_rule

    original_is_current_stream_capturing = torch.cuda.is_current_stream_capturing
    original_pragma = _delta_rule.xh_pragma_fx
    _delta_rule.xh_pragma_fx = _dynamo_export_pragma
    torch.cuda.is_current_stream_capturing = lambda: False
    try:
        yield
    finally:
        _delta_rule.xh_pragma_fx = original_pragma
        torch.cuda.is_current_stream_capturing = original_is_current_stream_capturing


@contextmanager
def _opaque_gptq_modules_for_dynamo(model: torch.nn.Module):
    """Hide packed GPTQ modules during Dynamo capture without touching xhquant."""

    from xhquant.nn import GPTQPackedLinear, GPTQPackedMoeBlock, MoeBlock

    selected_paths: list[str] = []
    replacements: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []
    originals: dict[int, torch.nn.Module] = {}
    module_kinds: dict[int, str] = {}
    next_module_id = 0

    for path, module in tuple(model.named_modules()):
        if not path or any(path == parent or path.startswith(parent + ".") for parent in selected_paths):
            continue
        if isinstance(module, (GPTQPackedMoeBlock, MoeBlock)):
            dtype_anchor = getattr(module, "expert_gate_proj_weight", None)
            if not torch.is_tensor(dtype_anchor):
                packed_experts = getattr(module, "expert_gate_proj_packed", None)
                first_packed = packed_experts[0] if packed_experts else None
                dtype_anchor = getattr(first_packed, "scales", None)
            replacement = _DynamoOpaqueGPTQMoe(
                module_id=next_module_id,
                dtype_anchor=dtype_anchor,
            )
            kind = "moe"
        elif isinstance(module, GPTQPackedLinear):
            packed_weight = getattr(module, "packed_weight", None)
            dtype_anchor = getattr(packed_weight, "scales", None)
            replacement = _DynamoOpaqueGPTQLinear(
                out_features=module.out_features,
                module_id=next_module_id,
                dtype_anchor=dtype_anchor,
            )
            kind = "linear"
        else:
            continue

        parent_path, _, child_name = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, child_name, replacement)
        replacements.append((parent, child_name, module))
        selected_paths.append(path)
        originals[next_module_id] = module
        module_kinds[next_module_id] = kind
        next_module_id += 1

    try:
        yield originals, module_kinds
    finally:
        for parent, child_name, original in reversed(replacements):
            setattr(parent, child_name, original)


def _restore_gptq_modules_after_dynamo(
    graph_module: GraphModule,
    originals: Mapping[int, torch.nn.Module],
    module_kinds: Mapping[int, str],
    original_module_paths: Mapping[int, str] | None = None,
) -> tuple[str, ...]:
    """Turn exporter-only opaque calls back into xhquant-recognized modules."""

    restored: list[str] = []
    targets_by_id: dict[int, str] = {}
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if target == _GPTQ_LINEAR_OP_TARGET:
            if len(node.args) != 3 or not isinstance(node.args[2], int):
                raise RuntimeError(f"Malformed opaque GPTQ linear node: {node.format_node()}")
            module_id = int(node.args[2])
            expected_kind = "linear"
            new_args = (node.args[0],)
        elif target == _GPTQ_MOE_OP_TARGET:
            if len(node.args) != 5 or not isinstance(node.args[4], int):
                raise RuntimeError(f"Malformed opaque GPTQ MoE node: {node.format_node()}")
            module_id = int(node.args[4])
            expected_kind = "moe"
            new_args = tuple(node.args[:4])
        else:
            continue

        original = originals.get(module_id)
        if original is None or module_kinds.get(module_id) != expected_kind:
            raise RuntimeError(f"Opaque GPTQ node references unknown module id {module_id}")
        if original_module_paths is not None and id(original) in original_module_paths:
            node.meta["qwen4_original_module_path"] = original_module_paths[id(original)]
        module_name = targets_by_id.get(module_id)
        if module_name is None:
            import copy

            module_name = f"_xh2_gptq_packed_{module_id}"
            # XH2 quantization changes module classes and releases packed
            # parameters in place. Keep the source reusable for another phase.
            graph_module.add_submodule(module_name, copy.deepcopy(original))
            targets_by_id[module_id] = module_name
            restored.append(module_name)
        node.op = "call_module"
        node.target = module_name
        node.args = new_args
        node.kwargs = {}
        scope = node.meta.get("scope")
        if isinstance(scope, dict):
            scope["module_path"] = module_name

    if restored:
        graph_module.graph.lint()
        graph_module.recompile()
    return tuple(restored)


@dataclass(frozen=True)
class Qwen4ExpHMONNXExportResult:
    """Artifact paths and adapter evidence produced by an export."""

    onnx_path: str
    hmonnx_path: str | None
    preflight_path: str
    io_manifest_path: str
    capability: dict[str, Any]
    converted_gptq_modules: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preflight_qwen4_exp_hmonnx_export(
    config_or_model: Any,
    *,
    require_hmonnx_runtime: bool = True,
) -> Qwen4ExpExportCapability:
    return inspect_qwen4_exp_export_capability(
        config_or_model,
        model=config_or_model if isinstance(config_or_model, torch.nn.Module) else None,
        require_hmonnx_runtime=require_hmonnx_runtime,
    )


def _build_qwen4_exp_gptq_export_device_map(config: Any) -> str | dict[str, str]:
    """Keep the oversized PLE table on CPU while sharding decoder layers.

    Qwen4-Exp's ngram embedding is roughly 95 GiB in FP16 and cannot fit on a
    single RTX 6000D.  GPTQModel's generic ``device_map="auto"`` assigns whole
    decoder layers round-robin, so the layer containing PLE otherwise OOMs.
    The native PLE implementation supports a CPU-resident table and transfers
    only the selected rows back to the decoder device.
    """

    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        return "auto"
    text_config = getattr(config, "text_config", config)
    num_hidden_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)
    if num_hidden_layers <= 0:
        return "auto"

    device_count = torch.cuda.device_count()
    last_device = f"cuda:{device_count - 1}"
    device_map: dict[str, str] = {"": "cpu"}
    device_map["model.language_model.embed_tokens"] = "cuda:0"
    ple_layer_indices = {
        int(ple_layer_id) - 1
        for ple_layer_id in (getattr(text_config, "ple_layer_ids", ()) or ())
        if 0 <= int(ple_layer_id) - 1 < num_hidden_layers
    }
    for layer_idx in range(num_hidden_layers):
        layer_device = "cpu" if layer_idx in ple_layer_indices else f"cuda:{layer_idx % device_count}"
        device_map[f"model.language_model.layers.{layer_idx}"] = layer_device
    device_map["model.language_model.hyper_connection_mixer"] = last_device
    device_map["model.language_model.rotary_emb"] = last_device
    device_map["model.visual"] = last_device
    device_map["lm_head"] = last_device
    return device_map


def _restore_qwen4_exp_ple_layer_devices(model: torch.nn.Module, config: Any) -> None:
    """Move PLE layers back to their GPU while leaving the huge table on CPU."""

    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        return
    text_config = getattr(config, "text_config", config)
    text_model = extract_qwen4_exp_text_model(model)
    for ple_layer_id in getattr(text_config, "ple_layer_ids", ()) or ():
        layer_idx = int(ple_layer_id) - 1
        if not 0 <= layer_idx < len(text_model.layers):
            continue
        layer = text_model.layers[layer_idx]
        ple = getattr(layer, "ple", None)
        ngram_owner = getattr(ple, "ple_embedding", None)
        ngram_embedding = getattr(ngram_owner, "ngram_embedding", None)
        if ngram_embedding is None:
            continue
        target_device = torch.device(f"cuda:{layer_idx % torch.cuda.device_count()}")
        ngram_owner.ngram_embedding = None
        try:
            layer.to(target_device)
        finally:
            ngram_owner.ngram_embedding = ngram_embedding


def load_qwen4_exp_model_for_export(
    model_dir: str | Path,
    *,
    device: str = "cpu",
    device_map: str | Mapping[str, Any] | None = None,
    dtype: torch.dtype = torch.float16,
) -> torch.nn.Module:
    """Load either a GPTQModel HF artifact or the original Qwen4-Exp model."""

    from transformers import AutoConfig

    model_dir = str(Path(model_dir).expanduser().resolve())
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    quantization_config = getattr(config, "quantization_config", None)
    if isinstance(quantization_config, Mapping):
        quant_method = quantization_config.get("quant_method")
    else:
        quant_method = getattr(quantization_config, "quant_method", None)
    quant_method = str(getattr(quant_method, "value", quant_method) or "").lower()
    if quant_method == "gptq":
        from ...base_model import XHBaseModel

        resolved_device_map = device_map or device
        if device_map == "auto" and str(getattr(config, "model_type", "")) == "qwen4_exp":
            resolved_device_map = _build_qwen4_exp_gptq_export_device_map(config)
        model = XHBaseModel._load_gptqmodel(
            model_dir,
            device_map=resolved_device_map,
            dtype=dtype,
            trust_remote_code=False,
            offload_to_disk=False,
        ).eval()
        if isinstance(resolved_device_map, Mapping):
            _restore_qwen4_exp_ple_layer_devices(model, config)
        return model

    try:
        from transformers import AutoModelForImageTextToText

        auto_model = AutoModelForImageTextToText
    except ImportError:  # pragma: no cover - older Transformers fallback.
        from transformers import AutoModelForCausalLM

        auto_model = AutoModelForCausalLM
    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": False,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    model = auto_model.from_pretrained(model_dir, **load_kwargs).eval()
    if device_map is None:
        model.to(device)
    return model


def _sample_mrope_position_inputs(*, batch_size: int, input_length: int, past_length: int, device):
    current = torch.arange(past_length, past_length + input_length, dtype=torch.int32, device=device)
    first_group = past_length // 4
    groups = torch.arange(first_group, first_group + (input_length + 3) // 4, dtype=torch.int32, device=device) * 4
    current = current.view(1, -1).expand(batch_size, -1)
    groups = groups.view(1, -1).expand(batch_size, -1)
    return (*(current.clone() for _ in range(3)), *(groups.clone() for _ in range(3)))


@torch.no_grad()
def build_qwen4_exp_decode_sample_inputs(
    native_model: torch.nn.Module,
    *,
    text_config: Any | None = None,
    prefill_ids: torch.Tensor | None = None,
    decode_ids: torch.Tensor | None = None,
    context_length: int | None = None,
) -> tuple[torch.nn.Module, Any, tuple[torch.Tensor, ...]]:
    """Build fixed-capacity decode inputs, retaining real prompt state.

    With a prompt, execute the same explicit-state wrapper used by the decoder.
    If decode_ids are absent, its final logits also select the first token.
    Without a prompt, retain the zero-state sample used for export smoke tests.
    """

    from xhquant.core.cache_tensor import CacheTensor

    text_model = extract_qwen4_exp_text_model(native_model).eval()
    resolved_config = extract_qwen4_exp_text_config(text_config or text_model)
    embedding = text_model.embed_tokens
    embedding_device = embedding.weight.device
    infer_first_token = decode_ids is None and prefill_ids is not None
    decode_ids = torch.tensor([[4]], device=embedding_device) if decode_ids is None else decode_ids.to(embedding_device)
    if decode_ids.ndim != 2:
        raise ValueError("Qwen4-Exp decode_ids must be rank-2 [batch, sequence]")
    batch_size = int(decode_ids.shape[0])
    if prefill_ids is not None and (prefill_ids.ndim != 2 or prefill_ids.shape[0] != batch_size
                                    or not prefill_ids.shape[1]):
        raise ValueError("Qwen4-Exp prefill_ids must be a nonempty rank-2 tensor with the decode batch size")
    past_length = int(prefill_ids.shape[1]) if prefill_ids is not None else 0
    capacity = int(context_length) if context_length is not None else max(past_length + int(decode_ids.shape[1]), 1)
    if capacity < 1:
        raise ValueError("Qwen4-Exp context length must be positive")
    if past_length + int(decode_ids.shape[1]) > capacity:
        raise ValueError("Qwen4-Exp sample exceeds the fixed context length")
    flattened_cache = _build_qwen4_exp_zero_cache_inputs(
        text_model,
        resolved_config,
        batch_size=batch_size,
        context_length=capacity,
    )
    contract = Qwen4ExpCacheContract.from_text_config(resolved_config)
    # SDK K/V writers mutate caller-owned storage. Keep full-capacity inputs
    # rather than relying on the possibly shortened cache views they return.
    flattened_cache = tuple(
        CacheTensor(value) if spec.state_kind in {"kv_key", "kv_value", "qsa_indexer_key"} else value
        for spec, value in zip(contract.tensors, flattened_cache, strict=True)
    )
    if prefill_ids is not None:
        prefill_ids = prefill_ids.to(embedding_device)
        wrapper = build_qwen4_exp_text_export_wrapper(
            native_model,
            text_config=resolved_config,
            batch_size=batch_size,
            input_sequence_length=past_length,
            cache_length=capacity,
        ).eval()
        from .modeling import Qwen4ExpExportableAttention
        from .qsa_host_controls import build_qsa_host_controls

        qsa = next((module for module in wrapper.modules() if isinstance(module, Qwen4ExpExportableAttention)), None)
        controls = () if qsa is None else build_qsa_host_controls(
            0, past_length, past_length, capacity, qsa.packed_qsa.candidate_topk,
            device=embedding_device, dtype=embedding.weight.dtype,
        )
        outputs = wrapper(
            embedding(prefill_ids),
            torch.zeros((1,), dtype=torch.int32, device=embedding_device),
            torch.tensor([past_length], dtype=torch.int32, device=embedding_device),
            prefill_ids.to(torch.int32),
            *_sample_mrope_position_inputs(
                batch_size=batch_size, input_length=past_length, past_length=0, device=embedding_device,
            ),
            *flattened_cache,
            *controls,
        )
        if infer_first_token:
            if wrapper.lm_head is None:
                raise ValueError("Qwen4-Exp prompt token selection requires an lm_head")
            decode_ids = outputs[0][:, -1, :].argmax(dim=-1, keepdim=True).to(embedding_device)
        warmed_cache = []
        for spec, original, updated in zip(contract.tensors, flattened_cache, outputs[1:], strict=True):
            inplace = spec.state_kind in {"kv_key", "kv_value", "qsa_indexer_key"}
            value = (original if inplace else updated).detach().clone()
            warmed_cache.append(CacheTensor(value) if inplace else value)
        flattened_cache = tuple(warmed_cache)
        contract.validate_cache_shapes(dict(zip(contract.expected_keys(), flattened_cache, strict=True)))
    sample_inputs = (
        text_model.embed_tokens(decode_ids),
        torch.tensor([past_length], dtype=torch.int32, device=embedding_device),
        torch.tensor([decode_ids.shape[1]], dtype=torch.int32, device=embedding_device),
        decode_ids.to(torch.int32),
        *_sample_mrope_position_inputs(
            batch_size=batch_size, input_length=int(decode_ids.shape[1]),
            past_length=past_length, device=embedding_device,
        ),
        *flattened_cache,
    )
    return native_model, resolved_config, sample_inputs


@torch.no_grad()
def build_qwen4_exp_serving_sample_inputs(
    native_model: torch.nn.Module,
    *,
    input_sequence_length: int,
    context_length: int,
    text_config: Any | None = None,
) -> tuple[torch.nn.Module, Any, tuple[torch.Tensor, ...]]:
    """Build one static prefill/decode sample for the serving exporter."""

    text_model = extract_qwen4_exp_text_model(native_model).eval()
    resolved_config = extract_qwen4_exp_text_config(text_config or text_model)
    embedding = text_model.embed_tokens
    device = embedding.weight.device
    batch_size = 1
    token_ids = torch.zeros(
        (batch_size, int(input_sequence_length)),
        dtype=torch.long,
        device=device,
    )
    cache_inputs = _build_qwen4_exp_zero_cache_inputs(
        text_model,
        resolved_config,
        batch_size=batch_size,
        context_length=int(context_length),
    )
    return (
        native_model,
        resolved_config,
        (
            embedding(token_ids),
            torch.zeros((1,), dtype=torch.int32, device=device),
            torch.full((1,), int(input_sequence_length), dtype=torch.int32, device=device),
            token_ids.to(torch.int32),
            *_sample_mrope_position_inputs(
                batch_size=batch_size, input_length=int(input_sequence_length), past_length=0, device=device,
            ),
            *cache_inputs,
        ),
    )


def _isolate_frontend_module_ownership(graph: GraphModule) -> GraphModule:
    """Give PTQ private module/parameter wrappers while borrowing tensor storage.

    SDK quantization changes module classes and releases their source weights to
    meta. Distinct wrappers keep those changes and device transfers local to the
    export, without copying the enormous native PLE table or packed projections.
    The memo retains tied-parameter and shared-buffer aliases within the graph.
    """
    import copy

    memo = {}
    for value in (*graph.parameters(), *graph.buffers()):
        if id(value) not in memo:
            memo[id(value)] = (
                torch.nn.Parameter(value.detach(), requires_grad=value.requires_grad)
                if isinstance(value, torch.nn.Parameter) else value.detach()
            )
    # FrontendGraph deliberately forbids deepcopy of its graph ownership.
    # Its root already belongs to this export; isolate only borrowed modules
    # and tensor registrations, retaining the SDK's graph metadata untouched.
    graph._modules = {name: copy.deepcopy(module, memo) for name, module in graph._modules.items()}
    graph._parameters = {name: memo.get(id(value), value) for name, value in graph._parameters.items()}
    graph._buffers = {name: memo.get(id(value), value) for name, value in graph._buffers.items()}
    return graph


def _module_device(module: torch.nn.Module, fallback: torch.device) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return buffer.device if buffer is not None else fallback


def _normalize_quantized_moe_call_signatures(graph_module: GraphModule) -> tuple[str, ...]:
    """Pass traced MoE's legacy ``fast_mode`` argument by keyword.

    ``GPTQPackedMoeBlock`` accepts ``fast_mode`` positionally, while its xh2a
    quantized subclass exposes the shared ``QMoeBlock`` signature with
    ``**kwargs``.  Dynamo's exporter rejects the positional boolean before it
    can reach that keyword sink, so normalize only these generated call sites
    after PTQ and before the export graph is captured.
    """

    normalized: list[str] = []
    for node in graph_module.graph.nodes:
        if node.op != "call_module" or len(node.args) < 4:
            continue
        module = graph_module.get_submodule(str(node.target))
        if not any(kind in type(module).__name__ for kind in ("QGPTQPackedMoeBlock", "QMoeBlock")):
            continue
        fast_mode = node.args[3]
        node.args = tuple(node.args[:3])
        kwargs = dict(node.kwargs)
        kwargs.setdefault("fast_mode", fast_mode)
        node.kwargs = kwargs
        normalized.append(node.name)
    if normalized:
        graph_module.graph.lint()
        graph_module.recompile()
    return tuple(normalized)


def _build_qwen4_exp_zero_cache_inputs(
    text_model: torch.nn.Module,
    text_config: Any,
    *,
    batch_size: int,
    context_length: int,
) -> tuple[torch.Tensor, ...]:
    contract = Qwen4ExpCacheContract.from_text_config(text_config)
    fallback_device = text_model.embed_tokens.weight.device
    fallback_dtype = text_model.embed_tokens.weight.dtype
    values: list[torch.Tensor] = []
    for spec in contract.tensors:
        shape = resolve_qwen4_exp_cache_shape(spec, batch_size=batch_size, context_length=context_length)
        if spec.layer_idx >= 0:
            device = _module_device(text_model.layers[spec.layer_idx], fallback_device)
        else:
            device = fallback_device
        dtype = fallback_dtype if spec.dtype == "floating" else torch.int32
        if spec.state_kind == "ple_ngram_history":
            eos_token_id = getattr(text_config, "eos_token_id", 0)
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0]
            values.append(torch.full(shape, int(eos_token_id), dtype=dtype, device=device))
        else:
            values.append(torch.zeros(shape, dtype=dtype, device=device))
    return tuple(values)


def sanitize_qwen4_exp_frontend_graph(frontend_graph: GraphModule) -> tuple[str, ...]:
    """Remove trace-only state mutations that do not belong in the decoder ABI."""

    removed: list[str] = []
    for node in list(frontend_graph.graph.nodes):
        if node.op == "call_function" and node.target in {
            torch._C._nn.linear,
            torch.nn.functional.linear,
        }:
            if len(node.args) < 2 or not isinstance(node.args[1], torch.fx.Node):
                continue
            weight_node = node.args[1]
            if weight_node.op != "get_attr" or "lm_head" not in str(weight_node.target):
                continue
            weight_owner_path, _, weight_name = str(weight_node.target).rpartition(".")
            weight_owner = frontend_graph.get_submodule(weight_owner_path)
            weight = getattr(weight_owner, weight_name)
            if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2:
                raise RuntimeError("Qwen4-Exp functional lm_head weight must be a rank-2 Parameter")

            bias = None
            bias_arg = node.args[2] if len(node.args) > 2 else None
            if bias_arg is not None:
                if not isinstance(bias_arg, torch.fx.Node) or bias_arg.op != "get_attr":
                    raise RuntimeError("Qwen4-Exp functional lm_head bias must be a Parameter or None")
                bias_owner_path, _, bias_name = str(bias_arg.target).rpartition(".")
                bias = getattr(frontend_graph.get_submodule(bias_owner_path), bias_name)
                if not isinstance(bias, torch.nn.Parameter):
                    raise RuntimeError("Qwen4-Exp functional lm_head bias must be a Parameter")

            module_name = "lm_head"
            existing_lm_head = getattr(frontend_graph, module_name, None)
            if isinstance(existing_lm_head, torch.nn.Module):
                linear = existing_lm_head
                linear.__class__ = xhnn.XHLinear
            else:
                linear = xhnn.XHLinear(
                    int(weight.shape[1]),
                    int(weight.shape[0]),
                    bias=bias is not None,
                    device="meta",
                    dtype=weight.dtype,
                )
                frontend_graph.add_module(module_name, linear)
            linear.weight = weight
            linear.bias = bias
            with frontend_graph.graph.inserting_before(node):
                replacement = frontend_graph.graph.call_module(module_name, (node.args[0],))
            replacement.meta.update(node.meta)
            node.replace_all_uses_with(replacement)
            removed.append(node.name)
            frontend_graph.graph.erase_node(node)
            continue
        scalar_tensor_ops = {"Where": (1, 2), "Mod": (0, 1)}
        module_type = type(frontend_graph.get_submodule(node.target)).__name__ if node.op == "call_module" else None
        if module_type in scalar_tensor_ops:
            args = list(node.args)
            for argument_index in scalar_tensor_ops[module_type]:
                argument = args[argument_index]
                if isinstance(argument, torch.fx.Node) or not isinstance(argument, (bool, int, float)):
                    continue
                buffer_name = f"_qwen4_exp_{node.name}_scalar_{argument_index}"
                dtype = None
                if module_type == "Mod":
                    for operand in args:
                        if isinstance(operand, torch.fx.Node):
                            dtype = _first_tensor_dtype(operand.meta.get("example_value")) or _first_tensor_dtype(
                                operand.meta.get("val")
                            )
                            if dtype is not None:
                                break
                frontend_graph.register_buffer(buffer_name, torch.tensor(argument, dtype=dtype), persistent=False)
                with frontend_graph.graph.inserting_before(node):
                    args[argument_index] = frontend_graph.graph.get_attr(buffer_name)
            node.args = tuple(args)
        if node.op == "call_function" and node.target is torch._C._set_grad_enabled:
            if node.users:
                raise RuntimeError(f"Qwen4-Exp grad-state node {node.name!r} unexpectedly has users")
            removed.append(node.name)
            frontend_graph.graph.erase_node(node)
            continue
        if node.op == "call_method" and node.target == "copy_":
            if len(node.args) < 2 or not isinstance(node.args[1], torch.fx.Node):
                raise RuntimeError(f"Qwen4-Exp cache copy node {node.name!r} has an unsupported signature")
            node.replace_all_uses_with(node.args[1])
            removed.append(node.name)
            frontend_graph.graph.erase_node(node)
    frontend_graph.graph.lint()
    frontend_graph.recompile()
    return tuple(removed)


def _first_tensor_device(value: Any) -> torch.device | None:
    if torch.is_tensor(value):
        return value.device
    if isinstance(value, (tuple, list)):
        for item in value:
            device = _first_tensor_device(item)
            if device is not None:
                return device
    if isinstance(value, Mapping):
        for item in value.values():
            device = _first_tensor_device(item)
            if device is not None:
                return device
    return None


def _node_tensor_device(node: torch.fx.Node) -> torch.device | None:
    for metadata_name in ("val", "example_value"):
        device = _first_tensor_device(node.meta.get(metadata_name))
        if device is not None and device.type != "meta":
            return device
    return None


def _first_tensor_dtype(value: Any) -> torch.dtype | None:
    if torch.is_tensor(value):
        return value.dtype
    if hasattr(value, "dtype") and isinstance(value.dtype, torch.dtype):
        return value.dtype
    if isinstance(value, (tuple, list)):
        for item in value:
            dtype = _first_tensor_dtype(item)
            if dtype is not None:
                return dtype
    if isinstance(value, Mapping):
        for item in value.values():
            dtype = _first_tensor_dtype(item)
            if dtype is not None:
                return dtype
    return None


def _node_tensor_dtype(node: torch.fx.Node) -> torch.dtype | None:
    for metadata_name in ("val", "example_value", "tensor_meta"):
        dtype = _first_tensor_dtype(node.meta.get(metadata_name))
        if dtype is not None:
            return dtype
    return None


def _remove_device_only_transfer_nodes(graph_module: GraphModule) -> tuple[str, ...]:
    """Erase trace-time device copies while preserving real dtype casts."""

    removed: list[str] = []
    for node in list(graph_module.graph.nodes):
        if not node.args or not isinstance(node.args[0], torch.fx.Node):
            continue
        is_transfer = node.op == "call_method" and node.target in {"to", "cpu", "cuda"}
        if node.op == "call_module":
            is_transfer = type(graph_module.get_submodule(node.target)).__name__ in {"To", "CPU", "CUDA"}
        if not is_transfer:
            continue
        source = node.args[0]
        if _node_tensor_dtype(source) != _node_tensor_dtype(node):
            continue
        node.replace_all_uses_with(source)
        graph_module.graph.erase_node(node)
        removed.append(node.name)
    graph_module.graph.eliminate_dead_code()
    graph_module.graph.lint()
    graph_module.recompile()
    return tuple(removed)


def _remove_export_identity_nodes(graph_module: GraphModule) -> tuple[str, ...]:
    """Remove SDK export identities before develop's ONNX identity cleanup.

    QIdentity exports an HMIdentity without quantization or device movement.
    The stock ONNX pass can otherwise delete unrelated tail nodes when it
    encounters these identities. Real casts and graph input/output order stay
    untouched.
    """
    from xhquant.quantization.xh2a.qmodules.identity import QIdentity

    removed: list[str] = []
    for node in list(graph_module.graph.nodes):
        is_identity = node.op == "call_function" and node.target == torch.ops.xh2a.HMIdentity.default
        if node.op == "call_module":
            is_identity = isinstance(graph_module.get_submodule(node.target), QIdentity)
        if not is_identity:
            continue
        if len(node.args) != 1 or node.kwargs or not isinstance(node.args[0], torch.fx.Node):
            raise RuntimeError(f"Unexpected QIdentity signature for {node.name!r}")
        node.replace_all_uses_with(node.args[0])
        graph_module.graph.erase_node(node)
        removed.append(node.name)
    if removed:
        graph_module.graph.lint()
        graph_module.recompile()
    return tuple(removed)


def _argument_node_devices(value: Any, *, exclude: torch.fx.Node) -> set[torch.device]:
    if isinstance(value, torch.fx.Node):
        device = None if value is exclude else _node_tensor_device(value)
        return set() if device is None else {device}
    if isinstance(value, (tuple, list)):
        devices: set[torch.device] = set()
        for item in value:
            devices.update(_argument_node_devices(item, exclude=exclude))
        return devices
    if isinstance(value, Mapping):
        devices = set()
        for item in value.values():
            devices.update(_argument_node_devices(item, exclude=exclude))
        return devices
    return set()


def _consumer_peer_devices(source: torch.fx.Node) -> set[torch.device]:
    devices: set[torch.device] = set()
    frontier = [source]
    visited = {source}
    fallback_devices: set[torch.device] = set()
    while frontier:
        current = frontier.pop()
        for user in current.users:
            peer_devices = _argument_node_devices((user.args, user.kwargs), exclude=current)
            if peer_devices:
                devices.update(peer_devices)
                continue
            if user not in visited:
                visited.add(user)
                frontier.append(user)
            if (device := _node_tensor_device(user)) is not None:
                fallback_devices.add(device)
    return devices or fallback_devices


def _align_graph_constants_with_consumers(graph_module: GraphModule) -> tuple[str, ...]:
    """Place folded tensor constants beside their sharded graph consumers."""

    relocated: list[str] = []
    for node in graph_module.graph.nodes:
        if node.op != "get_attr":
            continue
        target = str(node.target)
        module_path, _, attribute_name = target.rpartition(".")
        owner = graph_module.get_submodule(module_path)
        value = getattr(owner, attribute_name)
        if not torch.is_tensor(value):
            continue
        consumer_devices = _consumer_peer_devices(node)
        if len(consumer_devices) != 1:
            continue
        target_device = next(iter(consumer_devices))
        if value.device == target_device:
            continue
        moved = value.to(target_device)
        if isinstance(value, torch.nn.Parameter):
            moved = torch.nn.Parameter(moved, requires_grad=value.requires_grad)
        setattr(owner, attribute_name, moved)
        relocated.append(target)
    return tuple(relocated)


def _qwen4_exp_cache_input_names(contract: Qwen4ExpCacheContract) -> tuple[str, ...]:
    """Expose the conventional cache names consumed by the page-cache rewriter."""

    attention_ordinals = {
        layer.layer_idx: ordinal
        for ordinal, layer in enumerate(
            layer for layer in contract.layers if any(spec.state_kind == "kv_key" for spec in layer.tensors)
        )
    }
    names: list[str] = []
    for spec in contract.tensors:
        if spec.state_kind == "kv_key":
            names.append(f"past_key_cache_{attention_ordinals[spec.layer_idx]}")
        elif spec.state_kind == "kv_value":
            names.append(f"past_value_cache_{attention_ordinals[spec.layer_idx]}")
        else:
            names.append(spec.name)
    return tuple(names)


def export_qwen4_exp_hmonnx(
    native_model: torch.nn.Module,
    output_dir: str | Path,
    *,
    text_config: Any | None = None,
    sample_inputs: tuple[torch.Tensor, ...] | None = None,
    calibration_inputs: Sequence[tuple[torch.Tensor, ...]] | None = None,
    quant_config: Mapping[str, Any] | None = None,
    batch_size: int = 1,
    input_sequence_length: int | None = None,
    cache_length: int = 1,
    opset_version: int = 17,
    convert_to_hmonnx: bool = True,
    quant_device: str | torch.device | None = None,
    validate_export_graph_parity: bool = False,
    use_fused_gdr: bool = False,
    linear_attention_mode: str | None = None,
    return_hidden_states: bool = False,
    ple_quant_type: str = DEFAULT_PLE_QUANT_TYPE,
    export_wrapper: torch.nn.Module | None = None,
    export_input_names: tuple[str, ...] | None = None,
    export_output_names: tuple[str, ...] | None = None,
) -> Qwen4ExpHMONNXExportResult:
    """Adapt a float or GPTQModel Qwen4-Exp text model and export its decode graph.

    GPTQModel ``QuantLinear`` modules are transferred to ownership-preserving
    ``GPTQPackedLinear`` containers before tracing. XH2 PTQ then converts the
    canonical GPTQ words directly to HM storage without dense checkpoint
    expansion.
    """

    ple_quant_type = resolve_ple_quant_type(ple_quant_type)
    from .ple_export import extract_qwen4_exp_ple_specs

    native_ple_specs = (extract_qwen4_exp_ple_specs(native_model)
                        if convert_to_hmonnx and ple_quant_type == "sefp4" else [])
    if sample_inputs is None:
        raise ValueError("sample_inputs are required for Qwen4-Exp export")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    capability = inspect_qwen4_exp_export_capability(
        native_model if text_config is None else text_config,
        model=native_model,
        require_hmonnx_runtime=convert_to_hmonnx,
    )
    preflight_path = output_path / "qwen4_exp_hmonnx_preflight.json"
    _write_json(preflight_path, capability.to_dict())
    capability.assert_supported()
    converted_gptq_modules = _adapt_gptq_linears(native_model)

    wrapper = export_wrapper or build_qwen4_exp_text_export_wrapper(
        native_model,
        text_config=text_config,
        batch_size=batch_size,
        input_sequence_length=int(input_sequence_length or sample_inputs[0].shape[1]),
        cache_length=cache_length,
        require_hmonnx_runtime=convert_to_hmonnx,
        return_hidden_states=return_hidden_states,
    ).eval()
    from .modeling import adapt_qwen4_exp_rms_norms

    adapt_qwen4_exp_rms_norms(wrapper)
    if use_fused_gdr:
        from .modeling import Qwen4ExpExportableGatedDeltaNet

        for module in tuple(wrapper.modules()):
            if isinstance(module, Qwen4ExpExportableGatedDeltaNet):
                module.enable_fused_gdr(linear_attention_mode=linear_attention_mode)
    from .modeling import Qwen4ExpExportableAttention

    selected_modules = [module for module in wrapper.modules() if isinstance(module, Qwen4ExpExportableAttention)]
    if selected_modules and (batch_size != 1 or cache_length < 4):
        raise ValueError("SelectedKV export requires batch_size=1 and cache_length >= 4")
    cache_input_names = _qwen4_exp_cache_input_names(wrapper.cache_contract)
    from ...mrope import MROPE_CURRENT_POSITION_NAMES, MROPE_POOL_POSITION_NAMES

    input_names = (
        "inputs_embeds",
        "past_seq_length",
        "current_input_length",
        "ple_input_ids",
        *MROPE_CURRENT_POSITION_NAMES,
        *MROPE_POOL_POSITION_NAMES,
        *cache_input_names,
    )
    inplace_recurrent_layers = {
        module.layer_idx
        for module in wrapper.modules()
        if use_fused_gdr
        and getattr(module, "linear_attention_mode", None) == "chunk"
        and bool(getattr(getattr(module, "chunk_scan_op", None), "state_is_cache", False))
    }
    inplace_recurrent_names = {
        spec.name
        for spec in wrapper.cache_contract.tensors
        if spec.state_kind == "delta_recurrent" and spec.layer_idx in inplace_recurrent_layers
    }
    output_names = (
        "logits",
        *(("target_hidden_states",) if return_hidden_states else ()),
        *(f"present_{name}" for name in cache_input_names if name not in inplace_recurrent_names),
    )
    input_names = export_input_names or input_names
    output_names = export_output_names or output_names
    if selected_modules:
        from .qsa_host_controls import QSA_HOST_CONTROL_NAMES, build_qsa_host_controls

        topks = {module.packed_qsa.candidate_topk for module in selected_modules}
        if len(topks) != 1:
            raise ValueError("QSA layers must share the host-control candidate width")
        topk = topks.pop()
        base_names = tuple(input_names)
        if any(name in base_names for name in QSA_HOST_CONTROL_NAMES):
            raise ValueError("QSA host-control input names are reserved")

        def with_qsa_controls(values):
            if len(values) != len(base_names):
                raise ValueError("QSA export inputs must match the base input ABI")
            lookup = dict(zip(base_names, values, strict=True))
            reference = values[0]
            controls = build_qsa_host_controls(
                int(lookup["past_seq_length"].item()), int(lookup["current_input_length"].item()),
                int(reference.shape[1]), cache_length, topk, device=reference.device, dtype=reference.dtype,
            )
            return (*values, *controls)

        sample_inputs = with_qsa_controls(sample_inputs)
        if calibration_inputs is not None:
            calibration_inputs = tuple(with_qsa_controls(values) for values in calibration_inputs)
        input_names = (*base_names, *QSA_HOST_CONTROL_NAMES)
    if len(sample_inputs) != len(input_names):
        raise ValueError(
            f"Qwen4-Exp sample input count mismatch: expected {len(input_names)}, got {len(sample_inputs)}"
        )

    onnx_path = output_path / "qwen4_exp_text.onnx"
    graph_counts: dict[str, int] = {}
    removed_gdr_projection_roundtrips: tuple[str, ...] = ()
    removed_static_layout_nodes: tuple[str, ...] = ()
    runtime_parity: list[dict[str, Any]] = []
    repaired_identity_nodes: tuple[str, ...] = ()
    graph_sample_inputs = sample_inputs
    portable_cpu_graph = bool(getattr(wrapper, "_explicit_device_forward", False))
    if convert_to_hmonnx:
        from .xhquant_extensions import qwen4_exp_xhquant_extension_scope

        extension_scope = qwen4_exp_xhquant_extension_scope()
    else:
        extension_scope = nullcontext()

    with extension_scope:
        if convert_to_hmonnx:
            from xhquant.api import (
                ConfigDict,
                FrontendType,
                PrecisionMode,
                ptq_quantize,
                to_export_graph,
                to_frontend_graph,
                to_quant_graph,
            )

            with (
                _dynamo_export_compatibility_scope(),
                _opaque_gptq_modules_for_dynamo(wrapper) as (opaque_modules, opaque_kinds),
            ):
                frontend_graph = to_frontend_graph(
                    wrapper,
                    FrontendType.DynamoFX,
                    sample_inputs,
                    enable_fuse=False,
                )
            _restore_gptq_modules_after_dynamo(
                frontend_graph, opaque_modules, opaque_kinds,
                original_module_paths={id(module): path for path, module in wrapper.named_modules()},
            )
            _normalize_frontend_gather_overloads(frontend_graph)
            _normalize_frontend_rmsnorm_modules(frontend_graph)
            _normalize_frontend_rope_modules(frontend_graph)
            _normalize_frontend_placeholder_modules(frontend_graph)
            _normalize_functional_linear_modules(frontend_graph)
            _normalize_frontend_dynamic_slice_modules(frontend_graph)
            _normalize_frontend_llm_cache_modules(frontend_graph)
            _normalize_functional_conv1d_modules(frontend_graph)
            removed_gdr_projection_roundtrips = _remove_gdr_qkv_projection_roundtrips(frontend_graph)
            # Dynamo may leave functional convs until model normalization, after
            # the SDK's initial Conv1d pass. Lower those before quantization too.
            from xhquant.frontend.fx_transform.conv1d_to_conv2d import Conv1dToConv2d

            frontend_graph = Conv1dToConv2d()(frontend_graph, sample_inputs)
            removed_nodes = sanitize_qwen4_exp_frontend_graph(frontend_graph)
            frontend_graph = _isolate_frontend_module_ownership(frontend_graph)
            if portable_cpu_graph:
                removed_nodes += _remove_device_only_transfer_nodes(frontend_graph)
                frontend_graph.cpu()
                graph_sample_inputs = tuple(value.detach().cpu() for value in sample_inputs)
            if inplace_recurrent_names:
                from xhquant.core.cache_tensor import CacheTensor
            graph_counts["frontend"] = len(tuple(frontend_graph.graph.nodes))
            resolved_quant_config = ConfigDict(dict(quant_config or {"quant_type": "w8a8h1_sefp"}))
            quant_graph = to_quant_graph(frontend_graph, "xh2a", resolved_quant_config)
            graph_counts["quant"] = len(tuple(quant_graph.graph.nodes))
            calibration_source = calibration_inputs or (graph_sample_inputs,)
            calibration_data = [
                [value.detach().cpu() for value in values] if portable_cpu_graph else list(values)
                for values in calibration_source
            ]
            if inplace_recurrent_names:
                # Dynamo/ONNX capture needs ordinary Tensor inputs. ALIGNED
                # calibration executes the cache-mutating GDR backend instead.
                calibration_data = [
                    [
                        CacheTensor(value.detach().clone()) if name in inplace_recurrent_names else value
                        for name, value in zip(input_names, values, strict=True)
                    ]
                    for values in calibration_data
                ]
            resolved_device = torch.device(quant_device or sample_inputs[0].device)
            ptq_quantize(
                quant_graph,
                calibration_data,
                PrecisionMode.ALIGNED,
                [resolved_device],
                # The full 48-layer model produces more than one GPU's worth
                # of quantized tensors. Keep only the module currently being
                # converted on the execution GPU and offload completed module
                # state back to CPU.
                auto_release_unused_parameters=True,
                # DynamoFX has already populated shape metadata. Re-running
                # FakeTensor propagation would reject the explicit CPU/GPU
                # transfers required to trace an oversized sharded model.
                infer_shape=False,
            )
            quant_graph.fixed()
            _align_graph_constants_with_consumers(quant_graph)
            _normalize_quantized_moe_call_signatures(quant_graph)
            export_graph = to_export_graph(quant_graph, list(graph_sample_inputs))
            _align_graph_constants_with_consumers(export_graph)
            _remove_export_identity_nodes(export_graph)
            graph_counts["export"] = len(tuple(export_graph.graph.nodes))
            exported_path = _export_qwen4_exp_hmonnx_program(
                export_graph,
                graph_sample_inputs,
                onnx_path,
                input_names=input_names,
                output_names=output_names,
            )
            onnx_path = Path(exported_path)
        else:
            removed_nodes = ()
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    sample_inputs,
                    str(onnx_path),
                    input_names=list(input_names),
                    output_names=list(output_names),
                    opset_version=opset_version,
                )

        import onnx

        if convert_to_hmonnx:
            repaired_identity_nodes = _repair_duplicate_identity_nodes(onnx_path)
            from .serving_graph import (
                normalize_gather_index_graph,
                normalize_qwen4_static_layout_graph,
            )

            normalize_gather_index_graph(onnx_path)
            removed_static_layout_nodes = normalize_qwen4_static_layout_graph(onnx_path)
        onnx.checker.check_model(str(onnx_path))
        if convert_to_hmonnx and validate_export_graph_parity:
            parity_sample_inputs = graph_sample_inputs
            if inplace_recurrent_names:
                parity_sample_inputs = tuple(
                    CacheTensor(value.clone()) if name in inplace_recurrent_names else value
                    for name, value in zip(input_names, graph_sample_inputs, strict=True)
                )
            runtime_parity = _validate_hmonnx_runtime_parity(
                onnx_path,
                parity_sample_inputs,
                export_graph,
                device=quant_device or sample_inputs[0].device,
                reference_device="cpu" if portable_cpu_graph else quant_device or sample_inputs[0].device,
            )

    ple_quantization = {"status": "disabled", "quant_type": ple_quant_type}
    if convert_to_hmonnx and ple_quant_type == "sefp4":
        from .ple_export import export_ple_sefp4_hmonnx

        # Validate the captured quant graph before applying the intentional
        # lossy deployment compression. The codec has its own SDK parity tests.
        ple_quantization = export_ple_sefp4_hmonnx(onnx_path, onnx_path, ple_specs=native_ple_specs)
        onnx.checker.check_model(str(onnx_path))

    final_model = onnx.load(str(onnx_path), load_external_data=False)
    input_manifest = []
    dtype_names = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.DOUBLE: "float64"}
    for value in final_model.graph.input:
        tensor_type = value.type.tensor_type
        input_manifest.append({
            "name": value.name,
            "shape": [dim.dim_value if dim.HasField("dim_value") else dim.dim_param or None
                      for dim in tensor_type.shape.dim],
            "dtype": dtype_names.get(tensor_type.elem_type, onnx.TensorProto.DataType.Name(
                tensor_type.elem_type).lower()),
        })
    io_manifest_path = output_path / "qwen4_exp_hmonnx_io.json"
    _write_json(
        io_manifest_path,
        {
            "inputs": input_manifest,
            "outputs": list(output_names),
            "use_fused_gdr": use_fused_gdr,
            "inplace_recurrent_cache_inputs": sorted(inplace_recurrent_names),
            "cache_contract": wrapper.cache_contract.to_dict(),
            "converted_gptq_modules": list(converted_gptq_modules),
            "removed_frontend_nodes": list(removed_nodes),
            "removed_gdr_projection_roundtrips": list(removed_gdr_projection_roundtrips),
        "removed_static_layout_nodes": list(removed_static_layout_nodes),
            "graph_node_counts": graph_counts,
            "onnx_checker": "passed",
            "hmonnx_runtime_parity": runtime_parity,
            "hmonnx_runtime_parity_scope": (
                "before_ple_externalization" if ple_quantization.get("status") == "externalized" else "exported_graph"
            ),
            "ple_quantization": ple_quantization,
            "repaired_duplicate_identity_nodes": list(repaired_identity_nodes),
        },
    )
    return Qwen4ExpHMONNXExportResult(
        onnx_path=str(onnx_path),
        hmonnx_path=str(onnx_path) if convert_to_hmonnx else None,
        preflight_path=str(preflight_path),
        io_manifest_path=str(io_manifest_path),
        capability=capability.to_dict(),
        converted_gptq_modules=converted_gptq_modules,
    )


def convert_qwen4_exp_kv_to_llm_cache(
    input_path: str | Path,
    output_path: str | Path,
    *,
    context_length: int,
    num_kv_heads: int,
    head_dim: int,
) -> str:
    """Publish SDK-standard KVcache nodes with explicit in-place K/V storage.

    The runtime class and existing ABI remain named LLMCache; XH2 HMONNX
    uses KVcache, matching the SDK exporter and DeepSeek graphs.
    In-place updates are enabled on the SDK runtime, not by an ONNX attribute.
    A negative attention limit selects full storage rather than a sliding view.
    """
    import re

    import onnx
    from onnx import helper

    source, destination = Path(input_path), Path(output_path)
    model = onnx.load(str(source), load_external_data=False)
    if any(node.op_type == "PageKVCache" for node in model.graph.node):
        raise ValueError("explicit LLMCache conversion requires the raw graph, not PageKVCache")
    pattern = re.compile(r"^past_(key|value)_cache_(\d+)$")
    indexer_pattern = re.compile(r"^qsa_indexer_(?:key|pool)_cache_\d+$")
    graph_inputs = {value.name: value for value in model.graph.input}
    producers = {name: node for node in model.graph.node for name in node.output}
    writers: dict[tuple[str, int], Any] = {}
    indexer_writers: set[str] = set()
    for node in model.graph.node:
        if node.op_type not in {"KVcache", "LLMCache"} or len(node.input) < 4:
            continue
        name = node.input[3]
        visited: set[str] = set()
        while name not in graph_inputs and name not in visited:
            visited.add(name)
            alias = producers.get(name)
            if alias is None or alias.op_type not in {"Identity", "HMIdentity"}:
                break
            name = alias.input[0]
        if indexer_pattern.fullmatch(name):
            attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            if name not in graph_inputs or attrs.get("axis", 2) != 2 or attrs.get("only_handle_old_cache", 0):
                raise ValueError(f"Qwen4 indexer cache {name} must have an axis-2 cache writer")
            if name in indexer_writers:
                raise ValueError(f"duplicate Qwen4 indexer cache writer {name}")
            node.input[3] = name
            indexer_writers.add(name)
            continue
        match = pattern.fullmatch(name)
        if match is None:
            continue
        kind, ordinal = match.groups()
        key = (kind, int(ordinal))
        if key in writers:
            raise ValueError(f"duplicate Qwen4 {kind} cache writer {ordinal}")
        value = graph_inputs.get(name)
        shape = [] if value is None else [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        if shape != [1, num_kv_heads, context_length, head_dim]:
            raise ValueError(f"Qwen4 cache {name} has incompatible shape {shape}")
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if attrs.get("axis", 2) != 2 or attrs.get("only_handle_old_cache", 0):
            raise ValueError(f"Qwen4 cache {name} must be an axis-2 K/V writer")
        node.op_type = "KVcache"
        node.input[3] = name
        del node.attribute[:]
        node.attribute.extend([helper.make_attribute("axis", 2),
                               helper.make_attribute("attention_max_length", -1)])
        writers[key] = node
    keys = {ordinal for kind, ordinal in writers if kind == "key"}
    values = {ordinal for kind, ordinal in writers if kind == "value"}
    expected = {name for name in graph_inputs if pattern.fullmatch(name)}
    if not keys or keys != values or len(writers) != len(expected):
        raise ValueError("Qwen4 explicit LLMCache conversion requires all paired K/V writers")
    if indexer_writers != {name for name in graph_inputs if indexer_pattern.fullmatch(name)}:
        raise ValueError("Qwen4 explicit LLMCache conversion requires all indexer cache writers")
    for ordinal in keys:
        if list(writers[("key", ordinal)].input[1:3]) != list(writers[("value", ordinal)].input[1:3]):
            raise ValueError(f"Qwen4 K/V cache lengths differ for ordinal {ordinal}")
    # Normalize legacy indexer/state cache aliases too, preserving their ports
    # and attributes. The SDK parses both names with the same cache semantics.
    for node in model.graph.node:
        if node.domain == "XH2a" and node.op_type == "LLMCache":
            node.op_type = "KVcache"
    # K/V and indexer storage belongs to the caller and is updated in place.
    # Keep writer outputs inside the attention graph, but do not expose redundant
    # functional-cache ports at the serving boundary.
    removed_outputs = {f"present_{name}" for name in expected | indexer_writers}
    outputs = [value for value in model.graph.output if value.name not in removed_outputs]
    del model.graph.output[:]
    model.graph.output.extend(outputs)
    # Drop only now-unused aliases of those ports. Never run general DCE here:
    # cache writers have observable mutations even without a public output.
    pending = set(removed_outputs)
    while pending:
        name = pending.pop()
        alias = producers.get(name)
        if (alias is None or alias.op_type not in {"Identity", "HMIdentity"}
                or alias.domain not in {"", "XH2a", "ai.houmo.xh2a"}
                or len(alias.input) != 1 or list(alias.output) != [name]
                or any(name in node.input for node in model.graph.node)
                or any(value.name == name for value in model.graph.output)
                or any(attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
                       for node in model.graph.node for attr in node.attribute)):
            continue
        source_name = alias.input[0]
        model.graph.node.remove(alias)
        pending.add(source_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(destination), save_as_external_data=False)
    _link_qwen4_exp_external_data(model, source=source, destination=destination)
    onnx.checker.check_model(str(destination))
    return str(destination)


def convert_qwen4_exp_kv_to_page_cache(
    input_path: str | Path,
    output_path: str | Path,
    *,
    context_length: int,
    num_kv_heads: int,
    head_dim: int,
) -> str:
    """Replace Qwen4 K/V writers with PageKVCache while retaining QSA computation."""

    import re

    import onnx
    from onnx import helper

    source = Path(input_path)
    destination = Path(output_path)
    model = onnx.load(str(source), load_external_data=False)
    cache_pattern = re.compile(r"^past_(key|value)_cache_(\d+)$")
    producers = {output: node for node in model.graph.node for output in node.output if output}

    def resolve_transparent_source(value_name: str) -> str:
        visited: set[str] = set()
        while value_name not in visited:
            visited.add(value_name)
            producer = producers.get(value_name)
            if (
                producer is None
                or producer.op_type not in {"Cast", "HMIdentity", "Identity"}
                or len(producer.input) != 1
            ):
                break
            value_name = producer.input[0]
        return value_name

    key_nodes: dict[int, tuple[int, Any]] = {}
    value_nodes: dict[int, tuple[int, Any]] = {}
    for node_index, node in enumerate(model.graph.node):
        if node.op_type not in {"KVcache", "LLMCache"} or len(node.input) < 4:
            continue
        match = cache_pattern.fullmatch(resolve_transparent_source(node.input[3]))
        if match is None:
            continue
        kind, ordinal_text = match.groups()
        ordinal = int(ordinal_text)
        target = key_nodes if kind == "key" else value_nodes
        if ordinal in target:
            raise ValueError(f"duplicate Qwen4 {kind} cache writer {ordinal}")
        target[ordinal] = (node_index, node)
    if set(key_nodes) != set(value_nodes) or not key_nodes:
        raise ValueError(
            "Qwen4 PageKVCache conversion requires paired K/V writers; "
            f"key={sorted(key_nodes)}, value={sorted(value_nodes)}"
        )

    replacement_by_index: dict[int, Any] = {}
    removed_indices: set[int] = set()
    for ordinal in sorted(key_nodes):
        key_index, key_node = key_nodes[ordinal]
        value_index, value_node = value_nodes[ordinal]
        if list(key_node.input[1:3]) != list(value_node.input[1:3]):
            raise ValueError(f"Qwen4 K/V cache lengths differ for ordinal {ordinal}")
        insertion_index = max(key_index, value_index)
        replacement_by_index[insertion_index] = helper.make_node(
            "PageKVCache",
            inputs=[
                key_node.input[0],
                value_node.input[0],
                key_node.input[1],
                key_node.input[2],
            ],
            outputs=[key_node.output[0], value_node.output[0]],
            name=f"page_kv_cache_{ordinal}",
            domain=key_node.domain or "ai.houmo.xh2a",
            layer_idx=ordinal,
            cache_index=ordinal,
            attention_kind="sliding_attention",
            sliding_window=int(context_length),
            output_sequence_length=int(context_length),
            num_kv_heads=int(num_kv_heads),
            head_size=int(head_dim),
        )
        removed_indices.update((key_index, value_index))

    nodes = []
    for node_index, node in enumerate(model.graph.node):
        if node_index in removed_indices:
            replacement = replacement_by_index.get(node_index)
            if replacement is not None:
                nodes.append(replacement)
            continue
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    cache_outputs = re.compile(r"^present_past_(?:key|value)_cache_\d+$")
    outputs = [value for value in model.graph.output if cache_outputs.fullmatch(value.name) is None]
    del model.graph.output[:]
    model.graph.output.extend(outputs)

    required_values = {value.name for value in model.graph.output}
    retained_nodes = []
    for node in reversed(model.graph.node):
        if not required_values.intersection(node.output):
            continue
        retained_nodes.append(node)
        required_values.update(name for name in node.input if name)
    retained_nodes.reverse()
    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)
    graph_inputs = [value for value in model.graph.input
                    if value.name in required_values or not cache_pattern.fullmatch(value.name)]
    del model.graph.input[:]
    model.graph.input.extend(graph_inputs)

    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(destination), save_as_external_data=False)
    _link_qwen4_exp_external_data(model, source=source, destination=destination)
    onnx.checker.check_model(str(destination))
    return str(destination)


def _link_qwen4_exp_external_data(model: Any, *, source: Path, destination: Path) -> None:
    """Keep large HMONNX weights zero-copy when writing a sibling graph."""

    from onnx import TensorProto, external_data_helper

    locations: set[str] = set()
    for tensor in external_data_helper._get_all_tensors(model):
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        location = next((item.value for item in tensor.external_data if item.key == "location"), None)
        if location:
            locations.add(location)
    from .ple_export import _metadata_specs

    for spec in _metadata_specs(model, source.parent):
        for key in ("packed", "exponent"):
            locations.add(spec[key]["path"])
    for location in locations:
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe Qwen4 HMONNX external-data location: {location!r}")
        source_data = source.parent / relative
        destination_data = destination.parent / relative
        if not source_data.is_file():
            raise FileNotFoundError(f"Qwen4 HMONNX external data is missing: {source_data}")
        destination_data.parent.mkdir(parents=True, exist_ok=True)
        if destination_data.exists():
            if not os.path.samefile(source_data, destination_data):
                raise FileExistsError(f"refusing to replace external data at {destination_data}")
            continue
        os.link(source_data, destination_data)


@torch.no_grad()
def run_qwen4_exp_hmonnx_golden(
    hmonnx_path: str | Path,
    sample_inputs: tuple[torch.Tensor, ...],
    golden_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    step: int = 0,
) -> dict[str, torch.Tensor]:
    """Execute one cached decode turn and persist full HMONNX Golden tensors."""

    from .xhquant_extensions import qwen4_exp_xhquant_extension_scope

    with qwen4_exp_xhquant_extension_scope():
        from xhquant.api import HMONNXGoldenInference
        from xhquant.core.cache_tensor import CacheTensor

        session = HMONNXGoldenInference(str(hmonnx_path))
        session.exec_device = torch.device(device)
        session.save_golden = True
        session.golden_dir = str(golden_dir)
        session.step = int(step)
        session.initialize()
        input_names = session.get_input_names()
        from .host_ple_inputs import HostPLEInputs
        from .qsa_host_controls import QSA_HOST_CONTROL_NAMES, build_qsa_host_controls, qsa_host_control_spec

        host_ple = HostPLEInputs.from_graph(hmonnx_path) if Path(hmonnx_path).is_file() else None
        host_names = [spec["input_name"] for spec in host_ple.manifest["layers"]] if host_ple else []
        base_names = [name for name in input_names if name not in host_names]
        if len(sample_inputs) == len(input_names):
            values = dict(zip(input_names, sample_inputs, strict=True))
        elif host_ple and len(sample_inputs) == len(base_names):
            values = dict(zip(base_names, sample_inputs, strict=True))
        elif (qsa_host_control_spec(session) is not None
              and len(sample_inputs) == len(base_names) - len(QSA_HOST_CONTROL_NAMES)):
            names = [name for name in base_names if name not in QSA_HOST_CONTROL_NAMES]
            values = dict(zip(names, sample_inputs, strict=True))
        else:
            raise ValueError(
                f"Qwen4-Exp HMONNX input count mismatch: expected {len(input_names)}, got {len(sample_inputs)}"
            )
        qsa_spec = qsa_host_control_spec(session)
        if qsa_spec is not None and "qsa_attention_mask" not in values:
            topk, mask_dtype = qsa_spec
            pool_cache = next(value for name, value in values.items() if name.startswith("qsa_indexer_key_cache_"))
            controls = build_qsa_host_controls(
                int(values["past_seq_length"].item()), int(values["current_input_length"].item()),
                int(values["inputs_embeds"].shape[1]), int(pool_cache.shape[2]) * 4,
                topk, device=device, dtype=mask_dtype,
            )
            values.update(zip(QSA_HOST_CONTROL_NAMES, controls, strict=True))
        if host_ple:
            embeddings = host_ple.prepare(
                values["ple_input_ids"], values, int(values["current_input_length"].item()), device=device,
            )
            for name, embedding in zip(host_names, embeddings, strict=True):
                if name in values and not torch.equal(values[name].to(embedding), embedding):
                    raise ValueError(f"Provided PLE embeddings disagree with host lookup: {name}")
                values[name] = embedding
        input_feed = {
            name: (
                values[name].to(session.get_input(name).dtype)
                if isinstance(values[name], CacheTensor)
                else values[name].detach().to(session.get_input(name).dtype)
            )
            for name in input_names
        }
        if host_ple and any((Path(golden_dir) / f"step_{step}" / f"{name}.npy").exists() for name in host_names):
            raise FileExistsError("External PLE Golden step already exists")
        outputs = session.run(input_feed)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        output_map = dict(zip(session.get_output_names(), outputs, strict=True))
    if host_ple:
        import numpy as np

        from .golden_runtime import copy_ple_resources, record_ple_sample

        root = Path(golden_dir)
        manifest_path = root / "ple/manifest.json"
        if manifest_path.exists():
            entry = {"path": "ple/manifest.json", "manifest": json.loads(manifest_path.read_text())}
            saved_layers = entry["manifest"]["layers"]
            if len(saved_layers) != len(host_ple.manifest["layers"]):
                raise ValueError("Existing Golden PLE layer contract differs")
            for saved, original in zip(saved_layers, host_ple.manifest["layers"], strict=True):
                if ({k: v for k, v in saved.items() if k not in {"packed", "exponent"}}
                        != {k: v for k, v in original.items() if k not in {"packed", "exponent"}}):
                    raise ValueError("Existing Golden PLE hash contract differs")
                for kind in ("packed", "exponent"):
                    archived = (root / "ple" / saved[kind]["path"]).resolve()
                    original_path = host_ple.root / original[kind]["path"]
                    if not archived.is_relative_to((root / "ple").resolve()):
                        raise ValueError("Unsafe existing Golden PLE asset path")
                    if not os.path.samefile(archived, original_path):
                        if archived.stat().st_size != original_path.stat().st_size:
                            raise ValueError("Existing Golden PLE payload size differs")
                        with archived.open("rb") as saved_file, original_path.open("rb") as source_file:
                            while block := source_file.read(16 * 1024 * 1024):
                                if saved_file.read(len(block)) != block:
                                    raise ValueError("Existing Golden PLE payload bytes differ")
        else:
            entry = copy_ple_resources(root, {"decode": host_ple})["decode"]
        destination = root / f"step_{step}"
        for name in host_names:
            value = input_feed[name].detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            np.save(destination / f"{name}.npy", value.numpy(), allow_pickle=False)
        record_ple_sample(root, entry, "decode", int(step), destination, host_ple.last_indices)
    summary_path = Path(golden_dir) / f"step_{step}" / "qwen4_exp_runtime_summary.json"
    _write_json(
        summary_path,
        {
            "hmonnx_path": str(Path(hmonnx_path).resolve()),
            "inputs": [_tensor_manifest(name, value) for name, value in input_feed.items()],
            "outputs": [_tensor_manifest(name, value) for name, value in output_map.items()],
        },
    )
    return output_map


def _adapt_gptq_linears(model: torch.nn.Module) -> tuple[str, ...]:
    if not any(type(module).__module__.startswith("gptqmodel.") for module in model.modules()):
        return ()
    try:
        from gptqmodel.nn_modules.qlinear import PackableQuantLinear
    except ImportError:
        return ()
    if not any(isinstance(module, PackableQuantLinear) for module in model.modules()):
        return ()
    from ..._dequant_converter import replace_gptqmodel_quant_linears_with_packed

    return replace_gptqmodel_quant_linears_with_packed(model)


def _repair_duplicate_identity_nodes(onnx_path: str | Path) -> tuple[str, ...]:
    """Remove dangling duplicate identities created by the stock optimizer.

    The optimizer already renamed the producer to the identity output.  When
    two identities consume the same value, its second copy can remain as a
    self-shadowing node whose input no longer exists.  The output is already
    available, so removing that duplicate is semantics preserving.
    """

    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    available = {value.name for value in model.graph.input}
    available.update(value.name for value in model.graph.initializer)
    remove_indices: list[int] = []
    removed_names: list[str] = []
    for index, node in enumerate(model.graph.node):
        missing_inputs = [name for name in node.input if name and name not in available]
        is_duplicate_identity = node.op_type in {"Identity", "HMIdentity"} and all(
            output in available for output in node.output
        )
        if missing_inputs and is_duplicate_identity:
            remove_indices.append(index)
            removed_names.append(node.name)
            continue
        available.update(node.output)
    for index in reversed(remove_indices):
        del model.graph.node[index]
    if remove_indices:
        onnx.save(model, str(onnx_path))
    return tuple(removed_names)


def _export_qwen4_exp_hmonnx_program(
    export_graph: torch.nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    onnx_path: str | Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
) -> str:
    """Export the quantized graph through the public SDK exporter."""
    from xhquant.api import ConfigDict, to_export_hmonnx_v2

    placeholders = [node for node in export_graph.graph.nodes if node.op == "placeholder"]
    export_inputs = tuple(
        value.to(node.meta["val"].dtype)
        if isinstance(value, torch.Tensor) and isinstance(node.meta.get("val"), torch.Tensor)
        else value
        for node, value in zip(placeholders, sample_inputs, strict=True)
    )
    return to_export_hmonnx_v2(
        export_graph,
        export_inputs,
        str(onnx_path),
        cfg=ConfigDict(input_names=list(input_names), output_names=list(output_names)),
        save_as_external_data=True,
    )


@torch.no_grad()
def _validate_hmonnx_runtime_parity(
    hmonnx_path: str | Path,
    sample_inputs: tuple[torch.Tensor, ...],
    reference_model: torch.nn.Module,
    *,
    device: str | torch.device,
    reference_device: str | torch.device,
) -> list[dict[str, Any]]:
    """Prove that the serialized HMONNX graph matches its XH2 export graph."""

    from xhquant.api import HMONNXGoldenInference
    from xhquant.core.cache_tensor import CacheTensor

    session = HMONNXGoldenInference(str(hmonnx_path))
    session.exec_device = torch.device(device)
    session.save_golden = False
    session.initialize()
    input_names = session.get_input_names()
    if len(input_names) != len(sample_inputs):
        raise RuntimeError(
            f"Qwen4-Exp HMONNX parity input count mismatch: expected {len(input_names)}, got {len(sample_inputs)}"
        )
    runtime_feed = {
        name: (
            CacheTensor(value.detach().clone()).to(device=session.exec_device, dtype=session.get_input(name).dtype)
            if isinstance(value, CacheTensor)
            else value.detach().to(device=session.exec_device, dtype=session.get_input(name).dtype)
        )
        for name, value in zip(input_names, sample_inputs, strict=True)
    }
    runtime_outputs = session.run(runtime_feed)
    if not isinstance(runtime_outputs, (tuple, list)):
        runtime_outputs = (runtime_outputs,)
    reference_inputs = tuple(
        CacheTensor(value.detach().clone()).to(device=reference_device, dtype=session.get_input(name).dtype)
        if isinstance(value, CacheTensor)
        else value.detach().to(device=reference_device, dtype=session.get_input(name).dtype)
        for name, value in zip(input_names, sample_inputs, strict=True)
    )
    reference_outputs = reference_model(*reference_inputs)
    if not isinstance(reference_outputs, (tuple, list)):
        reference_outputs = (reference_outputs,)
    output_names = session.get_output_names()
    if len(runtime_outputs) != len(reference_outputs) or len(output_names) != len(runtime_outputs):
        raise RuntimeError(
            "Qwen4-Exp HMONNX parity output count mismatch: "
            f"runtime={len(runtime_outputs)}, reference={len(reference_outputs)}, names={len(output_names)}"
        )

    parity: list[dict[str, Any]] = []
    for name, actual, expected in zip(output_names, runtime_outputs, reference_outputs, strict=True):
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"Qwen4-Exp HMONNX output {name!r} shape mismatch: "
                f"runtime={tuple(actual.shape)}, reference={tuple(expected.shape)}"
            )
        if actual.is_floating_point() or expected.is_floating_point():
            actual_float = actual.float()
            expected_float = expected.float().to(actual_float.device)
            if not torch.isfinite(actual_float).all() or not torch.isfinite(expected_float).all():
                raise RuntimeError(f"Qwen4-Exp HMONNX output {name!r} contains non-finite values")
            max_abs_error = float((actual_float - expected_float).abs().max()) if actual.numel() else 0.0
            torch.testing.assert_close(actual_float, expected_float, atol=1e-3, rtol=1e-3)
        else:
            expected = expected.to(actual.device)
            max_abs_error = (
                float((actual.to(torch.int64) - expected.to(torch.int64)).abs().max()) if actual.numel() else 0.0
            )
            torch.testing.assert_close(actual, expected)
        parity.append(
            {
                "name": name,
                "shape": list(actual.shape),
                "dtype": str(actual.dtype).removeprefix("torch."),
                "max_abs_error": max_abs_error,
            }
        )
    return parity


def _tensor_manifest(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "Qwen4ExpHMONNXExportResult",
    "build_qwen4_exp_decode_sample_inputs",
    "build_qwen4_exp_serving_sample_inputs",
    "convert_qwen4_exp_kv_to_llm_cache",
    "convert_qwen4_exp_kv_to_page_cache",
    "export_qwen4_exp_hmonnx",
    "load_qwen4_exp_model_for_export",
    "preflight_qwen4_exp_hmonnx_export",
    "run_qwen4_exp_hmonnx_golden",
    "sanitize_qwen4_exp_frontend_graph",
]
