"""Inject compiler-bound expert LoRA initializers into MoeBlock ONNX nodes.

Each routed-expert tensor contains one active adapter. MoeBlock reserves input
15 for ``s_scalar`` and appends 12 LoRA qweight/scale inputs at 16:28.

The main graph carries canonical L0 initializers. The four adapter bundles are
initializer-only ONNX files with an identical schema; the compiler selects one
bundle and replaces main-graph initializer values by name before model build.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import onnx
import torch
from onnx import helper

from .lora_constant_bundle import MacaronLoRAConstantBundle
from .lora_export_config import LoRAQuantization
from .lora_quantization import (
    AlignedLoRAQuantSpec,
    quantize_aligned_lora_weight,
)
from .lora_weights import (
    ADAPTER_NAMES,
    MacaronLoRAAdapters,
    pack_expert_lora_layer,
)


DOMAIN = "ai.houmo.xh2a"
BASE_MOEBLOCK_INPUT_COUNT = 15
BASE_MOEBLOCK_WITH_S_SCALAR_INPUT_COUNT = 16
LORA_MOEBLOCK_INPUT_COUNT = 28
ROUTED_EXPERT_LORA_WEIGHT_KEYS = (
    "lora_gate_A",
    "lora_gate_B",
    "lora_up_A",
    "lora_up_B",
    "lora_down_A",
    "lora_down_B",
)
GROUP_SIZE = 64
LORA_QUANT_ATTRIBUTES = (
    "lora_mode",
    "lora_weight_exp_bit",
    "lora_weight_man_bit",
    "lora_weight_nshare",
    "lora_weight_rounding",
    "lora_weight_hidden_bit",
)


DEFAULT_ROUTED_EXPERT_LORA_QUANT_SPEC = AlignedLoRAQuantSpec(
    mode="ssfp",
    nshare=GROUP_SIZE,
    man_bit=8,
    hidden_bit=False,
    rounding="RNE",
    exp_bit=16,
)


@dataclass(frozen=True)
class RoutedExpertLoRABinding:
    """One MoeBlock layer and its validated LoRA weight formats."""

    layer_idx: int
    projection_specs: tuple[tuple[str, AlignedLoRAQuantSpec], ...]
    quant_override: AlignedLoRAQuantSpec | None = None

    @property
    def quant_specs(self) -> dict[str, AlignedLoRAQuantSpec]:
        return dict(self.projection_specs)


def quantize_routed_expert_lora_weight(
    weight: np.ndarray,
    *,
    quant_spec: AlignedLoRAQuantSpec = (DEFAULT_ROUTED_EXPERT_LORA_QUANT_SPEC),
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize one adapter's [experts, rows, columns] tensor as a Linear weight.

    The transpose and grouping intentionally match ``QLinear`` aligned export:
    [expert, output, input] -> [expert, input, output], grouped on input.
    """
    if weight.ndim != 3:
        raise ValueError(f"Active expert LoRA weight must be 3-D, got {weight.shape}")
    # QLinear quantizes its static weight in the configured fake dtype
    # (float16 for this export).  Preserve that contract so SEFP exponents are
    # int16 just like ordinary HMONNX Linear weights.
    linear_weight = torch.from_numpy(weight).to(torch.float16).transpose(-2, -1).contiguous()
    return quantize_aligned_lora_weight(
        linear_weight.numpy(),
        dim=1,
        spec=quant_spec,
    )


def moeblock_node_for_layer(graph, layer_idx: int):
    target_name = "node_moe_block" if layer_idx == 0 else f"node_moe_block_{layer_idx}"
    for node in graph.node:
        if node.op_type == "MoeBlock" and node.domain == DOMAIN and node.name == target_name:
            return node
    return None


def _attribute_values(node) -> dict[str, object]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def routed_lora_quant_specs(
    node,
) -> dict[str, AlignedLoRAQuantSpec]:
    """Read per-projection LoRA quantization formats from a base MoeBlock.

    Gate and up use the shared MoeBlock weight specification.  MoeBlock has an
    optional down-projection mantissa override, so down LoRA follows it too.
    """

    attrs = _attribute_values(node)
    mode = attrs.get("mode", "sefp")
    if isinstance(mode, bytes):
        mode = mode.decode()
    if not isinstance(mode, str):
        raise ValueError(f"MoeBlock mode must be a string, got {mode!r}")
    common = AlignedLoRAQuantSpec(
        mode=mode,
        nshare=int(attrs.get("hmfp_weight_nshare", GROUP_SIZE)),
        man_bit=int(attrs.get("hmfp_weight_man_bit", 8)),
        hidden_bit=bool(attrs.get("hmfp_weight_hidden_bit", 0 if mode == "ssfp" else 1)),
        rounding=(
            attrs.get("hmfp_weight_rounding", "RNE").decode()
            if isinstance(attrs.get("hmfp_weight_rounding", "RNE"), bytes)
            else str(attrs.get("hmfp_weight_rounding", "RNE"))
        ),
        exp_bit=int(attrs.get("hmfp_weight_exp_bit", 16)),
    )
    down_man_bit = attrs.get("hmfp_weight_man_bit_down")
    down = common if down_man_bit is None else replace(common, man_bit=int(down_man_bit))
    specs = {"gate": common, "up": common, "down": down}
    for spec in specs.values():
        spec.validate()
    return specs


def _validate_lora_activation_quantization(
    node,
    quantization: LoRAQuantization,
) -> None:
    if quantization.act_spec is None:
        return
    attrs = _attribute_values(node)
    actual = {
        "mode": "sefp",
        "man_bit": int(attrs.get("hmfp_act_man_bit", 8)),
        "hidden_bit": bool(attrs.get("hmfp_act_hidden_bit", 1)),
    }
    expected = dict(quantization.act_spec)
    if "nshare" in expected:
        actual["nshare"] = int(attrs.get("hmfp_act_nshare", GROUP_SIZE))
    if actual != expected:
        raise ValueError(
            "Routed-expert LoRA activation quantization must match its "
            f"MoeBlock: expected {expected}, got {actual}"
        )


def _resolve_lora_quant_override(
    node,
    quantization: LoRAQuantization,
    inherited: Mapping[str, AlignedLoRAQuantSpec],
) -> AlignedLoRAQuantSpec | None:
    if quantization.hmfp_spec is None:
        return None
    _validate_lora_activation_quantization(node, quantization)
    base = inherited["gate"]
    values = quantization.hmfp_spec
    spec = AlignedLoRAQuantSpec(
        mode=str(values["mode"]),
        nshare=int(values.get("nshare", base.nshare)),
        man_bit=int(values["man_bit"]),
        hidden_bit=bool(values["hidden_bit"]),
        rounding=base.rounding,
        exp_bit=base.exp_bit,
    )
    spec.validate()
    return spec


def build_routed_expert_lora_bindings(
    graph: onnx.GraphProto,
    max_layers: int,
    quantization: LoRAQuantization,
) -> tuple[RoutedExpertLoRABinding, ...]:
    """Capture every expected MoeBlock and its effective LoRA format."""

    bindings = []
    for layer_idx in range(max_layers):
        node = moeblock_node_for_layer(graph, layer_idx)
        if node is None:
            raise ValueError(f"Missing MoeBlock for layer {layer_idx}")
        if len(node.input) not in {
            BASE_MOEBLOCK_INPUT_COUNT,
            BASE_MOEBLOCK_WITH_S_SCALAR_INPUT_COUNT,
        }:
            raise ValueError(
                f"Layer {layer_idx} base MoeBlock must have "
                f"{BASE_MOEBLOCK_INPUT_COUNT} or "
                f"{BASE_MOEBLOCK_WITH_S_SCALAR_INPUT_COUNT} inputs, got "
                f"{len(node.input)}"
            )
        inherited = routed_lora_quant_specs(node)
        quant_override = _resolve_lora_quant_override(
            node,
            quantization,
            inherited,
        )
        specs = (
            inherited
            if quant_override is None
            else {name: quant_override for name in inherited}
        )
        bindings.append(
            RoutedExpertLoRABinding(
                layer_idx=layer_idx,
                projection_specs=tuple(sorted(specs.items())),
                quant_override=quant_override,
            )
        )
    return tuple(bindings)


def routed_expert_lora_input_names(
    layer_idx: int,
) -> dict[str, tuple[str, str]]:
    prefix = f"layer{layer_idx}_"
    return {
        key: (
            f"{prefix}{key.lower()}_qweight",
            f"{prefix}{key.lower()}_scale",
        )
        for key in ROUTED_EXPERT_LORA_WEIGHT_KEYS
    }


def _set_attribute(node, name: str, value) -> None:
    retained = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(retained)
    node.attribute.append(helper.make_attribute(name, value))


def _set_lora_quant_attributes(
    node,
    quant_spec: AlignedLoRAQuantSpec | None,
) -> None:
    if quant_spec is None:
        return
    values = {
        "lora_mode": quant_spec.mode,
        "lora_weight_exp_bit": quant_spec.exp_bit,
        "lora_weight_man_bit": quant_spec.man_bit,
        "lora_weight_nshare": quant_spec.nshare,
        "lora_weight_rounding": quant_spec.rounding,
        "lora_weight_hidden_bit": quant_spec.hidden_bit,
    }
    for name in LORA_QUANT_ATTRIBUTES:
        _set_attribute(node, name, values[name])


def inject_routed_expert_lora_constants(
    graph,
    layer_idx: int,
    canonical_references: Mapping[str, onnx.TensorProto],
    scaling: float,
    quant_spec: AlignedLoRAQuantSpec | None = None,
) -> int:
    """Append canonical LoRA initializers to one 28-input MoeBlock."""
    node = moeblock_node_for_layer(graph, layer_idx)
    if node is None:
        return 0
    if len(node.input) not in {
        BASE_MOEBLOCK_INPUT_COUNT,
        BASE_MOEBLOCK_WITH_S_SCALAR_INPUT_COUNT,
    }:
        raise ValueError(
            f"Layer {layer_idx} base MoeBlock must have "
            f"{BASE_MOEBLOCK_INPUT_COUNT} or "
            f"{BASE_MOEBLOCK_WITH_S_SCALAR_INPUT_COUNT} inputs, got "
            f"{len(node.input)}"
        )
    if len(node.input) == BASE_MOEBLOCK_INPUT_COUNT:
        node.input.append("")

    initializer_names = {initializer.name for initializer in graph.initializer}
    names = routed_expert_lora_input_names(layer_idx)
    for key in ROUTED_EXPERT_LORA_WEIGHT_KEYS:
        for name in names[key]:
            reference = canonical_references.get(name)
            if reference is None:
                raise ValueError(f"Canonical LoRA pack is missing {name}")
            if name not in initializer_names:
                graph.initializer.add().CopyFrom(reference)
                initializer_names.add(name)
            node.input.append(name)

    if len(node.input) != LORA_MOEBLOCK_INPUT_COUNT:
        raise AssertionError(
            f"Layer {layer_idx} MoeBlock expected 28 inputs, "
            f"got {len(node.input)}"
        )
    _set_attribute(
        node,
        "lora_scaling",
        scaling,
    )
    _set_lora_quant_attributes(node, quant_spec)
    return 1


def write_routed_expert_lora_constants(
    constant_bundle: MacaronLoRAConstantBundle,
    adapters: MacaronLoRAAdapters,
    bindings: Sequence[RoutedExpertLoRABinding],
) -> None:
    """Quantize and append routed-expert constants one layer at a time."""

    for binding in bindings:
        layer_idx = binding.layer_idx
        packed = pack_expert_lora_layer(adapters, layer_idx)
        if set(packed) != set(ROUTED_EXPERT_LORA_WEIGHT_KEYS):
            raise ValueError("Unexpected routed expert LoRA keys for " f"layer {layer_idx}: {sorted(packed)}")
        names = routed_expert_lora_input_names(layer_idx)
        for key in ROUTED_EXPERT_LORA_WEIGHT_KEYS:
            tensor = packed[key]
            if tensor.ndim != 4 or tensor.shape[0] != len(ADAPTER_NAMES):
                raise ValueError(
                    f"layer {layer_idx}:{key} must be [adapter, expert, "
                    f"rows, columns], got {tuple(tensor.shape)}"
                )
            projection = key.split("_")[1]
            quant_spec = binding.quant_specs[projection]
            qweights: dict[str, np.ndarray] = {}
            scales: dict[str, np.ndarray] = {}
            for adapter_index, adapter_name in enumerate(ADAPTER_NAMES):
                qweight, scale = quantize_routed_expert_lora_weight(
                    tensor[adapter_index].float().numpy(),
                    quant_spec=quant_spec,
                )
                qweights[adapter_name] = qweight
                scales[adapter_name] = scale
            qweight_name, scale_name = names[key]
            constant_bundle.add(qweight_name, qweights)
            constant_bundle.add(scale_name, scales)
        del packed


def bind_routed_expert_lora_graph(
    graph: onnx.GraphProto,
    bindings: Sequence[RoutedExpertLoRABinding],
    canonical_references: Mapping[str, onnx.TensorProto],
    *,
    scaling: float,
) -> int:
    """Bind every planned layer to the 28-input MoeBlock LoRA ABI."""

    modified = sum(
        inject_routed_expert_lora_constants(
            graph,
            binding.layer_idx,
            canonical_references,
            scaling,
            quant_spec=binding.quant_override,
        )
        for binding in bindings
    )
    if modified != len(bindings):
        raise ValueError(
            f"Bound {modified} routed-expert LoRA layers, "
            f"expected {len(bindings)}"
        )
    return modified


__all__ = [
    "BASE_MOEBLOCK_INPUT_COUNT",
    "BASE_MOEBLOCK_WITH_S_SCALAR_INPUT_COUNT",
    "LORA_MOEBLOCK_INPUT_COUNT",
    "LORA_QUANT_ATTRIBUTES",
    "ROUTED_EXPERT_LORA_WEIGHT_KEYS",
    "RoutedExpertLoRABinding",
    "bind_routed_expert_lora_graph",
    "build_routed_expert_lora_bindings",
    "inject_routed_expert_lora_constants",
    "moeblock_node_for_layer",
    "quantize_routed_expert_lora_weight",
    "routed_expert_lora_input_names",
    "routed_lora_quant_specs",
    "write_routed_expert_lora_constants",
]
