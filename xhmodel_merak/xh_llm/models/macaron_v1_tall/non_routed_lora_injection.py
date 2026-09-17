"""Plan, quantize, and bind non-routed Macaron LoRA Linear branches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper

from .lora_constant_bundle import MacaronLoRAConstantBundle
from .lora_quantization import (
    AlignedLoRAQuantSpec,
    quantize_aligned_lora_weight,
)
from .lora_weights import ADAPTER_NAMES, MacaronLoRAAdapters


DOMAIN = "ai.houmo.xh2a"

# Macaron's V projection is twice hidden size. Row bounds are expressed as
# hidden-size multipliers and resolved from the HF model config.
FUSED_MODULES = {
    "linear_attn.in_proj_qkv": (
        ("linear_attn.in_proj_q", 0, 1),
        ("linear_attn.in_proj_k", 1, 2),
        ("linear_attn.in_proj_v", 2, 4),
    ),
}

DIRECT_MODULES = {
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.out_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
}

ROUTED_MODULES = {
    "mlp.experts.base_layer",
    "mlp.experts",
}


@dataclass(frozen=True)
class NonRoutedLoRATarget:
    """One HF adapter pair mapped to one base HMONNX Linear."""

    qweight_name: str
    lora_key: str
    b_rows: tuple[int, int] | None

    @property
    def a_qweight_name(self) -> str:
        safe_key = self.lora_key.replace(".", "_")
        return f"lora_{safe_key}_a_qweight"

    @property
    def a_scale_name(self) -> str:
        safe_key = self.lora_key.replace(".", "_")
        return f"lora_{safe_key}_a_scale"

    @property
    def b_qweight_name(self) -> str:
        safe_qweight = self.qweight_name.replace(".", "_")
        return f"lora_{safe_qweight}_b_qweight"

    @property
    def b_scale_name(self) -> str:
        safe_qweight = self.qweight_name.replace(".", "_")
        return f"lora_{safe_qweight}_b_scale"

    @property
    def constant_names(self) -> tuple[str, str, str, str]:
        return (
            self.a_qweight_name,
            self.a_scale_name,
            self.b_qweight_name,
            self.b_scale_name,
        )


@dataclass(frozen=True)
class NonRoutedLoRABinding:
    """Graph-independent binding data validated for both export phases."""

    target: NonRoutedLoRATarget
    a_shape: tuple[int, int]
    b_shape: tuple[int, int]
    quant_spec: AlignedLoRAQuantSpec
    linear_attributes: tuple[tuple[str, object], ...]

    @property
    def attributes(self) -> dict[str, object]:
        return dict(self.linear_attributes)


def quantize_non_routed_lora_weight(
    weight: np.ndarray,
    *,
    dim: int = 1,
    mode: str,
    nshare: int,
    man_bit: int,
    hidden_bit: bool,
    rounding: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize LoRA with the same aligned HMFP path as a base Linear."""

    return quantize_aligned_lora_weight(
        weight,
        dim=dim,
        spec=AlignedLoRAQuantSpec(
            mode=mode,
            nshare=nshare,
            man_bit=man_bit,
            hidden_bit=hidden_bit,
            rounding=rounding,
        ),
    )


def _as_node_name(output_name: str) -> str:
    return "".join(character if character.isalnum() or character == "_" else "_" for character in output_name)


def _split_module_key(lora_key: str) -> tuple[int, str]:
    parts = lora_key.split(".")
    if len(parts) < 4 or parts[:2] != ["model", "layers"]:
        raise ValueError(f"Unsupported Macaron LoRA module key: {lora_key}")
    try:
        layer_idx = int(parts[2])
    except ValueError as error:
        raise ValueError(f"Invalid layer index in Macaron LoRA module key: {lora_key}") from error
    return layer_idx, ".".join(parts[3:])


def build_non_routed_lora_targets(
    adapters: MacaronLoRAAdapters,
) -> tuple[NonRoutedLoRATarget, ...]:
    """Resolve every supported non-routed PEFT pair to an HMONNX weight."""

    targets: list[NonRoutedLoRATarget] = []
    qweight_names: set[str] = set()
    for lora_key in adapters.module_keys():
        layer_idx, module_suffix = _split_module_key(lora_key)
        if module_suffix in ROUTED_MODULES:
            continue
        if module_suffix in FUSED_MODULES:
            hidden_size = adapters.model_shape.hidden_size
            onnx_targets = tuple(
                (
                    suffix,
                    (lower * hidden_size, upper * hidden_size),
                )
                for suffix, lower, upper in FUSED_MODULES[module_suffix]
            )
        elif module_suffix in DIRECT_MODULES:
            onnx_targets = ((module_suffix, None),)
        else:
            raise ValueError(f"Unsupported non-routed Macaron LoRA target: {lora_key}")

        for onnx_suffix, b_rows in onnx_targets:
            qweight_name = f"model.language_model.layers.{layer_idx}." f"{onnx_suffix}.qweight"
            if qweight_name in qweight_names:
                raise ValueError(f"Duplicate LoRA ONNX target: {qweight_name}")
            targets.append(
                NonRoutedLoRATarget(
                    qweight_name=qweight_name,
                    lora_key=lora_key,
                    b_rows=b_rows,
                )
            )
            qweight_names.add(qweight_name)
    return tuple(targets)


def _linear_attributes(
    node: onnx.NodeProto,
    quant_override: Mapping[str, object] | None,
) -> tuple[AlignedLoRAQuantSpec, tuple[tuple[str, object], ...]]:
    attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
    mode = attributes.get("mode", b"ssfp")
    if isinstance(mode, bytes):
        mode = mode.decode()
    nshare = int(attributes.get("hmfp_weight_nshare", 64))
    man_bit = int(attributes.get("hmfp_weight_man_bit", 8))
    hidden_bit = bool(
        attributes.get(
            "hmfp_weight_hidden_bit",
            0 if mode == "ssfp" else 1,
        )
    )
    rounding = attributes.get("hmfp_weight_rounding", b"RNE")
    if isinstance(rounding, bytes):
        rounding = rounding.decode()
    if quant_override:
        mode = str(quant_override.get("mode", mode))
        nshare = int(quant_override.get("nshare", nshare))
        man_bit = int(quant_override.get("man_bit", man_bit))
        hidden_bit = bool(quant_override.get("hidden_bit", hidden_bit))
        rounding = str(quant_override.get("rounding", rounding))
    if mode == "ssfp" and hidden_bit:
        raise ValueError("ssfp LoRA weights require hidden_bit=false")

    linear_attributes = {
        key: value for key, value in attributes.items() if key.startswith("hmfp_") or key in {"mode", "output_dtype"}
    }
    defaults = {
        "mode": mode,
        "hmfp_act_exp_bit": 5,
        "hmfp_act_man_bit": 8,
        "hmfp_act_nshare": 64,
        "hmfp_act_rounding": "RNE",
        "hmfp_act_hidden_bit": 1,
        "hmfp_act_max_exp_boost": 0,
        "hmfp_weight_exp_bit": 5,
        "hmfp_weight_man_bit": man_bit,
        "hmfp_weight_nshare": nshare,
        "hmfp_weight_rounding": "RNE",
        "hmfp_weight_hidden_bit": 0 if mode == "ssfp" else 1,
        "hmfp_psum_dtype": "fp24",
        "hmfp_psum_round_mode": "trunc",
        "output_dtype": "float16",
    }
    for name, value in defaults.items():
        linear_attributes.setdefault(name, value)
    linear_attributes.update(
        {
            "mode": mode,
            "hmfp_weight_nshare": nshare,
            "hmfp_weight_man_bit": man_bit,
            "hmfp_weight_hidden_bit": int(hidden_bit),
            "hmfp_weight_rounding": rounding,
        }
    )
    quant_spec = AlignedLoRAQuantSpec(
        mode=mode,
        nshare=nshare,
        man_bit=man_bit,
        hidden_bit=hidden_bit,
        rounding=rounding,
    )
    return quant_spec, tuple(sorted(linear_attributes.items()))


def build_non_routed_lora_bindings(
    graph: onnx.GraphProto,
    targets: Sequence[NonRoutedLoRATarget],
    *,
    rank: int,
    quant_override: Mapping[str, object] | None = None,
) -> tuple[NonRoutedLoRABinding, ...]:
    """Validate target coverage and capture phase-independent Linear data."""

    targets_by_qweight = {target.qweight_name: target for target in targets}
    bindings: list[NonRoutedLoRABinding] = []
    matched: set[str] = set()
    for node in graph.node:
        if node.op_type != "Linear" or len(node.input) < 2:
            continue
        target = targets_by_qweight.get(node.input[1])
        if target is None:
            continue
        if target.qweight_name in matched:
            raise ValueError(f"Duplicate base Linear for LoRA target: {target.qweight_name}")
        quant_spec, linear_attributes = _linear_attributes(node, quant_override)
        attributes = dict(linear_attributes)
        in_features = int(attributes.get("in_features", 0))
        out_features = int(attributes.get("out_features", 0))
        source_attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
        in_features = int(source_attributes.get("in_features", in_features))
        out_features = int(source_attributes.get("out_features", out_features))
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                "LoRA base Linear must define positive in_features and " f"out_features: {target.qweight_name}"
            )
        bindings.append(
            NonRoutedLoRABinding(
                target=target,
                a_shape=(in_features, rank),
                b_shape=(rank, out_features),
                quant_spec=quant_spec,
                linear_attributes=linear_attributes,
            )
        )
        matched.add(target.qweight_name)
    missing = set(targets_by_qweight) - matched
    if missing:
        raise ValueError(
            "Base graph is missing non-routed LoRA Linear targets: " f"{sorted(missing)[:3]} ({len(missing)} total)"
        )
    return tuple(bindings)


def _linear_lora_values(
    adapters: MacaronLoRAAdapters,
    binding: NonRoutedLoRABinding,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    target = binding.target
    values = {}
    for adapter_name in ADAPTER_NAMES:
        a_weight, b_weight = adapters.pair(adapter_name, target.lora_key)
        if target.b_rows is not None:
            b_weight = b_weight[slice(*target.b_rows), :]
        a_value = a_weight.float().numpy().T.astype(np.float16)
        b_value = (b_weight.float().numpy().T * adapters.scaling).astype(np.float16)
        if a_value.shape != binding.a_shape or b_value.shape != binding.b_shape:
            raise ValueError(
                f"{adapter_name}:{target.lora_key} shape differs from "
                f"base Linear: A={a_value.shape}, B={b_value.shape}, "
                f"expected A={binding.a_shape}, B={binding.b_shape}"
            )
        values[adapter_name] = (a_value, b_value)
    return values


def _quantize_by_adapter(
    values: Mapping[str, np.ndarray],
    spec: AlignedLoRAQuantSpec,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    qweights = {}
    scales = {}
    for adapter_name in ADAPTER_NAMES:
        qweights[adapter_name], scales[adapter_name] = quantize_non_routed_lora_weight(
            values[adapter_name],
            dim=0,
            mode=spec.mode,
            nshare=spec.nshare,
            man_bit=spec.man_bit,
            hidden_bit=spec.hidden_bit,
            rounding=spec.rounding,
        )
    return qweights, scales


def write_non_routed_lora_constants(
    bundle: MacaronLoRAConstantBundle,
    adapters: MacaronLoRAAdapters,
    bindings: Sequence[NonRoutedLoRABinding],
) -> None:
    """Write each non-routed initializer once for all four adapters."""

    shared_a_specs: dict[str, AlignedLoRAQuantSpec] = {}
    for binding in bindings:
        target = binding.target
        values = _linear_lora_values(adapters, binding)
        previous_spec = shared_a_specs.get(target.a_qweight_name)
        if previous_spec is not None and previous_spec != binding.quant_spec:
            raise ValueError("Shared LoRA A is used with different quantization formats: " f"{target.a_qweight_name}")
        if not bundle.has(target.a_qweight_name):
            qweights, scales = _quantize_by_adapter(
                {name: values[name][0] for name in ADAPTER_NAMES},
                binding.quant_spec,
            )
            bundle.add(target.a_qweight_name, qweights)
            bundle.add(target.a_scale_name, scales)
            shared_a_specs[target.a_qweight_name] = binding.quant_spec
        qweights, scales = _quantize_by_adapter(
            {name: values[name][1] for name in ADAPTER_NAMES},
            binding.quant_spec,
        )
        bundle.add(target.b_qweight_name, qweights)
        bundle.add(target.b_scale_name, scales)


def _append_initializer(
    graph: onnx.GraphProto,
    reference: TensorProto,
    initializer_names: set[str],
) -> None:
    if reference.name in initializer_names:
        return
    graph.initializer.add().CopyFrom(reference)
    initializer_names.add(reference.name)


def bind_non_routed_lora_graph(
    graph: onnx.GraphProto,
    bindings: Sequence[NonRoutedLoRABinding],
    canonical_references: Mapping[str, TensorProto],
) -> int:
    """Add two Linear nodes and one Add node for every planned target."""

    # Import the SDK opset before querying its registered schema. Older XH2a
    # Add schemas have no saturate switch, so retain their fixed SDK behavior.
    # Newer schemas require an explicit value; disable saturation there.
    from xhquant.export.onnx.onnx_opset import xh2a_opset1

    add_schema = onnx.defs.get_schema("Add", xh2a_opset1.version, DOMAIN)
    add_attributes = {"saturate": 0} if "saturate" in add_schema.attributes else {}

    bindings_by_qweight = {binding.target.qweight_name: binding for binding in bindings}
    rewritten_nodes: list[onnx.NodeProto] = []
    shared_a_outputs: dict[tuple[str, str], str] = {}
    initializer_names = {initializer.name for initializer in graph.initializer}
    modified = 0

    for node in graph.node:
        qweight_name = node.input[1] if len(node.input) >= 2 else None
        binding = bindings_by_qweight.get(qweight_name)
        if node.op_type != "Linear" or binding is None:
            rewritten_nodes.append(node)
            continue

        target = binding.target
        for name in target.constant_names:
            reference = canonical_references.get(name)
            if reference is None:
                raise ValueError(f"Canonical LoRA pack is missing initializer {name}")
            _append_initializer(graph, reference, initializer_names)

        activation = node.input[0]
        original_output = node.output[0]
        pre_lora_output = f"{original_output}_pre_lora"
        node.output[0] = pre_lora_output
        rewritten_nodes.append(node)

        safe_key = target.lora_key.replace(".", "_")
        safe_qweight = target.qweight_name.replace(".", "_")
        shared_a_key = (activation, target.a_qweight_name)
        linear_a_output = shared_a_outputs.get(shared_a_key)
        if linear_a_output is None:
            linear_a_output = f"lora_{safe_key}_linear_a"
            rewritten_nodes.append(
                helper.make_node(
                    "Linear",
                    [
                        activation,
                        target.a_qweight_name,
                        target.a_scale_name,
                        "",
                        "",
                    ],
                    [linear_a_output],
                    name=_as_node_name(linear_a_output),
                    domain=DOMAIN,
                    have_bias=0,
                    in_features=binding.a_shape[0],
                    out_features=binding.a_shape[1],
                    **binding.attributes,
                )
            )
            shared_a_outputs[shared_a_key] = linear_a_output

        linear_b_output = f"lora_{safe_qweight}_linear_b"
        rewritten_nodes.append(
            helper.make_node(
                "Linear",
                [
                    linear_a_output,
                    target.b_qweight_name,
                    target.b_scale_name,
                    "",
                    "",
                ],
                [linear_b_output],
                name=_as_node_name(linear_b_output),
                domain=DOMAIN,
                have_bias=0,
                in_features=binding.b_shape[0],
                out_features=binding.b_shape[1],
                **binding.attributes,
            )
        )
        rewritten_nodes.append(
            helper.make_node(
                "Add",
                [pre_lora_output, linear_b_output],
                [original_output],
                name=_as_node_name(original_output),
                domain=DOMAIN,
                **add_attributes,
            )
        )
        modified += 1

    if modified != len(bindings):
        raise ValueError(f"Bound {modified} non-routed LoRA targets, " f"expected {len(bindings)}")
    del graph.node[:]
    graph.node.extend(rewritten_nodes)
    return modified


__all__ = [
    "NonRoutedLoRABinding",
    "NonRoutedLoRATarget",
    "bind_non_routed_lora_graph",
    "build_non_routed_lora_bindings",
    "build_non_routed_lora_targets",
    "quantize_non_routed_lora_weight",
    "write_non_routed_lora_constants",
]
