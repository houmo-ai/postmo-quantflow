"""Build an immutable LoRA plan before mutating a Qwen base export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx

from .lora_export_config import MacaronLoRAExportConfig
from .lora_weights import MacaronLoRAAdapters
from .non_routed_lora_injection import (
    NonRoutedLoRABinding,
    build_non_routed_lora_bindings,
    build_non_routed_lora_targets,
)
from .routed_expert_lora_injection import (
    ROUTED_EXPERT_LORA_WEIGHT_KEYS,
    RoutedExpertLoRABinding,
    build_routed_expert_lora_bindings,
    routed_expert_lora_input_names,
)


PHASES = ("prefill", "decode")


@dataclass(frozen=True)
class MacaronLoRAExportPlan:
    """All validated decisions needed to build and bind a LoRA bundle."""

    graph_paths: tuple[tuple[str, Path], ...]
    non_routed: tuple[NonRoutedLoRABinding, ...]
    routed_expert: tuple[RoutedExpertLoRABinding, ...]

    @property
    def initializer_order(self) -> tuple[str, ...]:
        names: list[str] = []
        seen: set[str] = set()
        for binding in self.non_routed:
            for name in binding.target.constant_names:
                if name not in seen:
                    names.append(name)
                    seen.add(name)
        for binding in self.routed_expert:
            layer_names = routed_expert_lora_input_names(binding.layer_idx)
            for key in ROUTED_EXPERT_LORA_WEIGHT_KEYS:
                for name in layer_names[key]:
                    names.append(name)
        return tuple(names)


def _find_phase_graph(export_dir: Path, phase: str) -> Path:
    candidates = sorted((export_dir / phase).glob("*_with_act.onnx"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"{phase}: expected one *_with_act.onnx graph, " f"found {[path.name for path in candidates]}"
        )
    return candidates[0]


def _routed_module_keys(num_layers: int) -> set[str]:
    return {
        f"model.layers.{layer_idx}.mlp.experts{suffix}"
        for layer_idx in range(num_layers)
        for suffix in (".base_layer", "")
    }


def _validate_adapter_coverage(
    adapters: MacaronLoRAAdapters,
    non_routed: tuple[NonRoutedLoRABinding, ...],
) -> None:
    consumed = {binding.target.lora_key for binding in non_routed} | _routed_module_keys(
        adapters.model_shape.num_layers
    )
    available = set(adapters.module_keys())
    missing = consumed - available
    unexpected = available - consumed
    if missing or unexpected:
        raise ValueError(
            "Macaron LoRA target coverage is incomplete: "
            f"missing={sorted(missing)[:3]}, "
            f"unexpected={sorted(unexpected)[:3]}"
        )


def build_macaron_lora_export_plan(
    export_dir: str | Path,
    adapters: MacaronLoRAAdapters,
    export_config: MacaronLoRAExportConfig,
) -> MacaronLoRAExportPlan:
    """Validate Prefill/Decode and return one shared export plan."""

    export_dir = Path(export_dir)
    targets = build_non_routed_lora_targets(adapters)
    graph_paths = tuple((phase, _find_phase_graph(export_dir, phase)) for phase in PHASES)
    phase_plans = {}
    for phase, graph_path in graph_paths:
        graph = onnx.load(str(graph_path), load_external_data=False).graph
        phase_plans[phase] = (
            build_non_routed_lora_bindings(
                graph,
                targets,
                rank=adapters.rank,
                quant_override=(export_config.non_routed_quantization.hmfp_spec),
            ),
            build_routed_expert_lora_bindings(
                graph,
                adapters.model_shape.num_layers,
                export_config.routed_expert_quantization,
            ),
        )

    prefill_non_routed, prefill_routed = phase_plans["prefill"]
    decode_non_routed, decode_routed = phase_plans["decode"]
    if prefill_non_routed != decode_non_routed:
        raise ValueError("Prefill and Decode non-routed LoRA bindings differ")
    if prefill_routed != decode_routed:
        raise ValueError("Prefill and Decode routed-expert LoRA bindings differ")
    _validate_adapter_coverage(adapters, prefill_non_routed)
    plan = MacaronLoRAExportPlan(
        graph_paths=graph_paths,
        non_routed=prefill_non_routed,
        routed_expert=prefill_routed,
    )
    if len(plan.initializer_order) != len(set(plan.initializer_order)):
        raise ValueError("Macaron LoRA plan has duplicate initializer names")
    return plan


__all__ = [
    "MacaronLoRAExportPlan",
    "build_macaron_lora_export_plan",
]
