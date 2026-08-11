"""Validate and stream Macaron PEFT LoRA checkpoints."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


ADAPTER_NAMES = ("L0", "L1", "L2", "L3")
DEPLOYMENT_DTYPE = torch.float16
_EXPERT_TARGET_PARAMETERS = {
    "experts.gate_up_proj",
    "experts.down_proj",
}


@dataclass(frozen=True)
class MacaronModelShape:
    num_layers: int
    num_experts: int
    hidden_size: int
    moe_intermediate_size: int


@dataclass(frozen=True)
class MacaronLoRAConfig:
    rank: int
    scaling: float


@dataclass(frozen=True)
class _LoRAPair:
    a_key: str
    b_key: str


def _module_and_side(key: str) -> tuple[str, str]:
    for side in ("A", "B"):
        suffix = f".lora_{side}.weight"
        if key.endswith(suffix):
            return (
                key.removeprefix("base_model.model.").removesuffix(suffix),
                side,
            )
    raise ValueError(f"Unsupported non-LoRA safetensors key: {key}")


def _read_model_shape(model_dir: Path) -> MacaronModelShape:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing HF model config: {config_path}")
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = model_config.get("text_config")
    if not isinstance(text_config, Mapping):
        raise TypeError(f"Macaron config must contain text_config: {config_path}")

    def require_positive(name: str) -> int:
        value = int(text_config.get(name, 0))
        if value <= 0:
            raise ValueError(
                f"Macaron text_config.{name} must be positive: {config_path}"
            )
        return value

    return MacaronModelShape(
        num_layers=require_positive("num_hidden_layers"),
        num_experts=require_positive("num_experts"),
        hidden_size=require_positive("hidden_size"),
        moe_intermediate_size=require_positive("moe_intermediate_size"),
    )


def _read_adapter_config(adapter_dir: Path) -> MacaronLoRAConfig:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing PEFT adapter config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError(f"PEFT adapter config must be an object: {config_path}")
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"Only PEFT LoRA adapters are supported: {config_path}")
    if str(config.get("bias", "none")).lower() != "none":
        raise ValueError(f"Macaron LoRA bias must be none: {config_path}")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError(
            f"Macaron does not support PEFT rank/alpha patterns: {config_path}"
        )
    target_parameters = set(config.get("target_parameters") or ())
    if target_parameters != _EXPERT_TARGET_PARAMETERS:
        raise ValueError(
            "Macaron routed LoRA target_parameters must be "
            f"{sorted(_EXPERT_TARGET_PARAMETERS)}: {config_path}"
        )
    rank = int(config.get("r", 0))
    alpha = float(config.get("lora_alpha", 0.0))
    if rank <= 0 or alpha <= 0:
        raise ValueError(
            f"Macaron LoRA r and lora_alpha must be positive: {config_path}"
        )
    return MacaronLoRAConfig(rank=rank, scaling=alpha / rank)


class MacaronLoRAAdapters:
    """Validated, memory-mapped L0--L3 checkpoints for one export."""

    def __init__(
        self,
        readers: Mapping[str, Any],
        *,
        config: MacaronLoRAConfig,
        model_shape: MacaronModelShape,
    ) -> None:
        self._readers = dict(readers)
        if tuple(self._readers) != ADAPTER_NAMES:
            raise ValueError(
                "Macaron LoRA adapters must be exactly "
                f"{list(ADAPTER_NAMES)}, got {sorted(self._readers)}"
            )
        self.config = config
        self.model_shape = model_shape
        self._pairs = {
            name: self._index_adapter(name, reader)
            for name, reader in self._readers.items()
        }
        reference = self._pairs[ADAPTER_NAMES[0]]
        for name, pairs in self._pairs.items():
            if set(pairs) != set(reference):
                raise ValueError(
                    f"{name}: LoRA module keys differ from {ADAPTER_NAMES[0]}"
                )

    @staticmethod
    def _index_adapter(
        adapter_name: str,
        reader: Any,
    ) -> dict[str, _LoRAPair]:
        sides: dict[str, dict[str, str]] = {}
        for key in reader.keys():
            module, side = _module_and_side(key)
            sides.setdefault(module, {})[side] = key
        incomplete = {
            module: tuple(sorted(pair))
            for module, pair in sides.items()
            if set(pair) != {"A", "B"}
        }
        if incomplete:
            raise ValueError(
                f"{adapter_name}: incomplete LoRA A/B pairs: {incomplete}"
            )
        return {
            module: _LoRAPair(a_key=pair["A"], b_key=pair["B"])
            for module, pair in sides.items()
        }

    @property
    def rank(self) -> int:
        return self.config.rank

    @property
    def scaling(self) -> float:
        return self.config.scaling

    def module_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._pairs[ADAPTER_NAMES[0]]))

    def pair(
        self,
        adapter_name: str,
        module_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self._pairs[adapter_name].get(module_key)
        if pair is None:
            raise KeyError(f"{adapter_name}: missing LoRA module {module_key}")
        reader = self._readers[adapter_name]
        return (
            reader.get_tensor(pair.a_key).to(DEPLOYMENT_DTYPE),
            reader.get_tensor(pair.b_key).to(DEPLOYMENT_DTYPE),
        )


@contextmanager
def open_macaron_lora_adapters(
    model_dir: str | Path,
) -> Iterator[MacaronLoRAAdapters]:
    """Open all checkpoints after enforcing one shared PEFT schema."""

    model_dir = Path(model_dir)
    model_shape = _read_model_shape(model_dir)
    adapter_dirs = {name: model_dir / "loras" / name for name in ADAPTER_NAMES}
    configs = {
        name: _read_adapter_config(adapter_dir)
        for name, adapter_dir in adapter_dirs.items()
    }
    reference_config = configs[ADAPTER_NAMES[0]]
    if any(config != reference_config for config in configs.values()):
        raise ValueError("L0--L3 must use the same PEFT rank and LoRA scaling")

    with ExitStack() as stack:
        readers: dict[str, Any] = {}
        for name, adapter_dir in adapter_dirs.items():
            checkpoint = adapter_dir / "adapter_model.safetensors"
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Missing Macaron safetensors checkpoint: {checkpoint}"
                )
            readers[name] = stack.enter_context(
                safe_open(str(checkpoint), framework="pt", device="cpu")
            )
        yield MacaronLoRAAdapters(
            readers,
            config=reference_config,
            model_shape=model_shape,
        )


def _reshape_gate_up(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    *,
    shape: MacaronModelShape,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed PEFT gate/up tensors into the MoeBlock projection layout."""

    expected_a = (shape.num_experts * rank, shape.hidden_size)
    expected_b = (2 * shape.moe_intermediate_size, shape.num_experts * rank)
    if tuple(a_packed.shape) != expected_a or tuple(b_packed.shape) != expected_b:
        raise ValueError(
            "Unexpected packed gate/up LoRA shapes: "
            f"A={tuple(a_packed.shape)}, B={tuple(b_packed.shape)}, "
            f"expected A={expected_a}, B={expected_b}"
        )
    a = a_packed.reshape(shape.num_experts, rank, shape.hidden_size)
    b = b_packed.reshape(
        2 * shape.moe_intermediate_size,
        rank,
        shape.num_experts,
    ).permute(2, 0, 1).contiguous()
    return (
        a,
        b[:, : shape.moe_intermediate_size, :].contiguous(),
        a,
        b[:, shape.moe_intermediate_size :, :].contiguous(),
    )


def _reshape_down(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    *,
    shape: MacaronModelShape,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape packed PEFT down-projection tensors for MoeBlock."""

    expected_a = (shape.num_experts * rank, shape.moe_intermediate_size)
    expected_b = (shape.hidden_size, shape.num_experts * rank)
    if tuple(a_packed.shape) != expected_a or tuple(b_packed.shape) != expected_b:
        raise ValueError(
            "Unexpected packed down LoRA shapes: "
            f"A={tuple(a_packed.shape)}, B={tuple(b_packed.shape)}, "
            f"expected A={expected_a}, B={expected_b}"
        )
    return (
        a_packed.reshape(
            shape.num_experts,
            rank,
            shape.moe_intermediate_size,
        ),
        b_packed.reshape(
            shape.hidden_size,
            rank,
            shape.num_experts,
        ).permute(2, 0, 1).contiguous(),
    )


def pack_expert_lora_layer(
    adapters: MacaronLoRAAdapters,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """Read and reshape one layer of expert LoRA for all four adapters."""

    if not 0 <= layer_idx < adapters.model_shape.num_layers:
        raise ValueError(f"Invalid Macaron LoRA layer index: {layer_idx}")
    gate_up_key = f"model.layers.{layer_idx}.mlp.experts.base_layer"
    down_key = f"model.layers.{layer_idx}.mlp.experts"
    packed: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "lora_gate_A",
            "lora_gate_B",
            "lora_up_A",
            "lora_up_B",
            "lora_down_A",
            "lora_down_B",
        )
    }
    for name in ADAPTER_NAMES:
        gate_a, gate_b = adapters.pair(name, gate_up_key)
        down_a, down_b = adapters.pair(name, down_key)
        gate_a, gate_b, up_a, up_b = _reshape_gate_up(
            gate_a,
            gate_b,
            shape=adapters.model_shape,
            rank=adapters.rank,
        )
        down_a, down_b = _reshape_down(
            down_a,
            down_b,
            shape=adapters.model_shape,
            rank=adapters.rank,
        )
        for key, tensor in (
            ("lora_gate_A", gate_a),
            ("lora_gate_B", gate_b),
            ("lora_up_A", up_a),
            ("lora_up_B", up_b),
            ("lora_down_A", down_a),
            ("lora_down_B", down_b),
        ):
            packed[key].append(tensor)
    return {key: torch.stack(values) for key, values in packed.items()}


__all__ = [
    "ADAPTER_NAMES",
    "MacaronLoRAAdapters",
    "MacaronModelShape",
    "open_macaron_lora_adapters",
    "pack_expert_lora_layer",
]
