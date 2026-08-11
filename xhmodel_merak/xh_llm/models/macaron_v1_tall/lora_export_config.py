"""Read Macaron LoRA export settings from an effective workflow config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from xhquant.api import DeviceType, QuantScheme, create_quant_config

from .lora_weights import ADAPTER_NAMES


@dataclass(frozen=True)
class LoRAQuantization:
    """One optional LoRA weight format."""

    quant_type: str | None
    hmfp_spec: dict[str, Any] | None
    act_spec: dict[str, Any] | None


@dataclass(frozen=True)
class MacaronLoRAExportConfig:
    """LoRA settings consumed while producing the final Macaron export."""

    adapters: tuple[str, ...]
    non_routed_quantization: LoRAQuantization
    routed_expert_quantization: LoRAQuantization

    def manifest_config(self) -> dict[str, Any]:
        return {
            "canonical_adapter": "L0",
            "adapters": list(self.adapters),
            "quantization": {
                "non_routed": {
                    "quant_scheme": (self.non_routed_quantization.quant_type or "inherit"),
                },
                "routed_expert": {
                    "quant_scheme": (
                        self.routed_expert_quantization.quant_type
                        or "inherit"
                    ),
                },
            },
        }


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _require_only_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    path: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{path} has unsupported keys: {sorted(unknown)}")


def _parse_quant_scheme(value: Any, path: str) -> LoRAQuantization:
    if value in (None, "inherit"):
        return LoRAQuantization(
            quant_type=None,
            hmfp_spec=None,
            act_spec=None,
        )
    if not isinstance(value, str):
        raise TypeError(f"{path} must be 'inherit' or a quant_type string")
    try:
        config = create_quant_config(
            QuantScheme(
                target_device=DeviceType.XH2a,
                quant_type=value,
            )
        )
        config_dict = config.to_dict()
        weight = config_dict["w_schema"]
        activation = config_dict["act_schema"]
    except Exception as error:
        raise ValueError(f"{path} is not a supported XH2a quant_type: {value!r}") from error

    mode = str(weight.fp_mode)
    hidden_bit = bool(weight.hidden_bit)
    if mode not in {"sefp", "ssfp"}:
        raise ValueError(f"{path} must select sefp or ssfp, got {mode!r}")
    if mode == "ssfp" and hidden_bit:
        raise ValueError(f"{path} cannot use ssfp with hidden bit enabled")

    spec = {
        "mode": mode,
        "man_bit": int(weight.man_bit),
        "hidden_bit": hidden_bit,
    }
    if weight.nshare is not None:
        spec["nshare"] = int(weight.nshare)
    act_spec = {
        "mode": str(activation.fp_mode),
        "man_bit": int(activation.man_bit),
        "hidden_bit": bool(activation.hidden_bit),
    }
    if activation.nshare is not None:
        act_spec["nshare"] = int(activation.nshare)
    return LoRAQuantization(
        quant_type=value,
        hmfp_spec=spec,
        act_spec=act_spec,
    )


def _parse_adapters(
    adapter_config: Mapping[str, Any],
) -> tuple[str, ...]:
    _require_only_keys(
        adapter_config,
        allowed={"available"},
        path="macaron_lora.adapters",
    )
    adapters = tuple(adapter_config.get("available", ()))
    if adapters != ADAPTER_NAMES:
        raise ValueError(f"macaron_lora.adapters.available must be {list(ADAPTER_NAMES)}")
    return adapters


def _parse_quantization(
    quantization: Mapping[str, Any],
) -> tuple[LoRAQuantization, LoRAQuantization]:
    non_routed_path = "macaron_lora.quantization.non_routed"
    non_routed = _require_mapping(
        quantization.get("non_routed"),
        non_routed_path,
    )
    _require_only_keys(
        non_routed,
        allowed={"quant_scheme"},
        path=non_routed_path,
    )

    routed_path = "macaron_lora.quantization.routed_expert"
    routed = _require_mapping(quantization.get("routed_expert"), routed_path)
    _require_only_keys(
        routed,
        allowed={"quant_scheme"},
        path=routed_path,
    )
    return (
        _parse_quant_scheme(
            non_routed.get("quant_scheme", "inherit"),
            f"{non_routed_path}.quant_scheme",
        ),
        _parse_quant_scheme(
            routed.get("quant_scheme", "inherit"),
            f"{routed_path}.quant_scheme",
        ),
    )


def load_macaron_lora_export_config(
    config: Mapping[str, Any],
) -> MacaronLoRAExportConfig:
    """Parse Macaron settings from the effective workflow config."""

    data = _require_mapping(config, "workflow config")
    section = _require_mapping(data.get("macaron_lora"), "macaron_lora")
    adapters = _parse_adapters(_require_mapping(section.get("adapters"), "macaron_lora.adapters"))
    non_routed, routed_expert = _parse_quantization(
        _require_mapping(
            section.get("quantization"),
            "macaron_lora.quantization",
        )
    )
    return MacaronLoRAExportConfig(
        adapters=adapters,
        non_routed_quantization=non_routed,
        routed_expert_quantization=routed_expert,
    )


__all__ = [
    "LoRAQuantization",
    "MacaronLoRAExportConfig",
    "load_macaron_lora_export_config",
]
