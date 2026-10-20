"""Qwen4-Exp quantization precision contract.

The policy is intentionally data-only so workflow code can inspect it without
importing Transformers 5.16+ or the external GPTQModel recipe package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_PLE_QUANT_TYPE = "sefp4"


def resolve_ple_quant_type(value: str) -> str:
    """PLE checkpoint weights remain high precision; deployment defaults to SEFP4."""
    if value not in ("sefp4", "fp16"):
        raise ValueError("Qwen4 PLE quantization must be 'sefp4' or 'fp16'")
    return value


@dataclass(frozen=True)
class PrecisionRule:
    pattern: str
    bits: int
    activation_bits: int | None
    data_type: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Qwen4ExpPrecisionProfile:
    name: str
    base_bits: int = 8
    expert_bits: int = 4
    activation_bits: int = 8
    group_size: int = 64
    sym: bool = True
    language_rotation: str = "forbidden"
    vision_rotation: str = "optional_vision_only"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


QWEN4_EXP_PROFILE = Qwen4ExpPrecisionProfile(
    name="qwen4_exp_w4a8_routed_experts_w8a8_attention_shared_safe",
)

_LAYER_PREFIX = r"(?:model\.)?(?:language_model\.)?layers\.\d+"

QWEN4_EXP_PRECISION_RULES: tuple[PrecisionRule, ...] = (
    PrecisionRule(
        pattern=rf"{_LAYER_PREFIX}\.mlp\.experts(?:\.\d+)?\.(?:gate_proj|up_proj|gate_up_proj|down_proj)$",
        bits=4,
        activation_bits=8,
        data_type="int",
        reason="Routed MoE experts dominate storage and match the requested W4A8 target.",
    ),
    PrecisionRule(
        pattern=rf"{_LAYER_PREFIX}\.mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj|gate_up_proj)$",
        bits=8,
        activation_bits=8,
        data_type="int",
        reason="Shared expert participates in every token and is kept at the requested W8A8 precision.",
    ),
    PrecisionRule(
        pattern=rf"{_LAYER_PREFIX}\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$",
        bits=8,
        activation_bits=8,
        data_type="int",
        reason="Full/QSA attention projections are token-critical and requested as W8A8.",
    ),
    PrecisionRule(
        pattern=rf"{_LAYER_PREFIX}\.linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)$",
        bits=8,
        activation_bits=8,
        data_type="int",
        reason="Gated DeltaNet main projections are requested as attention W8A8.",
    ),
    PrecisionRule(
        pattern=rf"{_LAYER_PREFIX}\.self_attn\.(?:q_norm|k_norm|indexer(?:\.|$).*)",
        bits=16,
        activation_bits=16,
        data_type="bf16",
        reason="QSA index selection and query/key normalization define sparse block routing; keep high precision.",
    ),
    PrecisionRule(
        pattern=rf"{_LAYER_PREFIX}\.linear_attn\.(?:in_proj_a|in_proj_b|conv1d|A_log|dt_bias|norm)$",
        bits=16,
        activation_bits=16,
        data_type="bf16",
        reason="DeltaNet recurrence/state parameters, conv state, and norms are numerically sensitive.",
    ),
    PrecisionRule(
        pattern=rf"{_LAYER_PREFIX}\.(?:attn_hyper_connection|mlp_hyper_connection)(?:\.|$).*",
        bits=16,
        activation_bits=16,
        data_type="bf16",
        reason="Hyper-connections use sigmoid-gated elementwise mixing and are not rotation/low-bit friendly.",
    ),
    PrecisionRule(
        pattern=rf"{_LAYER_PREFIX}\.mlp\.(?:gate|shared_expert_gate)(?:\.|$).*",
        bits=16,
        activation_bits=16,
        data_type="bf16",
        reason="Routers and shared-expert gates control expert allocation and should not be low-bit quantized.",
    ),
    PrecisionRule(
        pattern=(
            rf"(?:{_LAYER_PREFIX}\.ple|"
            r"(?:model\.)?(?:language_model\.)?hyper_connection_mixer|"
            r"(?:model\.)?(?:ple|ngram|embed_ngrams|ngram_embedding|input_embed))"
            r"(?:\.|$).*"
        ),
        bits=16,
        activation_bits=16,
        data_type="bf16",
        reason="Keep source PLE/HC weights high precision; HMONNX packs the PLE lookup as SEFP4 by default.",
    ),
    PrecisionRule(
        pattern=(
            r"(?:(?:model\.)?(?:language_model\.)?embed_tokens|"
            r"(?:model\.)?lm_head|(?:model\.)?(?:visual|vision_model|vision_tower))"
            r"(?:\.|$).*"
        ),
        bits=8,
        activation_bits=16,
        data_type="int",
        reason=(
            "Embedding, lm_head, and vision bridge paths are kept at W8A16 "
            "unless a vision-only rotation recipe opts in."
        ),
    ),
)


def precision_rules_as_dicts() -> list[dict[str, Any]]:
    return [rule.to_dict() for rule in QWEN4_EXP_PRECISION_RULES]


def high_precision_patterns() -> tuple[str, ...]:
    return tuple(rule.pattern for rule in QWEN4_EXP_PRECISION_RULES if rule.bits >= 16)


def validate_qwen4_exp_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate architecture invariants required by the Qwen4-Exp quant policy."""

    if not isinstance(config, Mapping):
        raise TypeError("Qwen4-Exp config must be a mapping")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, Mapping):
        raise TypeError("Qwen4-Exp text_config must be a mapping")

    outer_type = config.get("model_type")
    if "text_config" in config and outer_type != "qwen4_exp":
        raise ValueError(f"Qwen4-Exp outer model_type must be 'qwen4_exp', got {outer_type!r}")
    text_type = text_config.get("model_type")
    if text_type != "qwen4_exp_text":
        raise ValueError(f"Qwen4-Exp text model_type must be 'qwen4_exp_text', got {text_type!r}")

    layer_types = _required_sequence(text_config, "layer_types")
    num_layers = _required_positive_int(text_config, "num_hidden_layers")
    if len(layer_types) != num_layers:
        raise ValueError("Qwen4-Exp layer_types length must match num_hidden_layers")
    full_attention_types = {"full_attention", "qwen_sparse_attention"}
    if not all(item == "linear_attention" or item in full_attention_types for item in layer_types):
        raise ValueError(
            "Qwen4-Exp layer_types must contain only linear_attention and full_attention/qwen_sparse_attention"
        )
    if "linear_attention" not in layer_types or not any(item in full_attention_types for item in layer_types):
        raise ValueError("Qwen4-Exp requires both linear_attention and full/QSA attention layers")
    interval = int(text_config.get("full_attention_interval", 0) or 0)
    if interval > 0:
        for index, layer_type in enumerate(layer_types, start=1):
            expects_full_attention = index % interval == 0
            if (expects_full_attention and layer_type not in full_attention_types) or (
                not expects_full_attention and layer_type != "linear_attention"
            ):
                raise ValueError("Qwen4-Exp layer_types do not match full_attention_interval")

    hc_count = _required_positive_int(text_config, "hc_count")
    if hc_count <= 1:
        raise ValueError("Qwen4-Exp hc_count must be greater than 1 for hyper-connections")

    num_experts = _required_positive_int(text_config, "num_experts")
    topk = _required_positive_int(text_config, "num_experts_per_tok")
    if num_experts <= 1 or topk > num_experts:
        raise ValueError("Qwen4-Exp num_experts must exceed 1 and num_experts_per_tok must be <= num_experts")
    _required_positive_int(text_config, "moe_intermediate_size")
    _required_positive_int(text_config, "shared_expert_intermediate_size")

    ple_layer_ids = _required_sequence(text_config, "ple_layer_ids")
    if not ple_layer_ids:
        raise ValueError("Qwen4-Exp requires at least one PLE layer id")
    for layer_id in ple_layer_ids:
        if not isinstance(layer_id, int) or layer_id < 1 or layer_id > num_layers:
            raise ValueError("Qwen4-Exp PLE layer ids must be one-indexed layer numbers")
        if layer_types[layer_id - 1] != "linear_attention":
            raise ValueError("Qwen4-Exp PLE layer ids must point to linear_attention layers")
    for key in (
        "ngram_size",
        "heads_per_ngram",
        "ngram_vocab_size_base",
        "ple_embed_dim",
        "ple_conv_kernel_size",
    ):
        _required_positive_int(text_config, key)

    for key in (
        "indexer_budget",
        "indexer_compress_ratio",
        "indexer_head_dim",
        "indexer_kv_heads",
        "indexer_n_heads",
    ):
        _required_positive_int(text_config, key)

    return {
        "model_type": outer_type or text_type,
        "text_model_type": text_type,
        "num_hidden_layers": num_layers,
        "linear_attention_layers": layer_types.count("linear_attention"),
        "full_attention_layers": sum(item in full_attention_types for item in layer_types),
        "hc_count": hc_count,
        "num_experts": num_experts,
        "num_experts_per_tok": topk,
        "ple_layer_ids": tuple(ple_layer_ids),
    }


def _required_positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Qwen4-Exp config field {key!r} must be a positive integer")
    return value


def _required_sequence(config: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = config.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Qwen4-Exp config field {key!r} must be a sequence")
    return value


__all__ = [
    "PrecisionRule",
    "QWEN4_EXP_PRECISION_RULES",
    "QWEN4_EXP_PROFILE",
    "Qwen4ExpPrecisionProfile",
    "high_precision_patterns",
    "precision_rules_as_dicts",
    "validate_qwen4_exp_config",
]
