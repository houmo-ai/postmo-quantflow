"""Qwen4-Exp text export wrapper and capability gates.

The real Qwen3.8-Flash-Next text stack is not a Qwen3.5-shaped decoder: it
has hyper-connection streams, PLE/ngram state, and QSA indexer cache.  This
module provides a fixed cache ABI and exportable tensor programs for those
architecture-specific paths.

Shape notation used by inline comments:
    B: batch size; S: current input/query capacity; L: KV-cache capacity;
    H: hidden size (2560); HC: hyper-connection streams (4);
    NQ/NKV/DH: main-attention query heads/KV heads/head dim (24/2/256);
    IQ/IK/DI: indexer query heads/KV heads/head dim (4/1/128);
    R: rotary dim (64); C: indexer compression ratio (4);
    NB: candidate blocks; KB: selected blocks (at most 512);
    T: token budget (2048); P: packed selected width (T + C - 1 = 2051);
    E/K/M: routed experts/top-k/expert intermediate size (512/10/640);
    CD/KD: DeltaNet convolution channels/kernel size (10240/4);
    NV/DK/DV: DeltaNet value heads/key dim/value dim (48/128/128);
    G/DE/NL: PLE hash heads/head embedding dim/base-64 limbs (16/160/11);
    GN: PLE heads per ngram (8); U: tokenizer vocabulary; SH: token-history length.
"""

from __future__ import annotations

import importlib
import types
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.nn as nn

import xhquant.nn as xhnn
from xhquant.core import CacheTensor
from xhquant.utils.config import ConfigDict

from ...mrope import StaticMRotaryEmbedding
from .selected_kv_attention import Qwen4ExpQSAConfig, Qwen4ExpSelectedKVAttention


try:
    from transformers import DynamicCache as TransformersDynamicCache
except ImportError:  # pragma: no cover - exercised by the version preflight.
    TransformersDynamicCache = None

from ..qwen3_5._hybrid_gated_delta_net import HybridGatedDeltaNetMixin, HybridRMSNormGatedMixin
from ..qwen3_5_moe.autoround_hybrid import AutoRoundGatedDeltaNetMixin, AutoRoundSparseMoeMixin
from .cache_contract import (
    FULL_ATTENTION,
    QWEN_SPARSE_ATTENTION,
    Qwen4ExpCacheContract,
)


MIN_TRANSFORMERS_VERSION = (5, 16, 0)
SUPPORTED_LAYER_TYPES = {"linear_attention", FULL_ATTENTION, QWEN_SPARSE_ATTENTION}


class Qwen4ExpExportableRMSNorm(nn.Module):
    """Fuse normalization while retaining Qwen4's grouped FP32 affine order."""

    def __init__(self, native: nn.Module) -> None:
        super().__init__()
        self.weight = native.weight
        self.eps = native.eps
        self.group_size = native.group_size
        width = int(self.group_size or self.weight.numel())
        # The checkpoint weight remains zero-centered and retains its key.
        # SDK axis is inferred from weight rank, so its unit scale must be 1-D.
        self.norm = xhnn.RMSNorm(width, eps=self.eps)
        del self.norm.weight
        self.norm.register_buffer(
            "weight", torch.ones(width, dtype=torch.float32, device=self.weight.device), persistent=False,
        )
        self.train(native.training)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values = hidden_states.float()
        if self.group_size is not None:
            values = values.reshape(*values.shape[:-1], -1, self.group_size)
        normalized = self.norm(values)
        if self.group_size is not None:
            normalized = normalized.flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(hidden_states.dtype)


class Qwen4ExpExportableRMSNormGated(HybridRMSNormGatedMixin, nn.Module):
    """Use the SDK's weighted head norm before the checkpoint-selected gate."""

    def __init__(self, native: nn.Module) -> None:
        super().__init__()
        from transformers.activations import ACT2FN

        self.weight = native.weight
        self.activation = native.activation
        self.gate_act = ACT2FN[self.activation]
        self.norm = xhnn.RMSNorm(self.weight.numel(), eps=native.variance_epsilon)
        del self.norm.weight
        # Keep the checkpoint's original key/Parameter, while the SDK leaf
        # consumes that same scale without an additional persistent key.
        self.norm.register_buffer("weight", self.weight, persistent=False)
        self.train(native.training)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.norm._buffers["weight"] = self.weight
        return result

    def _apply_output_gate(
        self, normalized: torch.Tensor, gate: torch.Tensor, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Qwen4 chooses its gate activation and applies it in FP32.
        return (normalized * self.gate_act(gate.float())).to(hidden_states.dtype)


def adapt_qwen4_exp_rms_norms(model: nn.Module) -> tuple[str, ...]:
    """Adapt loaded native text norms, including MTP's top-level HC norms."""
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in tuple(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if type(child).__name__ == "Qwen4ExpTextRMSNorm":
                setattr(parent, name, Qwen4ExpExportableRMSNorm(child))
                replaced.append(path)
            elif type(child).__name__ == "Qwen4ExpTextRMSNormGated":
                setattr(parent, name, Qwen4ExpExportableRMSNormGated(child))
                replaced.append(path)
            else:
                visit(child, path)

    visit(model, "")
    return tuple(replaced)


def _torch_is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if is_compiling is not None:
        return bool(is_compiling())
    dynamo = getattr(torch, "_dynamo", None)
    dynamo_is_compiling = getattr(dynamo, "is_compiling", None)
    if dynamo_is_compiling is not None:
        return bool(dynamo_is_compiling())
    return False


def _module_execution_device(module: nn.Module, fallback: torch.device) -> torch.device:
    first_non_meta: torch.device | None = None
    for tensor in (*module.parameters(), *module.buffers()):  # shape: module-specific parameter/buffer
        if tensor.device.type == "cuda":
            return tensor.device
        if tensor.device.type != "meta" and first_non_meta is None:
            first_non_meta = tensor.device
    return first_non_meta or fallback


def _move_cache_layer_to_device(cache_layer: Any, device: torch.device) -> None:
    """Keep a sharded decoder layer's mutable cache state on its execution device."""

    for state_name in ("conv_states", "recurrent_states"):
        states = getattr(cache_layer, state_name, None)  # container: heterogeneous cache tensors
        if states is None:
            continue
        for state_idx, state in enumerate(states):  # state shape: defined by cache contract
            if torch.is_tensor(state):
                states[state_idx] = state.to(device)  # same contract-defined shape as state
    for state_name in ("keys", "values", "indexer_keys", "indexer_tail", "key_tail", "value_tail"):
        state = getattr(cache_layer, state_name, None)  # shape: [B, heads, L, head_dim]
        if torch.is_tensor(state):
            setattr(cache_layer, state_name, state.to(device))
    if hasattr(cache_layer, "device"):
        cache_layer.device = device


class Qwen4ExpExportUnsupportedError(RuntimeError):
    """Raised when a Qwen4-Exp model cannot be exported by the current path."""

    def __init__(self, reasons: list[str], capability: "Qwen4ExpExportCapability | None" = None) -> None:
        self.reasons = tuple(reasons)
        self.capability = capability
        super().__init__("Qwen4-Exp HMONNX export is unsupported: " + "; ".join(reasons))


@dataclass(frozen=True)
class Qwen4ExpExportCapability:
    """Structured preflight result for Qwen4-Exp HMONNX export."""

    supported: bool
    reasons: tuple[str, ...] = ()
    transformers_version: str | None = None
    has_transformers_qwen4_exp: bool = False
    hmonnx_runtime_available: bool = False
    cache_contract: dict[str, Any] | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def assert_supported(self) -> None:
        if not self.supported:
            raise Qwen4ExpExportUnsupportedError(list(self.reasons), self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "reasons": list(self.reasons),
            "transformers_version": self.transformers_version,
            "has_transformers_qwen4_exp": self.has_transformers_qwen4_exp,
            "hmonnx_runtime_available": self.hmonnx_runtime_available,
            "cache_contract": self.cache_contract,
            "details": dict(self.details),
        }


class Qwen4ExpExportableQSAIndexer(nn.Module):
    """Build new C4 index keys and retain the pooled cache across steps."""

    def __init__(self, native_indexer: nn.Module, *, sequence_length: int) -> None:
        super().__init__()
        from ..deepseek_v4.static_cache import FixedCapacityCacheWriter

        self.layer_idx = int(native_indexer.layer_idx)  # scalar: decoder layer index
        self.index_n_heads = int(native_indexer.index_n_heads)  # scalar IQ=4
        self.index_kv_heads = int(native_indexer.index_kv_heads)  # scalar IK=1
        self.index_head_dim = int(native_indexer.index_head_dim)  # scalar DI=128
        self.token_budget = int(native_indexer.token_budget)  # scalar T=2048
        self.compress_ratio = int(native_indexer.compress_ratio)  # scalar C=4
        self.block_topk = int(native_indexer.block_topk)  # scalar KB<=512
        self.index_qk_proj = native_indexer.index_qk_proj  # Linear: H -> (IQ+IK)*DI = 640
        self.q_layernorm = native_indexer.q_layernorm  # RMSNorm: [...,DI] -> [...,DI]
        self.k_layernorm = native_indexer.k_layernorm  # RMSNorm: [...,DI] -> [...,DI]
        self.rope = xhnn.Rope()
        self.pool_writer = FixedCapacityCacheWriter(keep_cache_dims=True)
        self.register_buffer(
            "pool_padding", self.k_layernorm.weight.new_zeros(1, 3, self.index_head_dim), persistent=False,
        )
        self.set_sequence_length(sequence_length)

    def set_sequence_length(self, sequence_length: int) -> None:
        self.pool_groups = (int(sequence_length) + 3) // 4
        self.pool_candidate = xhnn.DynamicSlice([self.pool_groups * 4], [1], [1])
        self.pool_tail = xhnn.DynamicSlice([4], [1], [1])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return index queries and pooled block keys for SelectedKV."""

        batch_size, seq_length, _ = hidden_states.shape  # scalars B,S,H
        hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)  # tuple [B,S,heads,DI]
        cos, sin = position_embeddings  # each [B,S + pool_groups,R]
        if past_key_values is None:
            raise ValueError("Qwen4Exp serving export requires explicit cache state")
        current_cos, current_sin = cos[:, :seq_length], sin[:, :seq_length]

        qk = self.index_qk_proj(hidden_states)  # [B,S,(IQ+IK)*DI] = [B,S,640]
        q, token_k = torch.split(
            qk,
            [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
            dim=-1,
        )  # q [B,S,IQ*DI], token_k [B,S,IK*DI]
        q = q.reshape(*hidden_shape)  # [B,S,IQ,DI]
        raw_keys = token_k.reshape(*hidden_shape).squeeze(2)  # [B,S,DI], because IK=1
        q = self.q_layernorm(q)  # [B,S,IQ,DI]
        q = _apply_rotary(q, current_cos, current_sin, unsqueeze_dim=2, rope=self.rope)  # [B,S,IQ,DI]
        cache_layer = past_key_values.layers[self.layer_idx]
        pool_offset, pool_count, pool_start = past_key_values.qsa_host_controls[:3]
        keys = torch.cat((cache_layer.indexer_tail, raw_keys), dim=1)
        candidates = self.pool_candidate(torch.cat((keys, self.pool_padding), dim=1), pool_offset)
        candidates = candidates.reshape(batch_size, self.pool_groups, 4, self.index_head_dim)
        pooled = self.k_layernorm(candidates.float().mean(dim=2).to(raw_keys.dtype))
        block_keys = _apply_rotary(
            pooled.unsqueeze(1),
            cos[:, seq_length : seq_length + self.pool_groups],
            sin[:, seq_length : seq_length + self.pool_groups],
            unsqueeze_dim=1,
            rope=self.rope,
        ).squeeze(1)
        cache_layer.indexer_keys = self.pool_writer(cache_layer.indexer_keys, block_keys, pool_start, pool_count)
        cache_layer.indexer_tail = self.pool_tail(keys, past_key_values.current_input_length)
        return q.transpose(1, 2), cache_layer.indexer_keys


class Qwen4ExpExportableAttention(nn.Module):
    """QSA attention with a stable DynamoFX-friendly tensor program."""

    def __init__(self, native_attention: nn.Module, *, cache_length: int, sequence_length: int) -> None:
        super().__init__()
        self.config = native_attention.config
        self.layer_idx = int(native_attention.layer_idx)  # scalar: decoder layer index
        self.head_dim = int(native_attention.head_dim)  # scalar DH=256
        self.num_key_value_groups = int(native_attention.num_key_value_groups)  # scalar NQ/NKV=12
        self.scaling = float(native_attention.scaling)  # scalar 1/sqrt(DH)
        self.q_proj = native_attention.q_proj  # Linear: H -> NQ*(2*DH)
        self.k_proj = native_attention.k_proj  # Linear: H -> NKV*DH
        self.v_proj = native_attention.v_proj  # Linear: H -> NKV*DH
        self.o_proj = native_attention.o_proj  # Linear: NQ*DH -> H
        self.q_norm = native_attention.q_norm  # RMSNorm: [...,DH] -> [...,DH]
        self.k_norm = native_attention.k_norm  # RMSNorm: [...,DH] -> [...,DH]
        self.rope = xhnn.Rope()
        self.indexer = Qwen4ExpExportableQSAIndexer(native_attention.indexer, sequence_length=sequence_length)
        self.k_cache = xhnn.LLMCacheV2(axis=2, attention_max_length=-1)
        self.v_cache = xhnn.LLMCacheV2(axis=2, attention_max_length=-1)
        if cache_length < 4 or self.indexer.compress_ratio != 4:
            raise ValueError("SelectedKV requires compression=4 and at least four cache slots")
        self.packed_qsa = Qwen4ExpSelectedKVAttention(
            Qwen4ExpQSAConfig(
                num_heads=int(self.config.num_attention_heads),
                num_kv_heads=int(self.config.num_key_value_heads),
                head_dim=self.head_dim,
                scale=self.scaling,
            ),
            candidate_topk=min(self.indexer.block_topk, cache_length // 4),
            index_head_dim=self.indexer.index_head_dim,
            input_length=sequence_length,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Any = None,
        **_: Any,
    ) -> tuple[torch.Tensor, None]:
        """Shapes: hidden [B,S,H], cos/sin [B,S+ceil(S/4),R], output [B,S,H]."""

        if past_key_values is None:
            raise ValueError("Qwen4Exp serving export requires explicit cache state")
        index_query, block_keys = self.indexer(
            hidden_states,
            position_embeddings,
            past_key_values,
        )

        rotary_cos, rotary_sin = position_embeddings
        cos = rotary_cos[:, : hidden_states.shape[1]]
        sin = rotary_sin[:, : hidden_states.shape[1]]
        input_shape = hidden_states.shape[:-1]  # tuple [B,S]
        hidden_shape = (*input_shape, -1, self.head_dim)  # tuple [B,S,heads,DH]
        projected = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)  # [B,S,NQ,2*DH]
        query_states = projected[..., : self.head_dim]  # [B,S,NQ,DH]
        gate = projected[..., self.head_dim :]  # [B,S,NQ,DH]
        gate = gate.reshape(*input_shape, -1)  # [B,S,NQ*DH]
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)  # [B,NQ,S,DH]
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # [B,NKV,S,DH]
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B,NKV,S,DH]
        query_states, key_states = _apply_rotary_pair(
            query_states, key_states, cos, sin, rope=self.rope
        )  # q [B,NQ,S,DH], k [B,NKV,S,DH]
        current_key, current_value = key_states, value_states
        cache_layer = past_key_values.layers[self.layer_idx]  # container: layer-local cache
        key_states = self.k_cache(
            key_states,
            past_key_values.past_seq_length,
            past_key_values.current_input_length,
            cache_layer.keys,
        )  # [B,NKV,L,DH]
        value_states = self.v_cache(
            value_states,
            past_key_values.past_seq_length,
            past_key_values.current_input_length,
            cache_layer.values,
        )  # [B,NKV,L,DH]
        cache_layer.keys = key_states  # [B,NKV,L,DH]
        cache_layer.values = value_states  # [B,NKV,L,DH]

        controls = getattr(past_key_values, "qsa_host_controls", None)
        if controls is None:
            raise ValueError("SelectedKV requires host-prepared QSA controls")
        query_valid, attention_mask = controls[3:]
        attn_output, cache_layer.key_tail, cache_layer.value_tail = self.packed_qsa(
            query_states,
            index_query,
            block_keys,
            key_states,
            value_states,
            current_key,
            current_value,
            cache_layer.key_tail,
            cache_layer.value_tail,
            attention_mask,
            past_key_values.past_seq_length,
            past_key_values.current_input_length,
        )
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(*input_shape, -1)  # [B,S,NQ*DH]
        attn_output = attn_output * torch.sigmoid(gate)  # [B,S,NQ*DH]
        output = self.o_proj(attn_output) * query_valid.unsqueeze(-1).to(attn_output.dtype)
        return output, None  # [B,S,H], None


def _stack_qwen4_exp_expert_biases(experts: list[nn.Module], linear_name: str) -> torch.Tensor | None:
    biases = [getattr(expert, linear_name).bias for expert in experts]  # list[E] of [out_features] or None
    if all(bias is None for bias in biases):
        return None
    if any(bias is None for bias in biases):
        raise ValueError(f"Qwen4-Exp {linear_name} expert biases must be present for every expert or none")
    return torch.stack(biases)  # [E,out_features]


def _build_qwen4_exp_gptq_packed_moeblock(
    experts: Any,
    activation_name: str,
    top_k: int,
) -> nn.Module | None:
    """Build the Qwen4-only packed expert container without widening Qwen3.5."""

    from xhquant.nn import GPTQPackedLinear, GPTQPackedMoeBlock

    if hasattr(experts, "gate_up_proj"):
        return None
    try:
        expert_list = list(experts)  # container length E
    except TypeError:
        return None
    projections = [  # container length 3*E; each module maps H<->M
        getattr(expert, linear_name) for expert in expert_list for linear_name in ("gate_proj", "up_proj", "down_proj")
    ]
    packed_flags = [isinstance(projection, GPTQPackedLinear) for projection in projections]  # list[3*E]
    if not any(packed_flags):
        return None
    if not all(packed_flags):
        raise TypeError("Qwen4-Exp routed expert gate/up/down projections must all use GPTQPackedLinear")
    packed_moeblock = GPTQPackedMoeBlock(
        activation_type=activation_name,
        k=top_k,
        normalize_routing_weights=True,
        gate_proj=[expert.gate_proj.packed_weight for expert in expert_list],
        up_proj=[expert.up_proj.packed_weight for expert in expert_list],
        down_proj=[expert.down_proj.packed_weight for expert in expert_list],
        expert_gate_proj_bias=_stack_qwen4_exp_expert_biases(expert_list, "gate_proj"),
        expert_up_proj_bias=_stack_qwen4_exp_expert_biases(expert_list, "up_proj"),
        expert_down_proj_bias=_stack_qwen4_exp_expert_biases(expert_list, "down_proj"),
    )
    _bind_qwen4_exp_packed_moe_dtype_anchor(packed_moeblock)
    return packed_moeblock


def _bind_qwen4_exp_packed_moe_dtype_anchor(moeblock: nn.Module) -> None:
    """Expose the expert dtype without constructing a dense expert tensor."""

    packed_experts = getattr(moeblock, "expert_gate_proj_packed", None)  # container length E
    if packed_experts is None or len(packed_experts) == 0:
        raise TypeError("Qwen4 GPTQ packed MoE must contain at least one gate projection")
    scales = getattr(packed_experts[0], "scales", None)  # quantization-specific scale tensor
    if not torch.is_tensor(scales):
        raise TypeError("Qwen4 GPTQ packed MoE gate projection must expose floating-point scales")

    moeblock._buffers.pop("expert_gate_proj_weight", None)
    moeblock.expert_gate_proj_weight = scales.new_empty(())  # scalar [] dtype anchor


def _bind_qwen4_exp_packed_output_dtype_anchor(projection: nn.Module) -> None:
    """Expose only the dtype expected by the shared GatedDeltaNet forward.

    ``xhquant.GPTQPackedLinear`` deliberately has no dense ``weight``.  The
    shared lowering reads ``out_proj.weight.dtype`` before invoking the leaf
    module, so a scalar, unregistered tensor is sufficient for that metadata
    access without materializing or serializing a dense projection.
    """

    weight = getattr(projection, "weight", None)  # dense [out_features,in_features] or None
    if torch.is_tensor(weight):
        return

    from xhquant.nn import GPTQPackedLinear

    if not isinstance(projection, GPTQPackedLinear):
        raise TypeError(
            "Qwen4 gated-delta output projection must expose either a dense weight or use xhquant.GPTQPackedLinear"
        )
    scales = getattr(projection.packed_weight, "scales", None)  # quantization-specific scale tensor
    if not torch.is_tensor(scales):
        raise TypeError("Qwen4 GPTQ packed output projection must expose floating-point scales")

    projection._parameters.pop("weight", None)
    projection.weight = scales.new_empty(())  # scalar [] dtype anchor


class Qwen4ExpExportableSparseMoe(AutoRoundSparseMoeMixin, nn.Module):
    """Reuse the production Qwen3.5/Next XH2 ``MoeBlock`` lowering."""

    def _setup(self, cfg: Any = None):
        experts = self.experts  # container length E
        activation = getattr(experts, "act_fn", None)
        if activation is None:
            try:
                activation = experts[0].act_fn
            except (IndexError, TypeError):
                activation = None
        activation_name = (
            activation._get_name().lower()
            if activation is not None and hasattr(activation, "_get_name")
            else type(activation).__name__.lower()
        )
        packed_moeblock = _build_qwen4_exp_gptq_packed_moeblock(experts, activation_name, self.gate.top_k)
        if packed_moeblock is None:
            return super()._setup(cfg)

        self.input_seq_len = cfg.input_sequence_length
        self.batch_size = cfg.batch_size
        self.device = self.gate.weight.device  # device scalar; gate.weight [E,H]
        gate_weight = self.gate.weight.detach().clone()  # [E,H]
        self.gate = nn.Linear(
            gate_weight.shape[1],  # in_features H
            gate_weight.shape[0],  # out_features E
            bias=False,
            device=self.device,
            dtype=gate_weight.dtype,
        )
        self.gate.weight.data.copy_(gate_weight)  # [E,H]
        self.moeblock = packed_moeblock
        self.experts = nn.ModuleList()
        torch.cuda.empty_cache()
        return self

    @classmethod
    def from_native(cls, native_moe: nn.Module, *, batch_size: int, sequence_length: int) -> nn.Module:
        export_moe = cls()
        export_moe.gate = native_moe.gate
        export_moe.experts = native_moe.experts
        export_moe.shared_expert = native_moe.shared_expert
        export_moe.shared_expert_gate = native_moe.shared_expert_gate
        export_moe._setup(
            types.SimpleNamespace(
                batch_size=int(batch_size),
                input_sequence_length=int(sequence_length),
            )
        )
        return export_moe


class Qwen4ExpExportableGatedDeltaNet(AutoRoundGatedDeltaNetMixin, nn.Module):
    """Qwen4 DynamicCache adapter for the shared XH2 GatedDeltaNet lowering."""

    def _input_projection_dtype(self) -> torch.dtype:
        projection = self.in_proj_q if self.split_conv_cache else self.in_proj_qkv
        weight = getattr(projection, "weight", None)  # dense [out_features,H] or None
        if torch.is_tensor(weight):
            return weight.dtype

        packed_weight = getattr(projection, "packed_weight", None)  # packed projection container
        scales = getattr(packed_weight, "scales", None)  # quantization-specific scale tensor
        if torch.is_tensor(scales):
            return scales.dtype
        raise TypeError("Qwen4 gated-delta input projection must expose either a dense weight or GPTQ packed scales")

    @classmethod
    def from_native(
        cls,
        native_gated_delta_net: nn.Module,
        *,
        batch_size: int,
        sequence_length: int,
    ) -> "Qwen4ExpExportableGatedDeltaNet":
        native_gated_delta_net.__class__ = cls
        _bind_qwen4_exp_packed_output_dtype_anchor(native_gated_delta_net.out_proj)
        native_gated_delta_net._setup(
            ConfigDict(
                {
                    "use_cache": True,
                    "linear_attention_mode": "recurrent",
                    "linear_chunk_size": 64,
                    "return_cache": True,
                    "input_sequence_length": int(sequence_length),
                    "batch_size": int(batch_size),
                    "split_conv_cache": False,
                    "suppress_recurrent_state_outputs": False,
                    "fuse_gdr_ops": False,
                    "fuse_gdr_block_recurrent_ops": False,
                    "use_manual_depthwise_conv1d": False,
                }
            )
        )
        return native_gated_delta_net

    def enable_fused_gdr(
        self,
        *,
        linear_attention_mode: str | None = None,
        linear_chunk_size: int | None = None,
        fuse_gdr_block_recurrent_ops: bool = True,
    ) -> "Qwen4ExpExportableGatedDeltaNet":
        """Enable xhquant fused GDR ops for the current export phase.

        The default phase follows the fixed export shape: multi-token prefill
        uses chunk mode and one-token decode uses recurrent mode.
        """

        mode = linear_attention_mode
        if mode is None:
            mode = "recurrent" if int(self.input_sequence_length) == 1 else "chunk"
        if mode not in {"chunk", "recurrent"}:
            raise ValueError(f"Unsupported Qwen4 fused GDR linear_attention_mode: {mode!r}")
        chunk_size = int(self.linear_chunk_size if linear_chunk_size is None else linear_chunk_size)
        self._setup(
            ConfigDict(
                {
                    "use_cache": self.use_cache,
                    "linear_attention_mode": mode,
                    "linear_chunk_size": chunk_size,
                    "return_cache": self.return_cache,
                    "input_sequence_length": int(self.input_sequence_length),
                    "batch_size": int(self.batch_size),
                    "split_conv_cache": self.split_conv_cache,
                    "suppress_recurrent_state_outputs": self.suppress_recurrent_state_outputs,
                    "fuse_gdr_ops": True,
                    "fuse_gdr_block_recurrent_ops": bool(fuse_gdr_block_recurrent_ops),
                    "use_manual_depthwise_conv1d": self.use_manual_depthwise_conv1d,
                }
            )
        )
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Shapes: hidden/output [B,S,H], conv [B,CD,KD], recurrent [B,NV,DK,DV]."""

        if cache_params is None:
            raise ValueError("Qwen4Exp HMONNX GatedDeltaNet requires explicit cache state")
        cache_layer = cache_params.layers[self.layer_idx]  # container: layer-local cache
        conv_cache = cache_layer.conv_states[0]  # [B,CD,KD]
        recurrent_state = cache_layer.recurrent_states[0]  # [B,NV,DK,DV]
        chunk_state_is_cache = bool(getattr(getattr(self, "chunk_scan_op", None), "state_is_cache", False))
        if (
            chunk_state_is_cache
            and self.linear_attention_mode != "recurrent"
            and not isinstance(recurrent_state, CacheTensor)
            and not _torch_is_compiling()
        ):
            recurrent_state = CacheTensor(recurrent_state)
            cache_layer.recurrent_states[0] = recurrent_state
        if attention_mask is None:
            attention_mask = hidden_states.new_ones(hidden_states.shape[:2])  # [B,S]
        output, conv_cache_out, recurrent_state_out = HybridGatedDeltaNetMixin.forward(
            self,
            hidden_states,
            conv_cache=conv_cache,
            recurrent_state=recurrent_state,
            linear_attn_mask=attention_mask,
            current_input_length=cache_params.current_input_length,
        )  # output [B,S,H], conv [B,CD,KD], recurrent [B,NV,DK,DV]
        cache_layer.conv_states[0] = conv_cache_out  # [B,CD,KD]
        # xh::GDRChunkScan owns recurrent-state mutation through its cache
        # input and reports no separate state output.  Keep the explicit Qwen4
        # cache ABI populated by reusing the existing state tensor in that case.
        cache_layer.recurrent_states[0] = recurrent_state if recurrent_state_out is None else recurrent_state_out
        return output  # [B,S,H]

    # SDK tracing replaces forward with graph_forward. Keep the native cache
    # adapter instead of inheriting the shared mixin's flat-tensor graph ABI.
    graph_forward = forward


class Qwen4ExpTextExportWrapper(nn.Module):
    """Fixed-ABI wrapper around a native Qwen4-Exp text model.

    It accepts explicit cache tensors described by
    :class:`Qwen4ExpCacheContract`, reconstructs the Transformers
    ``DynamicCache``, and returns the updated cache in the same stable order.
    HMONNX capability is checked by the export facade rather than here so this
    wrapper can also serve as an executable cache-ABI oracle.
    """

    def __init__(
        self,
        native_text_model: nn.Module,
        text_config: Any,
        *,
        lm_head: nn.Module | None = None,
        batch_size: int = 1,
        input_sequence_length: int = 1,
        cache_length: int = 1,
        require_hmonnx_runtime: bool = False,
        return_hidden_states: bool = False,
        return_native_hidden_states: bool = False,
    ) -> None:
        super().__init__()
        self.native_text_model = native_text_model
        self.lm_head = lm_head
        self.return_hidden_states = return_hidden_states
        self.return_native_hidden_states = return_native_hidden_states
        _replace_native_modules_for_export(
            self.native_text_model,
            batch_size=batch_size,
            sequence_length=input_sequence_length,
            cache_length=cache_length,
        )
        self.text_config = text_config
        self.batch_size = int(batch_size)
        self.input_sequence_length = int(input_sequence_length)
        self.cache_length = int(cache_length)
        self.cache_contract = Qwen4ExpCacheContract.from_text_config(text_config)
        has_device_hooks = any(
            hasattr(module, "_hf_hook")
            for owner in (self.native_text_model, self.lm_head)
            if owner is not None
            for module in owner.modules()
        )
        if return_native_hidden_states and (return_hidden_states or has_device_hooks):
            raise ValueError("Native hidden-state output requires an unsharded model")
        self._explicit_device_forward = not return_native_hidden_states
        if self._explicit_device_forward:
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(self.native_text_model, recurse=True)
            if self.lm_head is not None:
                remove_hook_from_module(self.lm_head, recurse=True)
            fallback_device = _module_execution_device(
                self.native_text_model.embed_tokens, torch.device("cpu")
            )  # device scalar; embedding weight [V,H]
            self._layer_execution_devices = tuple(
                _module_execution_device(layer, fallback_device)
                for layer in self.native_text_model.layers[: int(self.text_config.num_hidden_layers)]
            )
            self._rotary_execution_device = _module_execution_device(self.native_text_model.rotary_emb, fallback_device)
            self._mixer_execution_device = _module_execution_device(
                self.native_text_model.hyper_connection_mixer, fallback_device
            )
            self._lm_head_execution_device = (
                _module_execution_device(self.lm_head, fallback_device) if self.lm_head is not None else fallback_device
            )
        self.capability = inspect_qwen4_exp_export_capability(
            text_config,
            model=native_text_model,
            require_hmonnx_runtime=require_hmonnx_runtime,
        )

    @classmethod
    def from_native_model(
        cls,
        native_model: nn.Module,
        *,
        text_config: Any | None = None,
        batch_size: int = 1,
        input_sequence_length: int = 1,
        cache_length: int = 1,
        require_hmonnx_runtime: bool = False,
        return_hidden_states: bool = False,
    ) -> "Qwen4ExpTextExportWrapper":
        text_model = extract_qwen4_exp_text_model(native_model)
        lm_head = getattr(native_model, "lm_head", None)
        if lm_head is None:
            lm_head = getattr(text_model, "lm_head", None)
        return cls(
            text_model,
            text_config or extract_qwen4_exp_text_config(native_model),
            lm_head=lm_head,
            batch_size=batch_size,
            input_sequence_length=input_sequence_length,
            cache_length=cache_length,
            require_hmonnx_runtime=require_hmonnx_runtime,
            return_hidden_states=return_hidden_states,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        ple_input_ids: torch.Tensor,
        time_position_ids: torch.Tensor,
        height_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
        pool_time_position_ids: torch.Tensor,
        pool_height_position_ids: torch.Tensor,
        pool_width_position_ids: torch.Tensor,
        *cache_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Shapes: embeds [B,S,H], lengths [1], PLE ids [B,S], logits [B,S,V]."""

        expected_cache_keys = self.cache_contract.expected_keys()  # list[num_cache_tensors]
        qsa_controls = None
        if len(cache_tensors) == len(expected_cache_keys) + 5:
            cache_tensors, qsa_controls = cache_tensors[:-5], cache_tensors[-5:]
        if len(cache_tensors) != len(expected_cache_keys):
            raise ValueError(
                "Qwen4ExpTextExportWrapper.forward expects flattened cache tensors in "
                "cache_contract.expected_keys() order."
            )
        cache_map = dict(zip(expected_cache_keys, cache_tensors, strict=True))  # name -> contract-shaped tensor
        self.cache_contract.validate_cache_shapes(cache_map)
        dynamic_cache = self._build_dynamic_cache(cache_map)  # container: heterogeneous cache tensors
        dynamic_cache.past_seq_length = past_seq_length  # [1]
        dynamic_cache.current_input_length = current_input_length  # [1]
        dynamic_cache.qsa_host_controls = qsa_controls

        query_capacity = inputs_embeds.shape[1]  # scalar S
        query_offsets = torch.arange(
            query_capacity,
            dtype=torch.int32,
            device=inputs_embeds.device,
        ).view(1, -1)  # [1,S]
        current_positions = past_seq_length.to(torch.int32).reshape(1, 1) + query_offsets  # [1,S]
        current_positions = current_positions.expand(inputs_embeds.shape[0], -1)  # [B,S]
        dynamic_cache.current_position_indices = current_positions  # [B,S]

        rope_positions = tuple(torch.cat((current, pooled), dim=-1) for current, pooled in zip(
            (time_position_ids, height_position_ids, width_position_ids),
            (pool_time_position_ids, pool_height_position_ids, pool_width_position_ids), strict=True,
        ))

        key_positions = torch.arange(
            self.cache_length,
            dtype=torch.int32,
            device=inputs_embeds.device,
        ).view(1, 1, 1, -1)  # [1,1,1,L]
        query_positions = current_positions.view(inputs_embeds.shape[0], 1, query_capacity, 1)  # [B,1,S,1]
        valid_query = query_offsets < current_input_length.to(torch.int32).reshape(1, 1)  # [1,S]
        visible = key_positions <= query_positions  # [B,1,S,L]
        visible = visible & (
            key_positions
            < (
                past_seq_length.to(torch.int32).reshape(1, 1, 1, 1)
                + current_input_length.to(torch.int32).reshape(1, 1, 1, 1)
            )
        )  # [B,1,S,L]
        visible = visible & valid_query.view(inputs_embeds.shape[0], 1, query_capacity, 1)  # [B,1,S,L]
        causal_mask_mapping = {
            "full_attention": visible,  # [B,1,S,L]
            "linear_attention": valid_query,  # [1,S] for the serving B=1 ABI
        }  # mapping: layer type -> bool mask
        if self._explicit_device_forward:
            hidden_states = self._forward_text_model_explicit_devices(
                inputs_embeds=inputs_embeds,
                causal_mask_mapping=causal_mask_mapping,
                position_ids=rope_positions,
                ple_input_ids=ple_input_ids,
                past_key_values=dynamic_cache,
            )  # [B,S,H]
        else:
            outputs = self.native_text_model(
                inputs_embeds=inputs_embeds,
                attention_mask=causal_mask_mapping,
                position_ids=rope_positions,
                ple_input_ids=ple_input_ids,
                past_key_values=dynamic_cache,
                use_cache=True,
            )  # model output container; last_hidden_state [B,S,H]
            hidden_states = getattr(outputs, "last_hidden_state", outputs[0])  # [B,S,H]
            if self.return_native_hidden_states:
                target_hidden_states = outputs.hidden_states[0]
        if self.return_hidden_states:
            hidden_states, target_hidden_states = hidden_states
        if self.lm_head is None:
            logits = hidden_states  # [B,S,H]
        else:
            if self._explicit_device_forward:
                hidden_states = hidden_states.to(self._lm_head_execution_device)  # [B,S,H]
            logits = self.lm_head(hidden_states)  # [B,S,V]
        auxiliary = (target_hidden_states,) if self.return_hidden_states or self.return_native_hidden_states else ()
        return (logits, *auxiliary, *self._flatten_dynamic_cache(dynamic_cache))  # ([B,S,V or H], *cache tensors)

    def _forward_text_model_explicit_devices(
        self,
        *,
        inputs_embeds: torch.Tensor,
        causal_mask_mapping: Mapping[str, torch.Tensor],
        position_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ple_input_ids: torch.Tensor,
        past_key_values: Any,
    ) -> torch.Tensor:
        """Run an Accelerate-sharded model without untraceable device hooks."""

        conv_mask = causal_mask_mapping.get("linear_attention")  # [1,S] or None
        if getattr(self.text_config, "ple_layer_ids", ()) and conv_mask is not None:
            eos_token_id = getattr(self.text_config, "eos_token_id", 0)
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0]
            ple_input_ids = torch.where(conv_mask.to(torch.bool), ple_input_ids, int(eos_token_id))  # [B,S]

        rotary_device = self._rotary_execution_device  # device scalar
        position_embeddings = self.native_text_model.rotary_emb(
            inputs_embeds.to(rotary_device),  # [B,S,H], used for dtype/device
            *(axis.to(rotary_device) for axis in position_ids),
        )  # tuple(cos,sin), each [B,L,R]
        hidden_states = inputs_embeds.repeat(1, 1, int(self.text_config.hc_count))  # [B,S,HC*H]
        full_attention_mask = causal_mask_mapping["full_attention"]  # [B,1,S,L]
        for layer_idx, decoder_layer in enumerate(
            self.native_text_model.layers[: int(self.text_config.num_hidden_layers)]
        ):
            layer_device = self._layer_execution_devices[layer_idx]  # device scalar
            past_key_values.past_seq_length = past_key_values.past_seq_length.to(layer_device)  # [1]
            past_key_values.current_input_length = past_key_values.current_input_length.to(layer_device)  # [1]
            past_key_values.current_position_indices = past_key_values.current_position_indices.to(
                layer_device
            )  # [B,S]
            if past_key_values.qsa_host_controls is not None:
                past_key_values.qsa_host_controls = tuple(
                    value.to(layer_device) for value in past_key_values.qsa_host_controls
                )
            _move_cache_layer_to_device(past_key_values.layers[layer_idx], layer_device)
            hidden_states = decoder_layer(
                hidden_states.to(layer_device),  # [B,S,HC*H]
                position_embeddings=tuple(value.to(layer_device) for value in position_embeddings),  # each [B,L,R]
                attention_mask=full_attention_mask.to(layer_device),  # [B,1,S,L]
                conv_mask=None if conv_mask is None else conv_mask.to(layer_device),  # [1,S] or None
                past_key_values=past_key_values,
                ple_input_ids=ple_input_ids.to(layer_device),  # [B,S]
            )  # [B,S,HC*H]
        mixer_device = self._mixer_execution_device  # device scalar
        hidden_states = hidden_states.to(mixer_device)
        collapsed = self.native_text_model.hyper_connection_mixer(hidden_states)
        return (collapsed, hidden_states) if self.return_hidden_states else collapsed  # [B,S,H]

    def _build_dynamic_cache(self, cache_map: Mapping[str, torch.Tensor]) -> Any:
        if TransformersDynamicCache is None:
            raise RuntimeError("transformers>=5.16.0 is required to construct Qwen4Exp cache state")
        dynamic_cache = TransformersDynamicCache(config=self.text_config)  # heterogeneous cache container
        for layer_contract in self.cache_contract.layers:
            cache_layer = dynamic_cache.layers[layer_contract.layer_idx]  # layer-local cache container
            first_tensor: torch.Tensor | None = None
            for spec in layer_contract.tensors:
                tensor = cache_map[spec.name]  # shape: spec.shape from Qwen4ExpCacheContract
                first_tensor = tensor if first_tensor is None else first_tensor  # same shape as first layer tensor
                if spec.state_kind in {"delta_conv", "ple_conv", "ple_ngram_history"}:
                    state_idx = _required_state_idx(spec)  # scalar cache slot
                    cache_layer.conv_states[state_idx] = tensor  # contract-specific [B,...]
                    cache_layer.conv_kernel_size[state_idx] = tensor.shape[-1]  # scalar final dimension
                    cache_layer.is_conv_states_initialized[state_idx] = True
                    cache_layer.has_previous_state[state_idx] = True
                elif spec.state_kind == "delta_recurrent":
                    state_idx = _required_state_idx(spec)  # scalar cache slot
                    cache_layer.recurrent_states[state_idx] = tensor  # [B,NV,DK,DV]
                    cache_layer.is_recurrent_states_initialized[state_idx] = True
                elif spec.state_kind == "kv_key":
                    cache_layer.keys = tensor  # [B,NKV,L,DH]
                    cache_layer.is_initialized = True
                elif spec.state_kind == "kv_value":
                    cache_layer.values = tensor  # [B,NKV,L,DH]
                    cache_layer.is_initialized = True
                elif spec.state_kind == "qsa_indexer_key":
                    cache_layer.indexer_keys = tensor  # [B,1,L//4,DI]
                    cache_layer.is_indexer_initialized = True
                    cache_layer.indexer_dtype = tensor.dtype
                    cache_layer.indexer_device = tensor.device
                elif spec.state_kind == "qsa_indexer_tail":
                    cache_layer.indexer_tail = tensor  # [B,4,DI]
                elif spec.state_kind == "qsa_key_tail":
                    cache_layer.key_tail = tensor
                elif spec.state_kind == "qsa_value_tail":
                    cache_layer.value_tail = tensor
                else:
                    raise ValueError(f"unsupported Qwen4Exp cache state kind {spec.state_kind!r}")
            if first_tensor is not None:
                cache_layer.device = first_tensor.device
                cache_layer.dtype = first_tensor.dtype
        return dynamic_cache

    def _linear_attention_uses_gdr_cache_state(self, layer_idx: int) -> bool:
        layers = getattr(self.native_text_model, "layers", ())
        if layer_idx >= len(layers):
            return False
        linear_attention = getattr(layers[layer_idx], "linear_attn", None)
        if not isinstance(linear_attention, Qwen4ExpExportableGatedDeltaNet):
            return False
        return linear_attention.linear_attention_mode != "recurrent" and bool(
            getattr(getattr(linear_attention, "chunk_scan_op", None), "state_is_cache", False)
        )

    def _flatten_dynamic_cache(self, dynamic_cache: Any) -> tuple[torch.Tensor, ...]:
        tensors: list[torch.Tensor] = []  # list[num_cache_tensors] of heterogeneous shapes
        for spec in self.cache_contract.tensors:
            cache_layer = dynamic_cache.layers[spec.layer_idx]  # layer-local cache container
            if spec.state_kind in {"delta_conv", "ple_conv", "ple_ngram_history"}:
                tensor = cache_layer.conv_states[_required_state_idx(spec)]  # contract-specific [B,...]
            elif spec.state_kind == "delta_recurrent":
                if self._linear_attention_uses_gdr_cache_state(spec.layer_idx):
                    continue
                tensor = cache_layer.recurrent_states[_required_state_idx(spec)]  # [B,NV,DK,DV]
            elif spec.state_kind == "kv_key":
                tensor = cache_layer.keys  # [B,NKV,L,DH]
            elif spec.state_kind == "kv_value":
                tensor = cache_layer.values  # [B,NKV,L,DH]
            elif spec.state_kind == "qsa_indexer_key":
                tensor = cache_layer.indexer_keys  # [B,1,L//4,DI]
            elif spec.state_kind == "qsa_indexer_tail":
                tensor = cache_layer.indexer_tail  # [B,4,DI]
            elif spec.state_kind == "qsa_key_tail":
                tensor = cache_layer.key_tail
            elif spec.state_kind == "qsa_value_tail":
                tensor = cache_layer.value_tail
            else:
                raise ValueError(f"unsupported Qwen4Exp cache state kind {spec.state_kind!r}")
            if tensor is None:
                raise RuntimeError(f"Qwen4Exp cache state {spec.name!r} was not initialized")
            tensors.append(tensor)  # append one contract-shaped cache tensor
        return tuple(tensors)  # tuple[num_cache_tensors], heterogeneous shapes


def build_qwen4_exp_text_export_wrapper(
    native_model: nn.Module,
    *,
    text_config: Any | None = None,
    batch_size: int = 1,
    input_sequence_length: int = 1,
    cache_length: int = 1,
    require_hmonnx_runtime: bool = False,
    return_hidden_states: bool = False,
) -> Qwen4ExpTextExportWrapper:
    return Qwen4ExpTextExportWrapper.from_native_model(
        native_model,
        text_config=text_config,
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
        cache_length=cache_length,
        require_hmonnx_runtime=require_hmonnx_runtime,
        return_hidden_states=return_hidden_states,
    )


def inspect_qwen4_exp_export_capability(
    config_or_model: Any,
    *,
    model: nn.Module | None = None,
    require_hmonnx_runtime: bool = False,
) -> Qwen4ExpExportCapability:
    text_config = extract_qwen4_exp_text_config(config_or_model)
    reasons: list[str] = []
    details: dict[str, Any] = {}

    transformers_version, has_qwen4_module = _transformers_qwen4_status()
    if not _version_at_least(transformers_version, MIN_TRANSFORMERS_VERSION):
        reasons.append(
            "transformers>=5.16.0 with transformers.models.qwen4_exp is required "
            f"(found {transformers_version or 'unavailable'})"
        )
    if not has_qwen4_module:
        reasons.append("transformers.models.qwen4_exp is not importable")

    hmonnx_runtime_available = _hmonnx_runtime_available()
    if require_hmonnx_runtime and not hmonnx_runtime_available:
        reasons.append("xhquant HMONNX runtime and public v2 export API are required; update xhquanttool")

    layer_types = tuple(str(item) for item in _config_get(text_config, "layer_types", ()))
    if not layer_types:
        reasons.append("text_config.layer_types is required for a fixed export ABI")
    unknown = sorted(set(layer_types).difference(SUPPORTED_LAYER_TYPES))
    if unknown:
        reasons.append(f"unsupported layer_types for Qwen4Exp cache contract: {unknown}")

    ple_layer_ids = tuple(int(item) for item in _config_get(text_config, "ple_layer_ids", ()) or ())
    hc_count = int(_config_get(text_config, "hc_count", 1) or 1)
    has_attention_indexer = any(layer_type in {FULL_ATTENTION, QWEN_SPARSE_ATTENTION} for layer_type in layer_types)

    has_raw_gptq_linears = model is not None and _contains_gptq_quant_linear(model)

    contract_dict = None
    try:
        contract_dict = Qwen4ExpCacheContract.from_text_config(text_config).to_dict()
    except Exception as exc:  # noqa: BLE001 - preflight should aggregate all gate failures.
        reasons.append(f"cache contract construction failed: {exc}")

    details.update(
        {
            "layer_types": layer_types,
            "ple_layer_ids": ple_layer_ids,
            "hc_count": hc_count,
            "has_qsa_indexer_layers": has_attention_indexer,
            "has_ple_layers": bool(ple_layer_ids),
            "has_raw_gptq_linears": has_raw_gptq_linears,
            "gptq_adapter": "xhquant.GPTQPackedLinear" if has_raw_gptq_linears else None,
        }
    )
    return Qwen4ExpExportCapability(
        supported=not reasons,
        reasons=tuple(reasons),
        transformers_version=transformers_version,
        has_transformers_qwen4_exp=has_qwen4_module,
        hmonnx_runtime_available=hmonnx_runtime_available,
        cache_contract=contract_dict,
        details=details,
    )


def extract_qwen4_exp_text_model(native_model: Any) -> nn.Module:
    candidate = native_model
    for attr in ("model", "language_model"):
        if hasattr(candidate, attr):
            candidate = getattr(candidate, attr)
    if not isinstance(candidate, nn.Module):
        raise TypeError(f"failed to extract Qwen4-Exp text nn.Module from {type(native_model).__name__}")
    return candidate


def extract_qwen4_exp_text_config(config_or_model: Any) -> Any:
    config = getattr(config_or_model, "config", config_or_model)
    return getattr(config, "text_config", config.get("text_config") if isinstance(config, Mapping) else config)


def _config_get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _required_state_idx(spec: Any) -> int:
    if spec.state_idx is None:
        raise ValueError(f"Qwen4Exp cache state {spec.name!r} has no state_idx")
    return int(spec.state_idx)


def _apply_rotary(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int | None,
    rope: nn.Module,
) -> torch.Tensor:
    """Rotate tensor [...,D] with cos/sin [B,S,R], preserving the final D-R channels."""

    if unsqueeze_dim is not None:
        cos = cos.unsqueeze(unsqueeze_dim)  # [B,1,S,R] or [B,S,1,R]
        sin = sin.unsqueeze(unsqueeze_dim)  # [B,1,S,R] or [B,S,1,R]
    rotary_dim = cos.shape[-1]  # scalar R
    if rotary_dim == tensor.shape[-1]:
        return rope(tensor, cos, sin)
    rotary, passthrough = tensor[..., :rotary_dim], tensor[..., rotary_dim:]  # [...,R], [...,D-R]
    return torch.cat((rope(rotary, cos, sin), passthrough), dim=-1)  # [...,D]


def _apply_rotary_pair(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rope: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _apply_rotary(query, cos, sin, unsqueeze_dim=1, rope=rope),  # query [B,NQ,S,DH]
        _apply_rotary(key, cos, sin, unsqueeze_dim=1, rope=rope),  # key [B,NKV,S,DH]
    )  # ([B,NQ,S,DH], [B,NKV,S,DH])


def _repeat_kv(states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat [B,NKV,S,DH] KV heads to [B,NKV*repeats,S,DH]."""

    if repeats == 1:
        return states  # [B,NKV,S,DH]
    batch_size, num_heads, seq_length, head_dim = states.shape  # scalars B,NKV,S,DH
    return (
        states[:, :, None, :, :]  # [B,NKV,1,S,DH]
        .expand(batch_size, num_heads, repeats, seq_length, head_dim)  # [B,NKV,repeats,S,DH]
        .reshape(batch_size, num_heads * repeats, seq_length, head_dim)  # [B,NKV*repeats,S,DH]
    )  # [B,NQ,S,DH] when repeats=NQ/NKV


def _shift_right_ignore_eos_exportable(module: nn.Module, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift token ids [B,SH] while preventing ngrams from crossing EOS boundaries."""

    if shift == 0:
        return token_ids  # [B,SH]
    shifted = torch.nn.functional.pad(token_ids[:, :-shift], (shift, 0), value=module.eos_token_id)  # [B,SH]
    positions = torch.arange(token_ids.shape[1], device=token_ids.device, dtype=torch.int32)  # [SH]
    valid = positions.view(1, -1) >= shift  # [1,SH]
    for offset in range(1, shift + 1):
        preceding = torch.nn.functional.pad(token_ids[:, :-offset], (offset, 0), value=module.eos_token_id)  # [B,SH]
        valid = valid & torch.eq(preceding, module.eos_token_id).logical_not()  # [B,SH]
    return torch.where(valid, shifted, torch.zeros_like(shifted) + module.eos_token_id)  # [B,SH]


_NGRAM_LIMB_BASE = 64
_NGRAM_LIMB_BITS = 6
_NGRAM_LIMB_COUNT = 11


def _multiply_base64_exportable(
    value: torch.Tensor,
    multiplier_limbs: torch.Tensor,
    token_limb_table: torch.Tensor,
    div64_table: torch.Tensor,
) -> torch.Tensor:
    """Multiply non-negative int63 values without overflowing XH2 int32."""

    value_limbs = torch.nn.functional.embedding(
        value.to(torch.long), token_limb_table
    )  # [B,SH,NL], token_limb_table [U,NL]
    carry = torch.zeros_like(value)  # [B,SH]
    product_limbs: list[torch.Tensor] = []  # list[NL] of [B,SH]
    for output_index in range(_NGRAM_LIMB_COUNT):
        total = carry  # [B,SH]
        for value_index in range(output_index + 1):
            total = (
                total + value_limbs[..., value_index] * multiplier_limbs[output_index - value_index]
            )  # [B,SH]; multiplier limb is scalar
        product_limbs.append(torch.remainder(total, _NGRAM_LIMB_BASE))  # append [B,SH]
        carry = torch.nn.functional.embedding(total.to(torch.long), div64_table).squeeze(
            -1
        )  # [B,SH], div64_table [65536,1]
    return torch.stack(product_limbs, dim=-1)  # [B,SH,NL]


def _xor_base64_exportable(
    left_limbs: torch.Tensor,
    right_limbs: torch.Tensor,
    limb_bit_table: torch.Tensor,
    bit_values: torch.Tensor,
) -> torch.Tensor:
    """Exact XOR for base-64 limbs using only small int32 arithmetic."""

    left_bits = torch.nn.functional.embedding(
        left_limbs.to(torch.long), limb_bit_table
    )  # [B,SH,NL,6], limb_bit_table [64,6]
    right_bits = torch.nn.functional.embedding(right_limbs.to(torch.long), limb_bit_table)  # [B,SH,NL,6]
    return (torch.remainder(left_bits + right_bits, 2) * bit_values).sum(dim=-1)  # [B,SH,NL]


def _base64_limbs_mod_exportable(
    limbs: torch.Tensor,
    moduli: torch.Tensor,
) -> torch.Tensor:
    """Reduce base-64 limbs modulo every PLE head vocabulary in int32."""

    remainder = torch.zeros_like(limbs[..., :1]).expand(
        *limbs.shape[:-1], moduli.shape[0]
    )  # [B,SH,GN], limbs [B,SH,NL], moduli [GN]
    for limb_index in range(_NGRAM_LIMB_COUNT - 1, -1, -1):
        remainder = torch.remainder(
            remainder * _NGRAM_LIMB_BASE + limbs[..., limb_index].unsqueeze(-1),
            moduli.view(1, 1, -1),
        )  # [B,SH,GN]
    return remainder  # [B,SH,GN]


def _ngram_embedding_forward_exportable(
    module: nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any,
) -> torch.Tensor:
    """Map input ids [B,S] plus cached context to flattened PLE embeddings [B,S,H]."""

    # XH2 integer arithmetic is int32; token ids and PLE hash limbs fit that
    # range after the explicit base-64 decomposition below.
    input_ids = input_ids.to(torch.int32)  # [B,S]
    cache_layer = past_key_values.layers[module.layer_idx]  # container: layer-local PLE cache
    previous_context = cache_layer.conv_states[2]  # [B,ngram_size-1]
    token_history = torch.cat((previous_context, input_ids), dim=-1)  # [B,SH]
    cache_layer.conv_states[2] = module.token_history_cache_slice(
        token_history,
        past_key_values.current_input_length,
    )  # [B,ngram_size-1]

    shifted_tokens = [
        module._shift_right_ignore_eos(token_history, shift) for shift in range(module.ngram_size)
    ]  # list[ngram_size] of [B,SH]
    blocks: list[torch.Tensor] = []  # list[ngram_size-1] of [B,SH,GN]
    for ngram in range(2, module.ngram_size + 1):
        start_idx = (ngram - 2) * module.heads_per_ngram  # scalar head offset
        end_idx = start_idx + module.heads_per_ngram  # scalar head end
        mixed_limbs = _multiply_base64_exportable(
            shifted_tokens[0],  # [B,SH]
            module.export_multiplier_limbs[0],  # [NL]
            module.export_token_limb_table,  # [U,NL]
            module.export_div64_table,  # [65536,1]
        )  # [B,SH,NL]
        for position in range(1, ngram):
            mixed_limbs = _xor_base64_exportable(
                mixed_limbs,  # [B,SH,NL]
                _multiply_base64_exportable(
                    shifted_tokens[position],  # [B,SH]
                    module.export_multiplier_limbs[position],  # [NL]
                    module.export_token_limb_table,  # [U,NL]
                    module.export_div64_table,  # [65536,1]
                ),  # [B,SH,NL]
                module.export_limb_bit_table,  # [64,6]
                module.export_limb_bit_values,  # [6]
            )  # [B,SH,NL]
        head_vocab_sizes = module.export_head_vocab_sizes[start_idx:end_idx]  # [GN]
        head_offsets = module.export_head_offsets[start_idx:end_idx]  # [GN]
        ngram_ids = _base64_limbs_mod_exportable(mixed_limbs, head_vocab_sizes)  # [B,SH,GN]
        blocks.append(ngram_ids + head_offsets.view(1, 1, -1))  # append [B,SH,GN]

    ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]  # [B,S,G]
    embedding_device = module.ngram_embedding.weight.device  # device scalar
    return (
        module.ngram_embedding(ngram_ids.to(embedding_device))  # [B,S,G,DE]
        .to(input_ids.device)
        .flatten(-2)
    )  # [B,S,G*DE] = [B,S,H]


def _ple_short_conv_exportable(
    module: nn.Module,
    hidden_states: torch.Tensor,
    past_key_values: Any,
) -> torch.Tensor:
    """Run PLE's dilated convolution while caching only valid prefill tokens."""

    hidden_states = hidden_states.transpose(1, 2)  # [B,HC*H,S]
    cache_layer = past_key_values.layers[module.layer_idx]  # container: layer-local PLE cache
    hidden_states = torch.cat((cache_layer.conv_states[1], hidden_states), dim=-1)  # [B,HC*H,K+S]
    cache_layer.conv_states[1] = module.ple_conv_cache_slice(
        hidden_states,
        past_key_values.current_input_length,
    )  # [B,HC*H,K]
    return torch.nn.functional.silu(module.conv1d(hidden_states)).transpose(1, 2)  # [B,S,HC*H]


def _replace_native_modules_for_export(
    model: nn.Module,
    *,
    batch_size: int,
    sequence_length: int,
    cache_length: int,
) -> None:
    adapt_qwen4_exp_rms_norms(model)
    rotary = getattr(model, "rotary_emb", None)
    if rotary is not None and type(rotary).__name__ == "Qwen4ExpTextRotaryEmbedding":
        device = rotary.inv_freq.device
        if device.type == "meta":
            device = torch.device("cpu")
        inv_freq, attention_scaling = type(rotary).compute_default_rope_parameters(rotary.config, device=device)
        model.rotary_emb = StaticMRotaryEmbedding(
            inv_freq,
            attention_scaling,
            int(rotary.config.max_position_embeddings),
            rotary.mrope_section,
        )
    for module in tuple(model.modules()):  # container traversal; no tensor shape
        linear_attention = getattr(module, "linear_attn", None)  # module: [B,S,H] -> [B,S,H]
        if linear_attention is not None and type(linear_attention).__name__ == "Qwen4ExpTextGatedDeltaNet":
            module.linear_attn = Qwen4ExpExportableGatedDeltaNet.from_native(
                linear_attention,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
        elif isinstance(linear_attention, Qwen4ExpExportableGatedDeltaNet):
            linear_attention.input_sequence_length = int(sequence_length)
    for module in tuple(model.modules()):  # container traversal; no tensor shape
        moe = getattr(module, "mlp", None)  # module: [B,S,H] -> [B,S,H]
        if moe is not None and type(moe).__name__ == "Qwen4ExpTextSparseMoeBlock":
            module.mlp = Qwen4ExpExportableSparseMoe.from_native(
                moe,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
        elif isinstance(moe, Qwen4ExpExportableSparseMoe):
            moe.batch_size = int(batch_size)
            moe.input_seq_len = int(sequence_length)
    for module in tuple(model.modules()):  # container traversal; no tensor shape
        attention = getattr(module, "self_attn", None)  # module: [B,S,H] -> [B,S,H]
        if attention is not None and type(attention).__name__ == "Qwen4ExpTextAttention":
            module.self_attn = Qwen4ExpExportableAttention(
                attention, cache_length=cache_length, sequence_length=sequence_length,
            )
        elif isinstance(attention, Qwen4ExpExportableAttention):
            attention.indexer.set_sequence_length(sequence_length)
            attention.packed_qsa.tail.input_length = int(sequence_length)
    for module in tuple(model.modules()):  # container traversal; no tensor shape
        indexer = getattr(module, "indexer", None)  # module: [B,S,H] -> [B,S,P]
        if indexer is None or isinstance(indexer, Qwen4ExpExportableQSAIndexer):
            continue
        if type(indexer).__name__ == "Qwen4ExpTextQSAIndexer":
            module.indexer = Qwen4ExpExportableQSAIndexer(indexer, sequence_length=sequence_length)
    for module in tuple(model.modules()):  # container traversal; no tensor shape
        if type(module).__name__ == "Qwen4ExpTextNGramEmbedding":
            multipliers = [int(value) for value in module.layer_multipliers.tolist()]  # list[ngram_size]
            multiplier_limbs = [
                [
                    (multiplier // (_NGRAM_LIMB_BASE**limb_index)) % _NGRAM_LIMB_BASE
                    for limb_index in range(_NGRAM_LIMB_COUNT)
                ]
                for multiplier in multipliers
            ]  # list[ngram_size][NL]
            head_vocab_sizes = [int(value) for value in module.ngram_heads_vocab_sizes.tolist()]  # list[G]
            head_offsets = [int(value) for value in module.ngram_heads_offsets.tolist()]  # list[G]
            max_export_index = max((*head_vocab_sizes, *head_offsets), default=0)  # scalar
            if max_export_index >= 1 << 31:
                raise ValueError("Qwen4Exp PLE head vocabulary indices must fit XH2 int32")
            buffer_reference = module.layer_multipliers  # [ngram_size]
            token_values = torch.arange(module.unigram_vocab_size, dtype=torch.int64)  # [U]
            token_limb_columns: list[torch.Tensor] = []  # list[NL] of [U]
            for _ in range(_NGRAM_LIMB_COUNT):
                token_limb_columns.append(torch.remainder(token_values, _NGRAM_LIMB_BASE))  # append [U]
                token_values = torch.div(token_values, _NGRAM_LIMB_BASE, rounding_mode="floor")  # [U]
            token_limb_table = torch.stack(token_limb_columns, dim=-1)  # [U,NL]
            div64_table = torch.div(
                torch.arange(65536, dtype=torch.int32),
                _NGRAM_LIMB_BASE,
                rounding_mode="floor",
            ).unsqueeze(-1)  # [65536,1]
            limb_values = torch.arange(_NGRAM_LIMB_BASE, dtype=torch.int32).unsqueeze(-1)  # [64,1]
            limb_shifts = torch.arange(_NGRAM_LIMB_BITS, dtype=torch.int32).unsqueeze(0)  # [1,6]
            limb_bit_table = torch.bitwise_and(torch.bitwise_right_shift(limb_values, limb_shifts), 1)  # [64,6]
            export_buffers = {
                "export_multiplier_limbs": buffer_reference.new_tensor(
                    multiplier_limbs, dtype=torch.int32
                ),  # [ngram_size,NL]
                "export_limb_bit_values": buffer_reference.new_tensor(
                    [1 << bit_index for bit_index in range(_NGRAM_LIMB_BITS)], dtype=torch.int32
                ),  # [6]
                "export_token_limb_table": token_limb_table.to(
                    device=buffer_reference.device,
                    dtype=torch.int32,
                ),  # [U,NL]
                "export_div64_table": div64_table.to(device=buffer_reference.device),  # [65536,1]
                "export_limb_bit_table": limb_bit_table.to(device=buffer_reference.device),  # [64,6]
                "export_head_vocab_sizes": buffer_reference.new_tensor(head_vocab_sizes, dtype=torch.int32),  # [G]
                "export_head_offsets": buffer_reference.new_tensor(head_offsets, dtype=torch.int32),  # [G]
            }  # mapping: buffer name -> tensor with shape documented above
            for buffer_name, value in export_buffers.items():  # value: one documented export-buffer tensor
                if not hasattr(module, buffer_name):
                    module.register_buffer(buffer_name, value, persistent=False)
            if not hasattr(module, "token_history_cache_slice"):
                module.token_history_cache_slice = xhnn.DynamicSlice([module.context_len], [1], [1])
            module._shift_right_ignore_eos = types.MethodType(_shift_right_ignore_eos_exportable, module)
            module.forward = types.MethodType(_ngram_embedding_forward_exportable, module)
        if type(module).__name__ == "Qwen4ExpTextPLELayer":
            if not hasattr(module, "ple_conv_cache_slice"):
                module.ple_conv_cache_slice = xhnn.DynamicSlice([module.short_conv_state_len], [2], [1])
            module._short_conv = types.MethodType(_ple_short_conv_exportable, module)


def _transformers_qwen4_status() -> tuple[str | None, bool]:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError:
        return None, False
    version = str(getattr(transformers, "__version__", "0"))
    try:
        importlib.import_module("transformers.models.qwen4_exp")
    except ImportError:
        return version, False
    return version, True


def _version_at_least(version: str | None, minimum: tuple[int, int, int]) -> bool:
    if version is None:
        return False
    parts: list[int] = []
    for raw in version.replace("-", ".").split("."):
        if not raw.isdigit():
            break
        parts.append(int(raw))
    parts.extend([0] * (len(minimum) - len(parts)))
    return tuple(parts[: len(minimum)]) >= minimum


def _hmonnx_runtime_available() -> bool:
    try:
        api = importlib.import_module("xhquant.api")
        importlib.import_module("xhquant.xhonnxruntime")
    except ImportError:
        return False
    exporter = getattr(api, "to_export_hmonnx_v2", None)
    return callable(exporter)


def _contains_gptq_quant_linear(model: nn.Module) -> bool:
    for module in model.modules():
        module_name = type(module).__name__.lower()
        module_path = type(module).__module__.lower()
        if "quantlinear" in module_name or ("gptqmodel" in module_path and "linear" in module_name):
            return True
    return False


__all__ = [
    "MIN_TRANSFORMERS_VERSION",
    "Qwen4ExpExportCapability",
    "Qwen4ExpExportUnsupportedError",
    "Qwen4ExpTextExportWrapper",
    "build_qwen4_exp_text_export_wrapper",
    "extract_qwen4_exp_text_config",
    "extract_qwen4_exp_text_model",
    "inspect_qwen4_exp_export_capability",
]
