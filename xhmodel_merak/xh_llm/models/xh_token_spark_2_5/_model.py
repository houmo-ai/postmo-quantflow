import math
import types
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xhmodel_merak.xh_llm.register import XHLLM_TRACEABLE_MODULES
from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import FlashAttention, LLMCache, MaskedAdd, MaskedSoftmax, RMSNorm, SoftmaxPlus
from xhquant.utils.registry import DynamicModule

from .spark_2_5.modeling_spark import (
    Spark2_5Attention,
    Spark2_5DecoderLayer,
    Spark2_5ForCausalLM,
    Spark2_5MLP,
    Spark2_5Model,
    Spark2_5RMSNorm,
    compute_rope_cos_sin,
)


def _as_config_dict(cfg: ConfigDict | dict[str, Any]) -> ConfigDict:
    return cfg if isinstance(cfg, ConfigDict) else ConfigDict(cfg)


if TYPE_CHECKING:

    class _Spark2_5RMSNormBase(DynamicModule, Spark2_5RMSNorm):  # type: ignore[misc]
        ...

else:
    _Spark2_5RMSNormBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({Spark2_5RMSNorm: "Spark2_5RMSNorm"})
class _Spark2_5RMSNorm(_Spark2_5RMSNormBase):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.norm(hidden_states)

    def _setup(self, cfg: ConfigDict | dict[str, Any] | None = None):
        self.norm = RMSNorm(self.weight.shape[0], eps=self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


if TYPE_CHECKING:

    class _Spark2_5MLPBase(DynamicModule, Spark2_5MLP):  # type: ignore[misc]
        ...

else:
    _Spark2_5MLPBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({Spark2_5MLP: "Spark2_5MLP"})
class _Spark2_5MLP(_Spark2_5MLPBase):
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def _setup(self, cfg: ConfigDict | dict[str, Any] | None = None):
        return self


if TYPE_CHECKING:

    class _Spark2_5AttentionBase(DynamicModule, Spark2_5Attention):  # type: ignore[misc]
        ...

else:
    _Spark2_5AttentionBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({Spark2_5Attention: "Spark2_5Attention"})
class _Spark2_5Attention(_Spark2_5AttentionBase):
    def _build_sliding_attention_mask(self, attention_weights: Tensor, past_seq_length: Tensor) -> Tensor:
        query_length = attention_weights.shape[-2]
        key_length = attention_weights.shape[-1]
        if self.sliding_window is None:
            raise ValueError("Spark2_5 sliding attention requires config.sliding_window")
        past = past_seq_length.reshape(-1)[0]
        total_length = past + query_length
        key_start = torch.clamp(total_length - key_length, min=0)
        query_positions = past + torch.arange(query_length, device=attention_weights.device)
        key_positions = key_start + torch.arange(key_length, device=attention_weights.device)
        allowed = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)) & (
            key_positions.unsqueeze(0) > query_positions.unsqueeze(1) - int(self.sliding_window)
        )
        mask_value = torch.finfo(attention_weights.dtype).min
        return torch.where(
            allowed.unsqueeze(0).unsqueeze(0),
            torch.zeros((), dtype=attention_weights.dtype, device=attention_weights.device),
            torch.full((), mask_value, dtype=attention_weights.dtype, device=attention_weights.device),
        )

    def _apply_rotary_pos_emb(self, states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # rope_head_dim = cos.shape[-1]
        rope_head_dim = self.rope_head_dim  # 在_Spark2_5Model中被赋值
        if rope_head_dim == self.head_dim:
            return self.rope(states, cos, sin)
        else:
            rotary_states = states[..., :rope_head_dim]
            pass_states = states[..., rope_head_dim:]
            rotary_states = self.rope(rotary_states, cos, sin)
            return torch.cat((rotary_states, pass_states), dim=-1)

    def graph_forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        sliding_attention_mask: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
    ) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        q_k_v_states = self.q_k_v_proj(hidden_states)
        query_states, key_states, value_states = torch.split(
            q_k_v_states,
            [int(self.q_dim), int(self.kv_dim), int(self.kv_dim)],
            dim=-1,
        )
        gate_score = self.g_proj(hidden_states) if self.g_proj is not None else None

        query_states = query_states.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("Spark2_5 attention requires position_embeddings")
        cos, sin = position_embeddings
        # if past_seq_length is not None:
        #     past_seq_length = past_seq_length.to(dtype=torch.int32).reshape(-1)
        # if current_input_length is not None:
        #     current_input_length = current_input_length.to(dtype=torch.int32).reshape(-1)
        query_states = self._apply_rotary_pos_emb(query_states, cos, sin)
        key_states = self._apply_rotary_pos_emb(key_states, cos, sin)

        if self.use_cache:
            key_states = self.k_cache(
                key_states,
                past_seq_length,
                current_input_length,
                past_k_cache,
            )
            value_states = self.v_cache(
                value_states,
                past_seq_length,
                current_input_length,
                past_v_cache,
            )

        if self.use_flash_attention:
            kv_window_start_abs = None
            kv_valid_length = None
            if self.use_cache and self.layer_type == "sliding_attention":
                if past_seq_length is None:
                    raise ValueError("Spark2_5 cached sliding attention requires past_seq_length")
                # LLMCache retains up to W-1 past tokens plus the current chunk.
                # Its physical capacity includes alignment/padding, not valid tokens.
                window_start = past_seq_length - (int(self.sliding_window) - 1)
                # Max preserves integer metadata in the XH2a quantized graph.
                kv_window_start_abs = self.kv_window_start_max(window_start, torch.zeros_like(window_start))
                query_length = current_input_length if current_input_length is not None else sequence_length
                kv_valid_length = past_seq_length - kv_window_start_abs + query_length
            attention_output = self.flash_attn(
                query_states,
                key_states,
                value_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                kv_window_start_abs=kv_window_start_abs,
                kv_valid_length=kv_valid_length,
            )
            # attention_output = attention_output.transpose(1, 2).contiguous()
            # attention_output = attention_output.reshape(batch_size, sequence_length, self.attn_hidden_dim)
        else:
            query_states = query_states * self.scaling
            key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
            value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)

            attention_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            if self.layer_type == "sliding_attention":
                if sliding_attention_mask is None:
                    raise ValueError("sliding_attention_mask is required for Spark2_5 sliding attention")
                attention_weights = self.masked_add(attention_weights, sliding_attention_mask)
                attention_weights = self.softmax(attention_weights)
            else:
                if sliding_attention_mask is not None:
                    raise ValueError(
                        f"sliding_attention_mask is only valid for Spark2_5 sliding attention, got {self.layer_type=!r}"
                    )
                attention_weights = self.masked_softmax(
                    attention_weights,
                    past_seq_length,
                )
            attention_output = torch.matmul(attention_weights, value_states)

        if gate_score is not None:
            gate_score = gate_score.view(batch_size, sequence_length, self.num_heads, 1).transpose(1, 2)
            if self.gate_attn_act_mode == "sigmoid":
                gate = torch.sigmoid(gate_score)
            else:
                gate = F.silu(gate_score)
            attention_output = attention_output * gate

        attention_output = attention_output.transpose(1, 2).contiguous()

        attention_output = attention_output.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        return self.out_proj(attention_output)

    def _setup(self, cfg: ConfigDict | dict[str, Any]):
        cfg = _as_config_dict(cfg)
        if self.gate_attn_act_mode not in {"sigmoid", "silu"}:
            raise NotImplementedError(f"Unsupported Spark2_5 attention gate mode: {self.gate_attn_act_mode}")

        self.rope = xhnn.Rope()
        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.masked_add = MaskedAdd()
        self.softmax = SoftmaxPlus(dim=-1)
        self.use_cache = cfg.use_cache

        attention_max_length = -1
        if self.layer_type == "sliding_attention":
            if self.sliding_window is None:
                raise ValueError("Spark2_5 sliding attention requires config.sliding_window")
            attention_max_length = int(self.sliding_window)
        else:
            attention_max_length = -1

        flash_attention_cfg = cfg.get("flash_attention", None)
        self.use_flash_attention = False
        if flash_attention_cfg is not None:
            cfg_get = flash_attention_cfg.get if hasattr(flash_attention_cfg, "get") else None
            if cfg_get is not None:
                if "s_bits" in flash_attention_cfg:
                    raise ValueError(
                        "flash_attention.s_bits was removed: scores use FP16; configure p_bits for probabilities"
                    )
                self.use_flash_attention = bool(cfg_get("enable", False))
                legacy_manbit = cfg_get("sefp_manbit", 8)
                legacy_manbit = legacy_manbit + 1 if legacy_manbit <= 7 else legacy_manbit
                self.flash_q_bits = cfg_get("q_bits", cfg_get("q_manbit", legacy_manbit))
                self.flash_k_bits = cfg_get("k_bits", cfg_get("k_manbit", 8))
                self.flash_v_bits = cfg_get("v_bits", cfg_get("v_manbit", 8))
                self.flash_p_bits = cfg_get("p_bits", cfg_get("p_manbit", 8))
                flash_bits = {
                    "q_bits": self.flash_q_bits,
                    "k_bits": self.flash_k_bits,
                    "v_bits": self.flash_v_bits,
                    "p_bits": self.flash_p_bits,
                }
                invalid_bits = {name: value for name, value in flash_bits.items() if value not in (8, 16)}
                if invalid_bits:
                    invalid = ", ".join(f"{name}={value}" for name, value in invalid_bits.items())
                    raise ValueError(f"flash_attention q_bits/k_bits/v_bits/p_bits must be 8 or 16, got {invalid}")
        if self.layer_type == "sliding_attention":
            self.use_flash_attention = False  # sliding window 窗口一般比较小，无需开启
        if self.use_flash_attention:
            if self.use_cache and self.layer_type == "sliding_attention":
                self.kv_window_start_max = xhnn.Max()
            self.flash_attn = FlashAttention(
                num_heads=self.num_heads,
                scale=1 / math.sqrt(self.head_dim),
                num_kv_heads=self.num_key_value_heads,
                is_causal=True,
                sliding_window=attention_max_length if self.layer_type == "sliding_attention" else None,
                q_bits=self.flash_q_bits,
                k_bits=self.flash_k_bits,
                v_bits=self.flash_v_bits,
                p_bits=self.flash_p_bits,
            )

        if self.use_cache:
            cache_axis = int(cast(Any, cfg.kv_cache.cache_axis))
            self.k_cache = LLMCache(axis=cache_axis, attention_max_length=attention_max_length)
            self.v_cache = LLMCache(axis=cache_axis, attention_max_length=attention_max_length)

        return self


if TYPE_CHECKING:

    class _Spark2_5DecoderLayerBase(DynamicModule, Spark2_5DecoderLayer):  # type: ignore[misc]
        ...

else:
    _Spark2_5DecoderLayerBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({Spark2_5DecoderLayer: "Spark2_5DecoderLayer"})
class _Spark2_5DecoderLayer(_Spark2_5DecoderLayerBase):
    def graph_forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        sliding_attention_mask: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            position_embeddings=position_embeddings,
            sliding_attention_mask=sliding_attention_mask,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def _setup(self, cfg: ConfigDict | dict[str, Any] | None = None):
        if self.layer_type not in {"full_attention", "sliding_attention"}:
            raise NotImplementedError(f"Spark2_5 layer type {self.layer_type!r} is not supported by the XH wrapper")
        self.self_attn.layer_type = self.layer_type
        return self


if TYPE_CHECKING:

    class _Spark2_5ModelBase(DynamicModule, Spark2_5Model):  # type: ignore[misc]
        ...

else:
    _Spark2_5ModelBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({Spark2_5Model: "Spark2_5Model"})
class _Spark2_5Model(_Spark2_5ModelBase):
    def graph_forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        sliding_attention_mask: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ) -> Tensor:
        if past_seq_length is None:
            raise ValueError("past_seq_length is required by the Spark2_5 XH wrapper")
        if current_input_length is None:
            raise ValueError("current_input_length is required by the Spark2_5 XH wrapper")
        # past_seq_length = past_seq_length.to(dtype=torch.int32).reshape(-1)
        # current_input_length = current_input_length.to(dtype=torch.int32).reshape(-1)
        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: (
                getattr(self, f"{layer_type}_cos_slice")(getattr(self, f"{layer_type}_cos_cached"), past_seq_length),
                getattr(self, f"{layer_type}_sin_slice")(getattr(self, f"{layer_type}_sin_cached"), past_seq_length),
            )
            for layer_type in set(self.config.layer_types)
        }

        for layer_index, decoder_layer in enumerate(self.layers):
            if self.use_cache:
                if past_key_cache is None or past_value_cache is None:
                    raise ValueError("Spark2_5 cache inputs are required when use_cache=True")
                past_k_cache = past_key_cache[layer_index]
                past_v_cache = past_value_cache[layer_index]
            else:
                past_k_cache = None
                past_v_cache = None
            layer_type = decoder_layer.layer_type

            hidden_states = decoder_layer(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                sliding_attention_mask=sliding_attention_mask if layer_type == "sliding_attention" else None,
                past_k_cache=past_k_cache,
                past_v_cache=past_v_cache,
                position_embeddings=position_embeddings[decoder_layer.layer_type],
            )
            if self.max_layers > 0 and layer_index + 1 >= self.max_layers:
                break

        if self.num_logits_to_keep == 1:
            if current_input_length is None:
                raise ValueError("current_input_length is required when num_logits_to_keep=1")
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        return self.norm(hidden_states)

    def _setup(self, cfg: ConfigDict | dict[str, Any]):
        cfg = _as_config_dict(cfg)
        layer_types = set(self.config.layer_types)
        unsupported_layer_types = layer_types - {"full_attention", "sliding_attention"}
        if unsupported_layer_types:
            raise NotImplementedError(f"Unsupported Spark2_5 layer types: {sorted(unsupported_layer_types)}")
        max_position_embeddings = cfg.get("max_position_embeddings", 32768)
        max_position_embeddings = max(max_position_embeddings, cfg.context_max_length)
        context_max_length = int(cast(Any, max_position_embeddings))
        embedding_weight = cast(Tensor, self.embedding.weight)
        positions = torch.arange(context_max_length, dtype=torch.long, device=embedding_weight.device)

        input_sequence_length = int(cast(Any, cfg.input_sequence_length))
        for layer_type in layer_types:
            rope_theta = self.config.get_rope_theta(layer_type)
            partial_rotary_factor = self.config.get_partial_rotary_factor(layer_type)
            cos_cached, sin_cached = compute_rope_cos_sin(
                positions,
                self.config.head_dim,
                rope_theta,
                partial_rotary_factor=partial_rotary_factor,
                device=str(positions.device),
            )
            rope_head_dim = int(self.config.head_dim * partial_rotary_factor)
            for decode_layer in self.layers:
                if decode_layer.layer_type == layer_type:
                    decode_layer.self_attn.rope_head_dim = rope_head_dim

            cos_cached = cos_cached.unsqueeze(0).unsqueeze(0)
            sin_cached = sin_cached.unsqueeze(0).unsqueeze(0)
            self.register_buffer(f"{layer_type}_cos_cached", cos_cached, persistent=False)
            self.register_buffer(f"{layer_type}_sin_cached", sin_cached, persistent=False)

            cos_slice = xhnn.DynamicSlice([input_sequence_length], [2], [1])
            sin_slice = xhnn.DynamicSlice([input_sequence_length], [2], [1])

            def _update_slice(module: xhnn.DynamicSlice, update_cfg: Optional[dict] = None):
                if update_cfg is not None:
                    module.valid_length = [int(update_cfg["input_sequence_length"])]

            cos_slice._update_cfg = types.MethodType(_update_slice, cos_slice)  # type: ignore[assignment]
            sin_slice._update_cfg = types.MethodType(_update_slice, sin_slice)  # type: ignore[assignment]
            setattr(self, f"{layer_type}_cos_slice", cos_slice)
            setattr(self, f"{layer_type}_sin_slice", sin_slice)

        self.use_cache = cfg.use_cache
        self.num_logits_to_keep = cfg.num_logits_to_keep
        if self.num_logits_to_keep not in {0, 1}:
            raise ValueError("Spark2_5 XH wrapper supports num_logits_to_keep values 0 and 1 only")

        configured_max_layers = cfg.get("max_layers", -1)
        self.max_layers = 1 if cfg.get("only_first_block", False) else (configured_max_layers or -1)
        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(1, input_sequence_length)

        def _update_gather(module: xhnn.BatchGather, update_cfg: Optional[dict] = None):
            if update_cfg is None:
                return
            module.update_offset_indices(
                int(update_cfg.get("batch_size", 1)),
                int(update_cfg["input_sequence_length"]),
            )

        self.llm_gather._update_cfg = types.MethodType(_update_gather, self.llm_gather)  # type: ignore[assignment]
        return self


if TYPE_CHECKING:

    class _Spark2_5ForCausalLMBase(DynamicModule, Spark2_5ForCausalLM):  # type: ignore[misc]
        ...

else:
    _Spark2_5ForCausalLMBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({Spark2_5ForCausalLM: "Spark2_5ForCausalLM"})
class _Spark2_5ForCausalLM(_Spark2_5ForCausalLMBase):
    def graph_forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        sliding_attention_mask: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ) -> Tensor:
        hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            sliding_attention_mask=sliding_attention_mask,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        if self.config.tie_word_embeddings:
            # embedding_weight = cast(Tensor, self.model.embedding.weight)
            # return F.linear(hidden_states.to(dtype=embedding_weight.dtype), embedding_weight)
            raise NotImplementedError(
                "Spark2_5 XH wrapper does not support tie_word_embeddings=True in the causal LM output path"
            )
        return self.lm_head(hidden_states)

    def _setup(self, cfg: ConfigDict | dict[str, Any] | None = None):
        return self


def register_wrap_modules(hf_model: Spark2_5ForCausalLM | None = None) -> None:
    return None
