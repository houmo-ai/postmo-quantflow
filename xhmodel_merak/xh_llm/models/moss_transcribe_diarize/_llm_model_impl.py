"""Register Text LLM wrap classes onto MOSS language-model types."""

from __future__ import annotations

from typing import Any

from xhmodel_merak.xh_llm.register import XHLLM_TRACEABLE_MODULES


def language_model_from_hf(hf_model: Any) -> Any:
    root = hf_model.model if hasattr(hf_model, "model") else hf_model
    if hasattr(root, "language_model"):
        return root.language_model
    return root


def _moss_attention_wrap():
    from xhmodel_merak.xh_llm.models.qwen3._model import _Qwen3Attention

    from .prefill_decode import parse_flash_attention

    class MossQwen3Attention(_Qwen3Attention):
        def graph_forward(
            self,
            hidden_states,
            past_seq_length=None,
            current_input_length=None,
            past_k_cache=None,
            past_v_cache=None,
            position_embeddings=None,
        ):
            if not getattr(self, "use_flash_attention", False):
                return super().graph_forward(
                    hidden_states,
                    past_seq_length=past_seq_length,
                    current_input_length=current_input_length,
                    past_k_cache=past_k_cache,
                    past_v_cache=past_v_cache,
                    position_embeddings=position_embeddings,
                )
            bsz, q_len, _ = hidden_states.size()
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            query_states = self.q_norm(
                query_states.view(bsz, q_len, self.num_heads, self.head_dim)
            ).transpose(1, 2)
            key_states = self.k_norm(
                key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            ).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = self.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=0
            )
            if self.use_cache:
                key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
                value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)
            attn_output = self.flash_attn(
                query_states,
                key_states,
                value_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.config.num_attention_heads * self.head_dim)
            return self.o_proj(attn_output), None, None

        def _setup(self, cfg):
            super()._setup(cfg)
            getter = cfg.get if hasattr(cfg, "get") else None
            flash_cfg = getter("flash_attention") if getter else getattr(cfg, "flash_attention", None)
            bits = parse_flash_attention(flash_cfg)
            self.use_flash_attention = bits is not None
            if not self.use_flash_attention:
                return self
            import math

            from xhquant.nn import FlashAttention, LLMCacheV2

            self.flash_attn = FlashAttention(
                num_heads=self.num_heads,
                scale=1 / math.sqrt(self.head_dim),
                num_kv_heads=self.num_key_value_heads,
                is_causal=True,
                q_bits=bits["q_bits"],
                k_bits=bits["k_bits"],
                v_bits=bits["v_bits"],
                p_bits=bits["p_bits"],
            )
            if self.use_cache:
                self.k_cache = LLMCacheV2(axis=cfg.kv_cache.cache_axis)
                self.v_cache = LLMCacheV2(axis=cfg.kv_cache.cache_axis)
            return self

    return MossQwen3Attention


def _moss_model_wrap():
    """Qwen3 backbone plus the in-graph output projection.

    Like every other registered model, MOSS applies ``lm_head`` inside the
    decoder graph, so the graph emits real vocabulary logits instead of the
    final hidden state.  The greedy pick stays on the host, also as elsewhere in
    this repository.  ``attach_lm_head`` binds an untied copy of the projection
    onto the language model before the wrap, keeping the Whisper encoder out of
    the traced subgraph.
    """
    from xhmodel_merak.xh_llm.models.qwen3._model import _Qwen3Model

    class MossQwen3Model(_Qwen3Model):
        def graph_forward(
            self,
            inputs_embeds=None,
            past_seq_length=None,
            current_input_length=None,
            past_key_cache=None,
            past_value_cache=None,
        ):
            hidden_states = super().graph_forward(
                inputs_embeds=inputs_embeds,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_key_cache=past_key_cache,
                past_value_cache=past_value_cache,
            )
            if getattr(self, "lm_head", None) is None:
                return hidden_states
            return self.lm_head(hidden_states)

    return MossQwen3Model


def attach_lm_head(hf_model: Any) -> bool:
    """Bind an untied ``lm_head`` copy onto the language model for in-graph logits.

    MOSS ties ``lm_head.weight`` to ``embed_tokens.weight``.  Quantizing the
    attached projection in place would also corrupt the token embedding that
    ``_wraped_post`` later deep-copies for host-side lookup, so copy the weight
    off the tied storage first.
    """
    import torch
    from torch import nn

    language = language_model_from_hf(hf_model)
    lm_head = getattr(hf_model, "lm_head", None)
    if lm_head is None or getattr(language, "lm_head", None) is not None:
        return False
    weight = lm_head.weight.detach()
    # Build on the source device and dtype. nn.Linear would otherwise allocate
    # fp32 on CPU, leaving the projection inconsistent with the module it is
    # attached to: a bf16/CUDA checkpoint gets an fp32/CPU child that only works
    # because init_wrap_model casts the whole language model afterwards, and the
    # fp32 copy of a 151936 x 1024 matrix costs 622MB of host memory first.
    untied = nn.Linear(
        weight.shape[1], weight.shape[0], bias=False,
        device=weight.device, dtype=weight.dtype,
    )
    with torch.no_grad():
        untied.weight.copy_(weight)
    language.lm_head = untied
    return True


def _moss_rotary_wrap():
    from xhmodel_merak.xh_llm.models.qwen3._model import _Qwen3RotaryEmbedding

    class MossQwen3RotaryEmbedding(_Qwen3RotaryEmbedding):
        def _setup(self, cfg=None):
            getter = cfg.get if cfg is not None and hasattr(cfg, "get") else None
            max_pe = getter("max_pe_length") if getter else getattr(cfg, "max_pe_length", None)
            context = getter("context_max_length") if getter else getattr(cfg, "context_max_length", None)
            table = int(max_pe or context or self.config.max_position_embeddings)
            if context:
                table = max(table, int(context))
            self.config.max_position_embeddings = table
            return super()._setup(cfg)

    return MossQwen3RotaryEmbedding


def restore_wrap_modules(replacements: dict[Any, Any] | None) -> None:
    if not replacements:
        return
    for source, previous in replacements.items():
        XHLLM_TRACEABLE_MODULES._registry[source] = previous


def register_wrap_modules(hf_model: Any | None = None) -> dict[Any, Any]:
    """Bind Qwen3 wrap implementations to the loaded MOSS language-model types.

    Standard Qwen3 attention/rotary types are already in the global registry.
    Replace them for the duration of this wrap so YAML ``flash_attention`` and
    ``max_pe_length`` take effect, then restore.
    """
    from xhmodel_merak.xh_llm.models.qwen3._model import (
        _Qwen3DecoderLayer,
        _Qwen3Model,
        _Qwen3RMSNorm,
        register_wrap_modules as register_qwen3_wrap_modules,
    )

    register_qwen3_wrap_modules(hf_model)
    if hf_model is None:
        return {}
    language = language_model_from_hf(hf_model)
    if not hasattr(language, "layers") or not language.layers:
        raise TypeError("MOSS language model must expose decoder layers")
    layer = language.layers[0]
    pairs = (
        (type(getattr(layer, "self_attn", None)), _moss_attention_wrap()),
        (type(layer), _Qwen3DecoderLayer),
        (type(getattr(layer, "input_layernorm", None)), _Qwen3RMSNorm),
        (type(getattr(language, "norm", None)), _Qwen3RMSNorm),
        (type(getattr(language, "rotary_emb", None)), _moss_rotary_wrap()),
        # The lm_head-aware backbone needs the projection already attached. Callers
        # that trace for export run attach_lm_head first; a bare language model
        # (CI fixtures, metadata-only paths) keeps the hidden-state Qwen3 wrap.
        (type(language), _moss_model_wrap() if getattr(language, "lm_head", None) is not None else _Qwen3Model),
    )
    replacements: dict[Any, Any] = {}
    for source, wrap in pairs:
        if source is None or source is type(None):
            continue
        previous = XHLLM_TRACEABLE_MODULES._registry.get(source)
        if previous is not None:
            replacements[source] = previous
            XHLLM_TRACEABLE_MODULES._registry[source] = wrap
            continue
        XHLLM_TRACEABLE_MODULES.register_module({source: source.__name__}, wrap)
    return replacements


__all__ = ["attach_lm_head", "language_model_from_hf", "register_wrap_modules", "restore_wrap_modules"]
