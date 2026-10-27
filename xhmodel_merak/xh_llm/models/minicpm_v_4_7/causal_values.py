"""Query-split P x V for MiniCPM decode and speculative verify.

Zero attention probabilities do not isolate the shared HMFP exponent used to
quantize V along the token axis. Each query therefore multiplies only the V
prefix that is visible at that step, including after a rejected draft.
"""

from __future__ import annotations

from functools import cache

import torch

from xhquant import nn as xhnn


class _CausalValueProduct(torch.nn.Module):
    def __init__(self, cache_capacity, query_capacity, kv_groups, *, device, dtype=torch.float16):
        super().__init__()
        self.query_capacity = int(query_capacity)
        self.kv_groups = int(kv_groups)
        self.visible = xhnn.LessOrEqual()
        self.select_values = xhnn.Where()
        self.register_buffer("zero", torch.zeros((), device=device, dtype=dtype), persistent=False)
        self.register_buffer(
            "key_positions",
            torch.arange(int(cache_capacity), dtype=torch.int32, device=device).view(1, 1, -1, 1),
            persistent=False,
        )

    def forward(self, probabilities, values, past_seq_length):
        past = past_seq_length.reshape(-1, 1, 1, 1)
        outputs = []
        for index in range(self.query_capacity):
            visible_values = self.select_values(
                self.visible(self.key_positions, past + index),
                values,
                self.zero,
            )
            visible_values = torch.repeat_interleave(visible_values, self.kv_groups, dim=1)
            outputs.append(torch.matmul(probabilities[:, :, index : index + 1, :], visible_values))
        return torch.cat(outputs, dim=2)


def install_causal_value_product(module, cfg) -> None:
    """Attach the product without replacing Qwen's registered attention class."""

    enabled = (
        bool(getattr(module, "use_cache", False))
        and not bool(getattr(module, "use_flash_attention", False))
        and bool(cfg.get("mask_causal_values", False))
        and cfg.get("linear_attention_mode") == "recurrent"
    )
    if not enabled:
        module.causal_value_product = None
        return
    module.causal_value_product = _CausalValueProduct(
        cfg.context_max_length,
        cfg.input_sequence_length,
        module.num_key_value_groups,
        device=module.q_proj.weight.device,
        dtype=module.q_proj.weight.dtype,
    )


def enable_minicpm_causal_values(module) -> None:
    """Specialize one wrapped attention; Qwen classes and registries stay intact."""

    if isinstance(module, _MiniCPMCausalAttentionMixin):
        return
    module.causal_value_product = None
    module.__class__ = _causal_attention_class(type(module))
    # BaseLLMModel binds graph_forward on each instance before this hook.
    # Resolve through the derived class so copies use their own state/config.
    module.__dict__.pop("forward", None)


def _causal_forward(
    self,
    hidden_states,
    past_seq_length=None,
    current_input_length=None,
    attention_mask=None,
    position_embeddings=None,
    past_k_cache=None,
    past_v_cache=None,
    **kwargs,
):
    if self.causal_value_product is None:
        return super(_MiniCPMCausalAttentionMixin, self).forward(
            hidden_states,
            past_seq_length,
            current_input_length,
            attention_mask,
            position_embeddings,
            past_k_cache,
            past_v_cache,
            **kwargs,
        )
    import torch

    batch, query_length, _ = hidden_states.size()
    projected = self.q_proj(hidden_states).view(batch, query_length, self.num_heads, self.head_dim * 2)
    query_states, gate = torch.split(projected, self.head_dim, dim=-1)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = self.q_norm(query_states.view(batch, query_length, self.num_heads, self.head_dim)).transpose(1, 2)
    key_states = self.k_norm(
        key_states.view(batch, query_length, self.num_key_value_heads, self.head_dim)
    ).transpose(1, 2)
    value_states = value_states.view(batch, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    if position_embeddings is not None:
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, *position_embeddings)
    if self.use_cache:
        key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
        value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)
    query_states = query_states * self.kv_scale
    key_states = torch.repeat_interleave(key_states.transpose(2, 3), self.num_key_value_groups, dim=1)
    weights = self.masked_softmax(torch.matmul(query_states, key_states), past_seq_length)
    output = self.causal_value_product(weights, value_states, past_seq_length)
    output = output.transpose(1, 2).reshape(batch, query_length, self.attn_hidden_dim)
    return self.o_proj(output * torch.sigmoid(gate.reshape(batch, query_length, -1)))


class _MiniCPMCausalAttentionMixin:
    def _update_cfg(self, cfg):
        parent_update = getattr(super(), "_update_cfg", None)
        if parent_update is not None:
            parent_update(cfg)
        install_causal_value_product(self, cfg)

    forward = _causal_forward
    graph_forward = _causal_forward


@cache
def _causal_attention_class(base):
    return type(f"MiniCPM{base.__name__}", (_MiniCPMCausalAttentionMixin, base), {"__module__": __name__})
