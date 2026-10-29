# Copyright 2025 HOUMO AI. SPDX-License-Identifier: Apache-2.0
"""Bonsai specializes Qwen's GDN core without changing shared normalization.

Cache setup, split projections and configuration updates remain inherited.
The forward override selects Bonsai's FP32 delta-rule helpers.
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..qwen3_5._llm_model_impl import _Qwen3_5GatedDeltaNet
from ..qwen3_5._hybrid_gated_delta_net import (
    _align_linear_cache_args,
    _manual_depthwise_conv1d_tail,
    _normalize_linear_conv_cache_rank,
    _resolve_linear_attn_mask_views,
    _resolve_python_int_length,
    _split_linear_qkv_tensor,
)
from .delta_rule import torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule


class BonsaiGatedDeltaNet(_Qwen3_5GatedDeltaNet):
    def _setup(self, cfg):
        parent = getattr(self, "in_proj_qkv", None)
        logical_bits = getattr(parent, "logical_bits", None)
        super()._setup(cfg)
        if self.split_conv_cache and logical_bits is not None:
            for child in (self.in_proj_q, self.in_proj_k, self.in_proj_v):
                child.logical_bits = logical_bits
        return self

    def graph_forward(self, hidden_states, conv_cache=None, recurrent_state=None,
                      linear_attn_mask=None, current_input_length=None):
        hidden, conv, state = BonsaiGatedDeltaNet.forward(
            self, hidden_states, conv_cache, recurrent_state, linear_attn_mask, current_input_length)
        if not self.split_conv_cache:
            return hidden, conv, state
        states = state if isinstance(state, (tuple, list)) else (() if state is None else (state,))
        return hidden, *conv, *states

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_cache: Optional[Tensor] = None,
        recurrent_state: Optional[Tensor] = None,
        linear_attn_mask: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
    ):
        mask, channel_mask, state_mask = _resolve_linear_attn_mask_views(
            linear_attn_mask,
            hidden_states.dtype,
        )
        batch_size, seq_len, _ = hidden_states.shape
        query, key, value, z, b, a = self._project_qkvzba(hidden_states)
        use_recurrent = self.linear_attention_mode == "recurrent"
        if self.linear_attention_mode == "auto" and current_input_length is not None:
            resolved_len = _resolve_python_int_length(current_input_length)
            use_recurrent = (resolved_len == 1) if resolved_len is not None else False
        _verify_intermediates = getattr(self, "_verify_output_intermediates", False)
        resolved_seq_len = _resolve_python_int_length(seq_len)
        # Speculative/MTP verification can feed one hidden-state seed while
        # tracing a fixed N-token recurrent transition.  Qwen3.5-MoE already
        # relied on this contract; keeping it in the canonical path makes the
        # behavior available to Qwen3-Next without a family-specific fork.
        _expand_verify_tokens = (
            _verify_intermediates and self.input_sequence_length > 1 and use_recurrent and resolved_seq_len == 1
        )
        attn_seq_len = self.input_sequence_length if _expand_verify_tokens else seq_len
        if self.split_conv_cache:
            # === Split conv_cache path (3 separate q/k/v tensors) ===
            query_states = query.reshape(batch_size, seq_len, self.key_dim)
            key_states = key.reshape(batch_size, seq_len, self.key_dim)
            value_states = value.reshape(batch_size, seq_len, self.value_dim)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            if isinstance(conv_cache, (list, tuple)):
                if len(conv_cache) != 3:
                    raise RuntimeError(f"Expected 3 conv caches for linear attention, got {len(conv_cache)}")
                conv_cache_q = _normalize_linear_conv_cache_rank(conv_cache[0])
                conv_cache_k = _normalize_linear_conv_cache_rank(conv_cache[1])
                conv_cache_v = _normalize_linear_conv_cache_rank(conv_cache[2])
            else:
                conv_cache, recurrent_state = _align_linear_cache_args(
                    conv_cache,
                    recurrent_state,
                    self.conv_dim,
                    self.num_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                )
                assert conv_cache is not None, "conv_cache is required"
                conv_cache = _normalize_linear_conv_cache_rank(conv_cache)
                conv_cache_q, conv_cache_k, conv_cache_v = _split_linear_qkv_tensor(
                    conv_cache,
                    self.key_dim,
                    self.value_dim,
                    dim=1,
                )
            query_states_new = torch.cat([conv_cache_q, query_states], dim=-1).to(self.conv1d_q.weight.dtype)
            key_states_new = torch.cat([conv_cache_k, key_states], dim=-1).to(self.conv1d_k.weight.dtype)
            value_states_new = torch.cat([conv_cache_v, value_states], dim=-1).to(self.conv1d_v.weight.dtype)
            if _verify_intermediates and self.input_sequence_length > 1 and use_recurrent:
                _kernel = int(self.conv_kernel_size)
                conv_cache_out = (
                    tuple(query_states_new[..., 1 + t : 1 + t + _kernel] for t in range(self.input_sequence_length))
                    + tuple(key_states_new[..., 1 + t : 1 + t + _kernel] for t in range(self.input_sequence_length))
                    + tuple(value_states_new[..., 1 + t : 1 + t + _kernel] for t in range(self.input_sequence_length))
                )
            else:
                conv_cache_out = (
                    self.conv_cache_slice(query_states_new, current_input_length),
                    self.conv_cache_slice(key_states_new, current_input_length),
                    self.conv_cache_slice(value_states_new, current_input_length),
                )
            if self.use_manual_depthwise_conv1d:
                query_states = _manual_depthwise_conv1d_tail(
                    query_states_new,
                    self.conv1d_q_manual_weight,
                    getattr(self, "conv1d_q_manual_bias", None),
                    self.input_sequence_length,
                )
                key_states = _manual_depthwise_conv1d_tail(
                    key_states_new,
                    self.conv1d_k_manual_weight,
                    getattr(self, "conv1d_k_manual_bias", None),
                    self.input_sequence_length,
                )
                value_states = _manual_depthwise_conv1d_tail(
                    value_states_new,
                    self.conv1d_v_manual_weight,
                    getattr(self, "conv1d_v_manual_bias", None),
                    self.input_sequence_length,
                )
            else:
                _k = self.conv_kernel_size
                _l = self.input_sequence_length
                query_states = self.conv1d_q(query_states_new)[:, :, _k : _k + _l]
                key_states = self.conv1d_k(key_states_new)[:, :, _k : _k + _l]
                value_states = self.conv1d_v(value_states_new)[:, :, _k : _k + _l]
            query_states = F.silu(query_states).to(query_states.dtype)
            key_states = F.silu(key_states).to(key_states.dtype)
            value_states = F.silu(value_states).to(value_states.dtype)
            mask_qkv = channel_mask
            query_states = query_states * mask_qkv
            key_states = key_states * mask_qkv
            value_states = value_states * mask_qkv
            query = query_states.transpose(1, 2).reshape(batch_size, attn_seq_len, -1, self.head_k_dim)
            key = key_states.transpose(1, 2).reshape(batch_size, attn_seq_len, -1, self.head_k_dim)
            value = value_states.transpose(1, 2).reshape(batch_size, attn_seq_len, -1, self.head_v_dim)
        else:
            # === Merged conv_cache path (single tensor, default) ===
            mixed_qkv = torch.cat(
                (
                    query.reshape(batch_size, seq_len, self.key_dim),
                    key.reshape(batch_size, seq_len, self.key_dim),
                    value.reshape(batch_size, seq_len, self.value_dim),
                ),
                dim=-1,
            ).transpose(1, 2)
            conv_cache, recurrent_state = _align_linear_cache_args(
                conv_cache,
                recurrent_state,
                self.conv_dim,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            )
            assert conv_cache is not None, "conv_cache is required"
            conv_cache = _normalize_linear_conv_cache_rank(conv_cache)
            hidden_states_new = torch.cat([conv_cache, mixed_qkv], dim=-1).to(self.conv1d.weight.dtype)
            if _verify_intermediates and self.input_sequence_length > 1 and use_recurrent:
                _kernel = int(self.conv_kernel_size)
                conv_cache_out = tuple(
                    hidden_states_new[..., 1 + t : 1 + t + _kernel] for t in range(self.input_sequence_length)
                )
            else:
                conv_cache_out = self.conv_cache_slice(hidden_states_new, current_input_length)
            if self.use_manual_depthwise_conv1d:
                conv_out = _manual_depthwise_conv1d_tail(
                    hidden_states_new,
                    self.conv1d_manual_weight,
                    getattr(self, "conv1d_manual_bias", None),
                    self.input_sequence_length,
                )
            else:
                _k = self.conv_kernel_size
                _l = self.input_sequence_length
                conv_out = self.conv1d(hidden_states_new)[:, :, _k : _k + _l]
            mixed_qkv = F.silu(conv_out).to(mixed_qkv.dtype)
            mask_qkv = channel_mask
            mixed_qkv = mixed_qkv * mask_qkv
            mixed_qkv = mixed_qkv.transpose(1, 2)
            query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            query = query.reshape(batch_size, attn_seq_len, -1, self.head_k_dim)
            key = key.reshape(batch_size, attn_seq_len, -1, self.head_k_dim)
            value = value.reshape(batch_size, attn_seq_len, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = self.A_log_exp * F.softplus(a + self.dt_bias)
        mask_qkv = state_mask
        value = value * mask_qkv
        beta = beta * mask
        g = g * mask
        if _expand_verify_tokens:
            beta = beta.repeat_interleave(self.input_sequence_length, dim=1)
            g = g.repeat_interleave(self.input_sequence_length, dim=1)
            mask_qkv = mask_qkv.repeat_interleave(self.input_sequence_length, dim=1)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        if use_recurrent:
            if _verify_intermediates and self.input_sequence_length > 1:
                _recurrent_snapshots = []
                _current_rs = recurrent_state
                _core_parts = []
                for _t in range(self.input_sequence_length):
                    _out_t, _current_rs = torch_recurrent_gated_delta_rule(
                        query[:, _t : _t + 1],
                        key[:, _t : _t + 1],
                        value[:, _t : _t + 1],
                        g=g[:, _t : _t + 1],
                        beta=beta[:, _t : _t + 1],
                        mask_qkv=mask_qkv[:, _t : _t + 1],
                        initial_state=_current_rs,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        num_heads=self.chunk_num_heads,
                        k_head_dim=self.chunk_k_head_dim,
                        v_head_dim=self.chunk_v_head_dim,
                        batch_size=self.batch_size,
                        scale=self.chunk_scale,
                        sequence_length=1,
                        recurrent_scan_op=self.recurrent_scan_op,
                    )
                    _core_parts.append(_out_t)
                    _recurrent_snapshots.append(_current_rs)
                core_attn_out = torch.cat(_core_parts, dim=1)
                last_recurrent_state = _recurrent_snapshots[-1]
            else:
                core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    mask_qkv=mask_qkv,
                    initial_state=recurrent_state,
                    output_final_state=self.use_cache,
                    use_qk_l2norm_in_kernel=True,
                    num_heads=self.chunk_num_heads,
                    k_head_dim=self.chunk_k_head_dim,
                    v_head_dim=self.chunk_v_head_dim,
                    batch_size=self.batch_size,
                    scale=self.chunk_scale,
                    # Decode is normally one token, while MTP decode traces a
                    # fixed multi-token input.  The recurrent helper invokes
                    # GDRRecurrentScan once per token, preserving the same
                    # state transition contract in both cases.
                    sequence_length=self.input_sequence_length,
                    recurrent_scan_op=self.recurrent_scan_op,
                )
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                mask_qkv=mask_qkv,
                initial_state=recurrent_state,
                output_final_state=self.use_cache,
                use_qk_l2norm_in_kernel=True,
                chunk_size=self.linear_chunk_size,
                chunk_mask_incl=self.chunk_mask_incl,
                chunk_mask_strict=self.chunk_mask_strict,
                chunk_eye_matrix=self.chunk_eye_matrix,
                chunk_eye_8_batched=self.chunk_eye_8_batched,
                cumsum_matrix=self.chunk_cumsum_matrix,
                input_sequence_length=self.input_sequence_length,
                num_heads=self.chunk_num_heads,
                k_head_dim=self.chunk_k_head_dim,
                v_head_dim=self.chunk_v_head_dim,
                batch_size=self.batch_size,
                scale=self.chunk_scale,
                chunk_row_masks=self.chunk_row_masks,
                cumsum_matmul=self.cumsum_matmul,
                block_tri_inverse_op=self.block_tri_inverse_op,
                chunk_scan_op=self.chunk_scan_op,
            )
        # GDRChunkScan variants whose ``state_is_cache`` metadata is true
        # update the CacheTensor input in place.  Exporting the same state as
        # an additional graph output would duplicate the cache contract and
        # can make HMONNX allocate a second recurrent-state buffer.  Derive
        # suppression from the concrete op class rather than a model-family
        # switch; the explicit cfg flag remains available for legacy ops.
        chunk_state_is_cache = bool(getattr(getattr(self, "chunk_scan_op", None), "state_is_cache", False))
        suppress_recurrent_state_outputs = (
            getattr(self, "suppress_recurrent_state_outputs", False) or chunk_state_is_cache
        ) and not use_recurrent
        if suppress_recurrent_state_outputs:
            recurrent_state_out = None
        elif _verify_intermediates and self.input_sequence_length > 1 and use_recurrent:
            recurrent_state_out = tuple(_recurrent_snapshots)
        else:
            recurrent_state_out = last_recurrent_state if last_recurrent_state is not None else recurrent_state
        # b_sz, s, n, h = z.shape
        # core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        # z = z.reshape(-1, z.shape[-1])
        if _expand_verify_tokens:
            z = z.repeat_interleave(self.input_sequence_length, dim=1)
        core_attn_out = self.norm(core_attn_out, z)
        # core_attn_out = core_attn_out.reshape(b_sz, s, n, h)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        core_attn_out = core_attn_out.to(self.out_proj.weight.dtype)
        output = self.out_proj(core_attn_out)
        output = output * mask
        return output, conv_cache_out, recurrent_state_out
