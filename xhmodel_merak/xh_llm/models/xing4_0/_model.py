"""FX-traceable DynamicModule adapters for Xing4.0 MLA + mHC."""

from __future__ import annotations

import math
import sys
import types
import warnings
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import LLMCache, MaskedSoftmax, RMSNorm
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
from ..deepseek_v4.norm import build_xh_rms_norm


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    if hasattr(cfg, "get"):
        value = cfg.get(name, default)
        return default if value is None else value
    value = getattr(cfg, name, default)
    return default if value is None else value


def _register_or_replace_traceable(mapping: dict[type, str]):
    def decorator(dynamic_cls):
        for hf_cls, key in mapping.items():
            XHLLM_TRACEABLE_MODULES._registry[hf_cls] = dynamic_cls
            XHLLM_TRACEABLE_MODULES._key_registry[hf_cls] = key
            XHLLM_TRACEABLE_MODULES._dynamic_classes.pop(hf_cls, None)
        return dynamic_cls

    return decorator


def _find_remote_classes(hf_model: nn.Module) -> dict[str, type[nn.Module]]:
    classes: dict[str, type[nn.Module]] = {}
    for module in hf_model.modules():
        name = type(module).__name__
        if name.startswith("Xing4_0"):
            classes.setdefault(name, type(module))
    # Top-level causal LM may not appear under named_modules() as a child of itself.
    root_name = type(hf_model).__name__
    if root_name.startswith("Xing4_0"):
        classes.setdefault(root_name, type(hf_model))
    return classes


def _maybe_register_quant_weight(linear: nn.Linear, quant_weight: Tensor | None) -> None:
    if quant_weight is not None:
        assert quant_weight.dtype in (torch.int8, torch.int16)
        linear.register_buffer("quant_weight", quant_weight.contiguous())


def _pack_expert_projection(
    experts: nn.ModuleList,
    projection_name: str,
) -> tuple[Tensor, Tensor | None]:
    """Pack one routed-expert projection and preserve optional GPTQ codes."""
    linears = [getattr(expert, projection_name) for expert in experts]
    packed_weight = torch.stack([linear.weight.data for linear in linears], dim=0)
    quant_weights = [getattr(linear, "quant_weight", None) for linear in linears]
    has_quant_weights = [torch.is_tensor(quant_weight) for quant_weight in quant_weights]
    if any(has_quant_weights) and not all(has_quant_weights):
        missing = [str(index) for index, present in enumerate(has_quant_weights) if not present]
        raise RuntimeError(
            f"Xing4.0 MoE {projection_name} quant_weight must exist for every expert; "
            f"missing experts: {', '.join(missing)}"
        )
    if not all(has_quant_weights):
        return packed_weight, None

    for expert_index, (linear, quant_weight) in enumerate(zip(linears, quant_weights, strict=True)):
        if tuple(quant_weight.shape) != tuple(linear.weight.shape):
            raise RuntimeError(
                f"Xing4.0 MoE expert {expert_index} {projection_name}.quant_weight shape "
                f"{tuple(quant_weight.shape)} does not match weight shape {tuple(linear.weight.shape)}"
            )
    return packed_weight, torch.stack([quant_weight.data for quant_weight in quant_weights], dim=0)


class _Xing4_0RMSNorm(DynamicModule):
    def graph_forward(self, hidden_states: Tensor) -> Tensor:
        return self.norm(hidden_states)

    def _setup(self, cfg: dict | None = None):
        del cfg
        hidden_size = int(self.weight.shape[0])
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


class _Xing4_0RotaryEmbedding(DynamicModule):
    def _setup(self, cfg: dict | None = None):
        max_seq_len = int(
            _cfg_get(
                cfg,
                "max_pe_length",
                _cfg_get(cfg, "context_max_length", getattr(self, "max_seq_len_cached", 4096)),
            )
        )
        max_seq_len = max(1, max_seq_len)
        self.max_seq_len_cached = max_seq_len
        self._setup_cos_sin_cache(seq_len=max_seq_len, dtype=self.inv_freq.dtype)
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()
        return self

    def _setup_cos_sin_cache(self, seq_len: int, dtype: torch.dtype) -> None:
        inv_freq = self.inv_freq.float()
        device = inv_freq.device
        if torch.cuda.is_available() and device.type != "cuda":
            inv_freq = inv_freq.cuda()
            device = inv_freq.device
        positions = torch.arange(seq_len, device=device, dtype=torch.float32).view(seq_len, 1)
        freqs = positions * inv_freq.view(1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * float(self.attention_scaling)).to(dtype=dtype)
        sin = (emb.sin() * float(self.attention_scaling)).to(dtype=dtype)
        # Match Glm4 MLA cache layout: [1, 1, S, D]
        self.register_buffer("cos_cached", cos.unsqueeze(0).unsqueeze(0).contiguous(), persistent=False)
        self.register_buffer("sin_cached", sin.unsqueeze(0).unsqueeze(0).contiguous(), persistent=False)

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.inv_freq = self.inv_freq.to(dtype)
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=dtype)


class _Xing4_0HyperConnection(DynamicModule):
    """FX-safe mHC using MHCSinkhorn; preserve Xing4 clamp / return order."""

    def graph_forward(self, hidden_streams: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Keep Xing4 clamp/return order, but avoid FX-unsafe ``view(*shape[:-1], ...)``.
        ori_dtype = hidden_streams.dtype
        hc = int(self.hc_mult)
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mixes = self.hc_proj(flat.to(ori_dtype)).float()
        pre_w = mixes[..., :hc]
        post_w = mixes[..., hc : 2 * hc]
        comb_w = mixes[..., 2 * hc :]
        pre_b = self.hc_base[:hc]
        post_b = self.hc_base[hc : 2 * hc]
        comb_b = self.hc_base[2 * hc :]
        pre_scale = self.hc_scale[0]
        post_scale = self.hc_scale[1]
        comb_scale = self.hc_scale[2]
        pre = torch.sigmoid(pre_w * pre_scale + pre_b)
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.reshape(comb_w.shape[0], comb_w.shape[1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb_logits = torch.clamp(
            comb_logits,
            min=self.mhc_h_res_clamp_min,
            max=self.mhc_h_res_clamp_max,
        )
        comb = self.sinkhorn(comb_logits)
        collapsed = (pre.unsqueeze(-1).to(ori_dtype) * hidden_streams).sum(dim=2)
        return post.to(ori_dtype), comb.to(ori_dtype), collapsed.to(ori_dtype)

    def _setup(self, cfg: dict | None = None):
        del cfg
        hidden_flat = int(self.hc_fn.shape[1])
        self.hc_proj = nn.Linear(hidden_flat, int(self.hc_fn.shape[0]), bias=False)
        self.hc_proj.weight = self.hc_fn
        self.input_norm = build_xh_rms_norm(hidden_flat, float(self.input_norm.eps))
        self.sinkhorn = xhnn.MHCSinkhorn(int(self.hc_sinkhorn_iters), float(self.hc_eps))
        return self


class _Xing4_0Attention(DynamicModule):
    def rotate_half(self, x: Tensor) -> Tensor:
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
        if self.enable_rope:
            return self.rope(q, cos, sin), self.rope(k, cos, sin)
        return (q * cos) + (self.rotate_half(q) * sin), (k * cos) + (self.rotate_half(k) * sin)

    def apply_rotary_pos_emb_interleave(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
        b, h, s, d = q.shape
        q = q.view(b, h, s, -1, 2).transpose(4, 3).reshape(b, h, s, d)
        b, h, s, d = k.shape
        k = k.view(b, h, s, -1, 2).transpose(4, 3).reshape(b, h, s, d)
        return self.apply_rotary_pos_emb(q, k, cos, sin)

    def graph_forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, None, None]:
        batch_size, seq_length, _ = hidden_states.shape

        if self.q_lora_rank is None:
            inp = hidden_states
        else:
            inp = self.q_a_layernorm(self.q_a_proj(hidden_states))

        q_pe = self.q_rope_proj(inp).view(batch_size, seq_length, self.num_heads, self.qk_rope_head_dim)
        q_content = self.q_absorbed_proj(inp).view(batch_size, seq_length, self.num_heads, self.kv_lora_rank)

        k_latent = self.kv_a_layernorm(self.kv_a_proj_latent(hidden_states))
        k_rot = self.kv_a_proj_rope(hidden_states).view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_pe = q_pe.transpose(1, 2)
        if self.config.rope_interleave:
            q_pe, k_rot = self.apply_rotary_pos_emb_interleave(q_pe, k_rot, cos, sin)
        else:
            q_pe, k_rot = self.apply_rotary_pos_emb(q_pe, k_rot, cos, sin)
        q_pe = q_pe.transpose(1, 2)

        query_states = torch.cat((q_content, q_pe), dim=-1).transpose(1, 2) * self.kv_scale
        k_rot = k_rot.squeeze(1)
        key_states = torch.cat((k_latent, k_rot), dim=-1).unsqueeze(1)
        value_states = k_latent.unsqueeze(1)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        key_states = key_states.transpose(2, 3)
        attn_weights = torch.matmul(query_states, key_states)
        attn_weights = self.masked_softmax(attn_weights, past_seq_length)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None, None

    def _setup(self, cfg: dict | None = None):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        if not hasattr(self, "num_heads"):
            self.num_heads = int(self.config.num_attention_heads)
        if not hasattr(self, "hidden_size"):
            self.hidden_size = int(self.config.hidden_size)

        self.enable_rope = bool(_cfg_get(cfg, "enable_rope", True))
        if self.enable_rope:
            self.rope = xhnn.Rope()

        # RoPE rotate_half slices operate on the rope sub-dimension.
        rope_dim = int(self.qk_rope_head_dim)
        self.slice_1 = xhnn.Slice([0], [rope_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([rope_dim // 2], [sys.maxsize], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.use_cache = bool(_cfg_get(cfg, "use_cache", True))
        if self.use_cache:
            cache_axis = int(_cfg_get(_cfg_get(cfg, "kv_cache", {}), "cache_axis", 2))
            self.k_cache = LLMCache(axis=cache_axis, attention_max_length=-1)
            self.v_cache = LLMCache(axis=cache_axis, attention_max_length=-1)
        else:
            self.k_cache = None
            self.v_cache = None

        # Prefer the HF Attention.scaling (includes yarn mscale) when present.
        self.kv_scale = float(getattr(self, "scaling", 1.0 / math.sqrt(float(self.qk_head_dim))))

        layout = getattr(self.config, "mla_weight_layout", "hf_unabsorbed")
        if layout == "xh2_absorbed_v1":
            transform = getattr(self.config, "mla_transform", None)
            if not isinstance(transform, dict) or any(
                transform.get(key) != value
                for key, value in {
                    "version": 1,
                    "matmul_dtype": "float32",
                    "cache_layout": "separate_k_latent_rope_v_latent",
                }.items()
            ):
                raise ValueError("Invalid Xing MLA transform metadata")
            if any(hasattr(self, name) for name in ("q_proj", "q_b_proj", "kv_b_proj")):
                raise ValueError("Xing MLA checkpoint mixes absorbed and unabsorbed projections")
            query_width = self.hidden_size if self.q_lora_rank is None else self.q_lora_rank
            expected = {
                "q_absorbed_proj": (self.num_heads * self.kv_lora_rank, query_width),
                "q_rope_proj": (self.num_heads * self.qk_rope_head_dim, query_width),
                "o_proj": (self.hidden_size, self.num_heads * self.kv_lora_rank),
                "kv_a_proj_with_mqa": (self.kv_lora_rank + self.qk_rope_head_dim, self.hidden_size),
            }
            for name, shape in expected.items():
                layer = getattr(self, name, None)
                if not isinstance(layer, nn.Linear) or tuple(layer.weight.shape) != shape:
                    raise ValueError(f"Xing MLA {name} disagrees with absorbed geometry {shape}")
        elif layout != "hf_unabsorbed" or getattr(self.config, "mla_transform", None) is not None:
            raise ValueError(f"Unsupported Xing MLA layout: {layout}")

        with torch.no_grad():
            if hasattr(self, "kv_a_proj_with_mqa"):
                w_kv_a = self.kv_a_proj_with_mqa.weight
                w_kv_a_quant = getattr(self.kv_a_proj_with_mqa, "quant_weight", None)
                has_bias = self.kv_a_proj_with_mqa.bias is not None
                proj_dtype = w_kv_a.dtype
                split_idx = int(self.kv_lora_rank)

                self.kv_a_proj_latent = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=has_bias)
                self.kv_a_proj_latent.weight = nn.Parameter(w_kv_a[:split_idx, :].contiguous().to(dtype=proj_dtype))
                _maybe_register_quant_weight(
                    self.kv_a_proj_latent,
                    None if w_kv_a_quant is None else w_kv_a_quant[:split_idx, :],
                )
                if has_bias:
                    self.kv_a_proj_latent.bias = nn.Parameter(
                        self.kv_a_proj_with_mqa.bias[:split_idx].contiguous().to(dtype=proj_dtype)
                    )

                self.kv_a_proj_rope = nn.Linear(self.hidden_size, self.qk_rope_head_dim, bias=has_bias)
                self.kv_a_proj_rope.weight = nn.Parameter(w_kv_a[split_idx:, :].contiguous().to(dtype=proj_dtype))
                _maybe_register_quant_weight(
                    self.kv_a_proj_rope,
                    None if w_kv_a_quant is None else w_kv_a_quant[split_idx:, :],
                )
                if has_bias:
                    self.kv_a_proj_rope.bias = nn.Parameter(
                        self.kv_a_proj_with_mqa.bias[split_idx:].contiguous().to(dtype=proj_dtype)
                    )

            if layout == "xh2_absorbed_v1":
                del self.kv_a_proj_with_mqa
                return self

            query = self.q_proj if self.q_lora_rank is None else self.q_b_proj
            if any(hasattr(layer, "quant_weight") for layer in (query, self.kv_b_proj, self.o_proj)):
                warnings.warn(
                    "Xing MLA hf_unabsorbed checkpoint uses legacy post-quantization absorption; "
                    "re-quantize original floating weights for xh2_absorbed_v1.",
                    UserWarning,
                    stacklevel=2,
                )
            device = self.kv_b_proj.weight.device
            weight_dtype = self.kv_b_proj.weight.dtype
            fusion_dtype = torch.float32
            w_up = self.kv_b_proj.weight.to(device=device, dtype=fusion_dtype)
            d_latent = int(w_up.shape[1])
            w_up_view = w_up.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, d_latent)
            w_uk = w_up_view[:, : self.qk_nope_head_dim, :].clone()
            w_uv = w_up_view[:, self.qk_nope_head_dim :, :].clone()

            w_q_all = query.weight.to(device=device, dtype=fusion_dtype)
            w_q_all_quant = getattr(query, "quant_weight", None)
            d_in = int(w_q_all.shape[1])
            w_q_view = w_q_all.view(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, d_in)
            w_q_quant_view = (
                w_q_all_quant.to(device=device).view(
                    self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, d_in
                )
                if w_q_all_quant is not None
                else None
            )
            w_q_nope = w_q_view[:, : self.qk_nope_head_dim, :]
            w_q_rope = w_q_view[:, self.qk_nope_head_dim :, :]
            w_q_rope_quant = (
                w_q_quant_view[:, self.qk_nope_head_dim :, :].clone() if w_q_quant_view is not None else None
            )

            self.q_rope_proj = nn.Linear(d_in, self.num_heads * self.qk_rope_head_dim, bias=False)
            self.q_rope_proj.weight = nn.Parameter(
                w_q_rope.reshape(-1, d_in).contiguous().to(device=device, dtype=weight_dtype)
            )
            _maybe_register_quant_weight(
                self.q_rope_proj,
                None if w_q_rope_quant is None else w_q_rope_quant.reshape(-1, d_in).to(device=device),
            )

            w_q_absorbed = torch.einsum("hni,hnc->hci", w_q_nope, w_uk)
            self.q_absorbed_proj = nn.Linear(d_in, self.num_heads * self.kv_lora_rank, bias=False)
            self.q_absorbed_proj.weight = nn.Parameter(
                w_q_absorbed.reshape(-1, d_in).contiguous().to(device=device, dtype=weight_dtype)
            )
            # Absorption sums products with different GPTQ group scales.
            # Multiplying integer codes cannot represent this new matrix (and
            # int16 accumulation overflows). Quantize the fused float weight.

            w_o = self.o_proj.weight.to(device=device, dtype=fusion_dtype)
            w_o_view = w_o.view(self.hidden_size, self.num_heads, self.v_head_dim)
            w_fused_vo = torch.einsum("xhd,hdc->xhc", w_o_view, w_uv)
            new_in_features = self.num_heads * self.kv_lora_rank
            has_bias = self.o_proj.bias is not None
            old_bias = self.o_proj.bias
            self.o_proj = nn.Linear(new_in_features, self.hidden_size, bias=has_bias)
            self.o_proj.weight = nn.Parameter(
                w_fused_vo.reshape(self.hidden_size, new_in_features).contiguous().to(device=device, dtype=weight_dtype)
            )
            if has_bias and old_bias is not None:
                self.o_proj.bias = nn.Parameter(old_bias.to(device=device, dtype=weight_dtype))

            if self.q_lora_rank is None:
                del self.q_proj
            else:
                del self.q_b_proj
            del self.kv_b_proj
            if hasattr(self, "kv_a_proj_with_mqa"):
                del self.kv_a_proj_with_mqa

        return self


class _Xing4_0MoE(DynamicModule):
    """Pack Xing4 routed experts into MoeBlock; keep shared_experts as dense MLP."""

    def graph_forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.router(self.router_input_cast(flat_hidden_states))
        scores = torch.sigmoid(router_logits)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        selected_experts = self.router_topk(scores_for_choice)[1]
        scores = self.routing_cast(scores)
        routed = self.moeblock(
            hidden_states,
            scores.view(batch_size, sequence_length, self.num_experts),
            selected_experts=selected_experts.view(batch_size, sequence_length, self.top_k),
        )
        if self.routed_scaling_factor != 1.0:
            routed = routed * self.routed_scaling_factor
        return routed + self.shared_experts(hidden_states)

    def _setup(self, cfg: dict | None = None):
        del cfg
        experts = self.experts
        gate = self.gate
        self.num_experts = len(experts)
        self.top_k = int(getattr(gate, "top_k", getattr(self.config, "num_experts_per_tok", 4)))
        self.norm_topk_prob = bool(getattr(gate, "norm_topk_prob", getattr(self.config, "norm_topk_prob", True)))
        self.routed_scaling_factor = float(
            getattr(gate, "routed_scaling_factor", getattr(self.config, "routed_scaling_factor", 1.0))
        )

        # Xing4 HF gate stores a Parameter ``weight`` and runs Linear in fp32.
        self.router = nn.Linear(
            gate.weight.shape[1],
            gate.weight.shape[0],
            bias=False,
            device=gate.weight.device,
            dtype=torch.float32,
        )
        self.router.weight = nn.Parameter(gate.weight.detach().float())
        self.register_buffer(
            "e_score_correction_bias",
            gate.e_score_correction_bias.detach().float(),
            persistent=False,
        )
        self.router_input_cast = xhnn.Cast(torch.float32).to(dtype=torch.float32)
        # XH2A HMONNX TopK requires sorted=1 (MiniCPM5/Laguna default).
        self.router_topk = xhnn.TopK(self.top_k, axis=-1)
        routing_dtype = experts[0].gate_proj.weight.dtype
        self.routing_cast = xhnn.Cast(routing_dtype).to(dtype=routing_dtype)

        self.moeblock = MoeBlock(
            "silu",
            self.top_k,
            normalize_routing_weights=self.norm_topk_prob,
            topk_outside=True,
        )
        for projection_name in ("gate_proj", "up_proj", "down_proj"):
            packed_weight, packed_quant_weight = _pack_expert_projection(experts, projection_name)
            setattr(self.moeblock, f"expert_{projection_name}_weight", nn.Parameter(packed_weight))
            setattr(self.moeblock, f"expert_{projection_name}_bias", None)
            if packed_quant_weight is not None:
                self.moeblock.register_buffer(
                    f"expert_{projection_name}_quant_weight",
                    packed_quant_weight,
                )

        del self.experts
        del self.gate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self


class _Xing4_0DecoderLayer(DynamicModule):
    def graph_forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs,
    ) -> tuple[Tensor]:
        del kwargs
        residual = hidden_states
        post, comb, collapsed = self.attn_hc(hidden_states)
        collapsed = self.input_layernorm(collapsed)
        attn_output, _, _ = self.self_attn(
            collapsed,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = post.unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(comb, residual)

        residual = hidden_states
        post, comb, collapsed = self.ffn_hc(hidden_states)
        collapsed = self.post_attention_layernorm(collapsed)
        mlp_output = self.mlp(collapsed)
        hidden_states = post.unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(comb, residual)
        return (hidden_states,)

    def _setup(self, cfg: dict | None = None):
        del cfg
        return self


class _Xing4_0Model(DynamicModule):
    def graph_forward(
        self,
        inputs_embeds: Tensor | None = None,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ) -> BaseModelOutputWithPast:
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()
        cos = self.cos_slice(self.rotary_emb.cos_cached, past_seq_length)
        sin = self.sin_slice(self.rotary_emb.sin_cached, past_seq_length)
        position_embeddings = (cos, sin)

        for idx, decoder_layer in enumerate(self.layers):
            if self.max_layers > 0 and idx >= self.max_layers:
                break
            layer_outputs = decoder_layer(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=past_key_cache[idx] if self.use_cache else None,
                past_v_cache=past_value_cache[idx] if self.use_cache else None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if self.only_first_block:
                break

        hidden_states = hidden_states.mean(dim=2)
        if self.num_logits_to_keep != 0:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    def _setup_cos_sin_embeding(self):
        _ = self.rotary_emb.cos_cached
        _ = self.rotary_emb.sin_cached

    def _setup(self, cfg: dict | None = None):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        self.batch_size = int(_cfg_get(cfg, "batch_size", 1))
        self.only_first_block = bool(_cfg_get(cfg, "only_first_block", False))
        max_layers = _cfg_get(cfg, "max_layers", -1)
        self.max_layers = 1 if self.only_first_block else (-1 if max_layers is None else int(max_layers))
        self.num_logits_to_keep = int(_cfg_get(cfg, "num_logits_to_keep", 0) or 0)
        assert self.num_logits_to_keep in [0, 1]
        self.use_cache = bool(_cfg_get(cfg, "use_cache", True))
        self.hc_mult = int(getattr(self.config, "hc_mult", 4))

        input_seq_len = int(_cfg_get(cfg, "input_sequence_length", 1))
        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(self.batch_size, input_seq_len)

        def _llm_gather_update_cfg(gather: xhnn.BatchGather, new_cfg: dict | None = None):
            gather.update_offset_indices(
                int(_cfg_get(new_cfg, "batch_size", 1)),
                int(_cfg_get(new_cfg, "input_sequence_length", 1)),
            )

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)
        self.cos_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])
        self.sin_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])

        def _sin_cos_slice_update_cfg(slice_module, new_cfg: dict | None = None):
            slice_module.valid_length = [int(_cfg_get(new_cfg, "input_sequence_length", 1))]

        self.sin_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.sin_slice)
        self.cos_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.cos_slice)

        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        else:
            self._setup_cos_sin_embeding()
        return self


class _Xing4_0ForCausalLM(DynamicModule):
    def graph_forward(
        self,
        inputs_embeds: Tensor | None = None,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ) -> Tensor:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        return self.lm_head(outputs.last_hidden_state)

    def _setup(self, cfg: dict | None = None):
        del cfg
        return self


_REQUIRED_WRAPPERS_BY_REMOTE_NAME = {
    "Xing4_0RMSNorm": _Xing4_0RMSNorm,
    "Xing4_0RotaryEmbedding": _Xing4_0RotaryEmbedding,
    "Xing4_0HyperConnection": _Xing4_0HyperConnection,
    "Xing4_0Attention": _Xing4_0Attention,
    "Xing4_0DecoderLayer": _Xing4_0DecoderLayer,
    "Xing4_0Model": _Xing4_0Model,
    "Xing4_0ForCausalLM": _Xing4_0ForCausalLM,
}

# MoE appears only after first_k_dense_replace; smoke/max_layers=1 may omit it.
_OPTIONAL_WRAPPERS_BY_REMOTE_NAME = {
    "Xing4_0MoE": _Xing4_0MoE,
}

_WRAPPERS_BY_REMOTE_NAME = {
    **_REQUIRED_WRAPPERS_BY_REMOTE_NAME,
    **_OPTIONAL_WRAPPERS_BY_REMOTE_NAME,
}


def register_wrap_modules(hf_model: nn.Module | None = None):
    """Register Xing4 remote-code classes into XHLLM_TRACEABLE_MODULES."""
    if hf_model is None:
        return None
    remote_classes = _find_remote_classes(hf_model)
    missing = [name for name in _REQUIRED_WRAPPERS_BY_REMOTE_NAME if name not in remote_classes]
    if missing:
        raise RuntimeError(f"Xing4.0 remote model is missing expected classes: {missing}")
    for remote_name, dynamic_cls in _REQUIRED_WRAPPERS_BY_REMOTE_NAME.items():
        _register_or_replace_traceable({remote_classes[remote_name]: remote_name})(dynamic_cls)
    for remote_name, dynamic_cls in _OPTIONAL_WRAPPERS_BY_REMOTE_NAME.items():
        if remote_name in remote_classes:
            _register_or_replace_traceable({remote_classes[remote_name]: remote_name})(dynamic_cls)

    from xhquant.nn.builder import FX_LEAF_MODULES
    from xhquant.quantization.xh2a.builder import register_none_quanted_module

    FX_LEAF_MODULES._module_dict["Xing4_0RotaryEmbedding"] = True
    register_none_quanted_module(remote_classes["Xing4_0RotaryEmbedding"])
    return None


__all__ = ["register_wrap_modules"]
