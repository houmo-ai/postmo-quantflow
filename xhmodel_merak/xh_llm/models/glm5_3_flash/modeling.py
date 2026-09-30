# Copyright 2026 the HuggingFace Team. All rights reserved.
# Copyright 2026 HOUMO AI (static graph adaptation)
# SPDX-License-Identifier: Apache-2.0
# Reference: transformers 5.16.1, models/glm5_next/modeling_glm5_next.py.
# See licenses/transformers/LICENSE and THIRD_PARTY_NOTICES.
"""Fixed tensor GLM-5.3 text graph: KDA, learned C4 DSA, mHC and MoE."""

import torch
import torch.nn.functional as F
from torch import nn

from xhquant import nn as xhnn

from ..deepseek_v4.moe import ExactClampedSwiGLURoutedExperts
from ..deepseek_v4.norm import build_xh_rms_norm, rms_norm_from_hf
from ..deepseek_v4.static_cache import FixedCapacityCacheWriter, LatentCacheGather
from ..ling_3_flash._kda_rule import chunk_kda
from .contract import LINEAR


class StaticRMSNorm(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight = source.weight
        self.norm = rms_norm_from_hf(source)

    def forward(self, x):
        return self.norm(x)


class StaticHyperConnection(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.fn, self.base, self.scale = source.fn, source.base, source.scale
        self.proj = nn.Linear(source.fn.shape[1], source.fn.shape[0], bias=False)
        self.proj.weight = self.fn
        self.hc = source.hc_mult
        self.eps, self.norm_eps = source.hc_eps, source.input_norm.eps
        self.iters = source.hc_sinkhorn_iters
        self.sinkhorn = xhnn.MHCSinkhorn(self.iters, self.eps)
        self.norm = build_xh_rms_norm(source.fn.shape[1], self.norm_eps).to(
            device=source.fn.device, dtype=source.fn.dtype
        )

    def forward(self, streams):
        flat = streams.flatten(2).float()
        normalized = self.norm(flat)
        mix = self.proj(normalized.to(self.proj.weight.dtype)).float()
        hc = self.hc
        pre = torch.sigmoid(mix[..., :hc] * self.scale[0] + self.base[:hc]) + self.eps
        post = 2 * torch.sigmoid(mix[..., hc : 2 * hc] * self.scale[1] + self.base[hc : 2 * hc])
        comb_logits = (mix[..., 2 * hc :] * self.scale[2] + self.base[2 * hc :]).reshape(1, -1, hc, hc)
        comb = self.sinkhorn(comb_logits)
        collapsed = (pre.unsqueeze(-1) * streams).sum(2).to(streams.dtype)
        return post.to(streams.dtype), comb.to(streams.dtype), collapsed


class StaticKDA(nn.Module):
    """Reuse Ling's KDA factorization; GLM has one packed q/k/v convolution."""

    def __init__(self, source, input_length, *, chunk_size=8):
        super().__init__()
        self.length, self.heads, self.dim = input_length, source.num_heads, source.head_dim
        self.width, self.kernel = source.qkv_dim, source.conv_kernel_size
        for name in ("q_proj", "k_proj", "v_proj", "conv1d", "b_proj", "g_a_proj", "g_b_proj", "o_proj"):
            setattr(self, name, getattr(source, name))
        self.forget_gate = source.forget_gate
        self.norm_weight, self.norm_eps = source.o_norm.weight, source.o_norm.variance_epsilon
        self.output_norm = build_xh_rms_norm(source.head_dim, self.norm_eps, weight=source.o_norm.weight)
        self.tail_slice = xhnn.DynamicSlice([self.kernel], [2], [1])
        self.chunk_size = chunk_size
        if input_length != 1 and input_length % chunk_size:
            raise ValueError("KDA prefill length must be divisible by chunk_size")
        self.register_buffer("causal", torch.tril(torch.ones(chunk_size, chunk_size)), persistent=False)
        self.register_buffer("strict", torch.tril(torch.ones(chunk_size, chunk_size), -1), persistent=False)
        from ..qwen3_5._gdr_ops import GDRBlockTriInverse

        self.tri_inverse = GDRBlockTriInverse(chunk_size, 8)
        self.register_buffer(
            "eye",
            torch.eye(8)
            .unsqueeze(0)
            .expand(max(1, input_length // chunk_size) * self.heads * (chunk_size // 8), 8, 8)
            .contiguous(),
            persistent=False,
        )

    def forward(self, hidden, current_length, validity, conv_state, recurrent_state):
        mask = validity.unsqueeze(-1)
        hidden = hidden * mask
        mixed = torch.cat((self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)), -1).transpose(1, 2)
        source = torch.cat((conv_state, mixed), -1)
        next_conv = self.tail_slice(source, current_length)
        convolved = F.silu(self.conv1d(source)[:, :, self.kernel : self.kernel + self.length]).transpose(1, 2)
        q, k, v = (
            convolved[..., i * self.width : (i + 1) * self.width].reshape(1, self.length, self.heads, self.dim).float()
            for i in range(3)
        )
        # GLM's reference uses rsqrt(sum(x*x)+eps), not F.normalize's max(norm,eps).
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        state_mask = mask.unsqueeze(-1)
        q, k, v = (x * state_mask for x in (q, k, v))
        gate = self.forget_gate
        raw = gate.f_b_proj(gate.f_a_proj(hidden)).float() + gate.dt_bias.float()
        raw = raw.reshape(1, self.length, self.heads, self.dim)
        rate = gate.A_log.float().exp().reshape(1, 1, self.heads, 1)
        log_decay = (
            gate.safe_gate_lower_bound * torch.sigmoid(rate * raw)
            if gate.safe_gate_lower_bound is not None
            else -rate * F.softplus(raw)
        ) * state_mask
        beta = torch.sigmoid(self.b_proj(hidden)).float() * mask
        # Accumulate in FP32, then commit using the explicit cache input dtype.
        if self.length == 1:
            state = recurrent_state.float() * log_decay[:, 0].exp().unsqueeze(-1)
            memory = (state * k[:, 0].unsqueeze(-1)).sum(-2)
            delta = (v[:, 0] - memory) * beta[:, 0].unsqueeze(-1)
            state = state + k[:, 0].unsqueeze(-1) * delta.unsqueeze(-2)
            core = ((state * q[:, 0].unsqueeze(-1)).sum(-2) * self.dim**-0.5).unsqueeze(1)
        else:
            core, state = self._chunk(q, k, v, log_decay, beta, recurrent_state)
        core = core.to(hidden.dtype).float()
        out_gate = self.g_b_proj(self.g_a_proj(hidden)).reshape(1, self.length, self.heads, self.dim)
        core = self.output_norm(core) * torch.sigmoid(out_gate.float())
        output = self.o_proj(core.to(hidden.dtype).reshape(1, self.length, self.width)) * mask
        return output, next_conv, state.to(recurrent_state.dtype)

    def _chunk(self, q, k, v, g, beta, state):
        # KDA's factorization accepts already-normalized Q/K via an explicit
        # shared helper option, keeping normalization owned by this model.
        return chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            state,
            chunk_size=self.chunk_size,
            scale=self.dim**-0.5,
            block_tri_inverse_op=self.tri_inverse,
            eye_block_batched=self.eye,
            chunk_scan_op=None,
            batch_size=1,
            sequence_length=self.length,
            num_heads=self.heads,
            key_dim=self.dim,
            value_dim=self.dim,
            causal_pair_mask=self.causal,
            strict_pair_mask=self.strict,
            normalize_qk=False,
        )


class StaticKPool(nn.Module):
    """Four rows of normalized raw index key and raw gate, without overlap."""

    def __init__(self, indexer, input_length):
        super().__init__()
        self.length, self.dim, self.ratio = input_length, indexer.head_dim, indexer.index_kpool
        self.count = (input_length + self.ratio - 1) // self.ratio
        self.wk, self.k_norm = indexer.wk, indexer.k_norm
        self.gate, self.ape = indexer.index_kpool_compress_gate, indexer.index_kpool_compress_ape
        self.gate_proj = nn.Linear(self.gate.shape[1], self.gate.shape[0], bias=False)
        self.gate_proj.weight = self.gate
        self.candidate = xhnn.DynamicSlice([self.count * self.ratio], [1], [1])
        self.tail = xhnn.DynamicSlice([self.ratio], [1], [1])
        self.register_buffer("pad", torch.zeros(1, self.ratio - 1, self.dim), persistent=False)

    def forward(self, x, offset, current_length, key_state, score_state):
        keys = torch.cat((key_state, self.k_norm(self.wk(x))), dim=1)
        scores = torch.cat((score_state, self.gate_proj(x)), dim=1)
        grouped_keys = self.candidate(torch.cat((keys, self.pad), 1), offset).reshape(
            1, self.count, self.ratio, self.dim
        )
        grouped_scores = self.candidate(torch.cat((scores, self.pad), 1), offset).reshape(
            1, self.count, self.ratio, self.dim
        )
        probabilities = torch.softmax(grouped_scores.float() + self.ape.float()[None, None], dim=2).to(
            grouped_keys.dtype
        )
        return (grouped_keys * probabilities).sum(2), self.tail(keys, current_length), self.tail(scores, current_length)


class StaticDSA(nn.Module):
    """TopK pools expand into original token latents; exact MLA factorization."""

    def __init__(self, source, spec, input_length):
        super().__init__()
        self.length, self.heads = input_length, source.num_heads
        self.q_dim, self.v_dim, self.latent_dim = source.qk_nope_head_dim, source.v_head_dim, source.kv_lora_rank
        self.scaling = source.scaling
        for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
            setattr(self, name, getattr(source, name))
        self.q_norm, self.kv_norm = StaticRMSNorm(source.q_a_layernorm), StaticRMSNorm(source.kv_a_layernorm)
        weight = source.kv_b_proj.weight.reshape(self.heads, self.q_dim + self.v_dim, self.latent_dim)
        self.k_weight = nn.Parameter(weight[:, : self.q_dim].detach())
        self.v_weight = nn.Parameter(weight[:, self.q_dim :].detach())
        # Name the two constant RHS projections separately from activation
        # matmuls so the exporter can apply weight precision to these factors.
        self.k_projection = xhnn.MatMul()
        self.v_projection = xhnn.MatMul()
        self.indexer = source.indexer
        self.pool = StaticKPool(source.indexer, input_length)
        self.writer = FixedCapacityCacheWriter()
        self.gather = LatentCacheGather(feature_dim=spec.latent_dim)
        self.gather_validity = xhnn.GatherElements(axis=2)
        self.topk = xhnn.TopK(spec.selected_pools, axis=-1)
        self.register_buffer("pool_offsets", torch.arange(spec.pool_size, dtype=torch.int32), persistent=False)
        self.ratio = spec.pool_size

    def forward(
        self,
        x,
        past_length,
        current_length,
        offset,
        new_count,
        write_start,
        query_valid,
        pool_valid,
        tail_indices,
        tail_valid,
        latent_cache,
        pool_cache,
        key_state,
        score_state,
    ):
        resid = self.q_norm(self.q_a_proj(x))
        query = self.q_b_proj(resid).reshape(1, self.length, self.heads, self.q_dim)
        latent = self.kv_norm(self.kv_a_proj_with_mqa(x))
        full_latent = self.writer(latent_cache, latent, past_length, current_length)
        candidates, next_key, next_score = self.pool(x, offset, current_length, key_state, score_state)
        full_pool = self.writer(pool_cache, candidates, write_start, new_count)
        iq = self.indexer.wq_b(resid).reshape(1, self.length, self.indexer.n_heads, self.indexer.head_dim)
        score = torch.relu(
            torch.matmul(iq.float(), full_pool.float().transpose(1, 2).unsqueeze(1)) * self.indexer.softmax_scale
        )
        head_weight = self.indexer.weights_proj(x).float() * self.indexer.n_heads**-0.5
        score = torch.matmul(head_weight.unsqueeze(-2), score).squeeze(-2)
        # Use the FP16 finite minimum for invalid pools without shifting
        # valid scores; scores below -32768 must still outrank padding.
        score = score * pool_valid + (pool_valid - 1.0) * 65504.0
        selected = self.topk(score)[1]
        selected_valid = self.gather_validity(pool_valid, selected)
        indices = (selected.unsqueeze(-1) * self.ratio + self.pool_offsets).flatten(-2)
        validity = selected_valid.unsqueeze(-1).expand(1, self.length, -1, self.ratio).flatten(-2)
        indices = torch.cat((indices, tail_indices), -1)
        validity = torch.cat((validity, tail_valid), -1)
        selected_latent = self.gather(full_latent, indices)
        # Absorb only the per-head K factor. No compressed index key participates in attention.
        latent_query = self.k_projection(query.transpose(1, 2), self.k_weight).transpose(1, 2)
        scores = torch.matmul(latent_query.float(), selected_latent.float().transpose(-1, -2)) * self.scaling
        scores = scores * validity.unsqueeze(2) + (validity.unsqueeze(2) - 1.0) * 32768.0
        probabilities = torch.softmax(scores, dim=-1) * validity.unsqueeze(2)
        context = torch.matmul(probabilities.to(selected_latent.dtype), selected_latent)
        values = self.v_projection(context.transpose(1, 2), self.v_weight.transpose(-1, -2)).transpose(1, 2)
        output = self.o_proj(values.reshape(1, self.length, -1)) * query_valid.unsqueeze(-1)
        return output, full_latent.unsqueeze(1), full_pool.unsqueeze(1), next_key, next_score


class StaticMoE(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.gate, self.shared_experts = source.gate, source.shared_experts
        if source.gate.num_group != 1 or not source.gate.norm_topk_prob:
            raise ValueError("released GLM-5.3 uses one expert group and normalized top-k weights")
        self.router = nn.Linear(source.gate.hidden_dim, source.gate.num_experts, bias=False)
        self.router.weight = source.gate.weight
        self.topk = xhnn.TopK(source.gate.top_k, axis=-1)
        self.experts = ExactClampedSwiGLURoutedExperts.from_hf(
            source.experts,
            top_k=source.gate.top_k,
            limit=source.config.swiglu_limit,
            # The eager Triton implementation reads full 128-channel tiles.
            fast_mode=(source.experts.hidden_dim % 128 == 0 and source.experts.intermediate_dim % 128 == 0),
        )

    def forward(self, x):
        scores = torch.sigmoid(self.router(x.to(self.router.weight.dtype)).float())
        choice = scores + self.gate.e_score_correction_bias.float()
        selected = self.topk(choice)[1]
        return self.experts(x, scores.to(x.dtype), selected) * self.gate.routed_scaling_factor + self.shared_experts(x)


class StaticDecoder(nn.Module):
    def __init__(self, source, spec, input_length, chunk_size):
        super().__init__()
        self.kind = source.block_type
        self.attn_hc, self.ffn_hc = StaticHyperConnection(source.attn_hc), StaticHyperConnection(source.ffn_hc)
        self.input_norm, self.ffn_norm = (
            StaticRMSNorm(source.input_layernorm),
            StaticRMSNorm(source.post_attention_layernorm),
        )
        self.attention = (
            StaticKDA(source.self_attn, input_length, chunk_size=chunk_size)
            if self.kind == LINEAR
            else StaticDSA(source.self_attn, spec, input_length)
        )
        self.mlp = (
            StaticFFNPlaceholder()
            if getattr(source.mlp, "_glm53_placeholder", False)
            else StaticMoE(source.mlp)
            if hasattr(source.mlp, "experts")
            else source.mlp
        )

    def forward(self, streams, controls, caches):
        past, length, _, offset, count, start, valid, pool_valid, tail, tail_valid = controls
        post, comb, x = self.attn_hc(streams)
        x = self.input_norm(x)
        if self.kind == LINEAR:
            result = self.attention(x, length, valid, *caches)
        else:
            result = self.attention(x, past, length, offset, count, start, valid, pool_valid, tail, tail_valid, *caches)
        streams = post.unsqueeze(-1) * result[0].unsqueeze(2) + comb.transpose(-1, -2) @ streams
        post, comb, x = self.ffn_hc(streams)
        streams = post.unsqueeze(-1) * self.mlp(self.ffn_norm(x)).unsqueeze(2) + comb.transpose(-1, -2) @ streams
        return streams, result[1:]


class StaticFFNPlaceholder(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class StaticGlm53ForCausalLM(nn.Module):
    def __init__(self, text_model, lm_head, spec, input_length, *, chunk_size=8):
        super().__init__()
        self.spec, self.hc_mult = spec, text_model.config.hc_mult
        self.layers = nn.ModuleList(
            StaticDecoder(layer, spec, input_length, chunk_size) for layer in text_model.layers[: len(spec.layer_types)]
        )
        # Stable execution boundaries let decode reuse prefill's block placement.
        self.layer_tags = nn.ModuleList(
            [xhnn.XHTag(f"layer_{index}", "LLM", f"layer_{index}") for index in range(len(self.layers))]
        )
        self.norm, self.lm_head = StaticRMSNorm(text_model.norm), lm_head
        self.last_token = xhnn.DynamicSlice([1], [1], [1])

    def forward(
        self,
        inputs_embeds,
        past_seq_length,
        current_input_length,
        last_token_index,
        pool_offset,
        pool_new_count,
        pool_write_start,
        query_validity,
        pool_validity,
        tail_indices,
        tail_validity,
        caches,
    ):
        # FX sees the nested cache argument as a Proxy: unroll its known arity
        # with static indexing instead of iterating over that Proxy.
        caches = tuple(caches[i] for i in range(sum(len(self.spec.cache_suffixes(k)) for k in self.spec.layer_types)))
        controls = (
            past_seq_length,
            current_input_length,
            last_token_index,
            pool_offset,
            pool_new_count,
            pool_write_start,
            query_validity,
            pool_validity,
            tail_indices,
            tail_validity,
        )
        streams = inputs_embeds.unsqueeze(2).expand(1, -1, self.hc_mult, -1)
        cursor, next_caches = 0, []
        for index, layer in enumerate(self.layers):
            count = 2 if layer.kind == LINEAR else 4
            streams, updated = layer(streams, controls, caches[cursor : cursor + count])
            streams = self.layer_tags[index](streams)
            next_caches.extend(updated)
            cursor += count
        hidden = self.norm(streams.mean(2))
        logits = self.lm_head(self.last_token(hidden, last_token_index))
        return (logits, *next_caches)
