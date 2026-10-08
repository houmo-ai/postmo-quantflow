# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Small real GLM blocks, including sparse selection and streaming state."""

import inspect

import pytest
import torch
from torch import nn
from transformers import DynamicCache
from transformers.models.glm5_next import modeling_glm5_next as hf_glm53
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

from xhmodel_merak.xh_llm.models.glm5_3_flash import hmonnx_inference as glm_hmonnx
from xhmodel_merak.xh_llm.models.glm5_3_flash.contract import DSA, Glm53CacheSpec, build_transition
from xhmodel_merak.xh_llm.models.glm5_3_flash.modeling import (
    StaticDSA,
    StaticGlm53ForCausalLM,
    StaticHyperConnection,
    StaticKDA,
    StaticKPool,
    StaticMoE,
)
from xhmodel_merak.xh_llm.models.glm5_3_flash.runtime import Glm53CacheMixin, Glm53Runner
from xhmodel_merak.xh_llm.models.ling_3_flash._kda_rule import chunk_kda, recurrent_kda_reference
from xhquant.core import CacheTensor
from xhquant.ops.xh.kv_cache import llm_cache_default_static, llm_cache_update_inplace
from xhquant.xhonnxruntime.parsers.llm_cache import LLMCache


def tiny_config():
    return Glm5NextTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        qk_head_dim=8,
        linear_num_heads=2,
        linear_head_dim=8,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        mlp_layer_types=["dense"] * 3 + ["sparse"],
        indexer_types=["full"] * 4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        n_group=1,
        topk_group=1,
        index_n_heads=8,
        index_head_dim=8,
        index_kpool=4,
        index_topk=8,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        mhc=True,
        mla_use_nope=True,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=1,
        swiglu_limit=1.0,
    )


def test_inplace_dsa_cache_update_preserves_zero_padded_output():
    reference = CacheTensor(torch.zeros(1, 1, 12, 3, dtype=torch.float16))
    reused = CacheTensor(reference.data.clone())
    for start, count in ((0, 3), (3, 1), (4, 0), (4, 4), (8, 4)):
        values = torch.arange(12, dtype=torch.float16).reshape(1, 1, 4, 3) + start
        position = torch.tensor([start])
        length = torch.tensor([count])
        expected = llm_cache_default_static(values, position, length, reference, 2, -1)
        actual = llm_cache_update_inplace(values, position, length, reused, 2, -1)
        assert actual.data_ptr() == reused.data_ptr()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.count_nonzero(actual[:, :, start + count :]) == 0
    with pytest.raises(ValueError, match="outside the fixed-capacity storage"):
        llm_cache_update_inplace(values, torch.tensor([11]), torch.tensor([2]), reused, 2, -1)


def test_inplace_step_staging_backs_up_only_written_dsa_rows():
    spec = Glm53CacheSpec((DSA, DSA), 12, 2, 8, 4, 3, 2, 4, 8)
    cache = Glm53CacheMixin(spec, object(), inplace_step_staging=True)
    cache.caches = spec.initial_caches(dtype=torch.float16)
    for layer in (0, 4):
        cache.caches[layer].data[:, :, :3].fill_(layer + 1)
    before = tuple(value.clone() for value in cache.caches)

    staged = cache.stage_step(3, 2)
    assert all(old is new for old, new in zip(cache.caches, staged, strict=True))
    assert len(cache._staged_rows) == 4
    assert sum(saved.numel() for _, _, saved in cache._staged_rows) < sum(value.numel() for value in cache.caches)
    for tensor, start, saved in cache._staged_rows:
        tensor.narrow(2, start, saved.shape[2]).fill_(42)
    cache.rollback_step()
    assert all(torch.equal(old, new) for old, new in zip(before, cache.caches, strict=True))

    staged = cache.stage_step(5, 1)
    assert len(cache._staged_rows) == 2  # No complete pool is written on this step.
    logits = torch.ones(1, 1)
    outputs = list(staged)
    for i in (0, 1, 4, 5):
        outputs[i] = staged[i].data
    assert cache.commit((logits, *outputs)) is logits
    assert all(isinstance(cache.caches[i], CacheTensor) for i in (0, 1, 4, 5))
    assert all(cache.caches[i].data_ptr() == staged[i].data_ptr() for i in (0, 1, 4, 5))
    assert cache._staged_rows is None


@pytest.mark.parametrize("past,pool_rows", [(3, 1), (4, 0), (11, 1)])
def test_inplace_dsa_rollback_at_pool_and_capacity_boundaries(past, pool_rows):
    spec = Glm53CacheSpec((DSA,), 12, 2, 8, 4, 3, 2, 4, 8)
    cache = Glm53CacheMixin(spec, object(), inplace_step_staging=True)
    cache.caches = spec.initial_caches(dtype=torch.float16)
    before = tuple(value.clone() for value in cache.caches)
    staged = cache.stage_step(past, 1)
    assert [(start, saved.shape[2]) for _, start, saved in cache._staged_rows] == (
        [(past, 1)] + ([(past // 4, 1)] if pool_rows else [])
    )
    for tensor, start, saved in cache._staged_rows:
        tensor.narrow(2, start, saved.shape[2]).fill_(42)
    bad_outputs = [torch.ones(1, 1), *staged]
    bad_outputs[-1] = torch.zeros(1)
    with pytest.raises(ValueError, match="incompatible cache shape"):
        cache.commit(bad_outputs)
    cache.rollback_step()
    assert cache._staged_rows is None
    assert all(torch.equal(old, new) for old, new in zip(before, cache.caches, strict=True))
    # A failed step can be retried at the same committed length.
    cache.stage_step(past, 1)
    cache.rollback_step()


def test_inplace_dsa_cache_session_preflight_and_rollback(monkeypatch):
    class Session:
        def __init__(self, count, result=None, raise_after_enable=False):
            self.graph_module = nn.ModuleList(LLMCache(axis=2) for _ in range(count))
            self.result = result
            self.raise_after_enable = raise_after_enable
            self.calls = 0
            self.alias_calls = 0
            self.static_rhs_calls = []

        def initialize(self):
            pass

        def enable_inplace_cache_updates(self):
            self.calls += 1
            for module in self.graph_module:
                module.enable_inplace_output_reuse()
            if self.raise_after_enable:
                raise RuntimeError("enable failed")
            return len(self.graph_module) if self.result is None else self.result

        def enable_reused_cache_output_aliases(self):
            self.alias_calls += 1
            return 1

        def enable_constant_matmul_rhs_cache(self, *, min_numel):
            self.static_rhs_calls.append(min_numel)
            return len(self.graph_module)

    monkeypatch.setattr(glm_hmonnx, "HMONNXInferenceV2", Session)
    spec = Glm53CacheSpec((DSA,), 12, 2, 8, 4, 3, 2, 4, 8)
    assert all(torch.count_nonzero(cache.data) == 0 for cache in spec.initial_caches()[:2])
    prefill, decode = Session(2), Session(2)
    assert glm_hmonnx._enable_inplace_dsa_caches(prefill, decode, spec)
    assert (prefill.calls, decode.calls) == (1, 1)
    assert (prefill.alias_calls, decode.alias_calls) == (1, 1)
    assert prefill.static_rhs_calls == decode.static_rhs_calls == [4_000_000]
    assert all(module.reuse_output_storage for session in (prefill, decode) for module in session.graph_module)

    two_dsa = Glm53CacheSpec((DSA, DSA), 12, 2, 8, 4, 3, 2, 4, 8)
    prefill, decode = Session(4), Session(4)
    glm_hmonnx._enable_inplace_dsa_caches(prefill, decode, two_dsa)
    assert (prefill.calls, decode.calls) == (1, 1)

    prefill, decode = Session(1), Session(2)
    with pytest.raises(RuntimeError, match="prefill cache writer count mismatch"):
        glm_hmonnx._enable_inplace_dsa_caches(prefill, decode, spec)
    assert (prefill.calls, decode.calls) == (0, 0)

    prefill, decode = Session(2), Session(1)
    with pytest.raises(RuntimeError, match="decode cache writer count mismatch"):
        glm_hmonnx._enable_inplace_dsa_caches(prefill, decode, spec)
    assert (prefill.calls, decode.calls) == (0, 0)
    assert not any(module.reuse_output_storage for module in prefill.graph_module)

    prefill, decode = Session(2), Session(2, result=1)
    with pytest.raises(RuntimeError, match="decode cache writer count mismatch"):
        glm_hmonnx._enable_inplace_dsa_caches(prefill, decode, spec)
    assert (prefill.calls, decode.calls) == (1, 1)
    assert not any(module.reuse_output_storage for session in (prefill, decode) for module in session.graph_module)

    for failing_side in ("prefill", "decode"):
        prefill = Session(2, raise_after_enable=failing_side == "prefill")
        decode = Session(2, raise_after_enable=failing_side == "decode")
        with pytest.raises(RuntimeError, match="enable failed"):
            glm_hmonnx._enable_inplace_dsa_caches(prefill, decode, spec)
        assert (prefill.calls, decode.calls) == ((1, 0) if failing_side == "prefill" else (1, 1))
        assert not any(module.reuse_output_storage for session in (prefill, decode) for module in session.graph_module)

    with pytest.raises(RuntimeError, match="same HMONNX backend"):
        glm_hmonnx._enable_inplace_dsa_caches(Session(2), object(), spec)
    with pytest.raises(RuntimeError, match="same HMONNX backend"):
        glm_hmonnx._enable_inplace_dsa_caches(object(), Session(2), spec)
    glm_hmonnx._enable_inplace_dsa_caches(object(), object(), spec)
    monkeypatch.setenv("XHMODEL_GLM53_INPLACE_CACHE_UPDATES", "0")
    prefill, decode = Session(2), Session(2)
    glm_hmonnx._enable_inplace_dsa_caches(prefill, decode, spec)
    assert (prefill.calls, decode.calls) == (0, 0)
    monkeypatch.setenv("XHMODEL_GLM53_INPLACE_CACHE_UPDATES", "invalid")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        glm_hmonnx._enable_inplace_dsa_caches(prefill, decode, spec)


@pytest.fixture
def hf_model(monkeypatch):
    # These numerical references run on CPU. HF's optional-kernel decorators
    # select installed FLA/causal-conv CUDA implementations even for CPU inputs.
    # Keep HF's own torch math, including both prefill and decode cache updates,
    # independent of which optional GPU packages the CI image has installed.
    for name in (
        "chunk_kimi_delta_attention",
        "recurrent_kimi_delta_attention",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        monkeypatch.setattr(hf_glm53, name, inspect.unwrap(getattr(hf_glm53, name)))
    torch.manual_seed(315)
    model = Glm5NextTextModel(tiny_config()).eval()
    # Make routing, clamping and mHC matrix orientation observable.
    with torch.no_grad():
        for layer in model.layers:
            layer.attn_hc.fn.normal_(0, 0.1)
            layer.ffn_hc.fn.normal_(0, 0.1)
        model.layers[3].mlp.gate.weight.normal_(0, 0.3)
        model.layers[3].mlp.gate.e_score_correction_bias.copy_(torch.tensor([-0.3, 0.2, -0.1, 0.4]))
    return model


@pytest.mark.parametrize("past,n,p", [(0, 1, 8), (0, 7, 8), (6, 7, 8), (7, 8, 8), (15, 1, 1), (27, 5, 8), (61, 3, 8)])
def test_transition_pool_matches_reference(hf_model, past, n, p):
    spec = Glm53CacheSpec.from_config(hf_model.config, 64)
    indexer = hf_model.layers[3].self_attn.indexer
    x = torch.randn(1, past + n, 32)
    keys = indexer.k_norm(indexer.wk(x))
    scores = torch.nn.functional.linear(x, indexer.index_kpool_compress_gate)
    state_k = torch.nn.functional.pad(keys[:, :past], (0, 0, max(0, 4 - past), 0))[:, -4:]
    state_s = torch.nn.functional.pad(scores[:, :past], (0, 0, max(0, 4 - past), 0), value=-32768)[:, -4:]
    current = torch.nn.functional.pad(x[:, past:], (0, 0, 0, p - n), value=19)
    t = build_transition(spec, past, n, p)
    pooled, next_k, next_s = StaticKPool(indexer, p)(
        current, t["pool_offset"], t["current_input_length"], state_k, state_s
    )
    packed = torch.cat((keys, scores, torch.ones(1, past + n, 1)), -1)
    ref, _, _ = indexer.get_pooled_states(packed)
    count = (past % 4 + n) // 4
    torch.testing.assert_close(pooled[:, :count], ref[:, past // 4 : past // 4 + count])
    torch.testing.assert_close(next_k, torch.nn.functional.pad(keys, (0, 0, max(0, 4 - past - n), 0))[:, -4:])
    torch.testing.assert_close(
        next_s, torch.nn.functional.pad(scores, (0, 0, max(0, 4 - past - n), 0), value=-32768)[:, -4:]
    )


def test_mhc_and_clamped_moe(hf_model):
    streams = torch.randn(1, 8, 4, 32) * 2
    original = hf_model.layers[3]
    with torch.no_grad():
        expected = original.attn_hc(streams)
        actual = StaticHyperConnection(original.attn_hc)(streams)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        inputs = streams.mean(2) * 20
        torch.testing.assert_close(StaticMoE(original.mlp)(inputs), original.mlp(inputs), rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize("sequence", [[7, 1, 1, 8, 5], [1, 1, 1, 1, 8, 8, 8]])
def test_kda_continuation_and_padding(hf_model, sequence):
    original = hf_model.layers[0].self_attn
    spec = Glm53CacheSpec.from_config(hf_model.config, 64)
    cache = DynamicCache(config=hf_model.config)
    state = spec.initial_caches()[:2]
    past = 0
    with torch.no_grad():
        for n in sequence:
            p = 1 if n == 1 else 8
            x = torch.randn(1, n, 32)
            expected = original(x, cache_params=cache, attention_mask=torch.ones(1, n, dtype=torch.bool))
            padded = torch.nn.functional.pad(x, (0, 0, 0, p - n), value=17)
            t = build_transition(spec, past, n, p)
            actual, *state = StaticKDA(original, p)(padded, t["current_input_length"], t["query_validity"], *state)
            torch.testing.assert_close(actual[:, :n], expected, rtol=5e-4, atol=2e-5)
            torch.testing.assert_close(state[1], cache.layers[0].recurrent_states[0], rtol=5e-4, atol=2e-5)
            torch.testing.assert_close(state[0], cache.layers[0].conv_states[0])
            assert torch.count_nonzero(actual[:, n:]) == 0
            past += n


def test_dsa_streaming_selects_original_tokens(hf_model):
    original = hf_model.layers[3].self_attn
    spec = Glm53CacheSpec.from_config(hf_model.config, 64)
    cache = DynamicCache(config=hf_model.config)
    state = spec.initial_caches()[-4:]
    past = 0
    with torch.no_grad():
        for n in (7, 1, 1, 8, 5, 8, 1, 8, 8, 8, 6, 3):
            p = 1 if n == 1 else 8
            x = torch.randn(1, n, 32)
            expected = original(x, attention_mask=torch.ones(1, n, dtype=torch.bool), past_key_values=cache)[0]
            t = build_transition(spec, past, n, p)
            padded = torch.nn.functional.pad(x, (0, 0, 0, p - n), value=31)
            actual, *state = StaticDSA(original, spec, p)(
                padded,
                t["past_seq_length"],
                t["current_input_length"],
                t["pool_offset"],
                t["pool_new_count"],
                t["pool_write_start"],
                t["query_validity"],
                t["pool_validity"],
                t["tail_indices"],
                t["tail_validity"],
                *state,
            )
            torch.testing.assert_close(actual[:, :n], expected, rtol=1e-4, atol=2e-5)
            assert torch.count_nonzero(actual[:, n:]) == 0
            past += n


def test_complete_hybrid_blocks(hf_model):
    spec = Glm53CacheSpec.from_config(hf_model.config, 64)
    head = nn.Linear(32, 64, bias=False)
    prefill = StaticGlm53ForCausalLM(hf_model, head, spec, 8)
    decode = StaticGlm53ForCausalLM(hf_model, head, spec, 1)
    cache = DynamicCache(config=hf_model.config)
    state = spec.initial_caches()
    past = 0
    with torch.no_grad():
        for n in (7, 1, 1, 8, 5):
            ids = torch.randint(1, 64, (1, n))
            expected = head(hf_model(ids, past_key_values=cache, use_cache=True).last_hidden_state[:, -1:])
            p = 1 if n == 1 else 8
            embed = torch.nn.functional.pad(hf_model.embed_tokens(ids), (0, 0, 0, p - n), value=9)
            t = build_transition(spec, past, n, p)
            actual, *state = (decode if p == 1 else prefill)(embed, *t.values(), state)
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=4e-5)
            past += n


def record_pool_selection(chosen):
    def hook(module, args, result):
        query, pool, payload, validity, weights = args
        head_scores = torch.relu(query.float() @ pool.float().transpose(1, 2).unsqueeze(1) * module.score_scale)
        scores = (weights.float() @ head_scores).squeeze(-2)
        scores = scores * validity + (validity - 1.0) * 65504.0
        indices = torch.topk(scores, module.candidate_topk, dim=-1).indices
        token_ids = (indices[..., None] * 4 + torch.arange(4, device=indices.device)).flatten(-2)
        expected_mask = validity.gather(-1, indices)[..., None].expand(*indices.shape, 4).flatten(-2)
        torch.testing.assert_close(result[0], payload.reshape(-1, payload.shape[-1])[token_ids], rtol=0, atol=0)
        torch.testing.assert_close(result[1], expected_mask, rtol=0, atol=0)
        chosen.update(scores=scores.clone(), indices=indices.clone())

    return hook


@pytest.mark.parametrize("minimum_score", [-10.0, -40000.0, -65000.0])
def test_dsa_negative_scores_keep_raw_topk_values_and_mask(hf_model, minimum_score):
    original = hf_model.layers[3].self_attn
    original.indexer.weights_proj.weight.data.fill_(-0.125)
    spec = Glm53CacheSpec.from_config(hf_model.config, 64)
    core = StaticDSA(original, spec, 16)
    chosen = {}
    hook = core.select_kv.register_forward_hook(record_pool_selection(chosen))
    x = torch.randn(1, 16, 32).abs() + 0.1
    t = build_transition(spec, 0, 16, 16)
    with torch.no_grad():
        outputs = core(
            x, t["past_seq_length"], t["current_input_length"], t["pool_offset"],
            t["pool_new_count"], t["pool_write_start"], t["query_validity"],
            t["pool_validity"], t["tail_indices"], t["tail_validity"],
            *spec.initial_caches()[-4:],
        )
        valid = t["pool_validity"].bool()
        # Scale a real indexer's learned head weights to exercise scores
        # below the former -32768 sentinel, including near the FP16 limit.
        scale = minimum_score / chosen["scores"][valid].min().item()
        original.indexer.weights_proj.weight.mul_(scale)
        outputs = core(
            x, t["past_seq_length"], t["current_input_length"], t["pool_offset"],
            t["pool_new_count"], t["pool_write_start"], t["query_validity"],
            t["pool_validity"], t["tail_indices"], t["tail_validity"],
            *spec.initial_caches()[-4:],
        )
        pool = outputs[2].squeeze(1)
        query = original.indexer.wq_b(core.q_norm(core.q_a_proj(x))).reshape(1, 16, 8, 8)
        head_scores = torch.relu(
            query.float() @ pool.float().transpose(1, 2).unsqueeze(1) * original.indexer.softmax_scale
        )
        head_weights = original.indexer.weights_proj(x).float() * 8**-0.5
        raw_scores = (head_weights.unsqueeze(-2) @ head_scores).squeeze(-2)
        assert bool((raw_scores[valid] < 0).any())
        assert bool((raw_scores[valid] > torch.finfo(torch.float16).min).all())
        torch.testing.assert_close(raw_scores[valid].min(), torch.tensor(minimum_score))
        if minimum_score < -32768:
            assert bool((raw_scores[valid] < -32768).any())
        torch.testing.assert_close(chosen["scores"][valid], raw_scores[valid])
        assert bool((chosen["scores"][~valid] == torch.finfo(torch.float16).min).all())
        assert torch.isfinite(chosen["scores"].to(torch.float16)).all()
        selected = torch.topk(chosen["scores"], spec.selected_pools, dim=-1).indices
        selected_valid = valid.gather(-1, selected)
        assert torch.equal(
            selected_valid.sum(-1), valid.sum(-1).clamp(max=spec.selected_pools)
        )
    hook.remove()


def test_dsa_tied_scores_select_valid_pools(hf_model):
    # torch.topk does not specify tie order, and padded capacity changes it.
    # Check the admissible set, then use HF attention to verify its raw tokens.
    original = hf_model.layers[3].self_attn
    original.indexer.wq_b.weight.data.zero_()
    spec = Glm53CacheSpec.from_config(hf_model.config, 64)
    core = StaticDSA(original, spec, 16)
    chosen = {}
    hook = core.select_kv.register_forward_hook(record_pool_selection(chosen))
    x = torch.randn(1, 16, 32)
    t = build_transition(spec, 0, 16, 16)
    with torch.no_grad():
        actual = core(
            x,
            t["past_seq_length"],
            t["current_input_length"],
            t["pool_offset"],
            t["pool_new_count"],
            t["pool_write_start"],
            t["query_validity"],
            t["pool_validity"],
            t["tail_indices"],
            t["tail_validity"],
            *spec.initial_caches()[-4:],
        )[0]
        selected = chosen["indices"]
        valid = t["pool_validity"].gather(-1, selected.long()).bool()
        assert torch.equal(valid.sum(-1), torch.minimum(torch.arange(1, 17) // 4, torch.tensor(2))[None])
        assert bool((selected[..., 0] != selected[..., 1]).all())
        raw = (selected[..., None] * 4 + torch.arange(4)).flatten(-2)
        raw = raw.masked_fill(~valid[..., None].expand(1, 16, 2, 4).flatten(-2), -1)
        raw = torch.cat((raw, t["tail_indices"].masked_fill(~t["tail_validity"].bool(), -1)), -1).int()
        ref_hook = original.indexer.register_forward_hook(lambda m, a, v: raw)
        expected = original(x, attention_mask=torch.ones(1, 16, dtype=torch.bool))[0]
        ref_hook.remove()
    hook.remove()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)


def test_ling_normalization_default_preserved():
    torch.manual_seed(35)
    q, k, v = [torch.randn(1, 8, 2, 8) * 0.3 for _ in range(3)]
    g = -torch.rand_like(q)
    beta = torch.rand(1, 8, 2)
    state = torch.randn(1, 2, 8, 8) * 0.1
    expected = recurrent_kda_reference(q, k, v, g, beta, state)
    actual = chunk_kda(
        q, k, v, g, beta, state, chunk_size=8, block_tri_inverse_op=None, chunk_scan_op=None, eye_block_batched=None
    )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("p,n", [(8, 7), (1, 1)])
def test_real_frontend(hf_model, p, n):
    from xhquant.api import FrontendType, to_frontend_graph

    spec = Glm53CacheSpec.from_config(hf_model.config, 64)
    core = StaticGlm53ForCausalLM(hf_model, nn.Linear(32, 64, bias=False), spec, p).eval()
    args = (torch.randn(1, p, 32), *build_transition(spec, 0, n, p).values(), spec.initial_caches())
    with torch.no_grad():
        expected = core(*args)
        graph = to_frontend_graph(core, FrontendType.TorchFX, args)
        actual = graph(*args[:-1], *args[-1])
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=5e-4, atol=2e-5)


@pytest.mark.parametrize("past,n,p", [(-1, 1, 8), (0, 0, 8), (0, 9, 8), (63, 2, 8)])
def test_invalid_transition(hf_model, past, n, p):
    with pytest.raises(ValueError):
        build_transition(Glm53CacheSpec.from_config(hf_model.config, 64), past, n, p)


def test_runner_commit_reset_and_failed_call(hf_model):
    spec = Glm53CacheSpec.from_config(hf_model.config, 64)
    head = nn.Linear(32, 64, bias=False)
    prefill = StaticGlm53ForCausalLM(hf_model, head, spec, 8)
    decode = StaticGlm53ForCausalLM(hf_model, head, spec, 1)
    runner = Glm53Runner(prefill, decode, hf_model.embed_tokens, spec, 8)
    ids = torch.arange(1, 8)[None]
    expected = runner.prefill_tokens(ids)
    old = tuple(t.data.clone() for t in runner.caches)

    def fail(*args):
        args[-1][-4].data.fill_(99)
        raise RuntimeError("execution failed after cache write")

    runner.decode = fail
    with pytest.raises(RuntimeError, match="execution failed"):
        runner.step(torch.tensor([[8]]))
    assert runner.past_length == 7
    for a, b in zip(runner.caches, old, strict=True):
        torch.testing.assert_close(a.data, b)
    runner.decode = decode
    runner.step(torch.tensor([[8]]))
    assert runner.past_length == 8
    runner.reset()
    assert runner.past_length == 0
    torch.testing.assert_close(runner.prefill_tokens(ids), expected)


def test_source_fp8_blocks_and_missing_scale(tmp_path):
    from safetensors.torch import save_file
    from transformers.models.glm5_next import Glm5NextConfig

    from xhmodel_merak.xh_llm.models.glm5_3_flash.checkpoint import Glm53Checkpoint

    config = Glm5NextConfig(text_config=tiny_config().to_dict())
    config.quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    config.save_pretrained(tmp_path)
    weight = torch.ones(257, 129).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    save_file({"probe.weight": weight, "probe.weight_scale_inv": scale}, tmp_path / "model.safetensors")
    reader = Glm53Checkpoint(tmp_path)
    value = reader.tensor("probe.weight", dtype=torch.float32)
    assert value[0, 0] == 1 and value[0, 128] == 2 and value[128, 0] == 3
    assert value[255, 128] == 4 and value[256, 0] == 5 and value[256, 128] == 6
    with pytest.raises(KeyError, match="missing required tensor"):
        reader.tensor("missing.weight")
    save_file({"probe.weight": weight}, tmp_path / "model.safetensors")
    reader = Glm53Checkpoint(tmp_path)
    with pytest.raises(KeyError, match="scale_inv"):
        reader.tensor("probe.weight")


def test_256k_cache_capacity_and_final_chunk_controls():
    cfg = tiny_config()
    cfg.max_position_embeddings = 1048576
    cfg.index_topk = 2048
    spec = Glm53CacheSpec.from_config(cfg, 262144)
    assert spec.pool_capacity == 65536
    assert spec.selected_pools == 512
    assert spec.cache_shapes("deepseek_sparse_attention")[:2] == (
        (1, 1, 262144, 16), (1, 1, 65536, 8),
    )
    controls = build_transition(spec, 262141, 3, 256)
    assert controls["pool_validity"].shape == (1, 256, 65536)
    assert controls["pool_new_count"].item() == 1
    assert controls["pool_write_start"].item() == 65535
    assert controls["query_validity"].sum().item() == 3
    with pytest.raises(ValueError):
        build_transition(spec, 262143, 2, 256)


def test_workflow_registration_and_configuration(tmp_path, hf_model):
    from transformers.models.glm5_next import Glm5NextConfig

    from xhmodel_merak.workflows import AutoWorkflow
    from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
    from xhmodel_merak.xh_llm.models.glm5_3_flash.workflow import Glm53Workflow
    from xhmodel_merak.xh_llm.models.glm5_3_flash.xh_glm53_config import XHGlm53ModelConfig

    Glm5NextConfig(text_config=hf_model.config.to_dict()).save_pretrained(tmp_path)
    recipe = "configs_merak/workflows/xh2a/llm_models/glm5_3_flash/glm5_3_flash_w8a16.yaml"
    workflow = AutoWorkflow.from_config(model_dir=str(tmp_path), config_path=recipe)
    assert isinstance(workflow, Glm53Workflow)
    assert workflow.workflow_config.export["model"]["quant_scheme"]["quant_type"] == "w8a16h0_sefp"
    assert workflow.workflow_config.export["model"]["context_max_length"] == 262144
    assert XHGlm53ModelConfig().context_max_length == 262144
    from xhmodel_merak.model_catalog import get_recipe
    assert get_recipe("glm-5.3-flash").default_context_length == 262144
    quant = workflow.quant(str(tmp_path / "unused"), "cpu")
    assert quant.skipped
    cfg = AutoLLMConfig.from_pretrained(
        dict(
            model_type="Glm5NextForConditionalGeneration",
            model_name="tiny",
            chip_arch="XH2a",
            hf_model=str(tmp_path),
            context_max_length=64,
            prefill_chunk_length=8,
            linear_chunk_size=8,
        )
    )
    model = AutoLLMModel.from_pretrained(config=cfg)
    assert len(model.cache_spec.layer_types) == 4
    with pytest.raises(RuntimeError, match="full-model loading is unsupported"):
        model.get_native_model()
    for invalid in (
        {"batch_size": 2},
        {"linear_chunk_size": 7},
        {"use_cache": False},
        {"max_layers": 46},
        {"spec_decode_mode": "mtp"},
        {"low_memory": False},
        {"prefill_chunk_length": 512, "context_max_length": 256},
    ):
        with pytest.raises(ValueError):
            XHGlm53ModelConfig(**invalid)


@pytest.mark.parametrize("mlp_type", ["sparse", "dense"])
def test_w8_export_reload_and_generation(tmp_path, monkeypatch, mlp_type):
    """Hybrid MoE and KDA-only dense prefixes through export and runtime."""
    import json
    import runpy
    import sys
    from pathlib import Path

    import onnx
    import yaml
    from safetensors.torch import save_file
    from transformers.models.glm5_next import Glm5NextConfig

    from xhmodel_merak.utils import calculate_file_md5
    from xhmodel_merak.xh_llm.hmonnx.auto_llm_hmonnx_model import AutoLLMHONNXModel

    torch.manual_seed(533)
    cfg = tiny_config()
    cfg.num_hidden_layers = 2
    cfg.layer_types = ["linear_attention", "deepseek_sparse_attention" if mlp_type == "sparse" else "linear_attention"]
    cfg.mlp_layer_types = ["dense", mlp_type]
    cfg.indexer_types = ["full", "full"]
    source = Glm5NextTextModel(cfg).eval()
    head = nn.Linear(32, 64, bias=False)
    model_dir = tmp_path / "checkpoint"
    Glm5NextConfig(text_config=cfg.to_dict()).save_pretrained(model_dir)
    weights = {"model.language_model." + k: v for k, v in source.state_dict().items()}
    weights["lm_head.weight"] = head.weight
    save_file(weights, model_dir / "model.safetensors")
    recipe = yaml.safe_load(
        Path("configs_merak/workflows/xh2a/llm_models/glm5_3_flash/glm5_3_flash_w8a16.yaml").read_text()
    )
    recipe["export"]["model"].update(
        prefill_chunk_length=8,
        linear_chunk_size=8,
        context_max_length=64,
        placeholder_export_workers=2 if mlp_type == "sparse" else 1,
        export_vision=False,
    )
    recipe_file = tmp_path / "w8.yaml"
    recipe_file.write_text(yaml.safe_dump(recipe))
    output = tmp_path / "export"
    monkeypatch.setattr(
        sys,
        "argv",
        ["glm53_workflow.py", "--model", str(model_dir), "--config", str(recipe_file), "--output", str(output)],
    )
    runpy.run_path("examples_merak/llm/glm5_3_flash/glm53_workflow.py", run_name="__main__")
    meta_path = next(output.rglob("golden_meta_info.json"))
    meta = json.loads(meta_path.read_text())
    assert meta["model_config"]["quant_scheme"]["quant_type"] == "w8a16h0_sefp"
    assert meta["glm53_export_mode"] == "streamed_ffn"
    spec = Glm53CacheSpec.from_config(cfg, 64)
    # Merak normalizes the first three names for its common HMONNX ABI.
    input_names = ["input_1", "valid_length", "current_length", *spec.input_names()[3:]]
    for stage in ("prefill", "decode"):
        path = meta_path.parent / meta[stage + "_hmonnx"]
        assert calculate_file_md5(str(path)) == meta[stage + "_hmonnx_md5"]
        graph = onnx.load(path, load_external_data=False).graph
        assert [t.name for t in graph.input] == input_names
        assert [t.name for t in graph.output] == spec.output_names()
        assert not any("placeholder" in node.op_type.lower() for node in graph.node)
        tags = [node for node in graph.node if node.op_type == "Tag"]
        assert [onnx.helper.get_attribute_value(next(a for a in n.attribute if a.name == "tag_name"))
                for n in tags] == [b"layer_0", b"layer_1"]
        assert not any(node.op_type == "ReduceMin" for node in graph.node)
        if mlp_type == "sparse":
            select = [node for node in graph.node if node.op_type == "SparseAttentionSelectKV"]
            assert len(select) == 1
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in select[0].attribute}
            assert attrs["score_mode"] == b"hmfp_pool" and attrs["pool_size"] == 4
            assert len(select[0].input) == 5 and len(select[0].output) == 2
            assert "pool_validity" == select[0].input[3]
        assert len(graph.output) == (7 if mlp_type == "sparse" else 5)
        assert all(o.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16 for o in graph.output)
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    model = AutoLLMHONNXModel.from_pretrained(str(meta_path), device_map=["cuda:0"])
    source = source.half().cuda()
    head = head.half().cuda()
    cache = DynamicCache(config=cfg)
    past = 0
    with torch.no_grad():
        # End at capacity with a padded final chunk (L=61, n=3, P=8).
        for n in (7, 1, 1, 8, 5, 8, 8, 8, 8, 7, 3):
            (model.set_decode if n == 1 else model.set_prefill)()
            ids = torch.arange(1, n + 1)[None]
            actual = model.forward(*model.get_data_preprocessor()({"input_ids": ids, "past_seq_length": past}))
            # HF DynamicCache cannot infer length from KDA state alone. Supply
            # the known positions so the KDA-only prefix is a valid reference.
            expected = head(
                source(
                    ids.cuda(),
                    position_ids=torch.arange(past, past + n, device="cuda")[None],
                    past_key_values=cache,
                    use_cache=True,
                ).last_hidden_state[:, -1:]
            )
            assert torch.isfinite(actual).all()
            cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
            assert cosine > 0.99
            assert all(
                t.dtype == torch.float16 and torch.isfinite(t.data).all() for t in model.get_kvcache_mixin().caches
            )
            past += n
    prompt = torch.arange(1, 14, device=model.device)[None]
    kwargs = dict(input_ids=prompt, max_new_tokens=3, do_sample=False, eos_token_id=None, pad_token_id=0)
    generated = model.generate(**kwargs)
    assert generated.shape == (1, 16)
    torch.testing.assert_close(model.generate(**kwargs), generated)
    with pytest.raises(ValueError, match="owns its caches"):
        model.generate(**kwargs, past_key_values=DynamicCache())

    if mlp_type == "sparse":
        # A broadcasted expert bank crosses the int32 address range with only
        # one expert's physical storage. Compare the SDK's bounded fallback
        # against a small bank with identical experts and different high IDs.
        runtime_moe = next(
            m
            for m in model.prefill_model.hmonnx_session.interpreter._graph_module.modules()
            if type(m).__name__ == "MoeBlock"
        )
        original = runtime_moe._buffers
        gate = original["expert_gate_proj_qweight"]
        large_count = 2**31 // gate[0].numel() + 1
        hidden_dim = original["expert_down_proj_qweight"].shape[-1]
        hidden = torch.randn(1, 1, hidden_dim, device=gate.device, dtype=torch.float16) * 0.1
        weights = torch.tensor([[[0.25, 0.75]]], device=gate.device, dtype=torch.float16)
        outputs = []
        try:
            for count in (2, large_count):
                runtime_moe._buffers = {
                    name: value[:1].expand(count, *value.shape[1:])
                    if name.startswith("expert_") and value is not None
                    else value
                    for name, value in original.items()
                }
                indices = torch.tensor([[[count - 2, count - 1]]], device=gate.device, dtype=torch.int64)
                outputs.append(type(runtime_moe).forward(runtime_moe, hidden, weights, indices))
            torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-3, atol=1e-3)
            assert all(torch.isfinite(value).all() for value in outputs[1].values())
        finally:
            runtime_moe._buffers = original
