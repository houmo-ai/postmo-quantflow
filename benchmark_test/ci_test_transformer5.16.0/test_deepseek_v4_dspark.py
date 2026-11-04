# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Real small DeepSeek blocks, DSpark conditioning and transient cache rollback."""
import pytest
import torch
from torch import nn
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

from _small_model_threads import small_model_cpu_threads  # noqa: F401

from xhmodel_merak.xh_llm.models.deepseek_v4.dspark import DSparkContext, DSparkDraft, DSparkMarkov, DSparkSpec
from xhmodel_merak.xh_llm.models.deepseek_v4.dspark_runtime import DSparkSlidingCache, dspark_attention_mask, sample_dspark
from xhmodel_merak.xh_llm.models.deepseek_v4.norm import build_xh_rms_norm


pytestmark = pytest.mark.usefixtures("small_model_cpu_threads")


def _case(device="cpu", *, hidden_size=128):
    torch.manual_seed(523)
    config = DeepseekV4Config(
        hidden_size=hidden_size, moe_intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=128,
        q_lora_rank=64, o_groups=1, o_lora_rank=64, n_routed_experts=2,
        n_shared_experts=1, num_experts_per_tok=1, num_hash_layers=0,
        layer_types=["sliding_attention"] * 3, mlp_layer_types=["moe"] * 3,
        sliding_window=16, index_topk=4, index_n_heads=2, index_head_dim=128,
        vocab_size=64, qk_rope_head_dim=64, max_position_embeddings=64,
    )
    target = DeepseekV4ForCausalLM(config).eval().half().to(device)
    spec = DSparkSpec(target_layer_ids=(0, 1, 2), hidden_size=hidden_size, head_dim=128,
                      sliding_window=16, context_chunk_length=8, vocab_size=64,
                      noise_token_id=63, markov_rank=8, max_context_length=256)
    draft = DSparkDraft.from_hf(target, target.lm_head, spec).eval().half().to(device)
    projection = nn.Linear(3 * hidden_size, hidden_size, bias=False).half().to(device)
    norm = build_xh_rms_norm(hidden_size, 1e-6, weight=torch.ones(hidden_size, dtype=torch.float16, device=device))
    context = DSparkContext(projection, norm, list(draft.layers), spec, input_sequence_length=8, position_rope=draft.position_rope).eval().half().to(device)
    context_decode = DSparkContext(projection, norm, list(draft.layers), spec,
                                    input_sequence_length=spec.block_size + 1, position_rope=draft.position_rope).eval().half().to(device)
    markov = DSparkMarkov(torch.randn(64, 8, device=device).half() * .1,
                          torch.randn(64, 8, device=device).half() * .1,
                          confidence_weight=torch.randn(1, hidden_size + spec.markov_rank, device=device).half() * .1)
    return target, spec, draft, context, context_decode, markov


@pytest.mark.parametrize("overrides", [
    {"block_size": True}, {"block_size": 1.5}, {"hc_mult": 0},
    {"rope_dim": 3}, {"rope_dim": 514}, {"noise_token_id": True},
    {"target_layer_ids": (True, 41, 42)},
])
def test_dspark_spec_rejects_incompatible_runtime_geometry(overrides):
    with pytest.raises(ValueError):
        DSparkSpec(**overrides)


def test_dspark_real_three_block_draft_uses_shared_head_and_all_anchor_positions():
    target, spec, draft, context, _, markov = _case()
    cache = DSparkSlidingCache(spec)
    features = torch.randn(1, 8, 384).half()
    cache.commit_context(context, features, current_length=7)
    snapshot = tuple(value.data.clone() for value in cache.caches)
    valid = tuple(value.cache_valid_len for value in cache.caches)
    ids = torch.full((1, spec.block_size), spec.noise_token_id, dtype=torch.int32)
    ids[:, 0] = 2
    cache.begin_proposal()
    logits, hidden = draft(target.model.embed_tokens(ids.long()),
                           torch.tensor([7], dtype=torch.int32), torch.tensor([5], dtype=torch.int32),
                           dspark_attention_mask(spec, 7), *cache.caches)
    assert draft.lm_head.weight.data_ptr() == target.lm_head.weight.data_ptr()
    assert logits.shape == (1, 5, 64) and hidden.shape == (1, 5, 128)
    assert torch.isfinite(logits).all()
    tokens, q, confidence = sample_dspark(logits, hidden, ids[:, 0], markov,
                                          generator=torch.Generator().manual_seed(7))
    assert tokens.shape == (1, 5) and confidence.shape == (1, 5, 1)
    torch.testing.assert_close(q.sum(-1), torch.ones(1, 5), rtol=1e-6, atol=1e-6)
    cache.discard_proposal()
    for actual, before, old_valid in zip(cache.caches, snapshot, valid, strict=True):
        torch.testing.assert_close(actual.data, before, rtol=0, atol=0)
        assert actual.cache_valid_len == old_valid
    assert cache.past_length == 7


def test_dspark_markov_conditioning_uses_corrected_proposal_probabilities():
    embedding = torch.eye(4)
    weight = torch.tensor([[0., 8., 0., 0.], [0., 0., 8., 0.],
                           [0., 0., 0., 8.], [8., 0., 0., 0.]])
    markov = DSparkMarkov(embedding, weight)
    base = torch.zeros(1, 3, 4)
    hidden = torch.zeros(1, 3, 4)
    tokens, q, confidence = sample_dspark(base, hidden, torch.tensor([0]), markov, temperature=0)
    assert tokens.tolist() == [[3, 2, 1]]
    assert confidence is None
    assert q[0].argmax(-1).tolist() == [3, 2, 1]
    tokens, q, _ = sample_dspark(base, hidden, torch.tensor([0]), markov,
                                 generator=torch.Generator().manual_seed(1))
    previous = torch.cat((torch.tensor([[0]]), tokens[:, :-1]), dim=1)
    for i in range(3):
        reference = torch.softmax(torch.nn.functional.linear(embedding[previous[:, i]], weight), dim=-1)
        torch.testing.assert_close(q[:, i], reference)


@pytest.mark.parametrize("past", [0, 3, 15, 16, 127, 128, 4096])
def test_dspark_noncausal_mask_uses_sliding_context_and_entire_query_block(past):
    spec = DSparkSpec()
    mask = dspark_attention_mask(spec, past)[0, 0]
    retained = min(past, 127)
    for i in range(5):
        expected = list(range(max(retained + i - 127, 0), retained + 5))
        assert (mask[i] == 0).nonzero().flatten().tolist() == expected


def test_dspark_context_commit_requires_transient_cache_discard():
    _, spec, _, context, _, _ = _case()
    cache = DSparkSlidingCache(spec)
    cache.begin_proposal()
    with pytest.raises(RuntimeError, match="Discard transient"):
        cache.commit_context(context, torch.randn(1, 8, 384).half(), current_length=1)
    cache.discard_proposal()
    assert cache.past_length == 0


def test_target_verify_emits_all_logits_and_ordered_mean_mhc_features():
    from xhmodel_merak.xh_llm.models.deepseek_v4.cache_abi import DeepSeekV4CacheABI
    from xhmodel_merak.xh_llm.models.deepseek_v4.full_model import StaticDeepSeekV4ForCausalLM
    from xhmodel_merak.xh_llm.models.deepseek_v4.runtime import DeepSeekV4CacheMixin, DeepSeekV4DataPreprocess
    from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import DeepSeekV4StaticCacheSpec

    target, spec, _, _, _, _ = _case()
    model = StaticDeepSeekV4ForCausalLM.from_hf(target, input_sequence_length=6,
                                               max_context_length=64, auxiliary_layer_ids=(0, 1, 2),
                                               return_all_logits=True, swa_backing_length=spec.cache_backing_length).half()
    cache_spec = DeepSeekV4StaticCacheSpec(max_context_length=64, prefill_chunk_length=8,
                                          decode_sequence_length=6, sliding_window=16,
                                          latent_head_dim=128, index_topk=4)
    cache = DeepSeekV4CacheMixin(DeepSeekV4CacheABI(spec=cache_spec, layer_types=("sliding_attention",) * 3))
    processor = DeepSeekV4DataPreprocess(token_embedding=target.model.embed_tokens,
                                        input_sequence_length=6, cache_mixin=cache)
    args = processor({"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "past_seq_length": 0})
    observed = []
    hooks = [layer.register_forward_hook(lambda _module, _args, output: observed.append(output.hidden_states.clone()))
             for layer in model.blocks.layers]
    try:
        output = model(*args)
    finally:
        for hook in hooks:
            hook.remove()
    assert output.logits.shape == (1, 6, 64)
    assert output.target_hidden.shape == (1, 6, 384)
    torch.testing.assert_close(output.target_hidden, torch.cat([value.mean(2) for value in observed], dim=-1))
    assert torch.isfinite(output.logits).all()


def test_dspark_target_verify_hmonnx_preserves_all_logits_and_features(tmp_path):
    from xhquant.api import (HMONNXInference, PrecisionMode, ptq_quantize,
                             to_export_hmonnx_from_quanted_graph, to_frontend_graph, to_quant_graph)
    from xhmodel_merak.xh_llm.models.deepseek_v4.cache_abi import DeepSeekV4CacheABI
    from xhmodel_merak.xh_llm.models.deepseek_v4.full_model import StaticDeepSeekV4ForCausalLM
    from xhmodel_merak.xh_llm.models.deepseek_v4.runtime import DeepSeekV4CacheMixin, DeepSeekV4DataPreprocess
    from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import DeepSeekV4StaticCacheSpec

    target, spec, _, _, _, _ = _case("cuda:0", hidden_size=64)
    model = StaticDeepSeekV4ForCausalLM.from_hf(target, input_sequence_length=6,
        max_context_length=64, auxiliary_layer_ids=(0, 1, 2), return_all_logits=True,
        swa_backing_length=spec.cache_backing_length).half()
    cache_spec = DeepSeekV4StaticCacheSpec(max_context_length=64, prefill_chunk_length=8,
                                          decode_sequence_length=6, sliding_window=16,
                                          latent_head_dim=128, index_topk=4)
    cache = DeepSeekV4CacheMixin(DeepSeekV4CacheABI(spec=cache_spec, layer_types=("sliding_attention",) * 3))
    cache.to("cuda:0")
    processor = DeepSeekV4DataPreprocess(token_embedding=target.model.embed_tokens,
                                        input_sequence_length=6, cache_mixin=cache).to("cuda:0")
    args = processor({"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "past_seq_length": 0})
    graph = to_quant_graph(to_frontend_graph(model, "TorchFX", args), "XH2a", spec.quant_scheme(8))
    flat_args = list(args[:-1]) + list(args[-1])
    ptq_quantize(graph, [flat_args], PrecisionMode.ALIGNED, ["cuda:0"], infer_shape=False)
    cache.reset_kv_cache()
    expected = [value.dequantize().clone() for value in graph(*flat_args)]
    assert expected[0].shape == (1, 6, 64)
    assert expected[-1].shape == (1, 6, 3 * spec.hidden_size)
    cache.reset_kv_cache()
    path = tmp_path / "target_decode.onnx"
    to_export_hmonnx_from_quanted_graph(graph, flat_args, str(path))
    session = HMONNXInference(str(path)).to("cuda:0")
    actual = session(*flat_args)
    actual = tuple(actual.values()) if isinstance(actual, dict) else actual
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value.float(), reference.float(), rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("bits", [4, 5, 8])
def test_dspark_config_resolves_decode_width_and_one_shared_head(bits):
    from xhmodel_merak.xh_llm.models.deepseek_v4.xh_deepseek_v4_config import XHDeepSeekV4ModelConfig

    config = XHDeepSeekV4ModelConfig(model_name="tiny-test", dspark={"head_weight_bits": bits})
    assert config.num_logits_to_keep == 6
    assert config.dspark["block_size"] == 5
    scheme = config.quant_scheme.to_dict() if hasattr(config.quant_scheme, "to_dict") else config.quant_scheme
    assert scheme["nodes"]["lm_head"]["quant_type"] == f"w{bits}a8h1_sefp"
    # Keep DSpark's W4 routed-expert and shared-head format overrides while
    # running the dense body only at W8A8.
    routed = DSparkSpec(head_weight_bits=bits).quant_scheme(8)
    assert routed["quant_type"] == "w8a8h1_sefp"
    assert routed["nodes_cfg"]["lm_head"]["quant_type"] == f"w{bits}a8h1_sefp"
    assert routed["ops_cfg"]["MoeBlock"]["w_schema"] == {"bits": 4, "nshare": 64}
    from xhmodel_merak.xh_llm.types import BaseLLMModelConfig
    assert BaseLLMModelConfig.from_dict(config.to_dict()).num_logits_to_keep == 6


def test_dspark_exporter_switches_to_six_token_verify_shape():
    from xhmodel_merak.xh_llm.models.deepseek_v4.deepseek_v4_model import XHDeepSeekV4Model
    from xhmodel_merak.xh_llm.models.deepseek_v4.xh_deepseek_v4_config import XHDeepSeekV4ModelConfig

    model = XHDeepSeekV4Model(XHDeepSeekV4ModelConfig(model_name="tiny-test", dspark={"block_size": 5}))
    model.embed_tokens = nn.Embedding(64, 4096).half()
    model.set_prefill()
    model.set_decode()
    assert model.wrap_cfg.input_sequence_length == 6
    assert model._decode_input_sequence_length == 6
    assert model.get_data_preprocessor().input_sequence_length == 6
    model.set_prefill()
    assert model.wrap_cfg.input_sequence_length == 256


@pytest.mark.parametrize("hybrid", [False, True])
def test_dspark_full_greedy_loop_matches_real_target_and_replays_rejected_prefixes(hybrid):
    from dataclasses import replace
    from xhmodel_merak.xh_llm.models.deepseek_v4.cache_abi import DeepSeekV4CacheABI
    from xhmodel_merak.xh_llm.models.deepseek_v4.dspark_engine import DSparkEngine
    from xhmodel_merak.xh_llm.models.deepseek_v4.full_model import StaticDeepSeekV4ForCausalLM
    from xhmodel_merak.xh_llm.models.deepseek_v4.runtime import DeepSeekV4CacheMixin, DeepSeekV4DataPreprocess
    from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import DeepSeekV4StaticCacheSpec

    target, spec, draft, context, context_decode, markov = _case()
    context_length = 256 if hybrid else 64
    budget = 24 if hybrid else 12
    if hybrid:
        config = DeepseekV4Config(**target.config.to_dict())
        config.num_hidden_layers = 4
        config.layer_types = ["sliding_attention", "sliding_attention",
                              "compressed_sparse_attention", "heavily_compressed_attention"]
        config.mlp_layer_types = ["moe"] * 4
        target = DeepseekV4ForCausalLM(config).half().eval()
        spec = replace(spec, target_layer_ids=(1, 2, 3))
        draft.lm_head = target.lm_head
    models = {length: StaticDeepSeekV4ForCausalLM.from_hf(
        target, input_sequence_length=length, max_context_length=context_length,
        auxiliary_layer_ids=spec.target_layer_ids, return_all_logits=True,
        swa_backing_length=spec.cache_backing_length).half().eval() for length in (8, 6)}
    cache_spec = DeepSeekV4StaticCacheSpec(max_context_length=context_length, prefill_chunk_length=8,
                                          decode_sequence_length=6, sliding_window=16,
                                          latent_head_dim=128, index_topk=4)
    cache = DeepSeekV4CacheMixin(DeepSeekV4CacheABI(spec=cache_spec, layer_types=tuple(target.config.layer_types)))
    processors = {length: DeepSeekV4DataPreprocess(token_embedding=target.model.embed_tokens,
                                                  input_sequence_length=length, cache_mixin=cache)
                  for length in (8, 6)}

    def forward(ids, past, prefill):
        length = 8 if prefill else 6
        output = models[length](*processors[length]({"input_ids": ids, "past_seq_length": past}))
        cache.apply_state_outputs(output.csa_states, output.hca_states)
        return output.logits, output.target_hidden

    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    if hybrid:
        # Cross the released HCA ratio=128, CSA ratio=4 and SWA window=16.
        # Proposals straddle compression boundaries before being rejected.
        prompt = (torch.arange(123) % 60 + 1).reshape(1, -1)
    engine = DSparkEngine(spec, target_forward=forward, target_cache=cache,
                          embedding=target.model.embed_tokens, context=context,
                          context_decode=context_decode, draft=draft, markov=markov)
    actual = engine.generate(prompt, max_new_tokens=budget, temperature=0)
    assert engine.statistics.rounds > 0 and engine.statistics.replayed_target_tokens > 0
    cache.reset_kv_cache()
    past = 0
    for chunk in prompt.split(8, dim=1):
        logits, _ = forward(chunk, past, True)
        past += chunk.shape[1]
    anchor = logits[:, chunk.shape[1] - 1].argmax(-1, keepdim=True)
    reference = [prompt, anchor]
    for _ in range(budget - 1):
        logits, _ = forward(anchor, past, False)
        past += 1
        anchor = logits[:, 0].argmax(-1, keepdim=True)
        reference.append(anchor)
    assert torch.equal(actual, torch.cat(reference, dim=1))


@pytest.mark.parametrize("component", ["context", "context_decode", "draft", "markov"])
def test_dspark_components_hmonnx_export_and_runtime(tmp_path, component):
    from xhquant.api import (HMONNXInference, PrecisionMode, ptq_quantize,
                             to_export_hmonnx_from_quanted_graph, to_frontend_graph, to_quant_graph)
    from xhmodel_merak.xh_llm.shared_head import bind_quantized_head, quantized_head

    assert torch.cuda.is_available(), "DSpark HMONNX regression requires a GPU"
    target, spec, draft, context, context_decode, markov = _case("cuda:0", hidden_size=64)
    past = torch.tensor([3], dtype=torch.int32, device="cuda:0")
    cache = DSparkSlidingCache(spec, device="cuda:0")
    for value in cache.caches:
        value.cache_valid_len = 3
    if component in ("context", "context_decode"):
        length = 8 if component == "context" else 6
        model = context if component == "context" else context_decode
        args = [torch.randn(1, length, 3 * spec.hidden_size, device="cuda:0").half(), past,
                torch.tensor([length - 1], dtype=torch.int32, device="cuda:0"),
                *cache.caches]
    elif component == "draft":
        model = draft
        ids = torch.full((1, 5), spec.noise_token_id, device="cuda:0", dtype=torch.int32)
        ids[:, 0] = 2
        args = [target.model.embed_tokens(ids.long()), past,
                torch.tensor([5], dtype=torch.int32, device="cuda:0"),
                dspark_attention_mask(spec, 3, device="cuda:0"), *cache.caches]
    else:
        model = markov
        args = [torch.tensor([[2]], device="cuda:0", dtype=torch.int32),
                torch.randn(1, 1, spec.hidden_size, device="cuda:0").half()]
    scheme = spec.quant_scheme(8)
    graph = to_quant_graph(to_frontend_graph(model, "TorchFX", args), "XH2a", scheme)
    if component == "draft":
        class TargetHead(nn.Module):
            def __init__(self, head):
                super().__init__()
                self.lm_head = head

            def forward(self, hidden):
                return self.lm_head(hidden)

        head_args = [torch.randn(1, 1, spec.hidden_size, device="cuda:0").half()]
        owner = to_quant_graph(to_frontend_graph(TargetHead(target.lm_head), "TorchFX", head_args),
                               "XH2a", {"quant_type": "w5a8h1_sefp"})
        ptq_quantize(owner, [head_args], PrecisionMode.ALIGNED, ["cuda:0"], infer_shape=False)
        shared = quantized_head(owner)
        bind_quantized_head(graph, shared)
        assert graph.lm_head.qweight.data_ptr() == shared.qweight.data_ptr()
        assert graph.lm_head.w_cfg.qspec.man_bit == 5
    initial = [(value.data.clone(), value.cache_valid_len) for value in cache.caches]
    ptq_quantize(graph, [args], PrecisionMode.ALIGNED, ["cuda:0"], infer_shape=False)
    if component == "draft":
        from xhquant.quantization.xh2a.qmodules.moeblock import QMoeBlock
        from xhquant.quantization.xh2a.qmodules.linear import QLinear

        experts = [module for module in graph.modules() if isinstance(module, QMoeBlock)]
        assert len(experts) == 3
        for module in experts:
            assert module.w_cfg.qspec.man_bit == module.down_w_cfg.qspec.man_bit == 4
            assert module.w_cfg.qspec.nshare == module.down_w_cfg.qspec.nshare == 64
        shared_linears = [module for name, module in graph.named_modules()
                          if "shared_expert" in name and isinstance(module, QLinear)]
        assert len(shared_linears) == 9
        assert all(module.w_cfg.qspec.man_bit == 8 for module in shared_linears)
    for value, (data, valid) in zip(cache.caches, initial, strict=True):
        value.data.copy_(data)
        value.cache_valid_len = valid
    outputs = graph(*args)
    expected = [value.dequantize().clone() for value in outputs]
    for value, (data, valid) in zip(cache.caches, initial, strict=True):
        value.data.copy_(data)
        value.cache_valid_len = valid
    path = tmp_path / f"{component}.onnx"
    to_export_hmonnx_from_quanted_graph(graph, args, str(path))
    import onnx
    exported = onnx.load(str(path), load_external_data=False).graph
    names = {value.name for value in exported.input}
    assert not names.intersection({"cos", "sin", "input_ids"})
    if component != "markov":
        assert sum(node.op_type == "DynamicSlice" for node in exported.node) == 2
        assert names == ({"noise_embedding", "past_length", "current_length", "attn_mask",
                          "cache_0", "cache_1", "cache_2"} if component == "draft" else
                         {"target_hidden", "past_length", "current_length", "cache_0", "cache_1", "cache_2"})
    session = HMONNXInference(str(path))
    session.to("cuda:0")
    actual = session(*args)
    if isinstance(actual, dict):
        actual = tuple(actual.values())
    assert len(actual) == len(expected)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value.float(), reference.float(), rtol=3e-3, atol=3e-3)
    if component == "draft":
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceV2
        from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader

        loader = MultiHMONNXLoader({"draft": str(path)})
        summary = loader.initializer_pool.summary()
        assert summary["runtime_w4_packed_initializers"] == 9
        assert summary["runtime_w4_packed_saved_nbytes"] == 3 * 2 * 3 * spec.hidden_size * 64 // 2
        packed = HMONNXInferenceV2.from_onnx_graph(str(path), loader.graphs["draft"])
        packed.to("cuda:0")
        for value, (data, valid) in zip(cache.caches, initial, strict=True):
            value.data.copy_(data)
            value.cache_valid_len = valid
        actual = packed(*args)
        if isinstance(actual, dict):
            actual = tuple(actual.values())
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value.float(), reference.float(), rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("head_bits", [4, 5, 8])
def test_dspark_bundle_exports_sequentially_and_shares_the_target_head(tmp_path, head_bits):
    import json
    import onnx
    from dataclasses import replace
    from xhquant.api import (PrecisionMode, ptq_quantize, to_export_hmonnx_from_quanted_graph,
                             to_frontend_graph, to_quant_graph)
    from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader
    from xhmodel_merak.xh_llm.models.deepseek_v4.dspark_export import _HeadOwner, export_dspark_components
    from xhmodel_merak.xh_llm.shared_head import quantized_head

    # Retain two G64 groups here for every head width. Other GPU tests use
    # a narrower residual model. Both body and head activations use A8.
    target, spec, draft, context, _, markov = _case("cuda:0")
    draft.spec = replace(spec, head_weight_bits=head_bits)
    args = [torch.zeros(1, 1, 128, dtype=torch.float16, device="cuda:0")]
    owner = to_quant_graph(to_frontend_graph(_HeadOwner(target.lm_head), "TorchFX", args),
                            "XH2a", {"quant_type": f"w{head_bits}a8h1_sefp"})
    ptq_quantize(owner, [args], PrecisionMode.ALIGNED, ["cuda:0"], infer_shape=False)
    head_path = tmp_path / "target_head.onnx"
    to_export_hmonnx_from_quanted_graph(owner, args, str(head_path), save_as_external_data=True)
    result = export_dspark_components(draft, context.main_proj, context.main_norm, markov,
                                       target.model.embed_tokens, tmp_path,
                                       activation_bits=8, device="cuda:0",
                                       fixed_target_head=quantized_head(owner))
    assert json.loads((tmp_path / "dspark_runtime_meta_info.json").read_text()) == json.loads(json.dumps(result))
    paths = {name: str(tmp_path / value["path"]) for name, value in result["components"].items()}
    assert len(paths) == 4
    for path in paths.values():
        for node in onnx.load(path, load_external_data=False).graph.node:
            if node.op_type == "Linear":
                attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                assert attributes["hmfp_act_man_bit"] == 8
    loader = MultiHMONNXLoader({"target_head": str(head_path), **paths})
    summary = loader.initializer_pool.summary()
    assert summary["runtime_w4_packed_initializers"] == (10 if head_bits == 4 else 9)
    assert summary["shared_saved_nbytes"] > 0

    def weight_name(path):
        graph = onnx.load(path, load_external_data=False).graph
        nodes = [node for node in graph.node if node.op_type == "Linear"
                 and any(attr.name == "hmfp_weight_man_bit" and attr.i == head_bits for attr in node.attribute)
                 and node.input[1] == "lm_head.qweight"]
        assert len(nodes) == 1
        return nodes[0].input[1]

    owner_weight = loader.initializer_pool.graphs["target_head"][weight_name(head_path)]
    draft_weight = loader.initializer_pool.graphs["dspark_draft_decode"][weight_name(paths["dspark_draft_decode"])]
    assert owner_weight is draft_weight


@pytest.mark.parametrize("hybrid", [True, False])
def test_complete_dspark_hmonnx_package_loads_shared_graphs_and_generates(tmp_path, monkeypatch, hybrid):
    import copy
    from types import SimpleNamespace
    from xhquant.api import (PrecisionMode, ptq_quantize, to_export_hmonnx_from_quanted_graph,
                             to_frontend_graph, to_quant_graph)
    from xhmodel_merak.xh_llm.models.deepseek_v4.cache_abi import DeepSeekV4CacheABI
    from xhmodel_merak.xh_llm.models.deepseek_v4.deepseek_v4_hmonnx_inference import XHDeepSeekV4HMONNXModel
    from xhmodel_merak.xh_llm.models.deepseek_v4.dspark_export import export_dspark_components
    from xhmodel_merak.xh_llm.models.deepseek_v4.full_model import StaticDeepSeekV4ForCausalLM
    from xhmodel_merak.xh_llm.models.deepseek_v4.runtime import DeepSeekV4CacheMixin, DeepSeekV4DataPreprocess
    from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import DeepSeekV4StaticCacheSpec
    from xhmodel_merak.xh_llm.shared_head import bind_quantized_head, quantized_head
    from xhmodel_merak.xh_llm.types import KVCacheConfig, LLMModelMeta

    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    target, spec, draft, context, _, markov = _case("cuda:0", hidden_size=64)
    draft_config = copy.deepcopy(target.config)
    layer_types = ["sliding_attention"] * 3
    maximum_context = 160 if hybrid else 64
    if hybrid:
        config = copy.deepcopy(target.config)
        config.max_position_embeddings = maximum_context
        layer_types = ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"]
        config.num_hidden_layers = len(layer_types)
        config.layer_types = layer_types
        config.mlp_layer_types = ["moe"] * len(layer_types)
        target = DeepseekV4ForCausalLM(config).half().eval().to("cuda:0")
    torch.manual_seed(524)
    draft_source = DeepseekV4ForCausalLM(draft_config).half().eval().to("cuda:0")
    from dataclasses import replace
    spec = replace(spec, max_context_length=maximum_context)
    draft = DSparkDraft.from_hf(draft_source, target.lm_head, spec)
    hf = tmp_path / "hf_config"
    target.config.save_pretrained(hf)
    embedding_file = tmp_path / "quant_embedding.pt"
    torch.save({"weight": target.model.embed_tokens.weight.detach().cpu()}, embedding_file)
    cache_spec = DeepSeekV4StaticCacheSpec(max_context_length=maximum_context, prefill_chunk_length=8,
                                          decode_sequence_length=6, sliding_window=16,
                                          latent_head_dim=128, index_topk=4)
    cache = DeepSeekV4CacheMixin(DeepSeekV4CacheABI(spec=cache_spec, layer_types=tuple(layer_types)))
    cache.to("cuda:0")
    models = {length: StaticDeepSeekV4ForCausalLM.from_hf(target, input_sequence_length=length,
              max_context_length=maximum_context, auxiliary_layer_ids=(0, 1, 2), return_all_logits=length == 6,
              swa_backing_length=spec.cache_backing_length).half() for length in (8, 6)}
    paths, head, frontends, inputs = {}, None, {}, {}
    for role, length in (("prefill", 8), ("decode", 6)):
        processor = DeepSeekV4DataPreprocess(token_embedding=target.model.embed_tokens,
                                             input_sequence_length=length, cache_mixin=cache).to("cuda:0")
        args = processor({"input_ids": torch.arange(1, length + 1).reshape(1, length), "past_seq_length": 0})
        frontends[role] = to_frontend_graph(models[length], "TorchFX", args)
        inputs[role] = list(args[:-1]) + list(args[-1])
    for role in ("prefill", "decode"):
        graph = to_quant_graph(frontends[role], "XH2a", spec.quant_scheme(8))
        if head is not None:
            bind_quantized_head(graph, head)
        flat = inputs[role]
        ptq_quantize(graph, [flat], PrecisionMode.ALIGNED, ["cuda:0"], infer_shape=False)
        if head is None:
            head = quantized_head(graph)
        paths[role] = tmp_path / role / f"{role}.onnx"
        cache.reset_kv_cache()
        to_export_hmonnx_from_quanted_graph(graph, flat, str(paths[role]), save_as_external_data=True)
    metadata = export_dspark_components(draft, context.main_proj, context.main_norm, markov,
                                         target.model.embed_tokens, tmp_path, activation_bits=8,
                                         device="cuda:0", fixed_target_head=head)
    config = SimpleNamespace(max_layers=len(layer_types), context_max_length=maximum_context, prefill_chunk_length=8,
                              num_logits_to_keep=6, batch_size=1, dspark={"block_size": 5})
    meta = LLMModelMeta(model_config=config, hf_config=str(hf), quant_embedding=str(embedding_file),
                         kv_cache=KVCacheConfig(num_layers=len(layer_types)), prefill_hmonnx=str(paths["prefill"]),
                         decode_hmonnx=str(paths["decode"]), deepseek_v4_state_abi_version=2,
                         layer_types=layer_types, dspark=metadata)
    devices = [f"cuda:{index}" for index in range(min(2, torch.cuda.device_count()))]
    runtime = XHDeepSeekV4HMONNXModel(meta, device_map=devices,
                                    enable_auto_offload=len(devices) > 1,
                                    enable_cuda_graph=True).to("cuda:0")
    assert runtime.shared_weight_summary["shared_initializer_references"] > 0
    draft_device = torch.device(f"cuda:{min(1, torch.cuda.device_count() - 1)}")
    engine = runtime.build_dspark_engine(device=str(draft_device))
    assert all(session.device == draft_device for session in
               (engine.context, engine.context_decode, engine.draft, engine.markov))
    target_forward = engine.target_forward
    calls = []

    def record_target(ids, past, prefill):
        logits, features = target_forward(ids, past, prefill)
        if not prefill:
            calls.append((ids.clone(), past, logits.clone()))
        return logits, features

    engine.target_forward = record_target
    # These prompts put proposal rounds across the HCA/CSA boundary at
    # 128 or the SWA boundary at 16. Eight output tokens retain multiple
    # rounds and rejected-prefix replay without repeatedly rebuilding the
    # same 123-token reference prefix for another sixteen decode steps.
    prompt_length, budget = (123, 8) if hybrid else (9, 8)
    assert prompt_length < (128 if hybrid else spec.sliding_window) < prompt_length + budget
    prompt = (torch.arange(prompt_length, device="cuda:0") % (spec.vocab_size - 2) + 1).unsqueeze(0)
    result = engine.generate(prompt, max_new_tokens=budget, temperature=0)
    assert result.shape == (1, prompt_length + budget)
    assert torch.equal(result[:, :prompt_length].to(prompt.device), prompt)
    assert engine.statistics.rounds >= 2
    engine.target_forward = target_forward
    if hybrid:
        # A8 compressor/attention rounding depends on the target query batch.
        # Compare identical queries against a clean committed prefix instead
        # of treating one-token and six-token target logits as bit-identical.
        groups = []
        for call in calls:
            if groups and groups[-1][0][1] == call[1]:
                groups[-1].append(call)
            else:
                groups.append([call])
        assert engine.statistics.replayed_target_tokens > 0
        for index, group in enumerate(groups):
            engine.target_cache.reset_kv_cache()
            past = 0
            for chunk in prompt.split(spec.context_chunk_length, dim=1):
                target_forward(chunk, past, True)
                past += chunk.shape[1]
            # Only replay each earlier round's committed call. Rejected
            # proposal suffixes never enter this independently rebuilt cache.
            for previous in groups[:index]:
                committed_ids, committed_past, _ = previous[-1]
                target_forward(committed_ids, committed_past, False)
            ids, past, recorded = group[0]
            clean, _ = target_forward(ids, past, False)
            torch.testing.assert_close(clean[:, :ids.shape[1]], recorded[:, :ids.shape[1]], rtol=0, atol=0)
        return
    # Check the complete loaded quantized runtime against its autoregressive
    # verifier. Rejection and cache replay must preserve the target sequence.
    engine.target_cache.reset_kv_cache()
    past = 0
    for chunk in prompt.split(spec.context_chunk_length, dim=1):
        logits, _ = engine.target_forward(chunk, past, True)
        past += chunk.shape[1]
    last = logits[:, 0] if logits.shape[1] == 1 else logits[:, chunk.shape[1] - 1]
    expected = [prompt]
    for _ in range(budget):
        token = last.argmax(-1, keepdim=True).to(prompt.device)
        expected.append(token)
        logits, _ = engine.target_forward(token, past, False)
        past += 1
        last = logits[:, 0]
    assert torch.equal(result.to(prompt.device), torch.cat(expected, dim=1))


@pytest.mark.parametrize("length", [5, 6, 256])
@pytest.mark.parametrize("past", [0, 127, 128, 262143])
def test_dspark_internal_rope_matches_absolute_checkpoint_positions(length, past):
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
    from xhmodel_merak.xh_llm.models.deepseek_v4.dspark import DSparkPositionRope
    from xhmodel_merak.xh_llm.models.deepseek_v4.full_model import _build_rope_cache

    rotary = DeepseekV4RotaryEmbedding(DeepseekV4Config())
    module = DSparkPositionRope(rotary, DSparkSpec(), length).half()
    actual = module(torch.tensor([past], dtype=torch.int32))
    expected = _build_rope_cache(rotary.main_inv_freq, rotary.main_attention_scaling,
                                 torch.arange(past, past + length))
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference.half(), rtol=0, atol=0)


def test_dspark_rope_rebuilds_fp32_frequencies_after_hf_half_cast():
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
    from xhmodel_merak.xh_llm.models.deepseek_v4.full_model import _build_rope_cache

    target, spec, draft, _, _, _ = _case()
    assert target.model.rotary_emb.main_inv_freq.dtype == torch.float16
    rotary = DeepseekV4RotaryEmbedding(target.config)
    expected = _build_rope_cache(rotary.main_inv_freq, rotary.main_attention_scaling, torch.arange(127, 132))
    actual = draft.position_rope(torch.tensor([127], dtype=torch.int32))
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference.half(), rtol=0, atol=0)
