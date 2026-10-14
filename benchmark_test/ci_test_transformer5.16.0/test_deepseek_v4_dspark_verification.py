import pytest
import torch

from xhmodel_merak.xh_llm.models.deepseek_v4.cache_abi import CSA, HCA, SLIDING, DeepSeekV4CacheABI
from xhmodel_merak.xh_llm.models.deepseek_v4.dspark_verification import DSparkTargetTransaction, verify_dspark
from xhmodel_merak.xh_llm.models.deepseek_v4.runtime import DeepSeekV4CacheMixin
from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import DeepSeekV4StaticCacheSpec


def test_dspark_greedy_verification_rejects_first_mismatch_and_emits_bonus():
    logits = torch.eye(4).unsqueeze(0) * 10
    proposals = torch.tensor([[0, 1, 2]])
    q = torch.nn.functional.one_hot(proposals, 4).float()
    result = verify_dspark(proposals, q, logits, temperature=0)
    assert result.emitted_tokens.tolist() == [[0, 1, 2, 3]]
    assert result.accepted_count == 3 and result.all_accepted
    proposals = torch.tensor([[0, 3, 2]])
    q = torch.nn.functional.one_hot(proposals, 4).float()
    result = verify_dspark(proposals, q, logits, temperature=0)
    assert result.emitted_tokens.tolist() == [[0, 1]]
    assert result.accepted_count == 1 and not result.all_accepted


def test_dspark_stochastic_rejection_recovers_the_target_distribution():
    p = torch.tensor([.05, .30, .65])
    q = torch.tensor([.80, .15, .05])
    logits = p.log().reshape(1, 1, 3).expand(1, 2, 3)
    generator = torch.Generator().manual_seed(523)
    counts = torch.zeros(3)
    for _ in range(2048):
        proposed = torch.multinomial(q, 1, generator=generator).reshape(1, 1)
        result = verify_dspark(proposed, q.reshape(1, 1, 3), logits, generator=generator)
        counts[result.emitted_tokens[0, 0]] += 1
    torch.testing.assert_close(counts / counts.sum(), p, atol=.04, rtol=0)


@pytest.mark.parametrize("past", [3, 4, 127, 128])
def test_dspark_target_rollback_restores_swa_compressor_and_appended_rows(past):
    spec = DeepSeekV4StaticCacheSpec(max_context_length=16384, prefill_chunk_length=8,
                                    decode_sequence_length=6, sliding_window=16,
                                    latent_head_dim=128, index_head_dim=128, index_topk=4)
    state = DeepSeekV4CacheMixin(DeepSeekV4CacheABI(spec=spec, layer_types=(CSA, HCA, SLIDING)))
    caches = state.cache_inputs()
    before = [[tensor.data.clone() for tensor in cache] for cache in caches]
    transaction = DSparkTargetTransaction(state, past_length=past, input_length=6)
    assert transaction.snapshot_nbytes < caches[0].main.numel() * caches[0].main.element_size()
    for kind, cache in zip(state.abi.layer_types, caches, strict=True):
        cache.swa_kv.data.fill_(7)
        cache.swa_kv.cache_valid_len = 6
        if kind != SLIDING:
            cache.main.data[:, :, past // (4 if kind == CSA else 128)].fill_(3)
        if kind == CSA:
            cache.index_k.data[:, :, past // 4].fill_(5)
        for name, tensor in zip(cache._fields, cache, strict=True):
            if name.endswith("_state"):
                tensor.data.fill_(11)
    transaction.rollback()
    for cache, saved in zip(caches, before, strict=True):
        for actual, expected in zip(cache, saved, strict=True):
            torch.testing.assert_close(actual.data, expected, rtol=0, atol=0)
        assert cache.swa_kv.cache_valid_len == 0
    with pytest.raises(RuntimeError, match="already closed"):
        transaction.commit()
