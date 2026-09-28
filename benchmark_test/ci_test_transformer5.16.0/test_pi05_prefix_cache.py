# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Prefix KV must equal native PiGemma, including padding and both mask modes."""

import pytest
import torch


@pytest.mark.parametrize("causal", [False, True])
def test_prefix_cache_matches_native_decoder(causal, monkeypatch):
    from lerobot.policies import pi_gemma
    from transformers import GemmaConfig
    from transformers.masking_utils import create_causal_mask
    from xhmodel_merak.xh_other_model.models.pi05._prefix_cache import collect_prefix_cache

    # The reference executes the complete native forward. Adapt only its old
    # mask-call signature; Transformers itself consumes the supplied 4D mask.
    def current_mask(*, cache_position, **kwargs):
        assert cache_position.shape == (8,)
        return create_causal_mask(**kwargs)

    monkeypatch.setattr(pi_gemma, "create_causal_mask", current_mask)
    config = GemmaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16, vocab_size=64)
    config._attn_implementation = "eager"
    torch.manual_seed(4891)
    model = pi_gemma.PiGemmaModel(config).eval()
    embeddings = torch.randn(1, 8, 32)
    positions = torch.tensor([[0, 1, 2, 3, 4, 5, 5, 5]])
    allowed = torch.ones(8, 8, dtype=torch.bool)
    allowed[:, 6:] = False
    if causal:
        allowed &= torch.ones(8, 8, dtype=torch.bool).tril()
    mask = torch.where(allowed, 0., -1e9)[None, None]
    with torch.no_grad():
        reference = model(inputs_embeds=embeddings, attention_mask=mask,
                          position_ids=positions, use_cache=True).past_key_values
    actual = collect_prefix_cache(model, embeddings, mask, positions)
    assert len(actual.layers) == len(reference.layers) == 2
    for a, b in zip(actual.layers, reference.layers):
        torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
        torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)


def test_prefix_cache_rejects_incomplete_mask():
    from xhmodel_merak.xh_other_model.models.pi05._prefix_cache import collect_prefix_cache

    with pytest.raises(ValueError, match="complete square 4D"):
        collect_prefix_cache(None, torch.empty(1, 4, 32), torch.ones(1, 4), torch.arange(4)[None])
