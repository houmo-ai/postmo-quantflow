"""MTP scheduling boundaries with real tensor state mutations and rollback."""

from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_llm.models.qwen4_exp import generation


def _logits(tokens):
    output = torch.full((1, len(tokens), 12), -100.0)
    for index, token in enumerate(tokens):
        output[0, index, (token + 1) % 12] = 100
    return output


@pytest.mark.parametrize("candidate_matches", [False, True])
def test_legacy_single_candidate_preserves_cache_and_captures_acceptance_and_rejection(monkeypatch, candidate_matches):
    states = {"recurrent": torch.zeros(1)}
    embedding = torch.nn.Embedding(12, 2)
    with torch.no_grad():
        embedding.weight[:, 0] = torch.arange(12)
        embedding.weight[:, 1] = -torch.arange(12)
    model = SimpleNamespace(
        device="cpu",
        embed_tokens=embedding,
        meta_info=SimpleNamespace(model_config=SimpleNamespace(prefill_chunk_length=4, context_max_length=32)),
        _qwen4_state=SimpleNamespace(
            tensors=states,
            group_start_position=torch.zeros(3, 1, 1, dtype=torch.int32),
            contract=SimpleNamespace(
                tensors=[
                    SimpleNamespace(name="recurrent", state_kind="delta_recurrent"),
                ]
            ),
        ),
        _data_processor=None,
    )
    model.prefill = False
    model.set_prefill = lambda: setattr(model, "prefill", True)
    model.set_decode = lambda: setattr(model, "prefill", False)
    model.verify_model = object()
    model._qwen4_verify_length = 4
    model.verify_calls = 0

    def set_verify():
        model.prefill = False
        model.verify_calls += 1

    model.set_verify = set_verify
    pages = [torch.zeros(2, 1, 1, 64, 1)]
    calls = []

    def target(model, caches, ids, past, *, position_ids, capture=None, return_all=False):
        tokens = ids[0].tolist()
        calls.append((model.prefill, tokens, capture is not None))
        model._qwen4_state.tensors["recurrent"].add_(sum(tokens))
        for index, token in enumerate(tokens):
            caches[0][:, 0, 0, past + index, 0] += token
        model.last_hidden_states = ids.to(torch.float32).unsqueeze(-1).expand(1, len(tokens), 4)
        value = _logits(tokens)
        return value if return_all else value[:, -1]

    monkeypatch.setattr(generation, "_page_step", target)

    class Draft:
        def reset(self):
            self.inputs = []
            self.captures = []
            self.past_length = 0

        def step(self, embeddings, hidden, *, position_ids, capture=None):
            self.past_length += 1
            self.inputs.append((embeddings.clone(), hidden.clone(), position_ids.clone()))
            self.captures.append(capture)
            token = int(embeddings[0, 0, 0])
            prediction = token + 1 if candidate_matches else 11
            value = torch.full((1, 1, 12), -100.0)
            value[0, 0, prediction % 12] = 100
            return value, hidden

        def snapshot(self, *, write_length):
            return self.past_length

        def restore(self, saved):
            self.past_length = saved

    draft = Draft()
    capture = SimpleNamespace(components=("prefill", "decode", "mtp"))
    draft.reset()
    output = generation._generate_mtp(
        model,
        pages,
        torch.tensor([[0, 1]]),
        torch.ones(1, 1, 4),
        _logits([1])[:, -1],
        draft,
        5,
        set(),
        2,
        capture,
    )
    assert output == [2, 3, 4, 5, 6]
    assert float(model._qwen4_state.tensors["recurrent"]) == float(pages[0][0].sum())
    assert float(pages[0][0].sum()) == sum(output[:-1])
    assert any(not prefill and captured for prefill, _, captured in calls)
    assert all(item is capture for item in draft.captures)
    assert model.verify_calls > 0
    assert not any(prefill for prefill, _, _ in calls)


@pytest.mark.parametrize("past", [2, 3, 62, 63, 127])
def test_verify_rollback_restores_written_qsa_and_recurrent_state(past):
    from xhquant.core import CacheTensor

    kinds = {
        "key": "kv_key",
        "value": "kv_value",
        "pool": "qsa_indexer_key",
        "tail": "qsa_indexer_tail",
        "recurrent": "delta_recurrent",
        "key_tail": "qsa_key_tail",
        "value_tail": "qsa_value_tail",
    }
    tensors = {
        "key": CacheTensor(torch.randn(1, 2, 256, 8)),
        "value": CacheTensor(torch.randn(1, 2, 256, 8)),
        "pool": CacheTensor(torch.randn(1, 1, 64, 8)),
        "tail": torch.randn(1, 4, 8),
        "key_tail": torch.randn(1, 2, 3, 8),
        "value_tail": torch.randn(1, 2, 3, 8),
        "recurrent": CacheTensor(torch.randn(1, 2, 8, 8)),
    }
    state = SimpleNamespace(
        tensors=tensors,
        group_start_position=torch.randn(3, 1, 1),
        contract=SimpleNamespace(tensors=[SimpleNamespace(name=name, state_kind=kind) for name, kind in kinds.items()]),
    )
    pages = [torch.randn(2, 4, 2, 64, 8)]
    before = {name: value.detach().clone() for name, value in tensors.items()}
    position_before = state.group_start_position.clone()
    page_before = pages[0].clone()
    snapshot = generation._snapshot_mtp_state(state, pages, past, 2)
    saved = {name: value for name, _, _, value in snapshot[0]}
    assert saved["key"].shape == (1, 2, 2, 8)
    assert snapshot[2].shape == (3, 1, 1)
    assert saved["pool"].shape[2] == (past % 4 + 2) // 4
    assert snapshot[1][0][1].shape[1] == (past + 1) // 64 - past // 64 + 1
    state.group_start_position.zero_()
    for name in ("key", "value"):
        state.tensors[name][:, :, past : past + 2].fill_(0)
    pool_start = past // 4
    pool_count = (past % 4 + 2) // 4
    state.tensors["pool"][:, :, pool_start : pool_start + pool_count].fill_(0)
    state.tensors["tail"] = torch.zeros_like(state.tensors["tail"])
    state.tensors["key_tail"] = torch.zeros_like(state.tensors["key_tail"])
    state.tensors["value_tail"] = torch.zeros_like(state.tensors["value_tail"])
    state.tensors["recurrent"].zero_()
    first_page, last_page = past // 64, (past + 1) // 64
    pages[0][:, first_page : last_page + 1].zero_()
    generation._restore_mtp_state(state, snapshot)
    for name, expected in before.items():
        torch.testing.assert_close(state.tensors[name], expected, atol=0, rtol=0)
    torch.testing.assert_close(state.group_start_position, position_before, atol=0, rtol=0)
    torch.testing.assert_close(pages[0], page_before, atol=0, rtol=0)


@pytest.mark.parametrize("visual", [False, True])
@pytest.mark.parametrize("reject_index", [None, 0, 2])
def test_n4_streaming_warmup_recursive_draft_and_partial_rejection(monkeypatch, reject_index, visual):
    embedding = torch.nn.Embedding(32, 2)
    with torch.no_grad():
        embedding.weight[:, 0] = torch.arange(32)
        embedding.weight[:, 1] = 0
    events = []
    tensor = torch.zeros(1)
    model = SimpleNamespace(
        device="cpu",
        embed_tokens=embedding,
        prefill=True,
        verify_model=object(),
        _qwen4_verify_length=5,
        qwen4_cache_abi=generation.QWEN4_EXP_LLM_CACHE_ABI,
        meta_info=SimpleNamespace(
            spec_decode={"num_draft_tokens": 4},
            model_config=SimpleNamespace(prefill_chunk_length=3, context_max_length=32),
        ),
        _qwen4_state=SimpleNamespace(tensors={"state": tensor}, group_start_position=torch.zeros(3, 1, 1)),
        _data_processor=None,
    )
    model.reset_internal_state = lambda: tensor.zero_()
    model.set_prefill = lambda: setattr(model, "prefill", True)
    model.set_decode = lambda: setattr(model, "prefill", False)
    model.set_verify = lambda: setattr(model, "prefill", False)
    model.is_prefill = lambda: model.prefill
    monkeypatch.setattr(generation, "_allocate_page_caches", lambda model: [])

    def target(model, caches, ids, past, **kwargs):
        events.append(("target", past))
        tensor.add_(ids.sum())
        model.last_hidden_states = ids.float().unsqueeze(-1).expand(1, -1, 4).clone()
        value = torch.full((1, ids.shape[1], 32), -100.0)
        value.scatter_(2, ((ids + 1) % 32).unsqueeze(-1), 100.0)
        return value if kwargs.get("return_all") else value[:, -1]

    monkeypatch.setattr(generation, "_page_step", target)

    class Draft:
        def reset(self):
            self.past_length = 0
            self.state = torch.zeros(1)
            self.warmups = []
            self.recursive_pairs = []
            self.proposing = False
            self.replaying = False

        def prefill(self, embeds, hidden, *, position_ids, capture=None):
            events.append(("warmup", embeds.shape[1]))
            self.warmups.append((embeds.clone(), hidden.clone(), position_ids.clone()))
            self.past_length += embeds.shape[1]
            self.state.add_(embeds[:, :, 0].sum())

        def snapshot(self, *, write_length):
            saved = self.past_length, self.state.clone()
            self.proposing, self.replaying, self.index = True, False, 0
            return saved

        def restore(self, saved):
            self.past_length, value = saved
            self.state.copy_(value)
            self.replaying = True

        def step(self, embeds, hidden, *, position_ids, capture=None):
            token = int(embeds[0, 0, 0])
            next_token = token + 1
            if self.proposing and not self.replaying:
                if self.index and self.index < 4:
                    self.recursive_pairs.append((hidden.clone(), self.previous_hidden.clone()))
                if self.index == reject_index:
                    next_token = 31
                self.index += 1
            self.state.add_(token)
            self.past_length += 1
            output_hidden = hidden + 10
            self.previous_hidden = output_hidden.clone()
            value = torch.full((1, 1, 32), -100.0)
            value[0, 0, next_token % 32] = 100.0
            return value, output_hidden

    draft = Draft()
    ids = torch.arange(6).reshape(1, -1)
    positions = torch.arange(18).reshape(3, 1, 6)
    fused = embedding(ids) + (100 if visual else 0)
    output = generation.generate_token_ids(
        model, ids, max_new_tokens=11, inputs_embeds=fused, position_ids=positions, mtp_runtime=draft
    )
    assert output == list(range(6, 17))
    assert events[:4] == [("target", 0), ("warmup", 2), ("target", 3), ("warmup", 3)]
    warm_embeddings = torch.cat([item[0] for item in draft.warmups], 1)
    warm_hidden = torch.cat([item[1] for item in draft.warmups], 1)
    warm_positions = torch.cat([item[2] for item in draft.warmups], 2)
    torch.testing.assert_close(warm_embeddings, fused[:, 1:])
    torch.testing.assert_close(warm_hidden[:, :, 0], ids[:, :-1].float())
    torch.testing.assert_close(warm_positions, positions[:, :, 1:])
    assert draft.recursive_pairs
    for actual, expected in draft.recursive_pairs:
        torch.testing.assert_close(actual, expected)
    assert tensor.item() == sum(range(16))
    assert draft.past_length == 15  # one shifted pair behind target's 16 consumed inputs
    assert draft.state.item() == sum(range(1, 16)) + (500 if visual else 0)
    stats = model.last_mtp_stats
    assert stats["num_rounds"] >= 2 and stats["draft_tokens_total"] >= 4
    assert stats["accepted_drafts_total"] == sum(stats["accepted_drafts_per_round"])
    assert stats["accepted_drafts_per_round"][0] == (4 if reject_index is None else reject_index)
    assert stats["target_replay_tokens"] == 0 if reject_index is None else stats["target_replay_tokens"] > 0
