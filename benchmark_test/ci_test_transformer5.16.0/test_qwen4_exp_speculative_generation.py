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
@pytest.mark.parametrize("chunk", [1, 4])
@pytest.mark.parametrize("golden", [False, True])
def test_mtp_matches_greedy_preserves_image_warmup_and_cache(monkeypatch, candidate_matches, chunk, golden):
    states = {"recurrent": torch.zeros(1)}
    embedding = torch.nn.Embedding(12, 2)
    with torch.no_grad():
        embedding.weight[:, 0] = torch.arange(12)
        embedding.weight[:, 1] = -torch.arange(12)
    model = SimpleNamespace(
        device="cpu",
        embed_tokens=embedding,
        meta_info=SimpleNamespace(model_config=SimpleNamespace(prefill_chunk_length=chunk, context_max_length=32)),
        _qwen4_state=SimpleNamespace(
            tensors=states,
            group_start_position=torch.zeros(3, 1, 1, dtype=torch.int32),
            contract=SimpleNamespace(tensors=[
                SimpleNamespace(name="recurrent", state_kind="delta_recurrent"),
            ]),
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
        assert not model.prefill or len(tokens) <= chunk
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

        def step(self, embeddings, hidden, *, position_ids, capture=None):
            self.inputs.append((embeddings.clone(), hidden.clone(), position_ids.clone()))
            self.captures.append(capture)
            token = int(embeddings[0, 0, 0])
            prediction = token + 1 if candidate_matches else 11
            value = torch.full((1, 1, 12), -100.0)
            value[0, 0, prediction % 12] = 100
            return value, hidden

    draft = Draft()
    prefill_calls = []
    def draft_prefill(embeddings, hidden, *, position_ids, capture=None):
        prefill_calls.append((embeddings.clone(), hidden.clone(), position_ids.clone()))
        for index in range(embeddings.shape[1]):
            draft.step(embeddings[:, index:index+1], hidden[:, index:index+1],
                       position_ids=position_ids[:, :, index:index+1])

    draft.prefill = draft_prefill
    capture = SimpleNamespace(components=("prefill", "decode", "mtp")) if golden else None
    prompt_embeddings = torch.tensor([[[100.0, 101.0], [200.0, 201.0]]])
    output = generation._generate_mtp(
        model,
        pages,
        torch.tensor([[0, 1]]),
        [torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]])],
        _logits([1])[:, -1],
        draft,
        5,
        set(),
        2,
        capture,
        prompt_embeds=prompt_embeddings,
        prompt_position_ids=torch.arange(2).reshape(1, 1, 2).expand(3, 1, 2),
    )
    assert output == [2, 3, 4, 5, 6]
    torch.testing.assert_close(draft.inputs[0][0], prompt_embeddings[:, 1:2])
    assert float(model._qwen4_state.tensors["recurrent"]) == float(pages[0][0].sum())
    assert float(pages[0][0].sum()) == sum(output[:-1])
    if chunk == 1:
        assert not any(prefill for prefill, _, _ in calls)
    if golden:
        assert any(not prefill and captured for prefill, _, captured in calls)
        # Every post-prompt assistant decode, including advancing an accepted
        # candidate, must be captured. Prompt warmup is checked separately.
        assert all(item is capture for item in draft.captures[1:])
    assert model.verify_calls > 0
    assert not any(prefill for prefill, _, _ in calls)
    assert len(prefill_calls) == 1
    torch.testing.assert_close(prefill_calls[0][0], prompt_embeddings[:, 1:])
    torch.testing.assert_close(prefill_calls[0][1], torch.zeros(1, 1, 4))
    assert prefill_calls[0][2].flatten().tolist() == [1, 1, 1]


@pytest.mark.parametrize("past", [2, 3, 62, 63, 127])
def test_verify_rollback_restores_written_qsa_and_recurrent_state(past):
    from xhquant.core import CacheTensor

    kinds = {
        "key": "kv_key", "value": "kv_value",
        "pool": "qsa_indexer_key", "tail": "qsa_indexer_tail", "recurrent": "delta_recurrent",
        "key_tail": "qsa_key_tail", "value_tail": "qsa_value_tail",
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
    state = SimpleNamespace(tensors=tensors, group_start_position=torch.randn(3, 1, 1), contract=SimpleNamespace(tensors=[
        SimpleNamespace(name=name, state_kind=kind) for name, kind in kinds.items()
    ]))
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
        state.tensors[name][:, :, past:past + 2].fill_(0)
    pool_start = past // 4
    pool_count = (past % 4 + 2) // 4
    state.tensors["pool"][:, :, pool_start:pool_start + pool_count].fill_(0)
    state.tensors["tail"] = torch.zeros_like(state.tensors["tail"])
    state.tensors["key_tail"] = torch.zeros_like(state.tensors["key_tail"])
    state.tensors["value_tail"] = torch.zeros_like(state.tensors["value_tail"])
    state.tensors["recurrent"].zero_()
    first_page, last_page = past // 64, (past + 1) // 64
    pages[0][:, first_page:last_page + 1].zero_()
    generation._restore_mtp_state(state, snapshot)
    for name, expected in before.items():
        torch.testing.assert_close(state.tensors[name], expected, atol=0, rtol=0)
    torch.testing.assert_close(state.group_start_position, position_before, atol=0, rtol=0)
    torch.testing.assert_close(pages[0], page_before, atol=0, rtol=0)


def test_mtp_prompt_prefill_aligns_all_shifted_hc_and_multiaxis_positions():
    embedding = torch.nn.Embedding(12, 2)
    hidden = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    prompt_embeddings = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    positions = torch.arange(15).reshape(3, 1, 5)
    model = SimpleNamespace(embed_tokens=embedding)
    calls = []
    draft = SimpleNamespace(reset=lambda: None,
                            prefill=lambda *args, **kwargs: calls.append((args, kwargs)))
    result = generation._generate_mtp(
        model, [], torch.arange(5).unsqueeze(0), [hidden[:, :3], hidden[:, 3:]],
        _logits([4])[:, -1], draft, 1, set(), 15, None, prompt_embeds=prompt_embeddings,
        prompt_position_ids=positions,
    )
    assert result == [5]
    assert len(calls) == 1
    args, kwargs = calls[0]
    torch.testing.assert_close(args[0], prompt_embeddings[:, 1:])
    torch.testing.assert_close(args[1], hidden[:, :-1])
    torch.testing.assert_close(kwargs["position_ids"], positions[:, :, 1:])
