"""Bounded MTP transactions preserve real state without cloning long cache prefixes."""

from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import Qwen4ExpTensorSpec
from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_runtime import MTPGraphRuntime, verify_draft_transaction
from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_snapshot import snapshot_state


def _states(capacity):
    specs = {
        name: Qwen4ExpTensorSpec(name, 0, kind, shape, dtype)
        for name, kind, shape, dtype in (
            ("key_cache_0", "kv_key", (1, 2, "cache_length", 4), "floating"),
            ("value_cache_0", "kv_value", (1, 2, "cache_length", 4), "floating"),
            ("qsa_indexer_key_cache_0", "qsa_indexer_key", (1, 1, "cache_length", 4), "floating"),
            ("past_position_ids", "past_position_ids", (3, 1, "cache_length"), "integer"),
            ("delta_recurrent_state_0", "delta_recurrent", (1, 2, 4, 4), "floating"),
            ("ple_ngram_history_0", "ple_ngram_history", (1, 3), "integer"),
        )
    }
    states = {}
    for name, spec in specs.items():
        shape = tuple(capacity if axis == "cache_length" else axis for axis in spec.shape)
        states[name] = torch.arange(torch.Size(shape).numel()).reshape(shape).to(
            torch.int32 if spec.dtype == "integer" else torch.float32
        )
    return states, specs


def _write(states, specs, past, width, valid):
    for name, value in states.items():
        shape = specs[name].shape
        if "cache_length" in shape:
            value.narrow(shape.index("cache_length"), past, width).add_(1000)
        else:
            value.add_(valid)


@pytest.mark.parametrize("capacity", [16, 262144])
def test_fixed_verify4_backs_up_padding_and_restores_tensor_identity(capacity):
    states, specs = _states(capacity)
    original = dict(states)
    expected = {name: value.clone() for name, value in states.items()}
    saved = snapshot_state(states, (), specs=specs, past_length=5, write_length=4)
    # Snapshot memory depends on the four written positions, not the 256K prefix.
    assert saved.nbytes == 4 * (2 * 2 * 4 + 4 + 3) * 4 + (2 * 4 * 4 + 3) * 4
    _write(states, specs, 5, 4, 2)
    # Commit_outputs can replace non-inplace state objects after graph execution.
    states["ple_ngram_history_0"] = torch.full((1, 3), 99, dtype=torch.int32)
    saved.restore(states, ())
    for name, value in states.items():
        assert value is original[name]
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("failure", [False, True])
def test_partial_acceptance_and_exception_restore_exact_write_set(failure):
    states, specs = _states(16)
    expected = {name: value.clone() for name, value in states.items()}
    past = 5

    def snapshot():
        return snapshot_state(states, (), specs=specs, past_length=past, write_length=4)

    def verify(tokens):
        _write(states, specs, past, 4, len(tokens))
        if failure:
            raise RuntimeError("verify failed after fixed-width cache writes")
        logits = torch.full((1, len(tokens), 12), -100.)
        logits[0, 0, 2] = 1
        logits[0, 1:, 7] = 1
        return logits, torch.zeros(1, len(tokens), 4)

    def replay(tokens):
        assert tokens == (1, 2)
        for name in states:
            torch.testing.assert_close(states[name], expected[name], rtol=0, atol=0)
        _write(states, specs, past, len(tokens), len(tokens))
        return torch.zeros(1, len(tokens), 12), torch.ones(1, len(tokens), 4)

    options = dict(pending_token=1, draft_tokens=[2, 9], snapshot=snapshot,
                   restore=lambda saved: saved.restore(states, ()), verify=verify, replay=replay)
    if failure:
        with pytest.raises(RuntimeError, match="verify failed"):
            verify_draft_transaction(**options)
    else:
        result = verify_draft_transaction(**options)
        assert result.committed_input_tokens == (1, 2)
        assert result.accepted_draft_tokens == 1
        _write(expected, specs, past, 2, 2)
    for name in states:
        torch.testing.assert_close(states[name], expected[name], rtol=0, atol=0)


def test_identity_page_map_snapshot_crosses_page_boundary_without_copying_other_slots():
    pages = [torch.arange(2 * 4 * 2 * 64 * 4.).reshape(2, 4, 2, 64, 4)]
    original = pages[0].clone()
    saved = snapshot_state({}, pages, specs={}, past_length=63, write_length=4)
    assert saved.nbytes == 2 * 2 * 4 * 4 * 4
    for position in range(63, 67):
        pages[0][:, position // 64, :, position % 64].add_(1000)
    saved.restore({}, pages)
    torch.testing.assert_close(pages[0], original, rtol=0, atol=0)


def test_draft_keeps_full_snapshot_compatibility_and_supports_step_transactions():
    runtime = object.__new__(MTPGraphRuntime)
    runtime.states, specs = _states(16)
    # This test runtime's spec keys already coincide with contract input names.
    runtime.contract = SimpleNamespace(
        tensors=tuple(specs.values()),
        layers=(SimpleNamespace(layer_idx=0, tensors=tuple(specs.values())),),
    )
    runtime.page_caches = []
    runtime.past_length = 5
    runtime.context_length = 16
    runtime.group_start_position = torch.zeros(3, 1, 1, dtype=torch.int32)
    original = {name: value.clone() for name, value in runtime.states.items()}
    full = runtime.snapshot()
    runtime.reset()
    runtime.restore(full)
    for name in original:
        torch.testing.assert_close(runtime.states[name], original[name], rtol=0, atol=0)
    saved = runtime.snapshot(write_length=1)
    _write(runtime.states, specs, 5, 1, 1)
    runtime.past_length += 1
    runtime.restore(saved)
    assert runtime.past_length == 5
    for name in original:
        torch.testing.assert_close(runtime.states[name], original[name], rtol=0, atol=0)


def test_restore_rejects_changed_layout_before_mutating_any_state():
    states, specs = _states(16)
    saved = snapshot_state(states, (), specs=specs, past_length=5, write_length=4)
    _write(states, specs, 5, 4, 2)
    changed = states["key_cache_0"].clone()
    states["ple_ngram_history_0"] = torch.zeros(1, 4, dtype=torch.int32)
    with pytest.raises(ValueError, match="layout"):
        saved.restore(states, ())
    torch.testing.assert_close(states["key_cache_0"], changed, rtol=0, atol=0)


def test_snapshot_preserves_sdk_cache_tensor_identity():
    from xhquant.core.cache_tensor import CacheTensor

    states, specs = _states(16)
    states["key_cache_0"] = CacheTensor(states["key_cache_0"])
    original = states["key_cache_0"]
    expected = original.clone()
    saved = snapshot_state(states, (), specs=specs, past_length=5, write_length=4)
    original[:, :, 5:9].add_(1000)
    saved.restore(states, ())
    assert states["key_cache_0"] is original
    torch.testing.assert_close(original, expected, rtol=0, atol=0)


def test_unknown_page_geometry_keeps_conservative_snapshot():
    page = torch.arange(24.).reshape(2, 3, 4)
    expected = page.clone()
    saved = snapshot_state({}, [page], specs={}, past_length=0, write_length=2)
    assert saved.nbytes == page.numel() * page.element_size()
    page.add_(1000)
    saved.restore({}, [page])
    torch.testing.assert_close(page, expected, rtol=0, atol=0)


@pytest.mark.parametrize("accept", [False, True])
def test_generation_uses_verify_static_width_and_leaves_committed_prefix(monkeypatch, accept):
    from xhmodel_merak.xh_llm.models.qwen4_exp import generation

    states, specs = _states(16)
    for tensor in states.values():
        tensor.zero_()
    originals = dict(states)
    embedding = torch.nn.Embedding(12, 2)
    with torch.no_grad():
        embedding.weight[:, 0] = torch.arange(12)
    model = SimpleNamespace(
        device="cpu", embed_tokens=embedding,
        meta_info=SimpleNamespace(model_config=SimpleNamespace(prefill_chunk_length=4, context_max_length=16)),
        qwen4_cache_contract=SimpleNamespace(tensors=tuple(specs.values())),
        _qwen4_state=SimpleNamespace(tensors=states, group_start_position=torch.zeros(3, 1, 1, dtype=torch.int32)), _qwen4_verify_length=4,
        verify_model=object(), _data_processor=None,
    )
    model.set_decode = lambda: setattr(model, "verifying", False)
    model.set_verify = lambda: setattr(model, "verifying", True)
    sizes = []
    actual_snapshot = generation.snapshot_state if hasattr(generation, "snapshot_state") else snapshot_state

    # _generate_mtp imports this callable locally; record the actual bounded
    # transaction while retaining its real snapshot and restore behavior.
    import xhmodel_merak.xh_llm.models.qwen4_exp.mtp_snapshot as snapshots

    def record(*args, **kwargs):
        result = actual_snapshot(*args, **kwargs)
        sizes.append((kwargs["write_length"], result.nbytes))
        return result

    monkeypatch.setattr(snapshots, "snapshot_state", record)

    def target(model, pages, ids, past, **kwargs):
        tokens = ids[0].tolist()
        width = 4 if model.verifying else 1
        _write(states, specs, past, width, len(tokens))
        # Every consumed token advances these small recurrent states.
        states["delta_recurrent_state_0"].fill_(past + len(tokens))
        model.last_hidden_states = ids.float().unsqueeze(-1).expand(1, len(tokens), 4)
        logits = torch.full((1, len(tokens), 12), -100.)
        for index, token in enumerate(tokens):
            logits[0, index, (token + 1) % 12] = 1
        return logits if kwargs.get("return_all") else logits[:, -1]

    monkeypatch.setattr(generation, "_page_step", target)

    class Draft:
        def reset(self):
            pass

        def prefill(self, *args, **kwargs):
            pass

        def step(self, embeddings, hidden, **kwargs):
            token = int(embeddings[0, 0, 0])
            logits = torch.full((1, 1, 12), -100.)
            logits[0, 0, token + 1 if accept else 11] = 1
            return logits, hidden

    logits = torch.zeros(1, 12)
    logits[0, 2] = 1
    result = generation._generate_mtp(
        model, [], torch.tensor([[0, 1]]), [torch.zeros(1, 2, 4)], logits,
        Draft(), 4, set(), 2, None,
        prompt_position_ids=torch.arange(2).reshape(1, 1, 2).expand(3, 1, 2),
    )
    assert result == [2, 3, 4, 5]
    assert sizes and all(width == 4 for width, _ in sizes)
    assert states["delta_recurrent_state_0"].flatten()[0].item() == 5
    assert all(states[name] is original for name, original in originals.items())
    if not accept:
        # verify's padding writes must not survive rejection and replay.
        assert torch.count_nonzero(states["key_cache_0"][:, :, 5:]) == 0


def test_replay_failure_restores_same_snapshot_a_second_time():
    states, specs = _states(16)
    expected = {name: value.clone() for name, value in states.items()}
    snapshot = snapshot_state(states, (), specs=specs, past_length=5, write_length=4)
    restores = []

    def verify(tokens):
        _write(states, specs, 5, 4, 2)
        return torch.zeros(1, 2, 12), torch.zeros(1, 2, 4)

    def restore(saved):
        saved.restore(states, ())
        restores.append(True)

    def replay(tokens):
        _write(states, specs, 5, 1, 1)
        states["ple_ngram_history_0"] = torch.full((1, 3), 99, dtype=torch.int32)
        raise RuntimeError("replay failed after mutation")

    with pytest.raises(RuntimeError, match="replay failed"):
        verify_draft_transaction(pending_token=1, draft_tokens=[2], snapshot=lambda: snapshot,
                                 restore=restore, verify=verify, replay=replay)
    assert len(restores) == 2
    for name in states:
        torch.testing.assert_close(states[name], expected[name], rtol=0, atol=0)


def test_snapshot_rejects_replaced_page_buffer_before_any_restore():
    states, specs = _states(16)
    page = torch.ones(2, 2, 1, 64, 4)
    snapshot = snapshot_state(states, [page], specs=specs, past_length=5, write_length=4)
    states["delta_recurrent_state_0"].add_(10)
    changed = states["delta_recurrent_state_0"].clone()
    with pytest.raises(ValueError, match="PageKV buffer identity"):
        snapshot.restore(states, [page.clone()])
    torch.testing.assert_close(states["delta_recurrent_state_0"], changed, rtol=0, atol=0)


def test_snapshot_rejects_mutated_backing_dtype_before_restoring():
    states, specs = _states(16)
    snapshot = snapshot_state(states, (), specs=specs, past_length=5, write_length=4)
    states["ple_ngram_history_0"].data = states["ple_ngram_history_0"].data.to(torch.int64)
    with pytest.raises(ValueError, match="layout"):
        snapshot.restore(states, ())
