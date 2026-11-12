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
        states[name] = (
            torch.arange(torch.Size(shape).numel())
            .reshape(shape)
            .to(torch.int32 if spec.dtype == "integer" else torch.float32)
        )
    return states, specs


def _write(states, specs, past, width, valid):
    for name, value in states.items():
        shape = specs[name].shape
        if "cache_length" in shape:
            value.narrow(shape.index("cache_length"), past, width).add_(1000)
        else:
            value.add_(valid)


@pytest.mark.parametrize("capacity,width", [(16, 4), (262144, 5)])
def test_fixed_verify_width_backs_up_padding_and_restores_tensor_identity(capacity, width):
    states, specs = _states(capacity)
    original = dict(states)
    expected = {name: value.clone() for name, value in states.items()}
    saved = snapshot_state(states, (), specs=specs, past_length=5, write_length=width)
    # Snapshot memory depends on the written positions, not the 256K prefix.
    assert saved.nbytes == width * (2 * 2 * 4 + 4 + 3) * 4 + (2 * 4 * 4 + 3) * 4
    _write(states, specs, 5, width, 2)
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
        logits = torch.full((1, len(tokens), 12), -100.0)
        logits[0, 0, 2] = 1
        logits[0, 1:, 7] = 1
        return logits, torch.zeros(1, len(tokens), 4)

    def replay(tokens):
        assert tokens == (1, 2)
        for name in states:
            torch.testing.assert_close(states[name], expected[name], rtol=0, atol=0)
        _write(states, specs, past, len(tokens), len(tokens))
        return torch.zeros(1, len(tokens), 12), torch.ones(1, len(tokens), 4)

    options = dict(
        pending_token=1,
        draft_tokens=[2, 9],
        snapshot=snapshot,
        restore=lambda saved: saved.restore(states, ()),
        verify=verify,
        replay=replay,
    )
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
    pages = [torch.arange(2 * 4 * 2 * 64 * 4.0).reshape(2, 4, 2, 64, 4)]
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
    page = torch.arange(24.0).reshape(2, 3, 4)
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
        device="cpu",
        embed_tokens=embedding,
        meta_info=SimpleNamespace(model_config=SimpleNamespace(prefill_chunk_length=4, context_max_length=16)),
        qwen4_cache_contract=SimpleNamespace(tensors=tuple(specs.values())),
        _qwen4_state=SimpleNamespace(tensors=states, group_start_position=torch.zeros(3, 1, 1, dtype=torch.int32)),
        _qwen4_verify_length=4,
        verify_model=object(),
        _data_processor=None,
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
        logits = torch.full((1, len(tokens), 12), -100.0)
        for index, token in enumerate(tokens):
            logits[0, index, (token + 1) % 12] = 1
        return logits if kwargs.get("return_all") else logits[:, -1]

    monkeypatch.setattr(generation, "_page_step", target)

    class Draft:
        def reset(self):
            pass

        def prefill(self, *args, **kwargs):
            pass

        def snapshot(self, **kwargs):
            return None

        def restore(self, saved):
            pass

        def step(self, embeddings, hidden, **kwargs):
            token = int(embeddings[0, 0, 0])
            logits = torch.full((1, 1, 12), -100.0)
            logits[0, 0, token + 1 if accept else 11] = 1
            return logits, hidden

    logits = torch.zeros(1, 12)
    logits[0, 2] = 1
    result = generation._generate_mtp(
        model,
        [],
        torch.tensor([[0, 1]]),
        torch.zeros(1, 1, 4),
        logits,
        Draft(),
        4,
        set(),
        2,
        None,
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
        verify_draft_transaction(
            pending_token=1, draft_tokens=[2], snapshot=lambda: snapshot, restore=restore, verify=verify, replay=replay
        )
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


def test_full_accept_tail_observer_failure_restores_draft_k_plus_one_across_page_and_qsa(monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen4_exp import generation
    from xhquant.core import CacheTensor

    capacity = 128
    specs = {
        "key": Qwen4ExpTensorSpec("key", 0, "kv_key", (1, 1, "cache_length", 2), "floating"),
        "pool": Qwen4ExpTensorSpec("pool", 0, "qsa_indexer_key", (1, 1, "qsa_pool_length", 2), "floating"),
        "tail": Qwen4ExpTensorSpec("tail", 0, "qsa_indexer_tail", (1, 3, 2), "floating"),
        "recurrent": Qwen4ExpTensorSpec("recurrent", 0, "delta_recurrent", (1, 1, 2, 2), "floating"),
    }

    def allocation():
        states = {
            "key": CacheTensor(torch.randn(1, 1, capacity, 2)),
            "pool": CacheTensor(torch.randn(1, 1, capacity // 4, 2)),
            "tail": torch.randn(1, 3, 2),
            "recurrent": CacheTensor(torch.randn(1, 1, 2, 2)),
        }
        return states, [torch.randn(2, 2, 1, 64, 2)], torch.randn(3, 1, 1)

    def write(states, pages, group, position, token):
        states["key"][:, :, position : position + 1].add_(token)
        states["pool"][:, :, position // 4 : position // 4 + 1].add_(token)
        states["tail"].add_(token)
        states["recurrent"].add_(token)
        pages[0][:, position // 64, :, position % 64].add_(token)
        group.add_(token)

    target_states, target_pages, target_group = allocation()
    draft_states, draft_pages, draft_group = allocation()
    originals = [dict(target_states), dict(draft_states)]
    expected = [{name: value.clone() for name, value in states.items()} for states in (target_states, draft_states)]
    expected_pages = [target_pages[0].clone(), draft_pages[0].clone()]
    expected_groups = [target_group.clone(), draft_group.clone()]
    model = SimpleNamespace(
        device="cpu",
        embed_tokens=torch.nn.Embedding(32, 2),
        verify_model=object(),
        _qwen4_verify_length=5,
        meta_info=SimpleNamespace(
            spec_decode={"runtime_contract_version": 4, "num_draft_tokens": 4},
            model_config=SimpleNamespace(context_max_length=capacity, prefill_chunk_length=4),
        ),
        qwen4_cache_contract=SimpleNamespace(tensors=tuple(specs.values())),
        _qwen4_state=SimpleNamespace(tensors=target_states, group_start_position=target_group),
        _data_processor=None,
    )
    with torch.no_grad():
        model.embed_tokens.weight[:, 0] = torch.arange(32)
    model.set_decode = lambda: None
    model.set_verify = lambda: None

    def target(model, pages, ids, past, **kwargs):
        for index, token in enumerate(ids[0].tolist()):
            write(target_states, pages, target_group, past + index, token)
        model.last_hidden_states = ids.float().unsqueeze(-1).expand(1, -1, 4).clone()
        logits = torch.full((1, ids.shape[1], 32), -100.0)
        logits.scatter_(2, ((ids + 1) % 32).unsqueeze(-1), 100.0)
        return logits

    monkeypatch.setattr(generation, "_page_step", target)

    class Draft:
        past_length = 60  # K=4 fills positions 60..63; full-accept tail writes 64.
        calls = 0

        def snapshot(self, *, write_length):
            self.saved_width = write_length
            return (
                snapshot_state(
                    draft_states, draft_pages, specs=specs, past_length=self.past_length, write_length=write_length
                ),
                self.past_length,
                draft_group.clone(),
            )

        def restore(self, saved):
            snapshot, self.past_length, group = saved
            snapshot.restore(draft_states, draft_pages)
            draft_group.copy_(group)

        def step(self, embeddings, hidden, **kwargs):
            token = int(embeddings[0, 0, 0])
            write(draft_states, draft_pages, draft_group, self.past_length, token)
            self.past_length += 1
            self.calls += 1
            if self.calls == 5:
                # Simulate an observer/capture failure after a successful tail
                # execution; this is outside a session's own atomic _run scope.
                raise RuntimeError("observer failed after draft tail write")
            logits = torch.full((1, 1, 32), -100.0)
            logits[0, 0, token + 1] = 100.0
            return logits, hidden + 1

    draft = Draft()
    logits = torch.full((1, 32), -100.0)
    logits[0, 7] = 100.0
    with pytest.raises(RuntimeError, match="observer failed"):
        generation._generate_mtp(
            model,
            target_pages,
            torch.zeros(1, 61, dtype=torch.long),
            torch.zeros(1, 1, 4),
            logits,
            draft,
            8,
            set(),
            61,
            None,
        )
    assert draft.calls == 5 and draft.past_length == 60
    for index, (states, pages, group) in enumerate(
        ((target_states, target_pages, target_group), (draft_states, draft_pages, draft_group))
    ):
        for name, value in states.items():
            assert value is originals[index][name]
            torch.testing.assert_close(value, expected[index][name], rtol=0, atol=0)
        torch.testing.assert_close(pages[0], expected_pages[index], rtol=0, atol=0)
        torch.testing.assert_close(group, expected_groups[index], rtol=0, atol=0)
    assert draft.saved_width == 5


def test_failed_draft_call_restores_qsa_tail_positions_and_length(monkeypatch):
    from transformers import Qwen4ExpTextConfig

    from xhmodel_merak.xh_llm.models.qwen4_exp import qsa_host_controls
    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import Qwen4ExpCacheContract

    config = Qwen4ExpTextConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=4,
        hc_count=2,
        hc_lowrank=4,
        layer_types=["full_attention"],
        ple_layer_ids=[],
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5},
    )
    runtime = object.__new__(MTPGraphRuntime)
    runtime.contract = Qwen4ExpCacheContract.from_text_config(config)
    runtime.context_length, runtime.past_length = 8, 3
    runtime.device = torch.device("cpu")
    runtime.page_modules, runtime.page_caches = [], []
    runtime.states = {"qsa_key_tail_0": torch.arange(24).reshape(1, 1, 3, 8).float()}
    runtime.group_start_position = torch.tensor([[[1]], [[2]], [[3]]], dtype=torch.int32)
    original = runtime.states["qsa_key_tail_0"].clone()
    group = runtime.group_start_position.clone()

    class FailingSession:
        def get_input(self, name):
            return SimpleNamespace(dtype=torch.float32)

        def run(self, feed):
            feed["qsa_key_tail_0"].fill_(42)
            raise RuntimeError("failed after writing draft state")

    runtime.session = FailingSession()
    # This fixture isolates failed session execution from QSA graph discovery.
    monkeypatch.setattr(qsa_host_controls, "qsa_host_control_spec", lambda session: None)
    with pytest.raises(RuntimeError, match="failed after writing"):
        runtime.step(torch.ones(1, 1, 16), torch.ones(1, 1, 32), position_ids=torch.tensor([[[7]], [[9]], [[5]]]))
    torch.testing.assert_close(runtime.states["qsa_key_tail_0"], original, rtol=0, atol=0)
    torch.testing.assert_close(runtime.group_start_position, group, rtol=0, atol=0)
    assert runtime.past_length == 3
