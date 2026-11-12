"""Real conditional-network N4 MTP, bounded rollback and persistent sessions.

The proposal-controlled cases exercise actual SDK updates but do not claim
natural draft acceptance. The production case leaves proposals untouched and
requires a real rejection before comparing every target state to greedy SDK.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import torch
from test_qwen4_exp_multimodal_e2e import (
    _assert_numerical,
    _export_package,
    _fused_and_positions,
    _load_package,
    _native_assistant_step,
    _native_inputs,
    _prefill,
    _processor_inputs,
)
from test_qwen4_exp_multimodal_e2e import (
    _bounded_cpu_threads as _bounded_cpu_threads,
)


@pytest.fixture(scope="module")
def alignment_package(tmp_path_factory):
    return _export_package(tmp_path_factory.mktemp("qwen4_mtp_n4"), mtp=True)


def _state(model):
    return {
        **{name: value.detach().clone() for name, value in model._qwen4_state.tensors.items()},
        "group_start_position": model._qwen4_state.group_start_position.detach().clone(),
    }


def _assert_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        tolerance = 0 if not actual[name].is_floating_point() or name == "group_start_position" else 5e-3
        torch.testing.assert_close(actual[name], expected[name], rtol=tolerance, atol=tolerance, msg=name)


def _decode(model, caches, tokens, *, past, next_position):
    from xhmodel_merak.xh_llm.models.qwen4_exp.generation import _page_step

    model.set_decode()
    values, hidden = [], []
    for index, token in enumerate(tokens):
        logits = _page_step(
            model,
            caches,
            torch.tensor([[token]]),
            past + index,
            position_ids=torch.full((3, 1, 1), next_position + index),
        )
        values.append(logits.unsqueeze(1))
        hidden.append(model.last_hidden_states[:, :1].clone())
    return torch.cat(values, 1), torch.cat(hidden, 1)


@torch.inference_mode()
def test_default_n4_metadata_and_real_native_draft_feedback(alignment_package):
    import onnx
    from onnx import numpy_helper
    from transformers import DynamicCache

    from xhmodel_merak.xh_llm.models.qwen4_exp.generation import create_mtp_runtime

    root, source = alignment_package
    meta = json.loads((root / "golden_meta_info.json").read_text())
    spec = meta["spec_decode"]
    assert spec["num_draft_tokens"] == 4
    assert spec["verify_length"] == 5
    assert spec["target_verify_graph"] == "decode_m5"
    assert spec["runtime_contract_version"] == 4
    assert spec["shared_head"]["owner"] == "target"
    assert spec["shared_head"]["logical_bits"] == 8
    paths = {
        meta["prefill_hmonnx"],
        meta["decode_hmonnx"],
        spec["target"]["verify_hmonnx"],
        spec["draft"]["prefill_hmonnx"],
        spec["draft"]["decode_hmonnx"],
    }
    assert len(paths) == 5 and all((root / path).is_file() for path in paths)
    heads = []
    for relative in sorted(paths):
        path = root / relative
        graph = onnx.load(path, load_external_data=False).graph
        (head,) = [
            node
            for node in graph.node
            if node.op_type == "Linear" and any("lm_head.qweight" in name for name in node.input)
        ]
        tensors = {value.name: value for value in graph.initializer}
        heads.append(
            tuple(
                (
                    tensors[name].data_type,
                    tuple(tensors[name].dims),
                    numpy_helper.to_array(tensors[name], base_dir=str(path.parent)).tobytes(),
                )
                for name in head.input[1:]
            )
        )
    assert all(head == heads[0] for head in heads)
    assert not source.path.exists()
    model = _load_package(root)
    try:
        assert model.get_input_sequence_length() == 4
        model.set_decode()
        assert model.get_input_sequence_length() == 1
        inputs = _processor_inputs(model.get_processor())
        fused, positions = _fused_and_positions(model, source, inputs)
        logits, _, hidden = _prefill(model, inputs["input_ids"], fused, positions)
        captured = []
        hook = source.native.model.language_model.hyper_connection_mixer.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone())
        )
        try:
            native = source.native(**_native_inputs(inputs), use_cache=True)
        finally:
            hook.remove()
        _assert_numerical(logits, native.logits[:, -1], cosine=0.98, atol=1e-2)
        _assert_numerical(hidden, captured[0], cosine=0.98, atol=1e-2)
        draft = create_mtp_runtime(model)
        try:
            cache = DynamicCache(config=source.assistant.config)
            actual = draft.prefill(fused[:, 1:], hidden[:, :-1], position_ids=positions[:, :, 1:])
            expected = _native_assistant_step(source, fused[:, 1:], hidden[:, :-1], positions[:, :, 1:], cache)
            for a, b in zip(actual, expected, strict=True):
                _assert_numerical(a[:, -1:], b[:, -1:], cosine=0.98, atol=1e-2)
            token = logits.argmax(-1).reshape(1, 1)
            feedback = hidden[:, -1:]
            start = int(positions.max()) + 1
            for offset in range(4):
                embedding = model.embed_tokens(token)
                position = torch.full((3, 1, 1), start + offset)
                actual = draft.step(embedding, feedback, position_ids=position)
                expected = _native_assistant_step(source, embedding, feedback, position, cache)
                for a, b in zip(actual, expected, strict=True):
                    _assert_numerical(a, b, cosine=0.98, atol=1e-2)
                token, feedback = actual[0].argmax(-1), actual[1]
            assert draft.past_length == inputs["input_ids"].shape[1] - 1 + 4
        finally:
            draft.close()
    finally:
        model.close()


@torch.inference_mode()
def test_legacy_v3_single_candidate_with_five_position_verify_uses_real_sdk(alignment_package, tmp_path):
    """Exercise v3/N1 with its supported padded verify5 geometry, reusing real graphs."""
    root, source = alignment_package
    legacy = tmp_path / "legacy_v3_n1_verify5"
    shutil.copytree(root, legacy)
    path = legacy / "golden_meta_info.json"
    metadata = json.loads(path.read_text())
    spec = metadata["spec_decode"]
    assert spec["verify_length"] == 5
    spec.update(runtime_contract_version=3, num_draft_tokens=1)
    spec.pop("shared_head")
    spec.pop("draft_head_weight_bits")
    path.write_text(json.dumps(metadata))
    model, oracle = _load_package(legacy), _load_package(root)
    try:
        inputs = _processor_inputs(model.get_processor())
        ids = inputs["input_ids"]
        fused, positions = _fused_and_positions(oracle, source, inputs)
        logits, caches, _ = _prefill(oracle, ids, fused, positions)
        tokens = []
        for offset in range(4):
            token = int(logits.argmax(-1).item())
            tokens.append(token)
            if offset < 3:
                logits, _ = _decode(
                    oracle,
                    caches,
                    [token],
                    past=ids.shape[1] + offset,
                    next_position=int(positions.max()) + 1 + offset,
                )
        actual = model.generate(**inputs, max_new_tokens=4, do_sample=False, eos_token_id=[])
        torch.testing.assert_close(actual, torch.cat((ids, ids.new_tensor([tokens])), 1), rtol=0, atol=0)
        assert model._qwen4_verify_length == 5
        assert model.last_mtp_stats["draft_capacity_per_round"] == 1
        assert model.last_mtp_stats["num_rounds"] > 0
    finally:
        model.close()
        oracle.close()


@pytest.mark.parametrize("accepted", [0, 2, 4])
@torch.inference_mode()
def test_real_sdk_controlled_acceptance_restores_bounded_target_state(alignment_package, accepted):
    root, source = alignment_package
    model = _load_package(root)
    try:
        inputs = _processor_inputs(model.get_processor())
        fused, positions = _fused_and_positions(model, source, inputs)
        _assert_controlled_production_round(model, root, source, inputs, fused, positions, accepted)
    finally:
        model.close()


def _draft_state(draft):
    return {
        **{name: value.clone() for name, value in draft.states.items()},
        **{f"page_{index}": value.clone() for index, value in enumerate(draft.page_caches)},
        "group_start_position": draft.group_start_position.clone(),
        "past_length": torch.tensor(draft.past_length),
    }


def _assert_controlled_production_round(model, root, source, inputs, fused, positions, accepted):
    """Control proposal selection after real SDK execution, preserving all HC/cache updates."""
    from xhmodel_merak.xh_llm.models.qwen4_exp import mtp_runtime
    from xhmodel_merak.xh_llm.models.qwen4_exp.generation import create_mtp_runtime
    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_snapshot import StateSnapshot

    oracle = _load_package(root)
    try:
        ids = inputs["input_ids"]
        past, position = ids.shape[1], int(positions.max()) + 1
        logits, caches, prompt_hidden = _prefill(oracle, ids, fused, positions)
        greedy = []
        for index in range(6):
            token = int(logits.argmax(-1).item())
            greedy.append(token)
            if index < 5:
                logits, _ = _decode(oracle, caches, [token], past=past + index, next_position=position + index)
        proposals = greedy[1:5]
        if accepted < 4:
            proposals[accepted] = (proposals[accepted] + 1) % source.native.config.text_config.vocab_size
        _prefill(oracle, ids, fused, positions)
        verify_logits, _ = _decode(oracle, caches, [greedy[0], *proposals], past=past, next_position=position)
        _prefill(oracle, ids, fused, positions)
        committed = greedy[: accepted + 1]
        _, target_hidden = _decode(oracle, caches, committed, past=past, next_position=position)
        expected_target = _state(oracle)
        reference_draft = create_mtp_runtime(oracle)
        reference_draft.prefill(fused[:, 1:], prompt_hidden[:, :-1], position_ids=positions[:, :, 1:])
        feedback = prompt_hidden[:, -1:]
        for index, token in enumerate(committed):
            if accepted == 4:
                # Full acceptance retains the recursive proposal HC; only
                # the final catch-up pair uses the target's previous HC.
                paired_hidden = feedback if index < 4 else target_hidden[:, index - 1 : index]
            else:
                # A rejection restores and replays verified target HC pairs.
                paired_hidden = prompt_hidden[:, -1:] if index == 0 else target_hidden[:, index - 1 : index]
            _, feedback = reference_draft.step(
                oracle.embed_tokens(torch.tensor([[token]])),
                paired_hidden,
                position_ids=torch.full((3, 1, 1), position + index),
            )
        expected_draft = _draft_state(reference_draft)
        draft = create_mtp_runtime(model)
        step, run, transaction = draft.step, draft._run, mtp_runtime.verify_draft_transaction
        selection, first_result, target_states, draft_states = [], [], [], []
        calls, snapshots = [], []

        def controlled_step(embedding, hidden, **options):
            values, actual_hidden = step(embedding, hidden, **options)
            if len(selection) < 4:
                token = proposals[len(selection)]
                selection.append(token)
                # Only proposal argmax is injected. The graph has already
                # executed and its real hidden output/cache updates remain.
                values = torch.full_like(values, -10000)
                values[..., token] = 10000
            return values, actual_hidden

        def observed_run(embedding, hidden, **options):
            values = run(embedding, hidden, **options)
            if first_result and not draft_states and draft.past_length == past + accepted:
                draft_states.append(_draft_state(draft))
            return values

        def observed_transaction(**options):
            if first_result:
                return transaction(**options)
            take, verify, replay = options["snapshot"], options["verify"], options["replay"]

            def snapshot():
                saved = take()
                assert isinstance(saved[0], StateSnapshot)
                snapshots.append(saved[0])
                return saved

            def checked_verify(tokens):
                calls.append("verify")
                assert tuple(tokens) == (greedy[0], *proposals)
                values = verify(tokens)
                _assert_numerical(values[0], verify_logits, cosine=0.99, atol=5e-3)
                return values

            def checked_replay(tokens):
                calls.append("replay")
                return replay(tokens)

            result = transaction(
                **{**options, "snapshot": snapshot, "verify": checked_verify, "replay": checked_replay}
            )
            first_result.append(result)
            target_states.append(_state(model))
            return result

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(draft, "step", controlled_step)
            patch.setattr(draft, "_run", observed_run)
            patch.setattr(mtp_runtime, "verify_draft_transaction", observed_transaction)
            generated = model.generate(**inputs, max_new_tokens=6, do_sample=False, eos_token_id=[])
        assert selection == proposals
        assert first_result[0].accepted_draft_tokens == accepted
        assert first_result[0].committed_input_tokens == tuple(committed)
        assert first_result[0].next_token == int(verify_logits[:, accepted].argmax(-1).item())
        assert calls == (["verify"] if accepted == 4 else ["verify", "replay"])
        assert len(snapshots) == 1
        for spec in model.qwen4_cache_contract.tensors:
            if "cache_length" in spec.shape:
                piece = snapshots[0].states[spec.name]
                assert piece.value.shape[spec.shape.index("cache_length")] == 5
                assert piece.value.numel() < piece.original.numel()
        assert model.last_mtp_stats["accepted_drafts_per_round"][0] == accepted
        torch.testing.assert_close(generated, torch.cat((ids, ids.new_tensor([greedy])), 1), rtol=0, atol=0)
        assert len(target_states) == len(draft_states) == 1
        _assert_state(target_states[0], expected_target)
        _assert_state(draft_states[0], expected_draft)
        _assert_numerical(first_result[0].target_hidden, target_hidden[:, -1:], cosine=0.99, atol=5e-3)
    finally:
        oracle.close()


def _assert_mtp_capture(capture, *, context_length):
    from xhmodel_merak.xh_llm.models.qwen4_exp.qsa_host_controls import QSA_HOST_CONTROL_NAMES, build_qsa_host_controls
    from xhmodel_merak.xh_llm.mrope import (
        MROPE_CURRENT_POSITION_NAMES,
        MROPE_POOL_POSITION_NAMES,
        prepare_mrope_position_inputs,
    )

    capture.finish()
    manifest = json.loads(capture.manifest_path.read_text())
    assert manifest["status"] == "complete"
    steps = [step for step in manifest["steps"] if step["component"] == "mtp"]
    assert {step["graph"] for step in steps} == {"mtp_prefill", "mtp_decode", "decode_m5"}
    for step in manifest["steps"]:
        names = {entry["name"] for entry in step["inputs"]}
        assert {*MROPE_CURRENT_POSITION_NAMES, *MROPE_POOL_POSITION_NAMES} <= names
        for key in ("runtime_before", "runtime_after"):
            assert "group_start_position" in {entry["name"] for entry in step[key]}
        directory = capture.root / step["directory"]
        arrays = {
            entry["name"]: torch.from_numpy(np.load(directory / entry["file"], allow_pickle=False))
            for entry in step["inputs"]
        }
        before, after = (
            {
                entry["name"]: torch.from_numpy(np.load(directory / entry["file"], allow_pickle=False))
                for entry in step[key]
            }
            for key in ("runtime_before", "runtime_after")
        )
        positions = torch.stack([arrays[name] for name in MROPE_CURRENT_POSITION_NAMES])
        width, valid = positions.shape[-1], int(arrays["current_input_length"].item())
        _, pool, expected_group = prepare_mrope_position_inputs(
            before["group_start_position"],
            positions[..., :valid],
            int(arrays["past_seq_length"].item()),
            width,
            valid,
        )
        for axis, name in enumerate(MROPE_POOL_POSITION_NAMES):
            torch.testing.assert_close(arrays[name], pool[axis], rtol=0, atol=0, check_dtype=False)
        torch.testing.assert_close(after["group_start_position"], expected_group, rtol=0, atol=0)
        if names.intersection(QSA_HOST_CONTROL_NAMES):
            assert set(QSA_HOST_CONTROL_NAMES) <= names
            mask = arrays["qsa_attention_mask"]
            topk = (mask.shape[-1] - 3) // 4
            controls = build_qsa_host_controls(
                int(arrays["past_seq_length"].item()),
                valid,
                width,
                context_length,
                topk,
                device="cpu",
                dtype=mask.dtype,
            )
            for name, expected in zip(QSA_HOST_CONTROL_NAMES, controls, strict=True):
                torch.testing.assert_close(arrays[name], expected, rtol=0, atol=0)
        for entry in (*step["inputs"], *step["outputs"], *step["runtime_before"], *step["runtime_after"]):
            value = np.load(directory / entry["file"], allow_pickle=False)
            assert list(value.shape) == entry["shape"] and str(value.dtype) == entry["dtype"]
        boundary = {entry["file"] for entry in (*step["inputs"], *step["outputs"])}
        assert any(
            path.name not in boundary and not path.name.startswith("runtime_") for path in directory.glob("*.npy")
        )


@torch.inference_mode()
def test_production_n4_natural_rejection_greedy_states_sessions_and_golden(alignment_package, monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.qwen4_exp import generation, mtp_runtime
    from xhmodel_merak.xh_llm.models.qwen4_exp.golden_runtime import GoldenCapture
    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_snapshot import StateSnapshot

    root, source = alignment_package
    model, oracle = _load_package(root), _load_package(root)
    events, invocations, snapshots, proposal_counts = [], [], [], []
    original_step, transaction = generation._page_step, mtp_runtime.verify_draft_transaction

    def observed_step(current, caches, ids, past, **kwargs):
        value = original_step(current, caches, ids, past, **kwargs)
        if current is model:
            invocations.append((past, ids.detach().clone(), kwargs.get("position_ids")))
        return value

    def observed_transaction(**kwargs):
        first = len(invocations)
        take = kwargs["snapshot"]

        def observed_snapshot():
            saved = take()
            assert isinstance(saved[0], StateSnapshot)
            snapshots.append(saved[0])
            return saved

        result = transaction(**{**kwargs, "snapshot": observed_snapshot})
        assert len(invocations) > first
        past = invocations[first][0]
        events.append((past, result, _state(model)))
        proposal_counts.append(len(kwargs["draft_tokens"]))
        return result

    monkeypatch.setattr(generation, "_page_step", observed_step)
    monkeypatch.setattr(mtp_runtime, "verify_draft_transaction", observed_transaction)
    try:
        inputs = _processor_inputs(model.get_processor())
        fused, positions = _fused_and_positions(model, source, inputs)
        ids = inputs["input_ids"]
        _, _, prompt_hidden = _prefill(oracle, ids, fused, positions)
        warmup = []
        run = mtp_runtime.MTPGraphRuntime._run

        def observed_draft_run(draft, embedding, hidden, *, position_ids, **options):
            past = draft.past_length
            if past < ids.shape[1] - 1:
                warmup.append((past, embedding.clone(), hidden.clone(), position_ids.clone(), len(invocations)))
            return run(draft, embedding, hidden, position_ids=position_ids, **options)

        capture = GoldenCapture(tmp_path / "mtp_golden", ["prefill", "decode", "mtp"])
        kwargs = dict(inputs, max_new_tokens=10, do_sample=False, eos_token_id=[])
        with monkeypatch.context() as patch:
            patch.setattr(mtp_runtime.MTPGraphRuntime, "_run", observed_draft_run)
            actual = model.generate(**kwargs, _capture=capture)
        assert sum(embedding.shape[1] for _, embedding, _, _, _ in warmup) == ids.shape[1] - 1
        assert warmup[0][-1] == 1  # Assistant starts before the second target prompt block.
        for start, embedding, hidden, actual_positions, _ in warmup:
            width = embedding.shape[1]
            torch.testing.assert_close(embedding, fused[:, start + 1 : start + 1 + width], rtol=0, atol=0)
            _assert_numerical(hidden, prompt_hidden[:, start : start + width], cosine=0.99, atol=5e-3)
            torch.testing.assert_close(actual_positions, positions[:, :, start + 1 : start + 1 + width], rtol=0, atol=0)
        stats = dict(model.last_mtp_stats)
        assert stats["num_rounds"] >= 1
        assert stats["draft_tokens_total"] >= 4
        assert stats["rejected_drafts_total"] > 0
        assert stats["target_verify_calls"] == stats["num_rounds"] == len(events)
        assert stats["accepted_drafts_total"] == sum(result.accepted_draft_tokens for _, result, _ in events)
        assert stats["draft_tokens_total"] == sum(proposal_counts)
        assert stats["rejected_drafts_total"] == stats["draft_tokens_total"] - stats["accepted_drafts_total"]
        assert stats["accepted_drafts_per_round"] == [result.accepted_draft_tokens for _, result, _ in events]
        assert stats["avg_accepted_per_round"] == pytest.approx(stats["accepted_drafts_total"] / stats["num_rounds"])
        assert stats["acceptance_rate"] == pytest.approx(stats["accepted_drafts_total"] / stats["draft_tokens_total"])
        assert stats["output_tokens"] == 10 and stats["draft_capacity_per_round"] == 4
        assert max(proposal_counts) == 4
        assert stats["target_replay_tokens"] > 0
        assert snapshots and all(saved.nbytes > 0 for saved in snapshots)
        assert any(
            piece.value.numel() < piece.original.numel() for saved in snapshots for piece in saved.states.values()
        )
        expected_logits, caches, _ = _prefill(oracle, ids, fused, positions)
        generated, event_map = (
            [],
            {past + len(result.committed_input_tokens): (result, state) for past, result, state in events},
        )
        for offset in range(10):
            token = int(expected_logits.argmax(-1).item())
            generated.append(token)
            if offset == 9:
                break
            expected_logits, hidden = _decode(
                oracle,
                caches,
                [token],
                past=ids.shape[1] + offset,
                next_position=int(positions.max()) + 1 + offset,
            )
            current = ids.shape[1] + offset + 1
            if current in event_map:
                result, state = event_map.pop(current)
                _assert_state(state, _state(oracle))
                _assert_numerical(result.target_hidden, hidden, cosine=0.99, atol=5e-3)
        assert not event_map
        torch.testing.assert_close(actual, torch.cat((ids, ids.new_tensor([generated])), 1), rtol=0, atol=0)
        _assert_state(_state(model), _state(oracle))
        for past, tokens, actual_positions in invocations:
            if past >= ids.shape[1]:
                expected_positions = (
                    torch.arange(
                        int(positions.max()) + 1 + past - ids.shape[1],
                        int(positions.max()) + 1 + past - ids.shape[1] + tokens.shape[1],
                    )
                    .view(1, 1, -1)
                    .expand(3, 1, -1)
                )
                torch.testing.assert_close(actual_positions, expected_positions, rtol=0, atol=0)
        _assert_mtp_capture(capture, context_length=int(model.meta_info.model_config.context_max_length))
        runtime = model._qwen4_mtp_runtime
        sessions = runtime.session, runtime.prefill_session
        resets, reset = [], runtime.reset

        def observed_reset():
            reset()
            assert runtime.past_length == 0 and not runtime.group_start_position.count_nonzero()
            assert all(not value.count_nonzero() for value in (*runtime.states.values(), *runtime.page_caches))
            resets.append(True)

        monkeypatch.setattr(runtime, "reset", observed_reset)
        torch.testing.assert_close(model.generate(**kwargs), actual, rtol=0, atol=0)
        assert model._qwen4_mtp_runtime is runtime
        assert (runtime.session, runtime.prefill_session) == sessions
        with pytest.raises(ValueError, match="grid|patch|image|placeholder|feature"):
            model.generate(**{**kwargs, "pixel_values": inputs["pixel_values"][:-1]})
        torch.testing.assert_close(model.generate(**kwargs), actual, rtol=0, atol=0)
        assert model._qwen4_mtp_runtime is runtime and (runtime.session, runtime.prefill_session) == sessions
        before_failure = []

        def fail_after_real_verify(current, caches, tokens, past, **options):
            is_verify = current is model and getattr(current, "_verify_active", False)
            if is_verify:
                before_failure.append(_state(current))
            result = observed_step(current, caches, tokens, past, **options)
            if is_verify:
                raise RuntimeError("injected failure after real target verify")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(generation, "_page_step", fail_after_real_verify)
            with pytest.raises(RuntimeError, match="injected failure after real target verify"):
                model.generate(**kwargs)
        assert len(before_failure) == 1
        _assert_state(_state(model), before_failure[0])
        torch.testing.assert_close(model.generate(**kwargs), actual, rtol=0, atol=0)
        assert model._qwen4_mtp_runtime is runtime and (runtime.session, runtime.prefill_session) == sessions
        assert len(resets) >= 4
        model.close()
        assert runtime.session is None and runtime.prefill_session is None
        torch.testing.assert_close(model.generate(**kwargs), actual, rtol=0, atol=0)
        reopened = model._qwen4_mtp_runtime
        assert reopened is not runtime
        assert reopened.session is not sessions[0] and reopened.prefill_session is not sessions[1]
    finally:
        model.close()
        oracle.close()
    assert reopened.session is None and reopened.prefill_session is None
