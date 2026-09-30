import types
from unittest.mock import patch

import pytest
import torch


def test_forward_uses_fast_path_not_slow_path():
    """Merged _model_forward must call the page-attention context fast path
    and must NOT call the full-cache load/scatter slow path."""
    src = open("vllm_merak/worker_v1/merak_model_runner_v1.py").read()
    assert "page_attention_block_ids" in src, "fast path kwargs missing"
    assert "_load_runtime_request_cache_from_paged" not in src, (
        "slow load path still present"
    )
    assert "_scatter_runtime_request_cache_to_paged" not in src, (
        "slow scatter path still present"
    )


def test_forward_keeps_mamba_bridge():
    src = open("vllm_merak/worker_v1/merak_model_runner_v1.py").read()
    assert "manage_linear_state" in src
    assert "_load_runtime_linear_state_from_store" in src
    assert "_scatter_runtime_linear_state_to_store" in src
    assert "_load_runtime_linear_state_from_vllm" not in src
    assert "_scatter_runtime_linear_state_to_vllm" not in src


def test_qwen_hybrid_state_contract_is_detected_without_proposer():
    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    fixed_width_runtime = types.SimpleNamespace(
        _get_spec_decode_verify_steps=lambda: 16
    )
    ordinary_runtime = types.SimpleNamespace(
        _get_spec_decode_verify_steps=lambda: 1
    )

    assert MerakModelRunnerV1._uses_qwen_hybrid_state_contract(
        fixed_width_runtime
    )
    assert not MerakModelRunnerV1._uses_qwen_hybrid_state_contract(
        ordinary_runtime
    )
    assert not MerakModelRunnerV1._uses_qwen_hybrid_state_contract(object())


def test_model_forward_passes_page_attention_context_to_run_request():
    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.input_batch = types.SimpleNamespace(req_ids=["req-0"])
    runner._resolve_mamba_group_id = lambda: None

    captured = {}

    class FakeModel:
        enable_page_attention = True
        placeholder_attn_layers = [types.SimpleNamespace(kv_cache=torch.ones(1))]
        logits_processor = types.SimpleNamespace(vocab_size=8)
        input_embeddings = types.SimpleNamespace(
            weight=torch.empty(1, dtype=torch.float32)
        )

        @staticmethod
        def _flatten_input_ids(input_ids):
            return input_ids.reshape(-1)

        @staticmethod
        def _flatten_inputs_embeds(inputs_embeds):
            return (
                None
                if inputs_embeds is None
                else inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
            )

        @staticmethod
        def _extract_request_multimodal_kwargs(**kwargs):
            del kwargs
            return {}, 0, 0

        @staticmethod
        def _run_request(past_len, **kwargs):
            captured["past_len"] = past_len
            captured.update(kwargs)
            return torch.arange(8, dtype=torch.float32)

    runner.model = FakeModel()
    runner._get_attn_metadata = lambda: types.SimpleNamespace(
        past_lens=torch.tensor([5], dtype=torch.long),
        subsequence_begins=torch.tensor([0, 2], dtype=torch.long),
        block_indices_begins=torch.tensor([0, 2], dtype=torch.long),
        block_indices=torch.tensor([4, 5], dtype=torch.long),
        slot_mapping=torch.tensor([256, 257], dtype=torch.long),
    )

    positions = torch.tensor([[5, 6], [15, 16], [25, 26]])
    logits = MerakModelRunnerV1._model_forward(
        runner,
        input_ids=torch.tensor([11, 12]),
        positions=positions,
    )

    assert captured["past_len"] == 5
    assert captured["page_attention_block_ids"] == (4, 5)
    assert captured["page_attention_slot_mapping"].tolist() == [256, 257]
    assert captured["input_ids"].tolist() == [11, 12]
    assert torch.equal(captured["positions"], positions)
    assert logits.shape == (2, 8)
    assert logits[0].tolist() == [0.0] * 8
    assert logits[1].tolist() == torch.arange(8, dtype=torch.float32).tolist()


def test_model_forward_routes_verify_and_keeps_every_target_logit_row():
    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.input_batch = types.SimpleNamespace(req_ids=["req-verify"])
    runner._resolve_mamba_group_id = lambda: None
    runner._merak_verify_req_ids = {"req-verify"}
    runner._merak_pending_verify_req_ids = set()
    forwarded = []

    class Proposer:
        artifact = types.SimpleNamespace(target_hidden_output_name="target_hidden")

        @staticmethod
        def on_target_forward(**kwargs):
            forwarded.append(kwargs)

    runner._merak_spec_proposer = Proposer()

    verify_logits = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    verify_hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    class FakeModel:
        enable_page_attention = True
        placeholder_attn_layers = [types.SimpleNamespace(kv_cache=torch.ones(1))]
        logits_processor = types.SimpleNamespace(vocab_size=6)
        input_embeddings = types.SimpleNamespace(
            weight=torch.empty(1, dtype=torch.float32)
        )

        @staticmethod
        def _flatten_input_ids(input_ids):
            return input_ids.reshape(-1)

        @staticmethod
        def _flatten_inputs_embeds(inputs_embeds):
            return inputs_embeds

        @staticmethod
        def _extract_request_multimodal_kwargs(**kwargs):
            del kwargs
            return {}, 0, 0

        @staticmethod
        def _run_request(past_len, **kwargs):
            assert past_len == 9
            assert kwargs["execution_phase"] == "verify"
            assert kwargs["return_result"] is True
            assert kwargs["target_hidden_output_name"] == "target_hidden"
            return types.SimpleNamespace(
                logits=verify_logits,
                hidden_states=verify_hidden,
            )

    runner.model = FakeModel()
    runner._get_attn_metadata = lambda: types.SimpleNamespace(
        past_lens=torch.tensor([9], dtype=torch.long),
        subsequence_begins=torch.tensor([0, 3], dtype=torch.long),
        block_indices_begins=torch.tensor([0, 1], dtype=torch.long),
        block_indices=torch.tensor([2], dtype=torch.long),
        slot_mapping=torch.tensor([9, 10, 11], dtype=torch.long),
    )

    merged_embeddings = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    logits = MerakModelRunnerV1._model_forward(
        runner,
        input_ids=torch.tensor([101, 102, 103]),
        inputs_embeds=merged_embeddings,
    )

    assert torch.equal(logits, verify_logits)
    assert runner._merak_pending_verify_req_ids == {"req-verify"}
    assert len(forwarded) == 1
    assert forwarded[0]["phase"] == "verify"
    assert forwarded[0]["past_len"] == 9
    assert torch.equal(forwarded[0]["input_embeddings"], merged_embeddings)
    assert torch.equal(forwarded[0]["result"].hidden_states, verify_hidden)


def test_acceptance_count_commits_target_state_before_next_draft():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.input_batch = types.SimpleNamespace(req_ids=["req-verify", "req-normal"])
    runner._merak_pending_verify_req_ids = {"req-verify"}
    events = []
    runner._commit_pending_merak_linear_verify = lambda req_id, *, accepted_steps: (
        events.append(("linear", req_id, accepted_steps))
    )

    class Proposer:
        @staticmethod
        def on_target_accept(**kwargs):
            events.append(
                (
                    "draft",
                    kwargs["req_id"],
                    kwargs["accepted_steps"],
                    kwargs["sampled_token_ids"].tolist(),
                )
            )

    runner._merak_spec_proposer = Proposer()
    sampled = torch.tensor(
        [[11, 12, -1], [21, -1, -1]],
        dtype=torch.int64,
    )

    with patch.object(
        GPUModelRunner,
        "_update_states_after_model_execute",
        lambda self, output_token_ids, scheduler_output: events.append(
            ("upstream", output_token_ids.shape)
        ),
    ):
        MerakModelRunnerV1._update_states_after_model_execute(
            runner,
            sampled,
            types.SimpleNamespace(),
        )

    assert events == [
        ("upstream", torch.Size([2, 3])),
        ("linear", "req-verify", 2),
        ("draft", "req-verify", 2, [11, 12, -1]),
    ]
    assert runner._merak_pending_verify_req_ids == set()


def test_custom_proposer_publishes_empty_drafts_near_max_model_len():
    """Token id 0 is data, not the custom-proposer empty sentinel."""

    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.input_batch = types.SimpleNamespace(req_ids=["req-near-limit"])
    runner._merak_spec_proposer = object()
    runner._draft_token_ids = torch.zeros((1, 9), dtype=torch.int32)
    runner._draft_token_req_ids = None

    runner._copy_draft_token_ids_to_cpu(
        types.SimpleNamespace(),
        zeros_only=True,
    )

    assert runner._draft_token_ids == [[]]
    assert runner._draft_token_req_ids == ["req-near-limit"]


@pytest.mark.parametrize("runtime_owner", [None, "req-0"])
def test_model_forward_restores_linear_state_for_resumed_request_without_local_idx(
    runtime_owner,
):
    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.input_batch = types.SimpleNamespace(
        req_ids=["req-0"],
        block_table=[
            types.SimpleNamespace(
                get_numpy_array=lambda: torch.tensor([[10, 11, 12]], dtype=torch.long)
            ),
            types.SimpleNamespace(
                get_numpy_array=lambda: torch.tensor([[20, 21, 22]], dtype=torch.long)
            ),
        ],
    )
    runner._resolve_mamba_group_id = lambda: 0
    # Group 0 deliberately owns two Linear layers. The runner must translate
    # each distinct group exactly once rather than passing per-layer IDs.
    runner._merak_layer_group_ids = [0, 0, 1]
    runner._mamba_block_size = lambda _gid: 4
    runner._mamba_num_speculative_blocks = lambda _gid: 0
    runner._merak_linear_logged = True
    runner.merak_linear_state_idx = {}
    runner._merak_runtime_linear_owner_req_id = runtime_owner

    events = []
    slot = object()

    class FakeStore:
        @staticmethod
        def load_slot(request_id, group_physical_ids, *, past_len):
            events.append(
                ("store-load", request_id, dict(group_physical_ids), past_len)
            )
            return slot

        @staticmethod
        def transition(request_id, group_physical_ids):
            events.append(("transition", request_id, dict(group_physical_ids)))

    runner.merak_linear_checkpoint_store = FakeStore()
    runner._begin_merak_hybrid_output_passthrough = (
        lambda runtime: events.append(("passthrough-begin", runtime))
    )
    runner._end_merak_hybrid_output_passthrough = (
        lambda runtime: events.append(("passthrough-end", runtime))
    )
    fixed_width_runtime = types.SimpleNamespace(
        _get_spec_decode_verify_steps=lambda: 5
    )

    class FakeModel:
        enable_page_attention = True
        inference_model = fixed_width_runtime
        placeholder_attn_layers = [types.SimpleNamespace(kv_cache=torch.ones(1))]
        logits_processor = types.SimpleNamespace(vocab_size=4)
        input_embeddings = types.SimpleNamespace(
            weight=torch.empty(1, dtype=torch.float32)
        )

        @staticmethod
        def _flatten_input_ids(input_ids):
            return input_ids.reshape(-1)

        @staticmethod
        def _flatten_inputs_embeds(inputs_embeds):
            return (
                None
                if inputs_embeds is None
                else inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
            )

        @staticmethod
        def _extract_request_multimodal_kwargs(**kwargs):
            del kwargs
            return {}, 0, 0

        @staticmethod
        def _clear_runtime_linear_caches():
            events.append(("clear",))

        @staticmethod
        def _load_runtime_linear_state_from_store(loaded_slot):
            events.append(("load", loaded_slot))

        @staticmethod
        def _scatter_runtime_linear_state_to_store(store, request_id):
            events.append(("scatter", store, request_id))

        @staticmethod
        def _run_request(past_len, **kwargs):
            events.append(("run", past_len, tuple(kwargs["page_attention_block_ids"])))
            return torch.arange(4, dtype=torch.float32)

    runner.model = FakeModel()
    runner._get_attn_metadata = lambda: types.SimpleNamespace(
        past_lens=torch.tensor([5], dtype=torch.long),
        subsequence_begins=torch.tensor([0, 1], dtype=torch.long),
        block_indices_begins=torch.tensor([0, 2], dtype=torch.long),
        block_indices=torch.tensor([7, 8], dtype=torch.long),
        slot_mapping=torch.tensor([321], dtype=torch.long),
    )

    MerakModelRunnerV1._model_forward(runner, input_ids=torch.tensor([99]))

    assert ("clear",) not in events
    expected = []
    if runtime_owner is None:
        expected.extend(
            [
                ("store-load", "req-0", {0: 11, 1: 21}, 5),
                ("load", slot),
            ]
        )
    expected.extend(
        [
            ("passthrough-begin", fixed_width_runtime),
            ("run", 5, (7, 8)),
            ("passthrough-end", fixed_width_runtime),
            ("scatter", runner.merak_linear_checkpoint_store, "req-0"),
            ("transition", "req-0", {0: 11, 1: 21}),
        ]
    )
    assert events[: len(expected)] == expected
    assert runner.merak_linear_state_idx == {"req-0": 1}


def test_model_forward_acquires_clean_slot_before_new_request_and_commits_after_run():
    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.input_batch = types.SimpleNamespace(
        req_ids=["req-new"],
        block_table=[
            types.SimpleNamespace(
                get_numpy_array=lambda: torch.tensor([[17]], dtype=torch.long)
            ),
        ],
    )
    runner._resolve_mamba_group_id = lambda: 0
    runner._merak_layer_group_ids = [0, 0]
    runner._mamba_block_size = lambda _gid: 4
    runner._mamba_num_speculative_blocks = lambda _gid: 0
    runner._merak_linear_logged = True
    runner.merak_linear_state_idx = {}
    events = []

    class FakeStore:
        @staticmethod
        def acquire(request_id, group_physical_ids, *, past_len):
            events.append(("acquire", request_id, dict(group_physical_ids), past_len))

        @staticmethod
        def transition(request_id, group_physical_ids):
            events.append(("transition", request_id, dict(group_physical_ids)))

    store = FakeStore()
    runner.merak_linear_checkpoint_store = store

    class FakeModel:
        enable_page_attention = False
        placeholder_attn_layers = [types.SimpleNamespace(kv_cache=torch.ones(1))]
        logits_processor = types.SimpleNamespace(vocab_size=2)
        input_embeddings = types.SimpleNamespace(
            weight=torch.empty(1, dtype=torch.float32)
        )

        @staticmethod
        def _flatten_input_ids(input_ids):
            return input_ids.reshape(-1)

        @staticmethod
        def _flatten_inputs_embeds(inputs_embeds):
            return inputs_embeds

        @staticmethod
        def _extract_request_multimodal_kwargs(**kwargs):
            del kwargs
            return {}, 0, 0

        @staticmethod
        def _clear_runtime_linear_caches():
            events.append(("clear",))

        @staticmethod
        def _run_request(past_len, **kwargs):
            del kwargs
            events.append(("run", past_len))
            return torch.tensor([1.0, 2.0])

        @staticmethod
        def _scatter_runtime_linear_state_to_store(actual_store, request_id):
            events.append(("scatter", actual_store, request_id))

    runner.model = FakeModel()
    runner._get_attn_metadata = lambda: types.SimpleNamespace(
        past_lens=torch.tensor([0], dtype=torch.long),
        subsequence_begins=torch.tensor([0, 1], dtype=torch.long),
        block_indices_begins=torch.tensor([0, 0], dtype=torch.long),
        block_indices=torch.tensor([], dtype=torch.long),
        slot_mapping=torch.tensor([0], dtype=torch.long),
    )

    MerakModelRunnerV1._model_forward(runner, input_ids=torch.tensor([42]))

    assert events == [
        ("acquire", "req-new", {0: 17}, 0),
        ("clear",),
        ("run", 0),
        ("scatter", store, "req-new"),
        ("transition", "req-new", {0: 17}),
    ]


def test_mamba_group_mapping_rejects_partial_layer_mapping():
    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    class MambaSpec:
        block_size = 4
        num_speculative_blocks = 0

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.model = types.SimpleNamespace(
        placeholder_linear_state_layers=[
            types.SimpleNamespace(layer_name="model.layers.0.linear_attn"),
            types.SimpleNamespace(layer_name="model.layers.1.linear_attn"),
        ]
    )
    runner.kv_cache_config = types.SimpleNamespace(
        kv_cache_groups=[
            types.SimpleNamespace(
                kv_cache_spec=MambaSpec(),
                layer_names=["model.layers.0.linear_attn"],
            )
        ]
    )

    with pytest.raises(RuntimeError, match="missing.*model.layers.1.linear_attn"):
        runner._resolve_mamba_group_id()


def test_mamba_group_mapping_rejects_duplicate_layer_ownership():
    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    class MambaSpec:
        block_size = 4
        num_speculative_blocks = 0

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.model = types.SimpleNamespace(
        placeholder_linear_state_layers=[
            types.SimpleNamespace(layer_name="model.layers.0.linear_attn")
        ]
    )
    runner.kv_cache_config = types.SimpleNamespace(
        kv_cache_groups=[
            types.SimpleNamespace(
                kv_cache_spec=MambaSpec(),
                layer_names=["model.layers.0.linear_attn"],
            ),
            types.SimpleNamespace(
                kv_cache_spec=MambaSpec(),
                layer_names=["model.layers.0.linear_attn"],
            ),
        ]
    )

    with pytest.raises(RuntimeError, match="duplicate.*model.layers.0.linear_attn"):
        runner._resolve_mamba_group_id()


def test_mamba_group_mapping_rejects_inconsistent_group_specs():
    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    class MambaSpec:
        def __init__(self, block_size):
            self.block_size = block_size
            self.num_speculative_blocks = 0

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.model = types.SimpleNamespace(
        placeholder_linear_state_layers=[
            types.SimpleNamespace(layer_name="model.layers.0.linear_attn"),
            types.SimpleNamespace(layer_name="model.layers.1.linear_attn"),
        ]
    )
    runner.kv_cache_config = types.SimpleNamespace(
        kv_cache_groups=[
            types.SimpleNamespace(
                kv_cache_spec=MambaSpec(4),
                layer_names=["model.layers.0.linear_attn"],
            ),
            types.SimpleNamespace(
                kv_cache_spec=MambaSpec(8),
                layer_names=["model.layers.1.linear_attn"],
            ),
        ]
    )

    with pytest.raises(RuntimeError, match="must share.*block_size"):
        runner._resolve_mamba_group_id()


def test_prefix_on_mamba_logical_index_overflow_is_not_clamped():
    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.cache_config = types.SimpleNamespace(
        enable_prefix_caching=True, mamba_cache_mode="none"
    )

    with pytest.raises(RuntimeError, match="prefix caching.*logical index 1.*width 1"):
        runner._normalize_mamba_logical_index(1, width=1, group_id=2)


def test_prefix_off_mamba_none_mode_normalizes_to_only_running_slot():
    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.cache_config = types.SimpleNamespace(
        enable_prefix_caching=False, mamba_cache_mode="none"
    )

    assert runner._normalize_mamba_logical_index(7, width=1, group_id=2) == 0


def test_qwen_v1_page_attention_context_keeps_positional_runtime_abi():
    from vllm_merak.model_executor.model_loader.merak_for_causal_lm import (
        MerakForCausalLM,
    )

    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.inference_device = torch.device("cpu")
    model.placeholder_attn_layers = [types.SimpleNamespace(kv_cache="qwen-cache")]
    model._get_block_size = lambda: 64
    captured = {}

    class Runtime:
        def set_page_attention_context(self, **kwargs):
            captured.update(kwargs)

    model._set_runtime_page_attention_context(
        Runtime(),
        (3, 5),
        torch.tensor([192], dtype=torch.int64),
    )

    assert set(captured) == {
        "paged_kv_caches",
        "block_ids",
        "slot_mapping",
        "block_size",
    }
    assert captured["paged_kv_caches"] == ("qwen-cache",)
