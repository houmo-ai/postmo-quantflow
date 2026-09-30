from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_merak.spec_decode.hmonnx_backend import (
    HMONNXDraftRuntimeBackend,
    _MTPPageCacheState,
    _PrivateCacheState,
    _dflash2_attention_mask,
    _dflash2_greedy_selector_walk,
    _dflash2_query_kv_ranges,
    _dflash2_sample_selector_walk,
    _gemma_readonly_sliding_cache_range,
)


@dataclass(frozen=True)
class _InputInfo:
    shape: tuple[int, ...]
    dtype: torch.dtype


class _DFlashContextSession:
    def __init__(self, *, hidden_capacity: int) -> None:
        self.inputs = {
            "target_hidden": _InputInfo(
                (1, hidden_capacity, 4),
                torch.float32,
            ),
            "past_seq_length": _InputInfo((1,), torch.int64),
            "current_input_length": _InputInfo((1,), torch.int64),
            "past_key_cache_0": _InputInfo((1, 1, 16, 2), torch.float32),
            "past_value_cache_0": _InputInfo(
                (1, 1, 16, 2),
                torch.float32,
            ),
        }
        self.feeds: list[dict[str, torch.Tensor]] = []
        self.copy_outputs = False

    def get_input_names(self) -> list[str]:
        return list(self.inputs)

    def get_input(self, name: str) -> _InputInfo:
        return self.inputs[name]

    @staticmethod
    def get_output_names() -> list[str]:
        return ["present_key_cache_0", "present_value_cache_0"]

    def run(self, feed: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        self.feeds.append(feed)
        outputs = (
            feed["past_key_cache_0"],
            feed["past_value_cache_0"],
        )
        if self.copy_outputs:
            return tuple(output.clone() for output in outputs)
        return outputs


class _MTPAppendSession:
    def __init__(self, *, hidden_capacity: int) -> None:
        self.inputs = {
            "post_norm_hidden": _InputInfo(
                (1, hidden_capacity, 4),
                torch.float32,
            ),
            "next_token_embedding": _InputInfo(
                (1, hidden_capacity, 4),
                torch.float32,
            ),
            "past_seq_length": _InputInfo((1,), torch.int64),
            "current_input_length": _InputInfo((1,), torch.int64),
            "past_key_cache_0": _InputInfo((1, 1, 16, 2), torch.float32),
        }
        self.feeds: list[dict[str, torch.Tensor]] = []

    def get_input_names(self) -> list[str]:
        return list(self.inputs)

    def get_input(self, name: str) -> _InputInfo:
        return self.inputs[name]

    @staticmethod
    def get_output_names() -> list[str]:
        return ["logits", "post_norm_out"]

    def run(self, feed: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        self.feeds.append(feed)
        capacity = self.inputs["post_norm_hidden"].shape[1]
        return (
            torch.zeros(1, capacity, 8),
            torch.zeros(1, capacity, 4),
        )


class _CapturedInterpreter:
    has_captured_graph = True
    capture_disabled_reason = None


class _BorrowingInterpreter:
    def __init__(self) -> None:
        self.indices: tuple[int, ...] | None = None

    def set_borrowed_output_indices(self, indices) -> None:
        self.indices = tuple(indices)


class _BorrowingSession:
    def __init__(self, output_names: list[str]) -> None:
        self._output_names = output_names
        self.interpreter = _BorrowingInterpreter()

    def get_output_names(self) -> list[str]:
        return self._output_names

    def enable_borrowed_cuda_graph_outputs(self, names) -> int:
        selected = set(names)
        indices = [
            index for index, name in enumerate(self._output_names) if name in selected
        ]
        self.interpreter.set_borrowed_output_indices(indices)
        return len(indices)


class _PrimeDecodeSession(_MTPAppendSession):
    def __init__(self, *, mode: str) -> None:
        super().__init__(hidden_capacity=1)
        if mode == "dflash":
            self.inputs.pop("post_norm_hidden")
            self.inputs.pop("next_token_embedding")
            self.inputs["noise_embedding"] = _InputInfo(
                (1, 16, 4),
                torch.float32,
            )
            self.inputs["attn_mask"] = _InputInfo(
                (1, 1, 16, 32),
                torch.float32,
            )
        self.interpreter = _CapturedInterpreter()

    def run(self, feed: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        self.feeds.append(feed)
        # Model capture warm-ups are stateful; priming must scrub them before
        # any real request claims this same persistent storage.
        feed["past_key_cache_0"].fill_(9)
        return (
            torch.zeros(1, 1, 8),
            torch.zeros(1, 1, 4),
        )


class _PrimeContextDecodeSession(_DFlashContextSession):
    def __init__(self, *, hidden_capacity: int) -> None:
        super().__init__(hidden_capacity=hidden_capacity)
        self.interpreter = _CapturedInterpreter()

    def run(self, feed: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        self.feeds.append(feed)
        feed["past_key_cache_0"].fill_(7)
        feed["past_value_cache_0"].fill_(8)
        return (
            feed["past_key_cache_0"],
            feed["past_value_cache_0"],
        )


def test_dflash2_symmetric_window_dense_and_flash_metadata_match() -> None:
    mask = _dflash2_attention_mask(
        (1, 4, 12),
        past_length=5,
        current_length=4,
        sliding_window=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    ranges = _dflash2_query_kv_ranges(
        (1, 4, 2),
        past_length=5,
        current_length=4,
        sliding_window=3,
        dtype=torch.int32,
        device=torch.device("cpu"),
    )

    assert ranges.tolist() == [[[3, 8], [4, 9], [5, 9], [6, 9]]]
    for query_index, (start, end) in enumerate(ranges[0].tolist()):
        visible = torch.where(mask[0, query_index] == 0)[0].tolist()
        assert visible == list(range(start, end))


def test_dflash2_scheme_b_walk_uses_selected_predecessor_row() -> None:
    candidate_ids = torch.tensor([[[10, 11, 12], [20, 21, 22], [30, 31, 32]]])
    first_scores = torch.tensor([[0.0, 4.0, 1.0]])
    transition_scores = torch.zeros(1, 2, 3, 3)
    transition_scores[0, 0, 1] = torch.tensor([0.0, 1.0, 7.0])
    transition_scores[0, 1, 2] = torch.tensor([5.0, 1.0, 0.0])

    selected = _dflash2_greedy_selector_walk(
        candidate_ids,
        first_scores,
        transition_scores,
        num_draft_tokens=3,
    )

    assert selected.tolist() == [[11, 22, 30]]


def test_dflash2_sampled_walk_returns_sparse_full_vocab_probabilities() -> None:
    candidate_ids = torch.tensor([[[1, 4], [2, 5]]])
    first_scores = torch.tensor([[0.0, 1.0]])
    transition_scores = torch.tensor([[[[4.0, 0.0], [0.0, 4.0]]]])
    generator = torch.Generator(device="cpu").manual_seed(7)

    tokens, probabilities = _dflash2_sample_selector_walk(
        candidate_ids,
        first_scores,
        transition_scores,
        num_draft_tokens=2,
        temperature=1.0,
        vocab_size=8,
        generator=generator,
    )
    repeated_tokens, repeated_probabilities = _dflash2_sample_selector_walk(
        candidate_ids,
        first_scores,
        transition_scores,
        num_draft_tokens=2,
        temperature=1.0,
        vocab_size=8,
        generator=torch.Generator(device="cpu").manual_seed(7),
    )

    assert tokens.shape == (1, 2)
    assert probabilities.shape == (1, 2, 8)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1, 2))
    assert torch.count_nonzero(probabilities[0, 0]).item() == 2
    assert torch.count_nonzero(probabilities[0, 1]).item() == 2
    assert int(tokens[0, 0]) in {1, 4}
    assert int(tokens[0, 1]) in {2, 5}
    torch.testing.assert_close(repeated_tokens, tokens)
    torch.testing.assert_close(repeated_probabilities, probabilities)


def test_gemma4_draft_device_discovers_unbound_target_flash_attention() -> None:
    observed_classes = []
    runtime_model = object()

    def attention_node_map(
        candidate,
        fallback,
        *,
        module_class_names,
    ):
        assert candidate is runtime_model
        assert fallback == torch.device("cuda:0")
        observed_classes.append(module_class_names)
        return [
            ("local_owner", torch.device("cuda:2")),
            ("global_owner", torch.device("cuda:2")),
        ]

    backend = HMONNXDraftRuntimeBackend.__new__(HMONNXDraftRuntimeBackend)
    backend.runner = SimpleNamespace(
        device=torch.device("cuda:0"),
        _iter_hmonnx_models=lambda _runtime: [runtime_model],
        _hmonnx_attention_node_map=attention_node_map,
    )
    backend.target_model = SimpleNamespace(
        inference_model=object(),
        page_attention_bindings=[
            SimpleNamespace(attention_kind="sliding_attention"),
            SimpleNamespace(attention_kind="full_attention"),
        ],
    )
    backend.artifact = SimpleNamespace(
        is_gemma4=True,
        draft_layer_attention_types=(
            "sliding_attention",
            "full_attention",
        ),
    )

    assert backend._select_draft_device() == torch.device("cuda:2")
    assert len(observed_classes) == 1
    assert "FlashAttention" in observed_classes[0]


def test_gemma4_draft_device_uses_physical_mixed_attention_bindings() -> None:
    runtime_model = object()
    physical_bindings = (
        SimpleNamespace(attention_kind="sliding_attention"),
        SimpleNamespace(attention_kind="full_attention"),
        SimpleNamespace(attention_kind="sliding_attention"),
    )
    logical_bindings = tuple(
        SimpleNamespace(attention_kind=kind)
        for kind in (
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        )
    )

    backend = HMONNXDraftRuntimeBackend.__new__(HMONNXDraftRuntimeBackend)
    backend.runner = SimpleNamespace(
        device=torch.device("cuda:0"),
        _iter_hmonnx_models=lambda _runtime: [runtime_model],
        _hmonnx_attention_node_map=lambda *_args, **_kwargs: [
            ("sliding_owner_0", torch.device("cuda:3")),
            ("full_owner_0", torch.device("cuda:3")),
            ("sliding_owner_1", torch.device("cuda:3")),
        ],
    )
    backend.target_model = SimpleNamespace(
        inference_model=object(),
        page_attention_bindings=logical_bindings,
        gemma4_cache_contract=SimpleNamespace(
            context_bindings=physical_bindings,
        ),
    )
    backend.artifact = SimpleNamespace(
        is_gemma4=True,
        draft_layer_attention_types=(
            "sliding_attention",
            "full_attention",
        ),
    )

    assert backend._select_draft_device() == torch.device("cuda:3")


@pytest.mark.parametrize(
    (
        "decode_graph",
        "context_graph",
        "expected_primes",
    ),
    [
        ("0", "0", []),
        ("1", "0", ["decode"]),
        ("0", "1", ["context_decode"]),
        ("1", "1", ["context_decode", "decode"]),
    ],
)
def test_dflash_cuda_graphs_are_independently_opt_in(
    monkeypatch,
    decode_graph: str,
    context_graph: str,
    expected_primes: list[str],
) -> None:
    monkeypatch.setenv(
        "VLLM_MERAK_SPEC_DRAFT_CUDA_GRAPH",
        decode_graph,
    )
    monkeypatch.setenv(
        "VLLM_MERAK_SPEC_DRAFT_CONTEXT_CUDA_GRAPH",
        context_graph,
    )
    monkeypatch.setattr(
        HMONNXDraftRuntimeBackend,
        "_select_draft_device",
        lambda _self: torch.device("cpu"),
    )
    monkeypatch.setattr(
        HMONNXDraftRuntimeBackend,
        "_load_graphs",
        lambda self: self._sessions.update(
            {
                "context_decode": object(),
                "decode": object(),
            }
        ),
    )
    monkeypatch.setattr(
        HMONNXDraftRuntimeBackend,
        "_initialize_private_cache",
        lambda _self: None,
    )
    primes: list[str] = []
    monkeypatch.setattr(
        HMONNXDraftRuntimeBackend,
        "_prime_private_context_decode_cuda_graph",
        lambda _self: primes.append("context_decode"),
    )
    monkeypatch.setattr(
        HMONNXDraftRuntimeBackend,
        "_prime_private_decode_cuda_graph",
        lambda _self: primes.append("decode"),
    )

    backend = HMONNXDraftRuntimeBackend(
        runner=SimpleNamespace(model=object()),
        artifact=SimpleNamespace(
            is_gemma4=False,
            mode="dflash",
        ),
    )

    assert backend._enable_private_decode_cuda_graph is (decode_graph == "1")
    assert backend._enable_private_context_decode_cuda_graph is (context_graph == "1")
    assert primes == expected_primes


def test_dflash_shared_page_slot_buffer_handles_context_and_decode_shapes() -> None:
    backend = object.__new__(HMONNXDraftRuntimeBackend)
    slot_mapping = torch.full((8,), -1, dtype=torch.int64)
    committed_block_ids = torch.arange(4, dtype=torch.int64)
    backend._dflash_page_cache = SimpleNamespace(
        block_ids=committed_block_ids.clone(),
        committed_block_ids=committed_block_ids,
        slot_mapping=slot_mapping,
        slot_offsets=torch.arange(8, dtype=torch.int64),
        max_sequence_length=32,
    )
    storage_ptr = slot_mapping.data_ptr()

    backend._stage_dflash_page_slots(
        past_length=5,
        current_length=2,
        physical_length=4,
    )

    assert slot_mapping.tolist() == [5, 6, -1, -1, -1, -1, -1, -1]
    assert slot_mapping.data_ptr() == storage_ptr

    backend._stage_dflash_page_slots(
        past_length=7,
        current_length=3,
        physical_length=3,
    )

    assert slot_mapping.tolist() == [7, 8, 9, -1, -1, -1, -1, -1]
    assert slot_mapping.data_ptr() == storage_ptr


def _fake_dflash_page_owner(num_blocks: int = 6) -> SimpleNamespace:
    return SimpleNamespace(
        k_qdata=torch.zeros(num_blocks, 1, 1),
        k_scale_exp=torch.zeros(num_blocks, 1, 1),
        v_qdata=torch.zeros(num_blocks, 1, 1),
        v_scale_exp=torch.zeros(num_blocks, 1, 1),
    )


def test_dflash_transient_tail_uses_cow_pages_then_restores_identity() -> None:
    backend = object.__new__(HMONNXDraftRuntimeBackend)
    owner = _fake_dflash_page_owner()
    for tensor in (
        owner.k_qdata,
        owner.k_scale_exp,
        owner.v_qdata,
        owner.v_scale_exp,
    ):
        tensor[1].fill_(7)
    committed_snapshot = tuple(
        tensor[:4].clone()
        for tensor in (
            owner.k_qdata,
            owner.k_scale_exp,
            owner.v_qdata,
            owner.v_scale_exp,
        )
    )
    committed_block_ids = torch.arange(4, dtype=torch.int64)
    backend._dflash_page_cache = SimpleNamespace(
        caches=(owner,),
        block_ids=committed_block_ids.clone(),
        committed_block_ids=committed_block_ids,
        slot_mapping=torch.full((8,), -1, dtype=torch.int64),
        slot_offsets=torch.arange(8, dtype=torch.int64),
        scratch_block_ids=(4, 5),
        block_size=64,
        max_sequence_length=256,
    )

    backend._stage_dflash_page_slots(
        past_length=126,
        current_length=4,
        physical_length=4,
        transient=True,
    )

    page = backend._dflash_page_cache
    assert page.block_ids.tolist() == [0, 4, 5, 3]
    assert page.slot_mapping.tolist() == [318, 319, 320, 321, -1, -1, -1, -1]
    for tensor, expected in zip(
        (
            owner.k_qdata,
            owner.k_scale_exp,
            owner.v_qdata,
            owner.v_scale_exp,
        ),
        committed_snapshot,
        strict=True,
    ):
        torch.testing.assert_close(tensor[:4], expected)
        assert tensor[4].item() == 7
        assert tensor[5].item() == 0

    backend._stage_dflash_page_slots(
        past_length=130,
        current_length=2,
        physical_length=4,
    )
    assert page.block_ids.tolist() == [0, 1, 2, 3]
    assert page.slot_mapping.tolist() == [130, 131, -1, -1, -1, -1, -1, -1]


def test_release_request_clears_external_page_owner_without_rebinding() -> None:
    backend = object.__new__(HMONNXDraftRuntimeBackend)
    owner = _fake_dflash_page_owner()
    for tensor in (
        owner.k_qdata,
        owner.k_scale_exp,
        owner.v_qdata,
        owner.v_scale_exp,
    ):
        tensor.fill_(3)
    storage_ptrs = tuple(
        tensor.data_ptr()
        for tensor in (
            owner.k_qdata,
            owner.k_scale_exp,
            owner.v_qdata,
            owner.v_scale_exp,
        )
    )
    committed_block_ids = torch.arange(4, dtype=torch.int64)
    backend._dflash_page_cache = SimpleNamespace(
        caches=(owner,),
        block_ids=torch.tensor([0, 4, 5, 3], dtype=torch.int64),
        committed_block_ids=committed_block_ids,
        slot_mapping=torch.arange(8, dtype=torch.int64),
        storage_ptrs=storage_ptrs,
    )
    backend._mtp_page_cache = None
    backend._private_cache = _PrivateCacheState(
        values={},
        storage_ptrs={},
        owner_req_id="first",
    )

    backend.release_request("first")

    assert backend._private_cache.owner_req_id is None
    assert backend._dflash_page_cache.block_ids.tolist() == [0, 1, 2, 3]
    assert backend._dflash_page_cache.slot_mapping.tolist() == [-1] * 8
    for tensor, pointer in zip(
        (
            owner.k_qdata,
            owner.k_scale_exp,
            owner.v_qdata,
            owner.v_scale_exp,
        ),
        storage_ptrs,
        strict=True,
    ):
        assert tensor.data_ptr() == pointer
        assert torch.count_nonzero(tensor).item() == 0


def test_target_cuda_graph_borrows_only_transactional_linear_outputs() -> None:
    prefill_session = _BorrowingSession(["logits", "conv_cache_out_q_0_0", "hidden"])
    decode_session = _BorrowingSession(
        [
            "logits",
            "conv_cache_out_q_0_0",
            "conv_cache_out_q_0_1",
            "recurrent_state_out_0_0",
            "recurrent_state_out_0_1",
            "target_hidden",
        ]
    )
    backend = HMONNXDraftRuntimeBackend.__new__(HMONNXDraftRuntimeBackend)
    backend.target_model = SimpleNamespace(
        inference_model=SimpleNamespace(
            prefill_model=SimpleNamespace(hmonnx_session=prefill_session),
            decode_model=SimpleNamespace(hmonnx_session=decode_session),
        )
    )

    assert backend._enable_target_transactional_output_borrowing() == 5
    assert prefill_session.interpreter.indices == (1,)
    assert decode_session.interpreter.indices == (1, 2, 3, 4)


@pytest.mark.parametrize(
    ("mode", "expected_current_length"),
    [("mtp", 1), ("dflash", 16)],
)
def test_private_decode_cuda_graph_primes_then_resets_same_cache(
    mode: str,
    expected_current_length: int,
) -> None:
    backend = HMONNXDraftRuntimeBackend.__new__(HMONNXDraftRuntimeBackend)
    backend.device = torch.device("cpu")
    backend.artifact = SimpleNamespace(mode=mode)
    session = _PrimeDecodeSession(mode=mode)
    backend._sessions = {"decode": session}
    cache = torch.zeros(1, 1, 16, 2)
    cache_ptr = cache.data_ptr()
    backend._private_cache = _PrivateCacheState(
        values={"past_key_cache_0": cache},
        storage_ptrs={"past_key_cache_0": cache_ptr},
    )

    backend._prime_private_decode_cuda_graph()

    assert len(session.feeds) == 1
    feed = session.feeds[0]
    assert feed["past_key_cache_0"] is cache
    assert feed["past_key_cache_0"].data_ptr() == cache_ptr
    assert feed["past_seq_length"].item() == 0
    assert feed["current_input_length"].item() == expected_current_length
    assert torch.count_nonzero(cache).item() == 0
    assert backend._private_cache.owner_req_id is None


def test_private_context_decode_cuda_graph_primes_shared_cache_then_resets():
    backend = HMONNXDraftRuntimeBackend.__new__(HMONNXDraftRuntimeBackend)
    backend.device = torch.device("cpu")
    backend.artifact = SimpleNamespace(mode="dflash")
    session = _PrimeContextDecodeSession(hidden_capacity=10)
    backend._sessions = {"context_decode": session}
    cache_values = {
        "past_key_cache_0": torch.zeros(1, 1, 16, 2),
        "past_value_cache_0": torch.zeros(1, 1, 16, 2),
    }
    backend._private_cache = _PrivateCacheState(
        values=cache_values,
        storage_ptrs={name: value.data_ptr() for name, value in cache_values.items()},
    )

    backend._prime_private_context_decode_cuda_graph()

    assert len(session.feeds) == 1
    feed = session.feeds[0]
    assert feed["current_input_length"].item() == 10
    assert feed["past_seq_length"].item() == 0
    assert torch.count_nonzero(feed["target_hidden"]).item() == 0
    for name, cache in cache_values.items():
        assert feed[name] is cache
        assert feed[name].data_ptr() == cache.data_ptr()
        assert torch.count_nonzero(cache).item() == 0
    assert backend._private_cache.owner_req_id is None


def test_mtp_single_pair_append_uses_decode_graph() -> None:
    backend = HMONNXDraftRuntimeBackend.__new__(HMONNXDraftRuntimeBackend)
    backend.device = torch.device("cpu")
    prefill = _MTPAppendSession(hidden_capacity=256)
    decode = _MTPAppendSession(hidden_capacity=1)
    backend._sessions = {"prefill": prefill, "decode": decode}
    cache = torch.zeros(1, 1, 16, 2)
    backend._private_cache = _PrivateCacheState(
        values={"past_key_cache_0": cache},
        storage_ptrs={"past_key_cache_0": cache.data_ptr()},
    )
    backend.target_model = SimpleNamespace(
        embed_input_ids=lambda token_ids: token_ids.unsqueeze(-1).expand(
            *token_ids.shape,
            4,
        )
    )

    backend.append_mtp_context(
        req_id="request",
        hidden_states=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        next_token_ids=torch.tensor([7]),
        past_length=12,
    )

    assert not prefill.feeds
    assert len(decode.feeds) == 1
    feed = decode.feeds[0]
    assert feed["past_seq_length"].item() == 12
    assert feed["current_input_length"].item() == 1
    assert feed["past_key_cache_0"] is cache
    torch.testing.assert_close(
        feed["post_norm_hidden"],
        torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),
    )
    torch.testing.assert_close(
        feed["next_token_embedding"],
        torch.full((1, 1, 4), 7.0),
    )


def test_mtp_context_append_prefers_merged_visual_embeddings() -> None:
    backend = HMONNXDraftRuntimeBackend.__new__(HMONNXDraftRuntimeBackend)
    backend.device = torch.device("cpu")
    prefill = _MTPAppendSession(hidden_capacity=256)
    decode = _MTPAppendSession(hidden_capacity=1)
    backend._sessions = {"prefill": prefill, "decode": decode}
    cache = torch.zeros(1, 1, 16, 2)
    backend._private_cache = _PrivateCacheState(
        values={"past_key_cache_0": cache},
        storage_ptrs={"past_key_cache_0": cache.data_ptr()},
    )

    def reject_token_lookup(_token_ids):
        raise AssertionError("visual MTP prefill must not embed placeholder ids")

    backend.target_model = SimpleNamespace(embed_input_ids=reject_token_lookup)
    visual_embedding = torch.tensor([[1.5, 2.5, 3.5, 4.5]])

    backend.append_mtp_context(
        req_id="request",
        hidden_states=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        next_token_ids=torch.tensor([151655]),
        next_token_embeddings=visual_embedding,
        past_length=12,
    )

    assert not prefill.feeds
    assert len(decode.feeds) == 1
    torch.testing.assert_close(
        decode.feeds[0]["next_token_embedding"],
        visual_embedding.unsqueeze(0),
    )


def test_mtp_draft_keeps_token_feedback_on_device_until_batched_result() -> None:
    backend = HMONNXDraftRuntimeBackend.__new__(HMONNXDraftRuntimeBackend)
    backend.device = torch.device("cpu")
    decode = _MTPAppendSession(hidden_capacity=1)
    backend._sessions = {"decode": decode}
    cache = torch.zeros(1, 1, 16, 2)
    backend._private_cache = _PrivateCacheState(
        values={"past_key_cache_0": cache},
        storage_ptrs={"past_key_cache_0": cache.data_ptr()},
    )
    backend.target_model = SimpleNamespace(
        embed_input_ids=lambda token_ids: token_ids.unsqueeze(-1).expand(
            *token_ids.shape,
            4,
        )
    )

    drafts = backend.draft_mtp(
        req_id="request",
        current_token_id=7,
        target_hidden=torch.ones(1, 4),
        past_length=3,
        num_draft_tokens=3,
    )

    assert drafts == [0, 0, 0]
    assert len(decode.feeds) == 3
    assert decode.feeds[0]["next_token_embedding"][0, 0, 0] == 7
    assert decode.feeds[1]["next_token_embedding"][0, 0, 0] == 0
    assert decode.feeds[2]["next_token_embedding"][0, 0, 0] == 0


def test_mtp_page_cache_transaction_restores_selected_draft_step() -> None:
    def cache(num_blocks: int) -> SimpleNamespace:
        return SimpleNamespace(
            num_blocks=num_blocks,
            k_qdata=torch.zeros(num_blocks, 2),
            k_scale_exp=torch.zeros(num_blocks, 1),
            v_qdata=torch.zeros(num_blocks, 3),
            v_scale_exp=torch.zeros(num_blocks, 1),
        )

    owner = cache(4)
    snapshot = cache(10)
    tensor_names = (
        "k_qdata",
        "k_scale_exp",
        "v_qdata",
        "v_scale_exp",
    )
    for tensor_index, tensor_name in enumerate(tensor_names, start=1):
        tensor = getattr(owner, tensor_name)
        tensor.copy_(torch.arange(tensor.numel()).reshape_as(tensor) * tensor_index)

    backend = HMONNXDraftRuntimeBackend.__new__(HMONNXDraftRuntimeBackend)
    backend.artifact = SimpleNamespace(num_draft_tokens=4)
    backend._mtp_page_cache = _MTPPageCacheState(
        cache=owner,
        snapshot_cache=snapshot,
        snapshot_blocks_per_step=2,
        modules={},
        contexts={},
        block_ids=torch.arange(4),
        slot_mappings={},
        block_size=64,
        max_sequence_length=256,
        storage_ptrs=(),
    )
    backend._begin_mtp_page_transaction(
        req_id="request",
        past_length=63,
        num_draft_tokens=4,
    )

    selected_state: tuple[torch.Tensor, ...] | None = None
    for step in range(1, 5):
        for tensor_name in tensor_names:
            tensor = getattr(owner, tensor_name)
            tensor[0].add_(step)
            tensor[1].add_(step * 10)
        backend._snapshot_mtp_page_transaction(step)
        if step == 2:
            selected_state = tuple(
                getattr(owner, tensor_name).clone() for tensor_name in tensor_names
            )

    assert selected_state is not None
    backend.commit_mtp_draft(
        req_id="request",
        accepted_steps=2,
    )
    for tensor_name, expected in zip(
        tensor_names,
        selected_state,
        strict=True,
    ):
        assert torch.equal(getattr(owner, tensor_name), expected)
    assert backend._mtp_page_cache.transaction is None


def _dflash_backend() -> tuple[
    HMONNXDraftRuntimeBackend,
    _DFlashContextSession,
    _DFlashContextSession,
]:
    backend = HMONNXDraftRuntimeBackend.__new__(HMONNXDraftRuntimeBackend)
    backend.device = torch.device("cpu")
    context = _DFlashContextSession(hidden_capacity=2)
    context_decode = _DFlashContextSession(hidden_capacity=1)
    backend._sessions = {
        "context": context,
        "context_decode": context_decode,
    }
    cache_values = {
        "past_key_cache_0": torch.zeros(1, 1, 16, 2),
        "past_value_cache_0": torch.zeros(1, 1, 16, 2),
    }
    backend._private_cache = _PrivateCacheState(
        values=cache_values,
        storage_ptrs={name: value.data_ptr() for name, value in cache_values.items()},
    )
    return backend, context, context_decode


def test_dflash_accepted_one_reuses_context_cache_and_fails_on_copy() -> None:
    backend, context, context_decode = _dflash_backend()

    backend.append_dflash_context(
        req_id="request",
        hidden_states=torch.arange(8, dtype=torch.float32).reshape(2, 4),
        past_length=0,
        decode_step=False,
    )
    backend.append_dflash_context(
        req_id="request",
        hidden_states=torch.tensor([[10.0, 11.0, 12.0, 13.0]]),
        past_length=2,
        decode_step=True,
    )

    prefill_feed = context.feeds[-1]
    accepted_feed = context_decode.feeds[-1]
    for cache_name, persistent in backend._private_cache.values.items():
        assert prefill_feed[cache_name] is persistent
        assert accepted_feed[cache_name] is persistent
        assert prefill_feed[cache_name].data_ptr() == persistent.data_ptr()
        assert accepted_feed[cache_name].data_ptr() == persistent.data_ptr()
    assert accepted_feed["past_seq_length"].item() == 2
    assert accepted_feed["current_input_length"].item() == 1
    torch.testing.assert_close(
        accepted_feed["target_hidden"],
        torch.tensor([[[10.0, 11.0, 12.0, 13.0]]]),
    )

    context_decode.copy_outputs = True
    with pytest.raises(RuntimeError, match="materialized/copied"):
        backend.append_dflash_context(
            req_id="request",
            hidden_states=torch.ones(1, 4),
            past_length=3,
            decode_step=True,
        )


@dataclass
class _PageAttentionContext:
    paged_kv_cache: object
    block_ids: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    kv_window_start_abs: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    mm_prefix_ranges: torch.Tensor | None = None


class PageAttentionReadOnly:
    def __init__(self) -> None:
        self.context: _PageAttentionContext | None = None
        self.is_causal = False
        self.has_query_kv_range_abs = True
        self.has_kv_window_start_abs = False
        self.has_kv_valid_length = False
        self.sliding_window = None

    def set_context(self, context: _PageAttentionContext) -> None:
        self.context = context


class PageKVCacheReadOnly(PageAttentionReadOnly):
    pass


class _RingCache:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values
        self.block_size = 1
        self.num_blocks = values.numel()
        self.device = values.device


def test_gemma4_mixed_readonly_binds_fp_sliding_and_full_page_owners() -> None:
    sliding_cache = torch.zeros(2, 8, 1, 1, 2, dtype=torch.float16)
    full_cache = _RingCache(torch.zeros(16))
    bindings = (
        SimpleNamespace(
            layer_idx=0,
            attention_kind="sliding_attention",
            cache_group_idx=1,
            cache_owner_layer_idx=0,
        ),
        SimpleNamespace(
            layer_idx=1,
            attention_kind="full_attention",
            cache_group_idx=2,
            cache_owner_layer_idx=1,
        ),
    )
    target_model = SimpleNamespace(
        page_attention_bindings=bindings,
        placeholder_attn_layers=(
            SimpleNamespace(kv_cache=sliding_cache),
            SimpleNamespace(kv_cache=full_cache),
        ),
    )
    sliding_module = PageKVCacheReadOnly()
    sliding_module.is_causal = True
    sliding_module.has_query_kv_range_abs = False
    sliding_module.layer_idx = 0
    sliding_module.attention_kind = "sliding_attention"
    sliding_module.sliding_window = 4
    full_module = PageAttentionReadOnly()
    full_module.is_causal = True
    full_module.has_query_kv_range_abs = False
    full_module.layer_idx = 1
    full_module.attention_kind = "full_attention"
    modules = {
        "assistant.sliding": sliding_module,
        "assistant.full": full_module,
    }
    graph_module = SimpleNamespace(
        graph=SimpleNamespace(
            nodes=tuple(
                SimpleNamespace(op="call_module", target=name) for name in modules
            )
        ),
        get_submodule=modules.__getitem__,
    )
    backend = HMONNXDraftRuntimeBackend.__new__(HMONNXDraftRuntimeBackend)
    backend.artifact = SimpleNamespace(
        draft_layer_attention_types=(
            "sliding_attention",
            "full_attention",
        ),
        uses_causal_readonly_target_page_cache=False,
        uses_mixed_readonly_target_page_cache=True,
    )
    backend.target_model = target_model
    backend.runner = SimpleNamespace(model_config=SimpleNamespace(max_model_len=16))
    backend._gemma_owner_bindings = None
    backend._gemma_page_bindings = None
    backend._sessions = {"decode": SimpleNamespace(graph_module=graph_module)}
    contexts = {
        1: SimpleNamespace(block_ids=torch.arange(16)),
        2: SimpleNamespace(block_ids=torch.arange(16)),
    }

    with patch(
        "xhquant.xhonnxruntime.parsers.PageAttentionContext",
        _PageAttentionContext,
    ):
        backend._bind_gemma_page_contexts(
            target_visible_length=5,
            page_attention_group_contexts=contexts,
        )

    assert sliding_module.context is not None
    assert sliding_module.context.paged_kv_cache is sliding_cache
    assert sliding_module.context.block_size == 1
    assert full_module.context is not None
    assert full_module.context.paged_kv_cache is full_cache


@pytest.mark.parametrize("causal_lowering", [False, True])
def test_gemma4_accepted_one_binds_readonly_owner_at_sliding_wrap(
    causal_lowering: bool,
) -> None:
    window = 4
    num_drafts = 2
    capacity = window + num_drafts
    old_visible_length = 10

    # The verify block writes [L, L + K]. At the minimum legal ring
    # capacity, L + K wraps onto L - W, which is exactly the entry discarded
    # after the mandatory accepted target step advances visibility to L + 1.
    ring = torch.full((capacity,), -1, dtype=torch.int64)
    for position in range(old_visible_length - window, old_visible_length):
        ring[position % capacity] = position
    for position in range(
        old_visible_length,
        old_visible_length + num_drafts + 1,
    ):
        ring[position % capacity] = position
    owner = _RingCache(ring)
    before_bind = ring.clone()

    owner_layer = 1
    group_idx = 3
    binding = SimpleNamespace(
        layer_idx=owner_layer,
        attention_kind="sliding_attention",
        cache_group_idx=group_idx,
        cache_owner_layer_idx=owner_layer,
    )
    target_model = SimpleNamespace(
        page_attention_bindings=(binding,),
        placeholder_attn_layers=(
            SimpleNamespace(kv_cache=None),
            SimpleNamespace(kv_cache=owner),
        ),
    )
    page_module = PageAttentionReadOnly()
    if causal_lowering:
        page_module.is_causal = True
        page_module.has_query_kv_range_abs = False
        page_module.has_kv_window_start_abs = True
        page_module.has_kv_valid_length = True
        page_module.sliding_window = window
        page_module.layer_idx = 0
        page_module.attention_kind = "sliding_attention"
    graph_module = SimpleNamespace(
        graph=SimpleNamespace(
            nodes=(
                SimpleNamespace(
                    op="call_module",
                    target="assistant.sliding_attention",
                ),
            )
        ),
        get_submodule=lambda target: (
            page_module
            if target == "assistant.sliding_attention"
            else (_ for _ in ()).throw(AttributeError(target))
        ),
    )

    backend = HMONNXDraftRuntimeBackend.__new__(HMONNXDraftRuntimeBackend)
    backend.artifact = SimpleNamespace(
        draft_layer_attention_types=("sliding_attention",),
        uses_causal_readonly_target_page_cache=causal_lowering,
    )
    backend.target_model = target_model
    backend.runner = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=old_visible_length + num_drafts + 1)
    )
    backend._gemma_owner_bindings = None
    backend._gemma_page_bindings = None
    backend._sessions = {"decode": SimpleNamespace(graph_module=graph_module)}

    verify_context = SimpleNamespace(
        # Absolute logical pages map into the compact physical sliding ring.
        block_ids=torch.arange(
            old_visible_length + num_drafts + 1,
            dtype=torch.int32,
        ).remainder(capacity),
        slot_mapping=torch.tensor(
            [
                position % capacity
                for position in range(
                    old_visible_length,
                    old_visible_length + num_drafts + 1,
                )
            ],
            dtype=torch.int64,
        ),
        seq_lens=torch.tensor(
            [old_visible_length + num_drafts + 1],
            dtype=torch.int64,
        ),
        kv_window_start_abs=torch.tensor(
            [old_visible_length - window],
            dtype=torch.int64,
        ),
    )
    accepted_visible_length = old_visible_length + 1
    with patch(
        "xhquant.xhonnxruntime.parsers.PageAttentionContext",
        _PageAttentionContext,
    ):
        backend._bind_gemma_page_contexts(
            target_visible_length=accepted_visible_length,
            page_attention_group_contexts={group_idx: verify_context},
        )

    bound = page_module.context
    assert bound is not None
    assert bound.paged_kv_cache is owner
    assert bound.seq_lens is not None
    assert bound.seq_lens.tolist() == [accepted_visible_length]
    assert bound.kv_window_start_abs is None
    assert bound.block_ids.data_ptr() != verify_context.block_ids.data_ptr()
    assert bound.block_ids[:accepted_visible_length].tolist() == (
        verify_context.block_ids[:accepted_visible_length].to(torch.int64).tolist()
    )
    assert bound.slot_mapping.tolist() == [-1]
    if causal_lowering:
        assert (
            backend._gemma_page_bindings["sliding_attention"].sliding_window == window
        )
    torch.testing.assert_close(owner.values, before_bind)

    # A later accepted length updates the stable capture buffers in-place
    # instead of rebinding the module or the target cache.
    context_id = id(bound)
    block_ids_ptr = bound.block_ids.data_ptr()
    seq_lens_ptr = bound.seq_lens.data_ptr()
    with patch(
        "xhquant.xhonnxruntime.parsers.PageAttentionContext",
        _PageAttentionContext,
    ):
        backend._bind_gemma_page_contexts(
            target_visible_length=accepted_visible_length + 1,
            page_attention_group_contexts={group_idx: verify_context},
        )
    assert id(page_module.context) == context_id
    assert page_module.context.block_ids.data_ptr() == block_ids_ptr
    assert page_module.context.seq_lens.data_ptr() == seq_lens_ptr
    assert page_module.context.seq_lens.tolist() == [accepted_visible_length + 1]

    accepted_slots = {
        position % capacity
        for position in range(
            accepted_visible_length - window,
            accepted_visible_length,
        )
    }
    rejected_slots = {
        position % capacity
        for position in range(
            accepted_visible_length,
            old_visible_length + num_drafts + 1,
        )
    }
    assert accepted_slots.isdisjoint(rejected_slots)
    assert sorted(owner.values[list(accepted_slots)].tolist()) == [
        7,
        8,
        9,
        10,
    ]


@pytest.mark.parametrize(
    ("target_visible_length", "expected"),
    [
        (3, (0, 3)),
        (4, (0, 4)),
        (5, (1, 4)),
        (9, (5, 4)),
        # Accepted-one rollback from a longer verify tail exposes only the new
        # committed suffix; rejected pages remain physically present but hidden.
        (6, (2, 4)),
    ],
)
def test_gemma4_readonly_sliding_range_hides_rejected_tail(
    target_visible_length: int,
    expected: tuple[int, int],
) -> None:
    assert _gemma_readonly_sliding_cache_range(target_visible_length, 4) == expected


def test_gemma4_draft_reuses_one_compact_range_for_all_proposals() -> None:
    class DraftSession:
        def __init__(self) -> None:
            self.inputs = {
                "input_1": _InputInfo((1, 1, 8), torch.float16),
                "valid_length": _InputInfo((1,), torch.int32),
                "current_length": _InputInfo((1,), torch.int32),
                "kv_window_start_abs": _InputInfo((1,), torch.int32),
                "kv_valid_length": _InputInfo((1,), torch.int32),
            }
            self.feeds: list[dict[str, torch.Tensor]] = []

        def get_input_names(self) -> list[str]:
            return list(self.inputs)

        def get_input(self, name: str) -> _InputInfo:
            return self.inputs[name]

        @staticmethod
        def get_output_names() -> list[str]:
            return ["logits", "assistant_hidden_state"]

        def run(
            self,
            feed: dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.feeds.append(feed)
            logits = torch.zeros(1, 1, 8, dtype=torch.float16)
            logits[..., 3] = 1
            hidden = torch.zeros(1, 1, 4, dtype=torch.float16)
            return logits, hidden

    session = DraftSession()
    backend = HMONNXDraftRuntimeBackend.__new__(HMONNXDraftRuntimeBackend)
    backend.device = torch.device("cpu")
    backend.artifact = SimpleNamespace(uses_causal_readonly_target_page_cache=True)
    backend.target_model = SimpleNamespace(
        embed_input_ids=lambda token_ids: torch.zeros(
            token_ids.shape[0],
            token_ids.shape[1],
            4,
            dtype=torch.float16,
        )
    )
    backend._sessions = {"decode": session}
    backend._gemma_page_bindings = {
        "sliding_attention": SimpleNamespace(sliding_window=4)
    }
    backend._bind_gemma_page_contexts = lambda **_kwargs: None
    backend._assert_private_cache_pointers = lambda: None

    tokens = backend.draft_gemma4(
        req_id="request",
        current_token_id=2,
        target_hidden=torch.zeros(1, 4, dtype=torch.float16),
        target_visible_length=5,
        num_draft_tokens=2,
        page_attention_group_contexts={},
    )

    assert tokens == [3, 3]
    assert len(session.feeds) == 2
    for feed in session.feeds:
        assert feed["valid_length"].tolist() == [4]
        assert feed["current_length"].tolist() == [1]
        assert feed["kv_window_start_abs"].tolist() == [1]
        assert feed["kv_valid_length"].tolist() == [4]
    for name in (
        "valid_length",
        "current_length",
        "kv_window_start_abs",
        "kv_valid_length",
    ):
        assert session.feeds[0][name].data_ptr() == session.feeds[1][name].data_ptr()


def test_gemma4_mixed_draft_builds_dense_sliding_mask() -> None:
    class DraftSession:
        def __init__(self) -> None:
            self.inputs = {
                "input_1": _InputInfo((1, 1, 8), torch.float16),
                "valid_length": _InputInfo((1,), torch.int32),
                "current_length": _InputInfo((1,), torch.int32),
                "sliding_attention_mask": _InputInfo((1, 1, 1, 8), torch.float16),
            }
            self.feeds: list[dict[str, torch.Tensor]] = []

        def get_input_names(self) -> list[str]:
            return list(self.inputs)

        def get_input(self, name: str) -> _InputInfo:
            return self.inputs[name]

        @staticmethod
        def get_output_names() -> list[str]:
            return ["logits", "assistant_hidden_state"]

        def run(
            self,
            feed: dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.feeds.append(feed)
            logits = torch.zeros(1, 1, 8, dtype=torch.float16)
            logits[..., 3] = 1
            hidden = torch.zeros(1, 1, 4, dtype=torch.float16)
            return logits, hidden

    session = DraftSession()
    backend = HMONNXDraftRuntimeBackend.__new__(HMONNXDraftRuntimeBackend)
    backend.device = torch.device("cpu")
    backend.artifact = SimpleNamespace(
        uses_causal_readonly_target_page_cache=False,
        uses_mixed_readonly_target_page_cache=True,
    )
    backend.target_model = SimpleNamespace(
        embed_input_ids=lambda token_ids: torch.zeros(
            token_ids.shape[0],
            token_ids.shape[1],
            4,
            dtype=torch.float16,
        )
    )
    backend._sessions = {"decode": session}
    backend._gemma_page_bindings = {
        "sliding_attention": SimpleNamespace(sliding_window=4)
    }
    backend._bind_gemma_page_contexts = lambda **_kwargs: None
    backend._assert_private_cache_pointers = lambda: None

    tokens = backend.draft_gemma4(
        req_id="request",
        current_token_id=2,
        target_hidden=torch.zeros(1, 4, dtype=torch.float16),
        target_visible_length=3,
        num_draft_tokens=2,
        page_attention_group_contexts={},
    )

    assert tokens == [3, 3]
    assert len(session.feeds) == 2
    expected_min = torch.finfo(torch.float16).min
    for feed in session.feeds:
        assert feed["sliding_attention_mask"].flatten().tolist() == [
            0.0,
            0.0,
            0.0,
            expected_min,
            expected_min,
            expected_min,
            expected_min,
            expected_min,
        ]
    assert (
        session.feeds[0]["sliding_attention_mask"].data_ptr()
        == session.feeds[1]["sliding_attention_mask"].data_ptr()
    )
