import pytest
import torch

from hybrid_page_padding_testkit import (
    assert_states_exact,
    recognizable_states,
)
from vllm_merak.model_executor.model_loader.merak_linear_checkpoint_store import (
    LinearStateLayerSpec,
    MerakLinearCheckpointStore,
)


def _layer_specs(*, groups: int = 3, layers_per_group: int = 2):
    example = recognizable_states(0)
    shapes = tuple(tuple(tensor.shape) for tensor in example)
    dtypes = tuple(tensor.dtype for tensor in example)
    return tuple(
        LinearStateLayerSpec(
            layer_name=f"linear.{group_id}.{layer_idx}",
            group_id=group_id,
            shapes=shapes,
            dtypes=dtypes,
            device=torch.device("cpu"),
        )
        for group_id in range(groups)
        for layer_idx in range(layers_per_group)
    )


def _physical_ids(base: int) -> dict[int, int]:
    return {0: base, 1: base + 10, 2: base + 20}


def _scatter_all_layers(
    store: MerakLinearCheckpointStore,
    request_id: str,
    *,
    value: int,
) -> dict[str, tuple[torch.Tensor, ...]]:
    expected = {}
    for layer_idx, layer_name in enumerate(store.layer_names):
        states = recognizable_states(value + layer_idx * 10)
        store.store_layer_state(request_id, layer_name, states)
        expected[layer_name] = states
    return expected


def _assert_slot_matches(
    store: MerakLinearCheckpointStore,
    request_id: str,
    expected: dict[str, tuple[torch.Tensor, ...]],
) -> None:
    for layer_name, expected_states in expected.items():
        assert_states_exact(
            store.layer_state(request_id, layer_name),
            expected_states,
        )


def test_sparse_linear_store_abc_resume_and_reuse_lifecycle() -> None:
    """Runner-level A/B/C lifecycle; not service preemption support."""
    store = MerakLinearCheckpointStore(_layer_specs(), max_num_seqs=2)
    a_old = _physical_ids(11)
    b_ids = _physical_ids(101)

    a_slot = store.acquire("A", a_old, past_len=0)
    a_initial = _scatter_all_layers(store, "A", value=10)
    store.transition("A", a_old)

    b_slot = store.acquire("B", b_ids, past_len=0)
    _scatter_all_layers(store, "B", value=70)
    store.transition("B", b_ids)
    assert a_slot.index != b_slot.index

    # A lost its runner-local logical index. Scheduler physical IDs remain the
    # sole source of truth and recover the same complete slot.
    resumed = store.load_slot("A", a_old, past_len=5)
    assert resumed.index == a_slot.index
    _assert_slot_matches(store, "A", a_initial)

    a_new = _physical_ids(12)
    a_updated = _scatter_all_layers(store, "A", value=110)
    store.transition("A", a_new)
    assert store.load_slot("A", a_new, past_len=6).index == a_slot.index
    _assert_slot_matches(store, "A", a_updated)
    for group_id, physical_id in a_old.items():
        assert not store.has_entry(group_id, physical_id)

    # B finishes; C reuses B's scheduler IDs and freed slot. Acquisition must
    # zero every layer state before C can write anything.
    store.release_request("B")
    c_slot = store.acquire("C", b_ids, past_len=0)
    assert c_slot.index == b_slot.index
    for layer_name in store.layer_names:
        for state in store.layer_state("C", layer_name):
            assert torch.count_nonzero(state).item() == 0
    c_states = _scatter_all_layers(store, "C", value=0)
    store.transition("C", b_ids)
    _assert_slot_matches(store, "C", c_states)


def test_sparse_linear_store_past_len_missing_checkpoint_fails_closed() -> None:
    store = MerakLinearCheckpointStore(_layer_specs(), max_num_seqs=1)

    with pytest.raises(
        RuntimeError,
        match=r"past_len=5.*missing complete linear checkpoint.*group 0.*physical ID 11",
    ):
        store.load_slot("A", _physical_ids(11), past_len=5)

    assert store.active_request_count == 0
    assert store.entry_count == 0


def test_sparse_linear_store_requires_complete_group_entry() -> None:
    store = MerakLinearCheckpointStore(_layer_specs(), max_num_seqs=1)
    physical_ids = _physical_ids(11)
    store.acquire("A", physical_ids, past_len=0)
    store.store_layer_state("A", "linear.0.0", recognizable_states(10))

    with pytest.raises(
        RuntimeError,
        match=r"checkpoint is incomplete.*linear\.0\.1",
    ):
        store.transition("A", physical_ids)

    assert not store.is_complete(0, 11)


def test_sparse_linear_store_reuse_requires_explicit_invalidation_or_release() -> None:
    store = MerakLinearCheckpointStore(_layer_specs(), max_num_seqs=2)
    reused_ids = _physical_ids(11)
    store.acquire("B", reused_ids, past_len=0)
    _scatter_all_layers(store, "B", value=50)
    store.transition("B", reused_ids)

    with pytest.raises(
        RuntimeError,
        match=r"group 0.*physical ID 11.*owned by request B",
    ):
        store.acquire("C", reused_ids, past_len=0)
    with pytest.raises(RuntimeError, match=r"belongs to request B, not C"):
        store.load_slot("C", reused_ids, past_len=1)

    store.invalidate_physical_ids({0: [11], 1: [21], 2: [31]})
    assert store.active_request_count == 0
    assert store.entry_count == 0
    store.acquire("C", reused_ids, past_len=0)


def test_sparse_linear_store_transition_is_atomic_on_collision() -> None:
    store = MerakLinearCheckpointStore(_layer_specs(), max_num_seqs=2)
    a_ids = _physical_ids(11)
    b_ids = _physical_ids(101)
    for request_id, physical_ids, value in (("A", a_ids, 10), ("B", b_ids, 50)):
        store.acquire(request_id, physical_ids, past_len=0)
        _scatter_all_layers(store, request_id, value=value)
        store.transition(request_id, physical_ids)

    colliding = dict(a_ids)
    colliding[2] = b_ids[2]
    _scatter_all_layers(store, "A", value=90)
    with pytest.raises(RuntimeError, match=r"group 2.*owned by request B"):
        store.transition("A", colliding)

    # Failed transition preserves the old key mapping, but it remains
    # deliberately incomplete because the slot state was already updated.
    assert all(
        store.has_entry(group_id, physical_id)
        for group_id, physical_id in a_ids.items()
    )
    assert not any(
        store.is_complete(group_id, physical_id)
        for group_id, physical_id in a_ids.items()
    )
    with pytest.raises(RuntimeError, match="missing complete linear checkpoint"):
        store.load_slot("A", a_ids, past_len=5)


def test_sparse_linear_store_rejects_shape_dtype_and_layer_order_mismatch() -> None:
    store = MerakLinearCheckpointStore(_layer_specs(), max_num_seqs=1)
    store.acquire("A", _physical_ids(11), past_len=0)
    correct = list(recognizable_states(10))

    wrong_shape = list(correct)
    wrong_shape[0] = torch.zeros((3, 2), dtype=wrong_shape[0].dtype)
    with pytest.raises(
        ValueError, match=r"linear\.0\.0 state 0 shape.*expected \(2, 2\)"
    ):
        store.store_layer_state("A", "linear.0.0", tuple(wrong_shape))

    wrong_dtype = list(correct)
    wrong_dtype[3] = wrong_dtype[3].to(torch.float16)
    with pytest.raises(
        ValueError, match=r"linear\.0\.0 state 3 dtype.*expected torch.float32"
    ):
        store.store_layer_state("A", "linear.0.0", tuple(wrong_dtype))

    with pytest.raises(KeyError, match="unknown linear layer"):
        store.store_layer_state("A", "linear.unknown", tuple(correct))


def test_sparse_linear_store_thirty_phase1_reuse_cycles_are_bounded() -> None:
    store = MerakLinearCheckpointStore(_layer_specs(), max_num_seqs=1)
    allocated_bytes = store.allocated_bytes
    storage_pointers = store.storage_pointers()

    for cycle in range(30):
        request_id = f"request-{cycle}"
        physical_ids = _physical_ids(100 + cycle)
        store.acquire(request_id, physical_ids, past_len=0)
        _scatter_all_layers(store, request_id, value=cycle * 10)
        store.transition(request_id, physical_ids)
        assert store.active_request_count == 1
        assert store.entry_count == 3
        assert store.slot_capacity == 1
        store.release_request(request_id)
        assert store.active_request_count == 0
        assert store.entry_count == 0

    assert store.allocated_bytes == allocated_bytes
    assert store.storage_pointers() == storage_pointers


def test_sparse_linear_store_slot_capacity_is_a_hard_bound() -> None:
    store = MerakLinearCheckpointStore(_layer_specs(), max_num_seqs=1)
    store.acquire("A", _physical_ids(11), past_len=0)

    with pytest.raises(
        RuntimeError, match=r"slots are exhausted.*active=1, capacity=1"
    ):
        store.acquire("B", _physical_ids(101), past_len=0)

    assert store.active_request_count == 1
    assert store.slot_capacity == 1


def test_partial_physical_invalidation_makes_resume_fail_closed() -> None:
    store = MerakLinearCheckpointStore(_layer_specs(), max_num_seqs=1)
    ids = _physical_ids(11)
    store.acquire("A", ids, past_len=0)
    _scatter_all_layers(store, "A", value=10)
    store.transition("A", ids)

    store.invalidate_physical_ids({1: [ids[1]]})

    assert store.active_request_count == 1
    assert store.entry_count == 2
    with pytest.raises(
        RuntimeError,
        match=r"past_len=5.*missing complete linear checkpoint.*group 1.*physical ID 21",
    ):
        store.load_slot("A", ids, past_len=5)


def test_sparse_linear_store_allocates_slots_not_global_num_blocks() -> None:
    specs = _layer_specs(groups=3, layers_per_group=1)
    store = MerakLinearCheckpointStore(specs, max_num_seqs=1)
    expected = sum(
        tensor.numel() * tensor.element_size() for tensor in recognizable_states(0)
    ) * len(specs)

    assert store.allocated_bytes == expected
    for layer_name in store.layer_names:
        assert all(
            tensor.shape[0] == 1 for tensor in store.storage_for_layer(layer_name)
        )

    with pytest.raises(TypeError, match="num_blocks"):
        MerakLinearCheckpointStore(specs, max_num_seqs=1, num_blocks=4_304)


def test_sparse_linear_store_adopts_bound_placeholder_storage_without_copy() -> None:
    specs = _layer_specs(groups=1, layers_per_group=2)
    storage_by_layer = {
        spec.layer_name: tuple(
            torch.empty((2, *shape), dtype=dtype)
            for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True)
        )
        for spec in specs
    }

    store = MerakLinearCheckpointStore(
        specs,
        max_num_seqs=2,
        storage_by_layer=storage_by_layer,
    )

    assert store.allocated_bytes == sum(
        tensor.numel() * tensor.element_size()
        for states in storage_by_layer.values()
        for tensor in states
    )
    for layer_name, external_states in storage_by_layer.items():
        adopted_states = store.storage_for_layer(layer_name)
        assert all(
            adopted is external
            for adopted, external in zip(adopted_states, external_states, strict=True)
        )
    store.acquire("A", {0: 11}, past_len=0)
    store.store_layer_state("A", specs[0].layer_name, recognizable_states(10))
    assert_states_exact(
        tuple(tensor[0] for tensor in storage_by_layer[specs[0].layer_name]),
        recognizable_states(10),
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing_layer", "storage layer set mismatch"),
        ("wrong_order", "storage layer order mismatch"),
        ("state_count", "state count"),
        ("leading_slots", "shape.*expected"),
        ("state_shape", "shape.*expected"),
        ("dtype", "dtype.*expected"),
        ("device", "device.*expected"),
    ],
)
def test_sparse_linear_store_validates_bound_placeholder_storage(
    mutation: str,
    error: str,
) -> None:
    specs = _layer_specs(groups=1, layers_per_group=2)
    storage_by_layer = {
        spec.layer_name: tuple(
            torch.empty((2, *shape), dtype=dtype)
            for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True)
        )
        for spec in specs
    }
    if mutation == "missing_layer":
        storage_by_layer.pop(specs[1].layer_name)
    elif mutation == "wrong_order":
        storage_by_layer = dict(reversed(tuple(storage_by_layer.items())))
    elif mutation == "state_count":
        storage_by_layer[specs[0].layer_name] = storage_by_layer[specs[0].layer_name][
            :-1
        ]
    elif mutation == "leading_slots":
        states = list(storage_by_layer[specs[0].layer_name])
        states[0] = torch.empty((1, *specs[0].shapes[0]), dtype=specs[0].dtypes[0])
        storage_by_layer[specs[0].layer_name] = tuple(states)
    elif mutation == "state_shape":
        states = list(storage_by_layer[specs[0].layer_name])
        states[0] = torch.empty((2, 3, 2), dtype=specs[0].dtypes[0])
        storage_by_layer[specs[0].layer_name] = tuple(states)
    elif mutation == "dtype":
        states = list(storage_by_layer[specs[0].layer_name])
        states[0] = states[0].to(torch.float32)
        storage_by_layer[specs[0].layer_name] = tuple(states)
    else:
        specs = (
            LinearStateLayerSpec(
                layer_name=specs[0].layer_name,
                group_id=specs[0].group_id,
                shapes=specs[0].shapes,
                dtypes=specs[0].dtypes,
                device=torch.device("meta"),
            ),
            specs[1],
        )

    with pytest.raises((TypeError, ValueError), match=error):
        MerakLinearCheckpointStore(
            specs,
            max_num_seqs=2,
            storage_by_layer=storage_by_layer,
        )


@pytest.mark.parametrize(
    ("linear_layers", "expected_bytes"),
    [(30, 33_423_360), (24, 26_738_688)],
)
def test_real_qwen_geometry_allocates_one_linear_page_per_layer_slot(
    linear_layers: int,
    expected_bytes: int,
) -> None:
    shapes = (
        (2_048, 4),
        (2_048, 4),
        (4_096, 4),
        (32, 128, 128),
    )
    specs = tuple(
        LinearStateLayerSpec(
            layer_name=f"model.layers.{layer_idx}.linear_attn",
            group_id=layer_idx % 3,
            shapes=shapes,
            dtypes=(torch.float16,) * 4,
        )
        for layer_idx in range(linear_layers)
    )

    store = MerakLinearCheckpointStore(specs, max_num_seqs=1)

    assert store.allocated_bytes == expected_bytes
    assert store.slot_capacity == 1
    assert all(
        tensor.shape[0] == 1
        for layer_name in store.layer_names
        for tensor in store.storage_for_layer(layer_name)
    )
