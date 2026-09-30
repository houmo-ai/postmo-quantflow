from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_merak.model_executor.model_loader.merak_linear_checkpoint_store import (
    LinearStateLayerSpec,
    MerakLinearCheckpointStore,
)
from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1


def test_runner_checkpoint_store_adopts_placeholder_storage_without_allocating() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    layer0_states = (
        torch.zeros((1, 2), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float16),
    )
    layer1_states = (
        torch.zeros((1, 4), dtype=torch.float32),
        torch.zeros((1, 5), dtype=torch.float16),
    )
    runner.model = SimpleNamespace(
        placeholder_linear_state_layers=[
            SimpleNamespace(layer_name="linear.0", kv_cache=layer0_states),
            SimpleNamespace(layer_name="linear.1", kv_cache=layer1_states),
        ]
    )
    runner.scheduler_config = SimpleNamespace(max_num_seqs=1)
    runner._merak_layer_group_ids = [0, 1]

    with patch(
        "vllm_merak.model_executor.model_loader.merak_linear_checkpoint_store.torch.zeros",
        side_effect=AssertionError("runner checkpoint store allocated duplicate state"),
    ):
        runner._initialize_linear_checkpoint_store()

    store = runner.merak_linear_checkpoint_store
    assert store.storage_for_layer("linear.0") == layer0_states
    assert store.storage_for_layer("linear.1") == layer1_states
    assert store.storage_pointers() == (
        ("linear.0", 0, layer0_states[0].data_ptr()),
        ("linear.0", 1, layer0_states[1].data_ptr()),
        ("linear.1", 0, layer1_states[0].data_ptr()),
        ("linear.1", 1, layer1_states[1].data_ptr()),
    )


def test_runner_checkpoint_store_is_disabled_without_linear_placeholders() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.model = SimpleNamespace(placeholder_linear_state_layers=[])

    runner._initialize_linear_checkpoint_store()

    assert runner.merak_linear_checkpoint_store is None


def test_serial_runtime_linear_owner_is_released_when_request_leaves_batch() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    store = _single_layer_store()
    _commit_request(store, "finished", 11)
    runner.merak_linear_checkpoint_store = store
    runner.merak_linear_state_idx = {"finished": 0}
    runner._merak_runtime_linear_owner_req_id = "finished"
    runner.input_batch = SimpleNamespace(req_ids=["finished"])

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_new_reqs = [
        SimpleNamespace(req_id="next", block_ids=[])
    ]
    scheduler_output.num_scheduled_tokens = {"next": 1}
    scheduler_output.total_num_scheduled_tokens = 1

    def update_parent(_scheduler_output):
        runner.input_batch.req_ids = ["next"]
        return "updated"

    with patch(
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states",
        side_effect=update_parent,
    ):
        assert runner._update_states(scheduler_output) == "updated"

    assert runner._merak_runtime_linear_owner_req_id is None


def test_update_states_invalidates_reused_blocks_and_releases_finished_requests() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.merak_linear_state_idx = {"finished": 0, "resumed": 1, "active": 2}
    runner.input_batch = SimpleNamespace(req_ids=["resumed", "active"])
    events = []

    class FakeStore:
        group_layer_names = {0: ("linear.0",), 2: ("linear.1",)}
        active_request_ids = ()

        @staticmethod
        def invalidate_physical_ids(group_physical_ids):
            events.append(("invalidate", dict(group_physical_ids)))

        @staticmethod
        def release_request(request_id):
            events.append(("release", request_id))

    runner.merak_linear_checkpoint_store = FakeStore()
    output = SimpleNamespace(
        finished_req_ids={"finished"},
        preempted_req_ids=set(),
        new_block_ids_to_zero=[11, 22],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=["resumed"]),
    )

    with patch(
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states",
        return_value="updated",
    ):
        assert runner._update_states(output) == "updated"

    assert events == [
        ("invalidate", {0: (11, 22), 2: (11, 22)}),
        ("release", "finished"),
    ]
    assert runner.merak_linear_state_idx == {"active": 2}


def test_update_states_rejects_phase1_preemption_before_mutating_parent_state() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.merak_linear_state_idx = {"req": 0}
    runner.merak_linear_checkpoint_store = SimpleNamespace()
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids={"req"},
        new_block_ids_to_zero=[],
        scheduled_cached_reqs=None,
    )

    with patch(
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states",
        side_effect=AssertionError("parent state mutated before fail-fast"),
    ):
        with pytest.raises(RuntimeError, match="Phase 1.*preemption"):
            runner._update_states(output)

    assert runner.merak_linear_state_idx == {"req": 0}


def _single_layer_store(*, capacity: int = 1) -> MerakLinearCheckpointStore:
    return MerakLinearCheckpointStore(
        (
            LinearStateLayerSpec(
                layer_name="linear.0",
                group_id=0,
                shapes=((2,),),
                dtypes=(torch.float32,),
            ),
        ),
        max_num_seqs=capacity,
    )


def _commit_request(
    store: MerakLinearCheckpointStore,
    request_id: str,
    physical_id: int,
) -> None:
    store.acquire(request_id, {0: physical_id}, past_len=0)
    store.store_layer_state(
        request_id,
        "linear.0",
        (torch.tensor([1.0, 2.0]),),
    )
    store.transition(request_id, {0: physical_id})


def test_update_states_reclaims_finished_owner_missing_from_next_real_scheduler_output() -> None:
    """Reproduce the sequential 9B request SchedulerOutput lifecycle."""
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    store = _single_layer_store()
    _commit_request(store, "first", 11)
    runner.merak_linear_checkpoint_store = store
    runner.merak_linear_state_idx = {"first": 0}
    runner.input_batch = SimpleNamespace(req_ids=["first"])

    # The finish-only scheduler step was not forwarded to the worker. The next
    # real model-execution output contains only the fresh request, exactly as
    # in the production failure dump, and therefore has no finished_req_ids.
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_new_reqs = [
        SimpleNamespace(req_id="second", block_ids=[])
    ]
    scheduler_output.num_scheduled_tokens = {"second": 1}
    scheduler_output.total_num_scheduled_tokens = 1
    assert scheduler_output.finished_req_ids == set()

    def update_parent(_scheduler_output):
        runner.input_batch.req_ids = ["second"]
        return "updated"

    with patch(
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states",
        side_effect=update_parent,
    ):
        assert runner._update_states(scheduler_output) == "updated"

    assert store.active_request_ids == ()
    assert runner.merak_linear_state_idx == {}
    # The sequential request can immediately reuse the sole compact slot.
    second_slot = store.acquire("second", {0: 12}, past_len=0)
    assert second_slot.index == 0


def test_update_states_keeps_active_and_resumed_checkpoint_owners() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    store = _single_layer_store(capacity=2)
    _commit_request(store, "active", 11)
    _commit_request(store, "resumed", 12)
    runner.merak_linear_checkpoint_store = store
    runner.merak_linear_state_idx = {"active": 0, "resumed": 0}
    runner.input_batch = SimpleNamespace(req_ids=[])

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_cached_reqs.resumed_req_ids = {"resumed"}

    def update_parent(_scheduler_output):
        runner.input_batch.req_ids = ["active", "resumed"]
        return "updated"

    with patch(
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states",
        side_effect=update_parent,
    ):
        runner._update_states(scheduler_output)

    assert store.active_request_ids == ("active", "resumed")
