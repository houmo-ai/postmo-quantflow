from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch

from hybrid_page_padding_testkit import assert_states_exact, recognizable_states
from vllm_merak.model_executor.model_loader.merak_for_causal_lm import MerakForCausalLM
from vllm_merak.model_executor.model_loader.merak_linear_checkpoint_store import (
    LinearCheckpointSlot,
    LinearStateLayerSpec,
    MerakLinearCheckpointStore,
)
from xhmodel_merak.xh_llm.models.qwen3_5.hybrid_cache_runtime import (
    commit_hybrid_cache_outputs,
)


class _ForbiddenDenseCache:
    def __len__(self) -> int:
        raise AssertionError("the sparse bridge must not inspect placeholder.kv_cache")

    def __getitem__(self, index: object) -> object:
        del index
        raise AssertionError("the sparse bridge must not index placeholder.kv_cache")


def _layer_spec(layer_name: str, *, group_id: int) -> LinearStateLayerSpec:
    states = recognizable_states(0)
    return LinearStateLayerSpec(
        layer_name=layer_name,
        group_id=group_id,
        shapes=tuple(tuple(state.shape) for state in states),
        dtypes=tuple(state.dtype for state in states),
    )


def _store(
    layer_names: tuple[str, ...] = ("linear.0", "linear.1"),
) -> MerakLinearCheckpointStore:
    return MerakLinearCheckpointStore(
        [_layer_spec(name, group_id=index) for index, name in enumerate(layer_names)],
        max_num_seqs=1,
    )


def _model(
    layer_names: tuple[str, ...] = ("linear.0", "linear.1"),
    *,
    values: tuple[int, ...] = (10, 20),
) -> MerakForCausalLM:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    torch.nn.Module.__init__(model)
    model.placeholder_linear_state_layers = [
        SimpleNamespace(layer_name=name, kv_cache=_ForbiddenDenseCache())
        for name in layer_names
    ]
    runtime_states = [recognizable_states(value) for value in values]
    mixin = SimpleNamespace(
        past_conv_caches=[
            tuple(state.unsqueeze(0) for state in states[:3])
            for states in runtime_states
        ],
        past_recurrent_states=[states[3].unsqueeze(0) for states in runtime_states],
    )
    model.inference_model = SimpleNamespace(get_kvcache_mixin=lambda: mixin)
    return model


def _runtime_states(model: MerakForCausalLM) -> dict[str, tuple[torch.Tensor, ...]]:
    result = {}
    for layer_name, conv_triple, recurrent in model._iter_runtime_linear_state_layers():
        result[layer_name] = tuple(
            (*[tensor[0] for tensor in conv_triple], recurrent[0])
        )
    return result


def _complete_slot(
    store: MerakLinearCheckpointStore,
    *,
    request_id: str = "request-0",
    values: tuple[int, ...] = (100, 200),
) -> LinearCheckpointSlot:
    physical_ids = {group_id: group_id + 10 for group_id in store.group_layer_names}
    store.acquire(request_id, physical_ids, past_len=0)
    for layer_name, value in zip(store.layer_names, values, strict=True):
        store.store_layer_state(request_id, layer_name, recognizable_states(value))
    store.transition(request_id, physical_ids)
    return store.load_slot(request_id, physical_ids, past_len=1)


def test_loads_sparse_slot_into_runtime_in_placeholder_order_without_dense_indices() -> (
    None
):
    model = _model()
    store = _store()
    slot = _complete_slot(store)

    model._load_runtime_linear_state_from_store(slot)

    actual = _runtime_states(model)
    for layer_name, expected in slot.states_by_layer.items():
        assert_states_exact(actual[layer_name], expected)


def test_scatters_runtime_into_sparse_store_without_dense_indices() -> None:
    model = _model(values=(31, 41))
    store = _store()
    physical_ids = {0: 51, 1: 61}
    store.acquire("request-0", physical_ids, past_len=0)

    model._scatter_runtime_linear_state_to_store(store, "request-0")
    slot = store.transition("request-0", physical_ids)

    expected_by_layer = _runtime_states(model)
    for layer_name, expected in expected_by_layer.items():
        assert_states_exact(slot.states_by_layer[layer_name], expected)


def test_fused_prefill_in_place_recurrent_mutation_reaches_checkpoint_store() -> (
    None
):
    """The no-output GDRChunkScan state remains request-persistent in vLLM."""

    model = _model(("linear.0",), values=(0,))
    runtime = model.inference_model
    mixin = runtime.get_kvcache_mixin()
    mixin.split_conv_cache = True
    runtime._kvcache_mixin = mixin
    runtime.meta_info = SimpleNamespace(
        model_config={"prefill_recurrent_state_uses_cache": True},
        spec_decode=None,
    )
    runtime.is_prefill = lambda: True
    output_names = [
        "logits",
        "conv_cache_out_q_0",
        "conv_cache_out_k_0",
        "conv_cache_out_v_0",
    ]
    runtime.prefill_model = SimpleNamespace(
        hmonnx_session=SimpleNamespace(
            get_output_names=lambda: output_names,
        )
    )

    # GDRChunkScan writes the recurrent CacheTensor input directly.  The graph
    # only returns logits and the three convolution-cache outputs.
    mixin.past_recurrent_states[0].fill_(91)
    conv_outputs = [tensor.clone().fill_(value) for tensor, value in zip(
        mixin.past_conv_caches[0],
        (11, 21, 31),
        strict=True,
    )]
    result = commit_hybrid_cache_outputs(
        runtime,
        [torch.zeros(1), *conv_outputs],
        model_label="Qwen fused prefill integration",
    )

    assert result[2] == []
    store = _store(("linear.0",))
    physical_ids = {0: 51}
    store.acquire("request-0", physical_ids, past_len=0)
    model._scatter_runtime_linear_state_to_store(store, "request-0")
    slot = store.transition("request-0", physical_ids)

    stored = slot.states_by_layer["linear.0"]
    assert torch.all(stored[0] == 11)
    assert torch.all(stored[1] == 21)
    assert torch.all(stored[2] == 31)
    assert torch.all(stored[3] == 91)


@pytest.mark.parametrize(
    "states_by_layer",
    [
        {"linear.0": recognizable_states(1)},
        {
            "linear.0": recognizable_states(1),
            "linear.1": recognizable_states(2),
            "linear.extra": recognizable_states(3),
        },
    ],
)
def test_load_rejects_layer_set_mismatch_before_mutating_runtime(
    states_by_layer: Mapping[str, tuple[torch.Tensor, ...]],
) -> None:
    model = _model(values=(10, 20))
    before = {
        name: tuple(t.clone() for t in states)
        for name, states in _runtime_states(model).items()
    }
    slot = LinearCheckpointSlot(
        index=0, request_id="request-0", states_by_layer=states_by_layer
    )

    with pytest.raises(RuntimeError, match=r"layer set mismatch.*missing=.*extra="):
        model._load_runtime_linear_state_from_store(slot)

    for name, expected in before.items():
        assert_states_exact(_runtime_states(model)[name], expected)


def test_load_rejects_layer_order_mismatch() -> None:
    model = _model()
    slot = LinearCheckpointSlot(
        index=0,
        request_id="request-0",
        states_by_layer={
            "linear.1": recognizable_states(1),
            "linear.0": recognizable_states(2),
        },
    )

    with pytest.raises(RuntimeError, match="layer order mismatch"):
        model._load_runtime_linear_state_from_store(slot)


def test_load_validates_all_shapes_before_mutating_any_runtime_layer() -> None:
    model = _model(values=(10, 20))
    before = {
        name: tuple(t.clone() for t in states)
        for name, states in _runtime_states(model).items()
    }
    wrong_second = list(recognizable_states(200))
    wrong_second[3] = torch.zeros((1,), dtype=wrong_second[3].dtype)
    slot = LinearCheckpointSlot(
        index=0,
        request_id="request-0",
        states_by_layer={
            "linear.0": recognizable_states(100),
            "linear.1": tuple(wrong_second),
        },
    )

    with pytest.raises(RuntimeError, match=r"linear\.1 state 3 shape mismatch"):
        model._load_runtime_linear_state_from_store(slot)

    for name, expected in before.items():
        assert_states_exact(_runtime_states(model)[name], expected)


def test_scatter_validates_all_dtypes_before_writing_any_store_layer() -> None:
    model = _model(values=(10, 20))
    mixin = model.inference_model.get_kvcache_mixin()
    mixin.past_recurrent_states[1] = mixin.past_recurrent_states[1].to(torch.float16)
    store = _store()
    physical_ids = {0: 51, 1: 61}
    slot = store.acquire("request-0", physical_ids, past_len=0)
    before = {
        name: tuple(t.clone() for t in states)
        for name, states in slot.states_by_layer.items()
    }

    with pytest.raises(RuntimeError, match=r"linear\.1 state 3 dtype mismatch"):
        model._scatter_runtime_linear_state_to_store(store, "request-0")

    for name, expected in before.items():
        assert_states_exact(store.layer_state("request-0", name), expected)
    with pytest.raises(RuntimeError, match="incomplete"):
        store.transition("request-0", physical_ids)


def test_load_requires_exactly_four_states_per_layer() -> None:
    model = _model()
    slot = LinearCheckpointSlot(
        index=0,
        request_id="request-0",
        states_by_layer={
            "linear.0": recognizable_states(1)[:3],
            "linear.1": recognizable_states(2),
        },
    )

    with pytest.raises(RuntimeError, match=r"linear\.0.*exactly 4"):
        model._load_runtime_linear_state_from_store(slot)


def test_scatter_rejects_store_layer_order_mismatch_before_writing() -> None:
    model = _model()
    store = _store(("linear.1", "linear.0"))
    store.acquire("request-0", {0: 51, 1: 61}, past_len=0)

    with pytest.raises(RuntimeError, match="layer order mismatch"):
        model._scatter_runtime_linear_state_to_store(store, "request-0")
