import inspect
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import SlidingWindowSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_merak.constants import HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST
from vllm_merak.hmfp_kv_cache import (
    HMFPFullAttentionSpec,
    HMFPPagedKVCache,
    HMFPSlidingWindowSpec,
    allocate_hmfp_paged_kv_cache,
)
from vllm_merak.model_executor.model_loader.gemma4_cache_contract import (
    Gemma4CacheContract,
    Gemma4LayerCacheBinding,
)
from vllm_merak.model_executor.model_loader.merak_for_causal_lm import MerakForCausalLM
from vllm_merak.worker_v1.merak_model_runner_v1 import (
    MerakModelRunnerV1,
    _validate_merak_narrow_query_length,
)


@pytest.mark.parametrize("query_length", range(1, 17))
def test_merak_packed_w4_narrow_query_contract_accepts_m1_through_m16(
    query_length,
) -> None:
    phase = "decode" if query_length == 1 else "verify"

    _validate_merak_narrow_query_length(phase, query_length)


@pytest.mark.parametrize(
    ("phase", "query_length"),
    [
        ("decode", 0),
        ("decode", 2),
        ("verify", 1),
        ("verify", 17),
    ],
)
def test_merak_packed_w4_narrow_query_contract_fails_closed(
    phase,
    query_length,
) -> None:
    with pytest.raises(RuntimeError, match="packed-W4|exactly one"):
        _validate_merak_narrow_query_length(phase, query_length)


def test_gemma4_prefill_result_accepts_only_final_target_hidden() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.logits_processor = SimpleNamespace(vocab_size=8)
    outputs = {
        "logits": torch.zeros(1, 1, 8),
        "target_hidden_state": torch.arange(4).reshape(1, 1, 4),
    }

    result = model._extract_request_result(
        SimpleNamespace(),
        outputs,
        current_input_length=26,
        hidden_output_name="target_hidden_state",
        require_all_logits=False,
        allow_last_hidden_only=True,
    )

    assert result.logits.shape == (1, 8)
    assert result.hidden_states is not None
    assert result.hidden_states.shape == (1, 4)


def test_non_gemma_prefill_result_rejects_short_target_hidden() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.logits_processor = SimpleNamespace(vocab_size=8)
    outputs = {
        "logits": torch.zeros(1, 1, 8),
        "target_hidden": torch.zeros(1, 1, 4),
    }

    with pytest.raises(RuntimeError, match="hidden output is shorter"):
        model._extract_request_result(
            SimpleNamespace(),
            outputs,
            current_input_length=26,
            hidden_output_name="target_hidden",
            require_all_logits=False,
        )


def test_target_page_verify_transaction_is_owned_until_acceptance() -> None:
    events = []

    class PageAttention:
        def __init__(self, slot_signature):
            self.cache_write_transaction_slot_signature = slot_signature
            self.cache_write_transaction_block_ids = (3, 7)

        def begin_cache_write_transaction(
            self,
            *,
            query_length,
            physical_block_ids=None,
        ):
            events.append(("begin", query_length, physical_block_ids))

        @staticmethod
        def commit_cache_write_transaction(*, accepted_steps):
            events.append(("commit", accepted_steps))

        @staticmethod
        def abort_cache_write_transaction():
            events.append(("abort",))

    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.enable_page_attention = True
    model._active_hmonnx_model = lambda _runtime: object()
    model._find_page_attention_modules = lambda _model: (
        ("page_attention.0", PageAttention((11, 64, 128))),
        ("page_attention.1", PageAttention((11, 64, 128))),
        ("page_attention.2", PageAttention((22, 64, 128))),
    )

    model._begin_page_attention_verify_transaction(
        object(),
        request_id="request",
        query_length=5,
    )
    assert model._page_attention_verify_transaction_request_id == "request"
    model._commit_page_attention_verify_transaction(
        request_id="request",
        accepted_steps=2,
    )

    assert events == [
        ("begin", 5, None),
        ("begin", 5, (3, 7)),
        ("begin", 5, None),
        ("commit", 2),
        ("commit", 2),
        ("commit", 2),
    ]
    assert model._page_attention_verify_transaction_request_id is None


def test_merak_verify_rejection_commits_only_valid_sampled_prefix(
    monkeypatch,
) -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner._merak_pending_verify_req_ids = {"request"}
    runner._merak_pending_page_verifies = {"request": 5}
    runner.input_batch = SimpleNamespace(req_ids=["request"])
    page_commits = []
    runner.model = SimpleNamespace(
        _commit_page_attention_verify_transaction=lambda **kwargs: page_commits.append(
            kwargs
        )
    )
    commits = []
    accepts = []
    runner._commit_pending_merak_linear_verify = lambda req_id, *, accepted_steps: (
        commits.append((req_id, accepted_steps))
    )
    runner._merak_spec_proposer = SimpleNamespace(
        on_target_accept=lambda **kwargs: accepts.append(kwargs)
    )
    monkeypatch.setattr(
        GPUModelRunner,
        "_update_states_after_model_execute",
        lambda _self, _output_token_ids, _scheduler_output: None,
    )

    sampled = torch.tensor([[101, 999, -1, -1, -1]])
    runner._update_states_after_model_execute(
        sampled,
        SimpleNamespace(),
    )

    assert commits == [("request", 2)]
    assert page_commits == [
        {
            "request_id": "request",
            "accepted_steps": 2,
        }
    ]
    assert runner._merak_pending_page_verifies == {}
    assert len(accepts) == 1
    assert accepts[0]["req_id"] == "request"
    assert accepts[0]["accepted_steps"] == 2
    assert torch.equal(accepts[0]["sampled_token_ids"], sampled[0])
    assert runner._merak_pending_verify_req_ids == set()


@pytest.mark.parametrize("runtime_owner", [None, "request"])
def test_merak_verify_rejection_selects_accepted_linear_snapshot(
    runtime_owner: str | None,
) -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runtime = object()
    store_events = []
    model_events = []

    class CheckpointStore:
        def load_slot(self, req_id, physical_ids, *, past_len):
            store_events.append(("load", req_id, physical_ids, past_len))
            return "previous-slot"

        def transition(self, req_id, physical_ids):
            store_events.append(("transition", req_id, physical_ids))

    runner.model = SimpleNamespace(
        inference_model=runtime,
        _load_runtime_linear_state_from_store=lambda slot: model_events.append(
            ("load-runtime", slot)
        ),
        _scatter_runtime_linear_state_to_store=lambda store, req_id: (
            model_events.append(("scatter", store, req_id))
        ),
    )
    runner.merak_linear_checkpoint_store = CheckpointStore()
    runner.merak_linear_state_idx = {}
    runner._merak_runtime_linear_owner_req_id = runtime_owner
    runner._merak_pending_linear_verifies = {
        "request": {
            "past_len": 63,
            "query_len": 5,
            "previous_physical_ids": {1: 7},
            "group_rows": {1: [10, 11, 12]},
            "block_size": 64,
            "num_speculative_blocks": 1,
            "transaction": "verify-transaction",
        }
    }
    hybrid_commits = []
    runner._commit_merak_hybrid_verify_transaction = (
        lambda active_runtime, *, accepted_steps, transaction: hybrid_commits.append(
            (active_runtime, accepted_steps, transaction)
        )
    )

    # One accepted draft plus the target recovery token: commit verify
    # snapshot index 1 and advance to the physical block containing token 64.
    runner._commit_pending_merak_linear_verify(
        "request",
        accepted_steps=2,
    )

    expected_store_events = []
    expected_model_events = []
    if runtime_owner is None:
        expected_store_events.append(("load", "request", {1: 7}, 63))
        expected_model_events.append(("load-runtime", "previous-slot"))
    expected_store_events.append(("transition", "request", {1: 11}))
    expected_model_events.append(
        ("scatter", runner.merak_linear_checkpoint_store, "request")
    )
    assert store_events == expected_store_events
    assert model_events == expected_model_events
    assert hybrid_commits == [
        (runtime, 2, "verify-transaction"),
    ]
    assert runner.merak_linear_state_idx == {"request": 1}
    assert runner._merak_pending_linear_verifies == {}


def _authoritative_gemma4_contract() -> Gemma4CacheContract:
    return Gemma4CacheContract(
        version=2,
        prefill_chunk_size=320,
        bindings=(),
        cache_shapes={},
    )


def _finalized_binding(
    layer_name: str,
    *,
    group_idx: int,
    cache_index: int,
) -> SimpleNamespace:
    is_sliding = layer_name.startswith("sliding.")
    return SimpleNamespace(
        layer_name=layer_name,
        attention_kind=("sliding_attention" if is_sliding else "full_attention"),
        cache_group_idx=group_idx,
        cache_index=cache_index,
        scheduler_block_size=64,
        physical_block_size=64,
        physical_pages_per_scheduler_block=1,
        fixed_physical_pages=(21 if is_sliding else None),
        kv_shape=(1, 1, 1344 if is_sliding else 262144, 64),
    )


def _page_attention_graph(*entries: tuple[str, str]):
    class PageAttention:
        def __init__(self, device: str):
            self.device = device

    nodes = [
        SimpleNamespace(op="call_module", target=target, name=target)
        for target, _device in entries
    ]
    modules = {target: PageAttention(device) for target, device in entries}
    return SimpleNamespace(
        graph=SimpleNamespace(nodes=nodes),
        get_submodule=lambda target: modules[target],
    )


def _indexed_page_attention_graph(*entries: tuple[str, str, int, int]):
    class PageAttention:
        def __init__(self, device: str, layer_idx: int, cache_index: int):
            self.device = device
            self.layer_idx = layer_idx
            self.cache_index = cache_index
            self.attention_kind = ""

    nodes = [
        SimpleNamespace(op="call_module", target=target, name=target)
        for target, _device, _layer_idx, _cache_index in entries
    ]
    modules = {
        target: PageAttention(device, layer_idx, cache_index)
        for target, device, layer_idx, cache_index in entries
    }
    return SimpleNamespace(
        graph=SimpleNamespace(nodes=nodes),
        get_submodule=lambda target: modules[target],
    )


def test_merak_model_runner_uses_gpu_model_runner() -> None:
    assert issubclass(MerakModelRunnerV1, GPUModelRunner)


def test_merak_model_runner_maps_readonly_page_attention_nodes() -> None:
    class PageAttentionReadOnly:
        device = "cuda:3"

    module = PageAttentionReadOnly()
    graph_module = SimpleNamespace(
        graph=SimpleNamespace(
            nodes=[
                SimpleNamespace(
                    op="call_module",
                    target="draft_attention",
                    name="draft_attention",
                )
            ]
        ),
        get_submodule=lambda target: (
            module
            if target == "draft_attention"
            else (_ for _ in ()).throw(AttributeError(target))
        ),
    )
    runtime = SimpleNamespace(hmonnx_session=SimpleNamespace(graph_module=graph_module))

    assert MerakModelRunnerV1._page_attention_node_map(
        runtime,
        torch.device("cpu"),
    ) == [("draft_attention", torch.device("cuda:3"))]


def test_merak_model_runner_maps_mixed_page_kv_cache_nodes() -> None:
    class PageKVCache:
        device = "cuda:2"

    module = PageKVCache()
    graph_module = SimpleNamespace(
        graph=SimpleNamespace(
            nodes=[
                SimpleNamespace(
                    op="call_module",
                    target="sliding_cache",
                    name="sliding_cache",
                )
            ]
        ),
        get_submodule=lambda target: (
            module
            if target == "sliding_cache"
            else (_ for _ in ()).throw(AttributeError(target))
        ),
    )
    runtime = SimpleNamespace(hmonnx_session=SimpleNamespace(graph_module=graph_module))

    assert MerakModelRunnerV1._page_attention_node_map(
        runtime,
        torch.device("cpu"),
    ) == [("sliding_cache", torch.device("cuda:2"))]


def test_merak_model_runner_maps_unbound_flash_attention_for_early_placement() -> None:
    class FlashAttention:
        device = "cuda:2"

    module = FlashAttention()
    graph_module = SimpleNamespace(
        graph=SimpleNamespace(
            nodes=[
                SimpleNamespace(
                    op="call_module",
                    target="target_attention",
                    name="target_attention",
                )
            ]
        ),
        get_submodule=lambda target: (
            module
            if target == "target_attention"
            else (_ for _ in ()).throw(AttributeError(target))
        ),
    )
    runtime = SimpleNamespace(hmonnx_session=SimpleNamespace(graph_module=graph_module))

    assert MerakModelRunnerV1._hmonnx_attention_node_map(
        runtime,
        torch.device("cpu"),
        module_class_names=frozenset(
            {"FlashAttention", "PageAttention", "PageAttentionReadOnly"}
        ),
    ) == [("target_attention", torch.device("cuda:2"))]
    assert (
        MerakModelRunnerV1._page_attention_node_map(
            runtime,
            torch.device("cpu"),
        )
        == []
    )


def test_merak_model_runner_uses_inference_cuda_for_legacy_cpu_metadata() -> None:
    class PageAttention:
        # Legacy parser modules do not expose a device after HMONNXModel.to().
        pass

    modules = [PageAttention(), PageAttention()]
    runtime = SimpleNamespace(
        hmonnx_session=SimpleNamespace(
            node_modules=modules,
            initialize=lambda: None,
            _session=SimpleNamespace(node_modules=modules),
        )
    )

    devices = MerakModelRunnerV1._page_or_flash_attention_node_map(
        runtime, torch.device("cuda:0")
    )
    assert devices == [
        ("PageAttention:index=0", torch.device("cuda:0")),
        ("PageAttention:index=1", torch.device("cuda:0")),
    ]


def test_merak_model_runner_passes_device_to_gpu_model_runner() -> None:
    vllm_config = SimpleNamespace()
    device = torch.device("cuda:1")
    captured = {}

    def fake_init(self, passed_vllm_config, passed_device):
        captured["vllm_config"] = passed_vllm_config
        captured["device"] = passed_device

    with patch.object(GPUModelRunner, "__init__", fake_init):
        runner = MerakModelRunnerV1(vllm_config, "sefp_k8_v8", device=device)  # type: ignore[arg-type]

    assert captured == {"vllm_config": vllm_config, "device": device}
    assert runner.device == device
    assert runner.kv_cache_dtype == "sefp_k8_v8"


def test_merak_model_runner_keeps_logits_on_runner_device() -> None:
    source = inspect.getsource(MerakModelRunnerV1._model_forward)

    assert "device = self.device" in source
    assert "device = flat_inputs.device" not in source


def test_merak_model_runner_tells_model_which_gpu_receives_embeddings() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.vllm_config = SimpleNamespace()
    runner.kv_cache_dtype = "float16"
    runner.device = torch.device("cuda:2")
    received_devices = []
    fake_model = SimpleNamespace(set_runner_device=received_devices.append)

    with patch(
        "vllm_merak.worker_v1.merak_model_runner_v1.get_model",
        return_value=fake_model,
    ):
        runner.load_model()

    assert runner.model is fake_model
    assert received_devices == [torch.device("cuda:2")]


def test_merak_model_runner_registers_gemma_placeholders_before_cache_specs() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.vllm_config = SimpleNamespace()
    runner.kv_cache_dtype = "sefp_k8_v8"
    runner.device = torch.device("cpu")
    hf_config = object()
    registered = []
    active_configs = []

    @contextmanager
    def current_config(config):
        active_configs.append(config)
        try:
            yield
        finally:
            active_configs.pop()

    def register(config):
        assert active_configs == [runner.vllm_config]
        registered.append(config)

    fake_model = SimpleNamespace(
        gemma4_cache_contract=object(),
        placeholder_attn_layers=[],
        config=SimpleNamespace(get_text_config=lambda: hf_config),
        _register_placeholder_attention_layers=register,
    )

    with (
        patch(
            "vllm_merak.worker_v1.merak_model_runner_v1.get_model",
            return_value=fake_model,
        ),
        patch("vllm.config.set_current_vllm_config", current_config),
    ):
        runner.load_model()

    assert registered == [hf_config]


def test_merak_model_runner_keeps_sefp_raw_cache_as_byte_layout() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.kv_cache_dtype = "sefp_k8_v8"
    runner.runner_only_attn_layers = set()

    spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    group = SimpleNamespace(
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        layer_names=["layer.0"],
    )
    runner._kv_cache_spec_attn_group_iterator = lambda: iter([group])

    raw_cache = torch.zeros(spec.page_size_bytes * 3, dtype=torch.int8)

    kv_caches = MerakModelRunnerV1._reshape_kv_cache_tensors(
        runner,
        {"layer.0": raw_cache},
        [64],
    )

    assert tuple(kv_caches["layer.0"].shape) == (3, spec.page_size_bytes)
    assert kv_caches["layer.0"].data_ptr() == raw_cache.data_ptr()


def test_shared_by_layers_reuse_one_raw_tensor_and_typed_handle() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.kv_cache_dtype = "sefp_k8_v8"
    runner.cache_dtype = "sefp_k8_v8"
    runner.scheduler_config = SimpleNamespace(max_num_seqs=2)
    runner.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=64))

    full_spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    sliding_spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    groups = [
        SimpleNamespace(
            kv_cache_spec=full_spec,
            kv_cache_group_id=0,
            layer_names=["full.0"],
        ),
        SimpleNamespace(
            kv_cache_spec=sliding_spec,
            kv_cache_group_id=1,
            layer_names=["sliding.1", "sliding.2"],
        ),
    ]
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(groups)

    def layer(name, spec):
        return SimpleNamespace(
            layer_name=name,
            kv_cache=None,
            num_kv_heads=1,
            head_size=64,
            get_kv_cache_spec=lambda _config: spec,
        )

    layers = [
        layer("full.0", full_spec),
        layer("sliding.1", sliding_spec),
        layer("sliding.2", sliding_spec),
    ]
    runner.model = SimpleNamespace(
        gemma4_cache_contract=_authoritative_gemma4_contract(),
        page_attention_bindings=(
            _finalized_binding("full.0", group_idx=0, cache_index=0),
            _finalized_binding("sliding.1", group_idx=1, cache_index=1),
            _finalized_binding("sliding.2", group_idx=1, cache_index=1),
        ),
        inference_device=torch.device("cpu"),
        inference_model=SimpleNamespace(hmonnx_session=None),
        placeholder_attn_layers=layers,
        cache_dtype="sefp_k8_v8",
    )
    runner._get_kv_cache_layer_devices = lambda: {
        layer.layer_name: torch.device("cpu") for layer in layers
    }
    config = SimpleNamespace(
        num_blocks=2,
        kv_cache_tensors=[
            SimpleNamespace(
                size=full_spec.real_page_size_bytes * 2, shared_by=["full.0"]
            ),
            SimpleNamespace(
                size=sliding_spec.real_page_size_bytes * 2,
                shared_by=["sliding.1", "sliding.2"],
            ),
        ],
        kv_cache_groups=groups,
    )

    raw = runner._allocate_kv_cache_tensors(config)
    assert raw["sliding.1"] is raw["sliding.2"]
    assert raw["full.0"].data_ptr() != raw["sliding.1"].data_ptr()

    shaped = runner._reshape_kv_cache_tensors(raw, [64, 64])
    for placeholder in layers:
        placeholder.kv_cache = shaped[placeholder.layer_name]
    runner._wrap_sefp_kv_cache_handles()

    assert isinstance(layers[1].kv_cache, HMFPPagedKVCache)
    assert layers[1].kv_cache is layers[2].kv_cache


def test_recycled_sliding_pages_are_zeroed_once_for_shared_cache(
    monkeypatch,
) -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    sliding_spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    group = SimpleNamespace(
        kv_cache_spec=sliding_spec,
        layer_names=["sliding.0", "sliding.1"],
    )
    shared = allocate_hmfp_paged_kv_cache(
        num_blocks=8,
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        device="cpu",
    )
    shared.k_qdata.fill_(7)
    shared.k_scale_exp.fill_(8)
    shared.v_qdata.fill_(9)
    shared.v_scale_exp.fill_(10)
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[group])
    runner.model = SimpleNamespace(
        placeholder_attn_layers=[
            SimpleNamespace(layer_name="sliding.0", kv_cache=shared),
            SimpleNamespace(layer_name="sliding.1", kv_cache=shared),
        ]
    )
    named_tensor_visits = []
    original_named_tensors = HMFPPagedKVCache.named_tensors

    def counted_named_tensors(cache):
        if cache is shared:
            named_tensor_visits.append(cache.k_qdata.data_ptr())
        yield from original_named_tensors(cache)

    monkeypatch.setattr(HMFPPagedKVCache, "named_tensors", counted_named_tensors)
    monkeypatch.setattr(GPUModelRunner, "_zero_block_ids", lambda _self, _ids: None)

    runner._zero_block_ids([4, 5])

    for tensor in shared.tensors():
        assert torch.count_nonzero(tensor[[4, 5]]) == 0
        assert torch.count_nonzero(tensor[[3, 6]]) > 0
    assert named_tensor_visits == [shared.k_qdata.data_ptr()]

    runner._zero_block_ids([5])

    assert named_tensor_visits == [
        shared.k_qdata.data_ptr(),
        shared.k_qdata.data_ptr(),
    ]


def test_gemma4_zeroing_preserves_colliding_block_id_in_other_group(
    monkeypatch, tmp_path
) -> None:
    from vllm_merak.gemma4_trace import flush_traces

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    full_spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    sliding_spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    full_cache = allocate_hmfp_paged_kv_cache(
        num_blocks=2, block_size=64, num_kv_heads=1, head_size=64, device="cpu"
    )
    sliding_cache = allocate_hmfp_paged_kv_cache(
        num_blocks=2, block_size=64, num_kv_heads=1, head_size=64, device="cpu"
    )
    for cache in (full_cache, sliding_cache):
        for tensor in cache.tensors():
            tensor.fill_(9)
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=full_spec, layer_names=["full"]),
            SimpleNamespace(kv_cache_spec=sliding_spec, layer_names=["sliding"]),
        ]
    )
    runner.model = SimpleNamespace(
        _uses_gemma4_page_attention_contract=lambda: True,
        placeholder_attn_layers=[
            SimpleNamespace(layer_name="full", kv_cache=full_cache),
            SimpleNamespace(layer_name="sliding", kv_cache=sliding_cache),
        ],
    )
    runner._gemma4_pending_zero_records = [
        {"request_id": "new", "group_idx": 0, "block_ids": [0]}
    ]
    trace = tmp_path / "kv.jsonl"
    monkeypatch.setenv("VLLM_MERAK_KV_TRACE_JSONL", str(trace))
    monkeypatch.setenv("VLLM_MERAK_TRACE_RUN_ID", "collision-run")
    monkeypatch.setattr(GPUModelRunner, "_zero_block_ids", lambda _self, _ids: None)

    runner._zero_block_ids([0])
    flush_traces()

    for tensor in full_cache.tensors():
        assert torch.count_nonzero(tensor[0]) == 0
    for tensor in sliding_cache.tensors():
        assert torch.count_nonzero(tensor[0]) > 0
    event = json.loads(trace.read_text())
    assert event["zero_records"] == [
        {
            "backing_ptr": full_cache.k_qdata.data_ptr(),
            "block_ids": [0],
            "group_idx": 0,
            "request_id": "new",
        }
    ]


def test_gemma4_new_request_zero_records_exclude_prefix_cache_blocks() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=64)),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=32)),
        ]
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="req",
                num_computed_tokens=128,
                block_ids=([3, 4, 5], [10, 11, 12, 13, 14]),
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
    )

    assert runner._gemma4_new_block_records(scheduler_output) == [
        {
            "request_id": "req",
            "group_idx": 0,
            "block_ids": [5],
            "logical_block_start": 2,
        },
        {
            "request_id": "req",
            "group_idx": 1,
            "block_ids": [14],
            "logical_block_start": 4,
        },
    ]


def test_gemma4_full_virtual_block_zeroes_both_physical_pages() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    full_spec = HMFPFullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    cache = allocate_hmfp_paged_kv_cache(
        num_blocks=20,
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        device="cpu",
    )
    for tensor in cache.tensors():
        tensor.fill_(9)
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=full_spec, layer_names=["full.0"])
        ]
    )
    runner.model = SimpleNamespace(
        page_attention_bindings=(),
        placeholder_attn_layers=[SimpleNamespace(layer_name="full.0", kv_cache=cache)],
    )

    runner._zero_gemma4_records(
        [{"request_id": "req", "group_idx": 0, "block_ids": [7]}]
    )

    for tensor in cache.tensors():
        assert torch.count_nonzero(tensor[[14, 15]]) == 0
        assert torch.count_nonzero(tensor[[7, 13, 16]]) > 0


def test_update_states_zeros_each_new_hmfp_backing_once_before_parent_continues(
    monkeypatch,
) -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    group = SimpleNamespace(kv_cache_spec=spec, layer_names=["sliding.0"])
    cache = allocate_hmfp_paged_kv_cache(
        num_blocks=8,
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        device="cpu",
    )
    for tensor in cache.tensors():
        tensor.fill_(7)
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[group])
    runner.model = SimpleNamespace(
        placeholder_attn_layers=[
            SimpleNamespace(layer_name="sliding.0", kv_cache=cache)
        ]
    )

    named_tensor_visits = []
    original_named_tensors = HMFPPagedKVCache.named_tensors

    def counted_named_tensors(visited_cache):
        if visited_cache is cache:
            named_tensor_visits.append(visited_cache.k_qdata.data_ptr())
        yield from original_named_tensors(visited_cache)

    monkeypatch.setattr(HMFPPagedKVCache, "named_tensors", counted_named_tensors)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )

    class InputBatch:
        req_id_to_index = {}

        def remove_request(self, _req_id):
            raise AssertionError("empty scheduler step removed a request")

        def condense(self):
            assert named_tensor_visits == [cache.k_qdata.data_ptr()]
            for tensor in cache.tensors():
                assert torch.count_nonzero(tensor[[4, 5]]) == 0
                assert torch.count_nonzero(tensor[[3, 6]]) > 0

        def refresh_metadata(self):
            pass

    runner.requests = {}
    runner.num_prompt_logprobs = {}
    runner.late_interaction_runner = SimpleNamespace(
        on_requests_finished=lambda _req_ids: None
    )
    runner.input_batch = InputBatch()
    runner.encoder_cache = {}
    runner.speculative_config = None
    runner.use_async_spec_decode = False
    runner.use_async_scheduling = False
    runner._may_reorder_batch = lambda _scheduler_output: None
    runner.merak_linear_state_idx = {}

    cached_reqs = SimpleNamespace(
        resumed_req_ids=set(),
        req_ids=[],
        new_block_ids=[([4, 5],)],
    )
    scheduler_output = SimpleNamespace(
        preempted_req_ids=[],
        finished_req_ids=[],
        new_block_ids_to_zero=[4, 5],
        free_encoder_mm_hashes=[],
        num_scheduled_tokens={},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_reqs,
        scheduled_spec_decode_tokens={},
    )

    runner._update_states(scheduler_output)

    assert named_tensor_visits == [cache.k_qdata.data_ptr()]


def test_update_states_emits_zero_proof_when_upstream_zero_ids_are_empty(
    monkeypatch, tmp_path
) -> None:
    from vllm_merak.gemma4_trace import flush_traces

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    group = SimpleNamespace(kv_cache_spec=spec, layer_names=["sliding.0"])
    cache = allocate_hmfp_paged_kv_cache(
        num_blocks=8,
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        device="cpu",
    )
    for tensor in cache.tensors():
        tensor.fill_(7)
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[group])
    runner.model = SimpleNamespace(
        _uses_gemma4_page_attention_contract=lambda: True,
        placeholder_attn_layers=[
            SimpleNamespace(layer_name="sliding.0", kv_cache=cache)
        ],
    )
    runner.requests = {}
    runner.merak_linear_checkpoint_store = None
    runner.merak_linear_state_idx = {}
    runner.input_batch = SimpleNamespace(req_ids=[])
    runner._gemma4_trace_released_block_ids = {(0, 4)}

    visits = []
    stream_order = []
    original_named_tensors = HMFPPagedKVCache.named_tensors

    def counted_named_tensors(visited_cache):
        if visited_cache is cache:
            visits.append(visited_cache.k_qdata.data_ptr())
            stream_order.append("zero")
        yield from original_named_tensors(visited_cache)

    parent_calls = []

    def parent_update(_self, _scheduler_output):
        parent_calls.append("parent")
        assert stream_order == ["zero"]
        for tensor in cache.tensors():
            assert torch.count_nonzero(tensor[[4, 5]]) == 0
            assert torch.count_nonzero(tensor[[3, 6]]) > 0
        return "updated"

    trace = tmp_path / "kv.jsonl"
    monkeypatch.setenv("VLLM_MERAK_KV_TRACE_JSONL", str(trace))
    monkeypatch.setenv("VLLM_MERAK_TRACE_RUN_ID", "empty-upstream-zero")
    monkeypatch.setattr(HMFPPagedKVCache, "named_tensors", counted_named_tensors)
    monkeypatch.setattr(GPUModelRunner, "_update_states", parent_update)
    scheduler_output = SimpleNamespace(
        preempted_req_ids=[],
        finished_req_ids=[],
        new_block_ids_to_zero=[],
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            resumed_req_ids=set(),
            req_ids=["req"],
            new_block_ids=[([4, 5],)],
        ),
    )

    assert runner._update_states(scheduler_output) == "updated"
    flush_traces()

    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    zero_rows = [row for row in rows if row["event"] == "zero_complete"]
    assert parent_calls == ["parent"]
    assert visits == [cache.k_qdata.data_ptr()]
    assert len(zero_rows) == 1
    assert zero_rows[0]["ordering"] == "before_model_execute"
    assert zero_rows[0]["zeroed_block_ids"] == [4, 5]
    assert zero_rows[0]["recycled_block_ids"] == [4]
    assert zero_rows[0]["recycled_records"] == [
        {"block_ids": [4], "group_idx": 0, "request_id": "req"}
    ]
    assert zero_rows[0]["zero_records"] == [
        {
            "backing_ptr": cache.k_qdata.data_ptr(),
            "block_ids": [4, 5],
            "group_idx": 0,
            "request_id": "req",
        }
    ]


def test_update_states_zeroes_non_gemma_hmfp_subclass_blocks_when_upstream_skips_them(
    monkeypatch,
) -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    cache = allocate_hmfp_paged_kv_cache(
        num_blocks=8,
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        device="cpu",
    )
    for tensor in cache.tensors():
        tensor.fill_(7)
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=["full.0"])]
    )
    runner.model = SimpleNamespace(
        _uses_gemma4_page_attention_contract=lambda: False,
        placeholder_attn_layers=[SimpleNamespace(layer_name="full.0", kv_cache=cache)],
    )
    runner.requests = {}
    runner.merak_linear_checkpoint_store = None
    runner.merak_linear_state_idx = {}
    runner.input_batch = SimpleNamespace(req_ids=[])

    def parent_update(_self, _scheduler_output):
        for tensor in cache.tensors():
            assert torch.count_nonzero(tensor[[4, 5]]) == 0
            assert torch.count_nonzero(tensor[[3, 6]]) > 0
        return "updated"

    monkeypatch.setattr(GPUModelRunner, "_update_states", parent_update)
    scheduler_output = SimpleNamespace(
        preempted_req_ids=[],
        finished_req_ids=[],
        new_block_ids_to_zero=[],
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            resumed_req_ids=set(),
            req_ids=["req"],
            new_block_ids=[([4, 5],)],
        ),
    )

    assert runner._update_states(scheduler_output) == "updated"


def test_shared_backing_zeroes_union_of_block_ids_from_all_groups(monkeypatch) -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    groups = [
        SimpleNamespace(kv_cache_spec=spec, layer_names=["sliding.0"]),
        SimpleNamespace(kv_cache_spec=spec, layer_names=["sliding.1"]),
    ]
    shared = allocate_hmfp_paged_kv_cache(
        num_blocks=8,
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        device="cpu",
    )
    for tensor in shared.tensors():
        tensor.fill_(11)
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    runner.model = SimpleNamespace(
        placeholder_attn_layers=[
            SimpleNamespace(layer_name="sliding.0", kv_cache=shared),
            SimpleNamespace(layer_name="sliding.1", kv_cache=shared),
        ]
    )

    visits = []
    original_named_tensors = HMFPPagedKVCache.named_tensors

    def counted_named_tensors(cache):
        visits.append(cache.k_qdata.data_ptr())
        yield from original_named_tensors(cache)

    monkeypatch.setattr(HMFPPagedKVCache, "named_tensors", counted_named_tensors)

    runner._zero_hmfp_group_blocks({0: {1, 3}, 1: {2, 3}})

    for tensor in shared.tensors():
        assert torch.count_nonzero(tensor[[1, 2, 3]]) == 0
        assert torch.count_nonzero(tensor[[0, 4]]) > 0
    assert visits == [shared.k_qdata.data_ptr()]


def test_qwen_hmfp_allocation_ignores_generic_shared_by_aliasing() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.kv_cache_dtype = "sefp_k8_v8"
    runner.scheduler_config = SimpleNamespace(max_num_seqs=1)
    runner.model = SimpleNamespace()
    runner._get_kv_cache_layer_devices = lambda: {
        "qwen.0": torch.device("cpu"),
        "qwen.1": torch.device("cpu"),
    }
    spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    group = SimpleNamespace(
        kv_cache_spec=spec,
        layer_names=["qwen.0", "qwen.1"],
    )
    runner._kv_cache_spec_attn_group_iterator = lambda: iter([group])
    config = SimpleNamespace(
        num_blocks=1,
        kv_cache_groups=[group],
        kv_cache_tensors=[
            SimpleNamespace(
                size=spec.real_page_size_bytes,
                shared_by=["qwen.0", "qwen.1"],
            )
        ],
    )

    raw = runner._allocate_kv_cache_tensors(config)

    assert raw["qwen.0"] is not raw["qwen.1"]
    assert raw["qwen.0"].data_ptr() != raw["qwen.1"].data_ptr()


def test_shared_by_rejects_incompatible_specs() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.kv_cache_dtype = "sefp_k8_v8"
    runner.scheduler_config = SimpleNamespace(max_num_seqs=1)
    runner.model = SimpleNamespace(
        gemma4_cache_contract=_authoritative_gemma4_contract(),
        page_attention_bindings=(
            _finalized_binding("layer.0", group_idx=0, cache_index=0),
            _finalized_binding("layer.1", group_idx=1, cache_index=1),
        ),
    )
    runner._get_kv_cache_layer_devices = lambda: {
        "layer.0": torch.device("cpu"),
        "layer.1": torch.device("cpu"),
    }
    spec0 = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    spec1 = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    groups = [
        SimpleNamespace(kv_cache_spec=spec0, layer_names=["layer.0"]),
        SimpleNamespace(kv_cache_spec=spec1, layer_names=["layer.1"]),
    ]
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(groups)
    config = SimpleNamespace(
        num_blocks=1,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            SimpleNamespace(
                size=spec1.real_page_size_bytes, shared_by=["layer.0", "layer.1"]
            )
        ],
    )

    with pytest.raises(RuntimeError, match="shared_by.*incompatible"):
        runner._allocate_kv_cache_tensors(config)


@pytest.mark.parametrize(
    ("layer_devices", "message"),
    [
        ({"sliding.0": torch.device("cpu")}, "no validated PageAttention device"),
        (
            {
                "sliding.0": torch.device("cuda:0"),
                "sliding.1": torch.device("cuda:1"),
            },
            "ambiguous physical owner devices",
        ),
    ],
)
def test_gemma_shared_by_rejects_missing_or_ambiguous_owner_devices(
    layer_devices, message
) -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.kv_cache_dtype = "sefp_k8_v8"
    runner.scheduler_config = SimpleNamespace(max_num_seqs=1)
    runner.model = SimpleNamespace(
        gemma4_cache_contract=_authoritative_gemma4_contract(),
        page_attention_bindings=(
            _finalized_binding("sliding.0", group_idx=0, cache_index=0),
            _finalized_binding("sliding.1", group_idx=0, cache_index=0),
        ),
    )
    runner._get_kv_cache_layer_devices = lambda: layer_devices
    spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    group = SimpleNamespace(
        kv_cache_spec=spec,
        layer_names=["sliding.0", "sliding.1"],
    )
    runner._kv_cache_spec_attn_group_iterator = lambda: iter([group])
    config = SimpleNamespace(
        num_blocks=1,
        kv_cache_groups=[group],
        kv_cache_tensors=[
            SimpleNamespace(
                size=spec.real_page_size_bytes,
                shared_by=["sliding.0", "sliding.1"],
            )
        ],
    )

    with pytest.raises(RuntimeError, match=message):
        runner._allocate_kv_cache_tensors(config)


def test_gemma_cache_indices_override_vllm_cross_group_tensor_pairing() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    full_spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    sliding_spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    config = SimpleNamespace(
        num_blocks=2,
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=full_spec, layer_names=["full.0"]),
            SimpleNamespace(
                kv_cache_spec=sliding_spec,
                layer_names=["sliding.1", "sliding.2"],
            ),
        ],
        # This is vLLM's generic cross-group pairing, which is not the
        # exported Gemma cache ownership contract.
        kv_cache_tensors=[
            SimpleNamespace(shared_by=["full.0", "sliding.1"]),
            SimpleNamespace(shared_by=["sliding.2"]),
        ],
    )
    bindings = (
        SimpleNamespace(layer_name="full.0", cache_index=0, cache_group_idx=0),
        SimpleNamespace(layer_name="sliding.1", cache_index=1, cache_group_idx=1),
        SimpleNamespace(layer_name="sliding.2", cache_index=1, cache_group_idx=1),
    )

    runner._apply_gemma4_cache_tensor_ownership(config, bindings)

    assert [tensor.shared_by for tensor in config.kv_cache_tensors] == [
        ["full.0"],
        ["sliding.1", "sliding.2"],
    ]
    assert [tensor.size for tensor in config.kv_cache_tensors] == [
        full_spec.real_page_size_bytes * 2,
        sliding_spec.real_page_size_bytes * 2,
    ]


def test_gemma_cache_ownership_resolves_uniform_shared_alias() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    owner_name = "placeholder_attn_layers.1"
    alias_name = "placeholder_attn_layers.2"
    group = SimpleNamespace(
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=64,
            kv_cache_specs={owner_name: spec},
        ),
        layer_names=[owner_name, alias_name],
    )
    config = SimpleNamespace(
        num_blocks=3,
        kv_cache_groups=[group],
        kv_cache_tensors=[],
    )
    bindings = (
        Gemma4LayerCacheBinding(
            layer_idx=1,
            layer_name=owner_name,
            attention_kind="sliding_attention",
            cache_group_idx=0,
            cache_index=1,
            kv_shape=(1, 1, 1344, 64),
            sliding_window=1024,
            cache_owner_layer_idx=1,
            scheduler_block_size=64,
        ),
        Gemma4LayerCacheBinding(
            layer_idx=2,
            layer_name=alias_name,
            attention_kind="sliding_attention",
            cache_group_idx=0,
            cache_index=1,
            kv_shape=(1, 1, 1344, 64),
            sliding_window=1024,
            cache_owner_layer_idx=1,
            scheduler_block_size=64,
        ),
    )

    runner._apply_gemma4_cache_tensor_ownership(config, bindings)

    assert len(config.kv_cache_tensors) == 1
    assert config.kv_cache_tensors[0].shared_by == [owner_name, alias_name]
    assert config.kv_cache_tensors[0].size == 21 * spec.real_page_size_bytes


def test_gemma_cache_ownership_recovers_alias_omitted_from_finalized_group() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    owner_name = "placeholder_attn_layers.1"
    alias_name = "placeholder_attn_layers.2"
    group = SimpleNamespace(
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=64,
            kv_cache_specs={owner_name: spec},
        ),
        # vLLM intentionally drops KV-sharing aliases from finalized groups.
        layer_names=[owner_name],
    )
    config = SimpleNamespace(
        num_blocks=3,
        kv_cache_groups=[group],
        kv_cache_tensors=[],
    )
    bindings = (
        Gemma4LayerCacheBinding(
            layer_idx=1,
            layer_name=owner_name,
            attention_kind="sliding_attention",
            cache_group_idx=0,
            cache_index=1,
            kv_shape=(1, 1, 1344, 64),
            sliding_window=1024,
            cache_owner_layer_idx=1,
            scheduler_block_size=64,
        ),
        Gemma4LayerCacheBinding(
            layer_idx=2,
            layer_name=alias_name,
            attention_kind="sliding_attention",
            cache_group_idx=0,
            cache_index=1,
            kv_shape=(1, 1, 1344, 64),
            sliding_window=1024,
            cache_owner_layer_idx=1,
            scheduler_block_size=64,
        ),
    )

    runner._apply_gemma4_cache_tensor_ownership(config, bindings)

    assert len(config.kv_cache_tensors) == 1
    assert config.kv_cache_tensors[0].shared_by == [owner_name]
    assert config.kv_cache_tensors[0].size == 21 * spec.real_page_size_bytes


def test_gemma_full_560_scheduler_pages_expose_1120_physical_pages() -> None:
    """A finalized 128-token full page owns two exported 64-token pages."""

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.kv_cache_dtype = "sefp_k8_v8"
    runner.cache_dtype = "sefp_k8_v8"
    runner.vllm_config = SimpleNamespace()
    runner.scheduler_config = SimpleNamespace(max_num_seqs=1)

    physical_spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=512,
        head_size_v=512,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    finalized_spec = HMFPFullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=512,
        head_size_v=512,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    layer_name = "full.0"
    physical_group = SimpleNamespace(
        kv_cache_spec=physical_spec,
        kv_cache_group_id=0,
        layer_names=[layer_name],
    )
    finalized_group = SimpleNamespace(
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=128,
            kv_cache_specs={layer_name: finalized_spec},
        ),
        layer_names=[layer_name],
    )
    binding = Gemma4LayerCacheBinding(
        layer_idx=0,
        layer_name=layer_name,
        attention_kind="full_attention",
        cache_group_idx=0,
        cache_index=0,
        kv_shape=(1, 2, 262144, 512),
        sliding_window=None,
        scheduler_block_size=128,
        physical_block_size=64,
    )
    placeholder = SimpleNamespace(
        layer_name=layer_name,
        kv_cache=None,
        num_kv_heads=2,
        head_size=512,
        get_kv_cache_spec=lambda _config: physical_spec,
    )
    runner.model = SimpleNamespace(
        gemma4_cache_contract=_authoritative_gemma4_contract(),
        page_attention_bindings=(binding,),
        placeholder_attn_layers=[placeholder],
        cache_dtype="sefp_k8_v8",
        _uses_gemma4_page_attention_contract=lambda: True,
    )
    runner._get_kv_cache_layer_devices = lambda: {layer_name: torch.device("cpu")}
    runner._kv_cache_spec_attn_group_iterator = lambda: iter([physical_group])
    config = SimpleNamespace(
        num_blocks=560,
        kv_cache_groups=[finalized_group],
        kv_cache_tensors=[],
    )
    runner.kv_cache_config = config

    runner._apply_gemma4_cache_tensor_ownership(config, (binding,))
    expected_bytes = 560 * finalized_spec.real_page_size_bytes
    assert config.kv_cache_tensors[0].size == expected_bytes
    assert expected_bytes // physical_spec.real_page_size_bytes == 1120

    # Reproduce the runner input seen on hardware: the spec iterator exposes
    # exported 64-token pages while vLLM's tensor declaration can still carry
    # only one such page per scheduler block.  The finalized binding ratio,
    # not that declaration, is authoritative for physical allocation.
    config.kv_cache_tensors = [
        SimpleNamespace(
            size=560 * physical_spec.real_page_size_bytes,
            shared_by=[layer_name],
        )
    ]

    with patch(
        "vllm_merak.worker_v1.merak_model_runner_v1.torch.zeros",
        side_effect=lambda size, **_kwargs: torch.empty(
            size, dtype=torch.int8, device="meta"
        ),
    ):
        raw = runner._allocate_kv_cache_tensors(config)

    # vLLM exposes the finalized 128-token iterator by reshape time.  The
    # exported binding must still force 64-token physical rows.
    finalized_iterator_group = SimpleNamespace(
        kv_cache_spec=finalized_spec,
        kv_cache_group_id=0,
        layer_names=[layer_name],
    )
    runner._kv_cache_spec_attn_group_iterator = lambda: iter([finalized_iterator_group])
    shaped = runner._reshape_kv_cache_tensors(raw, [64])
    assert shaped[layer_name].shape == (
        1120,
        physical_spec.real_page_size_bytes,
    )

    placeholder.kv_cache = shaped[layer_name]
    runner._wrap_sefp_kv_cache_handles()
    assert isinstance(placeholder.kv_cache, HMFPPagedKVCache)
    assert placeholder.kv_cache.num_blocks == 1120
    assert placeholder.kv_cache.block_size == 64


def test_gemma_mixed_sliding_cache_reshapes_native_fp_storage() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.kv_cache_dtype = "sefp_k8_v8"
    runner.runner_only_attn_layers = set()
    layer_name = "placeholder_attn_layers.0"
    spec = SlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.float16,
        sliding_window=512,
    )
    binding = Gemma4LayerCacheBinding(
        layer_idx=0,
        layer_name=layer_name,
        attention_kind="sliding_attention",
        cache_group_idx=0,
        cache_index=0,
        kv_shape=(1, 1, 832, 256),
        sliding_window=512,
        cache_owner_layer_idx=0,
        scheduler_block_size=64,
    )
    runner.model = SimpleNamespace(page_attention_bindings=(binding,))
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(
        [
            SimpleNamespace(
                kv_cache_spec=spec,
                kv_cache_group_id=0,
                layer_names=[layer_name],
            )
        ]
    )
    page_bytes = 2 * 64 * 256 * torch.empty((), dtype=torch.float16).element_size()
    raw = torch.zeros(13 * page_bytes, dtype=torch.int8)

    shaped = runner._reshape_kv_cache_tensors({layer_name: raw}, [64])

    assert shaped[layer_name].dtype is torch.float16
    assert shaped[layer_name].shape == (2, 13, 1, 64, 256)
    assert shaped[layer_name].view(torch.uint8).data_ptr() == raw.data_ptr()


def test_merak_padded_kv_first_cache_uses_physical_page_stride() -> None:
    class KVFirstBackend:
        @staticmethod
        def get_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str="auto",
        ):
            del cache_dtype_str
            return (2, num_blocks, num_kv_heads, block_size, head_size)

        @staticmethod
        def get_kv_cache_stride_order():
            return (0, 1, 2, 3, 4)

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.kv_cache_dtype = "float16"
    runner.runner_only_attn_layers = set()
    layer_name = "full.0"
    spec = SimpleNamespace(
        block_size=2,
        storage_block_size=2,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float16,
        real_page_size_bytes=16,
        page_size_bytes=24,
        page_size_padded=24,
    )
    group = SimpleNamespace(
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        layer_names=[layer_name],
        backend=KVFirstBackend,
    )
    runner._kv_cache_spec_attn_group_iterator = lambda: iter([group])
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_tensors=[
            SimpleNamespace(
                size=72,
                shared_by=[layer_name],
                offset=0,
                block_stride=0,
            )
        ]
    )
    raw = torch.zeros(72, dtype=torch.int8)

    shaped = runner._reshape_kv_cache_tensors({layer_name: raw}, [2])[layer_name]

    assert shaped.shape == (2, 3, 1, 2, 2)
    assert shaped.dtype is torch.float16
    assert shaped.stride() == (4, 12, 4, 2, 1)
    assert shaped.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()

    shaped[0, 0].fill_(1)
    shaped[1, 0].fill_(2)
    shaped[0, 1].fill_(3)
    raw_values = raw.view(torch.float16)
    assert torch.equal(raw_values[:4], torch.ones(4))
    assert torch.equal(raw_values[4:8], torch.full((4,), 2.0))
    assert torch.equal(raw_values[8:12], torch.zeros(4))
    assert torch.equal(raw_values[12:16], torch.full((4,), 3.0))


def test_merak_packed_padded_kv_first_cache_honors_layer_offset() -> None:
    class KVFirstBackend:
        @staticmethod
        def get_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str="auto",
        ):
            del cache_dtype_str
            return (2, num_blocks, num_kv_heads, block_size, head_size)

        @staticmethod
        def get_kv_cache_stride_order():
            return (0, 1, 2, 3, 4)

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.kv_cache_dtype = "float16"
    runner.runner_only_attn_layers = set()
    layer_name = "full.0"
    spec = SimpleNamespace(
        block_size=2,
        storage_block_size=2,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float16,
        real_page_size_bytes=16,
        page_size_bytes=24,
        page_size_padded=24,
    )
    group = SimpleNamespace(
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        layer_names=[layer_name],
        backend=KVFirstBackend,
    )
    runner._kv_cache_spec_attn_group_iterator = lambda: iter([group])
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_tensors=[
            SimpleNamespace(
                size=128,
                shared_by=[layer_name],
                offset=8,
                block_stride=64,
            )
        ]
    )
    raw = torch.zeros(128, dtype=torch.int8)

    shaped = runner._reshape_kv_cache_tensors({layer_name: raw}, [2])[layer_name]

    assert shaped.shape == (2, 2, 1, 2, 2)
    assert shaped.stride() == (4, 32, 4, 2, 1)
    shaped[1, 1].fill_(7)
    raw_values = raw.view(torch.float16)
    assert torch.equal(raw_values[40:44], torch.full((4,), 7.0))
    assert torch.count_nonzero(raw_values).item() == 4


def test_merak_contiguous_hybrid_uses_exact_attention_and_linear_backings() -> None:
    class KVFirstBackend:
        @staticmethod
        def get_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str="auto",
        ):
            del cache_dtype_str
            return (2, num_blocks, num_kv_heads, block_size, head_size)

        @staticmethod
        def get_kv_cache_stride_order():
            return (0, 1, 2, 3, 4)

    class MambaSpec:
        block_size = 2
        shapes = ((2,),)
        dtypes = (torch.float16,)
        page_size_bytes = 24
        page_size_padded = 24

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.kv_cache_dtype = "float16"
    runner.runner_only_attn_layers = set()
    runner.scheduler_config = SimpleNamespace(max_num_seqs=1)
    runner.model = SimpleNamespace(
        hybrid_cache_policy=HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST
    )
    runner._get_kv_cache_layer_devices = lambda: {"full.0": torch.device("cpu")}
    attention_spec = SimpleNamespace(
        block_size=2,
        storage_block_size=2,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float16,
        real_page_size_bytes=16,
        page_size_bytes=24,
        page_size_padded=24,
    )
    groups = [
        SimpleNamespace(
            kv_cache_spec=attention_spec,
            kv_cache_group_id=0,
            layer_names=["full.0"],
            backend=KVFirstBackend,
        ),
        SimpleNamespace(
            kv_cache_spec=MambaSpec(),
            kv_cache_group_id=1,
            layer_names=["linear.0"],
            backend=None,
        ),
    ]
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(groups)
    config = SimpleNamespace(
        num_blocks=3,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            SimpleNamespace(
                size=144,
                shared_by=["full.0"],
                offset=0,
                block_stride=48,
            ),
            SimpleNamespace(
                size=144,
                shared_by=["linear.0"],
                offset=24,
                block_stride=48,
            ),
        ],
    )
    runner.kv_cache_config = config

    raw = runner._allocate_kv_cache_tensors(config)

    assert raw["full.0"].numel() == 3 * attention_spec.real_page_size_bytes
    assert raw["linear.0"].numel() == 4
    assert raw["full.0"].data_ptr() != raw["linear.0"].data_ptr()

    shaped = runner._reshape_kv_cache_tensors(raw, [2, 2])

    assert shaped["full.0"].shape == (2, 3, 1, 2, 2)
    assert shaped["full.0"].is_contiguous()
    assert shaped["linear.0"][0].shape == (1, 2)


def test_gemma_sliding_cache_allocates_exported_window_not_global_blocks() -> None:
    """Sliding backing is window + prefill chunk, not the 256K block pool."""

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.kv_cache_dtype = "sefp_k8_v8"
    runner.scheduler_config = SimpleNamespace(max_num_seqs=1)

    full_name = "full.0"
    sliding_name = "sliding.0"
    full_spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    sliding_spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    full_binding = Gemma4LayerCacheBinding(
        layer_idx=0,
        layer_name=full_name,
        attention_kind="full_attention",
        cache_group_idx=0,
        cache_index=0,
        kv_shape=(1, 1, 262144, 64),
        sliding_window=None,
        cache_owner_layer_idx=0,
        scheduler_block_size=64,
    )
    sliding_binding = Gemma4LayerCacheBinding(
        layer_idx=1,
        layer_name=sliding_name,
        attention_kind="sliding_attention",
        cache_group_idx=1,
        cache_index=1,
        kv_shape=(1, 1, 1344, 64),
        sliding_window=1024,
        cache_owner_layer_idx=1,
        scheduler_block_size=64,
    )
    contract = Gemma4CacheContract(
        version=2,
        prefill_chunk_size=320,
        bindings=(full_binding, sliding_binding),
        cache_shapes={0: full_binding.kv_shape, 1: sliding_binding.kv_shape},
    )
    runner.model = SimpleNamespace(
        gemma4_cache_contract=contract,
        page_attention_bindings=(full_binding, sliding_binding),
    )
    runner._get_kv_cache_layer_devices = lambda: {
        full_name: torch.device("cpu"),
        sliding_name: torch.device("cpu"),
    }
    groups = [
        SimpleNamespace(kv_cache_spec=full_spec, layer_names=[full_name]),
        SimpleNamespace(kv_cache_spec=sliding_spec, layer_names=[sliding_name]),
    ]
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(groups)
    num_blocks = 2408
    config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            SimpleNamespace(
                size=num_blocks * full_spec.real_page_size_bytes,
                shared_by=[full_name],
            ),
            SimpleNamespace(
                size=num_blocks * sliding_spec.real_page_size_bytes,
                shared_by=[sliding_name],
            ),
        ],
    )

    with patch(
        "vllm_merak.worker_v1.merak_model_runner_v1.torch.zeros",
        side_effect=lambda size, **_kwargs: torch.empty(
            size, dtype=torch.int8, device="meta"
        ),
    ):
        raw = runner._allocate_kv_cache_tensors(config)

    assert raw[full_name].numel() == num_blocks * full_spec.real_page_size_bytes
    assert raw[sliding_name].numel() == 21 * sliding_spec.real_page_size_bytes
    assert (
        runner._gemma4_required_physical_blocks(full_binding, num_blocks) == num_blocks
    )
    assert runner._gemma4_required_physical_blocks(sliding_binding, num_blocks) == 21


def test_merak_model_runner_allocates_kv_cache_by_hmonnx_layer_device() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cuda:0")
    runner.runner_only_attn_layers = set()

    class PageAttention:
        def __init__(self, device):
            self.device = device

    node0 = SimpleNamespace(op="call_module", target="pa0", name="node0")
    node1 = SimpleNamespace(op="call_module", target="pa1", name="node1")
    modules = {
        "pa0": PageAttention("cuda:0"),
        "pa1": PageAttention("cuda:1"),
    }
    graph_module = SimpleNamespace(
        graph=SimpleNamespace(nodes=[node0, node1]),
        get_submodule=lambda target: modules[target],
    )
    runtime_model = SimpleNamespace(
        prefill_model=SimpleNamespace(
            hmonnx_session=SimpleNamespace(
                graph_module=graph_module,
                interpreter=SimpleNamespace(node_device_map={}),
            )
        )
    )
    runner.model = SimpleNamespace(
        inference_device=torch.device("cuda:0"),
        inference_model=runtime_model,
        placeholder_attn_layers=[
            SimpleNamespace(layer_name="layer.0"),
            SimpleNamespace(layer_name="layer.1"),
        ],
    )

    kv_cache_config = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(size=128, shared_by=["layer.0", "layer.1"])],
        kv_cache_groups=[SimpleNamespace(layer_names=["layer.0", "layer.1"])],
    )
    allocated = []

    def fake_zeros(size, *, dtype, device):
        allocated.append((size, dtype, torch.device(device)))
        return torch.empty(size, dtype=dtype, device="meta")

    with patch(
        "vllm_merak.worker_v1.merak_model_runner_v1.torch.zeros", side_effect=fake_zeros
    ):
        kv_cache_raw_tensors = MerakModelRunnerV1._allocate_kv_cache_tensors(
            runner, kv_cache_config
        )

    assert allocated == [
        (128, torch.int8, torch.device("cuda:0")),
        (128, torch.int8, torch.device("cuda:1")),
    ]
    assert kv_cache_raw_tensors["layer.0"] is not kv_cache_raw_tensors["layer.1"]


def test_merak_model_runner_rejects_prefill_decode_page_attention_device_mismatch() -> (
    None
):
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cuda:0")
    runner.model = SimpleNamespace(
        inference_device=torch.device("cuda:0"),
        inference_model=SimpleNamespace(
            prefill_model=SimpleNamespace(
                hmonnx_session=SimpleNamespace(
                    graph_module=_page_attention_graph(
                        ("pa0", "cuda:0"), ("pa1", "cuda:1")
                    ),
                    interpreter=SimpleNamespace(node_device_map={}),
                )
            ),
            decode_model=SimpleNamespace(
                hmonnx_session=SimpleNamespace(
                    graph_module=_page_attention_graph(
                        ("pa0", "cuda:1"), ("pa1", "cuda:1")
                    ),
                    interpreter=SimpleNamespace(node_device_map={}),
                )
            ),
        ),
        placeholder_attn_layers=[
            SimpleNamespace(layer_name="model.layers.3.attn"),
            SimpleNamespace(layer_name="model.layers.7.attn"),
        ],
    )

    with pytest.raises(RuntimeError, match="prefill/decode PageAttention.*device"):
        runner._get_kv_cache_layer_devices()


def test_merak_model_runner_matches_prefill_decode_page_attention_by_semantic_identity() -> (
    None
):
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cuda:0")
    runner.model = SimpleNamespace(
        inference_device=torch.device("cuda:0"),
        inference_model=SimpleNamespace(
            prefill_model=SimpleNamespace(
                hmonnx_session=SimpleNamespace(
                    graph_module=_indexed_page_attention_graph(
                        ("node_000639_flashattention", "cuda:0", 0, 0),
                        ("node_001389_flashattention", "cuda:1", 1, 1),
                    ),
                    interpreter=SimpleNamespace(node_device_map={}),
                )
            ),
            decode_model=SimpleNamespace(
                hmonnx_session=SimpleNamespace(
                    graph_module=_indexed_page_attention_graph(
                        ("node_000539_flashattention", "cuda:0", 0, 0),
                        ("node_001189_flashattention", "cuda:1", 1, 1),
                    ),
                    interpreter=SimpleNamespace(node_device_map={}),
                )
            ),
        ),
        placeholder_attn_layers=[
            SimpleNamespace(layer_name="model.layers.0.attn"),
            SimpleNamespace(layer_name="model.layers.1.attn"),
        ],
    )

    assert runner._get_kv_cache_layer_devices() == {
        "model.layers.0.attn": torch.device("cuda:0"),
        "model.layers.1.attn": torch.device("cuda:1"),
    }


def test_merak_model_runner_rejects_prefill_decode_page_attention_order_mismatch() -> (
    None
):
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cuda:0")
    runner.model = SimpleNamespace(
        inference_device=torch.device("cuda:0"),
        inference_model=SimpleNamespace(
            prefill_model=SimpleNamespace(
                hmonnx_session=SimpleNamespace(
                    graph_module=_page_attention_graph(
                        ("pa0", "cuda:0"), ("pa1", "cuda:1")
                    ),
                    interpreter=SimpleNamespace(node_device_map={}),
                )
            ),
            decode_model=SimpleNamespace(
                hmonnx_session=SimpleNamespace(
                    graph_module=_page_attention_graph(
                        ("pa1", "cuda:1"), ("pa0", "cuda:0")
                    ),
                    interpreter=SimpleNamespace(node_device_map={}),
                )
            ),
        ),
        placeholder_attn_layers=[
            SimpleNamespace(layer_name="model.layers.3.attn"),
            SimpleNamespace(layer_name="model.layers.7.attn"),
        ],
    )

    with pytest.raises(RuntimeError, match="identity/order mismatch"):
        runner._get_kv_cache_layer_devices()


def test_merak_model_runner_rejects_page_attention_count_mismatch_with_placeholders() -> (
    None
):
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cuda:0")
    graph = _page_attention_graph(("pa0", "cuda:0"))
    runner.model = SimpleNamespace(
        inference_device=torch.device("cuda:0"),
        inference_model=SimpleNamespace(
            prefill_model=SimpleNamespace(
                hmonnx_session=SimpleNamespace(
                    graph_module=graph,
                    interpreter=SimpleNamespace(node_device_map={}),
                )
            ),
            decode_model=SimpleNamespace(
                hmonnx_session=SimpleNamespace(
                    graph_module=graph,
                    interpreter=SimpleNamespace(node_device_map={}),
                )
            ),
        ),
        placeholder_attn_layers=[
            SimpleNamespace(layer_name="model.layers.3.attn"),
            SimpleNamespace(layer_name="model.layers.7.attn"),
        ],
    )

    with pytest.raises(RuntimeError, match="PageAttention count.*placeholder count"):
        runner._get_kv_cache_layer_devices()


def test_merak_model_runner_rejects_missing_page_attention_runtime_map() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cuda:0")
    runner.model = SimpleNamespace(
        enable_page_attention=True,
        inference_device=torch.device("cuda:0"),
        inference_model=SimpleNamespace(
            prefill_model=SimpleNamespace(
                hmonnx_session=SimpleNamespace(graph_module=None)
            ),
            decode_model=SimpleNamespace(
                hmonnx_session=SimpleNamespace(graph_module=None)
            ),
        ),
        placeholder_attn_layers=[SimpleNamespace(layer_name="model.layers.3.attn")],
    )

    with pytest.raises(RuntimeError, match="PageAttention count.*placeholder count"):
        runner._get_kv_cache_layer_devices()


def test_merak_model_runner_allocates_kv_cache_by_hmonnx_v2_page_attention_device() -> (
    None
):
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cuda:0")
    runner.runner_only_attn_layers = set()

    class PageAttention:
        device = "cpu"

    node0 = SimpleNamespace(op="call_module", target="pa0", name="node0")
    node1 = SimpleNamespace(op="call_module", target="pa1", name="node1")
    modules = {
        "pa0": PageAttention(),
        "pa1": PageAttention(),
    }
    graph_module = SimpleNamespace(
        graph=SimpleNamespace(nodes=[node0, node1]),
        get_submodule=lambda target: modules[target],
    )
    runtime_model = SimpleNamespace(
        graph_module=graph_module,
        interpreter=SimpleNamespace(node_device_map={}, device=torch.device("cuda:0")),
    )
    runner.model = SimpleNamespace(
        inference_device=torch.device("cuda:0"),
        inference_model=runtime_model,
        placeholder_attn_layers=[
            SimpleNamespace(layer_name="layer.0"),
            SimpleNamespace(layer_name="layer.1"),
        ],
    )

    kv_cache_config = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(size=128, shared_by=["layer.0", "layer.1"])],
        kv_cache_groups=[SimpleNamespace(layer_names=["layer.0", "layer.1"])],
    )
    allocated = []

    def fake_zeros(size, *, dtype, device):
        allocated.append((size, dtype, torch.device(device)))
        return torch.empty(size, dtype=dtype, device="meta")

    with patch(
        "vllm_merak.worker_v1.merak_model_runner_v1.torch.zeros", side_effect=fake_zeros
    ):
        kv_cache_raw_tensors = MerakModelRunnerV1._allocate_kv_cache_tensors(
            runner, kv_cache_config
        )

    assert allocated == [
        (128, torch.int8, torch.device("cuda:0")),
    ]
    assert kv_cache_raw_tensors["layer.0"] is kv_cache_raw_tensors["layer.1"]


def test_merak_model_runner_keeps_shared_kv_cache_on_same_device() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cuda:0")
    runner.runner_only_attn_layers = set()
    runner.model = SimpleNamespace(
        inference_device=torch.device("cuda:0"),
        inference_model=SimpleNamespace(hmonnx_session=None),
        placeholder_attn_layers=[
            SimpleNamespace(layer_name="layer.0"),
            SimpleNamespace(layer_name="layer.1"),
        ],
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(size=128, shared_by=["layer.0", "layer.1"])],
        kv_cache_groups=[SimpleNamespace(layer_names=["layer.0", "layer.1"])],
    )

    with patch("vllm_merak.worker_v1.merak_model_runner_v1.torch.zeros") as zeros:
        zeros.return_value = torch.empty(128, dtype=torch.int8, device="meta")
        kv_cache_raw_tensors = MerakModelRunnerV1._allocate_kv_cache_tensors(
            runner, kv_cache_config
        )

    zeros.assert_called_once()
    assert kv_cache_raw_tensors["layer.0"] is kv_cache_raw_tensors["layer.1"]


def test_merak_model_runner_preserves_packed_backing_allocation() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.model = SimpleNamespace(
        inference_device=torch.device("cpu"),
        inference_model=SimpleNamespace(hmonnx_session=None),
        placeholder_attn_layers=[
            SimpleNamespace(layer_name="layer.0"),
            SimpleNamespace(layer_name="layer.1"),
        ],
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_tensors=[
            SimpleNamespace(
                size=256, shared_by=["layer.0"], block_stride=128, offset=0
            ),
            SimpleNamespace(
                size=256, shared_by=["layer.1"], block_stride=128, offset=64
            ),
        ],
        kv_cache_groups=[SimpleNamespace(layer_names=["layer.0", "layer.1"])],
    )

    with patch("vllm_merak.worker_v1.merak_model_runner_v1.torch.zeros") as zeros:
        zeros.return_value = torch.empty(256, dtype=torch.int8, device="meta")
        kv_cache_raw_tensors = MerakModelRunnerV1._allocate_kv_cache_tensors(
            runner, kv_cache_config
        )

    zeros.assert_called_once_with(256, dtype=torch.int8, device=torch.device("cpu"))
    assert kv_cache_raw_tensors["layer.0"] is kv_cache_raw_tensors["layer.1"]


def test_merak_page_attention_context_is_passed_to_runtime() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.inference_device = torch.device("cpu")
    model.placeholder_attn_layers = [
        SimpleNamespace(kv_cache="cache0"),
        SimpleNamespace(kv_cache="cache1"),
    ]
    model._get_block_size = lambda: 64

    captured_context = {}

    class Runtime:
        def set_page_attention_context(self, **kwargs):
            captured_context.update(kwargs)

    model._set_runtime_page_attention_context(
        Runtime(),
        (3, 5),
        torch.tensor([192, 193], dtype=torch.long),
    )

    assert captured_context["paged_kv_caches"] == ("cache0", "cache1")
    assert captured_context["block_ids"].tolist() == [3, 5]
    assert captured_context["slot_mapping"].tolist() == [192, 193]
    assert captured_context["block_size"] == 64

    first_block_ids = captured_context["block_ids"]
    first_slot_mapping = captured_context["slot_mapping"]

    model._set_runtime_page_attention_context(
        Runtime(),
        (7, 9),
        torch.tensor([448, 449], dtype=torch.long),
    )

    assert captured_context["block_ids"] is first_block_ids
    assert captured_context["slot_mapping"] is first_slot_mapping
    assert captured_context["block_ids"].tolist() == [7, 9]
    assert captured_context["slot_mapping"].tolist() == [448, 449]

    model._set_runtime_page_attention_context(
        Runtime(),
        (11, 13, 15),
        torch.tensor([704, 705, 706], dtype=torch.long),
    )

    assert captured_context["block_ids"].tolist() == [11, 13, 15]
    assert captured_context["slot_mapping"].tolist() == [704, 705, 706]


def test_merak_page_attention_context_pads_short_prefill_tail_chunks() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.inference_device = torch.device("cpu")
    model.input_embeddings = SimpleNamespace(weight=torch.empty(1, dtype=torch.float32))
    model.logits_processor = SimpleNamespace(vocab_size=4)
    model.placeholder_attn_layers = [SimpleNamespace(kv_cache="cache0")]
    model._prefer_input_ids_runtime = True
    model._get_block_size = lambda: 2
    model._ensure_runtime_cache_tensors = lambda: None

    captured_contexts = []

    class Runtime:
        def set_prefill(self):
            pass

        def set_decode(self):
            raise AssertionError("six-token request should use prefill chunks")

        def get_input_sequence_length(self):
            return 4

        def get_data_preprocessor(self):
            return lambda data: (data["input_ids"],)

        def set_page_attention_context(self, **kwargs):
            captured_contexts.append(
                {
                    "block_ids": kwargs["block_ids"].clone(),
                    "slot_mapping": kwargs["slot_mapping"].clone(),
                }
            )

        def forward(self, *_args):
            return torch.arange(4, dtype=torch.float32)

    model.inference_model = Runtime()

    output = MerakForCausalLM._run_request(
        model,
        0,
        input_ids=torch.arange(6, dtype=torch.long),
        page_attention_block_ids=(10, 11, 12),
        page_attention_slot_mapping=torch.tensor(
            [20, 21, 22, 23, 24, 25], dtype=torch.long
        ),
    )

    assert output.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert len(captured_contexts) == 2
    assert captured_contexts[0]["block_ids"].tolist() == [10, 11, 12]
    assert captured_contexts[0]["slot_mapping"].tolist() == [20, 21, 22, 23]
    assert captured_contexts[1]["block_ids"].tolist() == [10, 11, 12, 12]
    assert captured_contexts[1]["slot_mapping"].tolist() == [24, 25, -1, -1]


def test_merak_page_attention_block_table_capacity_uses_growth_buckets() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=262144)
    )

    assert model._page_attention_block_table_capacity(64, 29) == 32
    assert model._page_attention_block_table_capacity(64, 33) == 64
    assert model._page_attention_block_table_capacity(64, 4096) == 4096


def test_merak_page_attention_block_table_capacity_tracks_current_context() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=262144)
    )
    runtime_model = SimpleNamespace(
        decode_model=SimpleNamespace(
            hmonnx_session=SimpleNamespace(interpreter=SimpleNamespace())
        )
    )

    initial_capacity = model._page_attention_block_table_capacity(64, 29)
    model._clear_runtime_cuda_graph_for_page_attention_capacity_change(
        runtime_model, initial_capacity
    )
    grown_capacity = model._page_attention_block_table_capacity(64, 33)
    model._clear_runtime_cuda_graph_for_page_attention_capacity_change(
        runtime_model, grown_capacity
    )

    assert initial_capacity == 32
    assert grown_capacity == 64
    assert model._page_attention_block_table_capacity(64, 29) == 32


def test_merak_page_attention_capacity_selects_cached_graph_by_bucket() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    selected = []

    class DecodeInterpreter:
        def set_capture_cache_key(self, key):
            selected.append(key)

        def clear(self, clear_disabled_reason=True):
            raise AssertionError(
                f"cached bucket selection must not clear graphs: {clear_disabled_reason}"
            )

    runtime_model = SimpleNamespace(
        _llm_prefill=False,
        decode_model=SimpleNamespace(
            hmonnx_session=SimpleNamespace(interpreter=DecodeInterpreter())
        ),
    )

    model._clear_runtime_cuda_graph_for_page_attention_capacity_change(runtime_model, 8)
    model._clear_runtime_cuda_graph_for_page_attention_capacity_change(
        runtime_model, 16
    )
    model._clear_runtime_cuda_graph_for_page_attention_capacity_change(runtime_model, 8)

    assert selected == [
        ("page_attention", "decode", 8),
        ("page_attention", "decode", 16),
        ("page_attention", "decode", 8),
    ]


def test_merak_page_attention_selects_same_bucket_for_prefill_and_decode() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    selected = []

    class Interpreter:
        def __init__(self, phase):
            self.phase = phase

        def set_capture_cache_key(self, key):
            selected.append((self.phase, key))

    runtime_model = SimpleNamespace(
        _llm_prefill=True,
        prefill_model=SimpleNamespace(
            hmonnx_session=SimpleNamespace(interpreter=Interpreter("prefill"))
        ),
        decode_model=SimpleNamespace(
            hmonnx_session=SimpleNamespace(interpreter=Interpreter("decode"))
        ),
    )

    model._clear_runtime_cuda_graph_for_page_attention_capacity_change(runtime_model, 8)
    runtime_model._llm_prefill = False
    model._clear_runtime_cuda_graph_for_page_attention_capacity_change(runtime_model, 8)

    assert selected == [
        ("prefill", ("page_attention", "decode", 8)),
        ("decode", ("page_attention", "decode", 8)),
    ]


def test_merak_page_attention_decode_avoids_full_256k_block_table_padding() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.inference_device = torch.device("cpu")
    model.input_embeddings = SimpleNamespace(weight=torch.empty(1, dtype=torch.float32))
    model.logits_processor = SimpleNamespace(vocab_size=4)
    model.placeholder_attn_layers = [SimpleNamespace(kv_cache="cache0")]
    model._prefer_input_ids_runtime = True
    model._get_block_size = lambda: 64
    model._ensure_runtime_cache_tensors = lambda: None
    model.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=262144)
    )

    captured_contexts = []

    class DecodeSession:
        def __init__(self):
            self.cleared = 0

        def clear(self, clear_disabled_reason=True):
            del clear_disabled_reason
            self.cleared += 1

    decode_session = DecodeSession()

    class Runtime:
        _llm_prefill = False
        decode_model = SimpleNamespace(
            hmonnx_session=SimpleNamespace(interpreter=decode_session)
        )

        def set_prefill(self):
            raise AssertionError("single-token request should use decode")

        def set_decode(self):
            pass

        def get_input_sequence_length(self):
            return 1

        def get_data_preprocessor(self):
            return lambda data: (data["input_ids"],)

        def set_page_attention_context(self, **kwargs):
            captured_contexts.append(
                {
                    "block_ids": kwargs["block_ids"].clone(),
                    "slot_mapping": kwargs["slot_mapping"].clone(),
                }
            )

        def forward(self, *_args):
            return torch.arange(4, dtype=torch.float32)

    model.inference_model = Runtime()

    output = MerakForCausalLM._run_request(
        model,
        1808,
        input_ids=torch.tensor([1], dtype=torch.long),
        page_attention_block_ids=tuple(range(29)),
        page_attention_slot_mapping=torch.tensor([1808], dtype=torch.long),
    )

    assert output.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert len(captured_contexts) == 1
    assert len(captured_contexts[0]["block_ids"]) == 32
    assert captured_contexts[0]["block_ids"].tolist()[:29] == list(range(29))
    assert captured_contexts[0]["block_ids"].tolist()[29:] == [28, 28, 28]
    assert captured_contexts[0]["slot_mapping"].tolist() == [1808]
    assert decode_session.cleared == 1


def test_merak_page_attention_q1_decode_pads_fixed_verify_graph_slots() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.inference_device = torch.device("cpu")
    model.input_embeddings = SimpleNamespace(weight=torch.empty(1, dtype=torch.float32))
    model.logits_processor = SimpleNamespace(vocab_size=4)
    model.placeholder_attn_layers = [SimpleNamespace(kv_cache="cache0")]
    model._prefer_input_ids_runtime = True
    model._get_block_size = lambda: 64
    model._ensure_runtime_cache_tensors = lambda: None
    model.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=262144)
    )

    captured_contexts = []

    class Runtime:
        _llm_prefill = False
        decode_model = SimpleNamespace(
            hmonnx_session=SimpleNamespace(
                interpreter=SimpleNamespace(clear=lambda **_kwargs: None)
            )
        )

        def set_prefill(self):
            raise AssertionError("single-token request should use decode")

        def set_decode(self):
            pass

        def get_input_sequence_length(self):
            # Target graph exported for K=15 speculative verification.
            return 16

        def get_data_preprocessor(self):
            return lambda data: (data["input_ids"],)

        def set_page_attention_context(self, **kwargs):
            captured_contexts.append(
                {
                    "block_ids": kwargs["block_ids"].clone(),
                    "slot_mapping": kwargs["slot_mapping"].clone(),
                }
            )

        def forward(self, *_args):
            return torch.arange(4, dtype=torch.float32)

    model.inference_model = Runtime()

    output = MerakForCausalLM._run_request(
        model,
        1808,
        input_ids=torch.tensor([1], dtype=torch.long),
        page_attention_block_ids=tuple(range(29)),
        page_attention_slot_mapping=torch.tensor([1808], dtype=torch.long),
        execution_phase="decode",
    )

    assert output.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert len(captured_contexts) == 1
    assert captured_contexts[0]["slot_mapping"].tolist() == [1808] + [-1] * 15


def test_merak_page_attention_context_falls_back_to_legacy_node_modules() -> None:
    from xhquant.xhonnxruntime.parsers.page_attention import PageAttention

    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.inference_device = torch.device("cpu")
    model.placeholder_attn_layers = [
        SimpleNamespace(kv_cache="cache0"),
        SimpleNamespace(kv_cache="cache1"),
    ]
    model._get_block_size = lambda: 64

    page_attention_0 = PageAttention.__new__(PageAttention)
    page_attention_1 = PageAttention.__new__(PageAttention)

    class Runtime:
        _llm_prefill = True
        prefill_model = SimpleNamespace(
            hmonnx_session=SimpleNamespace(
                node_modules=[page_attention_0, object(), page_attention_1]
            )
        )

        def set_page_attention_context(self, **kwargs):
            del kwargs
            raise AttributeError(
                "'HMONNXGolden' object has no attribute 'graph_module'"
            )

    model._set_runtime_page_attention_context(
        Runtime(),
        (3, 5),
        torch.tensor([192, 193], dtype=torch.long),
    )

    assert model._use_local_page_attention_context is True
    assert page_attention_0._page_attention_context.paged_kv_cache == "cache0"
    assert page_attention_1._page_attention_context.paged_kv_cache == "cache1"
    assert page_attention_0._page_attention_context.block_ids.dtype == torch.int64
    assert page_attention_0._page_attention_context.block_ids.tolist() == [3, 5]
    assert page_attention_0._page_attention_context.slot_mapping.tolist() == [192, 193]
    assert page_attention_0._page_attention_context.block_size == 64


def test_merak_page_attention_local_fallback_rejects_multi_device_caches() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.inference_device = torch.device("cuda:0")
    caches = (
        SimpleNamespace(device=torch.device("cuda:0")),
        SimpleNamespace(device=torch.device("cuda:2")),
    )
    model._active_hmonnx_model = lambda runtime_model: runtime_model
    model._find_page_attention_modules = lambda _runtime_model: [object(), object()]

    with pytest.raises(RuntimeError, match="multi-device.*local fallback"):
        model._set_runtime_page_attention_context_via_modules(
            runtime_model=object(),
            paged_kv_caches=caches,
            block_ids=torch.tensor([1], dtype=torch.int64),
            slot_mapping=torch.tensor([64], dtype=torch.int64),
            block_size=64,
        )


def test_merak_hybrid_layer_contract_rejects_placeholder_order_mismatch() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.num_layers = 2
    model.config = SimpleNamespace(linear_kv_cache_config={"num_layers": 2})
    model.placeholder_attn_layers = [
        SimpleNamespace(layer_name="model.layers.1.attn"),
        SimpleNamespace(layer_name="model.layers.3.attn"),
    ]
    model.placeholder_linear_state_layers = [
        SimpleNamespace(layer_name="model.layers.2.linear_attn"),
        SimpleNamespace(layer_name="model.layers.0.linear_attn"),
    ]
    hf_config = SimpleNamespace(
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
    )

    with pytest.raises(RuntimeError, match="Linear placeholder order"):
        model._validate_hybrid_layer_contract(hf_config)


def test_merak_hybrid_layer_contract_accepts_hf_and_golden_counts_and_order() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.num_layers = 2
    model.config = SimpleNamespace(linear_kv_cache_config={"num_layers": 2})
    model.placeholder_attn_layers = [
        SimpleNamespace(layer_name="model.layers.1.attn"),
        SimpleNamespace(layer_name="model.layers.3.attn"),
    ]
    model.placeholder_linear_state_layers = [
        SimpleNamespace(layer_name="model.layers.0.linear_attn"),
        SimpleNamespace(layer_name="model.layers.2.linear_attn"),
    ]
    hf_config = SimpleNamespace(
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
    )

    model._validate_hybrid_layer_contract(hf_config)


def test_merak_hybrid_layer_contract_rejects_golden_count_mismatch() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.num_layers = 3
    model.config = SimpleNamespace(linear_kv_cache_config={"num_layers": 2})
    model.placeholder_attn_layers = [
        SimpleNamespace(layer_name="model.layers.1.attn"),
        SimpleNamespace(layer_name="model.layers.3.attn"),
    ]
    model.placeholder_linear_state_layers = [
        SimpleNamespace(layer_name="model.layers.0.linear_attn"),
        SimpleNamespace(layer_name="model.layers.2.linear_attn"),
    ]
    hf_config = SimpleNamespace(
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
    )

    with pytest.raises(RuntimeError, match="HF/golden hybrid layer count mismatch"):
        model._validate_hybrid_layer_contract(hf_config)


def test_merak_runtime_linear_state_contract_rejects_recurrent_count_mismatch() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.placeholder_linear_state_layers = [
        SimpleNamespace(layer_name="model.layers.0.linear_attn"),
        SimpleNamespace(layer_name="model.layers.1.linear_attn"),
    ]
    mixin = SimpleNamespace(
        past_conv_caches=[
            (object(), object(), object()),
            (object(), object(), object()),
        ],
        past_recurrent_states=[object()],
    )
    model.inference_model = SimpleNamespace(get_kvcache_mixin=lambda: mixin)

    with pytest.raises(
        RuntimeError, match="2 placeholders, 2 conv caches, 1 recurrent states"
    ):
        model._validate_runtime_linear_state_contract()


def test_merak_runtime_contract_rejects_missing_page_attention_api() -> None:
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.model = SimpleNamespace(
        enable_page_attention=True,
        inference_model=SimpleNamespace(),
        placeholder_attn_layers=[SimpleNamespace(layer_name="model.layers.0.attn")],
        placeholder_linear_state_layers=[],
    )
    runner.kv_cache_config = None

    with pytest.raises(RuntimeError, match="set_page_attention_context"):
        runner._validate_runtime_contracts()


def test_merak_initialize_kv_cache_checks_runtime_api_before_parent_allocation() -> (
    None
):
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    events = []

    class Mixin:
        def prepare_kv_cache(self):
            events.append("prepare_kv_cache")

        def prepare_other_cache(self):
            events.append("prepare_other_cache")

    runner.model = SimpleNamespace(
        enable_page_attention=True,
        inference_model=SimpleNamespace(get_kvcache_mixin=lambda: Mixin()),
        placeholder_attn_layers=[SimpleNamespace(layer_name="model.layers.0.attn")],
        placeholder_linear_state_layers=[],
    )
    runner._wrap_sefp_kv_cache_handles = lambda: events.append("wrap")
    runner._install_cpu_slot_mapping_fallback = lambda: events.append("slot_mapping")

    with (
        patch.object(
            GPUModelRunner,
            "initialize_kv_cache",
            side_effect=lambda *_args, **_kwargs: events.append("parent-allocation"),
        ),
        pytest.raises(RuntimeError, match="set_page_attention_context"),
    ):
        runner.initialize_kv_cache(SimpleNamespace())  # type: ignore[arg-type]

    assert events == []


def test_merak_initialize_kv_cache_checks_mamba_groups_before_parent_allocation() -> (
    None
):
    class MambaSpec:
        block_size = 4
        num_speculative_blocks = 0

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.model = SimpleNamespace(
        enable_page_attention=False,
        inference_device=torch.device("cpu"),
        inference_model=SimpleNamespace(),
        placeholder_attn_layers=[],
        placeholder_linear_state_layers=[
            SimpleNamespace(layer_name="model.layers.0.linear_attn"),
            SimpleNamespace(layer_name="model.layers.2.linear_attn"),
        ],
    )
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=MambaSpec(),
                layer_names=["model.layers.0.linear_attn"],
            )
        ]
    )

    with (
        patch.object(
            GPUModelRunner,
            "initialize_kv_cache",
            side_effect=AssertionError("parent allocation ran before group validation"),
        ),
        pytest.raises(RuntimeError, match="missing.*model.layers.2.linear_attn"),
    ):
        runner.initialize_kv_cache(config)  # type: ignore[arg-type]
