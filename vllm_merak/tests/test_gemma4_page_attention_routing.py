from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)

from vllm_merak.attention.backends.merak import (
    MerakAttentionMetadata,
    MerakPageAttentionGroupContext,
)
from vllm_merak.hmfp_kv_cache import (
    HMFPFullAttentionSpec,
    HMFPPagedKVCache,
    HMFPSlidingWindowSpec,
    allocate_hmfp_paged_kv_cache,
    write_hmfp_kv_cache_by_slot,
)
from vllm_merak.model_executor.model_loader.gemma4_cache_contract import (
    Gemma4CacheContract,
    Gemma4LayerCacheBinding,
)
from vllm_merak.model_executor.model_loader.merak_for_causal_lm import MerakForCausalLM
from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1


def _binding(
    layer_idx: int,
    *,
    kind: str,
    group: int,
    cache: int,
    window: int | None,
    scheduler_block_size: int = 64,
) -> Gemma4LayerCacheBinding:
    return Gemma4LayerCacheBinding(
        layer_idx=layer_idx,
        layer_name=f"placeholder_attn_layers.{layer_idx}",
        attention_kind=kind,
        cache_group_idx=group,
        cache_index=cache,
        kv_shape=(1, 1, 1344, 64),
        sliding_window=window,
        scheduler_block_size=scheduler_block_size,
        physical_block_size=64,
    )


def _model() -> MerakForCausalLM:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.inference_device = torch.device("cpu")
    model.gemma4_cache_contract = Gemma4CacheContract(
        version=2,
        prefill_chunk_size=320,
        bindings=(
            _binding(0, kind="full_attention", group=0, cache=0, window=None),
            _binding(1, kind="sliding_attention", group=1, cache=1, window=1024),
            _binding(2, kind="sliding_attention", group=1, cache=1, window=1024),
        ),
        cache_shapes={0: (1, 1, 4096, 64), 1: (1, 1, 1344, 64)},
    )
    model.page_attention_bindings = model.gemma4_cache_contract.bindings
    model.placeholder_attn_layers = [
        SimpleNamespace(layer_name="placeholder_attn_layers.0", kv_cache="full-cache"),
        SimpleNamespace(layer_name="placeholder_attn_layers.1", kv_cache="local-cache"),
        SimpleNamespace(layer_name="placeholder_attn_layers.2", kv_cache="local-cache"),
    ]
    model._get_block_size = lambda: 64
    return model


def _meta(tag: str, *, past: int = 1500, current: int = 320, block: int = 0):
    metadata = MerakAttentionMetadata(
        past_lens=torch.tensor([past], dtype=torch.int32),
        subsequence_begins=torch.tensor([0, current], dtype=torch.int32),
        block_indices=torch.tensor([block, block + 1], dtype=torch.int32),
        block_indices_begins=torch.tensor([0, 2], dtype=torch.int32),
        max_context_len=torch.tensor(past + current, dtype=torch.int32),
        slot_mapping=torch.arange(current, dtype=torch.int64) + block * 64,
        seq_lens=torch.tensor([past + current], dtype=torch.int32),
    )
    metadata.tag = tag
    return metadata


@contextmanager
def _forward_context(attn_metadata):
    with patch(
        "vllm_merak.model_executor.model_loader.merak_for_causal_lm.get_forward_context",
        return_value=SimpleNamespace(attn_metadata=attn_metadata),
    ):
        yield


def test_metadata_resolves_by_binding_not_dictionary_order() -> None:
    model = _model()
    full = _meta("full", block=3)
    local = _meta("local", block=7)
    context = {
        "placeholder_attn_layers.2": local,
        "placeholder_attn_layers.1": _meta("local", block=7),
        "placeholder_attn_layers.0": full,
    }

    with _forward_context(context):
        grouped = model._get_attn_metadata_by_group()

    assert grouped[0].tag == "full"
    assert grouped[1].tag == "local"


def test_partial_group_metadata_raises_for_missing_bound_layer() -> None:
    model = _model()
    context = {
        "placeholder_attn_layers.2": _meta("local", block=7),
        "placeholder_attn_layers.0": _meta("full", block=3),
    }

    with _forward_context(context):
        with pytest.raises(
            RuntimeError, match=r"cache_group_idx=1.*placeholder_attn_layers\.1"
        ):
            model._get_attn_metadata_by_group()


def test_missing_group_metadata_raises_instead_of_using_first_value() -> None:
    model = _model()
    with _forward_context({"placeholder_attn_layers.0": _meta("full")}):
        with pytest.raises(RuntimeError, match="cache_group_idx=1"):
            model._get_attn_metadata_by_group()


def test_group_contexts_use_native_sliding_base_and_filtered_visual_ranges() -> None:
    model = _model()
    contexts = model._build_page_attention_group_contexts(
        metadata_by_group={
            0: _meta("full", past=1500, current=320, block=1),
            1: _meta("local", past=1500, current=320, block=9),
        },
        query_lengths=torch.tensor([320]),
        visual_ranges=torch.tensor([[[1490, 1769], [50, 60]]], dtype=torch.int32),
    )

    assert contexts[0].kv_window_start_abs is None
    assert contexts[0].mm_prefix_ranges is None
    assert contexts[1].kv_window_start_abs.tolist() == [477]
    assert contexts[1].seq_lens.tolist() == [1820]
    assert contexts[1].mm_prefix_ranges.tolist() == [[[1490, 1769], [0, 0]]]
    assert contexts[0].block_ids.tolist() == [1, 2]
    assert contexts[1].block_ids.tolist() == [0, 1]
    assert contexts[0].block_ids.data_ptr() != contexts[1].block_ids.data_ptr()
    assert contexts[0].slot_mapping.data_ptr() != contexts[1].slot_mapping.data_ptr()


def test_gemma4_runtime_reconstructs_exported_multimodal_token_types() -> None:
    model = _model()
    model.image_token_id = 101
    model.video_token_id = 102
    model.audio_token_id = 103

    token_types = model._gemma4_mm_token_type_ids(
        torch.tensor([7, 101, 101, 8, 102, 103], dtype=torch.int32)
    )

    assert token_types.dtype == torch.long
    assert token_types.shape == (1, 6)
    # Gemma4's compact attention contract treats only image/video as the
    # bidirectional visual block. Audio deliberately remains causal.
    assert token_types.tolist() == [[0, 1, 1, 0, 2, 0]]


def test_sliding_group_projects_global_scheduler_ids_into_fixed_ring() -> None:
    model = _model()
    full = _meta("full", past=0, current=283, block=2)
    full.block_indices = torch.tensor([6, 5, 4, 3, 2], dtype=torch.int32)
    local = _meta("local", past=0, current=283, block=1)
    local.block_indices = torch.tensor([1, 7, 8, 9, 10], dtype=torch.int32)

    context = model._build_page_attention_group_contexts(
        metadata_by_group={0: full, 1: local},
        query_lengths=torch.tensor([283]),
    )[1]

    assert context.block_ids.tolist() == [0, 1, 2, 3, 4]
    assert context.slot_mapping.tolist() == list(range(283))
    assert max(context.block_ids.tolist()) < 21


def test_sliding_zeroing_uses_logical_ring_pages_not_scheduler_ids() -> None:
    model = _model()
    cache = allocate_hmfp_paged_kv_cache(
        num_blocks=21,
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        device="cpu",
    )
    for tensor in cache.tensors():
        tensor.fill_(7)
    model.placeholder_attn_layers[1].kv_cache = cache
    model.placeholder_attn_layers[2].kv_cache = cache

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.model = model
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=HMFPFullAttentionSpec(
                    block_size=64,
                    num_kv_heads=1,
                    head_size=64,
                    head_size_v=64,
                    dtype=torch.uint8,
                    sefp_cache_dtype="sefp_k8_v8",
                ),
                layer_names=["placeholder_attn_layers.0"],
            ),
            SimpleNamespace(
                kv_cache_spec=HMFPSlidingWindowSpec(
                    block_size=64,
                    num_kv_heads=1,
                    head_size=64,
                    head_size_v=64,
                    dtype=torch.uint8,
                    sefp_cache_dtype="sefp_k8_v8",
                    sliding_window=1024,
                ),
                layer_names=[
                    "placeholder_attn_layers.1",
                    "placeholder_attn_layers.2",
                ],
            ),
        ]
    )

    runner._zero_gemma4_records(
        [
            {
                "request_id": "image",
                "group_idx": 1,
                "block_ids": [21, 22, 23, 24, 25],
                "logical_block_start": 0,
            }
        ]
    )

    for tensor in cache.tensors():
        assert torch.count_nonzero(tensor[:5]) == 0
        assert torch.count_nonzero(tensor[5:]) > 0


def test_full_metadata_keeps_vllm_projected_physical_64_pages() -> None:
    model = _model()
    full = _binding(
        0,
        kind="full_attention",
        group=0,
        cache=0,
        window=None,
        scheduler_block_size=128,
    )
    model.gemma4_cache_contract = Gemma4CacheContract(
        version=2,
        prefill_chunk_size=320,
        bindings=(full, *model.gemma4_cache_contract.bindings[1:]),
        cache_shapes=model.gemma4_cache_contract.cache_shapes,
    )
    model.page_attention_bindings = model.gemma4_cache_contract.bindings
    metadata = _meta("full", past=4096, current=1, block=321)
    metadata.block_indices = torch.tensor([642, 643, 716, 717], dtype=torch.int32)
    metadata.slot_mapping = torch.tensor([358 * 128], dtype=torch.int64)

    context = model._build_page_attention_group_contexts(
        metadata_by_group={
            0: metadata,
            1: _meta("local", past=4096, current=1, block=7),
        },
        query_lengths=torch.tensor([1]),
    )[0]

    assert context.block_ids.tolist() == [642, 643, 716, 717]
    assert context.slot_mapping.tolist() == [358 * 128]
    assert context.slot_mapping.item() == 716 * 64


def test_chunk_context_accepts_projected_full_table_for_4097_tokens() -> None:
    model = _model()
    physical_ids = torch.arange(642, 708, dtype=torch.int64)
    context = model._chunk_page_attention_group_contexts(
        {
            0: MerakPageAttentionGroupContext(
                block_ids=physical_ids,
                slot_mapping=torch.tensor([4096], dtype=torch.int64),
                seq_lens=torch.tensor([4097], dtype=torch.int64),
            )
        },
        chunk_start=0,
        chunk_end=1,
        chunk_past_len=4096,
        context_length=1,
        chunk_visual_ranges=None,
    )[0]

    assert context.block_ids.tolist() == physical_ids.tolist()
    assert context.slot_mapping.tolist() == [4096]


def test_chunk_context_rejects_short_physical_table_instead_of_alias_padding() -> None:
    model = _model()
    with pytest.raises(RuntimeError, match="repeating the final page would alias"):
        model._chunk_page_attention_group_contexts(
            {
                0: MerakPageAttentionGroupContext(
                    block_ids=torch.arange(33, dtype=torch.int64),
                    slot_mapping=torch.tensor([0], dtype=torch.int64),
                    seq_lens=torch.tensor([4097], dtype=torch.int64),
                )
            },
            chunk_start=0,
            chunk_end=1,
            chunk_past_len=4096,
            context_length=1,
            chunk_visual_ranges=None,
        )


def _contexts(blocks: int, *, ranges: int = 0):
    block_ids = torch.arange(blocks, dtype=torch.int64)
    mm = torch.tensor([[[4, 7]]], dtype=torch.int32) if ranges else None
    return {
        0: MerakPageAttentionGroupContext(
            block_ids=block_ids.clone(),
            slot_mapping=torch.tensor([64], dtype=torch.int64),
            seq_lens=torch.tensor([65], dtype=torch.int64),
            kv_window_start_abs=None,
            mm_prefix_ranges=None,
        ),
        1: MerakPageAttentionGroupContext(
            block_ids=block_ids.clone(),
            slot_mapping=torch.tensor([128], dtype=torch.int64),
            seq_lens=torch.tensor([65], dtype=torch.int64),
            kv_window_start_abs=torch.tensor([0], dtype=torch.int64),
            mm_prefix_ranges=mm,
        ),
    }


def test_decode_context_buffers_keep_identity_until_capacity_growth(
    monkeypatch,
) -> None:
    model = _model()

    class Runtime:
        def __init__(self):
            self.clear_count = 0

        def clear(self, clear_disabled_reason=True):
            del clear_disabled_reason
            self.clear_count += 1

    runtime = Runtime()
    model.inference_model = runtime

    first = model._stage_group_contexts(_contexts(4))
    first_ptr = first[1].block_ids.data_ptr()

    def forbid_item(*_args, **_kwargs):
        raise AssertionError("staging must not call Tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", forbid_item)
    second = model._stage_group_contexts(_contexts(4))
    assert second[1].block_ids.data_ptr() == first_ptr
    assert second[1].mm_prefix_ranges is None
    assert runtime.clear_count == 0

    third = model._stage_group_contexts(_contexts(9))
    assert runtime.clear_count == 1
    replay = model._stage_group_contexts(_contexts(9))
    assert replay[1].block_ids.data_ptr() == third[1].block_ids.data_ptr()
    assert runtime.clear_count == 1


def test_shared_context_buffer_growth_invalidates_prefill_and_decode_graphs() -> None:
    model = _model()

    class Interpreter:
        def __init__(self):
            self.clear_count = 0

        def clear(self, clear_disabled_reason=True):
            del clear_disabled_reason
            self.clear_count += 1

    prefill_interpreter = Interpreter()
    decode_interpreter = Interpreter()
    prefill_model = SimpleNamespace(
        hmonnx_session=SimpleNamespace(interpreter=prefill_interpreter)
    )
    decode_model = SimpleNamespace(
        hmonnx_session=SimpleNamespace(interpreter=decode_interpreter)
    )
    model.inference_model = SimpleNamespace(
        _llm_prefill=True,
        prefill_model=prefill_model,
        prefill_models={"prefill": prefill_model},
        decode_model=decode_model,
    )

    model._stage_group_contexts(_contexts(4))
    model._stage_group_contexts(_contexts(9))

    assert prefill_interpreter.clear_count == 1
    assert decode_interpreter.clear_count == 1


def test_decode_context_shape_is_stable_inside_capacity_bucket() -> None:
    model = _model()

    class Runtime:
        def __init__(self):
            self.clear_count = 0

        def clear(self, clear_disabled_reason=True):
            del clear_disabled_reason
            self.clear_count += 1

    model.inference_model = Runtime()
    first = model._stage_group_contexts(_contexts(5))
    second = model._stage_group_contexts(_contexts(7))

    assert first[1].block_ids.data_ptr() == second[1].block_ids.data_ptr()
    assert first[1].block_ids.shape == second[1].block_ids.shape == (8,)
    assert model.inference_model.clear_count == 0


def test_visual_prefill_uses_same_cuda_graph_path_as_text() -> None:
    model = _model()
    assert model._page_attention_execution_mode(_contexts(4, ranges=1)) == "cuda_graph"
    assert model._page_attention_execution_mode(_contexts(4, ranges=0)) == "cuda_graph"


def test_page_attention_modules_bind_by_explicit_layer_identity_not_order() -> None:
    model = _model()
    modules = {
        0: SimpleNamespace(layer_idx=0),
        1: SimpleNamespace(layer_idx=1),
        2: SimpleNamespace(layer_idx=2),
    }
    model._find_page_attention_modules = lambda _runtime: (
        ("pa.2", modules[2]),
        ("pa.0", modules[0]),
        ("pa.1", modules[1]),
    )

    bound = model._bound_page_attention_modules(object())

    assert [(binding.layer_idx, module.layer_idx) for binding, module in bound] == [
        (0, 0),
        (1, 1),
        (2, 2),
    ]


def test_page_attention_modules_without_explicit_identity_fail_closed() -> None:
    model = _model()
    model._find_page_attention_modules = lambda _runtime: (
        ("page_attention.0", SimpleNamespace()),
        ("page_attention.1", SimpleNamespace()),
        ("page_attention.2", SimpleNamespace()),
    )

    with pytest.raises(RuntimeError, match="no explicit layer identity"):
        model._bound_page_attention_modules(object())


@dataclass
class _V2PageAttentionContext:
    paged_kv_cache: object
    block_ids: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    kv_window_start_abs: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    mm_prefix_ranges: torch.Tensor | None = None


def test_contract_v2_fast_path_is_keyed_by_cache_index_and_full_has_no_overlay() -> (
    None
):
    model = _model()
    captured = {}

    class Runtime:
        def set_page_attention_context(self, **kwargs):
            captured.update(kwargs)

    staged = model._stage_group_contexts(_contexts(4, ranges=1))
    with patch(
        "xhquant.xhonnxruntime.parsers.page_attention.PageAttentionContext",
        _V2PageAttentionContext,
    ):
        model._set_runtime_page_attention_group_contexts(Runtime(), staged)

    assert set(captured) == {"contexts_by_cache_index"}
    assert set(captured["contexts_by_cache_index"]) == {0, 1}
    assert captured["contexts_by_cache_index"][0].paged_kv_cache == "full-cache"
    assert captured["contexts_by_cache_index"][0].kv_window_start_abs is None
    assert captured["contexts_by_cache_index"][0].mm_prefix_ranges is None
    assert captured["contexts_by_cache_index"][1].paged_kv_cache == "local-cache"
    assert captured["contexts_by_cache_index"][1].mm_prefix_ranges.tolist() == [
        [[4, 7]]
    ]


def test_contract_v2_fast_path_publishes_every_cache_in_same_group() -> None:
    model = _model()
    bindings = (
        _binding(0, kind="full_attention", group=0, cache=0, window=None),
        _binding(1, kind="sliding_attention", group=1, cache=1, window=1024),
        _binding(2, kind="sliding_attention", group=1, cache=2, window=1024),
    )
    model.gemma4_cache_contract = Gemma4CacheContract(
        version=2,
        prefill_chunk_size=320,
        bindings=bindings,
        cache_shapes={
            0: (1, 1, 4096, 64),
            1: (1, 1, 1344, 64),
            2: (1, 1, 1344, 64),
        },
    )
    model.page_attention_bindings = bindings
    model.placeholder_attn_layers[2].kv_cache = "local-cache-2"
    captured = {}

    class Runtime:
        def set_page_attention_context(self, **kwargs):
            captured.update(kwargs)

    with patch(
        "xhquant.xhonnxruntime.parsers.page_attention.PageAttentionContext",
        _V2PageAttentionContext,
    ):
        model._set_runtime_page_attention_group_contexts(
            Runtime(), model._stage_group_contexts(_contexts(4))
        )

    contexts = captured["contexts_by_cache_index"]
    assert set(contexts) == {0, 1, 2}
    assert contexts[1].paged_kv_cache == "local-cache"
    assert contexts[2].paged_kv_cache == "local-cache-2"
    assert contexts[1].block_ids.data_ptr() == contexts[2].block_ids.data_ptr()


def test_contract_v2_rejects_legacy_runtime_context_api() -> None:
    model = _model()

    class LegacyRuntime:
        def set_page_attention_context(
            self, paged_kv_caches, block_ids, slot_mapping, block_size
        ):
            del paged_kv_caches, block_ids, slot_mapping, block_size

    with patch(
        "xhquant.xhonnxruntime.parsers.page_attention.PageAttentionContext",
        _V2PageAttentionContext,
    ):
        with pytest.raises(RuntimeError, match="contract-v2.*contexts_by_cache_index"):
            model._set_runtime_page_attention_group_contexts(
                LegacyRuntime(), model._stage_group_contexts(_contexts(4))
            )


def test_gemma4_prefill_decode_release_and_recycle_keep_group_contract_end_to_end(
    monkeypatch,
    tmp_path,
) -> None:
    """Drive Task 7 through the production coordinators and scheduler lifecycle."""

    model = _model()
    model_trace_path = tmp_path / "model-trace.jsonl"
    kv_trace_path = tmp_path / "kv-trace.jsonl"
    monkeypatch.setenv("VLLM_MERAK_GEMMA4_TRACE_JSONL", str(model_trace_path))
    monkeypatch.setenv("VLLM_MERAK_KV_TRACE_JSONL", str(kv_trace_path))
    monkeypatch.setenv("VLLM_MERAK_TRACE_RUN_ID", "end-to-end-run")
    full_cache = allocate_hmfp_paged_kv_cache(
        num_blocks=128,
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        device="cpu",
    )
    local_cache = allocate_hmfp_paged_kv_cache(
        num_blocks=128,
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        device="cpu",
    )
    model.placeholder_attn_layers[0].kv_cache = full_cache
    model.placeholder_attn_layers[1].kv_cache = local_cache
    model.placeholder_attn_layers[2].kv_cache = local_cache
    model.input_embeddings = SimpleNamespace(weight=torch.empty(1, dtype=torch.float32))
    model.logits_processor = SimpleNamespace(vocab_size=4)
    model._prefer_input_ids_runtime = True
    model._ensure_runtime_cache_tensors = lambda: None
    model.max_mm_ranges_per_chunk = 2

    class Runtime:
        def __init__(self):
            self.page_modules = [
                SimpleNamespace(layer_idx=layer_idx, context=None)
                for layer_idx in range(3)
            ]
            self.context_steps = []
            self.execution_modes = []
            self.request_modes = []
            self.past_seq_lengths = []
            self.forward_lengths = []
            self.clear_count = 0

        def set_prefill(self):
            self.request_modes.append("prefill")

        def set_decode(self):
            self.request_modes.append("decode")

        def get_input_sequence_length(self):
            return 320

        def get_data_preprocessor(self):
            def preprocess(data):
                self.past_seq_lengths.append(data["past_seq_length"])
                return (data["input_ids"],)

            return preprocess

        def set_page_attention_context(self, *, contexts_by_cache_index):
            for binding, page_module in zip(
                model.page_attention_bindings,
                self.page_modules,
                strict=True,
            ):
                page_module.context = contexts_by_cache_index[binding.cache_index]
            self.context_steps.append(
                {
                    cache_index: SimpleNamespace(
                        paged_kv_cache=context.paged_kv_cache,
                        block_ids=context.block_ids.clone(),
                        slot_mapping=context.slot_mapping.clone(),
                        kv_window_start_abs=(
                            context.kv_window_start_abs.clone()
                            if context.kv_window_start_abs is not None
                            else None
                        ),
                        seq_lens=(
                            context.seq_lens.clone()
                            if context.seq_lens is not None
                            else None
                        ),
                        mm_prefix_ranges=(
                            context.mm_prefix_ranges.clone()
                            if context.mm_prefix_ranges is not None
                            else None
                        ),
                        block_ptr=context.block_ids.data_ptr(),
                        slot_ptr=context.slot_mapping.data_ptr(),
                    )
                    for cache_index, context in contexts_by_cache_index.items()
                }
            )

        def set_page_attention_execution_mode(self, mode):
            self.execution_modes.append(mode)

        def clear(self, clear_disabled_reason=True):
            del clear_disabled_reason
            self.clear_count += 1

        def get_page_attention_execution_state(self):
            return {
                "requested_mode": self.execution_modes[-1],
                "has_captured_graph": self.execution_modes[-1] == "cuda_graph",
                "replay_active": self.execution_modes.count("cuda_graph") > 1,
                "clear_count": self.clear_count,
            }

        def forward(self, input_ids):
            self.forward_lengths.append(input_ids.numel())
            return torch.arange(4, dtype=torch.float32)

    runtime = Runtime()
    model.inference_model = runtime

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
    # Size each pool to the first request's peak plus the native null block so
    # request 2 must receive released IDs, not unused tail capacity.
    full_pool = BlockPool(51, enable_caching=False, hash_block_size=64)
    sliding_pool = BlockPool(22, enable_caching=False, hash_block_size=64)
    full_manager = FullAttentionManager(
        full_spec,
        block_pool=full_pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=64,
    )
    sliding_bound = sliding_spec.max_admission_blocks_per_request(320, 32768)
    sliding_manager = SlidingWindowManager(
        sliding_spec,
        block_pool=sliding_pool,
        enable_caching=False,
        kv_cache_group_id=1,
        scheduler_block_size=64,
        max_admission_blocks_per_request=sliding_bound,
    )

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=full_spec,
                layer_names=["placeholder_attn_layers.0"],
            ),
            SimpleNamespace(
                kv_cache_spec=sliding_spec,
                layer_names=[
                    "placeholder_attn_layers.1",
                    "placeholder_attn_layers.2",
                ],
            ),
        ]
    )
    runner.model = model
    runner.requests = {}
    runner.num_prompt_logprobs = {}
    runner.late_interaction_runner = SimpleNamespace(
        on_requests_finished=lambda _req_ids: None
    )
    runner.encoder_cache = {}
    runner.speculative_config = None
    runner.use_async_spec_decode = False
    runner.use_async_scheduling = False
    runner._may_reorder_batch = lambda _scheduler_output: None
    runner.merak_linear_state_idx = {}

    class InputBatch:
        req_id_to_index = {}
        req_ids = []

        def remove_request(self, _req_id):
            raise AssertionError("empty scheduler step removed a request")

        def condense(self):
            pass

        def refresh_metadata(self):
            pass

    runner.input_batch = InputBatch()

    backing_visits = []
    original_named_tensors = HMFPPagedKVCache.named_tensors

    def counted_named_tensors(cache):
        backing_visits.append(cache.k_qdata.data_ptr())
        yield from original_named_tensors(cache)

    monkeypatch.setattr(HMFPPagedKVCache, "named_tensors", counted_named_tensors)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )

    class ObservableParentZeroer:
        def __init__(self):
            self.calls = []

        def zero_block_ids(self, block_ids):
            # Merak's dynamic-dispatch override must clear every typed backing
            # before delegating to vLLM's standard zeroer.
            for cache in (full_cache, local_cache):
                for tensor in cache.tensors():
                    assert torch.count_nonzero(tensor[block_ids]) == 0
            self.calls.append(list(block_ids))

    parent_zeroer = ObservableParentZeroer()
    runner._kv_block_zeroer = parent_zeroer

    def scheduler_output(block_ids):
        cached = CachedRequestData(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            preempted_req_ids=set(),
            new_block_ids_to_zero=list(block_ids),
        )

    def group_context(manager, token_positions):
        blocks = manager.req_to_blocks["request-1"]
        slot_mapping = torch.tensor(
            [
                blocks[position // 64].block_id * 64 + position % 64
                for position in token_positions
            ],
            dtype=torch.int64,
        )
        return MerakPageAttentionGroupContext(
            block_ids=torch.tensor(
                [block.block_id for block in blocks], dtype=torch.int64
            ),
            slot_mapping=slot_mapping,
            seq_lens=torch.tensor([len(token_positions)], dtype=torch.int64),
            kv_window_start_abs=None,
            mm_prefix_ranges=None,
        )

    full_owned_history = []
    sliding_owned_history = []
    first_full_ids = set()
    first_sliding_ids = set()
    prefill_step_offset = None
    for past in range(0, 3200, 320):
        full_manager.remove_skipped_blocks("request-1", past)
        sliding_manager.remove_skipped_blocks("request-1", past)
        new_full = full_manager.allocate_new_blocks("request-1", past + 320, past + 320)
        new_sliding = sliding_manager.allocate_new_blocks(
            "request-1", past + 320, past + 320
        )
        first_full_ids.update(block.block_id for block in new_full)
        first_sliding_ids.update(block.block_id for block in new_sliding)
        new_ids = sorted({block.block_id for block in (*new_full, *new_sliding)})
        ids_tensor = torch.tensor(new_ids, dtype=torch.long)
        for cache in (full_cache, local_cache):
            for tensor in cache.tensors():
                tensor.index_fill_(0, ids_tensor, 13)

        visit_start = len(backing_visits)
        runner._update_states(scheduler_output(new_ids))
        assert backing_visits[visit_start:] == [
            full_cache.k_qdata.data_ptr(),
            local_cache.k_qdata.data_ptr(),
        ]
        assert parent_zeroer.calls[-1] == new_ids
        for cache in (full_cache, local_cache):
            for tensor in cache.tensors():
                assert torch.count_nonzero(tensor[new_ids]) == 0

        full_owned_history.append(
            sum(
                block is not full_pool.null_block
                for block in full_manager.req_to_blocks["request-1"]
            )
        )
        sliding_owned_history.append(
            sum(
                block is not sliding_pool.null_block
                for block in sliding_manager.req_to_blocks["request-1"]
            )
        )

        if past + 320 == 1600:
            prefill_step_offset = len(runtime.context_steps)
            contexts = {
                0: group_context(full_manager, range(1600)),
                1: group_context(sliding_manager, range(1600)),
            }
            with patch(
                "xhquant.xhonnxruntime.parsers.page_attention.PageAttentionContext",
                _V2PageAttentionContext,
            ):
                output = model._run_request(
                    0,
                    input_ids=torch.arange(1600, dtype=torch.long),
                    page_attention_group_contexts=contexts,
                    visual_ranges=[(300, 579)],
                )
            assert output.tolist() == [0.0, 1.0, 2.0, 3.0]

    assert full_owned_history == list(range(5, 51, 5))
    assert sliding_bound == 22
    assert sliding_owned_history[:4] == [5, 10, 15, 20]
    assert sliding_owned_history[4:] == [21] * 6
    assert max(sliding_owned_history) <= sliding_bound

    # The coordinator must consume the atomic plan and publish each chunk.
    assert prefill_step_offset is not None
    prefill_steps = runtime.context_steps[prefill_step_offset:]
    assert runtime.request_modes == ["prefill"]
    assert runtime.past_seq_lengths == [0, 300, 620, 940, 1260, 1580]
    assert runtime.forward_lengths == [300, 320, 320, 320, 320, 20]
    assert runtime.execution_modes == ["cuda_graph"] * 6
    assert len(prefill_steps) == 6
    visual_step = prefill_steps[1]
    assert visual_step[0].kv_window_start_abs is None
    assert visual_step[0].mm_prefix_ranges is None
    assert visual_step[1].kv_window_start_abs.tolist() == [0]
    assert visual_step[1].seq_lens.tolist() == [620]
    assert visual_step[1].mm_prefix_ranges.tolist() == [[[300, 579], [0, 0]]]

    # Drive a normal one-token decode through the same production chain. Its
    # metadata comes from the native managers' current physical block tables.
    decode_contexts = {
        0: group_context(full_manager, [3134]),
        1: group_context(sliding_manager, [3134]),
    }
    with patch(
        "xhquant.xhonnxruntime.parsers.page_attention.PageAttentionContext",
        _V2PageAttentionContext,
    ):
        output = model._run_request(
            3134,
            input_ids=torch.tensor([1], dtype=torch.long),
            page_attention_group_contexts=decode_contexts,
        )
    assert output.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert runtime.request_modes[-1] == "decode"
    assert runtime.past_seq_lengths[-1] == 3134
    assert runtime.forward_lengths[-1] == 1
    assert runtime.execution_modes[-1] == "cuda_graph"

    decode_step = runtime.context_steps[-1]
    full = decode_step[0]
    local = decode_step[1]
    assert full.kv_window_start_abs is None
    assert full.mm_prefix_ranges is None
    assert local.kv_window_start_abs.tolist() == [2111]
    assert local.seq_lens.tolist() == [3135]
    assert local.mm_prefix_ranges.tolist() == [[[0, 0], [0, 0]]]
    assert local.paged_kv_cache is local_cache
    assert full.paged_kv_cache is full_cache
    assert full.paged_kv_cache is not local.paged_kv_cache
    assert full.block_ptr != local.block_ptr
    assert full.slot_ptr != local.slot_ptr
    assert model.inference_model.page_modules[1].context is (
        model.inference_model.page_modules[2].context
    )

    full_manager.free("request-1")
    sliding_manager.free("request-1")
    second_full = full_manager.allocate_new_blocks("request-2", 320, 320)
    second_sliding = sliding_manager.allocate_new_blocks("request-2", 320, 320)
    second_ids = sorted({block.block_id for block in (*second_full, *second_sliding)})
    assert {block.block_id for block in second_full} <= first_full_ids
    assert {block.block_id for block in second_sliding} <= first_sliding_ids
    second_ids_tensor = torch.tensor(second_ids, dtype=torch.long)
    for cache in (full_cache, local_cache):
        for tensor in cache.tensors():
            tensor.index_fill_(0, second_ids_tensor, 29)
    assert torch.count_nonzero(local_cache.v_qdata[second_ids]) > 0

    visit_start = len(backing_visits)
    runner._update_states(scheduler_output(second_ids))
    assert backing_visits[visit_start:] == [
        full_cache.k_qdata.data_ptr(),
        local_cache.k_qdata.data_ptr(),
    ]
    assert parent_zeroer.calls[-1] == second_ids
    for cache in (full_cache, local_cache):
        for tensor in cache.tensors():
            assert torch.count_nonzero(tensor[second_ids]) == 0

    recycled_block = second_sliding[0].block_id
    write_hmfp_kv_cache_by_slot(
        torch.zeros((1, 1, 64)),
        torch.ones((1, 1, 64)),
        local_cache,
        torch.tensor([recycled_block * 64]),
    )
    assert torch.count_nonzero(local_cache.v_qdata[recycled_block]) > 0

    from vllm_merak.gemma4_trace import flush_traces

    flush_traces()
    model_events = [
        __import__("json").loads(line)
        for line in model_trace_path.read_text().splitlines()
    ]
    plans = [event for event in model_events if event["event"] == "request_plan"]
    chunks = [event for event in model_events if event["event"] == "chunk_context"]
    assert plans[0]["chunk_plan"] == [
        [0, 300],
        [300, 620],
        [620, 940],
        [940, 1260],
        [1260, 1580],
        [1580, 1600],
    ]
    assert plans[0]["inclusive_mm_ranges"] == [[300, 579]]
    visual_local = next(
        event
        for event in chunks
        if event["chunk"] == [300, 620] and event["cache_group_idx"] == 1
    )
    assert visual_local["execution_mode"] == "cuda_graph"
    assert visual_local["kv_window_start_abs"] == [0]
    assert visual_local["seq_lens"] == [620]
    assert visual_local["mm_prefix_ranges"] == [[[300, 579], [0, 0]]]
    assert set(visual_local["metadata_ptrs"]) == {
        "block_ids",
        "slot_mapping",
        "seq_lens",
        "kv_window_start_abs",
        "mm_prefix_ranges",
    }
    decode_local = next(
        event
        for event in reversed(chunks)
        if event["phase"] == "decode" and event["cache_group_idx"] == 1
    )
    assert decode_local["execution_mode"] == "cuda_graph"
    assert decode_local["kv_window_start_abs"] == [2111]
    assert decode_local["seq_lens"] == [3135]
    assert decode_local["mm_prefix_ranges"] == [[[0, 0], [0, 0]]]
    assert decode_local["graph_runtime"] == {
        "requested_mode": "cuda_graph",
        "has_captured_graph": True,
        "replay_active": True,
        "clear_count": 1,
    }

    kv_events = [
        __import__("json").loads(line)
        for line in kv_trace_path.read_text().splitlines()
    ]
    zero_events = [event for event in kv_events if event["event"] == "zero_complete"]
    assert zero_events
    assert zero_events[-1]["zeroed_block_ids"] == second_ids
    assert zero_events[-1]["ordering"] == "before_model_execute"
    assert len(zero_events[-1]["unique_backings"]) == 2
    assert {item["ptr"] for item in zero_events[-1]["unique_backings"]} == {
        full_cache.k_qdata.data_ptr(),
        local_cache.k_qdata.data_ptr(),
    }


def test_gemma4_kv_trace_records_release_ownership_and_recycle(
    monkeypatch, tmp_path
) -> None:
    trace_path = tmp_path / "kv-lifecycle.jsonl"
    monkeypatch.setenv("VLLM_MERAK_KV_TRACE_JSONL", str(trace_path))
    monkeypatch.setenv("VLLM_MERAK_TRACE_RUN_ID", "kv-lifecycle-run")
    model = _model()
    full_cache = allocate_hmfp_paged_kv_cache(
        num_blocks=16,
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        device="cpu",
    )
    local_cache = allocate_hmfp_paged_kv_cache(
        num_blocks=16,
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        device="cpu",
    )
    model.placeholder_attn_layers[0].kv_cache = full_cache
    model.placeholder_attn_layers[1].kv_cache = local_cache
    model.placeholder_attn_layers[2].kv_cache = local_cache
    full_spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    local_spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.model = model
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=full_spec,
                layer_names=["placeholder_attn_layers.0"],
            ),
            SimpleNamespace(
                kv_cache_spec=local_spec,
                layer_names=[
                    "placeholder_attn_layers.1",
                    "placeholder_attn_layers.2",
                ],
            ),
        ]
    )
    runner.requests = {"req-1": SimpleNamespace(block_ids=[[1, 2], [7, 8]])}
    runner.merak_linear_state_idx = {}
    runner.input_batch = SimpleNamespace(req_ids=["req-1"])
    runner._kv_block_zeroer = SimpleNamespace(zero_block_ids=lambda _ids: None)

    def parent_update(actual_runner, scheduler_output):
        for req_id in scheduler_output.finished_req_ids:
            actual_runner.requests.pop(req_id, None)
        if scheduler_output.new_block_ids_to_zero:
            actual_runner._zero_block_ids(scheduler_output.new_block_ids_to_zero)

    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states",
        parent_update,
    )

    def output(*, finished=(), new=()):
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=[],
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[],
                num_computed_tokens=[],
                num_output_tokens=[],
            ),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(finished),
            free_encoder_mm_hashes=[],
            preempted_req_ids=set(),
            new_block_ids_to_zero=list(new),
        )

    # Model execution can skip a finish-only scheduler output. Snapshot request
    # 1 while active, then emulate the parent persistent batch replacing it.
    runner._update_states(output())
    runner.requests.pop("req-1")
    runner.requests["req-2"] = SimpleNamespace(block_ids=[[1], [7]])
    runner.input_batch.req_ids = ["req-2"]
    for cache in (full_cache, local_cache):
        for tensor in cache.tensors():
            tensor[7].fill_(9)
    runner._update_states(output(new={7}))

    from vllm_merak.gemma4_trace import flush_traces

    flush_traces()
    events = [
        __import__("json").loads(line) for line in trace_path.read_text().splitlines()
    ]
    release = next(event for event in events if event["event"] == "release")
    ownership = next(
        event
        for event in events
        if event["event"] == "ownership" and event["request_id"] == "req-2"
    )
    zero = next(event for event in events if event["event"] == "zero_complete")
    assert release["request_id"] == "req-1"
    assert release["released_block_ids_by_group"] == {"0": [1, 2], "1": [7, 8]}
    assert ownership["request_id"] == "req-2"
    assert ownership["owned_count_by_group"] == {"0": 1, "1": 1}
    assert zero["new_block_ids"] == [7]
    assert zero["recycled_block_ids"] == [7]
    assert zero["zeroed_block_ids"] == [7]
    assert len(zero["unique_backings"]) == 2
