from types import SimpleNamespace

import pytest
import torch

from vllm_merak.attention.backends.merak import MerakPageAttentionGroupContext
from vllm_merak.model_executor.model_loader.gemma4 import (
    _visual_ranges_for_chunk,
    plan_gemma4_prefill_chunks,
)
from vllm_merak.model_executor.model_loader.gemma4_cache_contract import (
    Gemma4CacheContract,
    Gemma4LayerCacheBinding,
)
from vllm_merak.model_executor.model_loader.merak_for_causal_lm import MerakForCausalLM


def test_atomic_planner_moves_boundary_before_visual_span_then_fills_next_chunk():
    assert plan_gemma4_prefill_chunks(700, 320, [(300, 579)]) == (
        (0, 300),
        (300, 620),
        (620, 700),
    )


def test_atomic_planner_accepts_maximum_280_token_span():
    assert plan_gemma4_prefill_chunks(640, 320, [(40, 319)]) == (
        (0, 320),
        (320, 640),
    )


@pytest.mark.parametrize("visual_ranges", [[(10, 290)], [(20, 19)]])
def test_atomic_planner_rejects_oversized_or_invalid_spans(visual_ranges):
    match = "280" if visual_ranges[0][0] < visual_ranges[0][1] else "inclusive"
    with pytest.raises(ValueError, match=match):
        plan_gemma4_prefill_chunks(700, 320, visual_ranges)


def test_atomic_planner_rejects_overlapping_and_upstream_split_ranges():
    with pytest.raises(ValueError, match="overlap"):
        plan_gemma4_prefill_chunks(700, 320, [(10, 20), (20, 30)])
    with pytest.raises(RuntimeError, match="visual span.*atomic"):
        plan_gemma4_prefill_chunks(320, 320, [(300, 579)])


def test_visual_ranges_for_chunk_are_absolute_inclusive_and_fixed_capacity():
    ranges = _visual_ranges_for_chunk(
        300,
        620,
        [(300, 579), (700, 710)],
        max_ranges=3,
    )

    assert ranges is not None
    assert ranges.dtype == torch.int32
    assert ranges.tolist() == [[[300, 579], [0, 0], [0, 0]]]
    assert _visual_ranges_for_chunk(620, 700, [(300, 579)], 3).tolist() == [
        [[0, 0], [0, 0], [0, 0]]
    ]


def test_visual_ranges_for_chunk_rejects_capacity_overflow():
    with pytest.raises(ValueError, match="capacity=1"):
        _visual_ranges_for_chunk(0, 320, [(10, 20), (30, 40)], 1)


def _binding(
    layer_idx: int, *, kind: str, group: int, cache: int, window: int | None
) -> Gemma4LayerCacheBinding:
    return Gemma4LayerCacheBinding(
        layer_idx=layer_idx,
        layer_name=f"placeholder_attn_layers.{layer_idx}",
        attention_kind=kind,
        cache_group_idx=group,
        cache_index=cache,
        kv_shape=(1, 1, 1344, 64),
        sliding_window=window,
    )


def _gemma_model(runtime_len: int = 320) -> MerakForCausalLM:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.inference_device = torch.device("cpu")
    model.input_embeddings = SimpleNamespace(weight=torch.empty(1, dtype=torch.float16))
    model.logits_processor = SimpleNamespace(vocab_size=2)
    model._prefer_input_ids_runtime = True
    model.max_mm_ranges_per_chunk = 2
    model.gemma4_cache_contract = Gemma4CacheContract(
        version=2,
        prefill_chunk_size=320,
        bindings=(
            _binding(0, kind="full_attention", group=0, cache=0, window=None),
            _binding(1, kind="sliding_attention", group=1, cache=1, window=1024),
        ),
        cache_shapes={0: (1, 1, 4096, 64), 1: (1, 1, 1344, 64)},
    )
    model.page_attention_bindings = model.gemma4_cache_contract.bindings

    class Runtime:
        def __init__(self):
            self.executed = False

        def set_prefill(self):
            pass

        def get_input_sequence_length(self):
            return runtime_len

        def get_data_preprocessor(self):
            self.executed = True
            raise AssertionError("must reject before HMONNX preprocessing")

    model.inference_model = Runtime()
    model._ensure_runtime_cache_tensors = lambda: (_ for _ in ()).throw(
        AssertionError("must reject before cache mutation")
    )
    return model


def test_run_request_rejects_scheduler_chunk_cutting_visual_span():
    model = _gemma_model()
    contexts = {
        0: MerakPageAttentionGroupContext(
            block_ids=torch.arange(10),
            slot_mapping=torch.arange(320),
            seq_lens=torch.tensor([320]),
            kv_window_start_abs=None,
            mm_prefix_ranges=None,
        ),
        1: MerakPageAttentionGroupContext(
            block_ids=torch.arange(10),
            slot_mapping=torch.arange(320),
            seq_lens=torch.tensor([320]),
            kv_window_start_abs=torch.tensor([0]),
            mm_prefix_ranges=torch.tensor([[[300, 579], [0, 0]]], dtype=torch.int32),
        ),
    }

    with pytest.raises(RuntimeError, match="visual span.*atomic"):
        model._run_request(
            0,
            input_ids=torch.arange(320),
            page_attention_group_contexts=contexts,
            visual_ranges=[(300, 579)],
        )

    assert not model.inference_model.executed


def test_run_request_rejects_visual_range_capacity_before_cache_mutation():
    model = _gemma_model()
    model.max_mm_ranges_per_chunk = 1

    with pytest.raises(ValueError, match="capacity=1"):
        model._run_request(
            0,
            input_ids=torch.arange(320),
            visual_ranges=[(10, 20), (30, 40)],
        )

    assert not model.inference_model.executed


def test_run_request_rejects_atomic_cut_before_embedding_preparation():
    model = _gemma_model()
    model._prefer_input_ids_runtime = False
    effects = []

    def fail_embedding(**_kwargs):
        effects.append("embedding")
        raise AssertionError("must reject before embedding preparation")

    model._prepare_request_embeds = fail_embedding

    with pytest.raises(RuntimeError, match="visual span.*atomic"):
        model._run_request(
            0,
            input_ids=torch.arange(320),
            inputs_embeds=torch.zeros((320, 1)),
            visual_ranges=[(300, 579)],
        )

    assert effects == []


def test_run_request_routes_multimodal_token_types_into_exported_preprocessor():
    model = _gemma_model()
    model.image_token_id = 101
    model.video_token_id = 102
    model.audio_token_id = 103
    model._ensure_runtime_cache_tensors = lambda: None
    captured = []

    class Runtime:
        def set_prefill(self):
            pass

        def get_input_sequence_length(self):
            return 320

        def get_data_preprocessor(self):
            def preprocess(data):
                captured.append(data)
                return (torch.zeros((1, 320, 2), dtype=torch.float16),)

            return preprocess

        def forward(self, _inputs):
            return torch.zeros((1, 1, 2), dtype=torch.float16)

    model.inference_model = Runtime()
    model._run_request(
        0,
        input_ids=torch.tensor([7, 101, 101, 8, 102, 103]),
        visual_ranges=[(1, 2), (4, 4)],
    )

    assert len(captured) == 1
    assert captured[0]["mm_token_type_ids"].tolist() == [[0, 1, 1, 0, 2, 0]]


@pytest.mark.parametrize("enable_page_attention", [False, True])
def test_public_forward_rejects_atomic_cut_before_any_side_effect(
    enable_page_attention,
):
    model = _gemma_model()
    model.enable_page_attention = enable_page_attention
    metadata = SimpleNamespace(
        past_lens=torch.tensor([0], dtype=torch.int64),
        subsequence_begins=torch.tensor([0, 320], dtype=torch.int64),
    )
    model._get_attn_metadata_by_group = lambda: {0: metadata}
    model._common_page_attention_metadata = lambda _grouped: metadata
    model._page_attention_visual_ranges = lambda *_args, **_kwargs: torch.tensor(
        [[[300, 579]]], dtype=torch.int32
    )
    effects = []

    def fail(effect):
        def callback(*_args, **_kwargs):
            effects.append(effect)
            raise AssertionError(f"must reject before {effect}")

        return callback

    model._prepare_request_embeds = fail("embedding")
    model._ensure_runtime_cache_tensors = fail("cache ensure")
    model._load_runtime_request_cache_from_paged = fail("cache load")
    model.inference_model.forward = fail("HMONNX")

    with pytest.raises(RuntimeError, match="visual span.*atomic"):
        model.forward(
            input_ids=torch.arange(320),
            inputs_embeds=torch.zeros((320, 1)),
        )

    assert effects == []
