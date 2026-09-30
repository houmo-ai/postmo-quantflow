import json
from types import SimpleNamespace

import pytest
import torch

from vllm_merak.config import MerakConfigParser
from vllm_merak.model_executor.model_loader import gemma4
from vllm_merak.model_executor.model_loader.merak_for_causal_lm import (
    MerakForCausalLM,
)
from vllm_merak.model_executor.register import get_registered_model_cls


def _write_unified_export(tmp_path) -> None:
    layer_types = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    hf_dir = tmp_path / "hf_config"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "model_type": "gemma4_unified",
                "image_token_id": 101,
                "audio_token_id": 102,
                "video_token_id": 103,
                "vision_config": {
                    "num_soft_tokens": 280,
                    "patch_size": 48,
                    "pooling_kernel_size": 1,
                },
                "video_config": {
                    "num_soft_tokens": 70,
                    "patch_size": 48,
                    "pooling_kernel_size": 1,
                },
                "audio_config": {"num_soft_tokens": 750},
                "text_config": {
                    "architectures": ["Gemma4ForCausalLM"],
                    "model_type": "gemma4_text",
                    "hidden_size": 128,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "num_hidden_layers": 3,
                    "vocab_size": 256,
                    "max_position_embeddings": 262144,
                    "layer_types": layer_types,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "merak_config.json").write_text(
        json.dumps({"xh_model": {"model_type": "hmonnx"}}),
        encoding="utf-8",
    )
    meta = {
        "attention_contract_version": 2,
        "layer_types": layer_types,
        "layer_cache_types": ["sliding_attention", "full_attention"],
        "layer_cache_indices": [0, 0, 1],
        "layer_cache_owner_indices": [0, 2],
        "layer_kv_shapes": [[1, 4, 1344, 128], [1, 4, 262464, 128]],
        "sliding_window": 1024,
        "sliding_kv_cache_input_mode": "slice_window",
        "prefill_chunk_length": 320,
        "bidirectional_vision_attention": True,
        "max_mm_ranges_per_chunk": 4,
        "model_config": {
            "model_type": "gemma4_unified",
            "spec_decode_mode": "none",
            "image_token_id": 101,
            "audio_token_id": 102,
            "video_token_id": 103,
        },
        "visual_config": {
            "hmonnx": "visual/visual.hmonnx",
            "num_image_tokens": 280,
            "frontend_kind": "encoder_free",
        },
        "video_visual_config": {
            "hmonnx": "video_visual/video_visual.hmonnx",
            "num_image_tokens": 70,
            "frontend_kind": "encoder_free",
        },
        "audio_config": {
            "hmonnx": "audio/audio.hmonnx",
            "input_feature_length": 750,
            "frontend_kind": "encoder_free",
        },
    }
    (tmp_path / "golden_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")


def test_unified_source_architecture_selects_dedicated_loader(tmp_path) -> None:
    _write_unified_export(tmp_path)

    config_dict, config = MerakConfigParser().parse(tmp_path, trust_remote_code=False)

    assert config_dict["architectures"] == ["Gemma4UnifiedMerakForCausalLM"]
    assert config.merak_loader_architecture == "Gemma4UnifiedMerakForCausalLM"
    assert config.audio_token_id == 102
    assert (
        get_registered_model_cls("Gemma4UnifiedForConditionalGeneration").__name__
        == "Gemma4UnifiedMerakForCausalLM"
    )


def test_unified_image_budget_is_checkpoint_owned(monkeypatch) -> None:
    info = gemma4.MerakGemma4UnifiedProcessingInfo.__new__(
        gemma4.MerakGemma4UnifiedProcessingInfo
    )
    info.ctx = SimpleNamespace(
        model_config=SimpleNamespace(
            get_multimodal_config=lambda: SimpleNamespace(limit_per_prompt={})
        ),
        get_merged_mm_kwargs=lambda _kwargs: {"max_soft_tokens": 64},
    )
    source_config = SimpleNamespace(
        vision_config=SimpleNamespace(num_soft_tokens=280),
        audio_config=None,
    )
    monkeypatch.setattr(info, "get_hf_config", lambda: source_config)

    budget = info.get_mm_max_tokens_per_item(262144, {"image": 1})

    assert budget is not None
    assert budget["image"] == 280


def test_unified_processor_rejects_runtime_soft_token_override() -> None:
    processor = object.__new__(gemma4.MerakGemma4UnifiedMultiModalProcessor)

    with pytest.raises(ValueError, match="checkpoint-owned"):
        processor._call_hf_processor(
            "<|image|>",
            {},
            {"images_kwargs": {"max_soft_tokens": 560}},
            {},
        )


def test_unified_audio_placeholder_uses_raw_640_sample_frames(monkeypatch) -> None:
    info = gemma4.MerakGemma4UnifiedProcessingInfo.__new__(
        gemma4.MerakGemma4UnifiedProcessingInfo
    )
    monkeypatch.setattr(
        info,
        "get_hf_config",
        lambda: SimpleNamespace(
            audio_config=SimpleNamespace(audio_samples_per_token=640),
            boa_token_id=10,
            eoa_token_index=11,
        ),
    )
    processor = SimpleNamespace(audio_seq_length=750, audio_token_id=12)

    details = info.get_audio_repl(audio_len=641, processor=processor)

    assert details.full == [10, 12, 12, 11]
    assert details.is_embed(None, details.full).tolist() == [
        False,
        True,
        True,
        False,
    ]


def _new_unified_model():
    model = gemma4.Gemma4UnifiedMerakForCausalLM.__new__(
        gemma4.Gemma4UnifiedMerakForCausalLM
    )
    model._runner_device = torch.device("cpu")
    model.input_embeddings = SimpleNamespace(weight=torch.empty(1, dtype=torch.float16))
    return model


def test_unified_image_and_video_graphs_are_mask_free_and_filter_on_host() -> None:
    calls: list[tuple[str, tuple[torch.Tensor, ...]]] = []

    class Graph:
        device = torch.device("cpu")
        dtype = torch.float16

        def __init__(self, name: str):
            self.name = name

        def forward(self, *args):
            calls.append((self.name, args))
            marker = 1 if self.name == "image" else 2
            return torch.full((1, 4, 3), marker, dtype=torch.float16)

    model = _new_unified_model()
    model.inference_model = SimpleNamespace(
        visual=Graph("image"), video_visual=Graph("video")
    )

    embeddings = model.embed_multimodal(
        pixel_values=torch.ones((1, 4, 8)),
        pixel_position_ids=torch.tensor([[[0, 0], [1, 0], [-1, -1], [-1, -1]]]),
        pixel_values_videos=torch.ones((2, 4, 8)),
        pixel_position_ids_videos=torch.tensor(
            [
                [[0, 0], [-1, -1], [-1, -1], [-1, -1]],
                [[0, 0], [1, 0], [-1, -1], [-1, -1]],
            ]
        ),
        video_frame_counts=torch.tensor([2]),
    )

    assert [name for name, _ in calls] == ["image", "video", "video"]
    assert all(len(args) == 2 for _, args in calls)
    assert torch.all(calls[0][1][1] >= 0)
    assert [tuple(item.shape) for item in embeddings] == [(2, 3), (3, 3)]
    assert embeddings[0].unique().item() == 1
    assert embeddings[1].unique().item() == 2


def test_unified_audio_graph_is_mask_free_and_filter_on_host() -> None:
    calls = []

    class Audio:
        device = torch.device("cpu")
        dtype = torch.float16

        @staticmethod
        def forward(*args):
            calls.append(args)
            return torch.arange(24, dtype=torch.float16).reshape(1, 4, 6)

    model = _new_unified_model()
    model.inference_model = SimpleNamespace(audio=Audio())

    (embedding,) = model.embed_multimodal(
        input_features_padded=torch.ones((1, 1, 4, 640)),
        input_features_mask=torch.tensor([[[True, True, False, False]]]),
    )

    assert len(calls) == 1 and len(calls[0]) == 1
    assert embedding.shape == (2, 6)


def test_unified_request_extracts_each_modality_embedding() -> None:
    model = _new_unified_model()
    model.image_token_id = 101
    model.audio_token_id = 102
    model.video_token_id = 103
    input_ids = torch.tensor([7, 101, 101, 8, 102, 102, 9, 103, 10])
    inputs_embeds = torch.arange(36, dtype=torch.float32).reshape(9, 4)

    kwargs, image_offset, video_offset = model._extract_request_multimodal_kwargs(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=None,
        image_grid_offset=0,
        video_grid_thw=None,
        video_grid_offset=0,
    )

    assert torch.equal(kwargs["image_embeds"], inputs_embeds[input_ids == 101])
    assert torch.equal(kwargs["audio_embeds"], inputs_embeds[input_ids == 102])
    assert torch.equal(kwargs["video_embeds"], inputs_embeds[input_ids == 103])
    assert image_offset == 0
    assert video_offset == 0


def test_page_attention_ranges_exclude_audio_tokens() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.config = SimpleNamespace(
        image_token_id=101, audio_token_id=102, video_token_id=103
    )
    model.image_token_id = 101
    model.audio_token_id = 102
    model.video_token_id = 103

    ranges = model._page_attention_visual_ranges(
        torch.tensor([101, 101, 7, 102, 102, 8, 103]),
        past_len=10,
    )

    assert ranges is not None
    assert ranges.tolist() == [[[10, 11], [16, 16]]]


def test_runtime_routes_all_modalities_to_matching_token_ids() -> None:
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.inference_device = torch.device("cpu")
    model.input_embeddings = SimpleNamespace(weight=torch.empty(1, dtype=torch.float16))
    model.logits_processor = SimpleNamespace(vocab_size=2)
    model.image_token_id = 101
    model.audio_token_id = 102
    model.video_token_id = 103
    model._prefer_input_ids_runtime = True
    model._ensure_runtime_cache_tensors = lambda: None
    captured = []

    class Processor:
        def __call__(self, data):
            captured.append(data)
            return (torch.zeros((1, 4, 2)),)

    class Runtime:
        def set_prefill(self):
            pass

        def get_input_sequence_length(self):
            return 4

        def get_data_preprocessor(self):
            return Processor()

        def forward(self, *args):
            return torch.zeros((1, 4, 2))

    model.inference_model = Runtime()
    input_ids = torch.tensor([101, 7, 102, 103])

    model._run_request(
        0,
        input_ids=input_ids,
        image_embeds=torch.ones((1, 2)),
        audio_embeds=torch.ones((1, 2)) * 2,
        video_embeds=torch.ones((1, 2)) * 3,
    )

    assert torch.equal(captured[0]["image_embeds"], torch.ones((1, 2)))
    assert torch.equal(captured[0]["audio_embeds"], torch.ones((1, 2)) * 2)
    assert torch.equal(captured[0]["video_embeds"], torch.ones((1, 2)) * 3)
