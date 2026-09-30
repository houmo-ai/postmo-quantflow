"""WeMM recipe and video contracts after the Qwen3.5 visual gear migration."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_catalog.registry import get_recipe
from xhmodel_merak.xh_llm.kv_cache_mixin import CacheList
from xhmodel_merak.xh_llm.models.wemm_embedding.data_preprocess import WeMMEmbeddingDataPreprocess
from xhmodel_merak.xh_llm.models.wemm_embedding.processor import (
    normalize_embedding_pooling_anchor,
    validate_embedding_pooling_anchor,
    video_frame_grids,
)


@pytest.mark.parametrize(
    ("size", "hidden_size", "num_layers"),
    (("2b", 2048, 24), ("4b", 2560, 32), ("9b", 4096, 32)),
)
def test_wemm_recipe_resolves_patch_gear_workflow(tmp_path, size, hidden_size, num_layers):
    model_dir = tmp_path / f"WeMM-Embedding-{size.upper()}"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {"hidden_size": hidden_size, "num_hidden_layers": num_layers},
            }
        ),
        encoding="utf-8",
    )
    recipe = get_recipe(f"wemm-embedding-{size}")
    plan = resolve_build_plan(
        model_id=recipe.model_id,
        model_dir=str(model_dir),
        output_dir=str(tmp_path / f"artifact-{size}"),
        device="cpu",
    )
    assert plan.vision_gears == (96, 196, 384, 704, 1536)
    assert plan.components == ("visual", "prefill")
    assert plan.quant_method == "autoround"
    assert recipe.modalities == ("text", "image", "video")
    workflow = yaml.safe_load(Path(plan.config_path).read_text(encoding="utf-8"))
    assert workflow["export"]["model"]["visual_config"]["visual_input_mode"] == "patches"
    assert workflow["export"]["model"]["visual_config"]["image_token_capacity"] == 1536
    assert workflow["export"]["model"]["quant_scheme"]["ops"] == {}
    assert workflow["export"]["model"]["visual_config"]["quant_scheme"]["ops"] == {}
    assert workflow["quant"]["calibration"]["nsamples"] == 256
    assert workflow["quant"]["iters"] == 400


def test_wemm_video_frame_grids_preserve_patch_order_and_reject_mismatch():
    grids = torch.tensor([[2, 2, 2], [1, 2, 4]])
    patches = torch.arange(16 * 6, dtype=torch.float32).reshape(16, 6)
    frame_grids = video_frame_grids(patches, grids, 2)
    torch.testing.assert_close(frame_grids, torch.tensor([[1, 2, 2], [1, 2, 2], [1, 2, 4]]))
    torch.testing.assert_close(grids, torch.tensor([[2, 2, 2], [1, 2, 4]]))
    assert [part.shape[0] for part in torch.split(patches, [int(grid.prod()) for grid in frame_grids])] == [4, 4, 8]
    with pytest.raises(ValueError, match="do not match"):
        video_frame_grids(patches[:-1], grids, 2)


@pytest.mark.parametrize(
    ("padding_side", "input_ids", "attention_mask", "expected"),
    (
        ("right", [[3, 9, 11, 0]], [[1, 1, 1, 0]], [[3, 11, 9, 0]]),
        ("left", [[0, 4, 9, 11]], [[0, 1, 1, 1]], [[0, 4, 11, 9]]),
    ),
)
def test_wemm_processor_moves_transformers_turn_suffix_before_pooling_anchor(
    padding_side,
    input_ids,
    attention_mask,
    expected,
):
    class Tokenizer:
        pad_token_id = 0

        @staticmethod
        def encode(text, *, add_special_tokens):
            assert text == "<embedding>"
            assert add_special_tokens is False
            return [9]

    tokenizer = Tokenizer()
    tokenizer.padding_side = padding_side
    model_inputs = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
    }

    normalize_embedding_pooling_anchor(model_inputs, tokenizer)
    validate_embedding_pooling_anchor(model_inputs, tokenizer)

    torch.testing.assert_close(model_inputs["input_ids"], torch.tensor(expected))


@pytest.mark.parametrize(
    ("input_ids", "expected_ids", "expected_mask"),
    (
        ([[1, 2, 3]], [[1, 2, 3, 9]], [[1, 1, 1, 1]]),
        ([[1, 9, 9]], [[1, 9, 0]], [[1, 1, 0]]),
    ),
)
def test_wemm_processor_normalizes_missing_or_duplicate_pooling_anchor(
    input_ids,
    expected_ids,
    expected_mask,
):
    class Tokenizer:
        pad_token_id = 0
        padding_side = "right"

        @staticmethod
        def encode(_text, *, add_special_tokens):
            assert add_special_tokens is False
            return [9]

    model_inputs = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    normalize_embedding_pooling_anchor(model_inputs, Tokenizer())
    validate_embedding_pooling_anchor(model_inputs, Tokenizer())
    torch.testing.assert_close(model_inputs["input_ids"], torch.tensor(expected_ids))
    torch.testing.assert_close(model_inputs["attention_mask"], torch.tensor(expected_mask))


def test_wemm_video_positions_and_features_use_local_preprocessor():
    embedding = torch.nn.Embedding(32, 4)
    processor = WeMMEmbeddingDataPreprocess(
        token_embedding=embedding,
        input_sequence_length=8,
        past_key_caches=CacheList(),
        past_value_caches=CacheList(),
        past_conv_caches=CacheList(),
        past_recurrent_states=CacheList(),
        image_token_id=3,
        video_token_id=5,
        vision_start_token_id=4,
        vision_end_token_id=6,
        spatial_merge_size=2,
    )
    input_ids = torch.tensor([[4, 5, 6, 4, 5, 6, 1]])
    video_grid = torch.tensor([[2, 2, 2]])
    video_features = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    outputs = processor(
        {
            "input_ids": input_ids,
            "video_embeds": video_features,
            "video_grid_thw": video_grid,
            "past_seq_length": 0,
        }
    )
    torch.testing.assert_close(outputs[0][0, [1, 4]], video_features)
    assert not torch.equal(outputs[1][1:2], outputs[1][4:5])
    torch.testing.assert_close(video_grid, torch.tensor([[2, 2, 2]]))


def test_wemm_runtime_routes_image_and_video_patches_through_visual_gears(monkeypatch):
    from xhmodel_merak.xh_llm.models.wemm_embedding import hmonnx_inference

    monkeypatch.setattr(hmonnx_inference, "LLMInferenceContextManager", lambda *args, **kwargs: nullcontext())
    model_cls = hmonnx_inference.XHWeMMEmbeddingHMONNXModel
    image_pixels = torch.arange(4 * 12, dtype=torch.float32).reshape(4, 12)
    video_pixels = torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12)
    calls = []

    class Visual:
        def encode_many(self, pixels, grids):
            calls.append((pixels, grids.clone()))
            value = float(len(calls))
            return [torch.full((1, int(grid.prod()) // 4, 3), value) for grid in grids]

    class Processor:
        def apply_chat_template(self, messages, add_generation_prompt=False):
            assert messages[0]["role"] == "user"
            assert not add_generation_prompt
            return {
                "input_ids": torch.tensor([[3, 5, 5, 1]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "pixel_values": image_pixels,
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "pixel_values_videos": video_pixels,
                "video_grid_thw": torch.tensor([[2, 2, 2]]),
            }

    def preprocess(data):
        torch.testing.assert_close(data["image_embeds"], torch.ones(1, 3))
        torch.testing.assert_close(data["video_embeds"], torch.full((2, 3), 2.0))
        return ()

    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        embed_tokens=torch.nn.Embedding(16, 3),
        visual=Visual(),
        meta_info=SimpleNamespace(
            embedding_dimension=3,
            model_config=SimpleNamespace(prefill_chunk_length=256, context_max_length=2048, spatial_merge_size=2),
        ),
        to=lambda device: None,
        set_prefill=lambda: None,
        get_data_preprocessor=lambda: preprocess,
        forward=lambda: torch.tensor([[[0.0, 0.0, 0.0]] * 3 + [[3.0, 4.0, 0.0]]]),
        _messages=model_cls._messages,
        _validate_input_length=model_cls._validate_input_length,
        _flatten_visual_features=model_cls._flatten_visual_features,
        _pool_hidden_states=model_cls._pool_hidden_states,
    )
    embedding = model_cls.embed(runtime, {"image": "image.jpg", "video": "video.mp4"}, processor=Processor())
    torch.testing.assert_close(embedding, torch.tensor([[0.6, 0.8, 0.0]]))
    assert len(calls) == 2
    torch.testing.assert_close(calls[0][0], image_pixels)
    torch.testing.assert_close(calls[0][1], torch.tensor([[1, 2, 2]]))
    torch.testing.assert_close(calls[1][0], video_pixels)
    torch.testing.assert_close(calls[1][1], torch.tensor([[1, 2, 2], [1, 2, 2]]))


def test_wemm_public_inference_adapter_returns_serializable_embedding(monkeypatch, tmp_path):
    import xhmodel_merak.xh_llm as xh_llm
    from xhmodel_merak.model_integrations.wemm_embedding.inference import adapter

    video = tmp_path / "clip.mp4"
    video.touch()

    class Model:
        def embed(self, item):
            assert item == {"text": "describe", "video": str(video)}
            return torch.tensor([[0.6, 0.8]], dtype=torch.float32)

    monkeypatch.setattr(xh_llm.AutoLLMHONNXModel, "from_pretrained", lambda *args, **kwargs: Model())
    result = adapter.run(
        recipe=get_recipe("wemm-embedding-2b"),
        root=tmp_path,
        manifest={},
        meta_file=tmp_path / "export_meta_info.json",
        inputs={"text": "describe", "video": str(video)},
        device="cpu",
        components=("visual", "prefill"),
        selected_golden=(),
        golden_artifacts={},
    )
    assert result.to_dict()["embedding"] == pytest.approx([0.6, 0.8])
    assert result.to_dict()["stats"] == {"dimension": 2}
