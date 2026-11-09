# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Execute production plugin methods against the real tiny Merak ViT.

Load only the methods under test so the CPU contracts do not require vLLM's
separate CUDA installation. Visual computation uses the wrapped native model.
"""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, cast

import pytest
import torch
from test_qwen36_video_vision import _frames, _reference_patches, tiny_vision  # noqa: F401


@pytest.fixture
def model(tiny_vision):
    source = Path(__file__).parents[2] / "vllm_merak/vllm_merak/model_executor/model_loader/qwen3_5.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5MerakForCausalLM")
    names = {"embed_multimodal", "_normalize_visual_embeds", "_count_multimodal_items",
             "_extract_request_image_embeds", "_extract_request_multimodal_kwargs"}
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in cls.body} == names
    cls.bases, cls.keywords, cls.decorator_list = [], [], []
    namespace = dict(torch=torch, Any=Any, Optional=Optional, cast=cast, MultiModalEmbeddings=tuple,
                     determinism_trace_enabled=lambda: False,
                     hmonnx_op_trace_scope=lambda *args, **kwargs: nullcontext())
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    model = namespace[cls.name]()
    model.supports_multimodal = True
    model.inference_model = SimpleNamespace(visual=tiny_vision[1])
    model._runner_device = "cpu"
    model.input_embeddings = torch.nn.Embedding(64, 32)
    model.image_token_id, model.video_token_id, model.vision_start_token_id = 4, 5, 6
    model._flatten_input_ids = lambda values: values.reshape(-1)
    model._flatten_inputs_embeds = lambda values: values.reshape(-1, values.shape[-1])
    return model


@pytest.mark.parametrize("video_first", [False, True])
def test_mixed_images_and_videos_follow_vllm_field_order(model, tiny_vision, video_first):
    native, _ = tiny_vision
    image, video = _reference_patches(_frames(1)), _reference_patches(_frames(5))
    images = dict(pixel_values=image, image_grid_thw=torch.tensor([[1, 4, 6]]))
    videos = dict(pixel_values_videos=video, video_grid_thw=torch.tensor([[3, 4, 6]]))
    with torch.no_grad():
        expected = [native(image, images["image_grid_thw"]).pooler_output,
                    native(video, videos["video_grid_thw"]).pooler_output]
        output = model.embed_multimodal(**(videos | images if video_first else images | videos))
    if video_first:
        expected.reverse()
    assert len(output) == 2
    for actual, reference in zip(output, expected):
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)


def test_multiple_video_items_use_runtime_group_encoding(model, tiny_vision):
    _, runtime = tiny_vision
    before = runtime.models[8].calls
    patches = torch.cat([_reference_patches(_frames(4)), _reference_patches(_frames(5))])
    with torch.no_grad():
        result = model.embed_multimodal(pixel_values_videos=patches, video_grid_thw=torch.tensor([[2, 4, 6], [3, 4, 6]]))
    assert [tuple(value.shape) for value in result] == [(12, 32), (18, 32)]
    assert runtime.models[8].calls - before == 5


@pytest.mark.parametrize("fields", [
    {"pixel_values_videos": torch.empty(48, 24)},
    {"video_grid_thw": torch.tensor([[2, 4, 6]])},
    {"pixel_values_videos": torch.empty(24, 24), "video_grid_thw": torch.tensor([[2, 4, 6]])},
    {"pixel_values_videos": torch.empty(48, 24), "video_grid_thw": torch.tensor([[2., 4., 6.]])},
])
def test_video_capability_rejects_incomplete_or_malformed_inputs(model, tiny_vision, fields):
    runtime = tiny_vision[1]
    before = runtime.models[8].calls
    with pytest.raises(ValueError):
        model.embed_multimodal(**fields)
    assert runtime.models[8].calls == before


def extract(model, ids, *, video_grids=None, offset=0, embeds=True):
    values = torch.arange(len(ids) * 32).reshape(len(ids), 32).float()
    result = model._extract_request_multimodal_kwargs(
        input_ids=torch.tensor(ids), inputs_embeds=values if embeds else None,
        image_grid_thw=None, image_grid_offset=0, video_grid_thw=video_grids, video_grid_offset=offset,
    )
    return result, values


def test_timestamp_spans_consume_temporal_group_grids_without_losing_request_offsets(model):
    grids = torch.tensor([[2, 4, 6], [1, 2, 4]])
    ids = [1, 6] + [5] * 6 + [7, 2, 6] + [5] * 6 + [7]
    (first, _, offset), values = extract(model, ids, video_grids=grids)
    assert offset == 2
    assert first["video_grid_thw"].tolist() == [[1, 4, 6], [1, 4, 6]]
    torch.testing.assert_close(first["video_embeds"], values[torch.tensor(ids) == 5])
    (second, _, offset), _ = extract(model, [6, 5, 5, 7], video_grids=grids, offset=offset)
    assert offset == 3 and second["video_grid_thw"].tolist() == [[1, 2, 4]]
    assert grids.tolist() == [[2, 4, 6], [1, 2, 4]]


@pytest.mark.parametrize("ids", [[5, 5, 7], [6, 5, 5, 7]])
def test_resumed_video_chunk_routes_cached_embeddings_without_grid(model, ids):
    (kwargs, _, offset), values = extract(model, ids, offset=2)
    assert offset == 2 and "video_grid_thw" not in kwargs
    torch.testing.assert_close(kwargs["video_embeds"], values[torch.tensor(ids) == 5])


def test_missing_video_embeddings_and_insufficient_grids_fail(model):
    with pytest.raises(ValueError, match="missing merged video embeddings"):
        extract(model, [6, 5, 5, 7], embeds=False)
    with pytest.raises(ValueError, match="video group grids"):
        extract(model, [6, 5, 7, 6, 5, 7], video_grids=torch.tensor([[1, 2, 4]]))
