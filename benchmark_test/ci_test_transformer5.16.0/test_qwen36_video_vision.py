"""Video preprocessing and native/Merak ViT equivalence on tiny real models.

The session boundary runs the actual wrapped Torch graph on CPU and reuses an
output buffer. HMONNX export, device kernels and full LLM video generation are
outside this test's scope.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import VisualTokenGearHMONNXModel
from xhmodel_merak.xh_llm.models.qwen3_5_moe import _vision_model_impl  # noqa: F401
from xhmodel_merak.xh_llm.wrap_model import wrap_llm_model
from xhquant.api import ConfigDict


def _processor():
    return Qwen3VLVideoProcessor(
        patch_size=2, temporal_patch_size=2, merge_size=2,
        do_resize=False, do_sample_frames=False, image_mean=[0.5] * 3, image_std=[0.5] * 3,
    )


def _frames(count):
    return torch.randint(0, 256, (count, 3, 8, 12), dtype=torch.uint8, generator=torch.Generator().manual_seed(53))


def _reference_patches(frames):
    """Explicit spatial/temporal order independent of the processor reshape."""
    normalized = (frames.float() / 255 - 0.5) / 0.5
    if len(frames) % 2:
        normalized = torch.cat((normalized, normalized[-1:]), dim=0)
    patches = []
    for time in range(0, len(normalized), 2):
        for block_y in range(0, 8, 4):
            for block_x in range(0, 12, 4):
                for dy in (0, 2):
                    for dx in (0, 2):
                        patch = normalized[time:time + 2, :, block_y + dy:block_y + dy + 2, block_x + dx:block_x + dx + 2]
                        patches.append(patch.permute(1, 0, 2, 3).reshape(-1))
    return torch.stack(patches)


@pytest.mark.parametrize("count", [1, 2, 4, 5])
@pytest.mark.parametrize("container", ["torch", "numpy", "pil"])
def test_preprocess_preserves_real_frame_pairs_and_only_pads_the_tail(count, container):
    frames = _frames(count)
    pixels = frames.permute(0, 2, 3, 1).numpy()
    source = {"torch": frames, "numpy": pixels, "pil": [Image.fromarray(frame) for frame in pixels]}[container]
    processor = _processor()
    processor.max_frames = 2  # Sampling limits must not discard supplied frames.
    inputs = processor(videos=[source], do_sample_frames=False, return_tensors="pt")

    assert inputs["video_grid_thw"].tolist() == [[(count + 1) // 2, 4, 6]]
    torch.testing.assert_close(inputs["pixel_values_videos"], _reference_patches(frames), rtol=1e-6, atol=1e-6)


class _TorchSession:
    """Execute the real wrapped ViT at the device-session boundary."""

    def __init__(self, wrapped, capacity, out_dim):
        self.wrapped = wrapped
        self.device = torch.device("cpu")
        self.buffer = torch.empty(1, capacity, out_dim)
        self.calls = 0
        self.hmonnx_session = SimpleNamespace(get_input=lambda name: SimpleNamespace(
            dtype=torch.int64 if name in ("position_ids", "rotary_position_ids") else torch.float32
        ))

    def forward(self, *inputs):
        self.calls += 1
        self.buffer.copy_(self.wrapped(*inputs))
        return self.buffer


@pytest.fixture(scope="module")
def tiny_vision():
    torch.manual_seed(31)
    config = Qwen3_5MoeVisionConfig(
        depth=2, hidden_size=64, intermediate_size=80, num_heads=1,
        in_channels=3, patch_size=2, temporal_patch_size=2,
        spatial_merge_size=2, out_hidden_size=32, num_position_embeddings=16,
    )
    config._attn_implementation = "eager"
    native = Qwen3_5MoeVisionModel(config).float().eval()
    capacity = 8  # Six valid merged tokens plus padding, per temporal group.
    wrap_cfg = ConfigDict(dict(
        patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
        image_token_capacity=capacity, visual_rope_cache_length=16,
    ))
    wrapped = wrap_llm_model(copy.deepcopy(native), wrap_cfg).float().eval()
    runtime = VisualTokenGearHMONNXModel.__new__(VisualTokenGearHMONNXModel)
    runtime.visual_meta = SimpleNamespace(
        patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
        num_position_embeddings=16, visual_rope_cache_length=16,
    )
    runtime.gears = (capacity,)
    runtime.models = {capacity: _TorchSession(wrapped, capacity, config.out_hidden_size)}
    return native, runtime


@pytest.mark.parametrize("count", [1, 2, 4, 5])
def test_video_features_match_native_conv3d_and_temporal_attention(tiny_vision, count):
    native, runtime = tiny_vision
    frames = _frames(count)
    grid = torch.tensor([[(count + 1) // 2, 4, 6]])
    session = runtime.models[8]
    calls_before = session.calls
    with torch.no_grad():
        expected = native(_reference_patches(frames), grid).pooler_output
        actual = runtime.encode_video(_reference_patches(frames), grid)

    assert actual.shape == ((count + 1) // 2 * 6, 32)
    assert session.calls - calls_before == (count + 1) // 2
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_changing_second_frame_changes_video_features(tiny_vision):
    _, runtime = tiny_vision
    frames = _frames(2)
    grid = torch.tensor([[1, 4, 6]])
    actual = runtime.encode_video(_reference_patches(frames), grid)
    repeated = runtime.encode_video(_reference_patches(frames[:1].repeat(2, 1, 1, 1)), grid)
    assert not torch.allclose(actual, repeated, rtol=1e-5, atol=1e-6)


def test_processor_round_trip_uses_checkpoint_parameters(tmp_path):
    _processor().save_pretrained(tmp_path)
    from transformers import AutoVideoProcessor
    processor = AutoVideoProcessor.from_pretrained(tmp_path, local_files_only=True)
    assert processor.temporal_patch_size == 2
    assert processor.patch_size == 2
    inputs = processor(videos=[_frames(5)], do_sample_frames=False, return_tensors="pt")
    torch.testing.assert_close(inputs["pixel_values_videos"], _reference_patches(_frames(5)), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("grid,pixels,message", [
    (torch.tensor([[1, 4, 6], [1, 4, 6]]), torch.empty(48, 24), "one video_grid_thw"),
    (torch.tensor([[1., 4., 6.]]), torch.empty(24, 24), "integer dtype"),
    (torch.tensor([[0, 4, 6]]), torch.empty(0, 24), "positive"),
    (torch.tensor([[2, 3, 6]]), torch.empty(36, 24), "divisible"),
    (torch.tensor([[2, 4, 6]]), torch.empty(24, 24), "requires 48"),
    (torch.tensor([[1, 4, 6]]), torch.empty(1, 24, 24), "flattened patches"),
])
def test_video_rejects_invalid_geometry_before_encoding(tiny_vision, grid, pixels, message):
    _, runtime = tiny_vision
    calls = runtime.models[8].calls
    with pytest.raises(ValueError, match=message):
        runtime.encode_video(pixels, grid)
    assert runtime.models[8].calls == calls


def test_video_preserves_per_group_gear_overflow_error(tiny_vision):
    _, runtime = tiny_vision
    calls = runtime.models[8].calls
    with pytest.raises(ValueError, match="exceeds the largest configured gear"):
        runtime.encode_video(torch.empty(2 * 6 * 6, 24), torch.tensor([[2, 6, 6]]))


    assert runtime.models[8].calls == calls


def test_multiple_videos_keep_owned_outputs_and_input_order(tiny_vision):
    native, runtime = tiny_vision
    grids = torch.tensor([[2, 4, 6], [1, 4, 6]])
    patches = [_reference_patches(_frames(4)), _reference_patches(_frames(1))]
    with torch.no_grad():
        expected = [native(pixels, grid.reshape(1, 3)).pooler_output for pixels, grid in zip(patches, grids)]
        actual = runtime.encode_videos(torch.cat(patches), grids)
        saved = [value.clone() for value in actual]
        runtime.encode_video(patches[1] + 1, grids[1:])
    for output, reference, snapshot in zip(actual, expected, saved):
        torch.testing.assert_close(output, reference, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(output, snapshot)


@pytest.mark.parametrize("field", ["patch_size", "temporal_patch_size", "spatial_merge_size"])
def test_video_rejects_processor_runtime_geometry_mismatch(tiny_vision, field):
    _, runtime = tiny_vision
    processor = _processor()
    setattr(processor, "merge_size" if field == "spatial_merge_size" else field, 4)
    with pytest.raises(ValueError, match=field):
        runtime.validate_video_processor(processor)
