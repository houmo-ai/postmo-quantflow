# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Video decoding, native positions and feature fusion at runtime boundaries."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from xhmodel_merak.xh_llm.models.qwen3_5.data_preprocess import Qwen3_5_DataPreprocess
from xhmodel_merak.xh_llm.models.qwen3_5.video_inputs import build_video_inputs, load_video


def processor_files(path):
    from _unified_small_model import read_spec, write_processor
    from transformers import AutoVideoProcessor, PreTrainedTokenizerFast
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_processor import XHQwen3_5Processor

    path.mkdir(exist_ok=True)
    write_processor(path, read_spec("qwen_moe"))
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
    template = "{% for m in messages %}{% for c in m['content'] %}{% if c['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>{% else %}{{ c['text'] }}{% endif %}{% endfor %}{% endfor %}"
    tokenizer.chat_template = template
    tokenizer.save_pretrained(path)
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    Qwen3VLVideoProcessor(patch_size=2, temporal_patch_size=2, merge_size=2,
                         do_resize=False, min_frames=2, max_frames=8, fps=2).save_pretrained(path)
    processor = XHQwen3_5Processor.from_pretrained(path)
    processor.chat_template = template
    assert isinstance(processor.video_processor, type(AutoVideoProcessor.from_pretrained(path)))
    return processor


def test_directory_frames_build_native_timestamps_and_placeholders(tmp_path):
    processor = processor_files(tmp_path / "processor")
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(5):
        Image.fromarray(np.full((8, 12, 3), i * 40, np.uint8)).save(frames / f"{i:02}.png")
    result = build_video_inputs(processor, frames, "hello", False)
    assert result["video_grid_thw"].tolist() == [[3, 4, 6]]
    assert int((result.input_ids == processor.video_token_id).sum()) == 18
    assert int((result.input_ids == processor.vision_start_token_id).sum()) == 3


def test_file_decoding_uses_native_sampling_and_real_frame_data(tmp_path):
    import av
    processor = processor_files(tmp_path / "processor")
    path = tmp_path / "video.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width, stream.height, stream.pix_fmt = 12, 8, "yuv420p"
        for color in (30, 90, 150, 210):
            frame = av.VideoFrame.from_ndarray(np.full((8, 12, 3), color, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    frames, metadata = load_video(processor, path)
    expected = processor.video_processor.sample_frames(metadata=metadata)
    assert list(metadata.frames_indices) == list(expected)
    assert tuple(frames.shape) == (len(expected), 3, 8, 12)
    for frame, index in zip(frames, expected):
        assert abs(float(np.asarray(frame).mean()) - (30, 90, 150, 210)[index]) < 5
    result = build_video_inputs(processor, path, "hello", False)
    assert result["video_grid_thw"].tolist() == [[len(expected) // 2, 4, 6]]


def _preprocessor(capacity=32):
    from xhmodel_merak.xh_llm.types import CacheList
    return Qwen3_5_DataPreprocess(
        token_embedding=torch.nn.Embedding(64, 4), input_sequence_length=capacity,
        image_token_id=4, video_token_id=5, vision_start_token_id=6, vision_end_token_id=7,
        spatial_merge_size=2, past_key_caches=CacheList([]), past_value_caches=CacheList([]),
        past_conv_caches=CacheList([]), past_recurrent_states=CacheList([]),
    )


def test_timestamped_video_rope_matches_native_and_preserves_input_grid():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    processor = _preprocessor()
    ids = torch.tensor([[1, 6, 5, 5, 7, 2, 6, 5, 5, 7, 3]])
    grids = torch.tensor([[2, 2, 4]])
    native = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)),
                             get_vision_position_ids=Qwen3_5Model.get_vision_position_ids)
    # Bind the native helper without allocating an unrelated language model.
    native.get_vision_position_ids = Qwen3_5Model.get_vision_position_ids.__get__(native)
    expected = Qwen3_5Model.get_rope_index(native, ids, (ids == 5).long() * 2, video_grid_thw=grids)
    actual = processor.get_rope_index(ids, video_grid_thw=grids)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)
    assert grids.tolist() == [[2, 2, 4]]


@pytest.mark.parametrize("capacity", [8, 32])
def test_video_features_reach_short_and_chunked_hf_prefill(capacity):
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import _Qwen3_5HFCompatible
    processor = _preprocessor(capacity)
    ids = torch.tensor([[1, 6, 5, 5, 7, 2, 6, 5, 5, 7, 3]])
    features = torch.arange(16).reshape(4, 4).float() + 100
    seen = []
    class Visual:
        def encode_videos(self, pixels, grid):
            assert grid.tolist() == [[2, 2, 4]]
            return [pixels]
    def forward(*args):
        seen.append(args[0].detach().clone())
        return (torch.zeros(1, capacity, 64),)
    owner = SimpleNamespace(visual=Visual(), get_data_preprocessor=lambda: processor,
                            get_input_sequence_length=lambda: capacity, forward=forward,
                            get_num_logits_to_keep=lambda: 0)
    compatible = SimpleNamespace(_llm_model=owner, _past_seq_length=0,
                                 get_input_embeddings=lambda: processor.embed_tokens)
    _Qwen3_5HFCompatible.forward(compatible, input_ids=ids, pixel_values_videos=features,
                               video_grid_thw=torch.tensor([[2, 2, 4]]))
    actual = torch.cat(seen, dim=1)[:, :ids.shape[1]]
    torch.testing.assert_close(actual[ids == 5], features)
    torch.testing.assert_close(actual[ids != 5], processor.embed_tokens(ids)[ids != 5])


def test_video_feature_token_mismatch_is_rejected():
    processor = _preprocessor()
    with pytest.raises(ValueError, match="Video features and video tokens do not match"):
        processor({"input_ids": torch.tensor([[6, 5, 7]]), "video_embeds": torch.ones(2, 4), "past_seq_length": 0})
