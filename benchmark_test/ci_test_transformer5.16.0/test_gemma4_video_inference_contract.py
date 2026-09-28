"""Validate local video inputs against real Gemma4 processors."""
import os
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="module")
def unified_processor():
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_processor import XHGemma4Processor

    model_dir = Path(os.getenv("GEMMA4_UNIFIED_PROCESSOR_DIR", "/data01/datasets/gemma-4-12B-it"))
    if not model_dir.is_dir():
        pytest.skip(f"Local Gemma4 Unified processor assets unavailable: {model_dir}")
    return XHGemma4Processor.from_pretrained(str(model_dir), local_files_only=True)


def test_gemma4_12b_inference_frame_directory_metadata_is_processor_compatible(unified_processor, tmp_path):
    from xhmodel_merak.model_integrations.gemma4.inference import _load_video

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for index in range(4):
        Image.new("RGB", (320, 240), color=(index, index, index)).save(frames_dir / f"{index:02d}.png")
    frames, metadata = _load_video(frames_dir, max_frames=32)

    inputs = unified_processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "video_metadata": metadata},
                    {"type": "text", "text": "describe the video"},
                ],
            }
        ]
    )

    assert tuple(inputs["pixel_values_videos"].shape) == (1, 4, 70, 6912)
    assert metadata["frames_indices"] == [0, 1, 2, 3]


def test_gemma4_12b_short_mp4_uses_compatible_torchcodec(unified_processor, tmp_path):
    from xhmodel_merak.model_integrations.gemma4.inference import _load_video

    video_file = tmp_path / "two_frames.mp4"
    _write_two_frame_mp4(video_file)

    frames, metadata = _load_video(video_file, max_frames=unified_processor.video_processor.num_frames)
    inputs = unified_processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "video", "video": frames, "video_metadata": metadata},
            {"type": "text", "text": "describe the video"},
        ]}]
    )

    assert metadata.video_backend == "torchcodec"
    assert metadata.frames_indices.tolist() == [0, 1]
    assert tuple(inputs["pixel_values_videos"].shape) == (1, 2, 70, 6912)


@pytest.mark.parametrize("model_name", ["gemma-4-E2B-it", "gemma-4-E4B-it"])
def test_gemma4_series_short_mp4_keeps_predecoded_frames(model_name, tmp_path):
    from xhmodel_merak.model_integrations.gemma4.inference import _load_video
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_processor import XHGemma4Processor

    model_dir = Path("/data01/datasets") / model_name
    if not model_dir.is_dir():
        pytest.skip(f"Local Gemma4 processor assets unavailable: {model_dir}")
    processor = XHGemma4Processor.from_pretrained(str(model_dir), local_files_only=True)
    video_file = tmp_path / "two_frames.mp4"
    _write_two_frame_mp4(video_file)
    frames, metadata = _load_video(video_file, max_frames=processor.video_processor.num_frames)
    message = [{"role": "user", "content": [
        {"type": "video", "video": frames, "video_metadata": metadata},
        {"type": "text", "text": "describe the video"},
    ]}]
    inputs = processor.apply_chat_template(message)

    assert metadata.frames_indices.tolist() == [0, 1]
    assert inputs["pixel_values_videos"].shape[1] == 2
    assert "video_metadata" in message[0]["content"][0]


def _write_two_frame_mp4(video_file):
    with av.open(str(video_file), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(2):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), index * 100, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
