"""Validate the frame-directory adapter against the real Unified processor."""
import os
from pathlib import Path

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
    frames, metadata = _load_video(frames_dir)

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
