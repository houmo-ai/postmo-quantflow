# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Local video loading for the Qwen processor and shared visual runtime."""
from pathlib import Path


def load_video(processor, path: Path):
    """Use native file sampling; frame directories are supplied in sorted order."""
    import numpy as np
    from PIL import Image
    from transformers.video_utils import VideoMetadata, load_video as decode_video

    if path.is_dir():
        paths = [p for p in sorted(path.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}]
        if not paths:
            raise ValueError(f"video frame directory contains no supported images: {path}")
        frames = []
        for frame_path in paths:
            with Image.open(frame_path) as image:
                frames.append(np.asarray(image.convert("RGB")))
        if any(frame.shape != frames[0].shape for frame in frames):
            raise ValueError("video directory frames must have identical dimensions")
        metadata = VideoMetadata(total_num_frames=len(frames), fps=2.0,
                                 frames_indices=list(range(len(frames))))
        return np.stack(frames), metadata
    if not path.is_file():
        raise FileNotFoundError(f"video does not exist: {path}")
    # Import/check the decoder only for file inputs. Text, images and frame
    # directories do not need TorchCodec or FFmpeg.
    from importlib.metadata import version
    from packaging.version import Version

    torch_version, codec_version = Version(version("torch")), Version(version("torchcodec"))
    if torch_version.release[:2] != (2, 8) or codec_version.release[:2] not in {(0, 6), (0, 7)}:
        raise ValueError("Qwen video files require torch==2.8 and torchcodec 0.6/0.7")
    return decode_video(str(path), backend="torchcodec", sample_indices_fn=processor.video_processor.sample_frames)


def build_video_inputs(processor, path: Path, prompt: str, think: bool):
    frames, metadata = load_video(processor, path)
    messages = [{"role": "user", "content": [
        {"type": "video", "video": str(path)}, {"type": "text", "text": prompt},
    ]}]
    # Render with the native template and call the existing processor directly:
    # the frames are already sampled and must not pass through a second loader.
    from transformers.processing_utils import ProcessorMixin
    text = ProcessorMixin.apply_chat_template(
        processor,
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=think,
    )
    return processor(text=[text], videos=[frames], video_metadata=[metadata],
                     do_sample_frames=False, padding=True, return_tensors="pt")
