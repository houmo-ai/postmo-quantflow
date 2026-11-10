# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Image/video inference after relocation of Qwen3.6 dense/MoE variants."""
from .transformer import prepare as prepare_transformer


def prepare(case, root, monkeypatch):
    kwargs, inputs = prepare_transformer(case, root, monkeypatch)
    if case.variant in {"full", "lora", "mtp", "lora_mtp", "dflash"}:
        import av
        import numpy as np
        from transformers import AutoImageProcessor, PreTrainedTokenizerFast
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

        source = root / "source"
        tokenizer = PreTrainedTokenizerFast.from_pretrained(source)
        tokenizer.chat_template = (
            "{% for m in messages %}{% if m['content'] is string %}{{ m['content'] }}"
            "{% else %}{% for c in m['content'] %}"
            "{% if c['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>"
            "{% elif c['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
            "{% else %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}{% endfor %}"
        )
        tokenizer.save_pretrained(source)
        image_processor = AutoImageProcessor.from_pretrained(source)
        image_processor.do_resize = False
        image_processor.save_pretrained(source)
        Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2,
                             do_resize=False, min_frames=2, max_frames=4, fps=2).save_pretrained(source)
        path = root / "video.mp4"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=2)
            stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
            for color in (30, 90, 150, 210):
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), color, np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    return kwargs, inputs


def validate_output(case, output, result, moved, inputs, root, record):
    if case.variant not in {"full", "lora", "mtp", "lora_mtp", "dflash"}:
        return
    from xhmodel_merak.model_inference import inference_model
    image = inference_model(artifact_dir=str(moved), device="cuda:0", inputs={
        "prompt": "hello world", "image": str(root / "image.png"), "max_new_tokens": 5,
    })
    assert image.model_id == case.model_id and image.output_tokens > 0
    record["image_inference"] = image.to_dict()
    video = inference_model(artifact_dir=str(moved), device="cuda:0", inputs={
        "prompt": "hello world", "video": str(root / "video.mp4"), "max_new_tokens": 5,
    })
    assert video.model_id == case.model_id and video.output_tokens > 0
    record["video_inference"] = video.to_dict()
