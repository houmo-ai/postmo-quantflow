"""vLLM model adapters for the official WeMM MMEB-v3 evaluator."""

from __future__ import annotations

import atexit
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch


IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
LONG_TEXT_SAFETY_MARGIN_TOKENS = 1
DEFAULT_VIDEO_TOTAL_PIXELS = 8192 * 32 * 32


def _flatten_video_frames(value: object) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [value]
    frames: list[Any] = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, list):
            frames.extend(frame for frame in item if frame is not None)
        else:
            frames.append(item)
    return frames


def _strip_media_tokens(text: str) -> str:
    return text.replace(IMAGE_TOKEN, "").replace(VIDEO_TOKEN, "").strip()


def build_vllm_request(
    tokenizer: Any,
    raw_text: str,
    images: object,
    videos: object,
    *,
    image_min_pixels: int,
    image_max_pixels: int,
    video_min_pixels: int,
    video_max_pixels: int,
    video_total_pixels: int = DEFAULT_VIDEO_TOTAL_PIXELS,
    max_model_len: int,
) -> object:
    """Build one vLLM request while preserving the official WeMM prompt order."""

    image_items = []
    if images is not None:
        image_items = images if isinstance(images, list) else [images]
        image_items = [image for image in image_items if image is not None]
    video_frames = _flatten_video_frames(videos)
    wants_video = VIDEO_TOKEN in raw_text

    content: list[dict[str, str]] = []
    multimodal_data: dict[str, object] = {}
    processor_kwargs: dict[str, object] = {}
    interleaved = False

    if video_frames or (image_items and wants_video):
        # Official MMEB video datasets deliver their sampled frame bundle via
        # the evaluator's legacy ``images`` field. Do not apply the 16-image
        # interleaving limit to that bundle: the official protocol keeps up to
        # 64 frames and lets qwen-vl-utils enforce ``max_frames``.
        frames = video_frames or image_items
        from qwen_vl_utils import process_vision_info

        video_content = {
            "type": "video",
            "video": frames,
            "min_pixels": int(video_min_pixels),
            "max_pixels": int(video_max_pixels),
            "total_pixels": int(video_total_pixels),
            "sample_fps": 1,
            "fps": 1,
            "max_frames": 64,
        }
        _, processed_videos, video_kwargs = process_vision_info(
            [{"role": "user", "content": [video_content]}],
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if not processed_videos or len(processed_videos) != 1:
            raise ValueError("WeMM expected exactly one processed video")
        processed_video = processed_videos[0]
        if not isinstance(processed_video, tuple) or len(processed_video) != 2:
            raise TypeError("WeMM video preprocessing must return video metadata")
        video, metadata = processed_video
        content.append({"type": "video"})
        multimodal_data["video"] = (video, metadata)
        processor_kwargs.update(video_kwargs)
    elif image_items and IMAGE_TOKEN in raw_text:
        image_items = image_items[:16]
        interleaved = True
        image_iter = iter(image_items)
        selected_images = []
        parts = raw_text.split(IMAGE_TOKEN)
        for index, part in enumerate(parts):
            part = part.strip()
            if part:
                content.append({"type": "text", "text": part})
            if index < len(parts) - 1:
                image = next(image_iter, None)
                if image is not None:
                    content.append({"type": "image"})
                    selected_images.append(image)
        if selected_images:
            multimodal_data["image"] = selected_images[0] if len(selected_images) == 1 else selected_images
            processor_kwargs.update(
                min_pixels=int(image_min_pixels),
                max_pixels=int(image_max_pixels),
            )
    elif image_items:
        image_items = image_items[:16]
        content.extend({"type": "image"} for _ in image_items)
        multimodal_data["image"] = image_items[0] if len(image_items) == 1 else image_items
        processor_kwargs.update(
            min_pixels=int(image_min_pixels),
            max_pixels=int(image_max_pixels),
        )

    if not interleaved:
        clean_text = _strip_media_tokens(raw_text)
        if clean_text or not content:
            content.append({"type": "text", "text": clean_text})

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    # Keep WeMM's model-specific token protocol in its vLLM model module.  The
    # evaluator only chooses whether text truncation is allowed.
    from vllm_merak.model_executor.model_loader.wemm_embedding_native import (
        prepare_wemm_prompt_token_ids,
    )

    max_input_tokens = None
    if not multimodal_data:
        # The current vLLM hybrid prefill path stalls when an input occupies
        # exactly max_model_len tokens. Reserve one scheduler token while the
        # WeMM helper preserves the final pooling anchor.
        max_input_tokens = int(max_model_len) - LONG_TEXT_SAFETY_MARGIN_TOKENS
    prompt_token_ids = prepare_wemm_prompt_token_ids(
        tokenizer,
        prompt,
        max_input_tokens=max_input_tokens,
    )
    tokenized_prompt: dict[str, object] = {"prompt_token_ids": prompt_token_ids}
    if not multimodal_data:
        return tokenized_prompt
    tokenized_prompt.update(
        {
            "multi_modal_data": multimodal_data,
            "mm_processor_kwargs": processor_kwargs,
        }
    )
    return tokenized_prompt


class VLLMWeMM:
    """MMEB model facade backed by one native vLLM pooling engine."""

    def __init__(
        self,
        model: str,
        max_model_len: int,
        image_min_pixels: int,
        image_max_pixels: int,
        video_min_pixels: int,
        video_max_pixels: int,
        video_total_pixels: int,
        max_images: int,
        gpu_memory_utilization: float,
        max_num_seqs: int = 1,
        *,
        merak: bool = False,
    ) -> None:
        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if merak and max_num_seqs != 1:
            raise ValueError("vLLM Merak hybrid linear state requires max_num_seqs=1")
        if merak:
            os.environ.setdefault("ENABLE_HMINFERENCE_V2", "1")
        os.environ["VLLM_PLUGINS"] = "merak" if merak else "wemm"
        from vllm import LLM
        from vllm.config import CompilationConfig, CUDAGraphMode

        model_path = Path(model).expanduser().resolve()
        compilation_config = CompilationConfig()
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        kwargs: dict[str, object] = {
            "model": str(model_path),
            "runner": "auto" if merak else "pooling",
            "max_num_seqs": int(max_num_seqs),
            "max_model_len": int(max_model_len),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "enforce_eager": True,
            "compilation_config": compilation_config,
            "trust_remote_code": True,
            "enable_prefix_caching": False,
            # Profiling the maximum 64-frame video at every group startup is
            # expensive and does not affect embeddings.
            "skip_mm_profiling": True,
            "limit_mm_per_prompt": {"image": int(max_images), "video": 1},
        }
        if merak:
            kwargs.update(config_format="merak_llm", load_format="merak_llm")
        else:
            chat_template = model_path / "embedding_chat_template.jinja"
            if chat_template.is_file():
                kwargs["chat_template"] = chat_template
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        config_name = "golden_meta_info.json" if merak else "config.json"
        config = json.loads((model_path / config_name).read_text())
        if merak:
            self.rep_dim = int(config["embedding_dimension"])
        else:
            self.rep_dim = int(config["text_config"]["hidden_size"])
        self.image_min_pixels = int(image_min_pixels)
        self.image_max_pixels = int(image_max_pixels)
        self.video_min_pixels = int(video_min_pixels)
        self.video_max_pixels = int(video_max_pixels)
        self.video_total_pixels = int(video_total_pixels)
        self.max_model_len = int(max_model_len)
        self._closed = False
        atexit.register(self.close)

    def to(self, *_args: object, **_kwargs: object) -> "VLLMWeMM":
        return self

    def eval(self) -> "VLLMWeMM":
        return self

    def close(self) -> None:
        if self._closed:
            return
        engine_core = getattr(getattr(self.llm, "llm_engine", None), "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
        self._closed = True

    def _requests(self, inputs: dict[str, object]) -> list[object]:
        texts = cast(Sequence[str], inputs.get("text") or [])
        images = cast(Sequence[object], inputs.get("images") or [None] * len(texts))
        videos = cast(Sequence[object], inputs.get("videos") or [None] * len(texts))
        requests = []
        for index, raw_text in enumerate(texts):
            requests.append(
                build_vllm_request(
                    self.tokenizer,
                    str(raw_text),
                    images[index] if index < len(images) else None,
                    videos[index] if index < len(videos) else None,
                    image_min_pixels=self.image_min_pixels,
                    image_max_pixels=self.image_max_pixels,
                    video_min_pixels=self.video_min_pixels,
                    video_max_pixels=self.video_max_pixels,
                    video_total_pixels=self.video_total_pixels,
                    max_model_len=self.max_model_len,
                )
            )
        return requests

    def __call__(self, qry=None, tgt=None):
        inputs = qry if qry is not None else tgt
        if inputs is None:
            raise ValueError("Expected query or target inputs")
        if inputs.get("audios") and any(audio is not None for audio in inputs["audios"]):
            raise ValueError("WeMM-Embedding does not support audio inputs")
        outputs = self.llm.embed(cast(Any, self._requests(inputs)), use_tqdm=False)
        rows = [torch.tensor(output.outputs.embedding, dtype=torch.float32) for output in outputs]
        if not rows:
            reps = torch.empty((0, self.rep_dim), dtype=torch.float32)
        else:
            reps = torch.stack(rows)
        return {"qry_reps" if qry is not None else "tgt_reps": reps}


class VLLMMerakWeMM(VLLMWeMM):
    """MMEB model facade backed by one prefill-only vLLM Merak engine."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs, merak=True)


__all__ = [
    "LONG_TEXT_SAFETY_MARGIN_TOKENS",
    "DEFAULT_VIDEO_TOTAL_PIXELS",
    "VLLMWeMM",
    "VLLMMerakWeMM",
    "build_vllm_request",
]
