"""Native WeMM-Embedding helpers used by Flickr30K and parity eval."""

from __future__ import annotations

from typing import Any

import torch


def load_native_embedder(
    model_dir: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=dtype).to(device).eval()
    return model, processor


def embed_native_items(
    model: Any,
    processor: Any,
    items: list[dict[str, Any]],
    device: str,
    batch_size: int = 4,
    image_size: int | None = None,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
) -> torch.Tensor:
    from qwen_vl_utils import process_vision_info

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    patch_size = int(getattr(getattr(model.config, "vision_config", None), "patch_size", 16))
    embeddings = []
    with torch.inference_mode():
        for offset in range(0, len(items), batch_size):
            batch = items[offset : offset + batch_size]
            conversations = [
                [
                    {
                        "role": "user",
                        "content": _content_for_item(
                            item,
                            image_size,
                            image_min_pixels=image_min_pixels,
                            image_max_pixels=image_max_pixels,
                        ),
                    }
                ]
                for item in batch
            ]
            texts = [
                processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                for conversation in conversations
            ]
            images, videos = process_vision_info(conversations, image_patch_size=patch_size)
            inputs = processor(
                text=texts,
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            ).to(device)
            embeddings.append(model.embedding(**inputs).float().cpu())
    return torch.cat(embeddings, dim=0)


def _content_for_item(
    item: dict[str, Any],
    image_size: int | None,
    *,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if item.get("image") is not None:
        image_item: dict[str, Any] = {"type": "image", "image": item["image"]}
        if image_size is not None:
            image_item["resized_height"] = image_size
            image_item["resized_width"] = image_size
        if image_min_pixels is not None:
            image_item["min_pixels"] = image_min_pixels
        if image_max_pixels is not None:
            image_item["max_pixels"] = image_max_pixels
        content.append(image_item)
    if item.get("text") is not None:
        content.append({"type": "text", "text": str(item["text"])})
    if not content:
        raise ValueError("WeMM embedding input must contain text or image")
    return content


__all__ = ["embed_native_items", "load_native_embedder"]
