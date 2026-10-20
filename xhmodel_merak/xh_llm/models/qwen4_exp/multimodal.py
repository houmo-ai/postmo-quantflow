"""Host image preprocessing, multimodal positions and embedding insertion."""

from __future__ import annotations

from types import SimpleNamespace

import torch


def image_rope_positions(config, input_ids, image_grid_thw, *, attention_mask=None):
    """Use native Qwen4 multimodal RoPE without allocating a language model."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel

    context = SimpleNamespace(config=config)
    context.get_vision_position_ids = Qwen4ExpModel.get_vision_position_ids.__get__(context)
    modality_ids = torch.zeros_like(input_ids)
    modality_ids[input_ids == config.image_token_id] = 1
    if torch.any(input_ids == config.video_token_id):
        raise ValueError("Qwen4 image preprocessing does not accept video tokens")
    merge = int(config.vision_config.spatial_merge_size)
    expected = int((image_grid_thw.prod(-1) // merge**2).sum())
    if int((modality_ids == 1).sum()) != expected:
        raise ValueError("image placeholder count does not match image_grid_thw")
    return Qwen4ExpModel.get_rope_index(
        context,
        input_ids,
        modality_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )


def prepare_image_prompt(model_dir, prompt, image, *, max_image_tokens):
    """Prepare one image/chat request with the checkpoint's native processor.

    Resize limits derive from the exported post-merge token capacity. The
    smallest-fit runtime rejects overflow rather than truncating image patches.
    """
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    if not isinstance(image, Image.Image):
        with Image.open(image) as loaded:
            image = loaded.convert("RGB")
    else:
        image = image.convert("RGB")
    max_image_tokens = int(max_image_tokens)
    if max_image_tokens <= 0:
        raise ValueError("max_image_tokens must be positive")
    pixel_unit = int(config.vision_config.patch_size * config.vision_config.spatial_merge_size) ** 2
    # A gear may be smaller than the checkpoint processor's usual min_pixels.
    # Clamp the minimum as well so its resize cannot exceed the selected gear.
    image_processor = processor.image_processor
    max_pixels = max_image_tokens * pixel_unit
    size = getattr(image_processor, "size", {})
    min_pixels = min(int(size.get("shortest_edge", max_pixels)), max_pixels)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        images_kwargs={"size": {"shortest_edge": min_pixels, "longest_edge": max_pixels}},
    )
    positions, deltas = image_rope_positions(
        config,
        inputs["input_ids"],
        inputs["image_grid_thw"],
        attention_mask=inputs.get("attention_mask"),
    )
    inputs["position_ids"] = positions
    inputs["rope_deltas"] = deltas
    return inputs


def merge_image_embeddings(input_ids, token_embedding, image_features, image_token_id):
    """Insert visual features exactly at image placeholders, retaining PLE IDs."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Qwen4 multimodal serving requires batch-one input_ids")
    embeddings = token_embedding(input_ids)
    features = torch.cat([value.reshape(-1, value.shape[-1]) for value in image_features], dim=0)
    mask = input_ids == int(image_token_id)
    if int(mask.sum()) != features.shape[0] or features.shape[-1] != embeddings.shape[-1]:
        raise ValueError("image features and image placeholder embeddings do not match")
    return embeddings.masked_scatter(
        mask.unsqueeze(-1).expand_as(embeddings), features.to(device=embeddings.device, dtype=embeddings.dtype)
    )


__all__ = ["image_rope_positions", "prepare_image_prompt", "merge_image_embeddings"]
