"""Host image preprocessing, multimodal positions and embedding insertion."""

from __future__ import annotations

from types import SimpleNamespace

import torch


def multimodal_rope_positions(config, input_ids, *, image_grid_thw=None, video_grid_thw=None,
                              mm_token_type_ids=None, attention_mask=None):
    """Native Qwen positions, including timestamp-separated video frame groups."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel

    context = SimpleNamespace(config=config)
    context.get_vision_position_ids = Qwen4ExpModel.get_vision_position_ids.__get__(context)
    if mm_token_type_ids is None:
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == config.image_token_id] = 1
        mm_token_type_ids[input_ids == config.video_token_id] = 2
    if (mm_token_type_ids.shape != input_ids.shape
            or not bool(((mm_token_type_ids >= 0) & (mm_token_type_ids <= 2)).all())):
        raise ValueError("mm_token_type_ids must match input_ids and contain text/image/video types")
    merge = int(config.vision_config.spatial_merge_size)
    for kind, grid in ((1, image_grid_thw), (2, video_grid_thw)):
        count = int((mm_token_type_ids == kind).sum())
        if grid is None:
            if count:
                raise ValueError("multimodal placeholders require the corresponding grid_thw")
            continue
        if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] == 0 or grid.dtype not in (torch.int32, torch.int64):
            raise ValueError("vision grid_thw must be nonempty integer [items,3]")
        if bool((grid <= 0).any()) or bool((grid[:, 1:] % merge != 0).any()):
            raise ValueError("vision grid dimensions must be positive and spatially merge-divisible")
        if count != int((grid.prod(-1) // merge**2).sum()):
            raise ValueError("multimodal placeholder count does not match grid_thw")
    return Qwen4ExpModel.get_rope_index(
        context, input_ids, mm_token_type_ids, image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw, attention_mask=attention_mask,
    )


def image_rope_positions(config, input_ids, image_grid_thw, *, attention_mask=None):
    """Compatibility entry for image-only callers."""
    if torch.any(input_ids == config.video_token_id):
        raise ValueError("Qwen4 image preprocessing does not accept video tokens")
    return multimodal_rope_positions(config, input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)


def create_qwen4_exp_processor(model_dir, max_image_tokens):
    """Reuse native media processing with budgets bounded by exported capacity."""
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    capacity = int(max_image_tokens)
    if capacity <= 0:
        raise ValueError("max_image_tokens must be positive")
    pixels = capacity * int(config.vision_config.patch_size * config.vision_config.spatial_merge_size) ** 2
    for name in ("image_processor", "video_processor"):
        media = getattr(processor, name, None)
        if media is not None:
            size = dict(getattr(media, "size", {}))
            media.size = {
                **size, "shortest_edge": min(int(size.get("shortest_edge", pixels)), pixels), "longest_edge": pixels,
            }
    return processor


def prepare_media_prompt(processor, prompt, *, image=None, video=None):
    """Native processor owns media loading, frame sampling and timestamp tokens."""
    images = list(image) if isinstance(image, (list, tuple)) else ([] if image is None else [image])
    videos = list(video) if isinstance(video, (list, tuple)) else ([] if video is None else [video])
    content = [{"type": "image", "image": value} for value in images]
    content += [{"type": "video", "video": value} for value in videos]
    content.append({"type": "text", "text": prompt})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, return_tensors="pt",
        return_dict=True, add_generation_prompt=True,
    )


def fuse_multimodal_embeddings(input_ids, embedding, encoder, config, *, pixel_values=None,
                               image_grid_thw=None, pixel_values_videos=None, video_grid_thw=None):
    """Fuse image/video features before text chunking while retaining PLE IDs."""
    if encoder is None:
        raise ValueError("this artifact has no visual encoder")
    embeds = embedding(input_ids.to(embedding.weight.device))
    for pixels, grid, token, encode in (
        (pixel_values, image_grid_thw, config.image_token_id, encoder.encode_many),
        (pixel_values_videos, video_grid_thw, config.video_token_id, encoder.encode_videos),
    ):
        mask = input_ids == int(token)
        if pixels is None and grid is None and not bool(mask.any()):
            continue
        if pixels is None or grid is None:
            raise ValueError("pixel values and grid_thw must be supplied together")
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 2:
            raise ValueError("visual pixels must be flattened patches [N, D]")
        if pixels.shape[0] != int(grid.prod(-1).sum()):
            raise ValueError("visual patch rows do not match grid_thw")
        features = encode(pixels, grid)
        if isinstance(features, (list, tuple)):
            features = torch.cat([x.reshape(-1, x.shape[-1]) for x in features], dim=0)
        else:
            features = features.reshape(-1, features.shape[-1])
        if int(mask.sum()) != features.shape[0] or features.shape[-1] != embeds.shape[-1]:
            raise ValueError("visual features and placeholder embeddings do not match")
        embeds = embeds.masked_scatter(mask.to(embeds.device).unsqueeze(-1).expand_as(embeds),
                                      features.to(device=embeds.device, dtype=embeds.dtype))
    return embeds


def prepare_image_prompt(model_dir, prompt, image, *, max_image_tokens):
    """Compatibility helper using the same native processor as model.generate."""
    from transformers import AutoConfig

    processor = create_qwen4_exp_processor(model_dir, max_image_tokens)
    inputs = prepare_media_prompt(processor, prompt, image=image)
    inputs["position_ids"], inputs["rope_deltas"] = multimodal_rope_positions(
        AutoConfig.from_pretrained(model_dir), inputs["input_ids"],
        image_grid_thw=inputs["image_grid_thw"],
        mm_token_type_ids=inputs.get("mm_token_type_ids"), attention_mask=inputs.get("attention_mask"),
    )
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


__all__ = ["image_rope_positions", "multimodal_rope_positions", "prepare_image_prompt",
           "merge_image_embeddings", "create_qwen4_exp_processor", "prepare_media_prompt", "fuse_multimodal_embeddings"]
