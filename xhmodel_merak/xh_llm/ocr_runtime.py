# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Chunked OCR prefill with positions computed for the complete prompt."""

import torch


def last_nonpad_index(input_ids, pad_token_id):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("OCR inference requires one token sequence")
    valid = torch.nonzero(input_ids[0] != pad_token_id).flatten()
    if not valid.numel():
        raise ValueError("OCR prompt has no non-padding tokens")
    return int(valid[-1])


def match_position_ranks(args, session):
    if hasattr(session, "initialize") and not getattr(session, "_initialized", True):
        session.initialize()
    args = list(args)
    for index, name in ((1, "time_position_ids"), (2, "hight_position_ids"), (3, "width_position_ids")):
        meta = session.get_input(name)
        while args[index].ndim < len(meta.shape):
            args[index] = args[index].unsqueeze(0)
    return tuple(args)


def prefill_logits(model, input_ids, *, image_embeds=None, position_ids=None):
    """Caller owns the inference/cache context; this function retains state."""
    length = int(input_ids.shape[1])
    context = int(model.meta_info.model_config.context_max_length)
    if not 0 < length <= context:
        raise ValueError(f"Prompt length {length} is outside context capacity {context}")
    model.set_prefill()
    processor = model.get_data_preprocessor()
    if position_ids is None:
        position_ids = torch.arange(length, device=model.device).view(1, 1, -1).expand(3, 1, -1)
    if tuple(position_ids.shape) != (3, 1, length):
        raise ValueError("position_ids must have shape [3, 1, prompt_length]")
    # Scatter all image features before slicing. Images may span any number of
    # chunks; their position IDs must never be recomputed on a partial image.
    embeds = model.embed_tokens(input_ids.to(model.device))
    image_mask = input_ids.to(model.device) == model.meta_info.model_config.image_token_id
    count = int(image_mask.sum())
    if count:
        if image_embeds is None or image_embeds.shape[0] != count:
            raise ValueError("Image placeholder count does not match image features")
        embeds[image_mask] = image_embeds.to(embeds)
    elif image_embeds is not None and image_embeds.numel():
        raise ValueError("Image features have no corresponding placeholders")
    chunk_length = int(model.meta_info.model_config.prefill_chunk_length)
    logits = None
    for offset in range(0, length, chunk_length):
        chunk = embeds[:, offset : offset + chunk_length]
        args = list(processor({"inputs_embeds": chunk, "past_seq_length": offset}))
        for axis in range(3):
            positions = position_ids[axis, 0, offset : offset + chunk.shape[1]]
            args[axis + 1] = torch.nn.functional.pad(positions, (0, chunk_length - positions.numel())).to(model.device)
        args = match_position_ranks(args, model.prefill_model.hmonnx_session)
        logits = model.forward(*args)[0]
    return logits
