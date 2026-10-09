# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Host token placement and fixed-prefix gear preparation for Qwen-Image-2.1."""

from dataclasses import dataclass
from math import prod

import torch


@dataclass(frozen=True)
class TokenLayout:
    encoder_tokens: int
    prefix_tokens: int
    prefix_capacity: int
    target_tokens: int
    target_capacity: int
    text_sources: tuple[int, ...]
    text_destinations: tuple[int, ...]
    image_destinations: tuple[int, ...]
    compact_image_mask: torch.Tensor
    image_ids: torch.Tensor
    key_valid: torch.Tensor

    @property
    def sequence_length(self):
        return self.prefix_capacity + self.target_capacity

    def assemble(self, text_embeddings, image_embeddings):
        if text_embeddings.ndim != 3 or tuple(text_embeddings.shape[:2]) != (1, self.encoder_tokens):
            raise ValueError("text embeddings must have shape [1, encoder_tokens, hidden]")
        expected = (1, len(self.image_destinations), text_embeddings.shape[-1])
        if tuple(image_embeddings.shape) != expected:
            raise ValueError(f"image embeddings must have shape {expected}")
        if text_embeddings.dtype != image_embeddings.dtype or text_embeddings.device != image_embeddings.device:
            raise ValueError("text and image embeddings must have the same dtype and device")
        hidden = text_embeddings.new_zeros(1, self.sequence_length, text_embeddings.shape[-1])
        hidden[:, self.text_destinations] = text_embeddings[:, self.text_sources]
        hidden[:, self.image_destinations] = image_embeddings
        return hidden

    def attention_bias(self, *, device, dtype):
        ids = self.image_ids.to(device)
        position = torch.arange(self.sequence_length, device=device)
        allowed = (position[:, None] >= position[None, :]) | ((ids[:, None] == ids[None, :]) & (ids[:, None] >= 0))
        allowed &= self.key_valid.to(device)[None, :]
        # Finite mask is representable by FP16 and the deployment input ABI.
        return torch.where(allowed, 0.0, -65504.0).to(dtype)[None, None]

    def pad_rope(self, compact_rope):
        if compact_rope.ndim != 2 or compact_rope.shape[0] != self.prefix_tokens + self.target_tokens:
            raise ValueError("RoPE must be built on the original unpadded joint sequence")
        gap = self.prefix_capacity - self.prefix_tokens
        padding = compact_rope.new_ones(gap, compact_rope.shape[1])
        tail = compact_rope.new_ones(self.target_capacity - self.target_tokens, compact_rope.shape[1])
        result = torch.cat((padding, compact_rope[: self.prefix_tokens], compact_rope[self.prefix_tokens :], tail))
        return result.real[None, None], result.imag[None, None]

    def attention_ranges(self, *, device):
        """Physical [start,end) ranges after right-aligning the valid prefix.

        Reference images attend bidirectionally within their own image block;
        target queries see the complete valid joint sequence. Padding queries
        have empty ranges. No dense attention mask is passed to deployment.
        """
        start = self.prefix_capacity - self.prefix_tokens
        result = torch.zeros(1, self.sequence_length, 2, dtype=torch.int32, device=device)
        ids = self.image_ids.tolist()
        ends = {image_id: index + 1 for index, image_id in enumerate(ids) if image_id >= 0}
        for index in range(start, self.prefix_capacity):
            result[0, index, 0] = start
            result[0, index, 1] = max(index + 1, ends.get(ids[index], 0))
        result[0, self.prefix_capacity : self.prefix_capacity + self.target_tokens, 0] = start
        result[0, self.prefix_capacity : self.prefix_capacity + self.target_tokens, 1] = (
            self.prefix_capacity + self.target_tokens
        )
        return result


def prepare_layout(image_slots, image_shapes, *, prefix_capacity=None, target_capacity=None, encoder_valid=None):
    """Build on Host; shapes list contains reference images followed by the target."""
    if image_slots.dtype != torch.bool or image_slots.ndim != 2 or image_slots.shape[0] != 1:
        raise ValueError("image_slots must be a batch-1 boolean matrix")
    if not image_shapes:
        raise ValueError("image_shapes must include a target image")
    for shape in image_shapes:
        if (
            len(shape) != 3
            or shape[0] != 1
            or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in shape)
        ):
            raise ValueError("image_shapes require (1, positive height, positive width)")
        if shape[1] % 2 or shape[2] % 2:
            raise ValueError("each image grid must have even height and width for the 2x2 VLM merge")
    slots = image_slots[0].cpu().tolist()
    sizes = [prod(shape) for shape in image_shapes]
    if sum(slots) * 4 != sum(sizes[:-1]):
        raise ValueError("reference image slots and latent grid sizes disagree")
    if encoder_valid is None:
        valid = [True] * len(slots)
    else:
        if encoder_valid.shape != image_slots.shape:
            raise ValueError("encoder_valid and image_slots must have the same shape")
        valid = encoder_valid[0].bool().cpu().tolist()
        if any(slot and not keep for slot, keep in zip(slots, valid, strict=False)):
            raise ValueError("image slots cannot be masked out")
    text_sources, text_destinations, image_destinations, compact_mask, key_valid = [], [], [], [], []
    cursor = 0
    for index, slot in enumerate(slots):
        if not valid[index]:
            continue
        count = 4 if slot else 1
        if slot:
            image_destinations.extend(range(cursor, cursor + count))
        else:
            text_sources.append(index)
            text_destinations.append(cursor)
        compact_mask.extend([slot] * count)
        key_valid.extend([True if slot else valid[index]] * count)
        cursor += count
    prefix = cursor
    capacity = prefix if prefix_capacity is None else prefix_capacity
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < prefix:
        raise ValueError(f"prefix_capacity must be an integer >= actual prefix length {prefix}")
    target_capacity = sizes[-1] if target_capacity is None else target_capacity
    if type(target_capacity) is not int or target_capacity < sizes[-1]:
        raise ValueError("target_capacity must cover the actual target length")
    gap = capacity - prefix
    text_destinations = [index + gap for index in text_destinations]
    image_destinations = [index + gap for index in image_destinations]
    # Build reference IDs from individual image shapes, not contiguous True runs.
    ids = [-1] * capacity + [len(sizes) - 1] * sizes[-1] + [-1] * (target_capacity - sizes[-1])
    offset = 0
    for image_id, length in enumerate(sizes[:-1]):
        destinations = image_destinations[offset : offset + length]
        if destinations != list(range(destinations[0], destinations[0] + length)):
            raise ValueError("slots belonging to one reference image must be contiguous")
        for position in destinations:
            ids[position] = image_id
        offset += length
    image_destinations.extend(range(capacity, capacity + sizes[-1]))
    compact_mask.extend([True] * sizes[-1])
    key_valid = [False] * gap + key_valid + [True] * sizes[-1] + [False] * (target_capacity - sizes[-1])
    return TokenLayout(
        len(slots),
        prefix,
        capacity,
        sizes[-1],
        target_capacity,
        tuple(text_sources),
        tuple(text_destinations),
        tuple(image_destinations),
        torch.tensor(compact_mask, dtype=torch.bool),
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(key_valid, dtype=torch.bool),
    )
