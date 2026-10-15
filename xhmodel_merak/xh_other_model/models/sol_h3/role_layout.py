# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Fixed AdaLN role slots with a broadcastable, noncontiguous key mask.

Roles describe modulation, not sparse video eligibility: role zero includes
both target video rows and Qwen vision-context rows. Original coordinates are
carried through the permutation rather than regenerated from the slot index.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm

import torch

ROLE_NAMES = ("video_time_video", "video_time_text", "audio_time_audio", "reference_video", "reference_audio")
ROLE_LAYOUT_FORMAT = "fixed_adaln_role_slots_v1"


def role_spans(capacities):
    """Return nonempty ``(start, stop, role)`` spans in numeric role order."""
    if len(capacities) != 5 or any(type(n) is not int or n < 0 for n in capacities) or not sum(capacities):
        raise ValueError("H3 fixed-role capacities must contain five nonnegative integers with a positive total")
    offset, spans = 0, []
    for role, count in enumerate(capacities):
        if count:
            spans.append((offset, offset + count, role))
        offset += count
    return tuple(spans)


def role_capacities_from_plan(plan, *, block_size=256):
    """Conservatively cover a native capacity plan without calibration counts.

    The presentation budget includes visual rows and their delimiters. Charge
    that whole non-prompt budget to role zero and charge the presentation
    overhead to role one as well: its text/vision split can vary with media.
    This small overestimate avoids treating vision delimiters as plain text.
    Shared branches take componentwise maxima across their per-mode results.
    """
    if block_size not in (64, 128, 256, 512):
        raise ValueError("H3 role attention block_size must be 64, 128, 256 or 512")
    fields = (
        "target_video_tokens",
        "target_audio_tokens",
        "reference_video_tokens",
        "reference_audio_tokens",
        "text_tokens",
        "prompt_max_tokens",
        "presentation_overhead",
    )
    if any(type(plan.get(key)) is not int or plan[key] < 0 for key in fields):
        raise ValueError("H3 role planning requires nonnegative native token counts")
    if plan["text_tokens"] < plan["prompt_max_tokens"]:
        raise ValueError("H3 presentation token budget cannot be below the prompt budget")
    alignment = plan.get("token_alignment", 256)
    if type(alignment) is not int or alignment <= 0:
        raise ValueError("H3 role token_alignment must be a positive integer")
    alignment = lcm(alignment, block_size)
    raw = (
        plan["target_video_tokens"] + plan["text_tokens"] - plan["prompt_max_tokens"],
        plan["prompt_max_tokens"] + plan["presentation_overhead"],
        plan["target_audio_tokens"],
        plan["reference_video_tokens"],
        plan["reference_audio_tokens"],
    )
    return tuple((value + alignment - 1) // alignment * alignment for value in raw)


def role_affine(value, scale, shift, spans):
    """Broadcast one short vector per fixed span, without token-shaped tables."""
    pieces = []
    for start, stop, role in spans:
        piece = value[:, start:stop] * scale[role : role + 1]
        if shift is not None:
            piece = piece + shift[role : role + 1]
        pieces.append(piece)
    return torch.cat(pieces, dim=1)


def valid_tokens_from_blocks(block_sizes, block_size):
    """O(N) validity used by the eager dense oracle, never an N-by-N mask."""
    offsets = torch.arange(block_size, device=block_sizes.device)
    return (offsets[None, None] < block_sizes[..., None]).flatten(1)


@dataclass(frozen=True)
class RoleLayout:
    capacities: tuple[int, ...]
    counts: tuple[int, ...]
    block_size: int
    positions: torch.Tensor
    roles: torch.Tensor
    pack: torch.Tensor
    head_unpack: torch.Tensor
    block_sizes: torch.Tensor
    attention_heads: int

    @property
    def valid_mask(self):
        return valid_tokens_from_blocks(self.block_sizes, self.block_size)

    @property
    def key_mask(self):
        """FP16 additive mask, broadcast over heads and queries: O(N), not O(N²)."""
        return torch.where(self.valid_mask[:, None, None, :], 0.0, -65504.0).half()

    def metadata(self):
        return dict(
            format=ROLE_LAYOUT_FORMAT,
            role_names=list(ROLE_NAMES),
            role_capacities=list(self.capacities),
            role_spans=[list(span) for span in role_spans(self.capacities)],
            attention_block_size=self.block_size,
            attention_heads=self.attention_heads,
            attention_mode="broadcast_key_mask",
            sequence_capacity=sum(self.capacities),
        )


def make_role_layout(lengths, capacities, roles, positions, role_capacities, *, attention_heads, block_size=256):
    """Map native valid rows into fixed role slots and prepare an exact KV mask.

    Native rows and coordinates arrive in compact video/audio/text order. Each
    stream's input still has its original static capacity. ``pack`` indexes
    projected input rows plus one explicit zero row; padding maps to that zero
    row. ``head_unpack`` restores only video/audio output slots, also using an
    explicit zero row for their padding. No layer needs a hidden-state repack.

    Every physical block contains a valid prefix followed by padding. Block
    sizes retain that geometry for future routing; the exported dense graph
    consumes their O(N) broadcast key mask. Query padding is never selected by
    the output map and cannot become a key for any live row in a later layer.
    """
    if (
        len(lengths) != 3
        or len(capacities) != 3
        or any(
            type(n) is not int or type(c) is not int or not 0 <= n <= c or c <= 0
            for n, c in zip(lengths, capacities, strict=True)
        )
    ):
        raise ValueError("H3 native row lengths must fit three positive source capacities")
    spans = role_spans(role_capacities)
    if block_size not in (64, 128, 256, 512) or any(capacity % block_size for capacity in role_capacities):
        raise ValueError("each H3 role capacity must align to the attention block size")
    if type(attention_heads) is not int or attention_heads <= 0:
        raise ValueError("H3 attention_heads must be a positive integer")
    valid = sum(lengths)
    if roles.shape != (valid,) or roles.dtype not in (torch.int32, torch.int64) or positions.shape != (valid, 3):
        raise ValueError("native H3 roles/positions must describe exactly the valid rows")
    roles, positions = roles.detach().cpu().long(), positions.detach().cpu().float()
    if not bool(torch.isfinite(positions).all()) or not bool(((roles >= 0) & (roles < 5)).all()):
        raise ValueError("native H3 coordinates must be finite and roles must lie in [0,4]")
    offset = 0
    for count, allowed in zip(lengths, ((0, 3), (2, 4), (0, 1)), strict=True):
        stream = roles[offset : offset + count]
        if not bool(((stream == allowed[0]) | (stream == allowed[1])).all()):
            raise ValueError("native H3 roles disagree with video/audio/text origins")
        offset += count

    source_rows, source_offset = [], 0
    for length, capacity in zip(lengths, capacities, strict=True):
        source_rows.append(torch.arange(source_offset, source_offset + length))
        source_offset += capacity
    source_rows = torch.cat(source_rows)
    total = sum(role_capacities)
    pack = torch.full((total,), source_offset, dtype=torch.int32)
    head_unpack = torch.full((capacities[0] + capacities[1],), total, dtype=torch.int32)
    destination_positions = torch.zeros((total, 3), dtype=torch.float32)
    fixed_roles = torch.empty(total, dtype=torch.int32)
    block_sizes = torch.zeros((1, total // block_size), dtype=torch.int32)
    counts = tuple(int((roles == role).sum()) for role in range(5))
    if any(count > capacity for count, capacity in zip(counts, role_capacities, strict=True)):
        raise ValueError(f"native H3 role counts {counts} exceed fixed capacities {tuple(role_capacities)}")
    for start, stop, role in spans:
        selected = torch.nonzero(roles == role, as_tuple=False).flatten()
        count = selected.numel()
        destination = torch.arange(start, start + count)
        original = source_rows[selected]
        pack[destination] = original.to(torch.int32)
        destination_positions[destination] = positions[selected]
        fixed_roles[start:stop] = role
        head = original < head_unpack.numel()
        head_unpack[original[head]] = destination[head].to(torch.int32)
        blocks = torch.arange((stop - start) // block_size)
        block_sizes[0, start // block_size : stop // block_size] = (count - blocks * block_size).clamp(0, block_size)

    return RoleLayout(
        tuple(role_capacities),
        counts,
        block_size,
        destination_positions,
        fixed_roles,
        pack,
        head_unpack,
        block_sizes,
        attention_heads,
    )
