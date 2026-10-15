# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Offline schedule-conditioned modulation; runtime receives indices only."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from itertools import chain

import torch
from torch import nn

VIDEO_SIGMAS = (1.0, 0.972972989, 0.923076928, 0.8, 0.0)
AUDIO_SIGMAS = (1.0, 0.9, 0.75, 0.5, 0.0)
REFINER_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)


@dataclass(frozen=True)
class H3Schedule:
    video_sigmas: tuple[float, ...] = VIDEO_SIGMAS
    audio_sigmas: tuple[float, ...] = AUDIO_SIGMAS

    def __post_init__(self):
        if len(self.video_sigmas) < 2 or len(self.video_sigmas) != len(self.audio_sigmas):
            raise ValueError("H3 requires matching video/audio schedules with at least two points")
        for name, values in (("video", self.video_sigmas), ("audio", self.audio_sigmas)):
            if (
                values[0] != 1.0
                or values[-1] != 0.0
                or any(not a > b for a, b in zip(values[:-1], values[1:], strict=True))
            ):
                raise ValueError(f"{name} sigmas must decrease strictly from one to zero")

    @property
    def steps(self):
        return len(self.video_sigmas) - 1

    def row_times(self, *, device="cpu") -> tuple[torch.Tensor, ...]:
        return tuple(
            (1 - torch.tensor([v, a], dtype=torch.float32, device=device)).unique(sorted=True)
            for v, a in zip(self.video_sigmas[:-1], self.audio_sigmas[:-1], strict=True)
        )

    def indices(self, step: int, token_tags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= step < self.steps:
            raise ValueError("step must index an H3 schedule update")
        if token_tags.ndim != 1 or bool(((token_tags < 0) | (token_tags > 2)).any()):
            raise ValueError("token_tags must be a vector of video=0, text=1, audio=2")
        rows = self.row_times(device=token_tags.device)[step]
        video_t, audio_t = 1 - torch.tensor(
            [self.video_sigmas[step], self.audio_sigmas[step]], dtype=torch.float32, device=token_tags.device
        )
        times = torch.where(token_tags == 2, audio_t, video_t)
        timestep_indices = torch.searchsorted(rows, times).to(torch.int32)
        return timestep_indices, timestep_indices * 3 + token_tags.to(torch.int32)


@dataclass(frozen=True)
class ConditionedH3Schedule(H3Schedule):
    """Include the pinned visual and clean audio reference timesteps."""

    condition_video_time: float = 0.999

    def row_times(self, *, device="cpu"):
        return tuple(
            torch.cat((times, times.new_tensor([self.condition_video_time, 1.0]))).unique(sorted=True)
            for times in super().row_times(device=device)
        )

    def role_indices(self, step, *, device="cpu"):
        # Generated video, prompt text, generated audio, visual ref, audio ref.
        # Match row_times and the native scheduler's FP32 subtraction. Computing
        # 1-sigma in Python can round above the stored row and select its successor.
        generated = 1 - torch.tensor(
            [self.video_sigmas[step], self.video_sigmas[step], self.audio_sigmas[step]],
            dtype=torch.float32,
            device=device,
        )
        times = torch.cat((generated, generated.new_tensor([self.condition_video_time, 1.0])))
        indices = torch.searchsorted(self.row_times(device=device)[step], times).long()
        return indices, indices * 3 + torch.tensor([0, 1, 2, 0, 2], device=device)


def conditioned_tables(folded, schedule=None):
    """Five small role tables, selected by runtime row roles inside the graph."""
    schedule = schedule or ConditionedH3Schedule()
    indices = [schedule.role_indices(step, device=folded.output.device) for step in range(schedule.steps)]
    blocks = tuple(
        torch.stack([table[step].index_select(1, indices[step][1]) for step in range(schedule.steps)])
        for table in folded.blocks
    )
    output = torch.stack([folded.output[step].index_select(1, indices[step][0]) for step in range(schedule.steps)])
    return blocks, output


DEFAULT_H3_SCHEDULE = H3Schedule()


@contextmanager
def staged(module, device):
    first = next(chain(module.parameters(), module.buffers()), None)
    original = first.device if first is not None else None
    try:
        module.to(device)
        yield module
    finally:
        if original is not None:
            module.to(original)


class IndexedModulation(nn.Module):
    """A finite table replaces every per-layer time projection in the graph."""

    def __init__(self, table: torch.Tensor, *, sdk_ops: bool = False):
        super().__init__()
        if table.ndim != 4 or table.shape[0] < 1 or table.shape[1] != 6:
            raise ValueError("H3 modulation table must have shape [steps,6,time*modality,hidden]")
        if not bool(torch.isfinite(table).all()):
            raise ValueError("modulation constants must be finite")
        self.register_buffer("table", table.contiguous())
        self.sdk_ops = sdk_ops
        if sdk_ops:
            from xhquant.nn import Gather

            self.gather = Gather(axis=0)

    def forward(self, step_index: torch.Tensor) -> tuple[torch.Tensor, ...]:
        rows = self.gather(self.table, step_index) if self.sdk_ops else self.table.index_select(0, step_index.long())
        rows = rows.squeeze(0)
        # Keep component selection as constant slices. A scalar tensor index
        # otherwise exports another Gather for each of the six small vectors.
        return tuple(rows[i : i + 1].squeeze(0) for i in range(6))


@dataclass(frozen=True)
class FoldedH3Conditioning:
    blocks: tuple[torch.Tensor, ...]
    output: torch.Tensor
    row_times: tuple[torch.Tensor, ...]
    original_projection_parameters: int


@torch.no_grad()
def fold_h3_conditioning(
    source, schedule: H3Schedule = DEFAULT_H3_SCHEDULE, *, compute_device=None
) -> FoldedH3Conditioning:
    """Evaluate merged native weights at their original mixed precision.

    SiLU runs before the cast into each projection's precision. This order is
    part of H3 semantics; casting the shared time feature first biases all layers.
    The caller merges the complete hybrid LoRA before entering this function.
    """
    original_device = source.time_embedder.linear_1.weight.device
    device = torch.device(compute_device) if compute_device is not None else original_device
    times = schedule.row_times(device=device)
    with ExitStack() as stack:
        stack.enter_context(staged(source.time_embedder, device))
        stack.enter_context(staged(source.time_proj, device))
        embeddings = tuple(
            source.time_embedder(source.time_proj(t).to(source.time_embedder.linear_1.weight.dtype)) for t in times
        )
    max_times = max(t.numel() for t in times)
    tables = []
    count = 0
    for block in source.transformer_blocks:
        entries = []
        with staged(block.adaln_proj, device):
            for embedding in embeddings:
                pieces = block.adaln_proj(embedding)
                padded = tuple(torch.cat((x, x.new_zeros(max_times * 3 - x.shape[0], x.shape[-1]))) for x in pieces)
                entries.append(torch.stack(padded))
        tables.append(torch.stack(entries).detach().to(original_device))
        count += sum(p.numel() for p in block.adaln_proj.parameters())
    output = []
    with staged(source.norm_out.linear, device):
        for embedding in embeddings:
            dtype = source.norm_out.linear.weight.dtype
            modulation = source.norm_out.linear(torch.nn.functional.silu(embedding).to(dtype))
            shift, scale = modulation.chunk(2, -1)
            rows = torch.stack((shift, scale))
            if rows.shape[1] < max_times:
                rows = torch.cat((rows, rows.new_zeros(2, max_times - rows.shape[1], rows.shape[-1])), dim=1)
            output.append(rows)
    count += sum(p.numel() for p in source.norm_out.linear.parameters())
    return FoldedH3Conditioning(
        tuple(tables), torch.stack(output).to(original_device), tuple(t.to(original_device) for t in times), count
    )


def modality_tables(folded: FoldedH3Conditioning, schedule: H3Schedule = DEFAULT_H3_SCHEDULE):
    """Select only the three live T2VA modality rows before graph construction.

    The graph broadcasts one vector over each contiguous modality segment. It
    never materializes a separate modulation constant for every video token.
    """
    tags = torch.tensor([0, 1, 2], dtype=torch.int32, device=folded.output.device)
    indices = [schedule.indices(step, tags) for step in range(schedule.steps)]
    blocks = tuple(
        torch.stack([table[step].index_select(1, indices[step][1].long()) for step in range(schedule.steps)])
        for table in folded.blocks
    )
    output = torch.stack(
        [folded.output[step].index_select(1, indices[step][0].long()) for step in range(schedule.steps)]
    )
    return blocks, output


def validate_fp16_constants(values: torch.Tensor, *, name: str) -> None:
    """Reject overflow before narrowing constants, never clip unknown outliers."""
    if not bool(torch.isfinite(values).all()) or bool((values.abs() > torch.finfo(torch.float16).max).any()):
        raise FloatingPointError(f"{name} exceeds finite FP16 range; build a compensated scale before export")
