"""Bounded cache write-set snapshots for Qwen4's single-request runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

import torch


@dataclass(frozen=True)
class _TensorPiece:
    original: torch.Tensor
    value: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    slices: tuple[slice, ...]

    @classmethod
    def capture(cls, tensor, slices):
        return cls(tensor, tensor[slices].detach().clone(), tuple(tensor.shape), tensor.dtype, tensor.device, slices)

    def validate(self, current):
        if (not torch.is_tensor(current) or tuple(current.shape) != self.shape
                or current.dtype != self.dtype or current.device != self.device):
            raise ValueError("MTP snapshot tensor layout differs from the live state")
        if (tuple(self.original.shape) != self.shape or self.original.dtype != self.dtype
                or self.original.device != self.device
                or tuple(self.original[self.slices].shape) != tuple(self.value.shape)):
            raise ValueError("MTP snapshot backing tensor layout changed")

    def restore(self):
        self.original[self.slices].copy_(self.value)


@dataclass(frozen=True)
class StateSnapshot:
    """Save full recurrent state and only the slots a fixed-width graph may write.

    The caller owns logical length, block IDs and slot mapping. Page storage here
    follows Qwen4 generation's preallocated identity page map: logical token p
    writes physical block p//64, offset p%64, with no allocation or page-table
    mutation. Other page tensor layouts retain the conservative full copy.
    """

    states: Mapping[str, _TensorPiece]
    pages: tuple[tuple[_TensorPiece, ...], ...]

    @property
    def nbytes(self):
        pieces = [*self.states.values(), *(piece for page in self.pages for piece in page)]
        return sum(piece.value.numel() * piece.value.element_size() for piece in pieces)

    def restore(self, states: MutableMapping[str, torch.Tensor], pages: Sequence[torch.Tensor]):
        # Validate the complete transaction before mutating any live object.
        if states.keys() != self.states.keys() or len(pages) != len(self.pages):
            raise ValueError("MTP snapshot layout differs from the live state")
        for name, piece in self.states.items():
            piece.validate(states[name])
        for current, pieces in zip(pages, self.pages, strict=True):
            for piece in pieces:
                if current is not piece.original:
                    raise ValueError("MTP snapshot PageKV buffer identity changed")
                piece.validate(current)
        for name, piece in self.states.items():
            piece.restore()
            states[name] = piece.original
        for current, pieces in zip(pages, self.pages, strict=True):
            for piece in pieces:
                # Page buffers are caller-owned and never replaced by sessions.
                current[piece.slices].copy_(piece.value)


def snapshot_state(states, pages, *, specs, past_length, write_length):
    """Snapshot a graph's static write width, including masked padding slots."""
    if isinstance(past_length, bool) or not isinstance(past_length, int) or past_length < 0:
        raise ValueError("MTP snapshot past_length must be a nonnegative integer")
    if isinstance(write_length, bool) or not isinstance(write_length, int) or write_length <= 0:
        raise ValueError("MTP snapshot write_length must be a positive integer")
    end = past_length + write_length
    saved = {}
    for name, tensor in states.items():
        spec = specs.get(name)
        slices = [slice(None)] * tensor.ndim
        if spec is not None and "cache_length" in spec.shape:
            axis = spec.shape.index("cache_length")
            if tensor.ndim != len(spec.shape) or end > tensor.shape[axis]:
                raise ValueError(f"MTP snapshot write range exceeds {name} capacity")
            slices[axis] = slice(past_length, end)
        elif spec is not None and "qsa_pool_length" in spec.shape:
            axis = spec.shape.index("qsa_pool_length")
            start_block, end_block = past_length // 4, (end + 3) // 4
            if tensor.ndim != len(spec.shape) or start_block > tensor.shape[axis]:
                raise ValueError(f"MTP snapshot write range exceeds {name} capacity")
            slices[axis] = slice(start_block, min(end_block, tensor.shape[axis]))
        saved[name] = _TensorPiece.capture(tensor, tuple(slices))
    saved_pages = []
    for page in pages:
        pieces = []
        if page.ndim == 5 and page.shape[0] == 2 and page.shape[3] == 64:
            if end > page.shape[1] * 64:
                raise ValueError("MTP snapshot write range exceeds PageKV capacity")
            position = past_length
            while position < end:
                block, offset = divmod(position, 64)
                stop = min(64, offset + end - position)
                slices = (slice(None), slice(block, block + 1), slice(None), slice(offset, stop), slice(None))
                pieces.append(_TensorPiece.capture(page, slices))
                position += stop - offset
        else:
            pieces.append(_TensorPiece.capture(page, (slice(None),) * page.ndim))
        saved_pages.append(tuple(pieces))
    return StateSnapshot(saved, tuple(saved_pages))
