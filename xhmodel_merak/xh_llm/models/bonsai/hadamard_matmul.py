"""Hadamard online rotation as ordinary MatMul weights (Phase A).

Bonsai folds a normalized Sylvester/Walsh Hadamard into weights offline and
applies a matching activation transform online:

    y = quant_matmul(fwht(x * signs), W_folded)

Phase A represents the online transform as an explicit MatMul / Linear weight
instead of a special FWHT op:

    H_block = normalized Hadamard(block_size)
    W_rot[i, j] = signs[j] * H_tiled[i, j]
    x_rot = x @ W_rot.T   # equivalent to blockwise fwht(x * signs)

For the embedding inverse path:

    out = (dequant(W) @ ...) then inverse fwht with signs after transform.
    Inverse of orthonormal Hadamard is itself; signs are applied after.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

import torch
from torch import Tensor


def build_block_hadamard_matrix(
    block_size: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    normalized: bool = True,
) -> Tensor:
    """Build a Sylvester/Walsh Hadamard matrix of size block_size x block_size."""
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError(f"Hadamard block_size must be a positive power of 2, got {block_size}")

    had = torch.ones((1, 1), dtype=torch.float32, device=device)
    while had.shape[0] < block_size:
        top = torch.cat([had, had], dim=1)
        bottom = torch.cat([had, -had], dim=1)
        had = torch.cat([top, bottom], dim=0)
    if normalized:
        had = had / math.sqrt(block_size)
    return had.to(dtype=dtype)


def build_hadamard_matmul_weight(
    width: int,
    signs: Tensor,
    *,
    block_size: int = 1024,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cpu",
    inverse: bool = False,
) -> Tensor:
    """Compose block-diagonal Hadamard (+ signs) into a dense MatMul weight [width, width].

    Forward (inverse=False):
        x_rot = x @ W.T  with W[i,j] = H_tiled[i,j] * signs[j]
        which equals blockwise FWHT(x * signs).

    Inverse (inverse=True):
        Since H is orthonormal, H^{-1}=H. Bonsai applies signs after the inverse
        transform: out = FWHT(x) * signs.
        Represented as W[i,j] = signs[i] * H_tiled[i,j].
    """
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError(f"Hadamard block_size must be a positive power of 2, got {block_size}")
    if signs.ndim != 1:
        raise ValueError(f"signs must be rank-1, got {tuple(signs.shape)}")
    if int(signs.numel()) != width:
        raise ValueError(f"signs length {signs.numel()} != width {width}")
    if width % block_size != 0:
        raise ValueError(f"width={width} must be divisible by block_size={block_size}")
    signs_f = signs.to(device=device, dtype=torch.float32)
    if not torch.isin(signs_f, torch.tensor([-1.0, 1.0], device=device)).all():
        bad = sorted({float(v) for v in signs_f.unique().tolist() if float(v) not in (-1.0, 1.0)})
        raise ValueError(f"Hadamard signs must be ±1, got extras={bad}")

    h = build_block_hadamard_matrix(block_size, dtype=torch.float32, device=device, normalized=True)
    # Tile block-diagonal Hadamard across width.
    num_blocks = width // block_size
    h_tiled = torch.zeros((width, width), dtype=torch.float32, device=device)
    for b in range(num_blocks):
        sl = slice(b * block_size, (b + 1) * block_size)
        h_tiled[sl, sl] = h

    if inverse:
        # out = FWHT(x) * signs  => W = diag(signs) @ H
        weight = signs_f.unsqueeze(1) * h_tiled
    else:
        # out = FWHT(x * signs) => W = H @ diag(signs)
        weight = h_tiled * signs_f.unsqueeze(0)
    return weight.to(dtype=dtype).contiguous()


def apply_block_hadamard_matmul(
    x: Tensor,
    signs: Tensor,
    *,
    block_size: int = 1024,
    inverse: bool = False,
) -> Tensor:
    """Apply the Phase A MatMul-equivalent Hadamard transform to activations."""
    width = x.shape[-1]
    weight = build_hadamard_matmul_weight(
        width,
        signs,
        block_size=block_size,
        dtype=torch.float32,
        device=x.device,
        inverse=inverse,
    )
    return (x.to(torch.float32) @ weight.T).to(x.dtype)


def fwht_reference(
    x: Tensor,
    signs: Tensor,
    *,
    block_size: int = 1024,
    inverse: bool = False,
) -> Tensor:
    """Reference blockwise FWHT matching Bonsai runtime semantics (float32)."""
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError(f"Hadamard block_size must be a positive power of 2, got {block_size}")
    if x.shape[-1] % block_size != 0:
        raise ValueError("Hadamard block does not divide activation width")
    orig_dtype = x.dtype
    x32 = x.to(torch.float32)
    signs32 = signs.to(device=x.device, dtype=torch.float32)
    if not inverse:
        x32 = x32 * signs32
    flat = x32.reshape(-1, block_size)
    # In-place style butterfly FWHT
    h = block_size
    while h > 1:
        half = h // 2
        flat = flat.view(-1, h)
        a = flat[:, :half]
        b = flat[:, half:]
        flat = torch.cat([a + b, a - b], dim=-1)
        h = half
    flat = flat.reshape(-1, block_size) / math.sqrt(block_size)
    out = flat.reshape_as(x32)
    if inverse:
        out = out * signs32
    return out.to(orig_dtype)


def build_hadamard_matmul_weight_bank(
    signs_by_width: Mapping[int, Tensor],
    *,
    block_size: int = 1024,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cpu",
    include_inverse: bool = True,
) -> Dict[str, Tensor]:
    """Build MatMul weights for each unique activation width."""
    bank: Dict[str, Tensor] = {}
    for width, signs in sorted(signs_by_width.items(), key=lambda kv: kv[0]):
        bank[f"hadamard_matmul.forward.w{width}"] = build_hadamard_matmul_weight(
            int(width),
            signs,
            block_size=block_size,
            dtype=dtype,
            device=device,
            inverse=False,
        )
        if include_inverse:
            bank[f"hadamard_matmul.inverse.w{width}"] = build_hadamard_matmul_weight(
                int(width),
                signs,
                block_size=block_size,
                dtype=dtype,
                device=device,
                inverse=True,
            )
    return bank


def collect_signs_by_width_from_state(
    state_dict: Mapping[str, Tensor],
) -> Dict[int, Tensor]:
    """Collect unique hadamard_signs vectors keyed by width from Phase A state."""
    by_width: Dict[int, Tensor] = {}
    for key, tensor in state_dict.items():
        if not key.endswith(".hadamard_signs"):
            continue
        if tensor.ndim != 1:
            raise ValueError(f"{key} signs must be rank-1, got {tuple(tensor.shape)}")
        width = int(tensor.numel())
        if width in by_width:
            if not torch.equal(by_width[width].cpu(), tensor.cpu()):
                raise ValueError(f"Conflicting Hadamard signs for width={width}")
        else:
            by_width[width] = tensor.detach().cpu().to(torch.float32).contiguous()
    return by_width
