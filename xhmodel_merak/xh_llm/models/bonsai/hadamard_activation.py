"""Online Hadamard activation as an ordinary MatMul (Phase A).

Bonsai folds a normalized Sylvester/Walsh Hadamard into Linear weights and
expects the matching activation transform at runtime:

    y = Linear(FWHT(x * signs), W_folded)

Phase A stores that transform as a dense weight

    W[i, j] = H_tiled[i, j] * signs[j]
    x_rot = x @ W.T

so the exported graph contains a normal MatMul / Linear, not an FWHT custom op.
Weights of the same input width share one matrix (signs are unique per width).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch
import torch.nn as nn
from torch import Tensor


class HadamardActivation(nn.Module):
    """``x_rot = x @ weight.T`` with a frozen ``[width, width]`` Hadamard weight.

    Keep the projection as a leaf ``nn.Linear`` for the XH2a quantizer.
    Hadamard entries are zero or +/- one shared amplitude, so sign(weight)
    is the exact SSFP mantissa and the exporter recovers the group scale.
    """

    def __init__(self, weight: Tensor):
        super().__init__()
        if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            raise ValueError(
                f"Hadamard MatMul weight must be square, got {tuple(weight.shape)}"
            )
        dense = weight.detach().contiguous()
        width = int(dense.shape[0])
        self.proj = nn.Linear(width, width, bias=False, device="meta", dtype=dense.dtype)
        self.proj.weight = nn.Parameter(dense, requires_grad=False)
        self.proj.register_buffer("quant_weight", dense.sign().to(torch.int8), persistent=False)
        # This Phase A projection uses an explicit SSFP4 payload format.
        self.proj.logical_bits = 4

    def forward(self, hidden_states: Tensor) -> Tensor:
        # The containing model casts this module to its execution dtype.
        # Branching on input.dtype breaks symbolic FX tracing; bypassing proj
        # with F.linear also loses the leaf module's SSFP quantization payload.
        return self.proj(hidden_states.to(self.proj.weight.dtype))


def _iter_linears(root: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def attach_hadamard_activations(
    root: nn.Module,
    signs_by_module: Mapping[str, Tensor],
    forward_weights: Mapping[int, Tensor],
    *,
    block_size: int = 1024,
) -> Dict[str, int]:
    """Register one shared ``HadamardActivation`` per input width and point signed Linears at it.

    ``signs_by_module`` keys are HF module names (``model.language_model....`` /
    ``lm_head``) matching ``named_modules()`` under ``root``. A Linear is wrapped
    only when its name is in that map, so unsigned projections (``in_proj_a/b``)
    stay untouched.

    Returns counts: attached, missing_weight, widths.
    """
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block_size must be a positive power of two")
    signs_by_width: Dict[int, Tensor] = {}
    for name, signs in signs_by_module.items():
        width = int(signs.numel())
        if signs.ndim != 1 or not ((signs == 1) | (signs == -1)).all():
            raise ValueError(f"{name} hadamard_signs must be a rank-1 +/-1 vector")
        if width in signs_by_width and not torch.equal(signs_by_width[width], signs):
            raise ValueError(f"Conflicting Hadamard signs for width={width}")
        signs_by_width[width] = signs

    banks: Dict[int, HadamardActivation] = {}
    for width, weight in forward_weights.items():
        if tuple(weight.shape) != (int(width), int(width)):
            raise ValueError(f"Hadamard weight for width={width} has shape {tuple(weight.shape)}")
        banks[int(width)] = HadamardActivation(weight)

    # One registered module per width, parented on the root so state_dict /
    # .to() see them even if several Linears alias the same instance.
    for width, bank in banks.items():
        root.add_module(f"_hadamard_act_w{width}", bank)

    attached = 0
    missing_weight = 0
    signed = set(signs_by_module)
    seen_signed = set()
    for name, linear in _iter_linears(root):
        if name not in signed:
            continue
        seen_signed.add(name)
        width = int(linear.in_features)
        bank = banks.get(width)
        if bank is None:
            missing_weight += 1
            continue
        signs = signs_by_module[name]
        if int(signs.numel()) != width:
            raise ValueError(
                f"{name} hadamard_signs length {int(signs.numel())} != in_features {width}"
            )
        if width % int(block_size) != 0:
            raise ValueError(
                f"{name} in_features {width} is not divisible by Hadamard block {block_size}"
            )
        linear.hadamard_activation = bank  # type: ignore[attr-defined]
        attached += 1

    missing_modules = sorted(signed - seen_signed)
    if missing_modules:
        # Partial checkpoints (layer0 smoke) only sign the modules they replace.
        # Callers that pass a full-model sign map against a truncated wrap should
        # filter first; a non-empty remainder here is a real name mismatch.
        preview = ", ".join(missing_modules[:8])
        raise KeyError(
            f"{len(missing_modules)} hadamard_signs modules were not found under "
            f"{type(root).__name__}: {preview}"
        )
    return {
        "attached": attached,
        "missing_weight": missing_weight,
        "widths": len(banks),
    }


def load_phase_a_hadamard_aux(aux_path: str) -> tuple[Dict[str, Tensor], Dict[int, Tensor], int]:
    """Read ``*.hadamard_signs`` and ``hadamard_matmul.forward.w{width}`` from aux."""
    aux = torch.load(aux_path, map_location="cpu", weights_only=True)
    metadata_path = Path(aux_path).with_name("phase_a_metadata.json")
    block_size = 1024
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        block_size = int(metadata["hadamard"]["block_size"])
    signs: Dict[str, Tensor] = {}
    forwards: Dict[int, Tensor] = {}
    for key, tensor in aux.items():
        if key.endswith(".hadamard_signs"):
            module_name = key[: -len(".hadamard_signs")]
            signs[module_name] = tensor.detach().to(torch.float32).contiguous()
        elif key.startswith("hadamard_matmul.forward.w"):
            width = int(key.rsplit("w", 1)[-1])
            forwards[width] = tensor.detach().contiguous()
    if not signs:
        raise KeyError(f"No *.hadamard_signs in {aux_path}")
    if not forwards:
        raise KeyError(f"No hadamard_matmul.forward.w* in {aux_path}")
    return signs, forwards, block_size
