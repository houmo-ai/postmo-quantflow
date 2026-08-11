"""Aligned XH2a quantization used by both Macaron LoRA paths.

This adapter deliberately stays in modelzoo: xhquanttool currently exposes
aligned weight quantization through QLinear/QMoeBlock rather than a public
standalone tensor API.  Keeping the one internal call here prevents the two
Macaron graph transforms from drifting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from xhquant.quantization.xh2a.quantizer.hmfp_quantizer import HMFPQuantizer


@dataclass(frozen=True)
class AlignedLoRAQuantSpec:
    mode: str
    nshare: int
    man_bit: int
    hidden_bit: bool
    rounding: str
    exp_bit: int = 16

    def validate(self) -> None:
        if self.mode not in {"sefp", "ssfp"}:
            raise ValueError(f"Unsupported LoRA HMFP mode: {self.mode}")
        if self.nshare <= 0 or self.man_bit <= 0:
            raise ValueError("LoRA HMFP nshare and man_bit must be positive")
        if self.exp_bit <= 0:
            raise ValueError("LoRA HMFP exp_bit must be positive")
        if self.mode == "ssfp" and self.hidden_bit:
            raise ValueError("ssfp LoRA weights require hidden_bit=false")


def quantize_aligned_lora_weight(
    weight: np.ndarray,
    *,
    dim: int,
    spec: AlignedLoRAQuantSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned qweight and scale/exponent for an already laid-out weight."""

    spec.validate()
    quantizer = HMFPQuantizer(
        {
            "mode": spec.mode,
            "man_bit": spec.man_bit,
            "hidden_bit": spec.hidden_bit,
            "nshare": spec.nshare,
            "rounding": spec.rounding,
            "calib_metric": "minmax",
        },
        is_weight_quant=True,
    )
    qweight, scale_or_exp = quantizer(
        torch.from_numpy(np.ascontiguousarray(weight)),
        dim=dim,
        return_padded=True,
        is_fake_quant=False,
    )
    return qweight.cpu().numpy(), scale_or_exp.cpu().numpy()


__all__ = ["AlignedLoRAQuantSpec", "quantize_aligned_lora_weight"]
