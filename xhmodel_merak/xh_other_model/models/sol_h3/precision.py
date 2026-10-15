# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Residual representation and explicit RMS precision policy, applied before PTQ."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

ATTENTION_VALUE_DIVISOR = 512.0
SCALED_RMS_POLICY = "max_rebased_augmented"
PLAIN_RMS_POLICY = "plain_wide_reduction"
FP16_MIN_POSITIVE = 2.0**-24


@dataclass(frozen=True)
class H3PrecisionPolicy:
    residual_scale: float = 1024.0
    rms_norm_mode: str = "plain"
    rms_norm_eps: float | None = None
    text_residual_scale: float = 4.0
    attention_output_scale: float = 16.0
    ffn_product_scale: float = 512.0
    ffn_output_scale: float = 512.0

    def __post_init__(self):
        for name in (
            "residual_scale",
            "text_residual_scale",
            "attention_output_scale",
            "ffn_product_scale",
            "ffn_output_scale",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 1
                or math.log2(value) % 1
            ):
                raise ValueError(f"{name} must be a finite power of two >= 1")

        if self.rms_norm_mode not in ("plain", "stable"):
            raise ValueError("rms_norm_mode must be plain or stable")
        if self.rms_norm_mode == "stable":
            if self.rms_norm_eps is not None:
                raise ValueError("stable RMSNorm uses source epsilon; set rms_norm_eps to null")
        else:
            epsilon = FP16_MIN_POSITIVE if self.rms_norm_eps is None else self.rms_norm_eps
            if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or not math.isfinite(epsilon):
                raise ValueError("rms_norm_eps must be finite")
            epsilon = float(torch.tensor(epsilon, dtype=torch.float32))
            if not math.isfinite(epsilon) or epsilon < 65504.0**-2:
                raise ValueError("rms_norm_eps must keep the FP16 reciprocal <= 65504 after FP32 rounding")
            object.__setattr__(self, "rms_norm_eps", epsilon)

    @property
    def scaled_rms_policy(self):
        return PLAIN_RMS_POLICY if self.rms_norm_mode == "plain" else SCALED_RMS_POLICY

    def residual_norm(self, source, *, sdk_ops=False):
        return StableRMSNorm(
            source,
            input_divisor=self.residual_scale,
            sdk_ops=sdk_ops,
            bounded_reciprocal=self.rms_norm_mode == "stable",
            epsilon=self.rms_norm_eps,
        )

    def to_dict(self):
        return asdict(self)


def scaled_linear(source: nn.Linear, *, output_divisor=1.0, input_multiplier=1.0) -> nn.Linear:
    """Fold a linear change of units in FP32, retaining the source's rounded weights."""
    if getattr(source, "packed_weight", None) is not None:
        from xhquant.nn import GPTQPackedLinear, GPTQPackedWeight

        from .conditioning import validate_fp16_constants

        if not isinstance(source, GPTQPackedLinear):
            raise TypeError("unsupported packed projection type")
        packed = source.packed_weight
        scales = packed.scales.detach().float() * (input_multiplier / output_divisor)
        validate_fp16_constants(scales, name="compensated GPTQ scales")
        weight = GPTQPackedWeight(
            packed.qweight,
            packed.qzeros,
            scales,
            packed.g_idx,
            **{
                key: getattr(packed, key)
                for key in (
                    "bits",
                    "group_size",
                    "in_features",
                    "out_features",
                    "pack_dtype_bits",
                    "qzero_format",
                    "sym",
                    "desc_act",
                    "logical_bits",
                )
            },
        )
        bias = None if source.bias is None else source.bias.detach().float() / output_divisor
        if bias is not None:
            validate_fp16_constants(bias, name="compensated GPTQ bias")
        return GPTQPackedLinear(weight, bias).requires_grad_(False)
    module = nn.Linear(
        source.in_features,
        source.out_features,
        source.bias is not None,
        device="meta",
        dtype=torch.float32,
    )
    with torch.no_grad():
        module.weight = nn.Parameter(
            source.weight.detach().float() * (input_multiplier / output_divisor), requires_grad=False
        )
        if source.bias is not None:
            module.bias = nn.Parameter(source.bias.detach().float() / output_divisor, requires_grad=False)
        from .conditioning import validate_fp16_constants

        validate_fp16_constants(module.weight, name="compensated Linear weight")
        if module.bias is not None:
            validate_fp16_constants(module.bias, name="compensated Linear bias")
    return module.requires_grad_(False)


def split_swiglu_projection(source: nn.Linear, product_divisor: float) -> nn.Linear:
    """Scale the linear half before the product; leave the SiLU argument intact."""
    if source.out_features % 2:
        raise ValueError("SwiGLU projection must have two equally sized halves")
    module = scaled_linear(source)
    half = source.out_features // 2
    with torch.no_grad():
        module.weight[:half].div_(product_divisor)
        if module.bias is not None:
            module.bias[:half].div_(product_divisor)
    return module


class StableRMSNorm(nn.Module):
    """Wide RMS with bounded reciprocal, or an explicitly selected lossy epsilon.

    Without an epsilon override, RMSNorm(x/s, epsilon/s²) preserves source units.
    """

    def __init__(self, source, *, input_divisor=1.0, sdk_ops=False, bounded_reciprocal=True, epsilon=None):
        super().__init__()
        self.eps = float(source.eps if hasattr(source, "eps") else source.variance_epsilon) / (
            input_divisor * input_divisor
        )
        if epsilon is not None:
            self.eps = float(epsilon)
        self.weight = nn.Parameter(source.weight.detach().float().clone(), requires_grad=False)
        self.sdk_ops = sdk_ops
        self.rebase = sdk_ops and input_divisor > 1 and bounded_reciprocal
        if sdk_ops and not bounded_reciprocal and self.eps < 65504.0**-2:
            raise ValueError("plain RMSNorm reciprocal can overflow FP16; use bounded_reciprocal")
        if sdk_ops:
            from xhquant.nn import RMSNorm

            if self.rebase:
                # XH2a narrows the reciprocal RMS to FP16 even with wide sums.
                # Add sqrt(N*eps) as a coordinate after dividing by row max:
                # the sum includes epsilon and the reciprocal stays <= sqrt(P).
                # Pad to 64 so the primitive retains an aligned hidden width.
                self.width = self.weight.numel()
                padded = ((self.width + 64) // 64) * 64
                self.padding = padded - self.width
                self.floor = float(torch.tensor(math.sqrt(self.width * self.eps), dtype=torch.float16))
                if not math.isfinite(self.floor) or self.floor <= 0:
                    raise ValueError("scaled RMS epsilon must have a finite nonzero FP16 augmented coordinate")
                self.norm = RMSNorm(padded, 0.0).requires_grad_(False)
                from xhquant.nn import Max, ReduceMax

                self.maximum = ReduceMax(dim=-1, keepdim=True)
                self.minimum_scale = Max()
                self.register_buffer("floor_value", torch.tensor(self.floor))
                with torch.no_grad():
                    self.weight.mul_(math.sqrt(self.width / padded))
            else:
                self.norm = RMSNorm(self.weight.numel(), self.eps)
                self.norm.weight = self.weight

    def forward(self, x):
        if self.sdk_ops:
            if self.rebase:
                maximum = self.minimum_scale(self.maximum(x.abs()), self.floor_value)
                unit = x / maximum
                augmented = torch.cat((unit, F.pad(self.floor_value / maximum, (0, self.padding - 1))), -1)
                return self.norm(augmented)[..., : self.width] * self.weight
            return self.norm(x)
        wide = x.float()
        y = wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).type_as(x)


def require_finite(values, *, name):
    """Host/runtime guard; an overflowing graph is rejected, never silently clipped."""
    for index, value in enumerate(values if isinstance(values, (list, tuple)) else [values]):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name}[{index}] produced non-finite values")
