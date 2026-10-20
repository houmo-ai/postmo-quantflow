"""Packed SEFP4 PLE storage with lookup-local SDK dequantization."""

from __future__ import annotations

import torch

from xhquant.backend.xh2a.functions.te.xh2a_matmul_triton import pack_signed_w4, unpack_signed_w4
from xhquant.quantization.xh2a.quantizer._sefp_utils import sefp
from xhquant.xhonnxruntime.base_module import BaseModule


PLE_SEFP4_OP = "Qwen4PLESEFP4Gather"


def quantize_ple_sefp4_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a bounded row block; the caller streams production-size tables."""
    if weight.ndim != 2 or weight.shape[1] < 1:
        raise ValueError("PLE weight must be a matrix with positive embedding width")
    if weight.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError("PLE SEFP4 encoding requires floating-point weights")
    if not torch.isfinite(weight).all().item():
        raise ValueError("PLE weight contains NaN or Inf")
    groups = (weight.shape[1] + 63) // 64
    if weight.shape[0] == 0:
        return (
            torch.empty((0, groups, 32), dtype=torch.int8, device=weight.device),
            torch.empty((0, groups, 1), dtype=torch.int16, device=weight.device),
        )
    # SDK sanitizes its input in place. The checkpoint mapping must stay intact.
    mantissa, exponent = sefp(
        weight.clone(), dim=1, manbit=3, nshare=64,
        keep_bit_for_exp=True, return_padded=True, assemble=False,
    )
    packed = pack_signed_w4(mantissa.unsqueeze(-1)).squeeze(-1)
    return packed.contiguous(), exponent.to(torch.int16).contiguous()


def lookup_ple_sefp4(packed, exponent, indices, *, embedding_dim: int, output_dtype=torch.float16):
    """Gather compressed rows before unpacking; negative ids follow ONNX Gather."""
    if embedding_dim < 1:
        raise ValueError("PLE embedding_dim must be positive")
    if output_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("PLE output_dtype must be float16 or bfloat16")
    groups = (embedding_dim + 63) // 64
    if packed.dtype != torch.int8 or packed.ndim != 3 or tuple(packed.shape[1:]) != (groups, 32):
        raise ValueError("PLE packed storage must be int8 [rows, groups, 32]")
    if exponent.dtype != torch.int16 or tuple(exponent.shape) != (packed.shape[0], groups, 1):
        raise ValueError("PLE exponent storage must be int16 [rows, groups, 1]")
    if packed.device != exponent.device or packed.device != indices.device:
        raise ValueError("PLE packed weights, exponent and ids must share a device")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("PLE indices must be int32 or int64")
    if indices.numel():
        minimum, maximum = torch.aminmax(indices)
        if minimum.item() < -packed.shape[0] or maximum.item() >= packed.shape[0]:
            raise IndexError("PLE indices out of bounds")
    # Advanced indexing preserves repetitions, arbitrary id rank and empty ids.
    selected = packed[indices.long()]
    selected_exp = exponent[indices.long()]
    mantissa = unpack_signed_w4(selected.unsqueeze(-1)).squeeze(-1)
    # Match SDK assemble: BF16 is assembled in FP32, retaining its wider
    # exponent range; FP16 preserves its original subnormal rounding.
    compute_dtype = torch.float32 if output_dtype == torch.bfloat16 else torch.float16
    values = mantissa * (2 ** selected_exp.to(compute_dtype))
    return values.reshape(*indices.shape, groups * 64)[..., :embedding_dim].to(output_dtype)


class PLESEFP4Gather(BaseModule):
    """Qwen4 deployment ABI for packed PLE weights; no whole-table expansion."""

    def __init__(self, embedding_dim: int, output_dtype=torch.float16):
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("PLE embedding_dim must be positive")
        if output_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("PLE output_dtype must be float16 or bfloat16")
        self.embedding_dim = embedding_dim
        self.output_dtype = output_dtype

    def _forward_aligned(self, packed, exponent, indices):
        return {self.output_names[0]: lookup_ple_sefp4(
            packed, exponent, indices, embedding_dim=self.embedding_dim, output_dtype=self.output_dtype,
        )}

    _forward_fast = _forward_aligned

    @classmethod
    def from_onnx_node(cls, node, context):
        del context
        if len(node.inputs) != 3 or len(node.outputs) != 1:
            raise ValueError("PLE SEFP4 Gather requires three inputs and one output")
        for name, expected in (("nshare", 64), ("manbit", 3), ("keep_bit_for_exp", 1)):
            if int(node.attrs.get(name, expected)) != expected:
                raise ValueError(f"PLE SEFP4 requires {name}={expected}")
        output_dtype = int(node.attrs.get("output_dtype", 10))
        if output_dtype not in (10, 16):
            raise ValueError("PLE output_dtype requires FLOAT16=10 or BFLOAT16=16")
        module = cls(
            int(node.attrs["embedding_dim"]), torch.bfloat16 if output_dtype == 16 else torch.float16,
        )
        module.input_names = [item.name for item in node.inputs]
        module.output_names = [node.outputs[0].name]
        return module
