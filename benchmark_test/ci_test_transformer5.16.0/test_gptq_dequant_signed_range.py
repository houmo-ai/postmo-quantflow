"""Reject incompatible GPTQ codes rather than hide a corrupt checkpoint."""

import pytest
import torch
from torch import nn

from xhmodel_merak.xh_llm._dequant_converter import gptqmodel_torch_qlinear_converter


class _PackedLinear(nn.Module):
    def __init__(self, zero: int) -> None:
        super().__init__()
        codes = (0, 1, 254, 255)
        def packed_word(values):
            word = sum(value << (8 * index) for index, value in enumerate(values))
            return word - (1 << 32) if word >= 1 << 31 else word

        self.register_buffer("qweight", torch.tensor([[packed_word(codes)] * 4], dtype=torch.int32))
        self.register_buffer("qzeros", torch.tensor([[packed_word((zero,) * 4)]], dtype=torch.int32))
        self.register_buffer("scales", torch.full((1, 4), 0.25))
        self.register_buffer("g_idx", torch.zeros(4, dtype=torch.int32))
        self.register_parameter("bias", None)
        self.register_buffer("wf_unsqueeze_zero", torch.arange(0, 32, 8).reshape(1, 1, 4))
        self.register_buffer("wf_unsqueeze_neg_one", torch.arange(0, 32, 8).reshape(1, 4, 1))
        self.bits = 8
        self.pack_factor = 4
        self.maxq = 255
        self.dequant_dtype = torch.int16


def test_gptq_signed_boundary_preserves_integer_codes():
    linear = _PackedLinear(128)
    gptqmodel_torch_qlinear_converter(linear)
    expected = torch.tensor([[-128, -127, 126, 127]] * 4, dtype=torch.int16)
    torch.testing.assert_close(linear.quant_weight, expected)
    torch.testing.assert_close(linear.weight, expected.float() * 0.25, rtol=0, atol=0)


@pytest.mark.parametrize("zero", [112, 127, 129])
def test_gptq_out_of_signed_range_is_rejected_without_requantization(zero):
    linear = _PackedLinear(zero)
    with pytest.raises(AssertionError, match="not in"):
        gptqmodel_torch_qlinear_converter(linear)
    assert hasattr(linear, "qweight")
    assert not hasattr(linear, "weight")
