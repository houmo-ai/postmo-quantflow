# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Expert modulation follows weight geometry, including unequal condition width."""

import copy

import pytest
import torch


@pytest.mark.parametrize("width,condition_width", [(64, 64), (64, 32), (1024, 1024)])
def test_adaptive_norm_matches_native(width, condition_width):
    from lerobot.policies.pi_gemma import PiGemmaRMSNorm
    from xhmodel_merak.xh_other_model.models.pi05._llm_model_impl_cond import _GemmaRMSNorm

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4891)
        native = PiGemmaRMSNorm(width, cond_dim=condition_width)
        with torch.no_grad():
            native.dense.weight.normal_(std=0.02)
        hidden = torch.randn(1, 3, width)
        condition = torch.randn(1, condition_width)
    wrapper_type = type("WrappedPiNorm", (_GemmaRMSNorm, PiGemmaRMSNorm), {})
    wrapped = wrapper_type.convert(copy.deepcopy(native))
    with torch.no_grad():
        expected, expected_gate = native(hidden, condition)
        actual, actual_gate = wrapped(hidden, condition)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gate, expected_gate)
