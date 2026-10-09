# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic validity only; does not claim model export or image quality coverage."""
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[2] / 'examples_merak/aigc/qwen_image21/float_probe.py'
    spec = importlib.util.spec_from_file_location('qwen_image21_float_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def monitor(probe):
    return probe.NumericalMonitor({'modules': {}, 'steps': []})


@pytest.mark.parametrize('operation', ['clamp', 'clip'])
@pytest.mark.parametrize('inference', [False, True])
def test_detects_overflow_hidden_by_clamp(monitor, operation, inference):
    x = torch.tensor([40000.], dtype=torch.float16)
    assert torch.isfinite((x + x).clamp(-65504, 65504)).all()
    with pytest.raises(FloatingPointError, match='fp16_pre_clamp'), torch.inference_mode(inference), monitor:
        getattr(x + x, operation)(-65504, 65504)
    assert monitor.report['first_nonfinite']['shape'] == [1]


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_real_linear_overflow_and_finite_baseline(monitor, dtype):
    linear = torch.nn.Linear(2, 1, bias=False, dtype=dtype)
    with torch.no_grad():
        linear.weight.fill_(2.)
    handle = linear.register_forward_hook(monitor.hook('linear'), with_kwargs=True)
    try:
        value = torch.tensor([[20000., 20000.]], dtype=dtype)
        if dtype == torch.float16:
            with pytest.raises(FloatingPointError, match='linear.output'):
                linear(value)
            assert monitor.report['first_nonfinite']['name'] == 'linear.output'
        else:
            assert torch.isfinite(linear(value)).all()
            assert monitor.report['modules']['linear.output']['absmax'] > 65504
            assert 'first_nonfinite' not in monitor.report
    finally:
        handle.remove()


def test_monitor_preserves_finite_arithmetic(monitor):
    value = torch.tensor([-4., 0., 5.], dtype=torch.float16)
    expected = (value * 2).clamp(-8, 8)
    with monitor:
        actual = (value * 2).clamp(-8, 8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert monitor.report['modules']['fp16_pre_clamp']['absmax'] == 10.


def test_missing_block_clip_observations_cannot_pass(probe):
    report = {'modules': {'fp16_pre_clamp': {'calls': 2}}}
    with pytest.raises(RuntimeError, match='Incomplete FP16 clipping monitoring'):
        probe.validate_clipping_coverage(report, 'float16', layers=32, steps=20)
    report['modules']['fp16_pre_clamp']['calls'] = 642
    probe.validate_clipping_coverage(report, 'float16', layers=32, steps=20)
    probe.validate_clipping_coverage({'modules': {}}, 'bfloat16', layers=32, steps=20)
