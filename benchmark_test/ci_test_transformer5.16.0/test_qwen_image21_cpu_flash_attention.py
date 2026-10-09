"""Guard bounded CPU FP calibration against the SDK token-wise golden."""

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("mode", ["padding", "ranges", "causal", "sliding", "multimodal", "window"])
def test_cpu_flash_chunked_calibration_preserves_attention_metadata(dtype, mode):
    from xhquant.backend.hmops.fused_ops.flash_attention.fp._flash_attention_fp_torch import (
        flash_attention_fp_torch,
    )
    from xhquant.backend.hmops.fused_ops.flash_attention.fp._flash_attention_fp_torch_opt import (
        flash_attention_fp_torch_opt,
    )

    torch.manual_seed(4)
    q = torch.randn(2, 4, 264, 16, dtype=dtype)
    k, v = [torch.randn(2, 2, 280, 16, dtype=dtype) for _ in range(2)]
    lengths = torch.tensor([259, 17])
    kv_lengths = torch.tensor([273, 22])
    ranges = None
    if mode == "ranges":
        ranges = torch.zeros(2, 264, 2, dtype=torch.int32)
        ranges[0, 9:259] = torch.tensor([13, 273])
        ranges[1, 3:17] = torch.tensor([2, 22])
    kwargs = dict(
        query_lengths=lengths,
        seq_lengths=kv_lengths,
        is_causal=mode in {"causal", "sliding", "multimodal", "window"},
        attention_mask=torch.randn(2, 264, 280) * 0.1,
        sinks=torch.randn(4),
        scale=16**-0.5,
        num_heads=4,
        sliding_window=11 if mode in {"sliding", "window"} else None,
        mm_prefix_range=torch.tensor([[[15, 38]], [[3, 8]]]) if mode == "multimodal" else None,
        query_start_abs=torch.tensor([100, 30]) if mode == "window" else None,
        kv_window_start_abs=torch.tensor([95, 28]) if mode == "window" else None,
        query_kv_range_abs=ranges,
    )
    expected = flash_attention_fp_torch(q, k, v, **kwargs)
    actual = flash_attention_fp_torch_opt(q, k, v, **kwargs)
    torch.testing.assert_close(
        actual, expected, rtol=1e-5 if dtype == torch.float32 else 0, atol=1e-6 if dtype == torch.float32 else 0
    )
    assert torch.count_nonzero(actual[0, :, 259:]) == 0
    assert torch.count_nonzero(actual[1, :, 17:]) == 0
    k[0, :, 273:] = 1000
    v[0, :, 273:] = -1000
    k[1, :, 22:] = 1000
    v[1, :, 22:] = -1000
    poisoned = flash_attention_fp_torch_opt(q, k, v, **kwargs)
    torch.testing.assert_close(poisoned, actual, atol=0, rtol=0)
