"""CPU contract for LingBot FlashAttention output layout.

Dense export keeps the official [B, H, S, D] scale path. The extra transpose
used by the 30B MoE adaptation stays behind an explicit router mode.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_other_model.models.lingbot_video.transformer_wrapper import (
    _LingBotVideoAttention,
)


@pytest.mark.transformer516
@pytest.mark.parametrize(
    ("router_precision", "legacy_transpose"),
    [
        (None, False),
        ("fp32", False),
        ("fixed_int32", True),
        ("half_add", True),
    ],
)
def test_lingbot_attention_layout_follows_router_choice_mode(monkeypatch, router_precision, legacy_transpose):
    if router_precision is None:
        monkeypatch.delenv("LINGBOT_MOE_ROUTER_PRECISION", raising=False)
    else:
        monkeypatch.setenv("LINGBOT_MOE_ROUTER_PRECISION", router_precision)

    batch_size, sequence_length, num_heads, head_dim = 1, 6, 2, 4
    value_scale = 4.0
    canonical_bhsd = torch.arange(
        batch_size * num_heads * sequence_length * head_dim,
        dtype=torch.float32,
    ).reshape(batch_size, num_heads, sequence_length, head_dim)
    current_input_length = torch.tensor([sequence_length - 1], dtype=torch.int32)

    class FakeFlashAttention:
        def __call__(self, query, key, value, *, current_input_length):
            assert query.shape == key.shape == value.shape == canonical_bhsd.shape
            assert current_input_length is expected_current_input_length
            return canonical_bhsd.clone()

    expected_current_input_length = current_input_length
    attention = SimpleNamespace(
        lingbot_export_batch_size=batch_size,
        lingbot_export_sequence_length=sequence_length,
        lingbot_flash_attention_value_scale=value_scale,
        num_heads=num_heads,
        head_dim=head_dim,
        to_q=torch.nn.Identity(),
        to_k=torch.nn.Identity(),
        to_v=torch.nn.Identity(),
        norm_q=torch.nn.Identity(),
        norm_k=torch.nn.Identity(),
        flash_attn=FakeFlashAttention(),
        to_out=torch.nn.Identity(),
        _apply_rotary=lambda hidden_states, rotary_emb: hidden_states,
    )
    hidden_states = torch.zeros(batch_size, sequence_length, num_heads * head_dim)

    actual = _LingBotVideoAttention.forward(
        attention,
        hidden_states,
        rotary_emb=(torch.empty(0), torch.empty(0)),
        current_input_length=current_input_length,
    )

    output = canonical_bhsd
    if legacy_transpose:
        output = output.transpose(1, 2).contiguous()
    output = output * value_scale
    expected = output.transpose(1, 2).reshape(batch_size, sequence_length, num_heads * head_dim)
    torch.testing.assert_close(actual, expected)

    dense_layout = (
        canonical_bhsd.mul(value_scale)
        .transpose(1, 2)
        .reshape(
            batch_size,
            sequence_length,
            num_heads * head_dim,
        )
    )
    if legacy_transpose:
        assert not torch.equal(actual, dense_layout)
    else:
        torch.testing.assert_close(actual, dense_layout)
    assert os.environ.get("LINGBOT_MOE_ROUTER_PRECISION") == router_precision
