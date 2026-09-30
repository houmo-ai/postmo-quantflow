import pytest

from vllm_merak.constants import (
    HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST,
    HYBRID_CACHE_POLICY_PAGED,
)
from vllm_merak.model_executor.model_loader.hybrid_cache_policy import (
    parse_full_attention_cache_shapes,
    resolve_hybrid_page_attention,
)


def test_cache_shapes_accept_legacy_symmetric_contract() -> None:
    shapes = parse_full_attention_cache_shapes(
        {"kv_cache_shape": [1, 2, 2048, 128], "cache_axis": 2}
    )

    assert shapes.key == shapes.value == (1, 2, 2048, 128)
    assert shapes.max_cache_tokens == 2048
    assert shapes.symmetric is True


def test_cache_shapes_accept_explicit_asymmetric_contract() -> None:
    shapes = parse_full_attention_cache_shapes(
        {
            "kv_cache_shape": [
                [1, 1, 262144, 576],
                [1, 1, 262144, 512],
            ],
            "cache_axis": 2,
        }
    )

    assert shapes.key == (1, 1, 262144, 576)
    assert shapes.value == (1, 1, 262144, 512)
    assert shapes.max_cache_tokens == 262144
    assert shapes.symmetric is False


def test_cache_shapes_reject_different_kv_capacities() -> None:
    with pytest.raises(ValueError, match="sequence capacities must match"):
        parse_full_attention_cache_shapes(
            {
                "kv_cache_shape": [
                    [1, 1, 2048, 576],
                    [1, 1, 1024, 512],
                ]
            }
        )


def test_hybrid_page_attention_policy_is_fail_closed() -> None:
    assert (
        resolve_hybrid_page_attention(
            cache_policy=HYBRID_CACHE_POLICY_PAGED,
            requested=False,
        )
        is True
    )
    assert (
        resolve_hybrid_page_attention(
            cache_policy=HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST,
            requested=False,
        )
        is False
    )
    with pytest.raises(ValueError, match="does not support PageAttention"):
        resolve_hybrid_page_attention(
            cache_policy=HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST,
            requested=True,
        )
