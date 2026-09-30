import pytest

from hybrid_page_padding_testkit import (
    PHASE1_RECOMMENDED_BLOCKS,
    QWEN35_9B,
    QWEN36_35B,
    exact_hmfp_bytes,
    phase1_linear_bytes,
    phase1_physical_bytes,
    recommended_blocks,
    required_blocks,
    virtual_scheduler_bytes,
)


@pytest.mark.parametrize(
    (
        "geometry",
        "expected_virtual",
        "expected_full",
        "expected_linear",
        "expected_total",
    ),
    [
        (
            QWEN36_35B,
            47_951_380_480,
            2_864_742_400,
            33_423_360,
            2_898_165_760,
        ),
        (
            QWEN35_9B,
            38_361_104_384,
            4_583_587_840,
            26_738_688,
            4_610_326_528,
        ),
    ],
)
def test_phase1_capacity_oracle_is_byte_exact(
    geometry,
    expected_virtual,
    expected_full,
    expected_linear,
    expected_total,
) -> None:
    assert (
        virtual_scheduler_bytes(geometry, PHASE1_RECOMMENDED_BLOCKS) == expected_virtual
    )
    assert exact_hmfp_bytes(geometry, PHASE1_RECOMMENDED_BLOCKS) == expected_full
    assert phase1_linear_bytes(geometry) == expected_linear
    assert phase1_physical_bytes(geometry, PHASE1_RECOMMENDED_BLOCKS) == expected_total


def test_phase1_256k_required_and_recommended_blocks_are_exact() -> None:
    minimum = required_blocks(max_model_len=262_144, block_size=64, linear_groups=3)

    assert minimum == 4_099
    assert recommended_blocks(minimum, reserve_ratio=0.05) == PHASE1_RECOMMENDED_BLOCKS
    assert PHASE1_RECOMMENDED_BLOCKS * 64 >= 262_144
    assert (minimum - 1) * 64 < 262_144 + 3 * 64


@pytest.mark.parametrize(
    ("geometry", "limit_gib"),
    [(QWEN36_35B, 2.8), (QWEN35_9B, 4.5)],
)
def test_phase1_physical_projection_meets_delivery_limit(geometry, limit_gib) -> None:
    projected = phase1_physical_bytes(geometry, PHASE1_RECOMMENDED_BLOCKS)

    assert projected <= int(limit_gib * 1024**3)
    assert projected < virtual_scheduler_bytes(geometry, PHASE1_RECOMMENDED_BLOCKS)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_model_len": 0}, "max_model_len"),
        ({"max_model_len": 1, "block_size": 0}, "block_size"),
        ({"max_model_len": 1, "linear_groups": -1}, "linear_groups"),
    ],
)
def test_capacity_oracle_rejects_invalid_inputs(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        required_blocks(**kwargs)
