from types import SimpleNamespace

import pytest

from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import XHGemma4SeriesModel


def _validate(*, prefill_chunk_length: int, bidirectional: bool, sliding_width: int) -> None:
    model = SimpleNamespace(
        sliding_window=1024,
        config=SimpleNamespace(
            prefill_chunk_length=prefill_chunk_length,
            bidirectional_vision_attention=bidirectional,
        ),
    )
    XHGemma4SeriesModel._validate_v2_export_metadata(
        model,
        layer_types=["sliding_attention"],
        layer_cache_types=["sliding_attention"],
        layer_cache_owner_indices=[0],
        layer_cache_indices=[0],
        layer_kv_shapes=[[1, 1, sliding_width, 256]],
    )


def test_nonvisual_contract_uses_actual_prefill_length_for_sliding_cache() -> None:
    _validate(prefill_chunk_length=128, bidirectional=False, sliding_width=1152)


def test_bidirectional_visual_contract_still_requires_280_prefill_tokens() -> None:
    with pytest.raises(ValueError, match="at least 280"):
        _validate(prefill_chunk_length=128, bidirectional=True, sliding_width=1152)


def test_contract_rejects_sliding_width_derived_from_legacy_fixed_320() -> None:
    with pytest.raises(ValueError, match="prefill_chunk_length"):
        _validate(prefill_chunk_length=128, bidirectional=False, sliding_width=1344)
