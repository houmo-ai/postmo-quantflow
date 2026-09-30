from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_merak.spec_decode.artifact import (
    load_hmonnx_spec_decode_artifact,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _touch_graphs(model_dir: Path, *names: str) -> None:
    for name in names:
        path = model_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test graph")


def test_no_spec_decode_contract_returns_none(tmp_path: Path) -> None:
    _write_json(tmp_path / "golden_meta_info.json", {"model_config": {}})

    assert load_hmonnx_spec_decode_artifact(tmp_path) is None
    with pytest.raises(ValueError, match="no speculative-decoding contract"):
        load_hmonnx_spec_decode_artifact(tmp_path, required=True)


def test_loads_qwen_mtp_contract_v2_from_merak_meta_path(
    tmp_path: Path,
) -> None:
    _touch_graphs(tmp_path, "draft/prefill.onnx", "draft/decode.onnx")
    _write_json(
        tmp_path / "merak_config.json",
        {"xh_model": {"meta_info": "meta/golden.json"}},
    )
    (tmp_path / "meta").mkdir()
    _write_json(
        tmp_path / "meta/golden.json",
        {
            "model_config": {"model_type": "Qwen3_5ForConditionalGeneration"},
            "spec_decode": {
                "runtime_contract_version": 2,
                "mode": "mtp",
                "num_draft_tokens": 4,
                "verify_length": 5,
                "target": {
                    "hidden_output_name": "post_norm_hidden",
                    "linear_state_outputs": "per_step",
                },
                "draft": {
                    "abi": "qwen_mtp_v1",
                    "prefill_hmonnx": "../draft/prefill.onnx",
                    "decode_hmonnx": "../draft/decode.onnx",
                    "cache_mutation": "in_place",
                },
            },
        },
    )

    with pytest.raises(ValueError, match="escapes the HMONNX artifact"):
        load_hmonnx_spec_decode_artifact(tmp_path, required=True)

    _touch_graphs(tmp_path / "meta", "draft/prefill.onnx", "draft/decode.onnx")
    metadata = json.loads((tmp_path / "meta/golden.json").read_text())
    metadata["spec_decode"]["draft"].update(
        prefill_hmonnx="draft/prefill.onnx",
        decode_hmonnx="draft/decode.onnx",
    )
    _write_json(tmp_path / "meta/golden.json", metadata)

    artifact = load_hmonnx_spec_decode_artifact(tmp_path, required=True)
    assert artifact is not None
    assert artifact.mode == "mtp"
    assert artifact.num_draft_tokens == 4
    assert artifact.verify_length == 5
    assert artifact.target_linear_state_outputs == "per_step"
    assert artifact.draft_abi == "qwen_mtp_v1"
    assert (
        artifact.draft_graphs.prefill
        == (tmp_path / "meta/draft/prefill.onnx").resolve()
    )


def test_loads_qwen_mtp_private_page_cache_contract_v2(
    tmp_path: Path,
) -> None:
    _touch_graphs(tmp_path, "draft/prefill.onnx", "draft/decode.onnx")
    _write_json(
        tmp_path / "golden_meta_info.json",
        {
            "model_config": {
                "model_type": "Qwen3_5ForConditionalGeneration",
            },
            "spec_decode": {
                "runtime_contract_version": 2,
                "mode": "mtp",
                "num_draft_tokens": 4,
                "verify_length": 5,
                "target": {
                    "hidden_output_name": "post_norm_hidden",
                    "linear_state_outputs": "per_step",
                },
                "draft": {
                    "abi": "qwen_mtp_paged_v2",
                    "prefill_hmonnx": "draft/prefill.onnx",
                    "decode_hmonnx": "draft/decode.onnx",
                    "cache_mutation": "page_attention",
                    "cache_binding": "private_draft",
                    "page_cache_block_size": 64,
                },
            },
        },
    )

    artifact = load_hmonnx_spec_decode_artifact(tmp_path, required=True)

    assert artifact is not None
    assert artifact.uses_private_mtp_page_cache
    assert artifact.is_qwen_mtp
    assert artifact.is_qwen_hybrid_spec_decode
    assert not artifact.is_qwen_dflash
    assert artifact.draft_cache_mutation == "page_attention"
    assert artifact.draft_cache_binding == "private_draft"


def test_loads_legacy_qwen_mtp_contract(tmp_path: Path) -> None:
    _touch_graphs(tmp_path, "mtp/prefill.onnx", "mtp/decode.onnx")
    _write_json(
        tmp_path / "golden_meta_info.json",
        {
            "model_config": {
                "model_type": "Qwen3_5MoeForConditionalGeneration",
                "num_draft_tokens": 2,
            },
            "spec_decode": {
                "mode": "mtp",
                "block_size": 2,
                "hidden_output_name": "post_norm_hidden",
                "draft_prefill_onnx": "mtp/prefill.onnx",
                "draft_decode_onnx": "mtp/decode.onnx",
            },
        },
    )

    artifact = load_hmonnx_spec_decode_artifact(tmp_path, required=True)
    assert artifact is not None
    assert artifact.runtime_contract_version == 1
    assert artifact.num_draft_tokens == 2
    assert artifact.verify_length == 3
    assert artifact.draft_abi == "qwen_mtp_v1"


@pytest.mark.parametrize("verify_length", range(2, 17))
def test_qwen_verify_length_accepts_full_packed_w4_m2_m16_contract(
    tmp_path: Path,
    verify_length: int,
) -> None:
    _touch_graphs(tmp_path, "mtp/prefill.onnx", "mtp/decode.onnx")
    _write_json(
        tmp_path / "golden_meta_info.json",
        {
            "model_config": {
                "model_type": "Qwen3_5ForConditionalGeneration",
            },
            "spec_decode": {
                "mode": "mtp",
                "verify_length": verify_length,
                "draft_prefill_onnx": "mtp/prefill.onnx",
                "draft_decode_onnx": "mtp/decode.onnx",
            },
        },
    )

    artifact = load_hmonnx_spec_decode_artifact(tmp_path, required=True)

    assert artifact is not None
    assert artifact.verify_length == verify_length
    assert artifact.num_draft_tokens == verify_length - 1


@pytest.mark.parametrize("verify_length", [1, 17])
def test_qwen_verify_length_rejects_outside_packed_w4_narrow_contract(
    tmp_path: Path,
    verify_length: int,
) -> None:
    _touch_graphs(tmp_path, "mtp/prefill.onnx", "mtp/decode.onnx")
    _write_json(
        tmp_path / "golden_meta_info.json",
        {
            "model_config": {
                "model_type": "Qwen3_5ForConditionalGeneration",
            },
            "spec_decode": {
                "mode": "mtp",
                "verify_length": verify_length,
                "draft_prefill_onnx": "mtp/prefill.onnx",
                "draft_decode_onnx": "mtp/decode.onnx",
            },
        },
    )

    with pytest.raises(ValueError, match="narrow-token contract"):
        load_hmonnx_spec_decode_artifact(tmp_path, required=True)


def test_dflash_derives_drafts_from_verify_length_and_requires_in_place_v2(
    tmp_path: Path,
) -> None:
    _touch_graphs(
        tmp_path,
        "dflash/context.onnx",
        "dflash/context_decode.onnx",
        "dflash/decode.onnx",
    )
    meta = {
        "model_config": {"model_type": "Qwen3_5ForConditionalGeneration"},
        "spec_decode": {
            "runtime_contract_version": 2,
            "mode": "dflash",
            "verify_length": 10,
            "target": {
                "hidden_output_name": "target_hidden",
                "linear_state_outputs": "per_step",
            },
            "draft": {
                "abi": "qwen_dflash_v1",
                "context_hmonnx": "dflash/context.onnx",
                "context_decode_hmonnx": "dflash/context_decode.onnx",
                "decode_hmonnx": "dflash/decode.onnx",
                "noise_token_id": 248070,
            },
        },
    }
    _write_json(tmp_path / "golden_meta_info.json", meta)

    with pytest.raises(ValueError, match="cache_mutation='in_place'"):
        load_hmonnx_spec_decode_artifact(tmp_path, required=True)

    meta["spec_decode"]["draft"]["cache_mutation"] = "in_place"
    _write_json(tmp_path / "golden_meta_info.json", meta)
    artifact = load_hmonnx_spec_decode_artifact(tmp_path, required=True)
    assert artifact is not None
    assert artifact.num_draft_tokens == 9
    assert artifact.draft_cache_mutation == "in_place"
    assert artifact.dflash_noise_token_id == 248070

    # Token id 0 is valid even though lengths and capacities must be positive.
    meta["spec_decode"]["draft"]["noise_token_id"] = 0
    _write_json(tmp_path / "golden_meta_info.json", meta)
    artifact = load_hmonnx_spec_decode_artifact(tmp_path, required=True)
    assert artifact is not None
    assert artifact.dflash_noise_token_id == 0


def test_qwen_contract_v2_rejects_non_per_step_linear_state_outputs(
    tmp_path: Path,
) -> None:
    _touch_graphs(tmp_path, "mtp/prefill.onnx", "mtp/decode.onnx")
    _write_json(
        tmp_path / "golden_meta_info.json",
        {
            "model_config": {"model_type": "Qwen3_5ForConditionalGeneration"},
            "spec_decode": {
                "runtime_contract_version": 2,
                "mode": "mtp",
                "num_draft_tokens": 4,
                "verify_length": 5,
                "target": {
                    "hidden_output_name": "post_norm_hidden",
                    "linear_state_outputs": "final_only",
                },
                "draft": {
                    "abi": "qwen_mtp_v1",
                    "prefill_hmonnx": "mtp/prefill.onnx",
                    "decode_hmonnx": "mtp/decode.onnx",
                    "cache_mutation": "in_place",
                },
            },
        },
    )

    with pytest.raises(ValueError, match="linear_state_outputs='per_step'"):
        load_hmonnx_spec_decode_artifact(tmp_path, required=True)


def test_dflash_mode_rejects_mtp_abi(tmp_path: Path) -> None:
    _touch_graphs(
        tmp_path,
        "draft/context.onnx",
        "draft/context_decode.onnx",
        "draft/decode.onnx",
    )
    _write_json(
        tmp_path / "golden_meta_info.json",
        {
            "model_config": {"model_type": "Qwen3_5ForConditionalGeneration"},
            "spec_decode": {
                "runtime_contract_version": 2,
                "mode": "dflash",
                "num_draft_tokens": 4,
                "verify_length": 5,
                "draft": {
                    "abi": "qwen_mtp_v1",
                    "context_hmonnx": "draft/context.onnx",
                    "context_decode_hmonnx": "draft/context_decode.onnx",
                    "decode_hmonnx": "draft/decode.onnx",
                },
            },
        },
    )

    with pytest.raises(ValueError, match="DFlash mode requires draft ABI"):
        load_hmonnx_spec_decode_artifact(tmp_path, required=True)


def test_loads_qwen_dflash_private_page_cache_contract_v2(
    tmp_path: Path,
) -> None:
    _touch_graphs(
        tmp_path,
        "draft/context.onnx",
        "draft/context_decode.onnx",
        "draft/decode.onnx",
    )
    _write_json(
        tmp_path / "golden_meta_info.json",
        {
            "model_config": {
                "model_type": "Qwen3_5ForConditionalGeneration",
            },
            "spec_decode": {
                "runtime_contract_version": 2,
                "mode": "dflash",
                "num_draft_tokens": 4,
                "verify_length": 5,
                "target": {
                    "hidden_output_name": "target_hidden",
                    "linear_state_outputs": "per_step",
                },
                "draft": {
                    "abi": "qwen_dflash_paged_v2",
                    "context_hmonnx": "draft/context.onnx",
                    "context_decode_hmonnx": "draft/context_decode.onnx",
                    "decode_hmonnx": "draft/decode.onnx",
                    "cache_mutation": "page_attention",
                    "cache_binding": "private_draft",
                    "page_cache_block_size": 64,
                    "noise_token_id": 248070,
                },
            },
        },
    )

    artifact = load_hmonnx_spec_decode_artifact(tmp_path, required=True)

    assert artifact is not None
    assert artifact.uses_private_dflash_page_cache
    assert artifact.draft_cache_mutation == "page_attention"
    assert artifact.draft_cache_binding == "private_draft"


def test_loads_qwen_dflash_shared_page_cache_writer_contract_v3(
    tmp_path: Path,
) -> None:
    _touch_graphs(
        tmp_path,
        "draft/context.onnx",
        "draft/context_decode.onnx",
        "draft/decode.onnx",
    )
    _write_json(
        tmp_path / "golden_meta_info.json",
        {
            "model_config": {
                "model_type": "Qwen3_5ForConditionalGeneration",
            },
            "spec_decode": {
                "runtime_contract_version": 2,
                "mode": "dflash",
                "num_draft_tokens": 9,
                "verify_length": 10,
                "target": {
                    "hidden_output_name": "target_hidden",
                    "linear_state_outputs": "per_step",
                },
                "draft": {
                    "abi": "qwen_dflash_paged_shared_v3",
                    "context_hmonnx": "draft/context.onnx",
                    "context_decode_hmonnx": "draft/context_decode.onnx",
                    "decode_hmonnx": "draft/decode.onnx",
                    "cache_mutation": "page_attention",
                    "cache_binding": "private_draft",
                    "page_cache_block_size": 64,
                    "noise_token_id": 248070,
                },
            },
        },
    )

    artifact = load_hmonnx_spec_decode_artifact(tmp_path, required=True)

    assert artifact is not None
    assert artifact.uses_private_dflash_page_cache
    assert artifact.uses_graph_dflash_page_cache_writes


@pytest.mark.parametrize(
    ("abi", "cache_mutation", "page_block_size", "uses_page_cache"),
    [
        ("qwen_dflash2_v1", "in_place", None, False),
        ("qwen_dflash2_paged_shared_v1", "page_attention", 64, True),
    ],
)
def test_loads_qwen_dflash2_selector_contract(
    tmp_path: Path,
    abi: str,
    cache_mutation: str,
    page_block_size: int | None,
    uses_page_cache: bool,
) -> None:
    _touch_graphs(
        tmp_path,
        "draft/context.onnx",
        "draft/context_decode.onnx",
        "draft/decode.onnx",
    )
    draft = {
        "abi": abi,
        "context_hmonnx": "draft/context.onnx",
        "context_decode_hmonnx": "draft/context_decode.onnx",
        "decode_hmonnx": "draft/decode.onnx",
        "cache_mutation": cache_mutation,
        "cache_binding": "private_draft",
        "noise_token_id": 248070,
        "architecture": "DFlash2DraftModel",
        "block_size": 8,
        "conv_kernel_size": 2,
        "conv_group_size": 256,
        "selector_rank": 256,
        "selector_top_k": 16,
        "sliding_window": 2048,
        "is_causal": False,
        "selector_scheme": "precomputed_kxk_v1",
        "decode_outputs": [
            "candidate_ids",
            "selector_first_scores",
            "selector_transition_scores",
        ],
    }
    if page_block_size is not None:
        draft["page_cache_block_size"] = page_block_size
    _write_json(
        tmp_path / "golden_meta_info.json",
        {
            "model_config": {
                "model_type": "Qwen3_5ForConditionalGeneration",
            },
            "spec_decode": {
                "runtime_contract_version": 2,
                "mode": "dflash",
                "num_draft_tokens": 7,
                "verify_length": 8,
                "target": {
                    "hidden_output_name": "target_hidden",
                    "linear_state_outputs": "per_step",
                },
                "draft": draft,
            },
        },
    )

    artifact = load_hmonnx_spec_decode_artifact(tmp_path, required=True)

    assert artifact is not None
    assert artifact.is_qwen_dflash
    assert artifact.is_qwen_dflash2
    assert artifact.uses_private_dflash_page_cache is uses_page_cache
    assert artifact.uses_graph_dflash_page_cache_writes is uses_page_cache
    assert artifact.dflash_selector_top_k == 16
    assert artifact.dflash_selector_rank == 256
    assert artifact.dflash_sliding_window == 2048
    assert artifact.dflash_selector_scheme == "precomputed_kxk_v1"


def test_legacy_gemma_mtp_is_marked_dense_shared_kv(tmp_path: Path) -> None:
    _touch_graphs(tmp_path, "mtp/decode.onnx")
    _write_json(
        tmp_path / "golden_meta_info.json",
        {
            "model_config": {
                "model_type": "Gemma4UnifiedForConditionalGeneration",
                "num_draft_tokens": 4,
            },
            "spec_decode": {
                "mode": "mtp",
                "verify_length": 5,
                "draft_decode_onnx": "mtp/decode.onnx",
                "hidden_output_name": "target_hidden_state",
            },
        },
    )

    artifact = load_hmonnx_spec_decode_artifact(tmp_path, required=True)
    assert artifact is not None
    assert artifact.is_gemma4
    assert artifact.draft_abi == "gemma4_mtp_dense_shared_kv_v1"
    assert not artifact.uses_readonly_target_page_cache


def test_gemma_paged_mtp_v2_requires_target_owner_binding(
    tmp_path: Path,
) -> None:
    _touch_graphs(tmp_path, "mtp/decode.onnx")
    meta = {
        "model_config": {
            "model_type": "Gemma4UnifiedForConditionalGeneration",
            "sliding_window": 1024,
        },
        "spec_decode": {
            "runtime_contract_version": 2,
            "mode": "mtp",
            "num_draft_tokens": 4,
            "verify_length": 5,
            "shared_sliding_cache_length": 1028,
            "draft": {
                "abi": "gemma4_mtp_readonly_page_attention_v1",
                "decode_hmonnx": "mtp/decode.onnx",
            },
        },
    }
    _write_json(tmp_path / "golden_meta_info.json", meta)

    with pytest.raises(ValueError, match="target_attention_type_owner"):
        load_hmonnx_spec_decode_artifact(tmp_path, required=True)

    meta["spec_decode"]["draft"]["cache_binding"] = "target_attention_type_owner"
    meta["spec_decode"]["draft"].update(
        cache_mutation="read_only",
        constant_draft_positions=True,
        position_semantics="target_kv_valid_length_minus_one",
        layer_attention_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        minimum_sliding_cache_slack=4,
    )
    _write_json(tmp_path / "golden_meta_info.json", meta)
    artifact = load_hmonnx_spec_decode_artifact(tmp_path, required=True)
    assert artifact is not None
    assert artifact.uses_readonly_target_page_cache
    assert artifact.draft_constant_positions
    # Legacy metadata is accepted, but normalized to the actual query-RoPE
    # position rather than preserving the old misnamed attention length.
    assert artifact.draft_position_semantics == "target_kv_valid_length"
    assert artifact.draft_layer_attention_types[-1] == "full_attention"


def test_gemma_causal_readonly_page_attention_v2_is_explicit(
    tmp_path: Path,
) -> None:
    _touch_graphs(tmp_path, "mtp/decode.onnx")
    draft = {
        "abi": "gemma4_mtp_readonly_page_attention_v2",
        "decode_hmonnx": "mtp/decode.onnx",
        "cache_binding": "target_attention_type_owner",
        "cache_mutation": "read_only",
        "constant_draft_positions": True,
        "query_rope_position_semantics": "target_kv_valid_length",
        "attention_past_length_semantics": ("target_kv_valid_length_minus_one"),
        "layer_attention_types": [
            "sliding_attention",
            "full_attention",
        ],
        "minimum_sliding_cache_slack": 4,
    }
    meta = {
        "model_config": {
            "model_type": "Gemma4UnifiedForConditionalGeneration",
            "sliding_window": 1024,
        },
        "spec_decode": {
            "runtime_contract_version": 2,
            "mode": "mtp",
            "num_draft_tokens": 4,
            "verify_length": 5,
            "shared_sliding_cache_length": 1028,
            "draft": draft,
        },
    }
    _write_json(tmp_path / "golden_meta_info.json", meta)
    with pytest.raises(ValueError, match="attention_lowering='causal'"):
        load_hmonnx_spec_decode_artifact(tmp_path, required=True)

    draft["attention_lowering"] = "causal"
    _write_json(tmp_path / "golden_meta_info.json", meta)
    artifact = load_hmonnx_spec_decode_artifact(tmp_path, required=True)
    assert artifact is not None
    assert artifact.uses_readonly_target_page_cache
    assert artifact.uses_causal_readonly_target_page_cache
    assert artifact.draft_position_semantics == "target_kv_valid_length"


def test_gemma_mixed_readonly_page_attention_v3_is_explicit(
    tmp_path: Path,
) -> None:
    _touch_graphs(tmp_path, "mtp/decode.onnx")
    draft = {
        "abi": "gemma4_mtp_readonly_mixed_attention_v3",
        "decode_hmonnx": "mtp/decode.onnx",
        "cache_binding": "target_attention_type_owner",
        "cache_mutation": "read_only",
        "constant_draft_positions": True,
        "query_rope_position_semantics": "target_kv_valid_length",
        "attention_past_length_semantics": "target_kv_valid_length_minus_one",
        "layer_attention_types": [
            "sliding_attention",
            "full_attention",
        ],
        "minimum_sliding_cache_slack": 4,
    }
    meta = {
        "model_config": {
            "model_type": "Gemma4UnifiedForConditionalGeneration",
            "sliding_window": 1024,
        },
        "spec_decode": {
            "runtime_contract_version": 2,
            "mode": "mtp",
            "num_draft_tokens": 4,
            "verify_length": 5,
            "shared_sliding_cache_length": 1028,
            "draft": draft,
        },
    }
    _write_json(tmp_path / "golden_meta_info.json", meta)
    with pytest.raises(ValueError, match="full_page_sliding_unfused"):
        load_hmonnx_spec_decode_artifact(tmp_path, required=True)

    draft["attention_lowering"] = "full_page_sliding_unfused"
    _write_json(tmp_path / "golden_meta_info.json", meta)
    artifact = load_hmonnx_spec_decode_artifact(tmp_path, required=True)

    assert artifact is not None
    assert artifact.uses_readonly_target_page_cache
    assert artifact.uses_mixed_readonly_target_page_cache
    assert not artifact.uses_causal_readonly_target_page_cache


def test_gemma_paged_mtp_rejects_insufficient_sliding_cache_slack(
    tmp_path: Path,
) -> None:
    _touch_graphs(tmp_path, "mtp/decode.onnx")
    _write_json(
        tmp_path / "golden_meta_info.json",
        {
            "model_config": {
                "model_type": "Gemma4UnifiedForConditionalGeneration",
                "sliding_window": 1024,
            },
            "spec_decode": {
                "runtime_contract_version": 2,
                "mode": "mtp",
                "num_draft_tokens": 4,
                "verify_length": 5,
                "shared_sliding_cache_length": 1027,
                "draft": {
                    "abi": "gemma4_mtp_readonly_page_attention_v1",
                    "decode_hmonnx": "mtp/decode.onnx",
                    "cache_binding": "target_attention_type_owner",
                    "cache_mutation": "read_only",
                    "constant_draft_positions": True,
                    "position_semantics": ("target_kv_valid_length_minus_one"),
                    "layer_attention_types": [
                        "sliding_attention",
                        "full_attention",
                    ],
                    "minimum_sliding_cache_slack": 4,
                },
            },
        },
    )

    with pytest.raises(ValueError, match="insufficient rollback slack"):
        load_hmonnx_spec_decode_artifact(tmp_path, required=True)
