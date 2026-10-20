"""Real tiny Qwen4 text model: image positions, HC output and Host state ABI."""

import pytest
import torch
from transformers import Qwen4ExpTextConfig

from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import Qwen4ExpCacheContract
from xhmodel_merak.xh_llm.models.qwen4_exp.qwen4_exp_hmonnx_inference import (
    Qwen4ExpServingDataPreprocess,
    Qwen4ExpServingState,
)
from xhmodel_merak.xh_llm.mrope import prepare_mrope_position_inputs


def _config():
    return Qwen4ExpTextConfig(
        max_position_embeddings=16,
        vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        num_experts=2, num_experts_per_tok=1, hc_count=2, hc_lowrank=4,
        layer_types=["full_attention"], ple_layer_ids=[],
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=4, indexer_compress_ratio=4,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5},
    )


def test_group_start_positions_match_multiaxis_history_across_chunks():
    positions = torch.tensor([[[0, 1, 1, 1, 2, 2, 2, 3, 4, 4]],
                              [[0, 0, 0, 1, 1, 2, 2, 2, 3, 4]],
                              [[0, 0, 1, 0, 1, 0, 1, 2, 0, 1]]], dtype=torch.int32)
    group_start = torch.zeros(3, 1, 1, dtype=torch.int32)
    past = 0
    for valid in (3, 2, 4, 1):
        current, groups, group_start = prepare_mrope_position_inputs(
            group_start, positions[:, :, past:past + valid], past, 4, valid,
        )
        torch.testing.assert_close(current[:, :, :valid], positions[:, :, past:past + valid])
        count = (past % 4 + valid) // 4
        starts = torch.arange(past // 4, past // 4 + count) * 4
        torch.testing.assert_close(groups[:, :, :count], positions.index_select(2, starts))
        remaining = (past + valid) % 4
        expected = (positions[:, :, past + valid - remaining:past + valid - remaining + 1]
                    if remaining else torch.zeros_like(group_start))
        torch.testing.assert_close(group_start, expected)
        past += valid


def test_host_rope_inputs_preserve_multiaxis_group_starts_and_padding():
    config = _config()
    contract = Qwen4ExpCacheContract.from_text_config(config)
    state = Qwen4ExpServingState(contract, batch_size=1, context_length=8, dtype=torch.float32, device="cpu")
    state.group_start_position.copy_(torch.tensor([[[4]], [[7]], [[1]]], dtype=torch.int32))
    processor = Qwen4ExpServingDataPreprocess(
        token_embedding=torch.nn.Embedding(32, 16), input_sequence_length=4,
        state=state, past_key_caches=None, past_value_caches=None,
        enable_page_attention=True, pad_token_id=0,
    )
    positions = torch.tensor([[[1, 1]], [[1, 2]], [[2, 3]]])
    inputs = processor({"input_ids": torch.tensor([[3, 4]]), "past_seq_length": 3, "position_ids": positions})
    current = torch.stack(inputs[4:7])
    group = torch.stack(inputs[7:10])
    torch.testing.assert_close(current[:, :, :2], positions.int())
    assert torch.count_nonzero(current[:, :, 2:]) == 0
    torch.testing.assert_close(group, torch.tensor([[[4]], [[7]], [[1]]], dtype=torch.int32))
    torch.testing.assert_close(state.group_start_position, positions[:, :, 1:2].int())
    assert "past_position_ids" not in state.tensors
    with pytest.raises(ValueError, match="shape"):
        processor({"input_ids": torch.tensor([[3, 4]]), "position_ids": torch.ones(1, 1, 2)})


def test_auxiliary_hc_output_does_not_shift_state_commits():
    contract = Qwen4ExpCacheContract.from_text_config(_config())
    state = Qwen4ExpServingState(contract, batch_size=1, context_length=8, dtype=torch.float32, device="cpu")
    key_tail = torch.full((1, 1, 3, 8), 7, dtype=torch.float32)
    logits, hidden = torch.randn(1, 1, 32), torch.randn(1, 1, 32)
    actual = state.commit_outputs((logits, hidden, key_tail),
                                  ("logits", "target_hidden_states", "present_qsa_key_tail_0"))
    assert actual is logits
    assert state.tensors["qsa_key_tail_0"] is key_tail
    assert "target_hidden_states" not in state.tensors
