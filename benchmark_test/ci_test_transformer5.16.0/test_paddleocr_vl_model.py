# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest
import torch
from transformers import PaddleOCRVLConfig, PaddleOCRVLForConditionalGeneration

from xhmodel_merak.xh_llm.models.paddleocr_vl._llm_model_impl import register_wrap_cls
from xhmodel_merak.xh_llm.models.paddleocr_vl.hf import load_paddleocr_config
from xhmodel_merak.xh_llm.wrap_model import wrap_llm_model
from xhquant.api import ConfigDict
from xhquant.core import CacheTensor
from xhquant.frontend.utils import module_graph_forward_guard


def tiny_config():
    return PaddleOCRVLConfig(
        tie_word_embeddings=False,
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         max_position_embeddings=64, tie_word_embeddings=False,
                         rope_parameters={"rope_type": "default", "rope_theta": 500000.0,
                                          "mrope_section": [2, 3, 3]}),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=4, image_size=28, patch_size=14),
        image_token_id=60, vision_start_token_id=61, vision_end_token_id=62,
    )


def _eager_model():
    model = PaddleOCRVLForConditionalGeneration(tiny_config()).eval()
    model.config._attn_implementation = "eager"
    model.config.text_config._attn_implementation = "eager"
    return model


def _wrap_cfg(input_sequence_length, *, max_sequence_length=64, num_logits_to_keep=0, use_cache=False):
    cfg = ConfigDict(
        input_sequence_length=input_sequence_length,
        max_sequence_length=max_sequence_length,
        max_pe_length=64,
        batch_size=1,
        num_logits_to_keep=num_logits_to_keep,
        use_cache=use_cache,
        enable_rope=False,
    )
    if use_cache:
        cfg.kv_cache = ConfigDict(cache_axis=2)
    return cfg


def _cache_lists_from_native(native_cache, max_sequence_length):
    key_caches = []
    value_caches = []
    for layer in native_cache.layers:
        keys = torch.zeros(
            *layer.keys.shape[:2], max_sequence_length, layer.keys.shape[-1],
            dtype=layer.keys.dtype,
        )
        values = torch.zeros(
            *layer.values.shape[:2], max_sequence_length, layer.values.shape[-1],
            dtype=layer.values.dtype,
        )
        keys[:, :, : layer.keys.shape[2], :] = layer.keys
        values[:, :, : layer.values.shape[2], :] = layer.values
        key_caches.append(CacheTensor(keys))
        value_caches.append(CacheTensor(values))
    return key_caches, value_caches


def test_explicit_untied_checkpoint_config_is_preserved(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"tie_word_embeddings": False, "model_type": "paddleocr_vl"}))
    config = load_paddleocr_config(tmp_path)
    assert not config.tie_word_embeddings
    assert not config.text_config.tie_word_embeddings


@pytest.mark.parametrize("different_axes", [False, True])
def test_rectangular_attention_and_mrope_match_native(different_axes):
    torch.manual_seed(3)
    model = _eager_model()
    inputs = torch.randn(1, 4, 32)
    positions = torch.arange(4).expand(3, 1, -1).clone()
    if different_axes:
        positions[1] = torch.tensor([[0, 1, 1, 2]])
        positions[2] = torch.tensor([[0, 0, 1, 2]])
    with torch.no_grad():
        expected = model.lm_head(model.model.language_model(
            inputs_embeds=inputs, position_ids=positions, use_cache=False,
        ).last_hidden_state)
    register_wrap_cls()
    wrapped = wrap_llm_model(copy.deepcopy(model), _wrap_cfg(4))
    assert "language_model" in wrapped.model._modules
    assert "visual" not in wrapped.model._modules
    assert "projector" not in wrapped.model._modules
    assert "lm_head" in wrapped._modules
    with torch.no_grad(), module_graph_forward_guard(wrapped):
        actual = wrapped(inputs, *positions[:, 0].unbind(), torch.tensor([0], dtype=torch.int32),
                         torch.tensor([4], dtype=torch.int32), None, None)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_default_single_logit_uses_current_input_length_for_underfull_chunk():
    torch.manual_seed(5)
    model = _eager_model()
    inputs = torch.randn(1, 4, 32)
    positions = torch.arange(4).expand(3, 1, -1).clone()
    positions[1] = torch.tensor([[0, 1, 2, 2]])
    positions[2] = torch.tensor([[0, 0, 1, 2]])
    with torch.no_grad():
        hidden = model.model.language_model(
            inputs_embeds=inputs, position_ids=positions, use_cache=False,
        ).last_hidden_state
        expected = model.lm_head(hidden[:, 2:3])

    register_wrap_cls()
    wrapped = wrap_llm_model(copy.deepcopy(model), _wrap_cfg(4, num_logits_to_keep=1))
    with torch.no_grad(), module_graph_forward_guard(wrapped):
        actual = wrapped(
            inputs,
            *positions[:, 0].unbind(),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
            None,
            None,
        )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_decode_with_real_prefill_kv_cache_matches_native_continuation():
    torch.manual_seed(7)
    model = _eager_model()
    prefill_inputs = torch.randn(1, 3, 32)
    padded_prefill_inputs = torch.cat([prefill_inputs, torch.zeros(1, 1, 32)], dim=1)
    decode_inputs = torch.randn(1, 1, 32)
    prefill_positions = torch.arange(3).expand(3, 1, -1).clone()
    prefill_positions[1] = torch.tensor([[0, 1, 1]])
    prefill_positions[2] = torch.tensor([[0, 0, 1]])
    padded_prefill_positions = torch.cat([prefill_positions, torch.zeros(3, 1, 1, dtype=torch.long)], dim=2)
    decode_positions = torch.tensor([[[3]], [[2]], [[2]]])

    with torch.no_grad():
        prefill = model.model.language_model(
            inputs_embeds=prefill_inputs,
            position_ids=prefill_positions,
            use_cache=True,
        )
        expected_prefill = model.lm_head(prefill.last_hidden_state[:, 2:3])
        expected = model.lm_head(model.model.language_model(
            inputs_embeds=decode_inputs,
            position_ids=decode_positions,
            past_key_values=prefill.past_key_values,
            use_cache=True,
        ).last_hidden_state)

    max_sequence_length = 8
    empty_cache = _cache_lists_from_native(prefill.past_key_values, max_sequence_length)
    past_key_cache = [CacheTensor(torch.zeros_like(cache.data)) for cache in empty_cache[0]]
    past_value_cache = [CacheTensor(torch.zeros_like(cache.data)) for cache in empty_cache[1]]
    register_wrap_cls()
    prefill_wrapped = wrap_llm_model(copy.deepcopy(model), _wrap_cfg(4, max_sequence_length=max_sequence_length,
                                                                     num_logits_to_keep=1, use_cache=True))
    with torch.no_grad(), module_graph_forward_guard(prefill_wrapped):
        actual_prefill = prefill_wrapped(
            padded_prefill_inputs,
            *padded_prefill_positions[:, 0].unbind(),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
            past_key_cache,
            past_value_cache,
        )
    torch.testing.assert_close(actual_prefill, expected_prefill, rtol=2e-5, atol=2e-5)

    decode_wrapped = wrap_llm_model(copy.deepcopy(model), _wrap_cfg(1, max_sequence_length=max_sequence_length,
                                                                    num_logits_to_keep=1, use_cache=True))
    with torch.no_grad(), module_graph_forward_guard(decode_wrapped):
        actual = decode_wrapped(
            decode_inputs,
            *decode_positions[:, 0].unbind(),
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            past_key_cache,
            past_value_cache,
        )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
