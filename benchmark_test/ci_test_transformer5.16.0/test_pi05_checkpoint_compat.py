# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Canonical Pi0.5 weights must load completely across SigLIP key layouts."""

import pytest
import torch

from xhmodel_merak.xh_other_model.models.pi05._export_utils import _normalize_pi05_checkpoint_keys


PREFIX = "model.paligemma_with_expert.paligemma.model.vision_tower."


@pytest.mark.parametrize("legacy", [False, True])
def test_visual_weights_keep_values_and_names(legacy):
    expected = {PREFIX + "embeddings.patch_embedding.weight": torch.empty(4, 3, 2, 2),
                "model.time_mlp_in.weight": torch.empty(4, 4)}
    visual = torch.arange(48).reshape(4, 3, 2, 2).float()
    time = torch.arange(16).reshape(4, 4).float()
    name = PREFIX + ("vision_model." if legacy else "") + "embeddings.patch_embedding.weight"
    actual = _normalize_pi05_checkpoint_keys({name: visual, "model.time_mlp_in.weight": time}, expected)
    assert set(actual) == set(expected)
    assert actual[PREFIX + "embeddings.patch_embedding.weight"] is visual
    assert actual["model.time_mlp_in.weight"] is time


def test_duplicate_visual_alias_is_rejected():
    name = PREFIX + "embeddings.position_embedding.weight"
    alias = PREFIX + "vision_model.embeddings.position_embedding.weight"
    with pytest.raises(ValueError, match="Duplicate Pi0.5"):
        _normalize_pi05_checkpoint_keys({name: torch.ones(2), alias: torch.zeros(2)}, {name: torch.empty(2)})


def test_unknown_keys_are_preserved_for_strict_load_to_reject():
    name = PREFIX + "vision_model.unknown.weight"
    weights = {name: torch.ones(2)}
    assert _normalize_pi05_checkpoint_keys(weights, {}) == weights


def test_old_model_keeps_old_key_layout():
    name = PREFIX + "vision_model.embeddings.position_embedding.weight"
    weights = {name: torch.arange(4)}
    assert _normalize_pi05_checkpoint_keys(weights, weights)[name] is weights[name]


@pytest.mark.parametrize("tied", [False, True])
def test_only_declared_tied_embedding_can_be_restored(tied):
    head = "model.paligemma_with_expert.paligemma.lm_head.weight"
    embedding = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    weight = torch.arange(8).reshape(4, 2).float()
    actual = _normalize_pi05_checkpoint_keys({head: weight}, {head: weight, embedding: weight},
                                             tied_embeddings=tied)
    assert (embedding in actual) is tied
    if tied:
        assert actual[embedding] is weight


def test_native_pi05_checkpoint_loads_every_tensor(tmp_path, monkeypatch):
    from _pi05_small_fixture import write_pi05
    from safetensors import safe_open
    from xhmodel_merak.xh_other_model.models.pi05._export_utils import load_pi05_policy

    source = tmp_path / "policy"
    write_pi05(source, tmp_path / "tokenizer", monkeypatch)
    policy = load_pi05_policy(str(source))
    actual = policy.state_dict()
    with safe_open(source / "model.safetensors", framework="pt") as weights:
        for name in weights.keys():
            torch.testing.assert_close(actual[name], weights.get_tensor(name), rtol=0, atol=0)
