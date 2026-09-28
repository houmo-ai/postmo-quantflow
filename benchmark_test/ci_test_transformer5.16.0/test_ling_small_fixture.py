# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Synthetic Ling weights and calibration must be independent of allocator state."""

import json

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from _unified_cases import CASES
from _unified_e2e_worker import prepare
from _unified_special_fixtures import write_ling


def test_ling_checkpoint_initializes_all_parameters(tmp_path):
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        # Real allocations are poisoned by PyTorch, without mocking model init.
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        write_ling(tmp_path / "first")
        write_ling(tmp_path / "second")
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = fill
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    first = load_file(tmp_path / "first" / "model.safetensors")
    second = load_file(tmp_path / "second" / "model.safetensors")
    assert "model.layers.0.attention.dt_bias" in first, "Keep the real KDA layer"
    assert first.keys() == second.keys()
    for name, value in first.items():
        assert torch.isfinite(value).all(), name
        assert torch.equal(value, second[name]), name


def test_ling_calibration_uses_two_local_rows_and_short_tuning(tmp_path, monkeypatch):
    import xhmodel_merak.model_build.api as api

    case = next(c for c in CASES if c.fixture == "ling")
    kwargs, inputs = prepare(case, tmp_path, monkeypatch)
    plan = api.resolve_build_plan(**kwargs)
    replacements = plan.config_replacements
    assert plan.quant_method == "autoround"
    assert replacements["quant.calibration.jsonl"] == str(tmp_path / "calibration.jsonl")
    assert replacements["quant.calibration.nsamples"] == 2
    assert replacements["quant.calibration.seqlen"] == 64
    assert replacements["quant.iters"] == 4
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "source", trust_remote_code=True)
    rows = [json.loads(line)["text"] for line in (tmp_path / "calibration.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for text in rows:
        ids = tokenizer(text)["input_ids"]
        assert len(ids) >= 64
        assert tokenizer.unk_token_id not in ids
        assert ids[:64].count(ids[63]) <= 32
    assert tokenizer.unk_token_id not in tokenizer(inputs["prompt"])["input_ids"]
