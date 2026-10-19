"""Eval parameter and loader contracts, without checkpoints or dataset downloads."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from hm_eval.__main__ import _parse_limit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", 0), ("1", 1), ("1.0", 1), ("10", 10), ("10.0", 10), ("0.1", 0.1)],
)
def test_limit_accepts_counts_and_fractions(raw, expected):
    value = _parse_limit(raw)
    assert value == expected
    assert type(value) is type(expected)


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "1e999", "-1", "2.5", "abc"])
def test_limit_rejects_invalid_cli_values(raw):
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=_parse_limit)
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([f"--limit={raw}"])
    assert exc.value.code == 2


@pytest.mark.parametrize(("raw", "expected"), [("0", None), ("1.0", 1), ("10", 10), ("0.1", 0.1)])
def test_limit_is_forwarded_without_evalscope_installation(monkeypatch, tmp_path, raw, expected):
    """Exercise real parsing/forwarding; Evalscope validation is outside this unit contract."""
    from hm_eval.core.eval_runner import run_evaluation

    tasks = []
    # Evalscope is an optional application dependency, absent from ModelZoo CI.
    # Substitute only its API boundary; never replace the runner or limit parser.
    api = {
        "evalscope": {
            "TaskConfig": SimpleNamespace,
            "run_task": lambda *, task_cfg: tasks.append(vars(task_cfg)),
        },
        "evalscope.api": {},
        "evalscope.api.messages": {"ChatMessage": object},
        "evalscope.api.model": {
            "GenerateConfig": SimpleNamespace, "ModelAPI": object, "ModelOutput": object,
        },
        "evalscope.api.registry": {"register_model_api": lambda **kwargs: lambda cls: cls},
        "evalscope.api.tool": {"ToolChoice": object, "ToolInfo": object},
    }
    for name, attributes in api.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        if name in {"evalscope", "evalscope.api"}:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    result = run_evaluation(
        backend=SimpleNamespace(backend_type="float"),
        model_display_name="limit-contract",
        datasets=["ceval"],
        work_dir=str(tmp_path),
        dataset_registry=None,
        limit=_parse_limit(raw),
    )
    assert result["datasets"]["ceval"]["status"] == "completed"
    assert len(tasks) == 1
    if expected is None:
        assert "limit" not in tasks[0]
    else:
        assert tasks[0]["limit"] == expected
        assert type(tasks[0]["limit"]) is type(expected)


@pytest.mark.parametrize(
    "config_id",
    ["gemma4_12b_unified", "gemma4_e2b_it", "gemma4_e4b_it", "gemma4_26b_a4b_it", "gemma4_31b_it"],
)
def test_gemma4_config_selects_its_hf_model_class(monkeypatch, config_id):
    import torch
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration,
    )

    from hm_eval.core.backends import create_backend
    from hm_eval.core.model_registry import load_model_config

    config_path = Path(__file__).resolve().parents[2] / "hm_eval/model_configs" / f"{config_id}.yaml"
    config = load_model_config(config_path)
    model = torch.nn.Linear(2, 2)
    processor = SimpleNamespace(tokenizer=object())
    calls = []

    def load_processor(path, **kwargs):
        assert path == config.hf_model_dir
        assert kwargs == {}
        return processor

    def load_model(cls, path, **kwargs):
        calls.append((cls.__name__, path, kwargs))
        return model

    # Check the real YAML -> factory -> loader dispatch; checkpoint loading is the boundary.
    monkeypatch.setattr(AutoProcessor, "from_pretrained", staticmethod(load_processor))
    for cls in (Gemma4ForConditionalGeneration, Gemma4UnifiedForConditionalGeneration):
        monkeypatch.setattr(cls, "from_pretrained", classmethod(load_model))
    backend = create_backend("float", config)

    expected_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if config_id != "gemma4_12b_unified":
        expected_kwargs["experts_implementation"] = "eager"
    assert calls == [(config.model_class, config.hf_model_dir, expected_kwargs)]
    assert backend.processor is processor
    assert backend.tokenizer is processor.tokenizer
    assert backend.model is model
    assert not model.training
