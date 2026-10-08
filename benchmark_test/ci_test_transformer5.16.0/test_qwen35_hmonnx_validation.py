"""Lightweight tests for Qwen3.5 HMONNX runtime quick-test helpers."""
from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_lightweight_xh_llm_packages(monkeypatch) -> None:
    packages = {
        "xhmodel_merak.xh_llm": REPO_ROOT / "xhmodel_merak/xh_llm",
        "xhmodel_merak.xh_llm.workflows": REPO_ROOT / "xhmodel_merak/xh_llm/workflows",
        "xhmodel_merak.xh_llm.models": REPO_ROOT / "xhmodel_merak/xh_llm/models",
        "xhmodel_merak.xh_llm.models.qwen3_5": REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_5",
    }
    for name, path in packages.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, module)


def _load_runtime_module(monkeypatch):
    _install_lightweight_xh_llm_packages(monkeypatch)
    return importlib.import_module("xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation")


def _install_fake_generation_dependencies(monkeypatch):
    import torch

    calls = {"max_new_tokens": [], "prompts": []}

    class Batch(dict):
        def __init__(self):
            self.input_ids = torch.tensor([[1, 2]])
            super().__init__(input_ids=self.input_ids)

        def to(self, _device):
            return self

    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, **_kwargs):
            calls["prompts"].append(messages[0]["content"])
            return "rendered"

        def __call__(self, _texts, **_kwargs):
            return Batch()

        def decode(self, _tokens, **_kwargs):
            return "ok"

    class Component:
        enable_golden = False

    class Model:
        prefill_model = Component()
        decode_model = Component()
        visual = Component()

        def get_tokenizer(self):
            return Tokenizer()

        def to(self, _device):
            return self

        def generate(self, **kwargs):
            calls["max_new_tokens"].append(kwargs["max_new_tokens"])
            return torch.cat([kwargs["input_ids"], torch.tensor([[3]])], dim=1)

    model = Model()

    class AutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return model

    class NullContext:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    xh_llm = sys.modules["xhmodel_merak.xh_llm"]
    monkeypatch.setattr(xh_llm, "AutoLLMHONNXModel", AutoModel, raising=False)
    monkeypatch.setattr(xh_llm, "LLMInferenceContextManager", NullContext, raising=False)

    transformers = types.ModuleType("transformers")
    transformers.TextStreamer = lambda _tokenizer: object()
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    xhquant = types.ModuleType("xhquant")
    xhquant.__path__ = []
    xhquant_api = types.ModuleType("xhquant.api")
    xhquant_api.get_xhquant_logger = lambda: types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    xhquant_api.xhquant_init = lambda *_args, **_kwargs: None
    xhquant_utils = types.ModuleType("xhquant.utils")
    xhquant_utils.ContextManagers = NullContext
    xhquant_utils.MemoryTracker = NullContext
    xhquant_utils.TimeProfiler = NullContext
    xhquant.api = xhquant_api
    xhquant.utils = xhquant_utils
    monkeypatch.setitem(sys.modules, "xhquant", xhquant)
    monkeypatch.setitem(sys.modules, "xhquant.api", xhquant_api)
    monkeypatch.setitem(sys.modules, "xhquant.utils", xhquant_utils)
    return calls


def test_find_hmonnx_meta_file_accepts_export_result_dir_and_meta(monkeypatch, tmp_path: Path):
    runtime = _load_runtime_module(monkeypatch)
    result_mod = importlib.import_module("xhmodel_merak.xh_llm.workflows.result")
    meta_file = tmp_path / "export" / "hmquant_qwen35" / "golden_meta_info.json"
    meta_file.parent.mkdir(parents=True)
    meta_file.write_text("{}", encoding="utf-8")

    export_result = result_mod.ExportResult(work_dir=str(tmp_path / "export"), config_file="effective.yaml")

    assert runtime.find_hmonnx_meta_file(export_result) == str(meta_file)
    assert runtime.find_hmonnx_meta_file(tmp_path / "export") == str(meta_file)
    assert runtime.find_hmonnx_meta_file(meta_file) == str(meta_file)


def test_find_hmonnx_meta_file_accepts_standalone_visual_export(monkeypatch, tmp_path: Path):
    runtime = _load_runtime_module(monkeypatch)
    visual_meta = tmp_path / "visual_meta_info.json"
    visual_meta.write_text('{"visual_input_mode": "patches", "gears": [{}]}', encoding="utf-8")

    assert runtime.find_hmonnx_meta_file(tmp_path) == str(visual_meta)


def test_quick_test_hmonnx_routes_plain_and_spec_decode_meta(monkeypatch, tmp_path: Path):
    runtime = _load_runtime_module(monkeypatch)
    calls: list[tuple[str, str, str | None]] = []

    def fake_generate(meta_file, **kwargs):
        calls.append(("generate", str(meta_file), os.environ.get("ENABLE_HMINFERENCE_V2")))
        return runtime.HMONNXQuickTestResult(
            meta_file=str(meta_file),
            output_text="plain",
            output_tokens=1,
            latency_s=0.5,
            tokens_per_second=2.0,
        )

    def fake_spec(meta_file, **kwargs):
        calls.append(("spec", str(meta_file), os.environ.get("ENABLE_HMINFERENCE_V2")))
        return runtime.HMONNXQuickTestResult(
            meta_file=str(meta_file),
            output_text="spec",
            output_tokens=2,
            latency_s=1.0,
            tokens_per_second=2.0,
            spec_decode_mode="mtp",
            draft_tokens_total=8,
            accepted_drafts_total=6,
            accept_rate=0.75,
        )

    def fake_visual(*, meta_file, device):
        calls.append(("visual", str(meta_file), os.environ.get("ENABLE_HMINFERENCE_V2")))
        return runtime.HMONNXQuickTestResult(
            meta_file=str(meta_file),
            output_text=f"visual:{device}",
            output_tokens=0,
            latency_s=1.0,
            tokens_per_second=0.0,
        )

    monkeypatch.setattr(runtime, "hmonnx_generate", fake_generate)
    monkeypatch.setattr(runtime, "spec_decode_generate", fake_spec)
    monkeypatch.setattr(runtime, "visual_hmonnx_smoke_test", fake_visual)

    plain_meta = tmp_path / "plain.json"
    plain_meta.write_text("{}", encoding="utf-8")
    spec_meta = tmp_path / "spec.json"
    spec_meta.write_text(json.dumps({"spec_decode": {"mode": "mtp"}}), encoding="utf-8")
    visual_meta = tmp_path / "visual_meta_info.json"
    visual_meta.write_text(
        json.dumps({"visual_input_mode": "patches", "gears": [{"image_token_capacity": 96}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv("ENABLE_HMINFERENCE_V2", raising=False)

    assert runtime.quick_test_hmonnx(plain_meta).output_text == "plain"
    assert runtime.quick_test_hmonnx(spec_meta).accept_rate == 0.75
    assert runtime.quick_test_hmonnx(visual_meta, device="cuda:3").output_text == "visual:cuda:3"
    assert calls == [
        ("generate", str(plain_meta), "1"),
        ("spec", str(spec_meta), "1"),
        ("visual", str(visual_meta), "1"),
    ]
    assert "ENABLE_HMINFERENCE_V2" not in os.environ


def test_build_text_inputs_disables_token_type_ids(monkeypatch):
    runtime = _load_runtime_module(monkeypatch)
    calls = {}

    class Batch:
        def to(self, device):
            calls["device"] = device
            return self

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls["messages"] = messages
            calls["template_kwargs"] = kwargs
            return "rendered prompt"

        def __call__(self, texts, **kwargs):
            calls["texts"] = texts
            calls["tokenizer_kwargs"] = kwargs
            return Batch()

    tokenizer = Tokenizer()
    model = types.SimpleNamespace(get_tokenizer=lambda: tokenizer)

    loaded_tokenizer, model_inputs = runtime._build_text_inputs(model, "hello", False, "cuda:6")

    assert loaded_tokenizer is tokenizer
    assert isinstance(model_inputs, Batch)
    assert calls["tokenizer_kwargs"] == {
        "return_tensors": "pt",
        "truncation": True,
        "return_token_type_ids": False,
    }
    assert calls["device"] == "cuda:6"


def test_preprocess_debug_saves_raw_input_rendered_text_and_token_ids(monkeypatch, tmp_path: Path):
    runtime = _load_runtime_module(monkeypatch)
    image_path = tmp_path / "input.png"
    meta_path = tmp_path / "golden_meta_info.json"
    meta_path.write_text("{}", encoding="utf-8")
    from PIL import Image

    Image.new("RGB", (3, 2)).save(image_path)

    class Tokenizer:
        def decode(self, token_ids, **kwargs):
            assert token_ids == [11, 12, 13]
            assert kwargs == {
                "skip_special_tokens": False,
                "clean_up_tokenization_spaces": False,
            }
            return "<|im_start|>user\nhello<|im_end|>"

    model_inputs = types.SimpleNamespace(input_ids=torch.tensor([[11, 12, 13]]))
    golden_input_dir = runtime._dump_golden_input(
        meta_file=meta_path,
        prompt="hello",
        image_path=str(image_path),
        image_used=True,
        tokenizer=Tokenizer(),
        model_inputs=model_inputs,
    )

    assert golden_input_dir == tmp_path / "golden_input"
    assert (golden_input_dir / "prompt.txt").read_text(encoding="utf-8") == "hello"
    assert (golden_input_dir / "tokenizer_output.txt").read_text(encoding="utf-8") == (
        "<|im_start|>user\nhello<|im_end|>"
    )
    assert torch.equal(
        torch.from_numpy(np.load(golden_input_dir / "token_ids.npy")), torch.tensor([[11, 12, 13]])
    )
    assert (golden_input_dir / "input_image.png").read_bytes() == image_path.read_bytes()
    debug_meta = json.loads((golden_input_dir / "input_meta.json").read_text(encoding="utf-8"))
    assert debug_meta["image"]["used"] is True
    assert debug_meta["image"]["size"] == [3, 2]

    next_golden_input_dir = runtime._dump_golden_input(
        meta_file=meta_path,
        prompt="next",
        image_path=None,
        image_used=False,
        tokenizer=Tokenizer(),
        model_inputs=model_inputs,
    )
    assert next_golden_input_dir == tmp_path / "golden_input"
    assert (next_golden_input_dir / "prompt.txt").read_text(encoding="utf-8") == "next"
    assert not (next_golden_input_dir / "input_image.png").exists()
    assert not (tmp_path / "golden_input/decode").exists()


def test_hmonnx_generate_keeps_path_like_prompts_literal_and_preserves_supported_golden_limits(
    monkeypatch, tmp_path: Path
):
    runtime = _load_runtime_module(monkeypatch)
    calls = _install_fake_generation_dependencies(monkeypatch)
    prompt = tmp_path / "worker-local-prompt.txt"
    prompt.write_text("filesystem contents must not replace the prompt", encoding="utf-8")

    runtime.hmonnx_generate(
        meta_file=tmp_path / "meta.json",
        prompt=str(prompt),
        device="cpu",
        max_new_tokens=7,
        golden_components=("decode",),
        stream_output=False,
    )
    runtime.hmonnx_generate(
        meta_file=tmp_path / "meta.json",
        prompt=str(prompt),
        device="cpu",
        max_new_tokens=1,
        golden_components=("prefill",),
        stream_output=False,
    )

    assert calls["prompts"] == [str(prompt), str(prompt)]
    assert calls["max_new_tokens"] == [7, 1]


def test_hmonnx_generate_dumps_request_input_only_with_golden(monkeypatch, tmp_path: Path):
    runtime = _load_runtime_module(monkeypatch)
    _install_fake_generation_dependencies(monkeypatch)
    meta_file = tmp_path / "golden_meta_info.json"
    meta_file.write_text("{}", encoding="utf-8")

    runtime.hmonnx_generate(
        meta_file=meta_file,
        prompt="plain",
        device="cpu",
        max_new_tokens=2,
        stream_output=False,
    )
    assert not (tmp_path / "golden_input").exists()

    runtime.hmonnx_generate(
        meta_file=meta_file,
        prompt="golden",
        device="cpu",
        max_new_tokens=2,
        golden=True,
        stream_output=False,
    )
    assert (tmp_path / "golden_input/prompt.txt").read_text(encoding="utf-8") == "golden"
    assert not (tmp_path / "golden_input/decode").exists()


def test_hmonnx_generate_rejects_decode_golden_without_a_decode_step(monkeypatch, tmp_path: Path):
    runtime = _load_runtime_module(monkeypatch)

    with pytest.raises(ValueError, match="Decode Golden requires max_new_tokens >= 2"):
        runtime.hmonnx_generate(
            meta_file=tmp_path / "meta.json",
            max_new_tokens=1,
            golden_components=("decode",),
        )


def test_spec_decode_stats_are_summarized_with_accept_rate(monkeypatch, tmp_path: Path):
    runtime = _load_runtime_module(monkeypatch)

    result = runtime.build_spec_decode_result(
        meta_file=tmp_path / "golden_meta_info.json",
        output_text="hello",
        output_tokens=12,
        latency_s=3.0,
        spec_decode_mode="dflash",
        block_size=5,
        stats={
            "num_rounds": 4,
            "draft_tokens_total": 20,
            "accepted_drafts_total": 7,
            "avg_accepted_per_round": 1.75,
            "accepted_drafts_per_round": [2, 1, 3, 1],
        },
    )

    assert result.spec_decode_mode == "dflash"
    assert result.tokens_per_second == 4.0
    assert result.draft_tokens_total == 20
    assert result.accepted_drafts_total == 7
    assert result.accept_rate == 0.35
    assert result.avg_accepted_per_round == 1.75
    assert result.stats["accepted_drafts_per_round"] == [2, 1, 3, 1]


def test_spec_decode_generate_enables_draft_golden_after_warmup(monkeypatch, tmp_path: Path):
    runtime = _load_runtime_module(monkeypatch)
    events = []
    prompts = []

    class FakeRuntime:
        block_size = 4

        def set_spec_draft_golden(self, enable: bool, *, reset_step: bool = True) -> None:
            events.append(("golden", enable, reset_step))

    def fake_load_runtime(**kwargs):
        return FakeRuntime(), object(), {"spec_decode": {"mode": "mtp", "block_size": 4}}

    def fake_run_once(*args, **kwargs):
        prompts.append(args[2])
        events.append(("run", kwargs["max_new_tokens"]))
        return "ok", {"draft_tokens_total": 4, "accepted_drafts_total": 2}, 0.5, 2

    monkeypatch.setattr(runtime, "_load_merak_spec_runtime", fake_load_runtime)
    monkeypatch.setattr(runtime, "_run_spec_decode_once", fake_run_once)

    prompt = tmp_path / "worker-local-prompt.txt"
    prompt.write_text("filesystem contents must not replace the prompt", encoding="utf-8")
    result = runtime.spec_decode_generate(
        meta_file=tmp_path / "golden_meta_info.json",
        prompt=str(prompt),
        max_new_tokens=2,
        warmup_runs=1,
        benchmark_runs=1,
        golden=True,
    )

    assert events == [("run", 2), ("golden", True, True), ("run", 2)]
    assert prompts == [str(prompt), str(prompt)]
    assert result.spec_decode_mode == "mtp"
    assert result.draft_tokens_total == 4
    assert result.accepted_drafts_total == 2


def test_component_golden_only_enables_selected_sessions(monkeypatch):
    runtime = _load_runtime_module(monkeypatch)

    class Component:
        enable_golden = False

    model = types.SimpleNamespace(
        prefill_model=Component(),
        decode_model=Component(),
        visual=Component(),
    )

    runtime._set_component_golden(model, ("visual", "decode"))

    assert model.prefill_model.enable_golden is False
    assert model.decode_model.enable_golden is True
    assert model.visual.enable_golden is True


def test_merak_qwen35_hmonnx_validation_does_not_import_xh_model_zoo():
    src = (REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_5/hmonnx_validation.py").read_text(encoding="utf-8")

    assert "xh_model_zoo" not in src


def test_dflash_noise_token_comes_from_contract_v2(monkeypatch):
    runtime = _load_runtime_module(monkeypatch)

    assert (
        runtime._dflash_noise_token_from_meta(
            {
                "spec_decode": {
                    "mode": "dflash",
                    "runtime_contract_version": 2,
                    "draft": {"noise_token_id": 0},
                }
            },
            spec_mode="dflash",
        )
        == 0
    )


def test_dflash_noise_token_v2_rejects_missing_value(monkeypatch):
    import pytest

    runtime = _load_runtime_module(monkeypatch)

    with pytest.raises(
        ValueError,
        match=r"contract v2 requires spec_decode\.draft\.noise_token_id",
    ):
        runtime._dflash_noise_token_from_meta(
            {
                "spec_decode": {
                    "mode": "dflash",
                    "runtime_contract_version": 2,
                }
            },
            spec_mode="dflash",
        )


def test_dflash_noise_token_legacy_fallback_is_v1_only(monkeypatch):
    runtime = _load_runtime_module(monkeypatch)

    assert (
        runtime._dflash_noise_token_from_meta(
            {"spec_decode": {"mode": "dflash"}},
            spec_mode="dflash",
        )
        == 248070
    )
    assert (
        runtime._dflash_noise_token_from_meta(
            {"spec_decode": {"mode": "mtp", "runtime_contract_version": 2}},
            spec_mode="mtp",
        )
        is None
    )


def test_qwen35_spec_runtime_uses_golden_session_wrapper():
    src = (REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_5/qwen3_5_onnx_model.py").read_text(encoding="utf-8")

    assert "HMONNXGraphGoldenInference" in src
    assert "session = HMONNXGrapInference(onnx_path)" not in src
    assert "session.initialize()" in src
