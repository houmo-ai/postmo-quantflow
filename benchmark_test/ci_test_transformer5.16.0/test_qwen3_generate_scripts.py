"""Contracts for the Qwen3 legacy and workflow generation examples."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCRIPT = REPO_ROOT / "examples_merak/llm/qwen3/debug_scripts/_qwen3_xh_generate.py"
WORKFLOW_SCRIPT = REPO_ROOT / "examples_merak/llm/qwen3/debug_scripts/qwen3_xh_generate_workflow.py"


class _TokenSequence(list):
    def __getitem__(self, item):
        value = super().__getitem__(item)
        return _TokenSequence(value) if isinstance(value, list) else value

    def tolist(self):
        return list(self)


class _ModelInputs(dict):
    def __init__(self, records):
        self.input_ids = [_TokenSequence([10, 11])]
        super().__init__(input_ids=self.input_ids)
        self._records = records

    def to(self, device):
        self._records.input_device = device
        return self


class _Tokenizer:
    eos_token_id = 7

    def __init__(self, records):
        self._records = records

    def encode(self, text):
        self._records.encoded.append(text)
        return [99]

    def decode(self, token_ids, *, skip_special_tokens):
        values = list(token_ids)
        self._records.decoded.append((values, skip_special_tokens))
        return ",".join(str(value) for value in values)


class _Processor:
    def __init__(self, records):
        self._records = records
        self.tokenizer = _Tokenizer(records)

    def apply_chat_template(self, messages, *, enable_think):
        self._records.chat_template = (messages, enable_think)
        return _ModelInputs(self._records)


class _RecordedContext:
    def __init__(self, records, name):
        self._records = records
        self._name = name

    def __enter__(self):
        self._records.entered_contexts.append(self._name)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._records.exited_contexts.append(self._name)
        return False


class _ContextManagers:
    def __init__(self, contexts):
        self._contexts = contexts
        self._stack = ExitStack()

    def __enter__(self):
        for context in self._contexts:
            self._stack.enter_context(context)
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._stack.__exit__(exc_type, exc, traceback)


class _Logger:
    def __init__(self, records):
        self._records = records

    def info(self, message):
        self._records.logs.append(str(message))


def _load_script(monkeypatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _install_import_stubs(monkeypatch):
    records = SimpleNamespace(
        encoded=[],
        decoded=[],
        entered_contexts=[],
        exited_contexts=[],
        logs=[],
        streamers=[],
    )

    class LLMModelState:
        @staticmethod
        def get_all_values():
            return ["wrap", "quanted_aligned"]

        @staticmethod
        def from_string(value):
            records.requested_state = value
            return f"state:{value}"

    class TextStreamer:
        def __init__(self, *, tokenizer):
            self.tokenizer = tokenizer
            records.streamers.append(self)

    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.no_grad = lambda: _RecordedContext(records, "no_grad")

    transformers = types.ModuleType("transformers")
    transformers.TextStreamer = TextStreamer

    xh_llm = types.ModuleType("xhmodel_merak.xh_llm")
    xh_llm.AutoLLMConfig = SimpleNamespace()
    xh_llm.AutoLLMModel = SimpleNamespace()
    xh_llm.LLMModelState = LLMModelState
    xh_llm.LLMInferenceContextManager = lambda model: _RecordedContext(records, "inference")

    workflows = types.ModuleType("xhmodel_merak.xh_llm.workflows")
    xhmodel_merak = types.ModuleType("xhmodel_merak")
    xhmodel_merak.__path__ = []
    xhmodel_merak.xh_llm = xh_llm
    xh_llm.__path__ = []
    xh_llm.workflows = workflows

    xhquant = types.ModuleType("xhquant")
    xhquant.__path__ = []
    xhquant_api = types.ModuleType("xhquant.api")
    xhquant_utils = types.ModuleType("xhquant.utils")
    records.current_context = SimpleNamespace(kvcache=SimpleNamespace(fast_mode=False))
    xhquant_context = types.ModuleType("xhquant.core.context")
    xhquant_context.get_current_context = lambda: records.current_context

    def xhquant_init(log_file, debug):
        records.xhquant_init = (log_file, debug)

    def set_random_seed(seed):
        records.seed = seed

    xhquant_api.Config = SimpleNamespace()
    xhquant_api.get_xhquant_logger = lambda: _Logger(records)
    xhquant_api.set_random_seed = set_random_seed
    xhquant_api.xhquant_init = xhquant_init
    xhquant_utils.ContextManagers = _ContextManagers
    xhquant_utils.TimeProfiler = lambda name, logger: _RecordedContext(records, f"profiler:{name}")
    xhquant_utils.MemoryTracker = lambda **kwargs: _RecordedContext(records, f"memory:{kwargs['device']}")

    modules = {
        "torch": torch,
        "transformers": transformers,
        "xhmodel_merak": xhmodel_merak,
        "xhmodel_merak.xh_llm": xh_llm,
        "xhmodel_merak.xh_llm.workflows": workflows,
        "xhquant": xhquant,
        "xhquant.api": xhquant_api,
        "xhquant.utils": xhquant_utils,
        "xhquant.core.context": xhquant_context,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return records, xh_llm, workflows, xhquant_api


def _make_model(records, *, generated_tokens):
    model_config_type = type("XHQwen3ModelConfig", (SimpleNamespace,), {})
    model_config = model_config_type(
        enable_auto_offload=False,
        enable_prefill_chunk=False,
        use_cache=False,
    )
    model_config.to_json_string = lambda: '{"model_type": "qwen3"}'

    class XHQwen3Model:
        def __init__(self):
            self.config = model_config
            self.processor = _Processor(records)

        def set_state(self, state):
            records.model_state = state

        def get_tf_processor(self):
            return self.processor

        def to(self, *, device, dtype):
            records.model_device = (device, dtype)
            return self

        def eval(self):
            records.eval_called = True
            return self

        def generate(self, **kwargs):
            records.generate_kwargs = kwargs
            return [_TokenSequence([10, 11, *generated_tokens])]

    return XHQwen3Model(), model_config


def _assert_generation_contract(records, model_config, *, prompt, think, expected_output_ids):
    assert records.seed == 1024
    assert records.model_state in {"state:wrap", "state:quanted_aligned"}
    assert records.chat_template == ([{"role": "user", "content": prompt}], think)
    assert records.input_device == "cpu"
    assert records.model_device == ("cpu", "float16")
    assert records.eval_called is True
    assert model_config.enable_auto_offload is True
    assert model_config.enable_prefill_chunk is True
    assert model_config.use_cache is True
    assert records.generate_kwargs["input_ids"] == [[10, 11]]
    assert records.generate_kwargs["max_new_tokens"] == 1024
    assert records.generate_kwargs["do_sample"] is False
    assert records.generate_kwargs["pad_token_id"] == 7
    assert records.generate_kwargs["streamer"] is records.streamers[0]
    assert records.decoded[-1] == (expected_output_ids, True)
    assert records.entered_contexts == ["profiler:generate", "memory:cpu", "inference", "no_grad"]
    assert records.exited_contexts == ["no_grad", "inference", "memory:cpu", "profiler:generate"]


@pytest.mark.transformer516
@pytest.mark.parametrize("override", [None, ""])
def test_legacy_generate_keeps_config_when_hf_override_is_empty(monkeypatch, override):
    _install_import_stubs(monkeypatch)
    module = _load_script(monkeypatch, "_ci_qwen3_legacy_generate_empty", LEGACY_SCRIPT)
    cfg = SimpleNamespace(hf_model_dir="original", model=SimpleNamespace(hf_model="original"))

    module._override_hf_model_dir(cfg, override)

    assert cfg.hf_model_dir == "original"
    assert cfg.model.hf_model == "original"


@pytest.mark.transformer516
def test_legacy_generate_main_applies_override_and_runs_generation(monkeypatch, tmp_path):
    records, _, _, _ = _install_import_stubs(monkeypatch)
    module = _load_script(monkeypatch, "_ci_qwen3_legacy_generate_main", LEGACY_SCRIPT)
    model, model_config = _make_model(records, generated_tokens=[30, 31])

    cfg = SimpleNamespace(
        hf_model_dir="original",
        model=SimpleNamespace(hf_model="original"),
        pretty_text="legacy config",
    )
    cfg.dump = lambda path: setattr(records, "dump_path", Path(path))
    module.Config = SimpleNamespace(fromfile=lambda path: setattr(records, "config_path", path) or cfg)
    module.AutoLLMConfig = SimpleNamespace(
        from_pretrained=lambda config: setattr(records, "auto_config_input", config) or model_config
    )
    module.AutoLLMModel = SimpleNamespace(
        from_pretrained=lambda *, config: setattr(records, "auto_model_config", config) or model
    )

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("legacy prompt", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        config="configs/qwen3.py",
        config_hf_model_dir="/models/qwen3",
        eval_type="wrap",
        prompt=str(prompt_path),
        think=True,
        debug=True,
        auto_offload=True,
        enable_prefill_chunk=True,
    )

    module.main(args)

    assert cfg.hf_model_dir == "/models/qwen3"
    assert cfg.model.hf_model == "/models/qwen3"
    assert cfg.seed == 1024
    assert records.config_path == "configs/qwen3.py"
    assert records.auto_config_input is cfg.model
    assert records.auto_model_config is model_config
    assert records.dump_path == Path("work_dirs/qwen3_debug/qwen3.py")
    assert records.xhquant_init == ("work_dirs/qwen3_debug/generate_wrap.log", True)
    _assert_generation_contract(records, model_config, prompt="legacy prompt", think=True, expected_output_ids=[30, 31])
    assert not any("thinking content" in message for message in records.logs)


@pytest.mark.transformer516
def test_workflow_generate_cli_contract(monkeypatch):
    records, _, _, _ = _install_import_stubs(monkeypatch)
    module = _load_script(monkeypatch, "_ci_qwen3_workflow_generate_cli", WORKFLOW_SCRIPT)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            WORKFLOW_SCRIPT.name,
            "--model-dir",
            "/models/qwen3",
            "--config-path",
            "workflow.yaml",
            "--quant-output-dir",
            "quant-output",
            "--export-output-dir",
            "export-output",
            "--device",
            "cpu",
            "--overwrite",
            "--dump-golden",
            "--export-from-quanted-model",
            "--bits",
            "4",
            "--eval-type",
            "wrap",
            "--prompt",
            "hello",
            "--think",
            "--debug",
            "--auto-offload",
            "--enable-prefill-chunk",
        ],
    )

    args = module.parse_args()

    assert vars(args) == {
        "model_dir": "/models/qwen3",
        "config_path": "workflow.yaml",
        "quant_output_dir": "quant-output",
        "export_output_dir": "export-output",
        "device": "cpu",
        "overwrite": True,
        "dump_golden": True,
        "export_from_quanted_model": True,
        "bits": 4,
        "eval_type": "wrap",
        "prompt": "hello",
        "think": True,
        "debug": True,
        "auto_offload": True,
        "enable_prefill_chunk": True,
    }
    assert records.logs == []


@pytest.mark.transformer516
def test_workflow_output_cleanup_requires_overwrite(monkeypatch, tmp_path):
    _install_import_stubs(monkeypatch)
    module = _load_script(monkeypatch, "_ci_qwen3_workflow_generate_cleanup", WORKFLOW_SCRIPT)
    output_dir = tmp_path / "export"
    output_dir.mkdir()
    artifact = output_dir / "model.hmonnx"
    artifact.write_text("artifact", encoding="utf-8")

    module._remove_output_dir_if_needed(str(output_dir), force=False)
    assert artifact.is_file()

    module._remove_output_dir_if_needed(str(output_dir), force=True)
    assert not output_dir.exists()
    module._remove_output_dir_if_needed(str(output_dir), force=True)


@pytest.mark.transformer516
def test_workflow_generate_main_builds_workflow_and_runs_generation(monkeypatch, tmp_path):
    records, _, workflows, _ = _install_import_stubs(monkeypatch)
    module = _load_script(monkeypatch, "_ci_qwen3_workflow_generate_main", WORKFLOW_SCRIPT)
    model, model_config = _make_model(records, generated_tokens=[20, 99, 30, 31])

    workflow = SimpleNamespace(
        workflow_config=SimpleNamespace(export={"model": {"chip_arch": "yuehui"}}),
    )

    def get_model(*, work_dir):
        records.workflow_work_dir = work_dir
        return model

    workflow.get_model = get_model

    class AutoLLMWorkflow:
        @staticmethod
        def from_config(**kwargs):
            records.workflow_kwargs = kwargs
            return workflow

    workflows.AutoLLMWorkflow = AutoLLMWorkflow

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("workflow prompt", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            model_dir="/models/qwen3",
            config_path="configs/qwen3_yuehui.yaml",
            quant_output_dir="quant-output",
            export_output_dir="export-output",
            device="cuda",
            overwrite=False,
            dump_golden=False,
            export_from_quanted_model=False,
            bits=None,
            eval_type="quanted_aligned",
            prompt=str(prompt_path),
            think=True,
            debug=True,
            auto_offload=True,
            enable_prefill_chunk=True,
        ),
    )

    module.main()

    assert records.workflow_kwargs == {
        "model_dir": "/models/qwen3",
        "config_path": "configs/qwen3_yuehui.yaml",
        "seed": 1024,
        "debug": True,
    }
    assert records.workflow_work_dir == "work_dirs/qwen3_yuehui_debug"
    assert records.xhquant_init == ("work_dirs/qwen3_yuehui_debug/generate.log", True)
    _assert_generation_contract(
        records,
        model_config,
        prompt="workflow prompt",
        think=True,
        expected_output_ids=[30, 31],
    )
    assert records.decoded[-2] == ([20, 99], True)
    assert "Config chip_arch: yuehui" in records.logs
    assert "20,99" in records.logs
    assert "30,31" in records.logs
