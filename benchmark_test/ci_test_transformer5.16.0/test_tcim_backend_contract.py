from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ASR_DEMOS = (
    REPO_ROOT / "examples_merak/asr/qwen3_asr/hmonnx_demo.py",
    REPO_ROOT / "examples_merak/asr/qwen3_asr/hmonnx_demo_chunk_prefix.py",
)
ASR_GOLDEN_DEMO = ASR_DEMOS[0]
ASR_GOLDEN_UTILS = REPO_ROOT / "examples_merak/asr/qwen3_asr/golden_utils.py"


def _load_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("demo_path", ASR_DEMOS)
def test_qwen3_asr_demo_resolves_config_cache_and_embedding_aliases(tmp_path, demo_path):
    module = _load_source(demo_path, f"_tcim_contract_{demo_path.stem}")

    assert module._resolve_cache_len({}, 77) == 77
    assert module._resolve_cache_len({"kv_cache_shape": [1, 8, 2048, 128]}, None) == 2048
    assert (
        module._resolve_cache_len(
            {"kv_cache": {"shape": [[1, 8, 4096, 128], [1, 8, 4096, 128]], "cache_axis": 2}},
            None,
        )
        == 4096
    )
    with pytest.raises(ValueError, match="must provide kv_cache_shape"):
        module._resolve_cache_len({}, None)

    for relative in ("encode/model.hmm", "prefill/model.hmm", "decode/model.hmm"):
        artifact = tmp_path / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"hmm")
    (tmp_path / "hf_config").mkdir()
    (tmp_path / "quant_embedding.pt").write_bytes(b"embedding")
    metadata = {
        "encoder": {
            "hmonnx_file": "encode/model.hmm",
            "model_cfg": {"fixed_max_audio_length": 1500},
        },
        "prefill_onnx_file": "prefill/model.hmm",
        "decode_onnx_file": "decode/model.hmm",
        "hf_config": "hf_config",
        "quant_embedding": "quant_embedding.pt",
        "kv_cache_shape": [1, 8, 2048, 128],
    }
    config = tmp_path / "hmm_meta_info.json"
    config.write_text(json.dumps(metadata), encoding="utf-8")

    artifacts = module._load_artifacts(config)
    assert artifacts["embedding"] == tmp_path / "quant_embedding.pt"
    assert artifacts["encoder"] == tmp_path / "encode/model.hmm"
    assert module._resolve_embedding_file({"token_embedding_file": "new.pt", "quant_embedding": "old.pt"}) == "new.pt"
    with pytest.raises(ValueError, match="token_embedding_file or quant_embedding"):
        module._resolve_embedding_file({})

    args = module.parse_args(["--config", str(config), "--audio", "audio.wav"])
    assert args.config == str(config)
    assert args.work_dir is None
    assert args.cache_len is None
    if demo_path == ASR_GOLDEN_DEMO:
        assert args.golden is False
        assert module.parse_args(["--config", str(config), "--audio", "audio.wav", "--golden"]).golden is True
    else:
        assert not hasattr(args, "golden")


def test_qwen3_asr_golden_input_overwrites_request_snapshot(tmp_path):
    module = _load_source(ASR_GOLDEN_UTILS, "_qwen3_asr_golden_utils")
    audio = tmp_path / "request.wav"
    audio.write_bytes(b"first-audio")

    recorder = module.GoldenInputRecorder(tmp_path, audio, 16_000, 8_000)
    recorder.record_tokenizer(prompt="raw request", tokenizer_output="<special>request", token_ids=[[1, 2]])
    stale = recorder.output_dir / "stale.txt"
    stale.write_text("stale", encoding="utf-8")

    audio.write_bytes(b"second-audio")
    recorder = module.GoldenInputRecorder(tmp_path, audio, 16_000, 16_000)
    recorder.record_tokenizer(prompt="next", tokenizer_output="<special>next", token_ids=[[3, 4, 5]])

    assert not stale.exists()
    assert (recorder.output_dir / "input_audio.wav").read_bytes() == b"second-audio"
    assert (recorder.output_dir / "prompt.txt").read_text(encoding="utf-8") == "next"
    assert (recorder.output_dir / "tokenizer_output.txt").read_text(encoding="utf-8") == "<special>next"
    np.testing.assert_array_equal(np.load(recorder.output_dir / "token_ids.npy"), [[3, 4, 5]])
    meta = json.loads((recorder.output_dir / "input_meta.json").read_text(encoding="utf-8"))
    assert meta["sample_rate"] == 16_000
    assert meta["sample_count"] == 16_000
    assert meta["prefill_count"] == 1
    assert meta["token_snapshot"] == "last_prefill"


def test_qwen3_asr_golden_uses_component_root_and_rejects_unsupported_backend(tmp_path):
    module = _load_source(ASR_GOLDEN_UTILS, "_qwen3_asr_golden_session")

    class Session:
        def __init__(self, reject=False):
            self.reject = reject
            self.golden_dir = None
            self.step = 7
            self._save_golden = False
            self.initialized = False

        @property
        def save_golden(self):
            return self._save_golden

        @save_golden.setter
        def save_golden(self, value):
            if value and self.reject:
                raise NotImplementedError("save_golden is not supported")
            self._save_golden = value

        def reset_step(self):
            self.step = 0

        def initialize(self):
            self.initialized = True

    model = tmp_path / "prefill/hmonnx/model.onnx"
    stale_step = tmp_path / "prefill/step_9"
    stale_step.mkdir(parents=True)
    session = Session()
    module.enable_component_golden(session, model)
    assert session.golden_dir == str(tmp_path / "prefill")
    assert session.save_golden is True
    assert session.step == 0
    assert session.initialized is True
    assert not stale_step.exists()

    protected_step = tmp_path / "prefill/step_8"
    protected_step.mkdir()
    with pytest.raises(NotImplementedError, match="not supported"):
        module.enable_component_golden(Session(reject=True), model)
    assert protected_step.is_dir()


def test_backend_factory_is_lazy_and_defaults_to_hmonnx(monkeypatch):
    from xhmodel_merak.exec_backends import factory

    imported = []
    real_import_module = factory.import_module

    def tracked_import(name, package=None):
        imported.append((name, package))
        return real_import_module(name, package=package)

    monkeypatch.delenv(factory.ENV_INFERENCE_BACKEND, raising=False)
    monkeypatch.setattr(factory, "import_module", tracked_import)
    from xhquant.api import HMONNXInference as NativeHMONNXInference

    assert factory.get_inference_backend() == "hmonnx"
    assert factory.resolve_backend_class("HMONNXInference") is NativeHMONNXInference
    assert all("tcim_hmonnx" not in name for name, _ in imported)

    monkeypatch.setenv(factory.ENV_INFERENCE_BACKEND, "tcim")
    assert factory.resolve_backend_class("HMONNXInference").__name__ == "TcimHMONNXInference"

    monkeypatch.setenv(factory.ENV_INFERENCE_BACKEND, "unknown")
    with pytest.raises(ValueError, match="expected one of hmonnx, tcim"):
        factory.get_inference_backend()


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", "hmonnx"),
        ("  HMONNX ", "hmonnx"),
        ("tcim", "tcim"),
        (" TCIM_RUNTIME ", "tcim"),
        ("tcim_lite", "tcim"),
    ],
)
def test_backend_factory_normalizes_selection(monkeypatch, value, expected):
    from xhmodel_merak.exec_backends import factory

    monkeypatch.setenv(factory.ENV_INFERENCE_BACKEND, value)
    assert factory.get_inference_backend() == expected


def test_backend_factory_supports_new_mapping_without_dispatch_changes(monkeypatch, tmp_path):
    from xhmodel_merak.exec_backends import factory

    module_name = "_test_merak_backend_extension"
    (tmp_path / f"{module_name}.py").write_text(
        "class Session:\n"
        "    def __init__(self, value): self.value = value\n"
        "    @classmethod\n"
        "    def from_onnx_graph(cls, value): return cls(value)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(
        factory.BACKENDS,
        "extension",
        {
            "HMONNXInference": (module_name, "Session"),
            "HMONNXInferenceV2": (module_name, "Session"),
        },
    )
    monkeypatch.setenv(factory.ENV_INFERENCE_BACKEND, "extension")
    try:
        assert factory.get_inference_backend() == "extension"
        assert module_name not in sys.modules
        session = factory.HMONNXInference(value=17)
        assert session.value == 17
        assert isinstance(session, factory.HMONNXInference)
        assert issubclass(type(session), factory.HMONNXInference)
        assert factory.HMONNXInferenceV2.from_onnx_graph(23).value == 23
        with pytest.raises(ValueError, match="extension.*CacheTensor"):
            factory.resolve_backend_class("CacheTensor")
        monkeypatch.setenv(factory.ENV_INFERENCE_BACKEND, "unknown")
        with pytest.raises(ValueError, match="expected one of extension, hmonnx, tcim"):
            factory.get_inference_backend()
    finally:
        sys.modules.pop(module_name, None)


def test_backend_declarations_do_not_import_runtime_dependencies():
    import subprocess

    code = """
import sys
from xhmodel_merak.exec_backends import factory
from xhmodel_merak.exec_backends import hmonnx_backend, tcim_backend
assert factory.BACKENDS['hmonnx'] is hmonnx_backend.CLASSES
assert factory.BACKENDS['tcim'] is tcim_backend.CLASSES
assert set(hmonnx_backend.CLASSES) == set(tcim_backend.CLASSES)
for alias in tcim_backend.ALIASES:
    assert factory.BACKEND_ALIASES[alias] == tcim_backend.NAME
assert 'xhmodel_merak.exec_backends.tcim_hmonnx' not in sys.modules
assert not any(name.split('.')[0] in {'torch', 'xhquant', 'tcim_runtime', 'tcim_lite'}
               for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)


def test_common_hmonnx_model_uses_backend_facade():
    from xhmodel_merak.exec_backends import CacheTensor, HMONNXGoldenInference, HMONNXGrapInference
    from xhmodel_merak.xh_other_model import llm_onnx_model
    from xhmodel_merak.xh_other_model.common import hmonnx_model

    assert hmonnx_model.HMONNXInference is HMONNXGoldenInference
    assert llm_onnx_model.HMONNXInference is HMONNXGoldenInference
    assert llm_onnx_model.HMONNXGrapInference is HMONNXGrapInference
    assert llm_onnx_model.CacheTensor is CacheTensor


class _FakeDeviceTensor:
    def __init__(self):
        self.zeroed = False

    def set_zero(self):
        self.zeroed = True


class _FakeHostTensor:
    def __init__(self, value):
        self._value = value

    def numpy(self):
        return self._value


class _FakeModule:
    def __init__(self):
        self.input_infos = {
            "tokens": SimpleNamespace(shape=[1, 1], dtype=np.int32),
            "past_key_cache_0": SimpleNamespace(shape=[1, 2, 4, 8], dtype=np.int8),
        }
        self.output_infos = {
            "logits": SimpleNamespace(shape=[1, 1, 3], dtype=np.float16),
            "present_key_cache_0": SimpleNamespace(shape=[1, 2, 4, 8], dtype=np.int8),
        }
        self.host_inputs = {}
        self.device_inputs = {}
        self.preallocated_inputs = {}
        self.device_outputs = {}
        self.run_count = 0

    def get_num_inputs(self):
        return len(self.input_infos)

    def get_num_outputs(self):
        return len(self.output_infos)

    def get_input_name(self, index):
        return tuple(self.input_infos)[index]

    def get_output_name(self, index):
        return tuple(self.output_infos)[index]

    def get_input_info(self, name):
        return self.input_infos[name]

    def get_output_info(self, name):
        return self.output_infos[name]

    def set_input(self, name, value):
        self.host_inputs[name] = value

    def get_dev_input(self, name):
        return self.preallocated_inputs.setdefault(name, _FakeDeviceTensor())

    def set_dev_input(self, name, value):
        self.device_inputs[name] = value

    def run(self):
        self.run_count += 1

    def sync(self):
        return None

    def get_output(self, name):
        assert name == "logits"
        return _FakeHostTensor(np.ones((1, 1, 3), dtype=np.float16))

    def get_dev_output(self, name):
        return self.device_outputs.setdefault(name, _FakeDeviceTensor())


class _FakeRuntime:
    class Option:
        def __init__(self, weight_manager):
            self.weight_manager = weight_manager

    def __init__(self):
        self.modules = []

    def load(self, model_path, option):
        assert model_path.endswith(".hmm")
        assert option.weight_manager == "weights"
        module = _FakeModule()
        self.modules.append(module)
        return module


def test_tcim_session_keeps_cache_on_device_between_sessions(tmp_path, monkeypatch):
    from xhmodel_merak.exec_backends import tcim_hmonnx

    model = tmp_path / "decode.hmm"
    model.write_bytes(b"hmm")
    runtime = _FakeRuntime()
    context = {"runtime": runtime, "device_manager": "device", "weight_manager": "weights"}
    monkeypatch.setattr(tcim_hmonnx, "_get_shared_context", lambda device_id: context)

    cache = tcim_hmonnx.TcimCacheTensor(torch.zeros((1, 2, 4, 8), dtype=torch.int8))
    first = tcim_hmonnx.TcimHMONNXInference(str(model))
    logits, returned_cache = first.run({"tokens": torch.ones((1, 1), dtype=torch.int32), "past_key_cache_0": cache})
    assert logits.dtype == torch.float16
    assert returned_cache is cache
    assert "past_key_cache_0" not in runtime.modules[0].host_inputs
    assert runtime.modules[0].preallocated_inputs["past_key_cache_0"].zeroed

    previous_device_cache = cache._dev_tensor
    second = tcim_hmonnx.TcimHMONNXInference(str(model))
    second.run({"tokens": torch.ones((1, 1), dtype=torch.int32), "past_key_cache_0": cache})
    assert runtime.modules[1].device_inputs["past_key_cache_0"] is previous_device_cache
    assert "past_key_cache_0" not in runtime.modules[1].host_inputs


def test_tcim_import_error_is_deferred_until_runtime_load(monkeypatch):
    from xhmodel_merak.exec_backends import tcim_hmonnx

    monkeypatch.setitem(sys.modules, "tcim_lite", None)
    with pytest.raises(ImportError, match="tcim_lite is required"):
        tcim_hmonnx._load_tcim_runtime()


def test_link_hmm_generates_runtime_metadata_without_mutating_source(tmp_path, monkeypatch):
    from xhmodel_merak.exec_backends import link_hmm

    hmm_root = tmp_path / "hmm"
    export_root = tmp_path / "export"
    hmm_root.mkdir()
    export_root.mkdir()
    hmm_files = {}
    for role in ("prefill", "decode", "visual_m96"):
        path = hmm_root / f"{role}.hmm"
        path.write_bytes(role.encode())
        hmm_files[role] = path

    source = {
        "model_config": {"context_max_length": 2048},
        "kv_cache": {
            "kv_cache_shape": [1, 2, 2048, 8],
            "cache_axis": 2,
            "cache_dtype": "float16",
        },
        "max_context_tokens": 2048,
        "prefill_hmonnx_md5": "old",
        "prefill_hmonnx": "prefill/model.onnx",
        "decode_hmonnx": "decode/model.onnx",
        "visual_config": {"hmonnx": "visual/visual_m96.onnx"},
    }
    source_path = export_root / "golden_meta_info.json"
    source_path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    original_source = source_path.read_bytes()

    def fake_model_info(path):
        if path.name.startswith("visual"):
            return {"input": []}
        return {
            "input": [
                {"name": "layer_0_kcache_input", "shape": [1, 2, 16, 8], "dtype": "INT8"},
                {"name": "layer_0_vcache_input", "shape": [1, 2, 16, 8], "dtype": "INT8"},
            ]
        }

    monkeypatch.setattr(link_hmm, "_load_hmm_model_info", fake_model_info)
    output_path = link_hmm.generate_hmm_meta(hmm_root, export_root)
    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == export_root / "hmm_meta_info.json"
    assert source_path.read_bytes() == original_source
    assert output["kv_cache"]["kv_cache_shape"] == [1, 2, 16, 8]
    assert output["kv_cache"]["cache_dtype"] == "int8"
    assert output["model_config"]["context_max_length"] == 16
    assert output["max_context_tokens"] == 16
    assert output["prefill_hmonnx"] == "../hmm/prefill.hmm"
    assert output["decode_hmonnx"] == "../hmm/decode.hmm"
    assert output["visual_config"]["hmonnx"] == "../hmm/visual_m96.hmm"
    assert output["prefill_hmonnx_md5"] == hashlib.md5(b"prefill", usedforsecurity=False).hexdigest()


@pytest.fixture
def session_boundary(monkeypatch):
    """Replace SDK execution only; run real backend policies and model orchestration."""
    from types import ModuleType
    from xhmodel_merak.exec_backends import factory

    module = ModuleType("_merak_session_boundary")

    class Session:
        def __init__(self, path, config=None):
            self.path = path
            self.config = config
            self.device = torch.device("cpu")
            self.enable_auto_offload = False
            self.moves = []
            self.graph = None

        @classmethod
        def from_onnx_graph(cls, path, graph, config):
            session = cls(path, config)
            session.graph = graph
            return session

        def to(self, device):
            self.moves.append(device)
            self.device = device
            return self

        def get_layer_infos(self):
            return ["shared-layer"]

    module.Session = Session
    module.Config = SimpleNamespace
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for name in ("hmonnx", "tcim"):
        mapping = dict(factory.BACKENDS[name])
        for interface in ("HMONNXInferenceV2", "HMONNXGoldenInference", "HMONNXCUDAGraphInference"):
            mapping[interface] = (module.__name__, "Session")
        mapping["HMONNXInferenceConfig"] = (module.__name__, "Config")
        monkeypatch.setitem(factory.BACKENDS, name, mapping)
    return module


@pytest.mark.parametrize("backend_name", ["hmonnx", "tcim"])
@pytest.mark.parametrize("with_graph", [False, True])
def test_bound_backend_creates_v2_session_and_preserves_capabilities(
    monkeypatch, session_boundary, backend_name, with_graph
):
    from xhmodel_merak.exec_backends import get_backend
    from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel

    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", backend_name)
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    backend = get_backend()
    # No later environment read may change session choice or capability handling.
    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "invalid-after-selection")
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "0")
    graph = object() if with_graph else None
    model = HMONNXModel(
        "model.onnx",
        onnx_graph=graph,
        enable_cuda_graph=True,
        enable_auto_offload=True,
        device_map="cpu",
        backend=backend,
    )
    session = model.hmonnx_session
    assert session.graph is graph
    assert session.config.exec_devices == ["cpu"]
    assert model.enable_cuda_graph is (backend_name == "hmonnx")
    assert session.config.enable_auto_offload is (backend_name == "hmonnx")
    model.enable_auto_offload = True
    assert model._enable_auto_offload is (backend_name == "hmonnx")
    assert session.enable_auto_offload is (backend_name == "hmonnx")
    model._set_device(torch.device("cuda:1"))
    assert session.moves == []  # V2 placement is owned by the session.


@pytest.mark.parametrize("cuda_graph", [False, True])
def test_bound_backend_keeps_legacy_creation_and_device_rules(monkeypatch, session_boundary, cuda_graph):
    from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel

    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "hmonnx")
    monkeypatch.delenv("ENABLE_HMINFERENCE_V2", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(FutureWarning, match="Legacy HMONNX"):
        model = HMONNXModel("decode/model.onnx", enable_cuda_graph=cuda_graph)
    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "tcim")
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    assert model.hmonnx_session.moves == [torch.device("cpu")]
    if cuda_graph:
        with pytest.raises(RuntimeError, match="Cannot enable auto offload"):
            model.enable_auto_offload = True
    else:
        assert model.hmonnx_session.golden_dir == "decode"
        assert model.hmonnx_session.legacy_mode is False
        model.enable_auto_offload = True
        assert model._enable_auto_offload is True
        model._set_device(torch.device("cuda:1"))
        assert model.hmonnx_session.moves[-1] == torch.device("cuda:1")


@pytest.mark.parametrize("fusion,page", [(False, False), (True, False), (False, True), (True, True)])
def test_backend_graph_preparation_policy(monkeypatch, fusion, page):
    from xhmodel_merak.exec_backends import get_backend

    calls = []

    def prepare(*paths, **options):
        calls.append((paths, options))
        return ("prepared-prefill", "prepared-decode")

    options = dict(
        enable_parallel_linear_fusion=fusion, enable_page_attention=page, convert_unfused_sliding_kv_cache=True
    )
    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "hmonnx")
    assert get_backend().prepare_graphs("p", "d", prepare=prepare, **options) == ("prepared-prefill", "prepared-decode")
    assert calls == [(("p", "d"), options)]
    calls.clear()
    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "tcim")
    if fusion or page:
        with pytest.raises(NotImplementedError, match="compiled with the requested graph ABI"):
            get_backend().prepare_graphs("p", "d", prepare=prepare, **options)
    else:
        assert get_backend().prepare_graphs(Path("p"), Path("d"), prepare=prepare, **options) == ("p", "d")
    assert calls == []


def test_llm_pins_prefill_decode_loader_and_cache_backend(monkeypatch, tmp_path, session_boundary):
    from transformers import LlamaConfig

    from xhmodel_merak.exec_backends import factory
    from xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model import BaseLLMHMONNXModel
    from xhmodel_merak.xh_llm.types import LLMModelMeta

    class Cache:
        def __init__(self, value):
            self.value = value

    class Loader:
        def __init__(self, *paths):
            self.prefill_graph = object()
            self.decode_graph = object()
            self.initializer_pool = SimpleNamespace(summary=lambda: {"weights": 1})
            # Simulate environment mutation during model initialization.
            monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "invalid-after-load")

    session_boundary.Cache = Cache
    session_boundary.Loader = Loader
    monkeypatch.setitem(factory.BACKENDS["tcim"], "CacheTensor", (session_boundary.__name__, "Cache"))
    monkeypatch.setitem(factory.BACKENDS["tcim"], "LLMHMONNXLoader", (session_boundary.__name__, "Loader"))
    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "tcim")
    embedding = tmp_path / "embedding.pt"
    torch.save({"weight": torch.zeros(4, 2)}, embedding)
    LlamaConfig(
        vocab_size=4,
        hidden_size=2,
        intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
    ).save_pretrained(tmp_path)
    meta = LLMModelMeta(
        hf_config=str(tmp_path),
        quant_embedding=str(embedding),
        prefill_hmonnx="p.hmm",
        decode_hmonnx="d.hmm",
        kv_cache=dict(num_layers=1, kv_cache_shape=[1, 1, 2, 2]),
    )
    model = BaseLLMHMONNXModel(meta, device_map="cpu")
    assert model.prefill_model._backend is model.decode_model._backend is model._backend
    assert model.decode_model.hmonnx_session.config.layers == ["shared-layer"]
    model._kvcache_mixin.prepare_kv_cache()
    assert isinstance(model._kvcache_mixin.past_key_caches[0], Cache)
    assert model.shared_weight_summary == {"weights": 1}
