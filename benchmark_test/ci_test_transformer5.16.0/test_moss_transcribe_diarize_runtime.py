from pathlib import Path
import sys
import types

import torch
import numpy as np
import pytest

from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.hmonnx_runtime import MossHMONNXRuntime
from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.hmonnx_runtime import _rewrapping_kvcache_slices
from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.hmonnx_runtime import _runtime_tool_path
from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.workflow import _golden_inputs


def test_runtime_uses_xh2a_input_names_and_last_hidden_state(tmp_path: Path, monkeypatch):
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.hmonnx_runtime import (
        MossHMONNXRuntime,
        _layer_cache_names,
    )

    names = (
        "inputs_embeds",
        "past_seq_length",
        "current_input_length",
        "model_layers_0_self_attn_kcache_input",
        "model_layers_0_self_attn_vcache_input",
    )
    assert _layer_cache_names(names) == (
        "model_layers_0_self_attn_kcache_input",
        "model_layers_0_self_attn_vcache_input",
    )

    component = tmp_path / "prefill.onnx"
    component.write_bytes(b"fixture")

    class Info:
        dtype = torch.float16
        shape = (1, 4096, 1024)

    class Session:
        def __init__(self, path, config=None):
            del path, config

        def to(self, device):
            del device
            return self

        def get_input_names(self):
            return ["inputs_embeds", "past_seq_length", "current_input_length"]

        def get_input(self, name):
            assert name in {"inputs_embeds", "past_seq_length", "current_input_length"}
            return Info()

        def get_output_names(self):
            return ["last_hidden_state"]

        def run(self, feed):
            assert set(feed) == {"inputs_embeds", "past_seq_length", "current_input_length"}
            return (torch.zeros(1, 1, 1024),)

    Session.__module__ = "moss_test_hmonnx"
    api = types.ModuleType("xhquant.api")
    api.HMONNXInference = Session
    monkeypatch.setitem(sys.modules, "xhquant.api", api)
    runtime = MossHMONNXRuntime(
        tmp_path,
        {"components": {"prefill_decode": {"prefill": {"hmonnx_file": "prefill.onnx"}}}},
        "cpu",
    )
    assert runtime.input_shape("prefill", "inputs_embeds") == (1, 4096, 1024)
    result = runtime.run(
        "prefill",
        {
            "inputs_embeds": torch.zeros(1, 4096, 1024),
            "past_seq_length": torch.zeros(1),
            "current_input_length": torch.zeros(1),
        },
    )
    assert list(result) == ["last_hidden_state"]


def test_runtime_maps_export_input_1_aliases():
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.hmonnx_runtime import _map_feed

    mapped = _map_feed(
        ("input_1", "valid_length", "current_length"),
        {
            "inputs_embeds": "embeds",
            "past_seq_length": "past",
            "current_input_length": "cur",
        },
    )
    assert mapped == {"input_1": "embeds", "valid_length": "past", "current_length": "cur"}


def test_published_runtime_graphs_accept_scatter_hmonnx():
    from xhmodel_merak.model_integrations.moss_transcribe_diarize.artifact import published_runtime_graphs

    meta = {
        "weights_file": "weights.pt",
        "components": {
            "encode": {
                "hmonnx_file": "encode/hmquant_moss_transcribe_diarize_09b_encode_with_act.onnx",
            },
            "prefill_decode": {
                "prefill": {"hmonnx_file": "prefill/model.onnx", "cache_mode": "scatter"},
                "decode": {"hmonnx_file": "decode/model.onnx", "cache_mode": "scatter"},
            },
        },
    }
    assert published_runtime_graphs(meta) == (
        "encode/hmquant_moss_transcribe_diarize_09b_encode_with_act.onnx",
        "prefill/model.onnx",
        "decode/model.onnx",
        "weights.pt",
    )


def _fa_cache_meta(*, fa_enable: bool, cache_mode: str, hmonnx_file: str = "prefill.onnx") -> dict:
    geometry = {
        "hmonnx_file": hmonnx_file,
        "cache_mode": cache_mode,
        "cache_length": 2048,
        "enable_page_attention": False,
    }
    return {
        "model_config": {
            "flash_attention": {"enable": fa_enable},
            "enable_page_attention": fa_enable,
        },
        "components": {
            "encode": {"hmonnx_file": "encode.onnx"},
            "prefill_decode": {"prefill": dict(geometry), "decode": dict(geometry, hmonnx_file="decode.onnx")},
        },
    }


def test_fa_off_export_metadata_declares_inplace_cache(tmp_path: Path):
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.configuration import XHMossTranscribeDiarizeConfig
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.model import XHMossTranscribeDiarizeModel

    model = XHMossTranscribeDiarizeModel(
        XHMossTranscribeDiarizeConfig(
            model_name="moss_transcribe_diarize_09b",
            flash_attention={"enable": False},
            encode_config={"quant_type": "w8a8_sefp", "max_audio_length": 3000},
        )
    )
    meta = model._stitch_asr_meta(
        {"hmonnx_file": "hmquant_moss_transcribe_diarize_09b_encode_with_act.onnx"},
        types.SimpleNamespace(prefill_hmonnx="prefill/x.onnx", decode_hmonnx="decode/x.onnx"),
        tmp_path,
        512,
        2048,
        4096,
    )
    assert meta["components"]["prefill_decode"]["prefill"]["cache_mode"] == "inplace"
    assert meta["components"]["prefill_decode"]["decode"]["cache_mode"] == "inplace"
    assert meta["components"]["encode"]["hmonnx_file"] == (
        "encode/hmquant_moss_transcribe_diarize_09b_encode_with_act.onnx"
    )
    assert "onnx_file" not in meta["components"]["encode"]
    assert "onnx_file" not in meta["encode_config"]
    assert meta["encode_config"]["quant_type"] == "w8a8_sefp"
    assert meta["model_config"]["enable_page_attention"] is False
    assert meta["model_config"]["flash_attention"]["enable"] is False


def test_fa_off_inplace_runtime_does_not_auto_convert_page_attention(tmp_path: Path, monkeypatch):
    for name in ("encode.onnx", "prefill.onnx", "decode.onnx"):
        (tmp_path / name).write_bytes(b"fixture")
    converted = []
    monkeypatch.setattr(
        MossHMONNXRuntime,
        "_materialize_page_attention_graphs",
        lambda self: converted.append("converted"),
    )
    runtime = MossHMONNXRuntime(tmp_path, _fa_cache_meta(fa_enable=False, cache_mode="inplace"), "cuda:0")
    assert converted == []
    assert runtime.uses_inplace_cache() is True
    assert runtime.uses_page_attention() is False


def test_fa_on_inplace_runtime_auto_converts_page_attention(tmp_path: Path, monkeypatch):
    for name in ("encode.onnx", "prefill.onnx", "decode.onnx"):
        (tmp_path / name).write_bytes(b"fixture")
    converted = []
    monkeypatch.setattr(
        MossHMONNXRuntime,
        "_materialize_page_attention_graphs",
        lambda self: converted.append("converted"),
    )
    runtime = MossHMONNXRuntime(tmp_path, _fa_cache_meta(fa_enable=True, cache_mode="inplace"), "cuda:0")
    assert converted == ["converted"]


def test_fa_on_converts_beside_published_graphs_and_reuses(tmp_path: Path, monkeypatch):
    import shutil as shutil_mod

    copies = []
    original_copytree = shutil_mod.copytree

    def tracking_copytree(*args, **kwargs):
        copies.append(args[0] if args else kwargs.get("src"))
        return original_copytree(*args, **kwargs)

    monkeypatch.setattr(shutil_mod, "copytree", tracking_copytree)
    for name in ("encode.onnx", "prefill.onnx", "decode.onnx"):
        (tmp_path / name).write_bytes(b"fixture")
    conversions = []

    def fake_convert(path, **kwargs):
        source = Path(path)
        conversions.append(source.name)
        output = source.with_name(f"{source.stem}_page_attention_cached{source.suffix}")
        output.write_bytes(b"converted")
        return str(output)

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.moss_transcribe_diarize.hmonnx_runtime.convert_published_graph_to_page_attention",
        fake_convert,
    )
    meta = _fa_cache_meta(fa_enable=True, cache_mode="inplace")
    for phase in ("prefill", "decode"):
        meta["components"]["prefill_decode"][phase].update(num_layers=1, n_kv_head=2, head_dim=4)

    first = MossHMONNXRuntime(tmp_path, meta, "cuda:0")
    assert first.uses_page_attention() is True
    assert meta["components"]["prefill_decode"]["prefill"]["enable_page_attention"] is False
    assert conversions == ["prefill.onnx", "decode.onnx"]
    assert copies == []
    assert (tmp_path / "prefill_page_attention_cached.onnx").is_file()

    conversions.clear()
    second = MossHMONNXRuntime(tmp_path, meta, "cuda:0")
    assert second.uses_page_attention() is True
    assert conversions == []
    assert second._graph_overrides["prefill"].name == "prefill_page_attention_cached.onnx"


def test_stitch_requires_published_hmonnx_graphs(tmp_path: Path):
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.configuration import XHMossTranscribeDiarizeConfig
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.model import XHMossTranscribeDiarizeModel

    model = XHMossTranscribeDiarizeModel(
        XHMossTranscribeDiarizeConfig(model_name="moss_transcribe_diarize_09b")
    )
    with pytest.raises(RuntimeError, match="published HMONNX"):
        model._stitch_asr_meta(
            {"onnx_file": "encoder_window.onnx"},
            types.SimpleNamespace(prefill_hmonnx="prefill/x.onnx", decode_hmonnx="decode/x.onnx"),
            tmp_path,
            512,
            2048,
            4096,
        )


def test_copy_official_runtime_files_lives_in_the_model_package(tmp_path: Path):
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.packaging import copy_official_runtime_files

    source = tmp_path / "src"
    destination = tmp_path / "dest"
    source.mkdir()
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    (source / "config.json").write_text("{}", encoding="utf-8")
    copy_official_runtime_files(source, destination)
    assert (destination / "tokenizer.json").is_file()
    assert (destination / "hf_config" / "config.json").is_file()


def test_workflow_export_does_not_import_model_integrations():
    import inspect
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.workflow import MossTranscribeDiarizeWorkflow

    source = inspect.getsource(MossTranscribeDiarizeWorkflow.export)
    assert "model_integrations" not in source
    assert "copy_official_runtime_files" in source


def test_hmonnx_runtime_reports_missing_published_components(tmp_path: Path):
    runtime = MossHMONNXRuntime(tmp_path, {"components": {}}, "cpu")

    assert runtime.available() is False


def test_rewrapping_kvcache_slices_wraps_plain_tensor():
    from xhquant.core import CacheTensor
    from xhquant.xhonnxruntime.parsers.llm_cache import LLMCache

    module = LLMCache(axis=2)
    seen = {}

    def original(states, past_seq_length, current_seq_length, past_cache, accepted_count=None):
        seen["is_cache"] = isinstance(past_cache, CacheTensor)
        return {"out": past_cache}

    module._orig_forward = original
    session = types.SimpleNamespace(interpreter=types.SimpleNamespace(_graph_module=torch.nn.Sequential(module)))
    _rewrapping_kvcache_slices(session)
    module._orig_forward(None, None, None, torch.zeros(1, 1, 2, 2), None)
    assert seen["is_cache"] is True


def test_hmonnx_runtime_is_not_available_on_cpu_even_with_published_files(tmp_path: Path):
    for name in ("encode.onnx", "prefill.onnx", "decode.onnx"):
        (tmp_path / name).write_bytes(b"fixture")
    runtime = MossHMONNXRuntime(
        tmp_path,
        {
            "components": {
                "encode": {"hmonnx_file": "encode.onnx"},
                "prefill_decode": {
                    "prefill": {"hmonnx_file": "prefill.onnx"},
                    "decode": {"hmonnx_file": "decode.onnx"},
                },
            }
        },
        "cpu",
    )

    assert runtime.available() is False


def test_hmonnx_runtime_enables_auto_offload_for_multiple_devices(tmp_path: Path, monkeypatch):
    component = tmp_path / "encode.onnx"
    component.write_bytes(b"fixture")
    seen = {}

    class Config:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.__dict__.update(kwargs)
            self.layers = kwargs.get("layers")

    class Session:
        def __init__(self, path, config):
            seen["path"] = path
            seen["config"] = config

        def to(self, device):
            seen["to"] = device

    module = types.ModuleType("xhquant.xhonnxruntime.hmonnx_inference_v2")
    module.HMONNXInferenceConfig = Config
    module.HMONNXInferenceV2 = Session
    monkeypatch.setitem(sys.modules, "xhquant.xhonnxruntime.hmonnx_inference_v2", module)
    monkeypatch.setitem(sys.modules, "xhquant.api", types.ModuleType("xhquant.api"))
    runtime = MossHMONNXRuntime(
        tmp_path,
        {"components": {"encode": {"hmonnx_file": "encode.onnx"}}},
        "cuda:2, cuda:3",
    )

    runtime._session("encode")

    assert runtime.device == "cuda:2"
    assert runtime.exec_devices == ("cuda:2", "cuda:3")
    assert seen["enable_auto_offload"] is True
    assert seen["enable_fast"] is False
    assert seen.get("enable_cuda_graph") is False
    assert seen["exec_devices"] == ["cuda:2", "cuda:3"]
    assert "to" not in seen


def _decoder_meta(tmp_path: Path) -> dict:
    (tmp_path / "prefill.onnx").write_bytes(b"fixture")
    (tmp_path / "decode.onnx").write_bytes(b"fixture")
    return {
        "components": {
            "prefill_decode": {
                "prefill": {"hmonnx_file": "prefill.onnx"},
                "decode": {"hmonnx_file": "decode.onnx"},
            }
        }
    }


def test_hmonnx_runtime_shares_prefill_decode_loader(tmp_path: Path, monkeypatch):
    seen: dict[str, list] = {"loader": [], "graphs": [], "constructed": []}

    class Loader:
        def __init__(self, prefill, decode):
            seen["loader"].append((Path(prefill).name, Path(decode).name))
            self.prefill_graph = "prefill-graph"
            self.decode_graph = "decode-graph"

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.layers = kwargs.get("layers")

    class Session:
        def __init__(self, path, config):
            seen["constructed"].append(Path(path).name)

        @classmethod
        def from_onnx_graph(cls, path, graph, config):
            session = cls.__new__(cls)
            session.path = path
            session.graph = graph
            session.config = config
            seen["graphs"].append(
                (
                    Path(path).name,
                    graph,
                    getattr(config, "layers", None),
                    bool(config.enable_fast),
                    bool(getattr(config, "enable_cuda_graph", False)),
                )
            )
            return session

        def to(self, device):
            return self

        def get_layer_infos(self):
            return ["prefill-layers"]

    v2 = types.ModuleType("xhquant.xhonnxruntime.hmonnx_inference_v2")
    v2.HMONNXInferenceConfig = Config
    v2.HMONNXInferenceV2 = Session
    loader_mod = types.ModuleType("xhquant.xhonnxruntime.llm_hmonnx_loader")
    loader_mod.LLMHMONNXLoader = Loader
    monkeypatch.setitem(sys.modules, "xhquant.xhonnxruntime.hmonnx_inference_v2", v2)
    monkeypatch.setitem(sys.modules, "xhquant.xhonnxruntime.llm_hmonnx_loader", loader_mod)
    monkeypatch.setitem(sys.modules, "xhquant.api", types.ModuleType("xhquant.api"))

    runtime = MossHMONNXRuntime(tmp_path, _decoder_meta(tmp_path), "cuda:0")
    prefill = runtime._session("prefill")
    decode = runtime._session("decode")

    assert seen["loader"] == [("prefill.onnx", "decode.onnx")]
    assert seen["constructed"] == []
    assert seen["graphs"] == [
        ("prefill.onnx", "prefill-graph", None, False, False),
        ("decode.onnx", "decode-graph", ["prefill-layers"], False, True),
    ]
    assert prefill is runtime._session("prefill")
    assert decode is runtime._session("decode")


def _install_decoder_v2(monkeypatch, *, loader_cls, session_cls):
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.layers = kwargs.get("layers")

    v2 = types.ModuleType("xhquant.xhonnxruntime.hmonnx_inference_v2")
    v2.HMONNXInferenceConfig = Config
    v2.HMONNXInferenceV2 = session_cls
    loader_mod = types.ModuleType("xhquant.xhonnxruntime.llm_hmonnx_loader")
    loader_mod.LLMHMONNXLoader = loader_cls
    monkeypatch.setitem(sys.modules, "xhquant.xhonnxruntime.hmonnx_inference_v2", v2)
    monkeypatch.setitem(sys.modules, "xhquant.xhonnxruntime.llm_hmonnx_loader", loader_mod)
    monkeypatch.setitem(sys.modules, "xhquant.api", types.ModuleType("xhquant.api"))


def test_shared_loader_does_not_leave_digest_cache_in_artifact(tmp_path: Path, monkeypatch):
    cache = tmp_path / ".xhquant_llm_hmonnx_sha256_prefill-decode.json"
    nested = tmp_path / "prefill" / ".xhquant_llm_hmonnx_sha256_nested.json"

    class Loader:
        def __init__(self, prefill, decode):
            del prefill, decode
            cache.write_text("{}", encoding="utf-8")
            nested.parent.mkdir(parents=True, exist_ok=True)
            nested.write_text("{}", encoding="utf-8")
            self.prefill_graph = "prefill-graph"
            self.decode_graph = "decode-graph"

    class Session:
        def __init__(self, path, config):
            del path, config

        @classmethod
        def from_onnx_graph(cls, path, graph, config):
            del path, graph, config
            return cls.__new__(cls)

        def get_layer_infos(self):
            return ["prefill-layers"]

    _install_decoder_v2(monkeypatch, loader_cls=Loader, session_cls=Session)
    runtime = MossHMONNXRuntime(tmp_path, _decoder_meta(tmp_path), "cuda:0")

    runtime._session("prefill")

    assert cache.is_file() is False
    assert nested.is_file() is False


def test_shared_loader_failure_falls_back_and_logs(tmp_path: Path, monkeypatch, caplog):
    import logging

    cache = tmp_path / ".xhquant_llm_hmonnx_sha256_prefill-decode.json"
    constructed = []

    class Loader:
        def __init__(self, prefill, decode):
            del prefill, decode
            cache.write_text("{}", encoding="utf-8")
            self.prefill_graph = "prefill-graph"
            self.decode_graph = "decode-graph"

    class Session:
        def __init__(self, path, config):
            del config
            constructed.append(Path(path).name)

        @classmethod
        def from_onnx_graph(cls, path, graph, config):
            del path, graph, config
            raise RuntimeError("shared graph parse failed")

        def get_layer_infos(self):
            return ["prefill-layers"]

    _install_decoder_v2(monkeypatch, loader_cls=Loader, session_cls=Session)
    runtime = MossHMONNXRuntime(tmp_path, _decoder_meta(tmp_path), "cuda:0")
    caplog.set_level(logging.ERROR)

    session = runtime._session("prefill")

    assert isinstance(session, Session)
    assert constructed == ["prefill.onnx"]
    assert cache.is_file() is False
    assert "shared prefill/decode load failed" in caplog.text
    assert "shared graph parse failed" in caplog.text


def test_decode_cuda_graph_only_on_single_cuda_session(tmp_path: Path, monkeypatch):
    configs = []

    class Loader:
        def __init__(self, prefill, decode):
            del prefill, decode
            self.prefill_graph = "prefill-graph"
            self.decode_graph = "decode-graph"

    class Session:
        def __init__(self, path, config):
            configs.append((Path(path).name, dict(config.__dict__)))

        @classmethod
        def from_onnx_graph(cls, path, graph, config):
            del graph
            session = cls.__new__(cls)
            session.config = config
            configs.append((Path(path).name, dict(config.__dict__)))
            return session

        def get_layer_infos(self):
            return ["prefill-layers"]

    _install_decoder_v2(monkeypatch, loader_cls=Loader, session_cls=Session)
    runtime = MossHMONNXRuntime(tmp_path, _decoder_meta(tmp_path), "cuda:0")
    runtime._session("prefill")

    by_name = {name: config for name, config in configs}
    assert by_name["prefill.onnx"]["enable_cuda_graph"] is False
    assert by_name["decode.onnx"]["enable_cuda_graph"] is True
    assert by_name["prefill.onnx"]["enable_fast"] is False
    assert by_name["decode.onnx"]["enable_fast"] is False


def test_decode_cuda_graph_disabled_for_multi_device_and_golden(tmp_path: Path, monkeypatch):
    configs = []

    class Loader:
        def __init__(self, prefill, decode):
            del prefill, decode
            self.prefill_graph = "prefill-graph"
            self.decode_graph = "decode-graph"

    class Session:
        def __init__(self, path, config):
            configs.append((Path(path).name, dict(config.__dict__)))

        @classmethod
        def from_onnx_graph(cls, path, graph, config):
            del graph
            session = cls.__new__(cls)
            session.config = config
            session.save_golden = False
            session.save_golden_dir = None
            configs.append((Path(path).name, dict(config.__dict__)))
            return session

        def get_layer_infos(self):
            return ["prefill-layers"]

    _install_decoder_v2(monkeypatch, loader_cls=Loader, session_cls=Session)
    multi = MossHMONNXRuntime(tmp_path, _decoder_meta(tmp_path), "cuda:2,cuda:3")
    multi._session("decode")
    multi_decode = [config for name, config in configs if name == "decode.onnx"][-1]
    assert multi_decode["enable_cuda_graph"] is False
    assert multi_decode["enable_auto_offload"] is True

    configs.clear()
    golden = MossHMONNXRuntime(tmp_path, _decoder_meta(tmp_path), "cuda:0")
    golden.enable_golden(("decode",))
    golden_decode = [config for name, config in configs if name == "decode.onnx"][-1]
    assert golden_decode["enable_cuda_graph"] is False
    assert golden._session("decode").save_golden is True


def testgenerate_hmonnx_gathers_decode_embeddings_on_device(tmp_path: Path):
    from xhmodel_merak.errors import ModelError
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.transcribe import generate_hmonnx
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.prefill_decode import MossLMHead

    (tmp_path / "generation_config.json").write_text('{"max_new_tokens": 2}', encoding="utf-8")
    head = MossLMHead(hidden_size=4, vocab_size=8).eval()
    seen = []

    class Runtime:
        device = "cpu"

        def input_shape(self, phase, name):
            del phase
            return (1, 4, 4) if name == "inputs_embeds" else (1, 1, 8, 2)

        def decoder_cache_length(self):
            return 8

        def prefill(self, *args):
            del args
            return {"last_hidden_state": torch.zeros(1, 4, 4), "present_key_cache": torch.zeros(1, 1, 8, 2),
                    "present_value_cache": torch.zeros(1, 1, 8, 2)}

        def decode(self, embed, *args):
            del args
            seen.append((tuple(embed.shape), embed.device.type, embed.dtype))
            return {"last_hidden_state": torch.zeros(1, 1, 4), "present_key_cache": torch.zeros(1, 1, 8, 2),
                    "present_value_cache": torch.zeros(1, 1, 8, 2)}

    tokenizer = types.SimpleNamespace(eos_token_id=-1, decode=lambda ids, **kw: "partial")
    with pytest.raises(ModelError, match="before EOS"):
        generate_hmonnx(np.zeros((3, 4), dtype=np.float32), np.ones(3), Runtime(), head, tokenizer, tmp_path)

    assert seen == [((1, 1, 4), "cpu", head.projection.weight.dtype)] * 2


def test_hmonnx_runtime_rejects_component_path_outside_artifact(tmp_path: Path):
    outside = tmp_path.parent / "outside.onnx"
    outside.write_bytes(b"fixture")
    runtime = MossHMONNXRuntime(
        tmp_path,
        {"components": {"encode": {"hmonnx_file": str(outside)}}},
        "cpu",
    )

    assert runtime.available() is False


def test_hmonnx_runtime_reuses_session_aligns_inputs_and_names_outputs(tmp_path: Path, monkeypatch):
    component = tmp_path / "encode.onnx"
    component.write_bytes(b"fixture")
    calls = []

    class Info:
        dtype = torch.float16
        shape = (1, 80, 3000)

    class FakeSession:
        def __init__(self, path):
            calls.append(("init", path))

        def to(self, device):
            calls.append(("to", device))
            return self

        def get_input_names(self):
            return ["input_features"]

        def get_input(self, name):
            assert name == "input_features"
            return Info()

        def get_output_names(self):
            return ["audio_embeds"]

        def run(self, feed):
            calls.append(("run", feed["input_features"].dtype, feed["input_features"].device.type))
            return (torch.ones(1, 375, 1024),)

    api = types.ModuleType("xhquant.api")
    api.HMONNXInference = FakeSession
    monkeypatch.setitem(sys.modules, "xhquant.api", api)

    runtime = MossHMONNXRuntime(
        tmp_path,
        {"components": {"encode": {"hmonnx_file": "encode.onnx"}}},
        "cpu",
    )
    first = runtime.encode(torch.zeros(1, 80, 3000, dtype=torch.float32))
    second = runtime.encode(torch.zeros(1, 80, 3000, dtype=torch.float32))

    assert tuple(first["audio_embeds"].shape) == (1, 375, 1024)
    assert tuple(second["audio_embeds"].shape) == (1, 375, 1024)
    assert [item[0] for item in calls].count("init") == 1
    assert [item[0] for item in calls].count("to") == 1
    assert [item[0] for item in calls].count("run") == 2
    assert all(item[1:] == (torch.float16, "cpu") for item in calls if item[0] == "run")


def test_hmonnx_runtime_configures_selected_golden_directory(tmp_path: Path, monkeypatch):
    component = tmp_path / "encode" / "hmonnx" / "encode.onnx"
    component.parent.mkdir(parents=True)
    component.write_bytes(b"fixture")

    class FakeSession:
        def __init__(self, path):
            self.path = path
            self.save_golden = False
            self.save_golden_dir = None

        def to(self, device):
            return self

        def get_input_names(self):
            return ["input_features"]

        def get_input(self, name):
            del name
            return type("Info", (), {"dtype": torch.float16, "shape": (1, 80, 3000)})()

        def get_output_names(self):
            return ["audio_embeds"]

    api = types.ModuleType("xhquant.api")
    api.HMONNXInference = FakeSession
    monkeypatch.setitem(sys.modules, "xhquant.api", api)

    runtime = MossHMONNXRuntime(
        tmp_path,
        {"components": {"encode": {"hmonnx_file": "encode/hmonnx/encode.onnx"}}},
        "cuda:0",
    )
    runtime.enable_golden(("encode",))
    session = runtime._session("encode")
    golden_dir = component.parent / "golden"

    assert session.save_golden is True
    assert session.save_golden_dir == str(golden_dir)
    assert golden_dir.is_dir()


def test_hmonnx_runtime_close_drops_sessions(tmp_path: Path, monkeypatch):
    component = tmp_path / "encode.onnx"
    component.write_bytes(b"fixture")
    closed = []

    class FakeSession:
        def __init__(self, path):
            del path

        def to(self, device):
            return self

        def close(self):
            closed.append("close")

        def get_input_names(self):
            return ["input_features"]

        def get_input(self, name):
            del name
            return type("Info", (), {"dtype": torch.float16, "shape": (1, 80, 3000)})()

        def get_output_names(self):
            return ["audio_embeds"]

    api = types.ModuleType("xhquant.api")
    api.HMONNXInference = FakeSession
    monkeypatch.setitem(sys.modules, "xhquant.api", api)
    runtime = MossHMONNXRuntime(
        tmp_path,
        {"components": {"encode": {"hmonnx_file": "encode.onnx"}}},
        "cpu",
    )
    runtime._session("encode")
    assert runtime._sessions
    runtime.close()
    assert runtime._sessions == {}
    assert closed == ["close"]


def test_runtime_tool_path_restores_process_environment(monkeypatch, tmp_path: Path):
    import os
    import sys

    environment_bin = Path(sys.executable).resolve().parent
    ninja = environment_bin / "ninja"
    assert ninja.is_file()
    original = os.environ.get("PATH")
    monkeypatch.setattr("shutil.which", lambda name: None if name == "ninja" else original)
    with _runtime_tool_path():
        assert str(environment_bin) in os.environ["PATH"].split(os.pathsep)
    assert os.environ.get("PATH") == original


def test_golden_inputs_follow_prefill_and_decode_abi():
    class Info:
        def __init__(self, name):
            self.dtype = torch.float16 if name == "inputs_embeds" else torch.int32
            self.shape = (1, 4, 8) if name == "inputs_embeds" else (1,)

    class FakeSession:
        def get_input_names(self):
            return ["inputs_embeds", "past_seq_length", "current_input_length"]

        def get_input(self, name):
            return Info(name)

    mel = __import__("numpy").zeros((80, 3000), dtype="float32")
    prefill = _golden_inputs(FakeSession(), "prefill", mel)
    decode = _golden_inputs(FakeSession(), "decode", mel)

    assert tuple(prefill["inputs_embeds"].shape) == (1, 4, 8)
    assert int(prefill["past_seq_length"].item()) == 0
    assert int(prefill["current_input_length"].item()) == 4
    assert int(decode["past_seq_length"].item()) == 4
    assert int(decode["current_input_length"].item()) == 1


def test_dump_golden_uses_hmonnx_save_golden_dir(monkeypatch, tmp_path: Path):
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize import workflow

    graph = tmp_path / "encode.onnx"
    graph.write_bytes(b"fixture")
    (tmp_path / "export_meta_info.json").write_text(
        '{"components":{"encode":{"hmonnx_file":"encode.onnx"},'
        '"prefill_decode":{"prefill":{"hmonnx_file":"encode.onnx"},'
        '"decode":{"hmonnx_file":"encode.onnx"}}}}',
        encoding="utf-8",
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")

    class FakeSession:
        def __init__(self, path):
            self.path = path
            self.save_golden = False
            self.save_golden_dir = None
            self.step = None

        def initialize(self):
            return None

        def to(self, device):
            return None

        def get_input_names(self):
            return []

        def __call__(self, *inputs):
            assert not inputs
            assert self.save_golden is True
            assert self.save_golden_dir is not None
            return None

    monkeypatch.setitem(__import__("sys").modules, "xhquant.api", type("API", (), {"HMONNXGoldenInference": FakeSession})())
    monkeypatch.setattr(workflow, "_golden_audio_path", lambda _: audio)
    monkeypatch.setattr(workflow, "_golden_inputs", lambda *_: {})
    monkeypatch.setattr(
        workflow,
        "_fused_embeds_from_encoder",
        lambda *_: __import__("numpy").ones((4, 8), dtype="float32"),
    )
    processing = __import__(
        "xhmodel_merak.xh_llm.models.moss_transcribe_diarize.processing",
        fromlist=["extract_moss_log_mel"],
    )
    monkeypatch.setattr(
        processing,
        "extract_moss_log_mel",
        lambda *_: (__import__("numpy").zeros((80, 3000), dtype="float32"), 1),
    )

    result = workflow.MossTranscribeDiarizeWorkflow.__new__(workflow.MossTranscribeDiarizeWorkflow)
    result.model_dir = str(tmp_path)
    export = type("Export", (), {"work_dir": str(tmp_path)})()
    assert workflow.MossTranscribeDiarizeWorkflow.dump_golden(result, export, "cuda:0", audio)


@pytest.mark.parametrize("prompt_length", [3, 8, 9, 17, 29])
def test_chunked_prefill_matches_single_pass_with_real_decoder(tmp_path, prompt_length, monkeypatch):
    """generate_hmonnx chunk math against a tiny fused stand-in, not the deleted product graph."""
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.transcribe import generate_hmonnx
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.prefill_decode import MossLMHead

    torch.manual_seed(71)

    class FakeFused(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x, past, valid, key, value):
            hidden = self.lin(x)
            return hidden, key, value

    graph = FakeFused().eval()
    head = MossLMHead(hidden_size=8, vocab_size=16).eval()
    fused = torch.randn(1, prompt_length, 8)
    shape = (2, 1, 32, 4)
    scalar = lambda n: torch.tensor([n], dtype=torch.int32)
    (tmp_path / "generation_config.json").write_text('{"max_new_tokens": 2}')
    expected, reference_steps = [], []
    with torch.no_grad():
        hidden, k, v = graph(fused, scalar(0), scalar(prompt_length),
                              torch.zeros(shape), torch.zeros(shape))
        reference_hidden = hidden.clone()
        for step in range(2):
            token = int(head(hidden[:, -1:]).argmax(-1).item())
            expected.append(token)
            hidden, k, v = graph(head.projection.weight[token][None, None],
                                scalar(prompt_length + step), scalar(1), k, v)
            reference_steps.append(hidden.clone())

    calls = []

    class Runtime:
        device = "cpu"

        def input_shape(self, phase, name):
            return (1, 8, 8) if name == "inputs_embeds" else shape

        def decoder_cache_length(self):
            padded = ((prompt_length + 7) // 8) * 8
            return max(padded, prompt_length + 2)

        def uses_page_attention(self):
            return False

        def uses_inplace_cache(self):
            return False

        def prefill(self, x, past, valid, key, value):
            offset, count = int(past.item()), int(valid.item())
            calls.append((offset, count))
            hidden, key, value = graph(x, past, valid, key, value)
            torch.testing.assert_close(hidden[:, :count], reference_hidden[:, offset:offset + count])
            return dict(last_hidden_state=hidden, present_key_cache=key, present_value_cache=value)

        def decode(self, x, past, valid, key, value):
            hidden, key, value = graph(x, past, valid, key, value)
            torch.testing.assert_close(hidden, reference_steps[int(past.item()) - prompt_length])
            return dict(last_hidden_state=hidden, present_key_cache=key, present_value_cache=value)

    from xhmodel_merak.errors import ModelError
    tokenizer = types.SimpleNamespace(eos_token_id=-1, decode=lambda ids, **kw: "partial")
    with torch.no_grad(), pytest.raises(ModelError, match="before EOS") as caught:
        generate_hmonnx(
            fused[0].numpy(),
            np.ones(prompt_length),
            Runtime(),
            head,
            tokenizer,
            tmp_path,
            max_new_tokens=2,
        )
    assert caught.value.details["generated_token_ids"] == expected
    assert caught.value.details["raw_transcript"] == "partial"
    assert caught.value.details["stop_reason"] == "max_new_tokens"
    assert calls == [(i, min(8, prompt_length - i)) for i in range(0, prompt_length, 8)]


def test_prefill_capacity_is_checked_before_execution(tmp_path):
    from xhmodel_merak.errors import ModelError
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.transcribe import generate_hmonnx

    (tmp_path / "generation_config.json").write_text('{"max_new_tokens": 1}')

    class Runtime:
        def input_shape(self, phase, name):
            return (1, 8, 8) if name == "inputs_embeds" else (2, 1, 30, 4)

        def decoder_cache_length(self):
            return 30

    with pytest.raises(ModelError, match="padded prefill"):
        generate_hmonnx(np.zeros((29, 8)), np.ones(29), Runtime(), None, None, tmp_path)


def test_workflow_example_passes_config_and_export_arguments(monkeypatch, tmp_path):
    """Run the real example orchestration; stub only expensive workflow execution."""
    import importlib.util
    import xhmodel_merak.workflows as workflows

    path = Path(__file__).resolve().parents[2] / (
        "examples_merak/asr/moss_transcribe_diarize/moss_transcribe_diarize_workflow.py"
    )
    spec = importlib.util.spec_from_file_location("moss_workflow_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    calls = []

    class Workflow:
        def quant(self, **kwargs):
            calls.append(("quant", kwargs))
            return "quant_result"

        def export(self, **kwargs):
            calls.append(("export", kwargs))
            return "export_result"

    def from_config(**kwargs):
        calls.append(("config", kwargs))
        return Workflow()

    monkeypatch.setattr(workflows.AutoWorkflow, "from_config", from_config)
    monkeypatch.setattr(sys, "argv", [
        str(path), "--model-dir", "/local/model", "--output-dir", str(tmp_path / "out"),
        "--device", "cpu",
    ])
    example.main()
    assert calls == [
        ("config", dict(model_dir="/local/model", config_path=example.DEFAULT_CONFIG_PATH, debug=False)),
        ("quant", dict(output_dir=str(tmp_path / "out" / "quant"), device="cpu")),
        ("export", dict(quant_result="quant_result", output_dir=str(tmp_path / "out"), device="cpu")),
    ]




def _rotary_artifact_meta():
    geometry = dict(prefill_length=512, cache_length=4096, rope_max_positions=4096,
                    rope_mode="fp32_table_integer_index",
                    onnx_file="graph.onnx", hmonnx_file="graph.onnx")
    return dict(quant_type="w8a8_sefp", weights_file="weights.pt", lm_head_file="weights.pt",
                components=dict(encode=dict(onnx_file="graph.onnx", hmonnx_file="graph.onnx"),
                                prefill_decode={phase: dict(geometry) for phase in ("prefill", "decode")}))


@pytest.mark.parametrize("change", [
    {"rope_mode": "dynamic"}, {"rope_max_positions": 2048},
    {"cache_length": True}, {"prefill_length": 0}, {"cache_length": 2048},
])
def test_rotary_artifact_contract_rejects_invalid_or_mixed_phases(change):
    from xhmodel_merak.model_integrations.moss_transcribe_diarize.artifact import validate_decoder_contract

    meta = _rotary_artifact_meta()
    validate_decoder_contract(meta)
    meta["components"]["prefill_decode"]["decode"].update(change)
    with pytest.raises(ValueError):
        validate_decoder_contract(meta)


def test_build_and_load_both_reject_pre_lookup_artifacts(tmp_path):
    import json
    from xhmodel_merak.errors import ModelError
    from xhmodel_merak.model_catalog import get_recipe
    from xhmodel_merak.model_integrations.moss_transcribe_diarize import build, inference

    for name in (
        "graph.onnx", "weights.pt", "config.json", "tokenizer.json", "vocab.json", "merges.txt",
        "tokenizer_config.json", "preprocessor_config.json", "generation_config.json",
        "configuration_moss_transcribe_diarize.py", "modeling_moss_transcribe_diarize.py",
        "processing_moss_transcribe_diarize.py",
    ):
        (tmp_path / name).write_text("fixture")
    recipe = get_recipe("moss-transcribe-diarize-0.9b")
    metadata = tmp_path / "export_meta_info.json"
    meta = _rotary_artifact_meta()
    _package_fixture_resources(tmp_path, meta)
    metadata.write_text(json.dumps(meta))
    validators = (
        lambda: build.validate_artifact(tmp_path, types.SimpleNamespace(
            model_id=recipe.model_id, components=recipe.components,
        )),
        lambda: inference.adapter.validate_artifact(
            recipe=recipe, root=tmp_path, manifest={}, meta_file=metadata,
        ),
    )
    for validate in validators:
        validate()
    del meta["components"]["prefill_decode"]["decode"]["rope_mode"]
    metadata.write_text(json.dumps(meta))
    for validate in validators:
        with pytest.raises(ModelError, match="re-export"):
            validate()


def test_flash_attention_artifact_accepts_inplace_hmonnx_without_onnx(tmp_path):
    import json
    from xhmodel_merak.model_catalog import get_recipe
    from xhmodel_merak.model_integrations.moss_transcribe_diarize import build, inference

    for name in (
        "encode.onnx", "prefill.hmonnx.onnx", "decode.hmonnx.onnx", "weights.pt",
        "config.json", "tokenizer.json", "vocab.json", "merges.txt",
        "tokenizer_config.json", "preprocessor_config.json", "generation_config.json",
        "configuration_moss_transcribe_diarize.py", "modeling_moss_transcribe_diarize.py",
        "processing_moss_transcribe_diarize.py",
    ):
        (tmp_path / name).write_text("fixture")
    geometry = dict(
        prefill_length=512, cache_length=4096, rope_max_positions=4096,
        rope_mode="fp32_table_integer_index", cache_mode="inplace",
        hmonnx_file="prefill.hmonnx.onnx",
    )
    meta = dict(
        quant_type="w8a8_sefp", weights_file="weights.pt", lm_head_file="weights.pt",
        components=dict(
            encode=dict(onnx_file="encode.onnx", hmonnx_file="encode.onnx"),
            prefill_decode=dict(
                prefill=dict(geometry, hmonnx_file="prefill.hmonnx.onnx"),
                decode=dict(geometry, hmonnx_file="decode.hmonnx.onnx"),
            ),
        ),
    )
    metadata = tmp_path / "export_meta_info.json"
    _package_fixture_resources(tmp_path, meta)
    metadata.write_text(json.dumps(meta))
    recipe = get_recipe("moss-transcribe-diarize-0.9b")
    plan = types.SimpleNamespace(model_id=recipe.model_id, components=recipe.components)
    assert build.validate_artifact(tmp_path, plan)["loadable"] is True
    inference.adapter.validate_artifact(
        recipe=recipe, root=tmp_path, manifest={}, meta_file=metadata,
    )


def _package_fixture_resources(root, meta):
    from xhmodel_merak.model_integrations.moss_transcribe_diarize.artifact import REQUIRED_RUNTIME_FILES

    destination = root / "hf_config"
    destination.mkdir()
    for name in REQUIRED_RUNTIME_FILES:
        (root / name).rename(destination / name)
    meta["hf_config"] = "hf_config"
