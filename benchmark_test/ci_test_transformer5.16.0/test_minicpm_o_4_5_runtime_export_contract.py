import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml


CONFIG_DIR = Path("configs_merak/workflows/xh2a/llm_models/minicpm_o_4_5")


@pytest.mark.parametrize("profile", ("offline", "session", "stream"))
def test_unified_plan_and_llm_checkpoint_identity(tmp_path, profile):
    from xhmodel_merak.errors import ModelError
    from xhmodel_merak.model_build.checkpoint import verify_quant_model_identity
    from xhmodel_merak.model_build.resolver import resolve_build_plan

    model_id = "minicpm-o-4.5"
    source = tmp_path / "source"
    source.mkdir()
    checkpoint_config = {
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
        "vocab_size": 16, "rope_theta": 1000000.0,
    }
    host_config = {
        **checkpoint_config, "model_type": "minicpmo", "version": "4.5", "architectures": ["MiniCPMO"],
        "bos_token_id": 1, "eos_token_id": 2,
    }
    (source / "config.json").write_text(json.dumps(host_config))
    plan = resolve_build_plan(
        model_id=model_id, model_dir=str(source), output_dir=str(tmp_path / "artifact"), device="cpu", profile=profile,
    )
    assert plan.variant == plan.config_replacements["export.mode"] == profile
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "config.json"
    checkpoint_path.write_text(json.dumps(checkpoint_config))
    verify_quant_model_identity(checkpoint_dir, [(checkpoint_path, checkpoint_config)], plan)
    for field in ("hidden_size", "rope_theta"):
        changed = {**checkpoint_config, field: checkpoint_config[field] * 2}
        with pytest.raises(ModelError, match="does not match"):
            verify_quant_model_identity(checkpoint_dir, [(checkpoint_path, changed)], plan)


@pytest.mark.parametrize(
    "name",
    (
        "minicpm_o_4_5_xh2a_w4a8_gptq.yaml",
        "minicpm_o_4_5_xh2a_w8a8_gptq.yaml",
    ),
)
def test_release_configs_define_each_export_mode(name, tmp_path, monkeypatch):
    from xhmodel_merak.exec_backends import CacheTensor, HMONNXGoldenInference
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import export_token2wav, workflow as workflow_module
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_token2wav import (
        _convert,
        add_stream_estimator_cache_writer,
        add_stream_frontend_cache_writer,
    )
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.minicpmo_token2wav_modules import (
        FlowEstimatorStepWrapper,
        FlowStreamingFrontendWrapper,
    )
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.workflow import MiniCPMO45Workflow
    from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXBaseModel, HMONNXModel
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        MiniCPMO45Token2WavHMONNXRuntime,
        _FlowCacheHMONNXModel,
    )
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    config = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
    export = config["export"]
    components = export["components"]

    assert export["mode"] == "offline"
    assert set(components) == set(MiniCPMO45Workflow.SUPPORTED_COMPONENTS)
    assert "tts_projection" not in components
    assert components["tts"]["input_names"][-2:] == ["semantic_hidden_states", "semantic_mask"]

    for mode in ("offline", "session", "stream"):
        MiniCPMO45Workflow._validate_export_config({**export, "mode": mode}, config["quant"])

    stream_components = {
        key: {"stream_contract": value["streaming"], "graphs": {}}
        for key, value in components.items() if key.startswith("token2wav_")
    }
    for component, roles in (
        ("token2wav_flow_frontend", ("stream_flow_frontend", "stream_flow_prompt_frontend")),
        ("token2wav_flow_decoder", ("stream_flow_estimator_step", "stream_flow_prompt_estimator_step")),
        ("token2wav_hift", ("stream_hift_part1", "stream_hift_part2")),
    ):
        stream_components[component]["graphs"] = {role: f"{role}.onnx" for role in roles}
    stream_components["token2wav_hift"].update(hop_length=480)
    stream_components["token2wav_hift"]["stream_contract"] = {
        **stream_components["token2wav_hift"]["stream_contract"], "harmonic_count": 1, "source_length": 1,
    }
    for requested in (False, True):
        runtime = object.__new__(MiniCPMO45Token2WavHMONNXRuntime)
        runtime._load_streaming_roles(
            tmp_path, stream_components,
            lambda _path, *, enable_cuda_graph=requested, **kwargs: SimpleNamespace(enable_cuda_graph=enable_cuda_graph, **kwargs),
        )
        assert runtime.estimator_step_session.enable_cuda_graph is requested
        assert runtime.flow_frontend_session.enable_cuda_graph is False
        assert runtime.estimator_step_session.resident_cache is True
        assert runtime.prompt_frontend_session.enable_cuda_graph is requested
        assert runtime.prompt_estimator_step_session.enable_cuda_graph is requested

    tails = []

    def capture_convert(module, _inputs, _names, output_file, *_args, **_kwargs):
        if isinstance(module, FlowStreamingFrontendWrapper):
            tails.append(module.cache_tail_length)
        return output_file

    def export_frontend(**kwargs):
        cfg = kwargs["component_cfg"]
        flow = SimpleNamespace(up_rate=2, setup_cache=lambda *_args, **_kwargs: {
            key: torch.empty(shape, device="meta") for key, shape in cfg["streaming"]["base_cache_shapes"].items()
        })
        return export_token2wav.export_flow_frontend(
            flow, Path(kwargs["work_dir"]) / "token2wav_flow_frontend", cfg, kwargs["target_device"],
        )

    def export_other(**kwargs):
        return {"graphs": {role: Path(kwargs["work_dir"]) / "unused.onnx" for role in (
            "main", "campplus", "speech_tokenizer",
        )}}

    with monkeypatch.context() as patch:
        patch.setattr(export_token2wav, "_convert", capture_convert)
        patch.setattr(workflow_module, "export_minicpm_o_4_5_token2wav_flow_frontend", export_frontend)
        for role in ("vision", "audio", "llm", "tts", "campplus", "speech_tokenizer", "token2wav_flow_decoder", "token2wav_hift"):
            patch.setattr(workflow_module, f"export_minicpm_o_4_5_{role}", export_other)
        workflow = MiniCPMO45Workflow(str(tmp_path), str(CONFIG_DIR / name))
        for mode in ("session", "stream"):
            for tail in (100, 50):
                workflow.export(QuantResult(raw_model_dir=str(tmp_path), skipped=True), str(tmp_path / f"{mode}-{tail}"), "cpu", {
                    "export.mode": mode,
                    "export.components.token2wav_flow_decoder.streaming.prompt_cache_policy.recent_mel_frames": tail,
                })
    assert tails == [100, 50, 100, 50]

    for past, current, tail, expected in (
        (4, 1, 2, [0, 3, 4, 5, 6, 7]),
        (4, 2, 2, [3, 4, 5, 6, 7, 8]),
        (6, 3, 2, [0, 0, 0, 1, 2, 6]),
        (6, 1, 4, [0, 1, 2, 7, 1, 2]),
    ):
        actual = FlowStreamingFrontendWrapper._updated_attention_cache(
            torch.arange(1, 10, dtype=torch.float32).reshape(1, 1, 1, 9, 1),
            torch.tensor([past]), torch.tensor([current]), torch.tensor([1]), tail, 6, is_base_cache=True,
        )
        torch.testing.assert_close(
            actual.flatten(), torch.tensor(expected, dtype=actual.dtype), rtol=0, atol=0,
        )

    class CacheUpdate(torch.nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind

        def forward(self, cache, current, past_length, current_length, prompt_length):
            if self.kind == "estimator":
                return FlowEstimatorStepWrapper._updated_stream_cache(
                    current, cache, past_length, current_length, prompt_length, 2,
                )
            updated = FlowStreamingFrontendWrapper._updated_attention_cache(
                torch.cat((cache, current), dim=3), past_length, current_length, prompt_length,
                2, 6, is_base_cache=self.kind == "base",
            )
            return current[:1], current[:, :, :1], current[:, :, :, :1], updated

    shape = (2, 1, 2, 6, 4)
    current = torch.arange(32, dtype=torch.float16).reshape(2, 1, 2, 2, 4)
    for kind in ("base", "up", "estimator"):
        component = "token2wav_flow_decoder" if kind == "estimator" else "token2wav_flow_frontend"
        cache_input = "past_estimator_att_cache" if kind == "estimator" else "past_conformer_att_cache"
        outputs = ("estimator_cache_write_done",) if kind == "estimator" else (
            "mu", "spks", "cnn", "conformer_cache_write_done",
        )
        finish = add_stream_estimator_cache_writer if kind == "estimator" else add_stream_frontend_cache_writer
        prompt = 1 if kind == "base" else 2
        owner = CacheTensor(torch.arange(96, dtype=torch.float16).reshape(shape))
        owner.data[:, :, :, :2] = 0
        expected = owner.data[:, :, :, -4:].clone()
        path = tmp_path / f"{kind}.onnx"
        _convert(
            CacheUpdate(kind),
            (
                owner.data, current, torch.tensor([4], dtype=torch.int32),
                torch.tensor([2], dtype=torch.int32), torch.tensor([prompt], dtype=torch.int32),
            ),
            ((cache_input, "current_cache", "past_length", "current_length", "prompt_length"), outputs),
            path, "XH2a", components[component]["quant_type"], finish=finish,
        )
        session = HMONNXGoldenInference(str(path)).to("cpu")
        session.initialize()
        for offset in (100, 200, 300):
            chunk = current + offset
            past_length = expected.shape[3]
            combined = torch.cat((chunk, expected) if kind == "estimator" else (expected, chunk), dim=3)
            if combined.shape[3] > 6:
                if kind == "base":
                    duplicated = combined.repeat(1, 1, 1, 2, 1)
                    combined = torch.cat((duplicated[:, :, :, :2 * prompt], duplicated[:, :, :, -4:]), dim=3)
                    combined = combined[:, :, :, :combined.shape[3] // 2]
                else:
                    combined = torch.cat((combined[:, :, :, :prompt], combined[:, :, :, -2:]), dim=3)
            expected = combined
            session(
                owner, chunk, torch.tensor([past_length], dtype=torch.int32),
                torch.tensor([2], dtype=torch.int32), torch.tensor([prompt], dtype=torch.int32),
            )
            packed = torch.nn.functional.pad(expected, (0, 0, 6 - expected.shape[3], 0))
            torch.testing.assert_close(owner.data, packed, rtol=0, atol=0)
        del session

        if kind == "estimator" and torch.cuda.is_available():
            monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
            eager = HMONNXModel(str(path), device_map="cuda:0")
            captured = _FlowCacheHMONNXModel(str(path), enable_cuda_graph=True, device_map="cuda:0")
            interpreter = captured.hmonnx_session.interpreter
            owners = [CacheTensor(owner.data.cuda() + bank) for bank in range(10)]
            references = [CacheTensor(value.data.clone()) for value in owners]
            for length in (4, 6, 4):
                for bank, (actual, reference) in enumerate(zip(owners, references)):
                    args = (
                        current.cuda() + bank * 10, torch.tensor([length], dtype=torch.int32, device="cuda"),
                        torch.tensor([2], dtype=torch.int32, device="cuda"),
                        torch.tensor([prompt], dtype=torch.int32, device="cuda"),
                    )
                    torch.testing.assert_close(captured(actual, *args), eager(reference, *args), rtol=0, atol=0)
                    # Check every bank, including those not called yet this round.
                    for left, right in zip(owners, references):
                        torch.testing.assert_close(left.data, right.data, rtol=0, atol=0)
            assert len(interpreter.cached_capture_keys) == 10
            assert interpreter.capture_disabled_reason is None
            assert interpreter.has_captured_graph
            runtime = object.__new__(MiniCPMO45Token2WavHMONNXRuntime)
            HMONNXBaseModel.__init__(runtime, device_map="cuda:0")
            runtime.estimator_step_session = captured
            runtime._stream_state = None
            runtime.release_state()
            assert interpreter.cached_capture_keys == ()
            actual, reference = (CacheTensor(value.data.clone()) for value in (owners[0], references[0]))
            torch.testing.assert_close(captured(actual, *args), eager(reference, *args), rtol=0, atol=0)
            torch.testing.assert_close(actual.data, reference.data, rtol=0, atol=0)
            assert interpreter.cached_capture_keys == (actual.data_ptr(),)
            runtime.release_state()


def test_tts_graph_fuses_masked_normalized_semantics_and_audio_head():
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5._tts_model_impl import _LlamaModel

    class Graph:
        use_cache = False
        num_logits_to_keep = 0
        layers = []
        rotary_emb = SimpleNamespace(cos_cached=torch.zeros(1), sin_cached=torch.zeros(1))
        cos_slice = staticmethod(lambda value, _past: value)
        sin_slice = staticmethod(lambda value, _past: value)
        norm = torch.nn.Identity()
        semantic_projector = torch.nn.Linear(4, 3, bias=True)
        audio_code_head = torch.nn.Linear(3, 5, bias=False)
        normalize_projected_hidden = True

    graph = Graph()
    with torch.no_grad():
        graph.semantic_projector.weight.copy_(
            torch.eye(3, 4, dtype=graph.semantic_projector.weight.dtype)
        )
        graph.semantic_projector.bias.zero_()
        graph.audio_code_head.weight.copy_(
            torch.arange(1, 16, dtype=graph.audio_code_head.weight.dtype).reshape(5, 3) / 16
        )
    embeddings = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    semantic = torch.tensor([[[1.0, 0.0, 0.5, -1.0], [3.0, 2.0, 1.0, 0.0]]])
    mask = torch.tensor([[[1.0], [0.0]]])
    output = _LlamaModel.forward(
        graph,
        inputs_embeds=embeddings,
        past_seq_length=torch.tensor([0], dtype=torch.int32),
        current_input_length=torch.tensor([2], dtype=torch.int32),
        attention_mask=torch.zeros((1, 1, 2, 2)),
        semantic_hidden_states=semantic,
        semantic_mask=mask,
    ).last_hidden_state
    projected = torch.nn.functional.normalize(graph.semantic_projector(semantic), p=2, dim=-1) * mask

    torch.testing.assert_close(output, graph.audio_code_head(embeddings + projected))
    torch.testing.assert_close((embeddings + projected)[:, 1], embeddings[:, 1])
