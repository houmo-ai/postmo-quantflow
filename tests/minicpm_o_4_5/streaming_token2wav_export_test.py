from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
import yaml
from onnx import helper, numpy_helper
from torch import nn


CONFIG_PATH = Path("configs_merak/workflows/xh2a/llm_models/minicpm_o_4_5/minicpm_o_4_5_xh2a_w8a8_gptq.yaml")


def test_token2wav_convert_passes_string_path_to_legacy_onnx_export(tmp_path, monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import export_token2wav

    exported_paths: list[object] = []
    saved_paths: list[object] = []

    def fake_export(module, inputs, output_file, **kwargs) -> None:
        del module, inputs, kwargs
        exported_paths.append(output_file)

    def fake_load(path):
        del path
        return onnx.ModelProto()

    def fake_save(model, path):
        del model
        saved_paths.append(path)

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    monkeypatch.setattr(onnx, "load", fake_load)
    monkeypatch.setattr(onnx, "save", fake_save)
    monkeypatch.setattr(onnx.checker, "check_model", lambda _path: None)
    monkeypatch.setattr(export_token2wav, "convert_onnx_to_hmonnx", lambda *_args, **_kwargs: None)

    export_token2wav._convert(
        nn.Identity(),
        (torch.zeros((1, 1)),),
        (("input",), ("output",)),
        tmp_path / "large.onnx",
        "XH2a",
        "w16a16_sefp",
    )

    assert len(exported_paths) == len(saved_paths) == 1
    assert isinstance(exported_paths[0], str)
    assert exported_paths == saved_paths
    assert Path(exported_paths[0]).name == "large_source.onnx"
    assert not Path(exported_paths[0]).is_relative_to(tmp_path)


def test_stream_export_copy_bounds_registered_flow_buffers_without_mutating_source() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_token2wav import (
        make_stream_export_flow,
    )

    class Estimator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.arange(3, dtype=torch.float32))
            self.register_buffer("att_cache_buffer", torch.ones(16, 2, 8, 1000, 128), persistent=False)
            self.register_buffer("cnn_cache_buffer", torch.ones(16, 2, 1024, 2), persistent=False)
            self.use_cuda_graph = True

    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.estimator = Estimator()
            self.register_buffer("att_cache_buffer", torch.ones(16, 16, 2, 8, 1000, 128), persistent=False)
            self.register_buffer("cnn_cache_buffer", torch.ones(16, 16, 2, 1024, 2), persistent=False)

    class Flow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.arange(2, dtype=torch.float32))
            self.decoder = Decoder()

    source = Flow()
    source_att = source.decoder.att_cache_buffer.clone()
    source_estimator_att = source.decoder.estimator.att_cache_buffer.clone()

    exported = make_stream_export_flow(source, cache_capacity=150, append_capacity=56)

    assert tuple(exported.decoder.att_cache_buffer.shape) == (16, 16, 2, 8, 206, 128)
    assert tuple(exported.decoder.estimator.att_cache_buffer.shape) == (16, 2, 8, 206, 128)
    assert exported.decoder.att_cache_buffer.dtype == source.decoder.att_cache_buffer.dtype
    assert exported.decoder.att_cache_buffer.device == source.decoder.att_cache_buffer.device
    assert "att_cache_buffer" in exported.decoder._non_persistent_buffers_set
    assert "att_cache_buffer" in exported.decoder.estimator._non_persistent_buffers_set
    assert exported.decoder.estimator.use_cuda_graph is False
    assert torch.equal(source.decoder.att_cache_buffer, source_att)
    assert torch.equal(source.decoder.estimator.att_cache_buffer, source_estimator_att)
    assert all(torch.equal(left, right) for left, right in zip(source.parameters(), exported.parameters(), strict=True))
    assert source.decoder.att_cache_buffer.shape == (16, 16, 2, 8, 1000, 128)
    assert source.decoder.estimator.att_cache_buffer.shape == (16, 2, 8, 1000, 128)


def test_streaming_frontend_exports_prompt_and_chunk_roles(tmp_path, monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import export_token2wav

    class Flow:
        up_rate = 2

        def setup_cache(self, token, mel, spk, n_timesteps):
            del token, mel, spk, n_timesteps
            return {
                "conformer_cnn_cache": torch.ones(1, 2, 1),
                "conformer_att_cache": torch.ones(1, 1, 1, 3, 1),
                "estimator_cnn_cache": torch.ones(1, 1, 1, 2, 1),
                "estimator_att_cache": torch.ones(1, 1, 1, 1, 3, 1),
            }

    converted_roles: list[str] = []

    def fake_convert(module, inputs, names, output_file, target_device, quant_type, finish=None):
        del module, inputs, names, target_device, quant_type, finish
        converted_roles.append(output_file.stem)
        return output_file

    monkeypatch.setattr(export_token2wav, "_convert", fake_convert)
    result = export_token2wav.export_flow_frontend(
        Flow(),
        tmp_path,
        {
            "export_mode": "stream",
            "token_capacity": 8,
            "mel_capacity": 8,
            "quant_type": "w16a16_sefp",
            "n_timesteps": 1,
            "cache_tail_length": 0,
            "streaming": {
                "prompt_token_capacity": 4,
                "prompt_mel_capacity": 6,
                "pre_lookahead_len": 3,
                "base_conformer_layers": 1,
                "cache_alignment": "right",
                "base_cache_valid_length": 3,
                "base_cache_shapes": {
                    "conformer_cnn_cache": [1, 2, 1],
                    "conformer_att_cache": [1, 1, 1, 3, 1],
                    "estimator_cnn_cache": [1, 1, 1, 2, 1],
                    "estimator_att_cache": [1, 1, 1, 1, 3, 1],
                },
                "cache_shapes": {
                    "conformer_cnn_cache": [1, 2, 1],
                    "conformer_att_cache": [1, 1, 1, 3, 1],
                    "estimator_cnn_cache": [1, 1, 1, 2, 1],
                    "estimator_att_cache": [1, 1, 1, 1, 3, 1],
                },
                "chunk_token_capacity": 4,
                "frontend_input_names": [
                    "tokens",
                    "token_valid_length",
                    "last_chunk",
                    "embedding",
                    "past_conformer_cnn_cache",
                    "past_conformer_att_cache",
                    "conformer_cache_valid_length",
                    "prompt_cache_valid_length",
                ],
                "frontend_output_names": [
                    "mu",
                    "spks",
                    "present_conformer_cnn_cache",
                    "conformer_cache_write_done",
                    "present_conformer_cache_valid_length",
                ],
                "output_mel_capacities": {"normal": 2, "final": 4},
            },
        },
        "XH2a",
    )

    assert set(result["graphs"]) == {"stream_flow_prompt_frontend", "stream_flow_frontend"}
    assert len(converted_roles) == 2
    assert result["stream_contract"]["chunk_token_capacity"] == 4
    assert result["stream_contract"]["base_conformer_layers"] == 1


def test_flow_frontend_normalization_does_not_export_dynamic_shape_expand(tmp_path) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.minicpmo_token2wav_modules import (
        FlowFrontendWrapper,
    )

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Identity()

        def forward(self, hidden, token_length):
            del token_length
            return hidden, hidden

    class Flow(nn.Module):
        up_rate = 2

        def __init__(self) -> None:
            super().__init__()
            self.spk_embed_affine_layer = nn.Identity()
            self.input_embedding = nn.Embedding(8, 4)
            self.encoder = Encoder()
            self.encoder_proj = nn.Identity()

    wrapper = FlowFrontendWrapper(Flow(), mel_capacity=6)
    inputs = (
        torch.zeros(1, 3, dtype=torch.int64),
        torch.tensor([3], dtype=torch.int64),
        torch.zeros(1, 6, 4),
        torch.tensor([0], dtype=torch.int64),
        torch.ones(1, 4),
    )
    torch.onnx.export(
        wrapper,
        inputs,
        tmp_path / "flow_frontend.onnx",
        input_names=("tokens", "token_length", "prompt_feat", "prompt_feat_length", "embedding"),
        output_names=("hidden", "mel_mask", "embedding_out", "condition"),
        opset_version=17,
    )
    model = onnx.load(tmp_path / "flow_frontend.onnx")
    shape_outputs = {output for node in model.graph.node if node.op_type == "Shape" for output in node.output}
    assert not [
        node
        for node in model.graph.node
        if node.op_type == "Expand" and any(value in shape_outputs for value in node.input)
    ]


def test_exportable_embedding_normalization_matches_torch_reference() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.minicpmo_token2wav_modules import (
        _normalize_embedding,
    )

    torch.manual_seed(7)
    embedding = torch.randn(3, 192)
    assert torch.allclose(_normalize_embedding(embedding), torch.nn.functional.normalize(embedding, dim=1))


def test_yaml_declares_token2wav_stream_contract() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    components = config["export"]["components"]

    assert components["token2wav_flow_frontend"]["quant_type"] == "w8a16_sefp"
    assert components["token2wav_flow_decoder"]["quant_type"] == "w8a16_sefp"
    assert components["token2wav_hift"]["quant_type"] == "w8a16_sefp"


def test_frontend_capacity_is_declared_per_role() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    streaming = config["export"]["components"]["token2wav_flow_frontend"]["streaming"]
    assert streaming["frontend_output_names"] == [
        "mu",
        "spks",
        "present_conformer_cnn_cache",
        "conformer_cache_write_done",
        "present_conformer_cache_valid_length",
    ]
    assert streaming["output_mel_capacities"] == {"normal": 50, "final": 56}


def test_final_frontend_masks_padded_tokens_after_biased_encoder_embedding() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.minicpmo_token2wav_modules import (
        FlowStreamingFrontendWrapper,
    )

    class BiasedEmbed(nn.Module):
        def forward(self, value, mask):
            return value + 3, torch.zeros(1, 1, 4), mask

        def position_encoding(self, offset, size):
            del offset
            return torch.zeros(1, size * 2 - 1, 4)

    class Lookahead(nn.Module):
        pre_lookahead_len = 3

        def forward_chunk(self, value, cache):
            return value[:, :-3] + value[:, 3:], cache

    class Upsample(nn.Module):
        stride = 2

        def forward_chunk(self, value, lengths, cache):
            del lengths
            return value.repeat_interleave(2, dim=2), None, cache

    class Layer(nn.Module):
        def forward(self, value, mask, pos_emb, att_cache):
            del pos_emb
            append = att_cache.new_zeros((1, 1, value.shape[1], 2))
            return value, mask, torch.cat((att_cache, append), dim=2), att_cache.new_zeros((0, 0, 0))

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = BiasedEmbed()
            self.up_embed = BiasedEmbed()
            self.pre_lookahead_layer = Lookahead()
            self.up_layer = Upsample()
            self.encoders = nn.ModuleList([Layer()])
            self.up_encoders = nn.ModuleList([Layer()])
            self.normalize_before = False

    class Flow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spk_embed_affine_layer = nn.Identity()
            self.input_embedding = nn.Embedding(8, 4)
            self.encoder = Encoder()
            self.encoder_proj = nn.Identity()

    torch.manual_seed(5)
    wrapper = FlowStreamingFrontendWrapper(Flow(), cache_tail_length=2)
    common = (
        torch.ones(1, 4),
        torch.zeros(1, 4, 6),
        torch.zeros(2, 1, 1, 4, 2),
        torch.tensor([0], dtype=torch.int32),
    )
    reference = wrapper(
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        *common,
        torch.tensor([0], dtype=torch.int32),
    )
    fixed_capacity = wrapper(
        torch.tensor([[1, 2, 7, 7]], dtype=torch.int64),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        *common,
        torch.tensor([0], dtype=torch.int32),
    )

    assert torch.allclose(fixed_capacity[0][:, :, : reference[0].shape[2]], reference[0])


def test_estimator_step_cache_outputs_depend_on_past_cache_and_current_input() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.minicpmo_token2wav_modules import (
        FlowEstimatorStepWrapper,
    )

    class Block(nn.Module):
        def forward_chunk(self, hidden, t_embed, cnn_cache, att_cache, mask):
            del t_embed, mask
            cache_signal = cnn_cache.mean().reshape(1, 1, 1)
            return hidden + cache_signal, cnn_cache + hidden.mean(), att_cache + hidden.mean()

    class Estimator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj = nn.Identity()
            self.blocks = nn.ModuleList([Block()])
            self.t_embedder = nn.Identity()

        def final_layer(self, hidden, t_embed):
            del t_embed
            return hidden

    wrapper = FlowEstimatorStepWrapper(Estimator())
    common = (
        torch.ones(2, 3, 2),
        torch.ones(2, 3, 2),
        torch.ones(2),
        torch.ones(2, 3),
        torch.zeros(2, 3, 2),
    )
    first = wrapper(*common, torch.zeros(1, 1, 1, 1, 1), torch.zeros(1, 1, 1, 1, 1, 1))
    second = wrapper(*common, torch.ones(1, 1, 1, 1, 1), torch.ones(1, 1, 1, 1, 1, 1))

    assert not torch.equal(first[0], second[0])
    assert not torch.equal(first[1], torch.zeros_like(first[1]))
    assert not torch.equal(first[2], torch.zeros_like(first[2]))


def test_estimator_step_contract_has_functional_cache_inputs_and_outputs() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    streaming = config["export"]["components"]["token2wav_flow_decoder"]["streaming"]
    assert streaming["estimator_step_input_names"][-5:] == [
        "past_estimator_cnn_cache",
        "past_estimator_att_cache",
        "past_cache_valid_length",
        "current_frame_valid_length",
        "prompt_cache_valid_length",
    ]
    assert streaming["estimator_step_output_names"] == [
        "derivative_cfg",
        "present_estimator_cnn_cache",
        "estimator_cache_write_done",
    ]


def test_streaming_frontend_declares_real_token_length_and_fixed_capacity_policy() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    streaming = config["export"]["components"]["token2wav_flow_frontend"]["streaming"]

    assert streaming["frontend_input_names"][:4] == ["tokens", "token_valid_length", "last_chunk", "embedding"]
    assert streaming["cache_alignment"] == "right"
    assert streaming["base_conformer_layers"] == 6
    assert streaming["output_mel_capacities"] == {"normal": 50, "final": 56}


def test_stream_config_contains_only_behavior_driving_values() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    frontend = config["export"]["components"]["token2wav_flow_frontend"]["streaming"]
    flow = config["export"]["components"]["token2wav_flow_decoder"]["streaming"]
    hift = config["export"]["components"]["token2wav_hift"]["streaming"]

    assert frontend["output_mel_capacities"] == {"normal": 50, "final": 56}
    assert flow["host_timestep_cache_banks"] == 10
    assert "roles" not in flow
    assert flow["prompt_cache_policy"] == {"recent_mel_frames": 100}
    assert hift == {
        "frame_capacity": 56,
        "mel_cache_length": 8,
        "source_cache_length": 3840,
        "speech_cache_length": 3840,
    }


def test_flow_stream_roles_use_explicit_cache_contract_and_seven_outputs(tmp_path, monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import export_token2wav

    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.estimator = nn.Identity()
            self.estimator.register_buffer("att_cache_buffer", torch.zeros(16, 2, 8, 1000, 128), persistent=False)
            self.register_buffer("att_cache_buffer", torch.zeros(16, 16, 2, 8, 1000, 128), persistent=False)
            self.inference_cfg_rate = 0.7
            self.rand_noise = torch.zeros(1)

    class Flow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = Decoder()

    calls: list[tuple[type[nn.Module], tuple[str, ...], tuple[str, ...], tuple[torch.Size, ...]]] = []

    def fake_convert(module, inputs, names, output_file, target_device, quant_type, finish=None):
        del target_device, quant_type, finish
        calls.append((type(module), tuple(names[0]), tuple(names[1]), tuple(value.shape for value in inputs)))
        return output_file

    monkeypatch.setattr(export_token2wav, "_convert", fake_convert)
    result = export_token2wav.export_flow_decoder(
        Flow(),
        tmp_path,
        {
            "export_mode": "stream",
            "mel_capacity": 8,
            "quant_type": "w16a16_sefp",
            "n_timesteps": 1,
            "streaming": {
                "chunk_token_capacity": 28,
                "cache_capacity": 150,
                "append_capacity": 56,
                "cache_alignment": "right",
                "base_cache_valid_length": 50,
                "cache_shapes": {
                    "conformer_cnn_cache": [1, 512, 6],
                    "conformer_att_cache": [10, 1, 8, 150, 128],
                    "estimator_cnn_cache": [16, 16, 2, 1024, 2],
                    "estimator_att_cache": [16, 16, 2, 8, 150, 128],
                },
                "prompt_cache_policy": {"recent_mel_frames": 100},
                "pre_lookahead_len": 3,
                "estimator_step_input_names": [
                    "x_cfg",
                    "mu_cfg",
                    "t_cfg",
                    "spks_cfg",
                    "cond_cfg",
                    "past_estimator_cnn_cache",
                    "past_estimator_att_cache",
                    "past_cache_valid_length",
                    "current_frame_valid_length",
                    "prompt_cache_valid_length",
                ],
                "estimator_step_output_names": [
                    "derivative_cfg",
                    "present_estimator_cnn_cache",
                    "estimator_cache_write_done",
                ],
                "estimator_step_cache_shapes": {
                    "input_cnn": [16, 2, 1024, 2],
                    "input_att": [16, 2, 8, 150, 128],
                },
                "host_timestep_cache_banks": 10,
            },
        },
        "XH2a",
    )

    assert set(result["graphs"]) == {"stream_flow_prompt_estimator_step", "stream_flow_estimator_step"}
    assert len(calls) == 2
    _, input_names, output_names, input_shapes = calls[1]
    assert input_names == tuple(
        [
            "x_cfg",
            "mu_cfg",
            "t_cfg",
            "spks_cfg",
            "cond_cfg",
            "past_estimator_cnn_cache",
            "past_estimator_att_cache",
            "past_cache_valid_length",
            "current_frame_valid_length",
            "prompt_cache_valid_length",
        ]
    )
    assert output_names == ("derivative_cfg", "present_estimator_cnn_cache", "estimator_cache_write_done")
    assert input_shapes[5] == torch.Size((16, 2, 1024, 2))
    assert input_shapes[6] == torch.Size((16, 2, 8, 150, 128))
    assert input_shapes[7:] == (torch.Size((1,)), torch.Size((1,)), torch.Size((1,)))


def test_hift_stream_part1_export_formal_inputs_match_wrapper_order() -> None:
    from inspect import signature

    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_token2wav import HiftStreamPart1Wrapper

    expected = (
        "speech_feat",
        "speech_feat_valid_length",
        "past_mel",
        "past_mel_valid_length",
    )
    parameters = tuple(signature(HiftStreamPart1Wrapper.forward).parameters)[1:]
    assert parameters == expected


def test_hift_stream_config_omits_exporter_derived_cache_metadata() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    streaming = config["export"]["components"]["token2wav_hift"]["streaming"]
    assert "cache_inputs" not in streaming
    assert "cache_outputs" not in streaming
    assert "valid_length_outputs" not in streaming
    assert "cache_axes" not in streaming


def _f0_module() -> nn.Module:
    class Module(nn.Module):
        def f0_predictor(self, value: torch.Tensor) -> torch.Tensor:
            return value[:, 0, :]

    return Module()


def test_hift_stream_part1_left_aligns_logical_mel_and_returns_f0() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_token2wav import HiftStreamPart1Wrapper

    class Module(nn.Module):
        def f0_predictor(self, value: torch.Tensor) -> torch.Tensor:
            return value[:, 0, :]

    wrapper = HiftStreamPart1Wrapper(Module(), mel_cache_length=4)
    current = torch.zeros(1, 80, 6)
    current[:, 0, :3] = torch.tensor([1.0, 2.0, 3.0])
    outputs = wrapper(
        current,
        torch.tensor([3], dtype=torch.int32),
        torch.full((1, 80, 4), 9.0),
        torch.tensor([0], dtype=torch.int32),
    )

    assert len(outputs) == 2
    prepared_mel, f0 = outputs
    assert torch.equal(prepared_mel[0, 0, :3], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.count_nonzero(prepared_mel[0, 0, 3:]) == 0
    assert torch.equal(f0, prepared_mel[:, 0, :])


@pytest.mark.parametrize("past_mel_valid", (0, 8))
def test_hift_stream_part1_f0_aligns_with_mel_frames(past_mel_valid: int) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_token2wav import HiftStreamPart1Wrapper

    frame_capacity = 58
    mel_cache_length = 8
    module = _f0_module()
    wrapper = HiftStreamPart1Wrapper(
        module,
        mel_cache_length=mel_cache_length,
    )

    args = (
        torch.zeros(1, 80, frame_capacity),
        torch.tensor([frame_capacity], dtype=torch.int32),
        torch.zeros(1, 80, mel_cache_length),
        torch.tensor([past_mel_valid], dtype=torch.int32),
    )
    outputs = wrapper(*args)

    prepared_mel = outputs[0]
    assert len(outputs) == 2
    assert tuple(outputs[1].shape) == (1, frame_capacity + mel_cache_length)
    assert prepared_mel.shape[2] == frame_capacity + mel_cache_length


def test_hift_stream_part1_graph_excludes_speech_cache_and_overlap_window(tmp_path) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_token2wav import HiftStreamPart1Wrapper

    hift = _f0_module()
    wrapper = HiftStreamPart1Wrapper(
        hift,
        mel_cache_length=8,
    )
    output = tmp_path / "stream_hift.onnx"
    torch.onnx.export(
        wrapper,
        (
            torch.zeros(1, 80, 58),
            torch.tensor([58], dtype=torch.int32),
            torch.zeros(1, 80, 8),
            torch.tensor([0], dtype=torch.int32),
        ),
        output,
        input_names=(
            "speech_feat",
            "speech_feat_valid_length",
            "past_mel",
            "past_mel_valid_length",
        ),
        output_names=("prepared_speech_feat", "f0"),
        opset_version=17,
    )
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_token2wav import prepare_hift_stream_onnx

    model = prepare_hift_stream_onnx(onnx.load(output))
    initializers = {value.name for value in model.graph.initializer}
    assert all(
        node.op_type != "Slice" or len(node.input) <= 3 or not node.input[3] or node.input[3] in initializers
        for node in model.graph.node
    )
    assert all(
        node.op_type not in {"Squeeze", "Unsqueeze"} or len(node.input) <= 1 or node.input[1] in initializers
        for node in model.graph.node
    )
    assert [value.name for value in model.graph.input] == [
        "speech_feat",
        "speech_feat_valid_length",
        "past_mel",
        "past_mel_valid_length",
    ]
    assert "last_chunk" not in [value.name for value in model.graph.input]
    assert [value.name for value in model.graph.output] == [
        "prepared_speech_feat",
        "f0",
    ]
    assert len(model.graph.output[1].type.tensor_type.shape.dim) == 2
    assert sum(node.op_type == "ConvTranspose" for node in model.graph.node) == 0
    assert all(node.op_type != "STFT" for node in model.graph.node)
    assert all("speech_window" not in initializer.name for initializer in model.graph.initializer)
    assert all(node.op_type != "Sin" for node in model.graph.node)


def test_workflow_metadata_defaults_to_offline_roles(tmp_path, monkeypatch) -> None:
    from xhmodel_merak.workflows import AutoWorkflow
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import workflow as workflow_module
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    workflow = AutoWorkflow.from_config(model_dir="/models/MiniCPM-o-4_5", config_path=str(CONFIG_PATH))

    def fake_export(**kwargs):
        name = kwargs["component_cfg"].get("model_type", "token2wav")
        component_cfg = kwargs["component_cfg"]
        # Distinguish by structural config keys, not quant_type.
        assert component_cfg["export_mode"] == "offline"
        roles = ("part1", "part2") if "frame_capacity" in component_cfg else ("main",)
        return {
            "graphs": {role: tmp_path / f"{name}_{role}.onnx" for role in roles},
            "stream_contract": {"cache_inputs": ["cache"], "cache_outputs": ["present_cache"]},
        }

    monkeypatch.setattr(workflow_module, "export_minicpm_o_4_5_token2wav_flow_frontend", fake_export)
    monkeypatch.setattr(workflow_module, "export_minicpm_o_4_5_token2wav_flow_decoder", fake_export)
    monkeypatch.setattr(workflow_module, "export_minicpm_o_4_5_token2wav_hift", fake_export)
    for name in ("vision", "audio", "llm", "tts"):
        monkeypatch.setattr(workflow_module, f"export_minicpm_o_4_5_{name}", fake_export)

    def fake_speaker_export(role: str):
        def export_component(**kwargs):
            return {
                "quant_type": "w8a16_sefp",
                "graphs": {role: f"speaker/{role}.onnx"},
            }

        return export_component

    monkeypatch.setattr(workflow_module, "export_minicpm_o_4_5_campplus", fake_speaker_export("campplus"))
    monkeypatch.setattr(
        workflow_module,
        "export_minicpm_o_4_5_speech_tokenizer",
        fake_speaker_export("speech_tokenizer"),
    )

    result = workflow.export(
        QuantResult(raw_model_dir="/models/MiniCPM-o-4_5", skipped=True),
        str(tmp_path),
        "cpu",
    )
    meta = json.loads((Path(result.work_dir) / "export_meta_info.json").read_text(encoding="utf-8"))
    assert set(meta["components"]["token2wav_flow_frontend"]["graphs"]) == {"main"}
    assert set(meta["components"]["token2wav_flow_decoder"]["graphs"]) == {"main"}
    assert set(meta["components"]["token2wav_hift"]["graphs"]) == {"part1", "part2"}


def _constant_rooted_expand_model() -> onnx.ModelProto:
    """Mimic the official Flow `ConstantOfShape/Equal/Where` shape-vector pattern.

    The Expand target shape `[2, 3]` is produced by float32 shape arithmetic
    over constants followed by an int64 `Cast` (`Where(Equal([0.,0.],
    ConstantOfShape([2]) * 1.), [2.,3.], [0.,0.]) -> Cast(int64)`). It is
    entirely constant-foldable but is still routed through a non-initializer
    tensor by the raw torch trace.
    """
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["shape_values"],
            value=numpy_helper.from_array(np.array([2], dtype=np.int64), name="shape_values"),
        ),
        helper.make_node("ConstantOfShape", ["shape_values"], ["zeros"]),
        helper.make_node("Constant", [], ["one"], value=numpy_helper.from_array(np.array(1.0, dtype=np.float32))),
        helper.make_node("Mul", ["zeros", "one"], ["mul_out"]),
        helper.make_node(
            "Constant", [], ["zero_target"], value=numpy_helper.from_array(np.array([0.0, 0.0], dtype=np.float32))
        ),
        helper.make_node("Equal", ["zero_target", "mul_out"], ["eq"]),
        helper.make_node(
            "Constant", [], ["ones_target"], value=numpy_helper.from_array(np.array([2.0, 3.0], dtype=np.float32))
        ),
        helper.make_node("Where", ["eq", "ones_target", "zero_target"], ["expanded_src"]),
        helper.make_node("Cast", ["expanded_src"], ["expand_shape"], to=onnx.TensorProto.INT64),
        helper.make_node(
            "Constant",
            [],
            ["data_const"],
            value=numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(2, 3)),
        ),
        helper.make_node("Expand", ["data_const", "expand_shape"], ["expanded"]),
    ]
    graph = helper.make_graph(
        nodes,
        "constant_rooted_expand",
        [],
        [helper.make_tensor_value_info("expanded", onnx.TensorProto.FLOAT, [2, 3])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_constantize_dynamic_expand_shapes_folds_constant_rooted_shape_vectors() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_token2wav import (
        constantize_dynamic_expand_shapes,
    )

    model = _constant_rooted_expand_model()
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    expand = next(node for node in model.graph.node if node.op_type == "Expand")
    assert expand.input[1] not in initializer_names, "fixture must begin with a non-initializer Expand shape"

    constantize_dynamic_expand_shapes(model)

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    expand = next(node for node in model.graph.node if node.op_type == "Expand")
    assert expand.input[1] in initializer_names
    shape_init = next(initializer for initializer in model.graph.initializer if initializer.name == expand.input[1])
    assert numpy_helper.to_array(shape_init).tolist() == [2, 3]
    onnx.checker.check_model(model)


def test_staticize_known_shape_nodes_removes_fixed_cache_shape_expand_inputs() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_token2wav import (
        staticize_known_shape_nodes,
    )

    data = helper.make_tensor_value_info("cache", onnx.TensorProto.FLOAT, [2, 8, 50, 128])
    expanded = helper.make_tensor_value_info("expanded", onnx.TensorProto.FLOAT, [2, 8, 50, 128])
    nodes = [
        helper.make_node("Shape", ["cache"], ["cache_shape"]),
        helper.make_node("Expand", ["cache", "cache_shape"], ["expanded"]),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "fixed_cache_shape", [data], [expanded]),
        opset_imports=[helper.make_opsetid("", 17)],
    )

    staticize_known_shape_nodes(model)

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    expand = next(node for node in model.graph.node if node.op_type == "Expand")
    assert expand.input[1] in initializer_names
    assert not [node for node in model.graph.node if node.op_type == "Shape"]
    shape_init = next(initializer for initializer in model.graph.initializer if initializer.name == expand.input[1])
    assert numpy_helper.to_array(shape_init).tolist() == [2, 8, 50, 128]
    onnx.checker.check_model(model)


def test_remove_identity_expands_rewires_equal_static_shapes() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_token2wav import remove_identity_expands

    data = helper.make_tensor_value_info("cache", onnx.TensorProto.FLOAT, [2, 8, 50, 128])
    output = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 8, 50, 128])
    shape = numpy_helper.from_array(np.array([2, 8, 50, 128], dtype=np.int64), name="shape")
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Expand", ["cache", "shape"], ["expanded"]),
                helper.make_node("Identity", ["expanded"], ["output"]),
            ],
            "identity_expand",
            [data],
            [output],
            initializer=[shape],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )

    remove_identity_expands(model)

    assert not [node for node in model.graph.node if node.op_type == "Expand"]
    identity = next(node for node in model.graph.node if node.op_type == "Identity")
    assert list(identity.input) == ["cache"]
    onnx.checker.check_model(model)
