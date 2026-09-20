"""Small-graph ABI for one MOSS Encoder component Audio window."""

import json
from inspect import signature
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
import yaml

from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.encoder import (
    ENCODER_AUDIO_TOKENS,
    ENCODER_MEL_BINS,
    ENCODER_MEL_LENGTH,
    LLM_HIDDEN_SIZE,
    build_tiny_encoder_component,
    export_encoder_component,
    slice_encoder_window,
    build_encoder_component,
    load_encoder_component,
    time_merge,
)
from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.processing import (
    SAMPLING_RATE,
    audio_token_length,
    extract_moss_log_mel,
)


_WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / "configs_merak/workflows/xh2a/llm_models/moss_transcribe_diarize/0_9b/moss_transcribe_diarize.yaml"
)


def test_encoder_window_graph_emits_fixed_audio_token_window():
    encoder = build_tiny_encoder_component()
    input_features = torch.zeros(1, ENCODER_MEL_BINS, ENCODER_MEL_LENGTH)

    audio_embeds = encoder(input_features)

    assert tuple(signature(encoder.forward).parameters) == ("input_features",)
    assert tuple(audio_embeds.shape) == (1, ENCODER_AUDIO_TOKENS, LLM_HIDDEN_SIZE)


def test_tiny_encoder_uses_official_checkpoint_key_names():
    encoder = build_tiny_encoder_component()
    keys = encoder.state_dict()
    assert "whisper_encoder.layers.0.self_attn.q_proj.weight" in keys
    assert "whisper_encoder.layers.0.self_attn_layer_norm.weight" in keys
    assert "whisper_encoder.layer_norm.weight" in keys
    assert "vq_adaptor.layers.2.weight" in keys
    assert not any("attn.q.weight" in key for key in keys)
    assert not any(key.startswith("whisper.blocks") for key in keys)


def test_moss_processor_returns_fixed_log_mel_and_valid_token_length(tmp_path: Path):
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "chunk_length": 30,
                "feature_extractor_type": "WhisperFeatureExtractor",
                "feature_size": 80,
                "hop_length": 160,
                "n_fft": 400,
                "n_samples": 480000,
                "nb_max_frames": 3000,
                "padding_side": "right",
                "padding_value": 0.0,
                "return_attention_mask": False,
                "sampling_rate": 16000,
            }
        ),
        encoding="utf-8",
    )
    mel, token_len = extract_moss_log_mel(
        np.zeros(SAMPLING_RATE, dtype=np.float32),
        tmp_path,
    )

    assert mel.shape == (ENCODER_MEL_BINS, ENCODER_MEL_LENGTH)
    assert mel.dtype.name == "float32"
    assert token_len == audio_token_length(SAMPLING_RATE)


def test_encoder_window_onnx_has_no_feature_lens(tmp_path: Path):
    encoder = build_tiny_encoder_component()

    meta = export_encoder_component(
        encoder,
        tmp_path,
        target_device="XH2a",
        quant_type="w8a8_sefp",
        convert_hmonnx=False,
    )
    proto = onnx.load(tmp_path / meta["onnx_file"])
    input_names = [value.name for value in proto.graph.input]
    output_names = [value.name for value in proto.graph.output]
    input_shape = tuple(d.dim_value for d in proto.graph.input[0].type.tensor_type.shape.dim)
    output_shape = tuple(d.dim_value for d in proto.graph.output[0].type.tensor_type.shape.dim)

    assert input_names == ["input_features"]
    assert output_names == ["audio_embeds"]
    assert input_shape == (1, ENCODER_MEL_BINS, ENCODER_MEL_LENGTH)
    assert output_shape == (1, ENCODER_AUDIO_TOKENS, LLM_HIDDEN_SIZE)
    assert "feature_lens" not in input_names
    assert meta["quant_type"] == "w8a8_sefp"
    assert meta["target_device"] == "XH2a"
    assert "onnx_file" in meta
    assert "hmonnx_file" not in meta
    assert not (tmp_path / "encoder_window.onnx").exists()
    assert not (tmp_path / "hmonnx").exists()


def test_short_audio_is_sliced_after_fixed_encoder_window():
    encoder = build_tiny_encoder_component()
    window = encoder(torch.zeros(1, ENCODER_MEL_BINS, ENCODER_MEL_LENGTH))
    token_len = audio_token_length(SAMPLING_RATE)

    sliced = slice_encoder_window(window, token_len)

    assert tuple(window.shape) == (1, ENCODER_AUDIO_TOKENS, LLM_HIDDEN_SIZE)
    assert token_len < ENCODER_AUDIO_TOKENS
    assert tuple(sliced.shape) == (1, token_len, LLM_HIDDEN_SIZE)


def test_encoder_hmonnx_uses_llm_phase_layout(tmp_path: Path, monkeypatch):
    import xhmodel_merak.xh_llm.models.moss_transcribe_diarize.encoder as encoder_mod

    def fake_convert(onnx_file, dummy, hmonnx_file, target_device, quant_type):
        del dummy, target_device, quant_type
        assert onnx_file.is_file()
        hmonnx_file.parent.mkdir(parents=True, exist_ok=True)
        hmonnx_file.write_bytes(b"hmonnx")
        hmonnx_file.with_name(f"{hmonnx_file.stem}_external_data").write_bytes(b"weights")

    monkeypatch.setattr(encoder_mod, "_convert_encoder_hmonnx", fake_convert)
    meta = export_encoder_component(
        build_tiny_encoder_component(),
        tmp_path,
        model_name="moss_transcribe_diarize_09b_encode",
        convert_hmonnx=True,
    )

    hmonnx_name = "hmquant_moss_transcribe_diarize_09b_encode_with_act.onnx"
    assert meta["hmonnx_file"] == hmonnx_name
    assert "onnx_file" not in meta
    assert (tmp_path / hmonnx_name).is_file()
    assert (tmp_path / "hmquant_moss_transcribe_diarize_09b_encode_external_data").is_file()
    leftover = sorted(
        path.name for path in tmp_path.rglob("*") if path.is_file() and path.name != hmonnx_name
    )
    assert leftover == ["hmquant_moss_transcribe_diarize_09b_encode_external_data"]
    assert not (tmp_path / "encoder_window.onnx").exists()
    assert not (tmp_path / "hmonnx").exists()


def test_encode_model_config_appends_phase_suffix():
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.configuration import (
        XHMossTranscribeDiarizeConfig,
    )

    cfg = XHMossTranscribeDiarizeConfig(model_name="moss_transcribe_diarize_09b")
    assert cfg.encode_model_config().model_name == "moss_transcribe_diarize_09b_encode"


def test_encoder_model_passes_serving_model_name(tmp_path: Path, monkeypatch):
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.configuration import XHMossEncoderConfig
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.encoder_model import XHMossTranscribeDiarizeEncoder

    captured: dict[str, object] = {}

    def fake_export(component, output_dir, **kwargs):
        del component
        captured.update(kwargs)
        captured["output_dir"] = output_dir
        return {"hmonnx_file": "x.onnx"}

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.moss_transcribe_diarize.encoder_model.export_encoder_component",
        fake_export,
    )
    encoder = XHMossTranscribeDiarizeEncoder(
        XHMossEncoderConfig(
            model_name="moss_transcribe_diarize_09b_encode",
            hf_model=build_tiny_encoder_component(),
        )
    )
    encoder.export_hmonnx(str(tmp_path))
    assert captured["model_name"] == "moss_transcribe_diarize_09b_encode"
    assert captured["convert_hmonnx"] is True


def test_encoder_export_rejects_non_xh2a_or_non_w8a8(tmp_path: Path):
    encoder = build_tiny_encoder_component()

    with pytest.raises(ValueError, match="XH2a"):
        export_encoder_component(
            encoder, tmp_path, target_device="XH3", quant_type="w8a8_sefp", convert_hmonnx=False
        )
    with pytest.raises(ValueError, match="w8a8_sefp"):
        export_encoder_component(
            encoder, tmp_path, target_device="XH2a", quant_type="w16a16_sefp", convert_hmonnx=False
        )


def test_checked_in_encoder_component_is_xh2a_w8a8_sefp():
    data = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))

    assert data["export"]["model"]["chip_arch"] == "XH2a"
    assert data["export"]["model"]["encode_config"]["quant_type"] == "w8a8_sefp"
    assert data["export"]["model"]["encode_config"]["max_audio_length"] == ENCODER_MEL_LENGTH


@pytest.mark.parametrize("valid_tokens", [(3,), (3, 1), (3, 2), (3, 3, 1)])
def test_per_window_adaptor_matches_recording_level_merge(valid_tokens):
    """Keep real attention/merge/adaptor, shrink only encoder dimensions."""
    torch.manual_seed(83)
    encoder = build_encoder_component(
        n_state=8, n_head=2, n_layer=1, n_mels=4, n_ctx=12, hidden_size=8,
    )
    mel = torch.randn(len(valid_tokens), 4, 24)
    with torch.no_grad():
        whisper = encoder.whisper_encoder(mel)
        official_order = encoder.vq_adaptor(time_merge(torch.cat([
            whisper[i:i + 1, :n * 4] for i, n in enumerate(valid_tokens)
        ], dim=1)))
        window_order = torch.cat([
            encoder(mel[i:i + 1])[:, :n] for i, n in enumerate(valid_tokens)
        ], dim=1)
    torch.testing.assert_close(window_order, official_order, rtol=1e-5, atol=1e-6)


def test_encoder_export_casts_bf16_whisper_to_float32(tmp_path: Path):
    encoder = build_tiny_encoder_component().to(dtype=torch.bfloat16)
    meta = export_encoder_component(
        encoder, tmp_path / "enc", target_device="XH2a", quant_type="w8a8_sefp", convert_hmonnx=False
    )
    assert (tmp_path / "enc" / meta["onnx_file"]).is_file()


def test_encoder_component_unwraps_hf_whisper_output():
    from transformers.models.whisper.configuration_whisper import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.encoder import (
        EncoderComponent,
        MossVQAdaptor,
        encoder_hidden_states,
    )

    cfg = WhisperConfig(
        num_mel_bins=8,
        d_model=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        max_source_positions=8,
        vocab_size=32,
    )
    whisper = WhisperEncoder(cfg).eval()
    adaptor = MossVQAdaptor(16 * 4, 8)
    encoder = EncoderComponent(whisper, adaptor, merge_size=4).eval()
    mel = torch.zeros(1, 8, 16)
    hidden = whisper(mel)
    assert not isinstance(hidden, torch.Tensor)
    with torch.no_grad():
        actual = encoder(mel)
        expected = adaptor(time_merge(encoder_hidden_states(hidden), 4))
    torch.testing.assert_close(actual, expected)


def test_encoder_load_requires_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        load_encoder_component(tmp_path)
