from __future__ import annotations

import io
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor


class Token2WavRuntimeProtocol(Protocol):
    up_rate: int

    def flow_inference(self, *args: Tensor | int) -> Tensor: ...

    def hift_inference(self, speech_feat: Tensor) -> Tensor: ...

    def setup_prompt_cache(
        self, token: Tensor, mel: Tensor, spk: Tensor, n_timesteps: int = 10
    ) -> Mapping[str, Tensor]: ...

    def init_stream_cache(self, base_cache: Mapping[str, Tensor], prompt_mel_length: int) -> Any: ...

    def reset_stream_cache(self) -> None: ...

    def stream_flow(
        self, tokens: Tensor, embedding: Tensor, last_chunk: bool
    ) -> tuple[Tensor, Mapping[str, Tensor]]: ...

    def stream_hift(
        self, mel: Tensor, cache_source: Tensor, last_chunk: bool | None = None
    ) -> tuple[Tensor, Tensor]: ...


class _HMONNXFlowAdapter:
    def __init__(self, runtime: Token2WavRuntimeProtocol) -> None:
        self.runtime = runtime
        self.up_rate = runtime.up_rate
        self.pre_lookahead_len = int(getattr(runtime, "pre_lookahead_len", 0))

    def inference(self, *args: Tensor | int) -> Tensor:
        reference = next((value for value in args if isinstance(value, Tensor)), None)
        device_type = reference.device.type if reference is not None else "cuda"
        with torch.autocast(device_type, enabled=False):
            return self.runtime.flow_inference(*args)

    def setup_cache(self, token: Tensor, mel: Tensor, spk: Tensor, n_timesteps: int = 10) -> Mapping[str, Tensor]:
        return self.runtime.setup_prompt_cache(token, mel, spk, n_timesteps=n_timesteps)

    def inference_chunk(self, *args: Tensor, **kwargs: object) -> tuple[Tensor, Mapping[str, Tensor]]:
        token = kwargs.get("token", args[0] if args else None)
        embedding = kwargs.get("spk", args[1] if len(args) > 1 else None)
        last_chunk = kwargs.get("last_chunk", False)
        if not isinstance(token, Tensor) or not isinstance(embedding, Tensor) or not isinstance(last_chunk, bool):
            raise RuntimeError("official Flow streaming arguments are incomplete")
        with torch.autocast(token.device.type, enabled=False):
            return self.runtime.stream_flow(token, embedding, last_chunk)


class _HMONNXHiFTAdapter:
    def __init__(self, runtime: Token2WavRuntimeProtocol) -> None:
        self.runtime = runtime

    def __call__(self, speech_feat: Tensor, cache_source: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        if cache_source is None:
            with torch.autocast(speech_feat.device.type, enabled=False):
                return self.runtime.hift_inference(speech_feat).float(), None
        stream_hift = getattr(self.runtime, "stream_hift", None)
        if stream_hift is None:
            raise RuntimeError("Token2Wav stream_hift role is unavailable after HMONNX attachment")
        with torch.autocast(speech_feat.device.type, enabled=False):
            return stream_hift(speech_feat, cache_source)


class _HMONNXCampplusAdapter:
    def __init__(self, runtime: Token2WavRuntimeProtocol, *, sequence_length: int) -> None:
        self._runtime = runtime
        self._sequence_length = sequence_length
        self._input_name = "input"

    def get_inputs(self) -> list[Any]:
        return [type("_In", (), {"name": self._input_name})()]

    def run(self, _output_names: Any, feed: dict[str, Any]) -> list[Any]:
        session = getattr(self._runtime, "campplus_session", None)
        if session is None:
            raise RuntimeError("HMONNX campplus session has been released")
        feat = torch.as_tensor(feed[self._input_name])
        if feat.shape[1] > self._sequence_length:
            feat = feat[:, : self._sequence_length]
        elif feat.shape[1] < self._sequence_length:
            feat = torch.nn.functional.pad(feat, (0, 0, 0, self._sequence_length - feat.shape[1]))
        return [torch.as_tensor(session(feat.to(torch.float16))).float().cpu().numpy()]


class _HMONNXSpeechTokenizerAdapter:
    def __init__(self, runtime: Token2WavRuntimeProtocol, *, feats_length: int, n_mels: int) -> None:
        self._runtime = runtime
        self._feats_length = feats_length
        self._n_mels = n_mels

    def quantize(self, mel: Tensor, mel_len: Tensor) -> tuple[Tensor, Tensor]:
        session = getattr(self._runtime, "speech_tokenizer_session", None)
        if session is None:
            raise RuntimeError("HMONNX speech tokenizer session has been released")
        mel = torch.as_tensor(mel)
        batch, n_mels, frames = mel.shape
        if batch != 1 or n_mels != self._n_mels:
            raise RuntimeError(f"HMONNX speech_tokenizer requires [1, {self._n_mels}, frames], got {tuple(mel.shape)}")
        valid = int(torch.as_tensor(mel_len).reshape(-1)[0].item())
        if frames > self._feats_length:
            mel = mel[:, :, : self._feats_length]
        elif frames < self._feats_length:
            mel = torch.nn.functional.pad(mel, (0, self._feats_length - frames, 0, 0, 0, 0))
        capacity = torch.tensor([self._feats_length], dtype=torch.int32, device=mel.device)
        tokens = torch.as_tensor(session(mel.to(torch.float16), capacity))
        token_len = _official_token_length(valid)
        return tokens[:, :token_len], torch.tensor([token_len], dtype=torch.int64, device=mel.device)


def _official_token_length(feats_length: int) -> int:
    x = (feats_length - 3) // 2
    x = int(x) + 1
    x = x - 3 + 2
    return int((x // 2) + 1)


def install_hmonnx_token2wav(audio_tokenizer: SimpleNamespace, runtime: Token2WavRuntimeProtocol) -> None:
    audio_tokenizer.flow = _HMONNXFlowAdapter(runtime)
    audio_tokenizer.hift = _HMONNXHiFTAdapter(runtime)
    audio_tokenizer.hift.inference_chunk = audio_tokenizer.hift
    if getattr(runtime, "campplus_session", None) is not None:
        audio_tokenizer.spk_model = _HMONNXCampplusAdapter(
            runtime, sequence_length=getattr(runtime, "campplus_sequence_length", 1000)
        )
    if getattr(runtime, "speech_tokenizer_session", None) is not None:
        audio_tokenizer.audio_tokenizer = _HMONNXSpeechTokenizerAdapter(
            runtime, feats_length=getattr(runtime, "speech_tokenizer_feats_length", 3000), n_mels=128
        )
    audio_tokenizer._xh_hmonnx_streaming_attached = True
    report = getattr(runtime, "report", None)
    if report is not None:
        report.streaming_attached = True


class HMONNXToken2Wav:
    """Official Token2Wav call surface without loading native Flow or HiFT."""

    def __init__(self, runtime: Token2WavRuntimeProtocol) -> None:
        self.runtime = runtime
        self.flow = _HMONNXFlowAdapter(runtime)
        self.hift = _HMONNXHiFTAdapter(runtime)
        self.spk_model = _HMONNXCampplusAdapter(runtime, sequence_length=runtime.campplus_sequence_length)
        self.audio_tokenizer = _HMONNXSpeechTokenizerAdapter(
            runtime, feats_length=runtime.speech_tokenizer_feats_length, n_mels=128
        )
        self.n_timesteps = runtime.n_timesteps
        self.cache = None
        self._stream_cache = None
        self.hift_cache_dict = {}
        self._xh_hmonnx_streaming_attached = True
        runtime.report.streaming_attached = True

    @property
    def device(self) -> torch.device:
        return self.runtime._noise_device

    @property
    def stream_cache(self):
        return self._stream_cache

    @stream_cache.setter
    def stream_cache(self, cache):
        self._stream_cache = cache
        if cache is not None:
            self.runtime.reset_stream_cache()

    def _prepare_prompt(self, prompt_wav: str):
        import s3tokenizer
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi
        from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram

        audio = s3tokenizer.load_audio(prompt_wav, sr=16000)
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        prompt_tokens, prompt_token_lengths = self.audio_tokenizer.quantize(
            mels.to(self.device), mels_lens.to(self.device)
        )
        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk = torch.as_tensor(
            self.spk_model.run(None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(0).numpy()})[0],
            device=self.device,
        )
        audio_24k, sample_rate = torchaudio.load(prompt_wav, backend="soundfile")
        audio_24k = audio_24k.mean(dim=0, keepdim=True)
        if sample_rate != 24000:
            audio_24k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio_24k)
        prompt_mel = mel_spectrogram(audio_24k).transpose(1, 2).to(self.device)
        prompt_mel_lengths = torch.tensor([prompt_mel.shape[1]], dtype=torch.int32, device=self.device)
        prompt_mel = torch.nn.functional.pad(
            prompt_mel,
            (0, 0, 0, prompt_tokens.shape[1] * self.flow.up_rate - prompt_mel.shape[1]),
            mode="replicate",
        )
        return prompt_tokens, prompt_token_lengths, spk, prompt_mel, prompt_mel_lengths

    def set_stream_cache(self, prompt_wav: str):
        self.cache = self._prepare_prompt(prompt_wav)
        tokens, _, spk, mel, _ = self.cache
        silence = torch.full((1, self.flow.pre_lookahead_len), 4218, dtype=tokens.dtype, device=self.device)
        base_cache = self.flow.setup_cache(torch.cat((tokens, silence), dim=1), mel, spk, self.n_timesteps)
        hift_cache = {
            "mel": torch.zeros((1, 80, 0), device=self.device),
            "source": torch.zeros((1, 1, 0), device=self.device),
            "speech": torch.zeros((1, 0), device=self.device),
        }
        return base_cache, hift_cache

    def __call__(self, generated_speech_tokens: list[int], prompt_wav: str) -> bytes:
        import torchaudio

        if self.cache is None:
            self.cache = self._prepare_prompt(prompt_wav)
        prompt_tokens, prompt_lengths, spk, prompt_mel, prompt_mel_lengths = self.cache
        tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device=self.device)
        lengths = torch.tensor([tokens.shape[1]], dtype=torch.int32, device=self.device)
        mel = self.runtime.flow_inference(
            tokens, lengths, prompt_tokens, prompt_lengths, prompt_mel, prompt_mel_lengths, spk, self.n_timesteps
        )
        wav = self.runtime.hift_inference(mel)
        output = io.BytesIO()
        torchaudio.save(output, wav.float().cpu(), sample_rate=24000, format="wav")
        return output.getvalue()

    def stream(
        self, generated_speech_tokens: list[int], prompt_wav: str | None,
        last_chunk: bool = False, return_waveform: bool = False,
    ) -> bytes | np.ndarray:
        if self.cache is None:
            self.cache = self._prepare_prompt(prompt_wav)
        spk = self.cache[2]
        tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device=self.device)
        waveform = self.runtime.stream(tokens, spk, last_chunk).float().cpu().numpy()
        if return_waveform:
            return waveform
        return (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


__all__ = [
    "HMONNXToken2Wav",
    "Token2WavRuntimeProtocol",
    "_HMONNXCampplusAdapter",
    "_HMONNXFlowAdapter",
    "_HMONNXHiFTAdapter",
    "_HMONNXSpeechTokenizerAdapter",
    "install_hmonnx_token2wav",
]
