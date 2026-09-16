from __future__ import annotations

import torch
import torch.nn.functional as F

from .minicpmo_token2wav_modules import HiftForwardWrapper, deterministic_hift_source


FLOW_CACHE_NAMES = (
    "conformer_cnn_cache",
    "conformer_att_cache",
    "estimator_cnn_cache",
    "estimator_att_cache",
)


class HiftStreamWrapper(HiftForwardWrapper):
    def __init__(
        self,
        hift_module: torch.nn.Module,
        source_cache_length: int = 3840,
        mel_cache_length: int = 8,
        waveform_length: int | None = None,
    ) -> None:
        super().__init__(hift_module, waveform_length=waveform_length)
        self.source_cache_length = int(source_cache_length)
        self.mel_cache_length = int(mel_cache_length)

    def _prepare_part1_outputs(
        self,
        speech_feat: torch.Tensor,
        speech_feat_valid_length: torch.Tensor,
        past_mel: torch.Tensor,
        past_mel_valid_length: torch.Tensor,
        past_source: torch.Tensor,
        past_source_valid_length: torch.Tensor,
        phase_noise: torch.Tensor,
        source_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_valid_frames = speech_feat_valid_length.reshape(()).to(torch.int64)
        past_valid_frames = past_mel_valid_length.reshape(()).to(torch.int64)
        current_mask = torch.arange(speech_feat.shape[2], device=speech_feat.device) < current_valid_frames
        speech_feat = speech_feat * current_mask.reshape(1, 1, -1)
        past_mask = torch.arange(self.mel_cache_length, device=speech_feat.device) < past_valid_frames
        past_mel = past_mel[:, :, -self.mel_cache_length :] * past_mask.reshape(1, 1, -1)
        total_capacity = self.mel_cache_length + speech_feat.shape[2]
        combined = torch.zeros(
            speech_feat.shape[0],
            speech_feat.shape[1],
            total_capacity,
            dtype=speech_feat.dtype,
            device=speech_feat.device,
        )
        for valid_length in range(self.mel_cache_length + 1):
            candidate = torch.cat((past_mel[:, :, :valid_length], speech_feat), dim=2)
            candidate = F.pad(candidate, (0, self.mel_cache_length - valid_length))
            selected = (past_valid_frames == valid_length).to(speech_feat.dtype)
            combined = combined + candidate * selected
        total_valid_frames = past_valid_frames + current_valid_frames
        total_mask = torch.arange(total_capacity, device=speech_feat.device) < total_valid_frames
        speech_feat = combined * total_mask.reshape(1, 1, -1)
        f0 = self.hift.f0_predictor(speech_feat)
        f0 = self.hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        sine_generator = self.hift.m_source.l_sin_gen
        sine_waves, _ = deterministic_hift_source(
            f0,
            phase_noise,
            source_noise,
            sampling_rate=int(sine_generator.sampling_rate),
            upsample_scale=int(sine_generator.upsample_scale),
            sine_amp=float(sine_generator.sine_amp),
            noise_std=float(sine_generator.noise_std),
            voiced_threshold=float(sine_generator.voiced_threshold),
        )
        source = self.hift.m_source.l_tanh(self.hift.m_source.l_linear(sine_waves)).transpose(1, 2)
        padded_past_source = F.pad(
            past_source[:, :, -self.source_cache_length :],
            (0, source.shape[2] - self.source_cache_length),
        )
        source_cache_mask = (
            torch.arange(source.shape[2], device=source.device) < past_source_valid_length.reshape(()).to(torch.int64)
        ).reshape(1, 1, -1)
        source = torch.where(source_cache_mask, padded_past_source.to(source), source)
        return speech_feat, source

    def _forward_full(
        self,
        speech_feat: torch.Tensor,
        speech_feat_valid_length: torch.Tensor,
        past_mel: torch.Tensor,
        past_mel_valid_length: torch.Tensor,
        past_source: torch.Tensor,
        past_source_valid_length: torch.Tensor,
        phase_noise: torch.Tensor,
        source_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        speech_feat, source = self._prepare_part1_outputs(
            speech_feat,
            speech_feat_valid_length,
            past_mel,
            past_mel_valid_length,
            past_source,
            past_source_valid_length,
            phase_noise,
            source_noise,
        )
        original_stft = getattr(self.hift, "_stft", None)
        original_istft = getattr(self.hift, "_istft", None)
        self.hift._stft = self._exportable_stft
        self.hift._istft = self._exportable_istft
        try:
            waveform = self.hift.decode(x=speech_feat, s=source)
        finally:
            if original_stft is not None:
                self.hift._stft = original_stft
            if original_istft is not None:
                self.hift._istft = original_istft
        source_length = source.shape[2]
        return waveform, source.reshape(1, 1, source_length)

    def forward(
        self,
        speech_feat: torch.Tensor,
        speech_feat_valid_length: torch.Tensor,
        past_mel: torch.Tensor,
        past_mel_valid_length: torch.Tensor,
        past_source: torch.Tensor,
        past_source_valid_length: torch.Tensor,
        phase_noise: torch.Tensor,
        source_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self._forward_full(
            speech_feat,
            speech_feat_valid_length,
            past_mel,
            past_mel_valid_length,
            past_source,
            past_source_valid_length,
            phase_noise,
            source_noise,
        )
        return outputs


class HiftStreamPart1Wrapper(HiftStreamWrapper):
    def forward(
        self,
        speech_feat: torch.Tensor,
        speech_feat_valid_length: torch.Tensor,
        past_mel: torch.Tensor,
        past_mel_valid_length: torch.Tensor,
        past_source: torch.Tensor,
        past_source_valid_length: torch.Tensor,
        phase_noise: torch.Tensor,
        source_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._prepare_part1_outputs(
            speech_feat,
            speech_feat_valid_length,
            past_mel,
            past_mel_valid_length,
            past_source,
            past_source_valid_length,
            phase_noise,
            source_noise,
        )


__all__ = [
    "FLOW_CACHE_NAMES",
    "HiftStreamPart1Wrapper",
    "HiftStreamWrapper",
]
