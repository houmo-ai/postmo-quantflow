from __future__ import annotations

import torch
import torch.nn.functional as F


FLOW_CACHE_NAMES = (
    "conformer_cnn_cache",
    "conformer_att_cache",
    "estimator_cnn_cache",
    "estimator_att_cache",
)


class HiftStreamPart1Wrapper(torch.nn.Module):
    def __init__(
        self,
        hift_module: torch.nn.Module,
        mel_cache_length: int = 8,
    ) -> None:
        super().__init__()
        self.hift = hift_module
        self.mel_cache_length = int(mel_cache_length)

    def _prepare_part1_outputs(
        self,
        speech_feat: torch.Tensor,
        speech_feat_valid_length: torch.Tensor,
        past_mel: torch.Tensor,
        past_mel_valid_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_valid_frames = speech_feat_valid_length.to(torch.int64)
        past_valid_frames = past_mel_valid_length.to(torch.int64)
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
        return speech_feat, f0

    def forward(
        self,
        speech_feat: torch.Tensor,
        speech_feat_valid_length: torch.Tensor,
        past_mel: torch.Tensor,
        past_mel_valid_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._prepare_part1_outputs(
            speech_feat,
            speech_feat_valid_length,
            past_mel,
            past_mel_valid_length,
        )


__all__ = [
    "FLOW_CACHE_NAMES",
    "HiftStreamPart1Wrapper",
]
