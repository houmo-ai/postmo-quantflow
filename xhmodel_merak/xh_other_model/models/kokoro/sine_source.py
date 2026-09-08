from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .graph import SourceMergeStatic


def _prefix_mask(length: Tensor, maximum: int, *, channel_axis: bool) -> Tensor:
    positions = torch.arange(maximum, device=length.device, dtype=torch.int64)
    active = positions < length.reshape(-1)[0].to(dtype=torch.int64)
    # Do not use ``active.to(torch.float32)`` here.  ONNX would freeze that
    # cast to FP32, so a W16 activation multiplied by the mask would promote
    # the complete downstream branch back to FP32.  Where keeps the mask in
    # the normal activation precision selected by xhquant.
    mask = torch.where(
        active,
        torch.ones(maximum, device=length.device, dtype=torch.float32),
        torch.zeros(maximum, device=length.device, dtype=torch.float32),
    )
    if channel_axis:
        return mask.reshape(1, 1, maximum)
    return mask.reshape(1, maximum, 1)


class StaticSineSource(nn.Module):
    """Fixed-seed SineGen and SourceMerge without ONNX random operators."""

    SAMPLE_RATE = 24_000
    HARMONICS = 9
    F0_UPSAMPLE = 300

    def __init__(self, source_merge: SourceMergeStatic, frame_max_length: int, seed: int) -> None:
        super().__init__()
        self.source_merge = source_merge
        self.frame_max_length = int(frame_max_length)
        self.waveform_length = 600 * self.frame_max_length
        self.register_buffer(
            "harmonic_numbers",
            torch.arange(1, self.HARMONICS + 1, dtype=torch.float32).reshape(1, 1, -1),
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            initial_phase = torch.rand((1, self.HARMONICS), dtype=torch.float32)
            initial_phase[:, 0] = 0.0
            noise_layout = torch.empty(
                1,
                self.HARMONICS,
                self.waveform_length,
                dtype=torch.float32,
            ).transpose(1, 2)
            noise = torch.randn_like(noise_layout)
        self.register_buffer("initial_phase", initial_phase)
        self.register_buffer("noise", noise.contiguous())

    def upsample_f0(self, f0_prediction: Tensor) -> Tensor:
        return F.interpolate(
            f0_prediction.unsqueeze(1),
            scale_factor=self.F0_UPSAMPLE,
        ).transpose(1, 2)

    def prepare_phase_increments(self, f0_up: Tensor) -> Tensor:
        radians = torch.remainder(
            f0_up * self.harmonic_numbers / self.SAMPLE_RATE,
            1.0,
        )
        radians = torch.cat(
            [
                radians[:, :1] + self.initial_phase.unsqueeze(1),
                radians[:, 1:],
            ],
            dim=1,
        )
        return F.interpolate(
            radians.transpose(1, 2),
            scale_factor=1 / self.F0_UPSAMPLE,
            mode="linear",
        ).transpose(1, 2)

    def phase_increments_to_sine(self, phase_increments: Tensor) -> Tensor:
        phase = torch.cumsum(phase_increments.float(), dim=1) * (2 * torch.pi)
        phase = F.interpolate(
            phase.transpose(1, 2) * self.F0_UPSAMPLE,
            scale_factor=self.F0_UPSAMPLE,
            mode="linear",
        ).transpose(1, 2)
        return torch.sin(phase) * 0.1

    def merge_source(
        self,
        sine: Tensor,
        f0_up: Tensor,
        waveform_mask: Tensor,
    ) -> Tensor:
        voiced = torch.where(
            f0_up > 10,
            torch.ones_like(f0_up),
            torch.zeros_like(f0_up),
        )
        noise_amplitude = voiced * 0.003 + (1.0 - voiced) * (0.1 / 3.0)
        sine = sine * voiced + noise_amplitude * self.noise
        source = self.source_merge(sine)
        return source * waveform_mask.transpose(1, 2)

    def forward(self, f0_prediction: Tensor, waveform_mask: Tensor) -> Tensor:
        f0_up = self.upsample_f0(f0_prediction)
        phase_increments = self.prepare_phase_increments(f0_up)
        sine = self.phase_increments_to_sine(phase_increments)
        return self.merge_source(sine, f0_up, waveform_mask)
