# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native scheduler parity, including reference preservation and actual NFE."""

import pytest
import torch
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

from xhmodel_merak.xh_other_model.models.sol_h3.host import conditioned_h3_updates
from xhmodel_merak.xh_other_model.models.sol_h3.native_schedule import native_schedule


@pytest.mark.parametrize("steps,shift", [(50, 12.0), (8, 6.0)])
def test_native_schedule_and_reference_solver(steps, shift):
    schedule = native_schedule(steps=steps, video_shift=shift)
    vs, aus = MiniMaxH3Scheduler(shift=shift), MiniMaxH3Scheduler(shift=3.0)
    for scheduler in (vs, aus):
        scheduler.set_timesteps(steps + 1)
    torch.testing.assert_close(torch.tensor(schedule.video_sigmas), vs.sigmas, rtol=0, atol=0)
    torch.testing.assert_close(torch.tensor(schedule.audio_sigmas), aus.sigmas, rtol=0, atol=0)
    video, audio = torch.randn(1, 5, 4), torch.randn(1, 4, 4)
    expected_v, expected_a = video.clone(), audio.clone()
    calls = []

    def invoke(v, a, text, step, *args):
        torch.testing.assert_close(v[:, :1], video[:, :1], rtol=0, atol=0)
        torch.testing.assert_close(a[:, :1], audio[:, :1], rtol=0, atol=0)
        calls.append(step.item())
        return v * 0.1, a * 0.2

    for i in range(steps):
        expected_v[:, 1:] = vs.step(expected_v[:, 1:] * 0.1, vs.timesteps[i], expected_v[:, 1:]).prev_sample
        expected_a[:, 1:] = aus.step(expected_a[:, 1:] * 0.2, aus.timesteps[i], expected_a[:, 1:]).prev_sample
    actual_v, actual_a = conditioned_h3_updates(
        invoke,
        video,
        audio,
        torch.zeros(1, 2, 4),
        None,
        None,
        2,
        torch.zeros(11),
        condition_video_rows=1,
        condition_audio_rows=1,
        schedule=schedule,
    )
    assert calls == list(range(steps))
    torch.testing.assert_close(actual_v, expected_v[:, 1:], rtol=0, atol=0)
    torch.testing.assert_close(actual_a, expected_a[:, 1:], rtol=0, atol=0)


@pytest.mark.parametrize("options", [{"steps": 0}, {"steps": True}, {"video_shift": float("nan")}, {"audio_shift": 0}])
def test_invalid_native_schedule(options):
    with pytest.raises(ValueError):
        native_schedule(**options)
