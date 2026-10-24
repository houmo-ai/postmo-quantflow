# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native H3 scheduler presets expressed as actual transformer evaluations."""

import math

from .conditioning import ConditionedH3Schedule


def native_schedule(*, steps=50, video_shift=12.0, audio_shift=3.0):
    """Preserve the native FP32 grids; its terminal zero is not an evaluation."""
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError("H3 steps must be an integer between 1 and 1000")
    grids = []
    for shift in (video_shift, audio_shift):
        if type(shift) not in (int, float) or not math.isfinite(shift) or shift <= 0:
            raise ValueError("H3 flow shifts must be finite positive numbers")
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(steps + 1)
        if len(scheduler.timesteps) != steps:
            raise ValueError("H3 flow shift collapses FP32 schedule points")
        grids.append(tuple(scheduler.sigmas.tolist()))
    return ConditionedH3Schedule(*grids)
