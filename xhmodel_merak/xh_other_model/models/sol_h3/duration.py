# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Static Sol-H3 temporal gears and their H3/LTX latent dimensions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DurationGear:
    seconds: int
    draft_frames: int
    refiner_frames: int
    fps: int = 24

    @property
    def h3_video_frames(self):
        return (self.draft_frames - 5) // 17 * 5 + 2

    @property
    def h3_audio_frames(self):
        return round(self.draft_frames / self.fps * 40)

    @property
    def ltx_video_frames(self):
        return (self.refiner_frames - 1) // 8 + 1

    @property
    def ltx_audio_frames(self):
        return round(self.refiner_frames / self.fps * 25)


GEARS = {
    2: DurationGear(2, 56, 49),
    5: DurationGear(5, 124, 121),
    15: DurationGear(15, 362, 361),
}


def gear_for_draft(draft_frames):
    for gear in GEARS.values():
        if draft_frames == gear.draft_frames:
            return gear
    raise ValueError(f"unsupported Sol-H3 H3 frame count: {draft_frames}")


def duration_gear(draft_frames, refiner_frames, fps=24):
    """Resolve an explicitly supported pair; never infer a gear by truncation."""
    for gear in GEARS.values():
        if (draft_frames, refiner_frames, fps) == (gear.draft_frames, gear.refiner_frames, gear.fps):
            return gear
    raise ValueError(f"unsupported Sol-H3 temporal gear: {(draft_frames, refiner_frames, fps)}")
