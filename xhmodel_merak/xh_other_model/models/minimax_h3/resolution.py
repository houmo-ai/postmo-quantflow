"""MiniMax-H3 resolution presets and transformer packing shape helpers.

Packed transformer sequence length follows the FL2VA host packing used by demos
and export YAML:

    Nv = (latent_frames + 1) * (latent_h // 2) * (latent_w // 2)
    Na = 2 * round(frames / fps * 40)
    S  = Nv + Na + text_tokens

``latent_frames = (frames - 5) // 17 * 5 + 2`` with ``frames = 17n + 5``.
Pixel height/width must be multiples of 32 so latent dims stay even after /16.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MiniMaxH3Resolution:
    name: str
    width: int
    height: int
    frames: int = 124
    fps: int = 24
    text_tokens: int = 512

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"resolution {self.name!r} requires positive width/height")
        if self.width % 32 != 0 or self.height % 32 != 0:
            raise ValueError(
                f"resolution {self.name!r} width/height must be multiples of 32, got {self.width}x{self.height}"
            )
        if self.frames % 17 != 5:
            raise ValueError(f"resolution {self.name!r} frames must have form 17*n+5, got {self.frames}")
        if self.text_tokens <= 0:
            raise ValueError(f"resolution {self.name!r} text_tokens must be positive")

    @property
    def latent_frames(self) -> int:
        return (self.frames - 5) // 17 * 5 + 2

    @property
    def latent_height(self) -> int:
        return self.height // 16

    @property
    def latent_width(self) -> int:
        return self.width // 16

    @property
    def rows_per_frame(self) -> int:
        return (self.latent_height // 2) * (self.latent_width // 2)

    @property
    def num_video_tokens(self) -> int:
        # +1 condition frame row pack used by FL2VA first-frame conditioning.
        return (self.latent_frames + 1) * self.rows_per_frame

    @property
    def num_audio_tokens(self) -> int:
        return 2 * round(self.frames / self.fps * 40)

    @property
    def packed_sequence_length(self) -> int:
        return self.num_video_tokens + self.num_audio_tokens + self.text_tokens

    def as_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "fps": self.fps,
            "text_tokens": self.text_tokens,
            "latent_frames": self.latent_frames,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "num_video_tokens": self.num_video_tokens,
            "num_audio_tokens": self.num_audio_tokens,
            "packed_sequence_length": self.packed_sequence_length,
        }


# Keep demo packing aligned with the historical default YAML (608x352).
RESOLUTION_PRESETS: dict[str, MiniMaxH3Resolution] = {
    "demo": MiniMaxH3Resolution(name="demo", width=608, height=352),
    "480p": MiniMaxH3Resolution(name="480p", width=832, height=480),
    # Official README short-side default is 768 with flexible aspect; 16:9 → 1344x768.
    "720p": MiniMaxH3Resolution(name="720p", width=1344, height=768),
    # Strict 16:9 720p that still satisfies the multiple-of-32 constraint.
    "720p_16x9": MiniMaxH3Resolution(name="720p_16x9", width=1280, height=704),
}


def get_resolution_preset(name: str) -> MiniMaxH3Resolution:
    key = str(name).strip().lower()
    if key not in RESOLUTION_PRESETS:
        supported = ", ".join(sorted(RESOLUTION_PRESETS))
        raise KeyError(f"Unknown MiniMax-H3 resolution preset {name!r}; expected one of: {supported}")
    return RESOLUTION_PRESETS[key]


def build_transformer_input_shapes(
    resolution: MiniMaxH3Resolution,
    *,
    valid_text_length: int = 280,
) -> list[dict[str, Any]]:
    """Build export ``input_shapes.transformer`` entries for a resolution."""
    nv = resolution.num_video_tokens
    na = resolution.num_audio_tokens
    text = resolution.text_tokens
    packed = resolution.packed_sequence_length
    if not 0 < valid_text_length <= text:
        raise ValueError(f"valid_text_length must be in [1, {text}], got {valid_text_length}")
    return [
        {"name": "hidden_states", "shape": [1, nv, 96], "dtype": "float16"},
        {"name": "audio_hidden_states", "shape": [1, na, 32], "dtype": "float16"},
        {"name": "encoder_hidden_states", "shape": [1, text, 5120], "dtype": "float16"},
        # Scalar denoise-eval index into the baked AdaLN / final-norm LUT.
        {"name": "step_index", "shape": [], "dtype": "int64"},
        {"name": "timestep_indices", "shape": [packed], "dtype": "int64"},
        {"name": "adaln_indices", "shape": [packed], "dtype": "int64"},
        {"name": "rotary_cos", "shape": [packed, 96], "dtype": "float16"},
        {"name": "rotary_sin", "shape": [packed, 96], "dtype": "float16"},
        {"name": "video_indices", "shape": [nv], "dtype": "int64"},
        {"name": "audio_indices", "shape": [na], "dtype": "int64"},
        {"name": "pack_gather_indices", "shape": [packed], "dtype": "int64"},
        {
            "name": "encoder_attention_mask",
            "shape": [1, 1, 1, text],
            "dtype": "float16",
            "valid_length": valid_text_length,
        },
        {"name": "packed_attention_mask", "shape": [1, 1, 1, packed], "dtype": "float16"},
    ]


__all__ = [
    "MiniMaxH3Resolution",
    "RESOLUTION_PRESETS",
    "build_transformer_input_shapes",
    "get_resolution_preset",
]
