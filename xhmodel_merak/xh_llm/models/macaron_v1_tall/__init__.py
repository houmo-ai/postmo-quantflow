"""Macaron model definitions built on the Qwen3.5-MoE implementation."""

from .macaron_v1_tall_hmonnx_inference import (
    XHMacaronV1TallHMONNXModel,
)
from .macaron_v1_tall_model import XHMacaronV1TallModel


__all__ = [
    "XHMacaronV1TallHMONNXModel",
    "XHMacaronV1TallModel",
]
