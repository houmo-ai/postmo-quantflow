"""Compatibility helpers for MiniCPM-V-4.7 Transformers processors."""

from __future__ import annotations

from typing import Any


def ensure_minicpm_v47_processor_compat() -> None:
    """Patch missing Transformers symbols required by MiniCPM remote video code.

    Current MiniCPM-V-4.7 remote ``video_processing_minicpmv4_7.py`` imports
    ``BASE_VIDEO_PROCESSOR_DOCSTRING`` from ``transformers.video_processing_utils``.
    Some Transformers builds no longer export that symbol. Provide an empty
    docstring so ``AutoProcessor.from_pretrained`` can import the remote module.
    """

    import transformers.video_processing_utils as video_processing_utils

    if not hasattr(video_processing_utils, "BASE_VIDEO_PROCESSOR_DOCSTRING"):
        video_processing_utils.BASE_VIDEO_PROCESSOR_DOCSTRING = ""


def load_minicpm_v47_processor(model_dir: str | Any, **kwargs: Any):
    """Load MiniCPM-V-4.7 ``AutoProcessor`` with the compatibility patch applied."""

    from transformers import AutoProcessor

    ensure_minicpm_v47_processor_compat()
    kwargs.setdefault("trust_remote_code", True)
    return AutoProcessor.from_pretrained(model_dir, **kwargs)


__all__ = [
    "ensure_minicpm_v47_processor_compat",
    "load_minicpm_v47_processor",
]
