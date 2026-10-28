"""Qwen3-VL pairwise relevance scoring using the existing VL backbone."""

from .model import XHQwen3VLRerankerModel, XHQwen3VLRerankerModelConfig


__all__ = ["XHQwen3VLRerankerModel", "XHQwen3VLRerankerModelConfig"]
