# SPDX-License-Identifier: Apache-2.0
"""Register the native floating-point WeMM model with vLLM."""

from __future__ import annotations

import os


_WEMM_MODEL = "vllm_merak.model_executor.model_loader.wemm_embedding_native:WeMMEmbedding"


def _native_wemm_selected() -> bool:
    """Whether the caller explicitly selected the native WeMM plugin.

    vLLM loads every installed general plugin when ``VLLM_PLUGINS`` is not
    set.  In that default mode the plugin only publishes the dedicated
    ``WeMMEmbedding`` architecture and leaves ordinary Qwen3.5 checkpoints
    alone.  Explicit ``VLLM_PLUGINS=wemm`` additionally maps the source
    checkpoint architecture to WeMM so the official, unmodified WeMM
    ``config.json`` can be loaded without an ``hf_overrides`` argument.
    """

    selected = os.environ.get("VLLM_PLUGINS")
    if selected is None:
        return False
    return "wemm" in {name.strip() for name in selected.split(",")}


def register() -> None:
    """Register WeMM lazily without importing model or CUDA code."""

    from vllm import ModelRegistry

    ModelRegistry.register_model("WeMMEmbedding", _WEMM_MODEL)
    if _native_wemm_selected():
        ModelRegistry.register_model("Qwen3_5ForConditionalGeneration", _WEMM_MODEL)


__all__ = ["register"]
