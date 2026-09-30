# SPDX-License-Identifier: Apache-2.0
"""Shared Hugging Face asset loading contract for Merak model loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from transformers import AutoConfig, PretrainedConfig
from vllm.config import ModelConfig

_T = TypeVar("_T")


def resolve_hf_assets_dir(model_id: str | Path) -> Path:
    """Resolve the copied source-HF assets for a Merak export."""

    model_dir = Path(model_id)
    hf_assets_dir = model_dir / "hf_config"
    return hf_assets_dir if hf_assets_dir.is_dir() else model_dir


def get_model_trust_remote_code(model_config: ModelConfig) -> bool:
    """Return the outer vLLM remote-code policy without changing its meaning."""

    trust_remote_code = model_config.trust_remote_code
    if type(trust_remote_code) is not bool:
        raise TypeError(
            "ModelConfig.trust_remote_code must be bool, got "
            f"{type(trust_remote_code).__name__}"
        )
    return trust_remote_code


def load_hf_pretrained(
    loader: Any,
    model_id: str | Path,
    *,
    trust_remote_code: bool,
    **kwargs: Any,
) -> _T:
    """Load one HF asset while preserving the caller's remote-code policy.

    ``loader`` can be ``AutoConfig`` or a concrete Processor/Tokenizer class.
    Keeping this as the only ``from_pretrained`` boundary prevents nested
    config/tokenizer loads from silently hard-coding or dropping the outer
    vLLM ``trust_remote_code`` setting.
    """

    if type(trust_remote_code) is not bool:
        raise TypeError(
            f"trust_remote_code must be bool, got {type(trust_remote_code).__name__}"
        )
    return cast(
        _T,
        loader.from_pretrained(
            str(resolve_hf_assets_dir(model_id)),
            trust_remote_code=trust_remote_code,
            **kwargs,
        ),
    )


def load_hf_config(
    model_id: str | Path,
    *,
    trust_remote_code: bool,
    **kwargs: Any,
) -> PretrainedConfig:
    """Load a source HF config through the shared policy boundary."""

    return load_hf_pretrained(
        AutoConfig,
        model_id,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


__all__ = [
    "get_model_trust_remote_code",
    "load_hf_config",
    "load_hf_pretrained",
    "resolve_hf_assets_dir",
]
