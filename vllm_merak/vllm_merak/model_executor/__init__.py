from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, cast

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_MERAK_BASE_ARCHITECTURE = "MerakForCausalLM"


def _coerce_architecture_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(architecture) for architecture in value if architecture]
    return []


def _get_architecture_candidates(hf_config: Any) -> tuple[str, list[str]]:
    loader_architectures = _coerce_architecture_list(
        getattr(hf_config, "merak_loader_architecture", None)
    )
    if loader_architectures:
        return "merak_loader_architecture", loader_architectures

    source_architectures = _coerce_architecture_list(
        getattr(hf_config, "source_architectures", None)
    )
    if source_architectures:
        return "source_architectures", source_architectures

    architectures = _coerce_architecture_list(getattr(hf_config, "architectures", None))
    if architectures:
        return "architectures", architectures

    text_config = getattr(hf_config, "text_config", None)
    text_architectures = _coerce_architecture_list(
        getattr(text_config, "architectures", None)
    )
    if text_architectures:
        return "text_config.architectures", text_architectures

    return "", []


def get_model(
    vllm_config: "VllmConfig",
    kv_cache_dtype: str,
    **kwargs,
):
    from vllm.config import set_current_vllm_config

    from .register import get_registered_custom_architectures, get_registered_model_cls

    model_config = vllm_config.model_config
    hf_config = model_config.hf_config

    architecture_field, architecture_candidates = _get_architecture_candidates(
        hf_config
    )
    if not architecture_candidates:
        raise ValueError(
            "Merak model config is missing architecture metadata. "
            "Expected 'source_architectures', 'architectures', or "
            "'text_config.architectures'. Check config_format='merak_llm' "
            "and <model>/hf_config/config.json."
        )

    merak_model_classes = get_registered_custom_architectures()
    architecture = architecture_candidates[0]
    model_cls = None
    for candidate in architecture_candidates:
        if candidate == _MERAK_BASE_ARCHITECTURE:
            continue
        model_cls = get_registered_model_cls(candidate) or merak_model_classes.get(
            candidate
        )
        if model_cls is not None:
            architecture = candidate
            break

    logger.info(
        "Detected architecture '%s' from MerakConfig.%s.",
        architecture,
        architecture_field,
    )
    if model_cls is None:
        if architecture_candidates == [_MERAK_BASE_ARCHITECTURE]:
            raise ValueError(
                "Merak model config only contains the generic 'MerakForCausalLM' architecture. "
                "Expected source HF architecture metadata in <model>/hf_config/config.json "
                "or a concrete Merak loader class in hf_config.architectures."
            )
        raise ValueError(
            f"Unsupported architectures {architecture_candidates} for Merak backend."
        )

    logger.info(
        "Using Merak model class '%s' for architecture '%s'.",
        model_cls.__name__,
        architecture,
    )
    with cast(AbstractContextManager[None], set_current_vllm_config(vllm_config)):
        return model_cls(
            vllm_config=vllm_config, kv_cache_dtype=kv_cache_dtype, **kwargs
        )


__all__ = [
    "get_model",
]
