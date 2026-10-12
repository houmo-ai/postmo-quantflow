import importlib
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch

# NOTE: Do NOT import ``.config`` / ``.hmfp_kv_cache`` / ``.model_executor.register``
# at module top level. This module is the ``vllm.platform_plugins`` entry point,
# so vLLM imports it *during* ``import vllm`` itself (``env_override`` ->
# ``resolve_current_platform_cls_qualname`` -> plugin discovery). At that moment
# ``vllm.v1.*`` and ``vllm.utils.torch_utils`` are only partially initialised, so
# pulling in ``.hmfp_kv_cache`` (which imports ``vllm.v1.kv_cache_interface``) or
# ``.config`` (which re-enters this same ``import vllm``) raises a circular-import
# ``ImportError`` and kills the whole Merak plugin, silently falling back to a
# default model. These symbols are only needed inside functions/methods that run
# later (general_plugins phase / runtime), so they are imported lazily there.
from .constants import (
    HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST,
    SUPPORTED_HYBRID_CACHE_POLICIES,
    MERAK_KV_CACHE_FORMAT,
)


def register() -> str:
    """Register the Merak platform plugin for vLLM.

    Keep this function before any vLLM imports.  vLLM discovers the
    ``vllm.platform_plugins`` entry point while it is importing itself; if this
    module re-enters vLLM before ``register`` exists, plugin loading sees a
    partially initialised module and fails with ``AttributeError``.
    """
    return "vllm_merak.platform.MerakPlatform"


logger = logging.getLogger(__name__)


def _coerce_optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _get_config_value(config: object, name: str) -> object:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _resolve_hmbf16_cache_dtype(vllm_config) -> str:
    k_man_bit = None
    v_man_bit = None

    for env_name, target in (
        ("VLLM_MERAK_KCACHE_MAN_BIT", "k"),
        ("VLLM_MERAK_VCACHE_MAN_BIT", "v"),
    ):
        man_bit = _coerce_optional_int(os.environ.get(env_name))
        if man_bit is not None:
            if target == "k":
                k_man_bit = man_bit
            else:
                v_man_bit = man_bit

    if k_man_bit is None and v_man_bit is None:
        fallback_man_bit = _coerce_optional_int(os.environ.get("VLLM_MERAK_HMBF16_MAN_BIT"))
        if fallback_man_bit is not None:
            k_man_bit = fallback_man_bit
            v_man_bit = fallback_man_bit

    if k_man_bit is None or v_man_bit is None:
        hf_config = getattr(vllm_config.model_config, "hf_config", None)
        if hf_config is not None:
            k_man_bit = _coerce_optional_int(getattr(hf_config, "kcache_man_bit", None))
            v_man_bit = _coerce_optional_int(getattr(hf_config, "vcache_man_bit", None))
            if k_man_bit is None or v_man_bit is None:
                for attr_name in ("man_bit", "mantissa_bits", "man_bits"):
                    man_bit = _coerce_optional_int(getattr(hf_config, attr_name, None))
                    if man_bit is not None:
                        if k_man_bit is None:
                            k_man_bit = man_bit
                        if v_man_bit is None:
                            v_man_bit = man_bit
                if k_man_bit is None or v_man_bit is None:
                    quant_configs = [getattr(hf_config, "quantization_config", None)]
                    source_weight_format = getattr(
                        hf_config, "merak_source_weight_format", None
                    )
                    if source_weight_format is not None:
                        quant_configs.extend(
                            _get_config_value(source_weight_format, scope)
                            for scope in ("model", "text")
                        )

                    for quant_config in quant_configs:
                        if quant_config is None:
                            continue
                        if k_man_bit is None:
                            k_man_bit = _coerce_optional_int(
                                _get_config_value(quant_config, "kcache_man_bit")
                            )
                        if v_man_bit is None:
                            v_man_bit = _coerce_optional_int(
                                _get_config_value(quant_config, "vcache_man_bit")
                            )
                        for attr_name in ("man_bit", "mantissa_bits", "man_bits"):
                            man_bit = _coerce_optional_int(
                                _get_config_value(quant_config, attr_name)
                            )
                            if man_bit is not None:
                                if k_man_bit is None:
                                    k_man_bit = man_bit
                                if v_man_bit is None:
                                    v_man_bit = man_bit
                        if k_man_bit is not None and v_man_bit is not None:
                            break

    if k_man_bit is None:
        k_man_bit = 8
    if v_man_bit is None:
        v_man_bit = 8

    k_dtype = "int16" if k_man_bit > 7 else "int8"
    v_dtype = "int16" if v_man_bit > 7 else "int8"
    return f"sefp_k{k_dtype}_v{v_dtype}"


def _normalize_vllm_cache_dtype(value: str, vllm_config=None) -> str:
    normalized_value = value.strip().lower()
    if normalized_value == "hmbf16":
        return _resolve_hmbf16_cache_dtype(vllm_config) if vllm_config is not None else "sefp_kint8_vint8"
    if normalized_value in {"bf16", "bfloat16"}:
        return "bfloat16"
    if normalized_value in {"float16", "fp16", "half"}:
        return "float16"
    if normalized_value in {"auto", ""}:
        return "auto"
    return normalized_value


def _resolve_merak_kv_cache_dtype(vllm_config) -> str:
    env_format = os.environ.get(MERAK_KV_CACHE_FORMAT)
    if env_format:
        return _normalize_vllm_cache_dtype(env_format, vllm_config)

    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    config_format = getattr(hf_config, "kv_cache_format", "float16")
    if not isinstance(config_format, str):
        return "float16"

    return _normalize_vllm_cache_dtype(config_format, vllm_config)


def _apply_merak_kv_cache_dtype(vllm_config) -> None:
    from .hmfp_kv_cache import get_hmfp_vllm_backing_cache_dtype, is_hmfp_cache_dtype

    cache_config = vllm_config.cache_config
    logical_cache_dtype = _resolve_merak_kv_cache_dtype(vllm_config)
    cache_config.merak_logical_kv_cache_dtype = logical_cache_dtype
    if is_hmfp_cache_dtype(logical_cache_dtype):
        cache_config.cache_dtype = get_hmfp_vllm_backing_cache_dtype(logical_cache_dtype)
    else:
        cache_config.cache_dtype = logical_cache_dtype


def _disable_merak_prefix_caching(cache_config, *, max_model_len: int) -> None:
    """Disable prefix caching and the Mamba fields derived from it."""

    if cache_config is None:
        return
    cache_config.enable_prefix_caching = False
    if hasattr(cache_config, "mamba_block_size"):
        # vLLM's Mamba layers require a concrete block size even when prefix
        # caching is disabled.  max_model_len is its sentinel for one
        # request-local state block and is exempt from the prefix-cache-only
        # mamba-block-size validation.
        cache_config.mamba_block_size = max_model_len
    if hasattr(cache_config, "user_specified_mamba_block_size"):
        cache_config.user_specified_mamba_block_size = False
    if hasattr(cache_config, "mamba_cache_mode"):
        cache_config.mamba_cache_mode = "none"


def _resolve_hf_assets_dir(model_path: object) -> Path | None:
    if not isinstance(model_path, str):
        return None

    hf_assets_dir = Path(model_path) / "hf_config"
    if not hf_assets_dir.is_dir():
        return None

    return hf_assets_dir


def register_architecture() -> None:
    from vllm import ModelRegistry

    from .model_executor.register import get_model_registration_paths

    registrations = get_model_registration_paths()
    supported_archs = set(ModelRegistry.get_supported_archs())
    for architecture_name, model_cls in registrations.items():
        if architecture_name not in supported_archs:
            ModelRegistry.register_model(architecture_name, model_cls)


def _install_merak_hybrid_cache_page_compatibility() -> None:
    """Pad HMFP pages without changing the exported logical block size.

    Merak hybrid models combine 64-token PageAttention pages with larger
    Mamba state pages.  The stock vLLM unifier tries to grow the logical
    attention block and cannot reconcile the non-divisible Mamba page.  HMFP
    specs instead support physical padding, so only attention allocations are
    padded while Mamba specs remain untouched.
    """

    import vllm.v1.core.kv_cache_utils as kv_cache_utils
    from vllm.v1.kv_cache_interface import AttentionSpec

    original_unify = kv_cache_utils.unify_kv_cache_spec_page_size
    if getattr(original_unify, "_merak_hmfp_compatible", False):
        return

    def unify_merak_kv_cache_spec_page_size(kv_cache_spec):
        if not kv_cache_spec or all(
            isinstance(spec, AttentionSpec) for spec in kv_cache_spec.values()
        ):
            return original_unify(kv_cache_spec)

        max_page_size = max(spec.page_size_bytes for spec in kv_cache_spec.values())
        prepared = {
            layer_name: (
                replace(spec, page_size_padded=max_page_size)
                if isinstance(spec, AttentionSpec)
                and spec.page_size_bytes < max_page_size
                else spec
            )
            for layer_name, spec in kv_cache_spec.items()
        }
        return original_unify(prepared)

    setattr(unify_merak_kv_cache_spec_page_size, "_merak_hmfp_compatible", True)
    kv_cache_utils.unify_kv_cache_spec_page_size = (
        unify_merak_kv_cache_spec_page_size
    )


def register_kv_cache_spec() -> None:
    import vllm.v1.core.single_type_kv_cache_manager as stkvcm

    from .hmfp_kv_cache import HMFPFullAttentionSpec, HMFPSlidingWindowSpec

    _install_merak_hybrid_cache_page_compatibility()

    FullAttentionManager = stkvcm.FullAttentionManager
    SlidingWindowManager = stkvcm.SlidingWindowManager

    # vLLM's KVCache spec->manager registration API changed across versions:
    #   * old: a plain ``spec_manager_map`` dict on the module.
    #   * new (>=0.24): ``KVCacheSpecRegistry.register(spec_cls, manager_class, ...)``.
    # Support both so this plugin keeps working across vLLM upgrades. In the new
    # API we register HMFPFullAttentionSpec under FullAttentionManager and declare
    # FullAttentionSpec as its uniform base so it groups with regular full
    # attention (HMFPFullAttentionSpec subclasses FullAttentionSpec).
    spec_manager_map = getattr(stkvcm, "spec_manager_map", None)
    if spec_manager_map is not None:
        spec_manager_map.setdefault(HMFPFullAttentionSpec, FullAttentionManager)
        spec_manager_map.setdefault(HMFPSlidingWindowSpec, SlidingWindowManager)
        return

    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    KVCacheSpecRegistry.register(
        HMFPFullAttentionSpec,
        manager_class=FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        HMFPSlidingWindowSpec,
        manager_class=SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowSpec,
    )


def _build_merak_platform_class():
    from vllm.platforms.interface import Platform, PlatformEnum

    class MerakPlatform(Platform):
        _enum = PlatformEnum.CPU
        device_name: str = "xh2"
        device_type: str = "cpu"
        dispatch_key: str = "CPU"
        dist_backend: str = "gloo"

        @classmethod
        def import_kernels(cls) -> None:
            """Load the CUDA sampler ops used after HMONNX inference.

            Merak presents a CPU-like vLLM platform because model execution is
            delegated to HMONNX, but logits sampling still runs on CUDA.
            ``Platform.import_kernels`` imports the legacy ``vllm._C`` module;
            CUDA-enabled vLLM 0.24 wheels instead ship the operators in
            ``vllm._C_stable_libtorch``.  Without this override, non-default
            repetition penalties crash when they first call
            ``torch.ops._C.apply_repetition_penalties_``.
            """
            try:
                importlib.import_module("vllm._C_stable_libtorch")
            except ImportError as exc:
                logger.warning(
                    "Failed to import vLLM stable CUDA kernels required by "
                    "Merak logits sampling: %r",
                    exc,
                )

        @classmethod
        def get_device_name(cls, device_id: int = 0) -> str:
            return "xh2"

        @classmethod
        def set_device(cls, device: torch.device) -> None:
            del device

        @classmethod
        def manual_seed_all(cls, seed: int) -> None:
            from xhquant.api import set_random_seed

            set_random_seed(seed)

        @classmethod
        def supports_v1(cls, model_config) -> bool:
            """Returns whether the current platform can support v1 for the supplied
            model configuration.
            """
            return True

        @property
        def supported_dtypes(self) -> list[torch.dtype]:
            """Returns the supported dtypes for the current platform."""
            # Be careful with the order of the dtypes. The first dtype will
            # be used as the default dtype fallback for the current platform,
            # when encountering unsupported dtypes in "auto" dtype.
            return [
                torch.float16,
            ]

        @classmethod
        def check_and_update_config(cls, vllm_config):
            from .config import MERAK_PREFERRED_BLOCK_SIZE

            parallel_config = vllm_config.parallel_config
            assert parallel_config.world_size == 1, "Merak only supports single CPU socket currently."

            if parallel_config.worker_cls == "auto":
                parallel_config.worker_cls = "vllm_merak.worker_v1.merak_worker_v1.MerakWorkerV1"

            register_kv_cache_spec()

            # check and update model config
            model_config = vllm_config.model_config
            if not model_config.enforce_eager:
                logger.warning("CUDA graph is not supported on Merak backend, fallback to the eager mode.")
                model_config.enforce_eager = True

            hf_config = getattr(model_config, "hf_config", None)
            loader_architecture = getattr(hf_config, "merak_loader_architecture", None)
            hybrid_cache_policy = getattr(
                hf_config, "merak_hybrid_cache_policy", None
            )
            if (
                hybrid_cache_policy is not None
                and hybrid_cache_policy not in SUPPORTED_HYBRID_CACHE_POLICIES
            ):
                raise ValueError(
                    f"Unsupported Merak hybrid cache policy {hybrid_cache_policy!r}"
                )
            is_linear_attention_hybrid = hybrid_cache_policy is not None
            if is_linear_attention_hybrid:
                scheduler_config = getattr(vllm_config, "scheduler_config", None)
                if scheduler_config is None or getattr(scheduler_config, "max_num_seqs", None) != 1:
                    raise ValueError("Hybrid Linear state requires max_num_seqs=1")

            # WeMM pooling is prefill-only and Merak owns its compact Linear
            # state slots.  vLLM's prefix-cache ``align`` preprocessor indexes
            # those compact tensors with global scheduler block IDs when a
            # prompt crosses a 64-token page, which is not a valid ownership
            # mapping.  Disable that unsupported optimization before cache
            # finalization; PageAttention itself remains enabled.
            if loader_architecture == "WeMMEmbeddingMerakForEmbedding":
                _disable_merak_prefix_caching(
                    getattr(vllm_config, "cache_config", None),
                    max_model_len=model_config.max_model_len,
                )
            if (
                hybrid_cache_policy
                == HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST
            ):
                if getattr(hf_config, "enable_page_attention", False):
                    raise ValueError(
                        "Contiguous-single-request Merak hybrid models do not "
                        "support PageAttention"
                    )
                _disable_merak_prefix_caching(
                    getattr(vllm_config, "cache_config", None),
                    max_model_len=model_config.max_model_len,
                )

            is_gemma4_contract_v2 = bool(
                loader_architecture
                in {
                    "Gemma4MerakForCausalLM",
                    "Gemma4UnifiedMerakForCausalLM",
                }
                and getattr(hf_config, "attention_contract_version", None) == 2
            )
            if is_gemma4_contract_v2:
                scheduler_config = getattr(vllm_config, "scheduler_config", None)
                if scheduler_config is None:
                    raise ValueError("Gemma4 contract-v2 requires an explicit scheduler config")
                scheduler_budget = getattr(scheduler_config, "max_num_batched_tokens", None)
                if type(scheduler_budget) is not int or scheduler_budget < 320:
                    raise ValueError(
                        "Gemma4 contract-v2 requires max_num_batched_tokens >= 320 "
                        "so one bidirectional visual span cannot be scheduler-split"
                    )
                scheduler_config.disable_chunked_mm_input = True
                if getattr(scheduler_config, "max_num_seqs", None) != 1:
                    raise ValueError("Gemma4 contract-v2 fixed sliding cache requires max_num_seqs=1")

                # The exported sliding cache has one 21-page ring rather than
                # one scheduler-owned page per cached prefix block.  Retaining
                # multiple logical prefixes would alias that ring across
                # requests, so only full request-local ownership is valid.
                cache_config = vllm_config.cache_config
                _disable_merak_prefix_caching(
                    cache_config,
                    max_model_len=model_config.max_model_len,
                )

            model_path = getattr(model_config, "model", None)
            hf_assets_dir = _resolve_hf_assets_dir(model_path)

            hf_config_path = getattr(model_config, "hf_config_path", None)
            if hf_assets_dir is not None and hf_config_path in (None, "", model_path):
                model_config.hf_config_path = str(hf_assets_dir)
                logger.info(
                    "Merak backend resolved Hugging Face config path to %s.",
                    model_config.hf_config_path,
                )

            tokenizer_path = getattr(model_config, "tokenizer", None)
            if hf_assets_dir is not None and isinstance(tokenizer_path, str) and tokenizer_path == model_path:
                model_config.tokenizer = str(hf_assets_dir)
                logger.info(
                    "Merak backend resolved tokenizer path to %s.",
                    model_config.tokenizer,
                )

            # check and update cache config
            cache_config = vllm_config.cache_config
            if cache_config and cache_config.block_size is None:
                cache_config.block_size = MERAK_PREFERRED_BLOCK_SIZE

            _apply_merak_kv_cache_dtype(vllm_config)
            cache_config.block_size = MERAK_PREFERRED_BLOCK_SIZE
            cache_config.user_specified_block_size = True

            assert vllm_config.device_config.device_type == "cpu"
            assert vllm_config.lora_config is None, "Merak backend doesn't support LoRA"

        @classmethod
        def update_block_size_for_backend(cls, vllm_config) -> None:
            """Keep Merak's SEFP PageAttention logical block size stable.

            vLLM 0.24's generic hybrid attention/Mamba reconciliation grows
            the attention block size until the attention page is larger than the
            Mamba page, then pads the MambaSpec to match.  That is the wrong
            direction for Merak:

            * HMONNX SEFP PageAttention exports are built around the Merak
              64-token logical KV block.
            * MambaSpec is not block-stride indexed in vLLM's cache unifier, so
              padding it trips ``NotImplementedError``.
            * HMFPFullAttentionSpec *is* block-stride indexed and opts into
              physical page padding; the later cache unifier can safely pad the
              SEFP attention allocation up to the larger Mamba page while
              preserving the logical block size seen by the runtime.

            Therefore this platform intentionally does not call the inherited
            ``Platform.update_block_size_for_backend`` / CPU hybrid aligner.
            """
            from .config import MERAK_PREFERRED_BLOCK_SIZE

            cache_config = getattr(vllm_config, "cache_config", None)
            if cache_config is None:
                return

            if getattr(cache_config, "block_size", None) in (None, 0):
                cache_config.block_size = MERAK_PREFERRED_BLOCK_SIZE
            cache_config.user_specified_block_size = True
            if hasattr(cache_config, "mamba_page_size_padded"):
                cache_config.mamba_page_size_padded = None

        @classmethod
        def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
            logger.info("Using Merak Attention backend.")
            return "vllm_merak.attention.backends.merak.MerakAttentionBackend"

        @classmethod
        def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
            return False

        @classmethod
        def support_hybrid_kv_cache(cls) -> bool:
            """Merak Qwen3.5 is a hybrid model (full-attention + linear-attention).

            Returning True lets vLLM keep the hybrid KV cache manager enabled so it
            allocates a ``MambaSpec`` cache group for the Merak linear-state
            placeholder layers alongside the full-attention paged KV cache.
            """
            return True

        @classmethod
        def inference_mode(cls):
            return torch.inference_mode(mode=True)

        @classmethod
        def is_pin_memory_available(cls) -> bool:
            return False

        @classmethod
        def num_compute_units(cls, device_id: int = 0) -> int:
            if not torch.cuda.is_available():
                return 1
            return int(torch.cuda.get_device_properties(device_id).multi_processor_count)

    return MerakPlatform


_MERAK_PLATFORM_CLASS = None


def _get_merak_platform_class():
    global _MERAK_PLATFORM_CLASS
    if _MERAK_PLATFORM_CLASS is None:
        _MERAK_PLATFORM_CLASS = _build_merak_platform_class()
    return _MERAK_PLATFORM_CLASS


def __getattr__(name: str):
    if name == "MerakPlatform":
        cls = _get_merak_platform_class()
        globals()[name] = cls
        return cls
    raise AttributeError(name)
