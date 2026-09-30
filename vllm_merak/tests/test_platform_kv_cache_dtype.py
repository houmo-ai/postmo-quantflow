from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_merak.hmfp_kv_cache import HMFPFullAttentionSpec, HMFPSlidingWindowSpec
from vllm_merak.constants import (
    HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST,
    HYBRID_CACHE_POLICY_PAGED,
)
from vllm_merak.platform import (
    MerakPlatform,
    _apply_merak_kv_cache_dtype,
    _normalize_vllm_cache_dtype,
    _resolve_merak_kv_cache_dtype,
    register_kv_cache_spec,
)


class _DummyModelConfig:
    def __init__(self, hf_config):
        self.hf_config = hf_config


class _DummyVllmConfig:
    def __init__(self, hf_config):
        self.model_config = _DummyModelConfig(hf_config)
        self.cache_config = SimpleNamespace(cache_dtype="auto")


def test_normalize_vllm_cache_dtype_supports_hmbf16() -> None:
    assert _normalize_vllm_cache_dtype("HMBF16") == "sefp_kint8_vint8"
    assert _normalize_vllm_cache_dtype("float16") == "float16"
    assert _normalize_vllm_cache_dtype("auto") == "auto"


def test_resolve_merak_kv_cache_dtype_uses_man_bit() -> None:
    config = _DummyVllmConfig(SimpleNamespace(kv_cache_format="HMBF16", man_bit=7))
    assert _resolve_merak_kv_cache_dtype(config) == "sefp_kint8_vint8"

    config = _DummyVllmConfig(SimpleNamespace(kv_cache_format="HMBF16", man_bit=8))
    assert _resolve_merak_kv_cache_dtype(config) == "sefp_kint16_vint16"


def test_resolve_merak_kv_cache_dtype_uses_kcache_and_vcache_man_bits() -> None:
    config = _DummyVllmConfig(
        SimpleNamespace(kv_cache_format="HMBF16", kcache_man_bit=7, vcache_man_bit=8)
    )
    assert _resolve_merak_kv_cache_dtype(config) == "sefp_kint8_vint16"

    config = _DummyVllmConfig(
        SimpleNamespace(kv_cache_format="HMBF16", kcache_man_bit=7, vcache_man_bit=7)
    )
    assert _resolve_merak_kv_cache_dtype(config) == "sefp_kint8_vint8"

    config = _DummyVllmConfig(SimpleNamespace(kv_cache_format="sefp_kint16_vint16"))
    assert _resolve_merak_kv_cache_dtype(config) == "sefp_kint16_vint16"


def test_resolve_merak_kv_cache_dtype_uses_isolated_source_weight_metadata() -> None:
    config = _DummyVllmConfig(
        SimpleNamespace(
            kv_cache_format="HMBF16",
            merak_source_weight_format={
                "model": {"kcache_man_bit": 7},
                "text": {"vcache_man_bit": 8},
            },
        )
    )

    assert _resolve_merak_kv_cache_dtype(config) == "sefp_kint8_vint16"


def test_apply_merak_kv_cache_dtype_keeps_sefp_logical_dtype_out_of_vllm_cache_dtype() -> (
    None
):
    config = _DummyVllmConfig(SimpleNamespace(kv_cache_format="sefp_k8_v8"))

    _apply_merak_kv_cache_dtype(config)

    assert config.cache_config.cache_dtype == "fp8"
    assert config.cache_config.merak_logical_kv_cache_dtype == "sefp_k8_v8"


def test_register_kv_cache_spec_registers_sefp_full_attention_spec() -> None:
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    register_kv_cache_spec()

    spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype="auto",
    )
    assert KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager
    assert KVCacheSpecRegistry.get_uniform_type_base_spec(spec) is FullAttentionSpec


def test_hmfp_sliding_spec_uses_native_sliding_manager() -> None:
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
    from vllm.v1.kv_cache_interface import SlidingWindowSpec
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    register_kv_cache_spec()

    spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=4,
        head_size=128,
        head_size_v=128,
        dtype=torch.float8_e4m3fn,
        sefp_cache_dtype="sefp_kint8_vint8",
        sliding_window=1024,
    )
    assert KVCacheSpecRegistry.get_manager_class(spec) is SlidingWindowManager
    assert KVCacheSpecRegistry.get_uniform_type_base_spec(spec) is SlidingWindowSpec
    manager = SlidingWindowManager(
        spec,
        block_pool=SimpleNamespace(null_block=None),
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=64,
    )
    assert manager.get_num_skipped_tokens(0) == 0
    assert manager.get_num_skipped_tokens(1023) == 0
    assert manager.get_num_skipped_tokens(1024) == 1


def test_hmfp_sliding_spec_registers_with_legacy_manager_map(monkeypatch) -> None:
    import vllm.v1.core.single_type_kv_cache_manager as stkvcm

    legacy_map = {}
    monkeypatch.setattr(stkvcm, "spec_manager_map", legacy_map, raising=False)

    register_kv_cache_spec()

    assert legacy_map[HMFPFullAttentionSpec] is stkvcm.FullAttentionManager
    assert legacy_map[HMFPSlidingWindowSpec] is stkvcm.SlidingWindowManager


def test_merak_platform_num_compute_units_uses_cuda_properties() -> None:
    props = SimpleNamespace(multi_processor_count=132)
    with (
        patch("vllm_merak.platform.torch.cuda.is_available", return_value=True),
        patch(
            "vllm_merak.platform.torch.cuda.get_device_properties", return_value=props
        ) as get_props,
    ):
        assert MerakPlatform.num_compute_units(6) == 132

    get_props.assert_called_once_with(6)


def test_merak_platform_num_compute_units_cpu_fallback() -> None:
    with patch("vllm_merak.platform.torch.cuda.is_available", return_value=False):
        assert MerakPlatform.num_compute_units(0) == 1


def test_merak_platform_imports_stable_cuda_sampler_kernels() -> None:
    with patch("vllm_merak.platform.importlib.import_module") as import_module:
        MerakPlatform.import_kernels()

    import_module.assert_called_once_with("vllm._C_stable_libtorch")


def test_merak_platform_does_not_pad_mamba_or_grow_attention_block_size() -> None:
    cache_config = SimpleNamespace(
        block_size=64,
        user_specified_block_size=True,
        mamba_page_size_padded=12345,
    )
    config = SimpleNamespace(cache_config=cache_config)

    MerakPlatform.update_block_size_for_backend(config)

    assert cache_config.block_size == 64
    assert cache_config.user_specified_block_size is True
    assert cache_config.mamba_page_size_padded is None


def test_merak_hybrid_page_unification_pads_sefp_attention_not_mamba() -> None:
    from vllm.v1.kv_cache_interface import MambaSpec

    register_kv_cache_spec()
    from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size

    attention_spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    mamba_page_size = attention_spec.real_page_size_bytes + 128
    mamba_spec = MambaSpec(
        block_size=2048,
        shapes=((mamba_page_size,),),
        dtypes=(torch.uint8,),
    )

    unified = unify_kv_cache_spec_page_size(
        {
            "model.layers.0.attn": attention_spec,
            "model.layers.1.linear_attn": mamba_spec,
        }
    )

    unified_attention = unified["model.layers.0.attn"]
    unified_mamba = unified["model.layers.1.linear_attn"]

    assert isinstance(unified_attention, HMFPFullAttentionSpec)
    assert unified_attention.block_size == 64
    assert unified_attention.page_size_padded == mamba_page_size
    assert unified_attention.page_size_bytes == mamba_page_size
    assert unified_mamba is mamba_spec
    assert unified_mamba.page_size_padded is None


def _platform_config(
    *,
    architecture: str,
    contract_version: int | None,
    budget: int,
    hybrid_cache_policy: str | None = None,
    enable_page_attention: bool = False,
):
    hf_config = SimpleNamespace(
        kv_cache_format="float16",
        merak_loader_architecture=architecture,
        attention_contract_version=contract_version,
        merak_hybrid_cache_policy=hybrid_cache_policy,
        enable_page_attention=enable_page_attention,
    )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=1, worker_cls="auto"),
        model_config=SimpleNamespace(
            enforce_eager=True,
            max_model_len=2048,
            model=None,
            hf_config_path=None,
            tokenizer=None,
            hf_config=hf_config,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=budget,
            max_num_seqs=1,
            disable_chunked_mm_input=False,
        ),
        cache_config=SimpleNamespace(
            block_size=64,
            cache_dtype="auto",
            enable_prefix_caching=True,
            mamba_block_size=256,
            user_specified_mamba_block_size=True,
            mamba_cache_mode="align",
        ),
        device_config=SimpleNamespace(device_type="cpu"),
        lora_config=None,
    )


def test_gemma4_platform_disables_upstream_multimodal_chunk_splitting() -> None:
    config = _platform_config(
        architecture="Gemma4MerakForCausalLM", contract_version=2, budget=320
    )

    with patch("vllm_merak.platform.register_kv_cache_spec"):
        MerakPlatform.check_and_update_config(config)

    assert config.scheduler_config.disable_chunked_mm_input is True
    assert config.cache_config.enable_prefix_caching is False
    assert config.cache_config.mamba_block_size == 2048
    assert config.cache_config.user_specified_mamba_block_size is False
    assert config.cache_config.mamba_cache_mode == "none"


def test_gemma4_fixed_sliding_cache_rejects_multiple_concurrent_sequences() -> None:
    config = _platform_config(
        architecture="Gemma4MerakForCausalLM", contract_version=2, budget=320
    )
    config.scheduler_config.max_num_seqs = 2

    with (
        patch("vllm_merak.platform.register_kv_cache_spec"),
        pytest.raises(ValueError, match="max_num_seqs=1"),
    ):
        MerakPlatform.check_and_update_config(config)


def test_gemma4_platform_requires_one_atomic_320_token_scheduler_budget() -> None:
    config = _platform_config(
        architecture="Gemma4MerakForCausalLM", contract_version=2, budget=319
    )

    with (
        patch("vllm_merak.platform.register_kv_cache_spec"),
        pytest.raises(ValueError, match="320"),
    ):
        MerakPlatform.check_and_update_config(config)


def test_qwen_platform_scheduler_policy_is_unchanged() -> None:
    config = _platform_config(
        architecture="Qwen3_5MerakForCausalLM",
        contract_version=1,
        budget=128,
        hybrid_cache_policy=HYBRID_CACHE_POLICY_PAGED,
    )

    with patch("vllm_merak.platform.register_kv_cache_spec"):
        MerakPlatform.check_and_update_config(config)

    assert config.scheduler_config.disable_chunked_mm_input is False
    assert config.cache_config.enable_prefix_caching is True


def test_wemm_platform_disables_unsafe_prefix_cache_alignment() -> None:
    config = _platform_config(
        architecture="WeMMEmbeddingMerakForEmbedding",
        contract_version=1,
        budget=256,
        hybrid_cache_policy=HYBRID_CACHE_POLICY_PAGED,
        enable_page_attention=True,
    )

    with patch("vllm_merak.platform.register_kv_cache_spec"):
        MerakPlatform.check_and_update_config(config)

    assert config.cache_config.enable_prefix_caching is False
    assert config.cache_config.mamba_block_size == 2048
    assert config.cache_config.user_specified_mamba_block_size is False
    assert config.cache_config.mamba_cache_mode == "none"


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3_5MerakForCausalLM",
        "Qwen3_5MerakTextForCausalLM",
        "Qwen3NextMerakForCausalLM",
    ],
)
def test_linear_attention_hybrid_rejects_multiple_sequences(
    architecture: str,
) -> None:
    config = _platform_config(
        architecture=architecture,
        contract_version=2,
        budget=256,
        hybrid_cache_policy=HYBRID_CACHE_POLICY_PAGED,
    )
    config.scheduler_config.max_num_seqs = 2

    with (
        patch("vllm_merak.platform.register_kv_cache_spec"),
        pytest.raises(ValueError, match="Hybrid Linear state.*max_num_seqs=1"),
    ):
        MerakPlatform.check_and_update_config(config)


def test_contiguous_hybrid_disables_prefix_cache_and_rejects_page_attention() -> None:
    config = _platform_config(
        architecture="Ling3FlashMerakForCausalLM",
        contract_version=None,
        budget=256,
        hybrid_cache_policy=HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST,
    )

    with patch("vllm_merak.platform.register_kv_cache_spec"):
        MerakPlatform.check_and_update_config(config)

    assert config.cache_config.enable_prefix_caching is False

    config.model_config.hf_config.enable_page_attention = True
    with (
        patch("vllm_merak.platform.register_kv_cache_spec"),
        pytest.raises(ValueError, match="do not support PageAttention"),
    ):
        MerakPlatform.check_and_update_config(config)
