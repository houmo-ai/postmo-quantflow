import types

import torch

from vllm_merak.hmfp_kv_cache import (
    HMFPFullAttentionSpec,
    hmfp_paged_kv_cache_from_raw_tensor,
)


def _fake_group(group_id, spec):
    return types.SimpleNamespace(
        kv_cache_group_id=group_id,
        kv_cache_spec=spec,
        layer_names=[f"layer.{group_id}"],
    )


def test_reshape_skips_mamba_group_without_raising():
    """A MambaSpec group (no sefp_cache_dtype) must not trigger the
    NotImplementedError meant for genuinely mixed SEFP/non-SEFP full-attn groups."""
    from vllm_merak.worker_v1.merak_model_runner_v1 import _spec_is_mamba

    class MambaSpec:  # _spec_is_mamba keys on the class name
        pass

    mamba_spec = MambaSpec()  # no sefp_cache_dtype attr
    assert _spec_is_mamba(mamba_spec) is True

    sefp_spec = types.SimpleNamespace(sefp_cache_dtype="sefp_kint8_vint8")
    assert _spec_is_mamba(sefp_spec) is False


def test_reshape_ignores_virtual_packing_for_exact_sefp_and_compact_mamba():
    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    class MambaSpec:  # _spec_is_mamba keys on the class name
        block_size = 1
        page_size_bytes = 16
        shapes = ((2,), (1,))
        dtypes = (torch.float16, torch.float32)

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.kv_cache_dtype = "sefp_k8_v8"
    runner.runner_only_attn_layers = set()

    sefp_spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    sefp_page_bytes = sefp_spec.real_page_size_bytes
    num_blocks = 3
    full_backing = torch.empty(num_blocks * sefp_page_bytes, dtype=torch.uint8)
    mamba_backing = torch.empty(8, dtype=torch.uint8)

    runner.kv_cache_config = types.SimpleNamespace(
        kv_cache_tensors=[
            types.SimpleNamespace(
                block_stride=sefp_page_bytes + 64,
                offset=0,
                shared_by=["model.layers.0.attn"],
            ),
            types.SimpleNamespace(
                block_stride=sefp_page_bytes + 64,
                offset=sefp_page_bytes,
                shared_by=["model.layers.1.linear_attn"],
            ),
        ]
    )
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(
        [
            types.SimpleNamespace(
                kv_cache_group_id=0,
                kv_cache_spec=sefp_spec,
                layer_names=["model.layers.0.attn"],
            ),
            types.SimpleNamespace(
                kv_cache_group_id=1,
                kv_cache_spec=MambaSpec(),
                layer_names=["model.layers.1.linear_attn"],
            ),
        ]
    )

    kv_caches = MerakModelRunnerV1._reshape_kv_cache_tensors(
        runner,
        {
            "model.layers.0.attn": full_backing,
            "model.layers.1.linear_attn": mamba_backing,
        },
        [64, 1],
    )

    sefp_raw = kv_caches["model.layers.0.attn"]
    assert tuple(sefp_raw.shape) == (num_blocks, sefp_page_bytes)
    assert sefp_raw.stride(0) == sefp_page_bytes
    assert sefp_raw.data_ptr() == full_backing.data_ptr()

    paged_cache = hmfp_paged_kv_cache_from_raw_tensor(
        sefp_raw,
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        cache_dtype="sefp_k8_v8",
    )
    assert paged_cache.k_qdata.stride(0) == sefp_page_bytes

    mamba_states = kv_caches["model.layers.1.linear_attn"]
    assert len(mamba_states) == 2
    assert tuple(mamba_states[0].shape) == (1, 2)
    assert tuple(mamba_states[1].shape) == (1, 1)
    assert mamba_states[0].data_ptr() == mamba_backing.data_ptr()
