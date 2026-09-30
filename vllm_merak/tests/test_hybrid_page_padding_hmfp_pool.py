from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from hybrid_page_padding_testkit import LINEAR_PAGE_BYTES
from vllm_merak.hmfp_kv_cache import (
    HMFPFullAttentionSpec,
    hmfp_paged_kv_cache_from_raw_tensor,
)
from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1


@pytest.mark.parametrize(
    (
        "num_kv_heads",
        "cache_dtype",
        "expected_real_page_bytes",
        "k_qdata_dtype",
        "v_qdata_dtype",
    ),
    [
        (2, "sefp_k8_v8", 66_560, torch.int8, torch.int8),
        (4, "sefp_k8_v8", 133_120, torch.int8, torch.int8),
        (2, "sefp_kint16_vint16", 132_096, torch.int16, torch.int16),
        (2, "sefp_kint8_vint16", 99_328, torch.int8, torch.int16),
    ],
)
def test_full_pool_allocates_exact_hmfp_bytes_and_stride(
    num_kv_heads: int,
    cache_dtype: str,
    expected_real_page_bytes: int,
    k_qdata_dtype: torch.dtype,
    v_qdata_dtype: torch.dtype,
) -> None:
    """Task-B contract: scheduler padding must not materialize physically.

    Interface assumption: the runner allocation strategy obtains ``num_blocks``
    and the HMFP spec from its normal KV config/group iterator.  No standalone
    pool class is required.
    """
    num_blocks = 3
    layer_name = "model.layers.3.attn"
    spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=num_kv_heads,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        sefp_cache_dtype=cache_dtype,
        page_size_padded=LINEAR_PAGE_BYTES,
    )
    assert spec.real_page_size_bytes == expected_real_page_bytes
    assert spec.page_size_bytes == LINEAR_PAGE_BYTES

    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.kv_cache_dtype = cache_dtype
    runner.runner_only_attn_layers = set()
    runner._get_kv_cache_layer_devices = lambda: {layer_name: torch.device("cpu")}
    runner.kv_cache_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            SimpleNamespace(
                size=num_blocks * LINEAR_PAGE_BYTES,
                shared_by=[layer_name],
                block_stride=LINEAR_PAGE_BYTES,
                offset=0,
            )
        ],
        kv_cache_groups=[SimpleNamespace(layer_names=[layer_name])],
    )
    group = SimpleNamespace(
        kv_cache_group_id=0,
        kv_cache_spec=spec,
        layer_names=[layer_name],
    )
    runner._kv_cache_spec_attn_group_iterator = lambda: iter([group])

    raw = runner._allocate_kv_cache_tensors(runner.kv_cache_config)[layer_name]
    assert raw.numel() == num_blocks * expected_real_page_bytes

    cache = runner._reshape_kv_cache_tensors({layer_name: raw}, [64])[layer_name]
    assert cache.shape == (num_blocks, expected_real_page_bytes)
    assert cache.stride() == (expected_real_page_bytes, 1)
    assert cache.data_ptr() == raw.data_ptr()

    raw.fill_(0x5A)
    cache[0].zero_()
    assert torch.count_nonzero(cache[0]).item() == 0
    assert torch.all(cache[1:] == 0x5A)

    rebound = runner._reshape_kv_cache_tensors({layer_name: raw}, [64])[layer_name]
    assert rebound.data_ptr() == cache.data_ptr()
    assert rebound.stride() == cache.stride()

    sections = hmfp_paged_kv_cache_from_raw_tensor(
        cache,
        block_size=64,
        num_kv_heads=num_kv_heads,
        head_size=256,
        cache_dtype=cache_dtype,
    )
    assert sections.k_qdata.dtype is k_qdata_dtype
    assert sections.v_qdata.dtype is v_qdata_dtype
    for section in sections.tensors():
        assert section.stride(0) * section.element_size() == expected_real_page_bytes


def test_full_layers_do_not_alias_one_device_wide_virtual_packed_backing() -> None:
    """Distinct Full slots on one device receive distinct exact backings."""
    num_blocks = 2
    layer_names = ["model.layers.3.attn", "model.layers.7.attn"]
    spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        page_size_padded=LINEAR_PAGE_BYTES,
    )
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.kv_cache_dtype = "sefp_k8_v8"
    runner.runner_only_attn_layers = set()
    runner._get_kv_cache_layer_devices = lambda: {
        name: torch.device("cpu") for name in layer_names
    }
    runner.kv_cache_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            SimpleNamespace(
                size=num_blocks * 2 * LINEAR_PAGE_BYTES,
                shared_by=[layer_names[0]],
                block_stride=2 * LINEAR_PAGE_BYTES,
                offset=0,
            ),
            SimpleNamespace(
                size=num_blocks * 2 * LINEAR_PAGE_BYTES,
                shared_by=[layer_names[1]],
                block_stride=2 * LINEAR_PAGE_BYTES,
                offset=LINEAR_PAGE_BYTES,
            ),
        ],
        kv_cache_groups=[SimpleNamespace(layer_names=layer_names)],
    )
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(
        [
            SimpleNamespace(
                kv_cache_group_id=0, kv_cache_spec=spec, layer_names=layer_names
            )
        ]
    )

    raw = runner._allocate_kv_cache_tensors(runner.kv_cache_config)

    assert set(raw) == set(layer_names)
    assert raw[layer_names[0]].numel() == num_blocks * spec.real_page_size_bytes
    assert raw[layer_names[1]].numel() == num_blocks * spec.real_page_size_bytes
    assert raw[layer_names[0]].data_ptr() != raw[layer_names[1]].data_ptr()


def test_runner_binds_int16_sefp_raw_page_as_hmfp_cache() -> None:
    layer_name = "model.layers.3.attn"
    cache_dtype = "sefp_kint16_vint16"
    spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        sefp_cache_dtype=cache_dtype,
    )
    raw_cache = torch.zeros((2, spec.real_page_size_bytes), dtype=torch.int8)
    placeholder = SimpleNamespace(
        layer_name=layer_name,
        kv_cache=raw_cache,
        num_kv_heads=2,
        head_size=256,
        get_kv_cache_spec=lambda _config: spec,
    )
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.cache_dtype = cache_dtype
    runner.vllm_config = SimpleNamespace()
    runner.model = SimpleNamespace(
        cache_dtype=cache_dtype,
        placeholder_attn_layers=[placeholder],
    )

    runner._wrap_sefp_kv_cache_handles()

    bound = placeholder.kv_cache
    assert bound.k_qdata.dtype is torch.int16
    assert bound.v_qdata.dtype is torch.int16
    assert bound.k_qdata.stride(0) * bound.k_qdata.element_size() == 132_096
    assert bound.v_qdata.stride(0) * bound.v_qdata.element_size() == 132_096


def test_full_layers_allocate_exact_bytes_on_each_page_attention_device() -> None:
    num_blocks = 2
    layer_names = ["model.layers.3.attn", "model.layers.7.attn"]
    spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        page_size_padded=LINEAR_PAGE_BYTES,
    )
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cuda:0")
    runner.kv_cache_dtype = "sefp_k8_v8"
    runner.runner_only_attn_layers = set()
    runner._get_kv_cache_layer_devices = lambda: {
        layer_names[0]: torch.device("cuda:0"),
        layer_names[1]: torch.device("cuda:2"),
    }
    runner.kv_cache_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[SimpleNamespace(layer_names=layer_names)],
    )
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(
        [
            SimpleNamespace(
                kv_cache_group_id=0, kv_cache_spec=spec, layer_names=layer_names
            )
        ]
    )
    allocations = []

    def fake_zeros(size, *, dtype, device):
        allocations.append((size, dtype, torch.device(device)))
        return torch.empty(size, dtype=dtype, device="meta")

    with patch(
        "vllm_merak.worker_v1.merak_model_runner_v1.torch.zeros", side_effect=fake_zeros
    ):
        raw = runner._allocate_kv_cache_tensors(runner.kv_cache_config)

    assert allocations == [
        (num_blocks * spec.real_page_size_bytes, torch.int8, torch.device("cuda:0")),
        (num_blocks * spec.real_page_size_bytes, torch.int8, torch.device("cuda:2")),
    ]
    assert raw[layer_names[0]] is not raw[layer_names[1]]


def test_mamba_allocation_and_reshape_use_compact_phase1_slot_count() -> None:
    class MambaSpec:
        block_size = 262_144
        page_size_bytes = LINEAR_PAGE_BYTES
        shapes = ((2,), (1,))
        dtypes = (torch.float16, torch.float32)

    full_name = "model.layers.3.attn"
    linear_name = "model.layers.0.linear_attn"
    full_spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        page_size_padded=LINEAR_PAGE_BYTES,
    )
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.kv_cache_dtype = "sefp_k8_v8"
    runner.runner_only_attn_layers = set()
    runner.scheduler_config = SimpleNamespace(max_num_seqs=1)
    runner._get_kv_cache_layer_devices = lambda: {full_name: torch.device("cpu")}
    runner.kv_cache_config = SimpleNamespace(
        num_blocks=4_304,
        kv_cache_tensors=[
            SimpleNamespace(
                size=4_304 * LINEAR_PAGE_BYTES,
                shared_by=[full_name, linear_name],
                block_stride=LINEAR_PAGE_BYTES,
                offset=0,
            )
        ],
        kv_cache_groups=[SimpleNamespace(layer_names=[full_name, linear_name])],
    )
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(
        [
            SimpleNamespace(
                kv_cache_group_id=0, kv_cache_spec=full_spec, layer_names=[full_name]
            ),
            SimpleNamespace(
                kv_cache_group_id=1,
                kv_cache_spec=MambaSpec(),
                layer_names=[linear_name],
            ),
        ]
    )

    raw = runner._allocate_kv_cache_tensors(runner.kv_cache_config)
    assert raw[linear_name].numel() == 8

    caches = runner._reshape_kv_cache_tensors(raw, [64, 262_144])
    assert [tuple(tensor.shape) for tensor in caches[linear_name]] == [(1, 2), (1, 1)]
