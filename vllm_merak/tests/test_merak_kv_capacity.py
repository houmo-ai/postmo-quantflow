from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import pytest
import torch
from unittest.mock import patch
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_merak.hmfp_kv_cache import HMFPFullAttentionSpec, HMFPSlidingWindowSpec
from vllm_merak.kv_capacity import (
    CapacityGeometry,
    PhysicalLayer,
    build_capacity_plan,
    derive_capacity_geometry,
    derive_capacity_geometry_from_groups,
    format_capacity_log_lines,
    physical_layers_from_groups,
    parse_virtual_gib_budget,
    project_physical_storage,
    recommended_blocks,
    required_blocks,
)
from vllm_merak.worker_v1.merak_worker_v1 import MerakWorkerV1


def test_worker_capacity_logger_uses_configured_vllm_namespace() -> None:
    from vllm_merak.worker_v1 import merak_worker_v1

    assert merak_worker_v1.logger.name == "vllm.vllm_merak.worker_v1.merak_worker_v1"
    assert merak_worker_v1.logger.isEnabledFor(logging.INFO)


MAMBA_PAGE_BYTES = 1_114_112


def _real_specs() -> dict[str, object]:
    full_spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    mamba_spec = MambaSpec(
        block_size=262_144,
        shapes=((2_048, 4), (2_048, 4), (4_096, 4), (32, 128, 128)),
        dtypes=(torch.float16,) * 4,
        mamba_cache_mode="none",
    )
    return {
        **{f"full.{idx}": full_spec for idx in range(10)},
        **{f"linear.{idx}": mamba_spec for idx in range(30)},
    }


def _geometry(*, full_layers: int) -> CapacityGeometry:
    return CapacityGeometry(
        max_model_len=262_144,
        attention_group_block_demand=(4_096,),
        mamba_group_block_demand=(1, 1, 1),
        virtual_page_size_bytes=MAMBA_PAGE_BYTES,
        virtual_group_size=full_layers,
    )


def test_required_and_recommended_blocks_for_256k_hybrid_model() -> None:
    assert required_blocks(262_144, 64, 3) == 4_099
    assert recommended_blocks(4_099) == 4_304


@pytest.mark.parametrize("full_layers", [10, 8])
def test_default_plan_reproduces_requested_global_block_count(full_layers: int) -> None:
    geometry = _geometry(full_layers=full_layers)

    plan = build_capacity_plan(geometry)

    assert plan.required_blocks == 4_099
    assert plan.recommended_blocks == 4_304
    assert plan.num_blocks == 4_304
    assert plan.virtual_bytes_per_block == full_layers * MAMBA_PAGE_BYTES
    assert plan.virtual_budget_bytes == 4_304 * full_layers * MAMBA_PAGE_BYTES
    assert plan.virtual_effective_bytes == plan.virtual_budget_bytes
    assert plan.max_concurrency == pytest.approx(4_304 / 4_099)
    assert plan.token_capacity == int((4_304 / 4_099) * 262_144)


def test_explicit_virtual_budget_is_floored_without_reinterpretation() -> None:
    geometry = _geometry(full_layers=10)
    bytes_per_block = 10 * MAMBA_PAGE_BYTES
    requested_bytes = 4_099 * bytes_per_block + 123

    plan = build_capacity_plan(
        geometry,
        virtual_budget_bytes=requested_bytes,
        source="VLLM_CPU_KVCACHE_SPACE",
    )

    assert plan.source == "VLLM_CPU_KVCACHE_SPACE"
    assert plan.virtual_budget_bytes == requested_bytes
    assert plan.virtual_effective_bytes == 4_099 * bytes_per_block
    assert plan.num_blocks == 4_099


def test_one_block_short_plan_fails_before_allocation() -> None:
    geometry = _geometry(full_layers=10)

    with pytest.raises(ValueError, match=r"4098 global blocks.*requires at least 4099"):
        build_capacity_plan(
            geometry, requested_blocks=4_098, source="explicit block override"
        )


@pytest.mark.parametrize("value", ["", "nan", "inf", "-1", "0", "not-a-number"])
def test_invalid_virtual_gib_override_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="positive finite GiB"):
        parse_virtual_gib_budget(value, variable_name="VLLM_CPU_KVCACHE_SPACE")


def test_virtual_gib_override_is_converted_deterministically() -> None:
    assert parse_virtual_gib_budget(
        "1.5", variable_name="VLLM_CPU_KVCACHE_SPACE"
    ) == int(1.5 * (1 << 30))


@pytest.mark.parametrize(
    ("full_layers", "full_page_bytes", "linear_layers", "limit_gib"),
    [
        (10, 66_560, 30, 2.8),
        (8, 133_120, 24, 4.5),
    ],
)
def test_physical_projection_uses_exact_full_pages_and_sparse_linear_slots(
    full_layers: int,
    full_page_bytes: int,
    linear_layers: int,
    limit_gib: float,
) -> None:
    plan = build_capacity_plan(_geometry(full_layers=full_layers))
    full = tuple(
        PhysicalLayer(
            name=f"full.{idx}", device="cuda:0", page_size_bytes=full_page_bytes
        )
        for idx in range(full_layers)
    )
    linear = tuple(
        PhysicalLayer(
            name=f"linear.{idx}", device="cuda:0", page_size_bytes=MAMBA_PAGE_BYTES
        )
        for idx in range(linear_layers)
    )

    projection = project_physical_storage(
        num_blocks=plan.num_blocks,
        full_layers=full,
        linear_layers=linear,
        linear_slots=1,
        virtual_effective_bytes=plan.virtual_effective_bytes,
    )

    assert projection.full_bytes == full_layers * full_page_bytes * 4_304
    assert projection.linear_bytes == linear_layers * MAMBA_PAGE_BYTES
    assert projection.total_bytes == projection.full_bytes + projection.linear_bytes
    assert projection.total_bytes <= math.floor(limit_gib * (1 << 30))
    assert (
        projection.padding_saved_bytes
        == plan.virtual_effective_bytes - projection.total_bytes
    )


def test_physical_projection_reconciles_per_device_totals() -> None:
    full = (
        PhysicalLayer("full.0", "cuda:0", 100),
        PhysicalLayer("full.1", "cuda:1", 100),
    )
    linear = (
        PhysicalLayer("linear.0", "cuda:0", 20),
        PhysicalLayer("linear.1", "cuda:1", 30),
    )

    projection = project_physical_storage(
        num_blocks=4,
        full_layers=full,
        linear_layers=linear,
        linear_slots=1,
        virtual_effective_bytes=1_000,
    )

    assert projection.by_device["cuda:0"].full_bytes == 400
    assert projection.by_device["cuda:0"].linear_bytes == 20
    assert projection.by_device["cuda:1"].full_bytes == 400
    assert projection.by_device["cuda:1"].linear_bytes == 30
    assert (
        sum(item.total_bytes for item in projection.by_device.values())
        == projection.total_bytes
    )


def test_log_lines_distinguish_virtual_scheduler_and_physical_backing() -> None:
    plan = build_capacity_plan(_geometry(full_layers=10))
    projection = project_physical_storage(
        num_blocks=plan.num_blocks,
        full_layers=(PhysicalLayer("full.0", "cuda:0", 66_560),),
        linear_layers=(PhysicalLayer("linear.0", "cuda:0", MAMBA_PAGE_BYTES),),
        linear_slots=1,
        virtual_effective_bytes=plan.virtual_effective_bytes,
    )

    lines = format_capacity_log_lines(plan, projection)

    assert any(
        "virtual scheduler budget" in line and "global_blocks=4304" in line
        for line in lines
    )
    assert any(
        "physical cache backing only" in line and "device=cuda:0" in line
        for line in lines
    )
    assert any("padding saved" in line for line in lines)
    assert all("total process memory" not in line for line in lines)


def test_geometry_is_derived_from_real_layer_specs_without_model_hardcoding() -> None:
    geometry = derive_capacity_geometry(
        _real_specs(), max_model_len=262_144, max_num_batched_tokens=320
    )

    assert geometry.attention_group_block_demand == (4_096,)
    assert geometry.mamba_group_block_demand == (1, 1, 1)
    assert geometry.virtual_page_size_bytes == MAMBA_PAGE_BYTES
    assert geometry.virtual_group_size == 10
    assert geometry.minimum_blocks == 4_099


def test_final_group_geometry_and_physical_layers_reconcile_devices() -> None:
    specs = _real_specs()
    original_full_spec = specs["full.0"]
    full_spec = HMFPFullAttentionSpec(
        block_size=original_full_spec.block_size,
        num_kv_heads=original_full_spec.num_kv_heads,
        head_size=original_full_spec.head_size,
        head_size_v=original_full_spec.head_size_v,
        dtype=original_full_spec.dtype,
        sefp_cache_dtype=original_full_spec.sefp_cache_dtype,
        page_size_padded=MAMBA_PAGE_BYTES,
    )
    mamba_spec = specs["linear.0"]
    groups = [
        KVCacheGroupSpec([f"full.{idx}" for idx in range(10)], full_spec),
        *[
            KVCacheGroupSpec(
                [f"linear.{idx}" for idx in range(offset, 30, 3)],
                mamba_spec,
            )
            for offset in range(3)
        ],
    ]

    geometry = derive_capacity_geometry_from_groups(
        groups, max_model_len=262_144, max_num_batched_tokens=320
    )
    full, linear = physical_layers_from_groups(
        groups,
        layer_devices={f"full.{idx}": f"cuda:{idx % 2}" for idx in range(10)},
        linear_default_device="cuda:0",
    )

    assert geometry.minimum_blocks == 4_099
    assert len(full) == 10
    assert len(linear) == 30
    assert {layer.page_size_bytes for layer in full} == {66_560}
    assert {layer.page_size_bytes for layer in linear} == {MAMBA_PAGE_BYTES}
    assert {layer.device for layer in full} == {"cuda:0", "cuda:1"}
    assert {layer.device for layer in linear} == {"cuda:0"}


def _gemma4_uniform_wrapper_groups():
    sliding_spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=8,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    full_spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=512,
        head_size_v=512,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        page_size_padded=sliding_spec.page_size_bytes,
    )
    full_specs = {f"full.{idx}": full_spec for idx in range(5)}
    sliding_specs = {f"sliding.{idx}": sliding_spec for idx in range(5)}
    return [
        KVCacheGroupSpec(
            [*full_specs, "full.shared"],
            UniformTypeKVCacheSpecs(
                block_size=64,
                kv_cache_specs=full_specs,
            ),
        ),
        KVCacheGroupSpec(
            [*sliding_specs, "sliding.shared"],
            UniformTypeKVCacheSpecs(
                block_size=64,
                kv_cache_specs=sliding_specs,
            ),
        ),
    ]


def test_final_gemma4_uniform_wrappers_expand_real_members_and_geometry() -> None:
    groups = _gemma4_uniform_wrapper_groups()

    geometry = derive_capacity_geometry_from_groups(
        groups,
        max_model_len=8192,
        max_num_batched_tokens=320,
    )

    assert geometry.attention_group_block_demand == (128, 22)
    assert geometry.mamba_group_block_demand == ()
    assert geometry.virtual_page_size_bytes == 5 * 266_240
    assert geometry.virtual_group_size == 1
    assert geometry.virtual_bytes_per_block == 5 * 266_240


def test_final_uniform_wrapper_rejects_mixed_full_and_sliding_members() -> None:
    full_group, sliding_group = _gemma4_uniform_wrapper_groups()
    mixed_specs = {
        "full.0": full_group.kv_cache_spec.kv_cache_specs["full.0"],
        "sliding.0": sliding_group.kv_cache_spec.kv_cache_specs["sliding.0"],
    }
    mixed = KVCacheGroupSpec(
        list(mixed_specs),
        UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=mixed_specs),
    )

    with pytest.raises(ValueError, match="mix.*FullAttention.*SlidingWindow"):
        derive_capacity_geometry_from_groups(
            [mixed],
            max_model_len=8192,
            max_num_batched_tokens=320,
        )


def test_physical_projection_expands_wrapper_children_without_shared_aliases() -> None:
    groups = _gemma4_uniform_wrapper_groups()
    layer_devices = {
        layer_name: "cuda:0"
        for group in groups
        for layer_name in group.kv_cache_spec.kv_cache_specs
    }

    attention, linear = physical_layers_from_groups(
        groups,
        layer_devices=layer_devices,
        linear_default_device="cpu",
    )

    assert linear == ()
    assert len(attention) == 10
    assert {layer.name for layer in attention} == {
        *(f"full.{idx}" for idx in range(5)),
        *(f"sliding.{idx}" for idx in range(5)),
    }
    assert {layer.page_size_bytes for layer in attention} == {133_120, 266_240}


@pytest.mark.parametrize("max_model_len", [32_768, 262_144])
def test_sliding_group_admission_plateaus_after_window(max_model_len: int) -> None:
    spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    geometry = derive_capacity_geometry_from_groups(
        [KVCacheGroupSpec(["sliding.0"], spec)],
        max_model_len=max_model_len,
        max_num_batched_tokens=320,
    )
    expected = math.ceil(((1024 - 1) + 320) / 64) + 1

    assert spec.max_admission_blocks_per_request(320, max_model_len) == expected
    assert geometry.attention_group_block_demand == (expected,)
    assert geometry.minimum_blocks == expected


def test_pregrouped_sliding_admission_uses_native_plateau_bound() -> None:
    spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )

    geometry = derive_capacity_geometry(
        {"sliding.0": spec},
        max_model_len=262_144,
        max_num_batched_tokens=320,
    )

    assert geometry.attention_group_block_demand == (22,)
    assert geometry.minimum_blocks == 22


def test_final_sliding_group_uses_formula_when_vllm_has_no_admission_api() -> None:
    spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=8,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )

    with patch(
        "vllm.v1.kv_cache_interface.SlidingWindowSpec.max_admission_blocks_per_request",
        None,
    ):
        geometry = derive_capacity_geometry_from_groups(
            [KVCacheGroupSpec(["sliding.0"], spec)],
            max_model_len=8192,
            max_num_batched_tokens=320,
        )

    assert geometry.attention_group_block_demand == (22,)


def test_full_group_still_scales_with_max_context() -> None:
    spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )

    geometry = derive_capacity_geometry_from_groups(
        [KVCacheGroupSpec(["full.0"], spec)],
        max_model_len=262_144,
        max_num_batched_tokens=320,
    )

    assert geometry.attention_group_block_demand == (4_096,)
    assert geometry.minimum_blocks == 4_096


def test_virtual_attention_packing_keeps_per_layer_physical_backing() -> None:
    spec = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    groups = [KVCacheGroupSpec(["sliding.0", "sliding.1"], spec)]
    tensors = [
        KVCacheTensor(
            size=spec.real_page_size_bytes * 22,
            shared_by=["sliding.0", "sliding.1"],
        )
    ]

    attention, linear = physical_layers_from_groups(
        groups,
        layer_devices={"sliding.0": "cuda:0", "sliding.1": "cuda:0"},
        linear_default_device="cpu",
        kv_cache_tensors=tensors,
    )

    assert [(layer.name, layer.page_size_bytes) for layer in attention] == [
        ("sliding.0", spec.real_page_size_bytes),
        ("sliding.1", spec.real_page_size_bytes),
    ]
    assert {layer.device for layer in attention} == {"cuda:0"}
    assert linear == ()


def test_qwen_hybrid_virtual_packing_keeps_native_physical_layers() -> None:
    """Virtual shared_by slots do not declare HMONNX physical ownership.

    Qwen3.5 places one FullAttention layer and three Mamba layers in each
    finalized virtual tensor because their padded page sizes match.  Merak
    nevertheless allocates one native HMFP cache per full layer and one
    compact state cache per linear layer.
    """

    specs = _real_specs()
    full = specs["full.0"]
    linear = specs["linear.0"]
    groups = [
        KVCacheGroupSpec(["full.0"], full),
        KVCacheGroupSpec(["linear.0"], linear),
        KVCacheGroupSpec(["linear.1"], linear),
        KVCacheGroupSpec(["linear.2"], linear),
    ]
    virtual_tensors = [
        KVCacheTensor(
            size=MAMBA_PAGE_BYTES * 4_304,
            shared_by=["full.0", "linear.0", "linear.1", "linear.2"],
        )
    ]

    attention, linear_layers = physical_layers_from_groups(
        groups,
        layer_devices={"full.0": "cuda:0"},
        linear_default_device="cuda:0",
        kv_cache_tensors=virtual_tensors,
    )

    assert [(layer.name, layer.page_size_bytes) for layer in attention] == [
        ("full.0", full.real_page_size_bytes)
    ]
    assert [(layer.name, layer.page_size_bytes) for layer in linear_layers] == [
        ("linear.0", MAMBA_PAGE_BYTES),
        ("linear.1", MAMBA_PAGE_BYTES),
        ("linear.2", MAMBA_PAGE_BYTES),
    ]


def test_virtual_packing_does_not_require_homogeneous_physical_specs() -> None:
    full = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    sliding = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    groups = [
        KVCacheGroupSpec(["full.0"], full),
        KVCacheGroupSpec(["sliding.0"], sliding),
    ]

    attention, linear = physical_layers_from_groups(
        groups,
        layer_devices={"full.0": "cuda:0", "sliding.0": "cuda:0"},
        linear_default_device="cpu",
        kv_cache_tensors=[
            KVCacheTensor(
                size=max(full.page_size_bytes, sliding.page_size_bytes),
                shared_by=["full.0", "sliding.0"],
            )
        ],
    )

    assert [(layer.name, layer.page_size_bytes) for layer in attention] == [
        ("full.0", full.real_page_size_bytes),
        ("sliding.0", sliding.real_page_size_bytes),
    ]
    assert linear == ()


def test_gemma4_physical_projection_ignores_hybrid_virtual_tensor_packing() -> None:
    """vLLM packs one layer from each hybrid group into a virtual tensor.

    Gemma4 contract-v2 has a separate physical cache for every cache_index in
    the 26B export, so those heterogeneous ``shared_by`` entries must neither
    alias storage nor be rejected as incompatible physical owners.
    """

    sliding = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=8,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    full = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=512,
        head_size_v=512,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        page_size_padded=sliding.page_size_bytes,
    )
    groups = [
        KVCacheGroupSpec(
            [f"placeholder_attn_layers.{offset + 6 * row}" for row in range(5)],
            full if offset == 5 else sliding,
        )
        for offset in range(6)
    ]
    tensors = [
        KVCacheTensor(
            size=sliding.page_size_bytes,
            shared_by=[
                f"placeholder_attn_layers.{offset + 6 * row}" for offset in range(6)
            ],
        )
        for row in range(5)
    ]
    owner_by_layer = {
        f"placeholder_attn_layers.{layer_idx}": (f"placeholder_attn_layers.{layer_idx}")
        for layer_idx in range(30)
    }

    attention, linear = physical_layers_from_groups(
        groups,
        layer_devices={name: "cuda:0" for name in owner_by_layer},
        linear_default_device="cpu",
        kv_cache_tensors=tensors,
        physical_owner_by_layer=owner_by_layer,
    )

    assert linear == ()
    assert len(attention) == 30
    assert len({layer.name for layer in attention}) == 30
    assert (
        sum(
            layer.page_size_bytes == sliding.real_page_size_bytes for layer in attention
        )
        == 25
    )
    assert (
        sum(layer.page_size_bytes == full.real_page_size_bytes for layer in attention)
        == 5
    )


def test_gemma4_physical_projection_accepts_omitted_logical_sharing_aliases() -> None:
    spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    groups = [KVCacheGroupSpec(["placeholder_attn_layers.0"], spec)]

    attention, linear = physical_layers_from_groups(
        groups,
        layer_devices={"placeholder_attn_layers.0": "cuda:0"},
        linear_default_device="cpu",
        physical_owner_by_layer={
            "placeholder_attn_layers.0": "placeholder_attn_layers.0",
            "placeholder_attn_layers.1": "placeholder_attn_layers.0",
        },
    )

    assert [layer.name for layer in attention] == ["placeholder_attn_layers.0"]
    assert linear == ()


def test_gemma4_projection_keeps_sliding_pages_fixed_at_window_plus_chunk() -> None:
    page_bytes = 532_480
    projection = project_physical_storage(
        num_blocks=2_408,
        full_layers=(
            PhysicalLayer("full.0", "cuda:0", page_bytes),
            PhysicalLayer("sliding.0", "cuda:0", page_bytes, num_pages=21),
        ),
        linear_layers=(),
        linear_slots=1,
        virtual_effective_bytes=0,
    )

    assert projection.full_bytes == (2_408 + 21) * page_bytes
    assert projection.by_device["cuda:0"].full_bytes == (2_408 + 21) * page_bytes


def test_int16_sefp_physical_projection_uses_native_132096_byte_page() -> None:
    spec = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_kint16_vint16",
        page_size_padded=MAMBA_PAGE_BYTES,
    )
    groups = [KVCacheGroupSpec(["full.0"], spec)]

    full, linear = physical_layers_from_groups(
        groups,
        layer_devices={"full.0": "cuda:0"},
        linear_default_device="cpu",
    )

    assert spec.real_page_size_bytes == 132_096
    assert spec.page_size_bytes == MAMBA_PAGE_BYTES
    assert [layer.page_size_bytes for layer in full] == [132_096]
    assert linear == ()


class _FakeRunner:
    def __init__(self, specs: dict[str, object]) -> None:
        self.specs = specs
        self.device = torch.device("cpu")
        self.initialized = False

    def get_kv_cache_spec(self):
        return self.specs

    def _get_kv_cache_layer_devices(self):
        return {
            name: torch.device("cpu") for name in self.specs if name.startswith("full.")
        }

    def initialize_kv_cache(self, config):
        self.initialized = True


def _worker_without_runtime() -> MerakWorkerV1:
    # Production workers are created after MerakPlatform.check_and_update_config,
    # which registers Merak's cache specs and vLLM-version bridge.  Keep this
    # direct-construction fixture independent of test execution order.
    from vllm_merak.platform import register_kv_cache_spec

    register_kv_cache_spec()
    worker = MerakWorkerV1.__new__(MerakWorkerV1)
    worker.model_runner = _FakeRunner(_real_specs())
    worker.model_config = SimpleNamespace(max_model_len=262_144)
    worker.cache_config = SimpleNamespace(
        block_size=64,
        kv_cache_memory_bytes=None,
        num_gpu_blocks_override=None,
    )
    worker.scheduler_config = SimpleNamespace(
        disable_hybrid_kv_cache_manager=False,
        max_num_seqs=1,
        max_num_batched_tokens=320,
    )
    worker.vllm_config = SimpleNamespace(
        scheduler_config=worker.scheduler_config,
        cache_config=worker.cache_config,
    )
    worker.kv_cache_config = None
    return worker


def _final_config(num_blocks: int) -> KVCacheConfig:
    specs = _real_specs()
    original_full_spec = specs["full.0"]
    full_spec = HMFPFullAttentionSpec(
        block_size=original_full_spec.block_size,
        num_kv_heads=original_full_spec.num_kv_heads,
        head_size=original_full_spec.head_size,
        head_size_v=original_full_spec.head_size_v,
        dtype=original_full_spec.dtype,
        sefp_cache_dtype=original_full_spec.sefp_cache_dtype,
        page_size_padded=MAMBA_PAGE_BYTES,
    )
    mamba_spec = specs["linear.0"]
    groups = [
        KVCacheGroupSpec([f"full.{idx}" for idx in range(10)], full_spec),
        *[
            KVCacheGroupSpec(
                [f"linear.{idx}" for idx in range(offset, 30, 3)],
                mamba_spec,
            )
            for offset in range(3)
        ],
    ]
    virtual_tensors = [
        KVCacheTensor(
            size=num_blocks * MAMBA_PAGE_BYTES,
            shared_by=[
                f"full.{row}",
                f"linear.{3 * row}",
                f"linear.{3 * row + 1}",
                f"linear.{3 * row + 2}",
            ],
        )
        for row in range(10)
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=virtual_tensors,
        kv_cache_groups=groups,
    )


def test_worker_default_budget_drives_4304_blocks_without_gpu() -> None:
    worker = _worker_without_runtime()

    available = worker.determine_available_memory()

    assert available == 4_304 * 10 * MAMBA_PAGE_BYTES


def test_worker_default_budget_uses_finalized_vllm_group_geometry() -> None:
    """The byte handoff must preserve the post-unification block target."""

    full = HMFPFullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=512,
        head_size_v=512,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
    )
    sliding = HMFPSlidingWindowSpec(
        block_size=64,
        num_kv_heads=8,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        sefp_cache_dtype="sefp_k8_v8",
        sliding_window=1024,
    )
    worker = MerakWorkerV1.__new__(MerakWorkerV1)
    worker.model_runner = _FakeRunner({"full.0": full, "sliding.0": sliding})
    worker.model_config = SimpleNamespace(max_model_len=8_192)
    worker.cache_config = SimpleNamespace(
        block_size=64,
        kv_cache_memory_bytes=None,
        num_gpu_blocks_override=None,
    )
    worker.scheduler_config = SimpleNamespace(
        disable_hybrid_kv_cache_manager=False,
        max_num_seqs=1,
        max_num_batched_tokens=320,
    )
    worker.vllm_config = SimpleNamespace(
        scheduler_config=worker.scheduler_config,
        cache_config=worker.cache_config,
    )

    available = worker.determine_available_memory()

    # vLLM widens the smaller FullAttention page from 64 to 128 tokens before
    # final grouping. The final demand is 64 Full + 22 Sliding blocks, not the
    # pre-grouping 128 Full + 22 Sliding blocks.
    assert worker._capacity_geometry.attention_group_block_demand == (64, 22)
    assert worker._capacity_plan.required_blocks == 86
    assert worker._capacity_plan.recommended_blocks == 91
    assert worker._capacity_plan.num_blocks == 91
    assert available == 91 * sliding.page_size_bytes


def test_worker_keeps_cpu_space_override_virtual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker_without_runtime()
    monkeypatch.setenv("VLLM_CPU_KVCACHE_SPACE", "48")

    available = worker.determine_available_memory()

    assert available == 48 * (1 << 30)
    assert worker._capacity_plan.source == "VLLM_CPU_KVCACHE_SPACE"


def test_worker_rejects_final_config_before_allocator_call() -> None:
    worker = _worker_without_runtime()
    worker.determine_available_memory()

    with pytest.raises(ValueError, match=r"4098 global blocks.*requires at least 4099"):
        worker.initialize_from_config(_final_config(4_098))

    assert worker.model_runner.initialized is False


def test_worker_logs_projection_then_initializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker_without_runtime()
    worker.determine_available_memory()
    messages: list[str] = []
    monkeypatch.setattr(
        "vllm_merak.worker_v1.merak_worker_v1.logger.info",
        lambda _format, message: messages.append(message),
    )

    worker.initialize_from_config(_final_config(4_304))

    assert worker.model_runner.initialized is True
    assert any("virtual scheduler budget" in message for message in messages)
    assert any("physical cache backing only" in message for message in messages)
