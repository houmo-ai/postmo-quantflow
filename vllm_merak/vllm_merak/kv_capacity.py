"""Deterministic accounting for Merak's virtual and physical KV capacity.

vLLM converts a byte budget into one global block count using the unified
hybrid page size.  Merak uses that count for scheduling, but stores Full
Attention pages at their native HMFP size and Linear Attention state in a
small number of request slots.  This module keeps those two accounting planes
explicit and contains no allocator or device-runtime calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import prod
from typing import Mapping, Sequence

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


GIB = 1 << 30


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator}")
    return (numerator + denominator - 1) // denominator


def required_blocks(
    max_model_len: int, full_block_size: int, mamba_group_block_demand: int
) -> int:
    """Return global IDs needed by one request across all cache groups."""

    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be positive, got {max_model_len}")
    if full_block_size <= 0:
        raise ValueError(f"full_block_size must be positive, got {full_block_size}")
    if mamba_group_block_demand < 0:
        raise ValueError(
            "mamba_group_block_demand must be non-negative, "
            f"got {mamba_group_block_demand}"
        )
    return _ceil_div(max_model_len, full_block_size) + mamba_group_block_demand


def recommended_blocks(minimum_blocks: int, headroom: float = 0.05) -> int:
    """Round a fractional scheduling headroom upward to whole global IDs."""

    if minimum_blocks <= 0:
        raise ValueError(f"minimum_blocks must be positive, got {minimum_blocks}")
    if not math.isfinite(headroom) or headroom < 0:
        raise ValueError(f"headroom must be finite and non-negative, got {headroom}")
    return math.ceil(minimum_blocks * (1.0 + headroom))


def parse_virtual_gib_budget(value: str, *, variable_name: str) -> int:
    """Parse a legacy GiB setting without implying it is physical storage."""

    try:
        gib = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{variable_name} must be a positive finite GiB value, got {value!r}"
        ) from exc
    if not math.isfinite(gib) or gib <= 0:
        raise ValueError(
            f"{variable_name} must be a positive finite GiB value, got {value!r}"
        )
    return int(gib * GIB)


@dataclass(frozen=True)
class CapacityGeometry:
    """Scheduler geometry after identifying attention and Mamba groups."""

    max_model_len: int
    attention_group_block_demand: tuple[int, ...]
    mamba_group_block_demand: tuple[int, ...]
    virtual_page_size_bytes: int
    virtual_group_size: int

    def __post_init__(self) -> None:
        if self.max_model_len <= 0:
            raise ValueError(
                f"max_model_len must be positive, got {self.max_model_len}"
            )
        if not self.attention_group_block_demand or any(
            demand <= 0 for demand in self.attention_group_block_demand
        ):
            raise ValueError(
                "Attention group block demand must be non-empty and positive"
            )
        if self.virtual_page_size_bytes <= 0:
            raise ValueError(
                "virtual_page_size_bytes must be positive, "
                f"got {self.virtual_page_size_bytes}"
            )
        if self.virtual_group_size <= 0:
            raise ValueError(
                f"virtual_group_size must be positive, got {self.virtual_group_size}"
            )
        if any(demand < 0 for demand in self.mamba_group_block_demand):
            raise ValueError("Mamba group block demand must be non-negative")

    @property
    def minimum_blocks(self) -> int:
        return sum(self.attention_group_block_demand) + sum(
            self.mamba_group_block_demand
        )

    @property
    def virtual_bytes_per_block(self) -> int:
        return self.virtual_page_size_bytes * self.virtual_group_size


def _uniform_positive(values: set[int], *, label: str) -> int:
    if len(values) != 1:
        raise ValueError(
            f"Merak capacity planning requires one {label}, got {sorted(values)}"
        )
    value = values.pop()
    if value <= 0:
        raise ValueError(f"{label} must be positive, got {value}")
    return value


def _mamba_blocks_per_group(spec: MambaSpec, max_model_len: int) -> int:
    speculative = int(spec.num_speculative_blocks)
    if spec.mamba_cache_mode == "all":
        return _ceil_div(max_model_len, int(spec.block_size)) + speculative
    if spec.mamba_cache_mode == "align":
        return 2 + speculative
    if spec.mamba_cache_mode == "none":
        return 1 + speculative
    raise ValueError(f"Unsupported Mamba cache mode {spec.mamba_cache_mode!r}")


def _sliding_blocks_per_group(
    spec: SlidingWindowSpec,
    max_num_batched_tokens: int,
    max_model_len: int,
) -> int:
    """Return vLLM's admission demand across old and new spec APIs."""

    if max_num_batched_tokens <= 0:
        raise ValueError(
            f"max_num_batched_tokens must be positive, got {max_num_batched_tokens}"
        )
    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be positive, got {max_model_len}")

    native_admission = getattr(spec, "max_admission_blocks_per_request", None)
    if callable(native_admission):
        return int(native_admission(max_num_batched_tokens, max_model_len))

    sliding_window = int(spec.sliding_window)
    block_size = int(spec.block_size)
    if sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")
    # Mirrors vLLM's pre-0.24 SlidingWindowSpec admission formula: a
    # request can retain W - 1 prior tokens and admit one complete scheduler
    # chunk, capped by the model context, plus the allocator's look-ahead page.
    num_tokens = min(
        sliding_window - 1 + max_num_batched_tokens,
        max_model_len,
    )
    return _ceil_div(num_tokens, block_size) + 1


def _general_group_size(type_counts: Sequence[int]) -> int:
    """Mirror vLLM's general hybrid grouping rule for layer-type counts."""

    if not type_counts or any(count <= 0 for count in type_counts):
        raise ValueError(
            f"KV layer-type counts must be positive, got {tuple(type_counts)}"
        )
    minimum = min(type_counts)
    maximum = max(type_counts)
    return maximum if maximum < minimum * 1.5 else minimum


def resolve_finalized_group_layer_specs(
    group: KVCacheGroupSpec,
    *,
    shared_layer_targets: Mapping[str, str] | None = None,
) -> dict[str, KVCacheSpec]:
    """Resolve a finalized group's per-layer specs without flattening geometry.

    ``UniformTypeKVCacheSpecs`` owns one child spec per physical scheduler
    layer. vLLM may append KV-sharing aliases to ``group.layer_names`` without
    duplicating those children. Callers that need aliases resolved must provide
    their explicit target mapping; storage/accounting callers intentionally see
    physical children only.
    """

    layer_names = tuple(str(name) for name in group.layer_names)
    if not layer_names:
        raise ValueError("Finalized Merak KV cache group must contain layer names")
    if len(set(layer_names)) != len(layer_names):
        raise ValueError(
            "Finalized Merak KV cache layer appears more than once in one group"
        )
    outer_spec = group.kv_cache_spec
    if not isinstance(outer_spec, UniformTypeKVCacheSpecs):
        return {layer_name: outer_spec for layer_name in layer_names}

    physical_specs = dict(outer_spec.kv_cache_specs)
    if not physical_specs:
        raise ValueError("Finalized UniformTypeKVCacheSpecs must contain child specs")
    missing_members = sorted(set(physical_specs) - set(layer_names))
    if missing_members:
        raise ValueError(
            "Finalized UniformTypeKVCacheSpecs contains children outside its "
            f"group: {missing_members}"
        )
    if any(
        isinstance(child, UniformTypeKVCacheSpecs) for child in physical_specs.values()
    ):
        raise ValueError("Nested UniformTypeKVCacheSpecs is not supported")
    if any(
        int(child.block_size) != int(outer_spec.block_size)
        for child in physical_specs.values()
    ):
        raise ValueError(
            "Finalized UniformTypeKVCacheSpecs children disagree on block size"
        )

    if shared_layer_targets is None:
        return physical_specs

    resolved = dict(physical_specs)
    for layer_name in layer_names:
        if layer_name in resolved:
            continue
        target = shared_layer_targets.get(layer_name)
        if target is None:
            raise ValueError(
                "Finalized UniformTypeKVCacheSpecs layer has no physical child "
                f"or declared sharing target: {layer_name!r}"
            )
        target_spec = physical_specs.get(str(target))
        if target_spec is None:
            raise ValueError(
                "Finalized UniformTypeKVCacheSpecs sharing target is not a "
                f"physical child: layer={layer_name!r}, target={target!r}"
            )
        resolved[layer_name] = target_spec
    return resolved


def derive_capacity_geometry(
    kv_cache_specs: Mapping[str, KVCacheSpec],
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
) -> CapacityGeometry:
    """Derive pre-grouping hybrid geometry from model-emitted cache specs."""

    sliding_specs = [
        spec for spec in kv_cache_specs.values() if isinstance(spec, SlidingWindowSpec)
    ]
    full_specs = [
        spec for spec in kv_cache_specs.values() if isinstance(spec, FullAttentionSpec)
    ]
    mamba_specs = [
        spec for spec in kv_cache_specs.values() if isinstance(spec, MambaSpec)
    ]
    if len(sliding_specs) + len(full_specs) + len(mamba_specs) != len(kv_cache_specs):
        unsupported = sorted(
            {type(spec).__name__ for spec in kv_cache_specs.values()}
            - {
                type(spec).__name__
                for spec in (*sliding_specs, *full_specs, *mamba_specs)
            }
        )
        raise ValueError(
            f"Unsupported Merak KV cache specs for capacity planning: {unsupported}"
        )
    if not full_specs and not sliding_specs:
        raise ValueError(
            "Merak capacity planning requires at least one attention cache layer"
        )

    type_counts = [count for count in (len(full_specs), len(sliding_specs)) if count]
    if mamba_specs:
        type_counts.append(len(mamba_specs))
    group_size = _general_group_size(type_counts)
    attention_demand: tuple[int, ...] = ()
    if full_specs:
        full_block_size = _uniform_positive(
            {int(spec.block_size) for spec in full_specs},
            label="FullAttention block size",
        )
        attention_demand += (_ceil_div(max_model_len, full_block_size),) * (
            _ceil_div(len(full_specs), group_size)
        )
    if sliding_specs:
        sliding_per_group = _uniform_positive(
            {
                _sliding_blocks_per_group(spec, max_num_batched_tokens, max_model_len)
                for spec in sliding_specs
            },
            label="SlidingWindow admission block demand",
        )
        attention_demand += (sliding_per_group,) * _ceil_div(
            len(sliding_specs), group_size
        )

    mamba_demand: tuple[int, ...] = ()
    if mamba_specs:
        per_group_demands = {
            _mamba_blocks_per_group(spec, max_model_len) for spec in mamba_specs
        }
        per_group = _uniform_positive(per_group_demands, label="Mamba block demand")
        mamba_demand = (per_group,) * _ceil_div(len(mamba_specs), group_size)

    return CapacityGeometry(
        max_model_len=max_model_len,
        attention_group_block_demand=attention_demand,
        mamba_group_block_demand=mamba_demand,
        virtual_page_size_bytes=max(
            int(spec.page_size_bytes) for spec in kv_cache_specs.values()
        ),
        virtual_group_size=group_size,
    )


def derive_capacity_geometry_from_groups(
    kv_cache_groups: Sequence[KVCacheGroupSpec],
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
) -> CapacityGeometry:
    """Derive authoritative geometry from vLLM's finalized cache groups."""

    attention_demand: list[int] = []
    mamba_demand: list[int] = []
    unsupported: set[str] = set()
    wrapped_group_page_sizes: list[int] = []
    saw_wrapped_group = False
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            saw_wrapped_group = True
            child_specs = resolve_finalized_group_layer_specs(group)
            child_kinds = {
                "SlidingWindow"
                if isinstance(child, SlidingWindowSpec)
                else "FullAttention"
                if isinstance(child, FullAttentionSpec)
                else "Mamba"
                if isinstance(child, MambaSpec)
                else type(child).__name__
                for child in child_specs.values()
            }
            supported_kinds = {"SlidingWindow", "FullAttention", "Mamba"}
            if not child_kinds <= supported_kinds:
                unsupported.update(child_kinds - supported_kinds)
                continue
            if len(child_kinds) != 1:
                raise ValueError(
                    "Finalized UniformTypeKVCacheSpecs must not mix cache kinds; "
                    f"got {sorted(child_kinds)}"
                )

            child_kind = next(iter(child_kinds))
            if child_kind == "SlidingWindow":
                demand = _uniform_positive(
                    {
                        _sliding_blocks_per_group(
                            child, max_num_batched_tokens, max_model_len
                        )
                        for child in child_specs.values()
                    },
                    label="wrapped SlidingWindow admission block demand",
                )
                attention_demand.append(demand)
            elif child_kind == "FullAttention":
                demand = _uniform_positive(
                    {
                        _ceil_div(max_model_len, int(child.block_size))
                        for child in child_specs.values()
                    },
                    label="wrapped FullAttention block demand",
                )
                attention_demand.append(demand)
            else:
                demand = _uniform_positive(
                    {
                        _mamba_blocks_per_group(child, max_model_len)
                        for child in child_specs.values()
                    },
                    label="wrapped Mamba block demand",
                )
                mamba_demand.append(demand)

            child_page_size = sum(
                int(child.page_size_bytes) for child in child_specs.values()
            )
            if int(spec.page_size_bytes) != child_page_size:
                raise ValueError(
                    "Finalized UniformTypeKVCacheSpecs page size does not equal "
                    "the sum of its child page sizes"
                )
            wrapped_group_page_sizes.append(child_page_size)
            continue
        if isinstance(spec, SlidingWindowSpec):
            attention_demand.append(
                _sliding_blocks_per_group(spec, max_num_batched_tokens, max_model_len)
            )
        elif isinstance(spec, FullAttentionSpec):
            attention_demand.append(_ceil_div(max_model_len, int(spec.block_size)))
        elif isinstance(spec, MambaSpec):
            mamba_demand.append(_mamba_blocks_per_group(spec, max_model_len))
        else:
            unsupported.add(type(spec).__name__)
    if unsupported:
        raise ValueError(
            f"Unsupported finalized Merak KV cache groups: {sorted(unsupported)}"
        )
    if not attention_demand:
        raise ValueError("Finalized Merak KV cache config has no attention group")

    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    if group_size <= 0:
        raise ValueError("Finalized Merak KV cache groups contain no layers")
    if saw_wrapped_group:
        if len(wrapped_group_page_sizes) != len(kv_cache_groups):
            raise ValueError(
                "Finalized Merak capacity groups must not mix wrapped and "
                "unwrapped KV cache specs"
            )
        virtual_page_size_bytes = _uniform_positive(
            set(wrapped_group_page_sizes),
            label="finalized wrapped virtual page size",
        )
        virtual_group_size = 1
    else:
        virtual_page_size_bytes = _uniform_positive(
            {int(group.kv_cache_spec.page_size_bytes) for group in kv_cache_groups},
            label="finalized virtual page size",
        )
        virtual_group_size = group_size

    return CapacityGeometry(
        max_model_len=max_model_len,
        attention_group_block_demand=tuple(attention_demand),
        mamba_group_block_demand=tuple(mamba_demand),
        virtual_page_size_bytes=virtual_page_size_bytes,
        virtual_group_size=virtual_group_size,
    )


@dataclass(frozen=True)
class CapacityPlan:
    source: str
    max_model_len: int
    required_blocks: int
    recommended_blocks: int
    num_blocks: int
    virtual_page_size_bytes: int
    virtual_bytes_per_block: int
    virtual_budget_bytes: int
    virtual_effective_bytes: int
    token_capacity: int
    max_concurrency: float


def build_capacity_plan(
    geometry: CapacityGeometry,
    *,
    requested_blocks: int | None = None,
    virtual_budget_bytes: int | None = None,
    source: str = "recommended 5% headroom",
) -> CapacityPlan:
    """Resolve a desired global block count and its vLLM virtual byte budget."""

    if requested_blocks is not None and virtual_budget_bytes is not None:
        raise ValueError(
            "Specify either requested_blocks or virtual_budget_bytes, not both"
        )

    minimum = geometry.minimum_blocks
    recommended = recommended_blocks(minimum)
    bytes_per_block = geometry.virtual_bytes_per_block

    if requested_blocks is not None:
        num_blocks = int(requested_blocks)
        budget_bytes = num_blocks * bytes_per_block
    elif virtual_budget_bytes is not None:
        budget_bytes = int(virtual_budget_bytes)
        if budget_bytes <= 0:
            raise ValueError(
                f"virtual_budget_bytes must be positive, got {budget_bytes}"
            )
        num_blocks = budget_bytes // bytes_per_block
    else:
        num_blocks = recommended
        budget_bytes = num_blocks * bytes_per_block

    if num_blocks < minimum:
        raise ValueError(
            f"Merak hybrid KV capacity source {source!r} provides {num_blocks} global blocks, "
            f"but this configuration requires at least {minimum} "
            f"({geometry.max_model_len} max tokens plus hybrid state groups)."
        )

    concurrency = num_blocks / minimum
    return CapacityPlan(
        source=source,
        max_model_len=geometry.max_model_len,
        required_blocks=minimum,
        recommended_blocks=recommended,
        num_blocks=num_blocks,
        virtual_page_size_bytes=geometry.virtual_page_size_bytes,
        virtual_bytes_per_block=bytes_per_block,
        virtual_budget_bytes=budget_bytes,
        virtual_effective_bytes=num_blocks * bytes_per_block,
        token_capacity=int(concurrency * geometry.max_model_len),
        max_concurrency=concurrency,
    )


@dataclass(frozen=True)
class PhysicalLayer:
    name: str
    device: str
    page_size_bytes: int
    num_pages: int | None = None

    def __post_init__(self) -> None:
        if self.page_size_bytes <= 0:
            raise ValueError(
                f"Physical page size for {self.name!r} must be positive, "
                f"got {self.page_size_bytes}"
            )
        if self.num_pages is not None and self.num_pages <= 0:
            raise ValueError(
                f"Fixed physical page count for {self.name!r} must be positive, "
                f"got {self.num_pages}"
            )


def _native_mamba_page_size_bytes(spec: MambaSpec) -> int:
    if len(spec.shapes) != len(spec.dtypes):
        raise ValueError(
            "Mamba state shape/dtype count mismatch: "
            f"{len(spec.shapes)} shapes versus {len(spec.dtypes)} dtypes"
        )
    return sum(
        prod(shape) * int(dtype.itemsize)
        for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True)
    )


def physical_layers_from_groups(
    kv_cache_groups: Sequence[KVCacheGroupSpec],
    *,
    layer_devices: Mapping[str, object],
    linear_default_device: object,
    kv_cache_tensors: Sequence[KVCacheTensor] | None = None,
    physical_owner_by_layer: Mapping[str, str] | None = None,
    physical_pages_by_layer: Mapping[str, int] | None = None,
) -> tuple[tuple[PhysicalLayer, ...], tuple[PhysicalLayer, ...]]:
    """Project model-owned physical caches from finalized layer specs.

    ``KVCacheTensor.shared_by`` describes vLLM's virtual allocator packing.  A
    hybrid model can therefore place FullAttention and Mamba layers in one
    virtual tensor even though Merak allocates distinct native HMONNX caches
    for those layers.  Virtual tensors validate layer coverage only; they never
    define physical ownership.

    Models with true physical cache sharing must provide an explicit
    ``physical_owner_by_layer`` contract.  Gemma4 contract-v2 is currently the
    only such model.
    """

    finalized_layer_names = {
        str(layer_name) for group in kv_cache_groups for layer_name in group.layer_names
    }

    has_explicit_physical_owners = physical_owner_by_layer is not None
    if has_explicit_physical_owners:
        assert physical_owner_by_layer is not None
        normalized_owners = {
            str(layer_name): str(owner_name)
            for layer_name, owner_name in physical_owner_by_layer.items()
        }
        missing_owner_entries = finalized_layer_names - set(normalized_owners)
        if missing_owner_entries:
            raise ValueError(
                "Physical KV owner contract does not cover finalized layers: "
                f"missing={sorted(missing_owner_entries)}"
            )
        # vLLM omits KV-sharing aliases from finalized cache groups. The
        # model contract still lists those logical layers so runtime binding
        # can resolve them to their physical owner; capacity accounting only
        # needs the finalized owner entries.
        logical_aliases = set(normalized_owners) - finalized_layer_names
        invalid_alias_targets = sorted(
            alias
            for alias in logical_aliases
            if normalized_owners[alias] not in finalized_layer_names
        )
        if invalid_alias_targets:
            raise ValueError(
                "Physical KV sharing aliases target non-finalized owners: "
                f"{invalid_alias_targets}"
            )
        normalized_owners = {
            layer_name: normalized_owners[layer_name]
            for layer_name in finalized_layer_names
        }
        physical_owners = set(normalized_owners.values())
        unknown_owners = sorted(physical_owners - finalized_layer_names)
        if unknown_owners:
            raise ValueError(
                f"Physical KV owner contract targets unknown layers: {unknown_owners}"
            )
        noncanonical_owners = sorted(
            owner for owner in physical_owners if normalized_owners.get(owner) != owner
        )
        if noncanonical_owners:
            raise ValueError(
                f"Physical KV owners must map to themselves: {noncanonical_owners}"
            )

    spec_by_layer: dict[str, KVCacheSpec] = {}
    for group in kv_cache_groups:
        resolved_specs = (
            resolve_finalized_group_layer_specs(
                group,
                shared_layer_targets=normalized_owners,
            )
            if has_explicit_physical_owners
            else resolve_finalized_group_layer_specs(group)
        )
        for layer_name, layer_spec in resolved_specs.items():
            if layer_name in spec_by_layer:
                raise ValueError(
                    f"KV cache layer {layer_name!r} appears in more than one group"
                )
            spec_by_layer[layer_name] = layer_spec
    if has_explicit_physical_owners and set(spec_by_layer) != finalized_layer_names:
        raise ValueError(
            "Physical KV owner contract could not resolve every finalized layer spec"
        )

    if has_explicit_physical_owners:
        for layer_name, owner_name in normalized_owners.items():
            if spec_by_layer[layer_name] != spec_by_layer[owner_name]:
                raise ValueError(
                    "Physical KV sharing alias has incompatible owner spec: "
                    f"layer={layer_name!r}, owner={owner_name!r}"
                )
    else:
        normalized_owners = {layer_name: layer_name for layer_name in spec_by_layer}

    if kv_cache_tensors is not None:
        packed_layers = [
            str(layer_name)
            for tensor in kv_cache_tensors
            for layer_name in tensor.shared_by
        ]
        if len(packed_layers) != len(set(packed_layers)):
            raise ValueError("Finalized virtual KVCacheTensor packing repeats a layer")
        if set(packed_layers) != finalized_layer_names:
            raise ValueError(
                "Finalized virtual KVCacheTensor packing does not cover "
                "finalized cache layers"
            )

    normalized_page_counts = {
        str(layer_name): int(num_pages)
        for layer_name, num_pages in (physical_pages_by_layer or {}).items()
    }
    unexpected_page_counts = sorted(
        set(normalized_page_counts) - set(normalized_owners.values())
    )
    if unexpected_page_counts:
        raise ValueError(
            "Fixed physical page counts target non-owner layers: "
            f"{unexpected_page_counts}"
        )
    invalid_page_counts = sorted(
        layer_name
        for layer_name, num_pages in normalized_page_counts.items()
        if num_pages <= 0
    )
    if invalid_page_counts:
        raise ValueError(
            f"Fixed physical page counts must be positive: {invalid_page_counts}"
        )

    attention_layers: list[PhysicalLayer] = []
    linear_layers: list[PhysicalLayer] = []
    emitted_owners: set[str] = set()
    for layer_name in spec_by_layer:
        owner_name = normalized_owners[layer_name]
        if owner_name in emitted_owners:
            continue
        emitted_owners.add(owner_name)
        spec = spec_by_layer[owner_name]
        if isinstance(spec, (SlidingWindowSpec, FullAttentionSpec)):
            page_size = int(spec.real_page_size_bytes)
            destination = attention_layers
            device = layer_devices.get(owner_name, linear_default_device)
        elif isinstance(spec, MambaSpec):
            page_size = _native_mamba_page_size_bytes(spec)
            destination = linear_layers
            device = linear_default_device
        else:
            raise ValueError(
                f"Unsupported physical Merak KV cache owner spec {type(spec).__name__}"
            )
        destination.append(
            PhysicalLayer(
                name=owner_name,
                device=str(device),
                page_size_bytes=page_size,
                num_pages=normalized_page_counts.get(owner_name),
            )
        )
    if emitted_owners != set(normalized_owners.values()):
        raise ValueError("Physical KV owner projection omitted declared owners")
    return tuple(attention_layers), tuple(linear_layers)


@dataclass(frozen=True)
class DeviceProjection:
    full_bytes: int = 0
    linear_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.full_bytes + self.linear_bytes


@dataclass(frozen=True)
class PhysicalProjection:
    num_blocks: int
    linear_slots: int
    full_bytes: int
    linear_bytes: int
    total_bytes: int
    padding_saved_bytes: int
    by_device: dict[str, DeviceProjection]


def project_physical_storage(
    *,
    num_blocks: int,
    full_layers: tuple[PhysicalLayer, ...],
    linear_layers: tuple[PhysicalLayer, ...],
    linear_slots: int,
    virtual_effective_bytes: int,
) -> PhysicalProjection:
    """Project exact HMFP pages and slot-bounded Linear state by device."""

    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")
    if linear_slots <= 0:
        raise ValueError(f"linear_slots must be positive, got {linear_slots}")

    per_device: dict[str, list[int]] = {}
    for layer in full_layers:
        totals = per_device.setdefault(layer.device, [0, 0])
        page_count = layer.num_pages if layer.num_pages is not None else num_blocks
        totals[0] += page_count * layer.page_size_bytes
    for layer in linear_layers:
        totals = per_device.setdefault(layer.device, [0, 0])
        totals[1] += linear_slots * layer.page_size_bytes

    by_device = {
        device: DeviceProjection(full_bytes=totals[0], linear_bytes=totals[1])
        for device, totals in sorted(per_device.items())
    }
    full_bytes = sum(item.full_bytes for item in by_device.values())
    linear_bytes = sum(item.linear_bytes for item in by_device.values())
    total_bytes = full_bytes + linear_bytes
    return PhysicalProjection(
        num_blocks=num_blocks,
        linear_slots=linear_slots,
        full_bytes=full_bytes,
        linear_bytes=linear_bytes,
        total_bytes=total_bytes,
        padding_saved_bytes=virtual_effective_bytes - total_bytes,
        by_device=by_device,
    )


def _bytes_and_gib(value: int) -> str:
    return f"{value} bytes ({value / GIB:.6f} GiB)"


def format_capacity_log_lines(
    plan: CapacityPlan,
    projection: PhysicalProjection,
) -> tuple[str, ...]:
    """Return byte-reconcilable startup messages without allocator claims."""

    lines = [
        "Merak virtual scheduler budget: "
        f"source={plan.source}; budget={_bytes_and_gib(plan.virtual_budget_bytes)}; "
        f"effective={_bytes_and_gib(plan.virtual_effective_bytes)}; "
        f"virtual_page={plan.virtual_page_size_bytes} bytes; "
        f"virtual_bytes_per_global_block={plan.virtual_bytes_per_block}; "
        f"global_blocks={plan.num_blocks} (required={plan.required_blocks}, "
        f"recommended={plan.recommended_blocks}); token_capacity={plan.token_capacity}; "
        f"max_concurrency={plan.max_concurrency:.6f}x",
    ]
    for device, item in projection.by_device.items():
        lines.append(
            "Merak physical cache backing only: "
            f"device={device}; HMFP={_bytes_and_gib(item.full_bytes)}; "
            f"Linear={_bytes_and_gib(item.linear_bytes)}; "
            f"total={_bytes_and_gib(item.total_bytes)}"
        )
    lines.append(
        "Merak physical cache total: "
        f"HMFP={_bytes_and_gib(projection.full_bytes)}; "
        f"Linear={_bytes_and_gib(projection.linear_bytes)} "
        f"({projection.linear_slots} slot(s)/layer); "
        f"total={_bytes_and_gib(projection.total_bytes)}; "
        f"virtual padding saved={_bytes_and_gib(projection.padding_saved_bytes)}. "
        "Model weights, runtime workspaces, allocator reservations, and other process memory are excluded."
    )
    return tuple(lines)
