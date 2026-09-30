# SPDX-License-Identifier: Apache-2.0
"""Sparse, slot-backed checkpoints for Merak Linear Attention state.

The vLLM hybrid scheduler exposes global physical block IDs for every Mamba
group.  Those IDs are scheduling identities, not a reason to allocate one
state tensor per global block.  This store keeps a fixed number of state slots
per Linear layer (``max_num_seqs``) and binds scheduler identities to those
slots with explicit request ownership and completeness.

Phase 1 configures one slot, prefix caching off, and serial execution.  The
group-qualified key and invalidation contract are nevertheless explicit so a
later phase can expand lifecycle support without changing the data model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch


LayerState = tuple[torch.Tensor, ...]
GroupPhysicalIDs = Mapping[int, int]


@dataclass(frozen=True)
class LinearStateLayerSpec:
    """Storage metadata for one HMONNX Linear Attention layer."""

    layer_name: str
    group_id: int
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    device: torch.device | str = torch.device("cpu")

    def __post_init__(self) -> None:
        if not self.layer_name:
            raise ValueError("linear layer_name must not be empty")
        if self.group_id < 0:
            raise ValueError(f"linear layer {self.layer_name} has negative group_id")
        normalized_shapes = tuple(
            tuple(int(dim) for dim in shape) for shape in self.shapes
        )
        if not normalized_shapes:
            raise ValueError(
                f"linear layer {self.layer_name} must expose at least one state tensor"
            )
        if len(normalized_shapes) != len(self.dtypes):
            raise ValueError(
                f"linear layer {self.layer_name} has {len(normalized_shapes)} shapes "
                f"but {len(self.dtypes)} dtypes"
            )
        for state_idx, shape in enumerate(normalized_shapes):
            if not shape or any(dim <= 0 for dim in shape):
                raise ValueError(
                    f"linear layer {self.layer_name} state {state_idx} has invalid shape {shape}"
                )
        if any(not isinstance(dtype, torch.dtype) for dtype in self.dtypes):
            raise TypeError(
                f"linear layer {self.layer_name} dtypes must be torch.dtype values"
            )
        object.__setattr__(self, "shapes", normalized_shapes)
        object.__setattr__(self, "dtypes", tuple(self.dtypes))
        object.__setattr__(self, "device", torch.device(self.device))


@dataclass(frozen=True)
class LinearCheckpointSlot:
    """Stable views for one active request's compact state slot."""

    index: int
    request_id: str
    states_by_layer: Mapping[str, LayerState]


@dataclass(frozen=True)
class _Binding:
    slot: int
    request_id: str
    complete: bool


class MerakLinearCheckpointStore:
    """Fixed-capacity sparse checkpoint store keyed by ``(group, physical)``.

    No constructor argument represents the global scheduler block count.  The
    only leading storage dimension is ``max_num_seqs``.
    """

    def __init__(
        self,
        layer_specs: Sequence[LinearStateLayerSpec],
        *,
        max_num_seqs: int,
        storage_by_layer: Mapping[str, Sequence[torch.Tensor]] | None = None,
    ) -> None:
        specs = tuple(layer_specs)
        if not specs:
            raise ValueError(
                "Merak Linear checkpoint store requires at least one layer"
            )
        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")

        layer_names = [spec.layer_name for spec in specs]
        duplicates = sorted(
            {name for name in layer_names if layer_names.count(name) > 1}
        )
        if duplicates:
            raise ValueError(
                "duplicate Linear checkpoint layer name(s): " + ", ".join(duplicates)
            )

        self._specs = specs
        self._spec_by_layer = {spec.layer_name: spec for spec in specs}
        group_layers: dict[int, list[str]] = {}
        for spec in specs:
            group_layers.setdefault(spec.group_id, []).append(spec.layer_name)
        self._group_layers = {
            group_id: tuple(names) for group_id, names in sorted(group_layers.items())
        }
        self._group_ids = tuple(self._group_layers)
        self._max_num_seqs = int(max_num_seqs)

        if storage_by_layer is None:
            self._storage = {
                spec.layer_name: tuple(
                    torch.zeros(
                        (self._max_num_seqs, *shape),
                        dtype=dtype,
                        device=spec.device,
                    )
                    for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True)
                )
                for spec in specs
            }
        else:
            expected_names = tuple(layer_names)
            storage_names = tuple(storage_by_layer)
            if set(storage_names) != set(expected_names):
                raise ValueError(
                    "Linear checkpoint storage layer set mismatch: "
                    f"missing={sorted(set(expected_names) - set(storage_names))}, "
                    f"extra={sorted(set(storage_names) - set(expected_names))}"
                )
            if storage_names != expected_names:
                raise ValueError(
                    "Linear checkpoint storage layer order mismatch: "
                    f"expected={expected_names}, got={storage_names}"
                )

            adopted_storage: dict[str, LayerState] = {}
            for spec in specs:
                states = tuple(storage_by_layer[spec.layer_name])
                if len(states) != len(spec.shapes):
                    raise ValueError(
                        f"Linear checkpoint storage {spec.layer_name} state count "
                        f"{len(states)}; expected {len(spec.shapes)}"
                    )
                for state_idx, (state, shape, dtype) in enumerate(
                    zip(states, spec.shapes, spec.dtypes, strict=True)
                ):
                    if not isinstance(state, torch.Tensor):
                        raise TypeError(
                            f"Linear checkpoint storage {spec.layer_name} state {state_idx} "
                            "must be a Tensor"
                        )
                    expected_shape = (self._max_num_seqs, *shape)
                    if tuple(state.shape) != expected_shape:
                        raise ValueError(
                            f"Linear checkpoint storage {spec.layer_name} state {state_idx} "
                            f"shape {tuple(state.shape)}; expected {expected_shape}"
                        )
                    if state.dtype != dtype:
                        raise ValueError(
                            f"Linear checkpoint storage {spec.layer_name} state {state_idx} "
                            f"dtype {state.dtype}; expected {dtype}"
                        )
                    if state.device != spec.device:
                        raise ValueError(
                            f"Linear checkpoint storage {spec.layer_name} state {state_idx} "
                            f"device {state.device}; expected {spec.device}"
                        )
                adopted_storage[spec.layer_name] = states
            self._storage = adopted_storage

        self._bindings: dict[tuple[int, int], _Binding] = {}
        self._request_slot: dict[str, int] = {}
        self._request_keys: dict[str, set[tuple[int, int]]] = {}
        self._written_layers: dict[str, set[str]] = {}

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(spec.layer_name for spec in self._specs)

    @property
    def group_layer_names(self) -> Mapping[int, tuple[str, ...]]:
        return MappingProxyType(self._group_layers)

    @property
    def slot_capacity(self) -> int:
        return self._max_num_seqs

    @property
    def active_request_count(self) -> int:
        return len(self._request_slot)

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        """Request owners in stable slot-acquisition order."""
        return tuple(self._request_slot)

    @property
    def entry_count(self) -> int:
        return len(self._bindings)

    @property
    def allocated_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for states in self._storage.values()
            for tensor in states
        )

    def storage_for_layer(self, layer_name: str) -> LayerState:
        try:
            return self._storage[layer_name]
        except KeyError as exc:
            raise KeyError(f"unknown linear layer {layer_name}") from exc

    def storage_pointers(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (spec.layer_name, state_idx, tensor.data_ptr())
            for spec in self._specs
            for state_idx, tensor in enumerate(self._storage[spec.layer_name])
        )

    def _normalize_physical_ids(
        self,
        group_physical_ids: GroupPhysicalIDs,
    ) -> tuple[tuple[int, int], ...]:
        supplied = set(group_physical_ids)
        expected = set(self._group_ids)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        if missing or extra:
            raise ValueError(
                "Linear checkpoint physical IDs must cover every group exactly once: "
                f"missing={missing}, extra={extra}"
            )
        keys = []
        for group_id in self._group_ids:
            physical_id = int(group_physical_ids[group_id])
            if physical_id < 0:
                raise ValueError(
                    f"Linear checkpoint group {group_id} has negative physical ID {physical_id}"
                )
            keys.append((group_id, physical_id))
        return tuple(keys)

    def _slot_views(self, request_id: str, slot: int) -> LinearCheckpointSlot:
        states_by_layer = {
            spec.layer_name: tuple(
                tensor[slot] for tensor in self._storage[spec.layer_name]
            )
            for spec in self._specs
        }
        return LinearCheckpointSlot(
            index=slot,
            request_id=request_id,
            states_by_layer=MappingProxyType(states_by_layer),
        )

    def _zero_slot(self, slot: int) -> None:
        for states in self._storage.values():
            for tensor in states:
                tensor[slot].zero_()

    def _free_slots(self) -> list[int]:
        used = set(self._request_slot.values())
        return [slot for slot in range(self._max_num_seqs) if slot not in used]

    def acquire(
        self,
        request_id: str,
        group_physical_ids: GroupPhysicalIDs,
        *,
        past_len: int,
    ) -> LinearCheckpointSlot:
        """Reserve a clean slot for a new request, or fail closed on resume."""
        if past_len > 0:
            return self.load_slot(request_id, group_physical_ids, past_len=past_len)
        if past_len < 0:
            raise ValueError("past_len must be non-negative")
        if not request_id:
            raise ValueError("request_id must not be empty")
        if request_id in self._request_slot:
            raise RuntimeError(
                f"request {request_id} already owns a Linear checkpoint slot"
            )

        keys = self._normalize_physical_ids(group_physical_ids)
        for group_id, physical_id in keys:
            binding = self._bindings.get((group_id, physical_id))
            if binding is not None:
                raise RuntimeError(
                    f"Linear checkpoint group {group_id}, physical ID {physical_id} "
                    f"is owned by request {binding.request_id}; invalidate or release it before reuse."
                )

        free_slots = self._free_slots()
        if not free_slots:
            raise RuntimeError(
                "Merak Linear checkpoint slots are exhausted: "
                f"active={len(self._request_slot)}, capacity={self._max_num_seqs}."
            )
        slot = free_slots[0]
        self._zero_slot(slot)
        self._request_slot[request_id] = slot
        self._request_keys[request_id] = set(keys)
        self._written_layers[request_id] = set()
        for key in keys:
            self._bindings[key] = _Binding(
                slot=slot,
                request_id=request_id,
                complete=False,
            )
        return self._slot_views(request_id, slot)

    def load_slot(
        self,
        request_id: str,
        group_physical_ids: GroupPhysicalIDs,
        *,
        past_len: int,
    ) -> LinearCheckpointSlot:
        """Resolve a complete scheduler checkpoint for a resumed request."""
        if past_len <= 0:
            raise ValueError("load_slot requires past_len > 0")
        keys = self._normalize_physical_ids(group_physical_ids)
        slot: int | None = None
        for group_id, physical_id in keys:
            binding = self._bindings.get((group_id, physical_id))
            if binding is None:
                raise RuntimeError(
                    f"past_len={past_len} request {request_id} is missing complete linear "
                    f"checkpoint for group {group_id}, physical ID {physical_id}; fail closed."
                )
            if binding.request_id != request_id:
                raise RuntimeError(
                    f"Linear checkpoint group {group_id}, physical ID {physical_id} "
                    f"belongs to request {binding.request_id}, not {request_id}."
                )
            if not binding.complete:
                raise RuntimeError(
                    f"past_len={past_len} request {request_id} is missing complete linear "
                    f"checkpoint for group {group_id}, physical ID {physical_id}; fail closed."
                )
            if slot is None:
                slot = binding.slot
            elif binding.slot != slot:
                raise RuntimeError(
                    f"request {request_id} Linear checkpoint groups resolve to different slots: "
                    f"{slot} and {binding.slot}."
                )

        request_slot = self._request_slot.get(request_id)
        if (
            slot is None
            or request_slot != slot
            or self._request_keys.get(request_id) != set(keys)
        ):
            raise RuntimeError(
                f"request {request_id} Linear checkpoint ownership is incomplete or stale; fail closed."
            )
        return self._slot_views(request_id, slot)

    def layer_state(self, request_id: str, layer_name: str) -> LayerState:
        try:
            slot = self._request_slot[request_id]
        except KeyError as exc:
            raise KeyError(
                f"request {request_id} does not own a Linear checkpoint slot"
            ) from exc
        states = self.storage_for_layer(layer_name)
        return tuple(tensor[slot] for tensor in states)

    def store_layer_state(
        self,
        request_id: str,
        layer_name: str,
        states: Sequence[torch.Tensor],
    ) -> None:
        try:
            slot = self._request_slot[request_id]
        except KeyError as exc:
            raise KeyError(
                f"request {request_id} does not own a Linear checkpoint slot"
            ) from exc
        try:
            spec = self._spec_by_layer[layer_name]
        except KeyError as exc:
            raise KeyError(f"unknown linear layer {layer_name}") from exc
        values = tuple(states)
        if len(values) != len(spec.shapes):
            raise ValueError(
                f"{layer_name} state count {len(values)} does not match expected {len(spec.shapes)}"
            )
        for state_idx, (value, shape, dtype) in enumerate(
            zip(values, spec.shapes, spec.dtypes, strict=True)
        ):
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{layer_name} state {state_idx} shape {tuple(value.shape)}; expected {shape}"
                )
            if value.dtype != dtype:
                raise ValueError(
                    f"{layer_name} state {state_idx} dtype {value.dtype}; expected {dtype}"
                )

        for destination, value in zip(self._storage[layer_name], values, strict=True):
            destination[slot].copy_(value)
        self._written_layers[request_id].add(layer_name)
        # Once state is being updated the previous scheduler binding no longer
        # represents an immutable complete checkpoint. Keep it fail-closed
        # until an atomic transition commits every layer.
        for key in self._request_keys[request_id]:
            binding = self._bindings[key]
            self._bindings[key] = _Binding(
                slot=binding.slot,
                request_id=binding.request_id,
                complete=False,
            )

    def transition(
        self,
        request_id: str,
        group_physical_ids: GroupPhysicalIDs,
    ) -> LinearCheckpointSlot:
        """Atomically commit complete state under replacement physical IDs."""
        try:
            slot = self._request_slot[request_id]
        except KeyError as exc:
            raise KeyError(
                f"request {request_id} does not own a Linear checkpoint slot"
            ) from exc
        new_keys = self._normalize_physical_ids(group_physical_ids)

        for group_id, physical_id in new_keys:
            binding = self._bindings.get((group_id, physical_id))
            if binding is not None and binding.request_id != request_id:
                raise RuntimeError(
                    f"Linear checkpoint group {group_id}, physical ID {physical_id} "
                    f"is owned by request {binding.request_id}; transition aborted."
                )

        written = self._written_layers[request_id]
        missing_layers = [
            layer_name for layer_name in self.layer_names if layer_name not in written
        ]
        if missing_layers:
            raise RuntimeError(
                f"request {request_id} Linear checkpoint is incomplete; missing layer state(s): "
                + ", ".join(missing_layers)
            )

        old_keys = self._request_keys[request_id]
        for key in old_keys:
            binding = self._bindings.get(key)
            if binding is not None and binding.request_id == request_id:
                del self._bindings[key]
        committed_keys = set(new_keys)
        for key in committed_keys:
            self._bindings[key] = _Binding(
                slot=slot,
                request_id=request_id,
                complete=True,
            )
        self._request_keys[request_id] = committed_keys
        self._written_layers[request_id] = set()
        return self._slot_views(request_id, slot)

    def _release_request_storage(self, request_id: str) -> None:
        slot = self._request_slot.pop(request_id)
        self._request_keys.pop(request_id, None)
        self._written_layers.pop(request_id, None)
        self._zero_slot(slot)

    def invalidate_physical_ids(
        self,
        physical_ids_by_group: Mapping[int, Iterable[int]],
    ) -> None:
        """Invalidate scheduler IDs, releasing owners once no bindings remain."""
        unknown_groups = sorted(set(physical_ids_by_group) - set(self._group_ids))
        if unknown_groups:
            raise ValueError(f"unknown Linear checkpoint group(s): {unknown_groups}")
        for group_id, physical_ids in physical_ids_by_group.items():
            for raw_physical_id in physical_ids:
                physical_id = int(raw_physical_id)
                if physical_id < 0:
                    raise ValueError(
                        f"Linear checkpoint group {group_id} has negative physical ID {physical_id}"
                    )
                key = (group_id, physical_id)
                binding = self._bindings.pop(key, None)
                if binding is None:
                    continue
                request_keys = self._request_keys[binding.request_id]
                request_keys.discard(key)
                if not request_keys:
                    self._release_request_storage(binding.request_id)

    def release_request(self, request_id: str) -> None:
        keys = self._request_keys.get(request_id)
        if keys is None:
            return
        for key in tuple(keys):
            binding = self._bindings.get(key)
            if binding is not None and binding.request_id == request_id:
                del self._bindings[key]
        self._release_request_storage(request_id)

    def has_entry(self, group_id: int, physical_id: int) -> bool:
        return (int(group_id), int(physical_id)) in self._bindings

    def is_complete(self, group_id: int, physical_id: int) -> bool:
        binding = self._bindings.get((int(group_id), int(physical_id)))
        return binding is not None and binding.complete


__all__ = [
    "GroupPhysicalIDs",
    "LayerState",
    "LinearCheckpointSlot",
    "LinearStateLayerSpec",
    "MerakLinearCheckpointStore",
]
