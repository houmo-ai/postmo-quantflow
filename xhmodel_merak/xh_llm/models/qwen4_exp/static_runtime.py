"""Model-owned static runtime pieces for Qwen4-Exp export.

The public Qwen4 export boundary is a fixed tensor program.  This module keeps
the heterogeneous cache contract next to that boundary so the generic Merak
state machine can trace, quantize, and export the same prefill/decode wrapper.
It deliberately contains no ONNX conversion or serving-package lowering.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from xhquant.core import CacheTensor

from ...kv_cache_mixin import KVCacheMixin
from ...llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor
from ...types import KVCacheConfig
from .cache_contract import Qwen4ExpCacheContract, resolve_qwen4_exp_cache_shape


def _mrope_inputs(*, batch_size: int, input_length: int, past_length: int, device: torch.device):
    current = torch.arange(
        past_length, past_length + input_length, dtype=torch.int32, device=device
    ).view(1, -1).expand(batch_size, -1)
    first_group = past_length // 4
    pooled = (
        torch.arange(
            first_group,
            first_group + (input_length + 3) // 4,
            dtype=torch.int32,
            device=device,
        )
        .mul(4)
        .view(1, -1)
        .expand(batch_size, -1)
    )
    return (*(current.clone() for _ in range(3)), *(pooled.clone() for _ in range(3)))


class Qwen4ExpCacheMixin(KVCacheMixin):
    """Own the heterogeneous Qwen4 cache tensors during generic export.

    The generic ``KVCacheMixin`` only knows paired K/V lists.  Qwen4 also has
    DeltaNet, PLE, QSA indexer and tail state, so all tensors are allocated and
    returned in ``Qwen4ExpCacheContract`` order here.  K/V and indexer cache
    entries retain ``CacheTensor`` identity because the graph mutates them.
    """

    def __init__(
        self,
        contract: Qwen4ExpCacheContract,
        *,
        batch_size: int,
        context_length: int,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
        pad_token_id: int = 0,
        ple_eos_token_id: int | None = None,
        use_cache: bool = True,
    ):
        attention_layers = sum(
            any(spec.state_kind == "kv_key" for spec in layer.tensors)
            for layer in contract.layers
        )
        cache_config = KVCacheConfig(
            num_layers=attention_layers,
            kv_cache_shape=[
                [batch_size, contract.num_key_value_heads, context_length, contract.head_dim],
                [batch_size, contract.num_key_value_heads, context_length, contract.head_dim],
            ],
            cache_axis=2,
            batch_size=int(batch_size),
            cache_dtype=str(dtype).removeprefix("torch."),
            use_cache=bool(use_cache),
        )
        super().__init__(cache_config)
        self.contract = contract
        self.batch_size = int(batch_size)
        self.context_length = int(context_length)
        self.cache_dtype = dtype
        self.pad_token_id = int(pad_token_id)
        self.ple_eos_token_id = self.pad_token_id if ple_eos_token_id is None else int(ple_eos_token_id)
        self._device = torch.device(device)
        # KVCacheMixin initializes this flag today, but keep the model-local
        # contract explicit so cache allocation remains stable across SDK
        # versions and direct construction in tests.
        self._cache_initialized = False
        self.tensors: dict[str, torch.Tensor] = {}
        self.inplace_state_names: set[str] = set()

    @staticmethod
    def _cache_dtype_for_spec(spec, floating_dtype):
        return floating_dtype if spec.dtype == "floating" else torch.int32

    def _allocate(self, device: torch.device) -> None:
        self.tensors = {}
        self.past_key_caches.clear()
        self.past_value_caches.clear()
        for spec in self.contract.tensors:
            shape = resolve_qwen4_exp_cache_shape(
                spec, batch_size=self.batch_size, context_length=self.context_length
            )
            cache_dtype = self._cache_dtype_for_spec(spec, self.cache_dtype)
            if spec.state_kind == "ple_ngram_history":
                value = torch.full(shape, self.ple_eos_token_id, dtype=cache_dtype, device=device)
            else:
                value = torch.zeros(shape, dtype=cache_dtype, device=device)
            if spec.state_kind in {"kv_key", "kv_value", "qsa_indexer_key"} or spec.name in self.inplace_state_names:
                value = CacheTensor(value)
            self.tensors[spec.name] = value
            if spec.state_kind == "kv_key":
                self.past_key_caches.append(value)
            elif spec.state_kind == "kv_value":
                self.past_value_caches.append(value)

    def prepare_kv_cache(self):
        if not self.use_cache:
            return
        if not self._cache_initialized:
            self._allocate(torch.device(self._device or "cpu"))
            self._cache_initialized = True

    def clear_kv_cache(self):
        self.tensors.clear()
        self.past_key_caches.clear()
        self.past_value_caches.clear()
        self._cache_initialized = False

    def prepare_other_cache(self):
        # All Qwen4 state is allocated together to keep one stable ABI.
        self.prepare_kv_cache()

    def clear_other_cache(self):
        # ``clear_kv_cache`` owns the shared allocation.  Keep this method for
        # KVCacheMixin's lifecycle contract and avoid a second clear.
        return None

    def _prepare_other_cache_on_device(self, device):
        requested = torch.device(device)
        self._device = torch.device("cpu") if requested.type == "meta" else requested
        self.prepare_other_cache()

    def to(self, device=None, dtype=None):
        super().to(device=device, dtype=dtype)
        if device is not None:
            self._device = torch.device(device)
        if dtype is not None:
            self.cache_dtype = dtype
            self.kvcache_config.cache_dtype = str(dtype).removeprefix("torch.")
        if not self.tensors:
            return self
        for name, value in tuple(self.tensors.items()):
            target_dtype = dtype if dtype is not None and value.is_floating_point() else None
            moved = value.to(device=self._device, dtype=target_dtype)
            # CacheTensor.to mutates in place in the runtime.  Preserve the
            # original object even with SDK versions that return a view.
            if isinstance(value, CacheTensor):
                self.tensors[name] = value
            else:
                self.tensors[name] = moved
        self.past_key_caches[:] = [self.tensors[s.name] for s in self.contract.tensors if s.state_kind == "kv_key"]
        self.past_value_caches[:] = [self.tensors[s.name] for s in self.contract.tensors if s.state_kind == "kv_value"]
        return self

    def inputs_in_contract_order(self) -> tuple[torch.Tensor, ...]:
        if not self._cache_initialized:
            self.prepare_kv_cache()
        missing = [name for name in self.contract.expected_keys() if name not in self.tensors]
        if missing:
            raise RuntimeError(f"Qwen4-Exp cache is not initialized: {missing}")
        return tuple(self.tensors[name] for name in self.contract.expected_keys())

    def commit(self, outputs, output_names=None):
        """Commit explicit wrapper state only after validating every update."""
        values = tuple(outputs)
        if not values:
            raise RuntimeError("Qwen4-Exp wrapper returned no outputs")
        names = tuple(output_names or ())
        if not names:
            raise ValueError("Qwen4-Exp cache commit requires explicit output names")
        if len(values) != len(names):
            raise ValueError("Qwen4-Exp cache output count does not match the declared ABI")
        if len(set(names)) != len(names):
            raise ValueError("Qwen4-Exp cache output names must be unique")
        pending: dict[str, torch.Tensor] = {}
        for name, value in zip(names[1:], values[1:], strict=True):
            state_name = str(name).removeprefix("present_")
            if state_name not in self.tensors:
                raise ValueError(f"Unknown Qwen4-Exp cache output: {name}")
            expected = self.tensors[state_name]
            if not torch.is_tensor(value) or value.dtype != expected.dtype:
                raise ValueError(f"Qwen4-Exp cache state {state_name!r} has incompatible dtype")
            if tuple(value.shape) != tuple(expected.shape):
                raise ValueError(
                    f"Qwen4-Exp cache state {state_name!r} shape changed: "
                    f"{tuple(value.shape)} != {tuple(expected.shape)}"
                )
            pending[state_name] = value
        for state_name, value in pending.items():
            current = self.tensors[state_name]
            if isinstance(current, CacheTensor):
                current.copy_(value)
            else:
                self.tensors[state_name] = value
        self.past_key_caches[:] = [
            self.tensors[s.name] for s in self.contract.tensors if s.state_kind == "kv_key"
        ]
        self.past_value_caches[:] = [
            self.tensors[s.name] for s in self.contract.tensors if s.state_kind == "kv_value"
        ]
        return values[0]


class Qwen4ExpDataPreprocess(BaseLLMInputProcessor):
    """Build the static wrapper inputs with an explicit cache/control tuple."""

    def __init__(self, *, token_embedding, input_sequence_length: int, cache_mixin: Qwen4ExpCacheMixin,
                 pad_token_id: int, qsa_topk: int | None = None):
        super().__init__(
            BaseInputProcessorConfig(
                token_embedding, input_sequence_length,
                cache_mixin.past_key_caches, cache_mixin.past_value_caches,
                pad_token_id=pad_token_id,
            )
        )
        self.cache_mixin = cache_mixin
        self.qsa_topk = qsa_topk

    def forward(self, data: Mapping[str, Any]):
        if not isinstance(data, Mapping):
            raise TypeError(f"Qwen4ExpDataPreprocess expects a mapping, got {type(data).__name__}")
        input_ids = data.get("input_ids")
        inputs_embeds = data.get("inputs_embeds")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            if input_ids.ndim != 2 or input_ids.shape[0] != self.cache_mixin.batch_size:
                raise ValueError("Qwen4-Exp export requires a fixed batch")
            supplied_length = int(input_ids.shape[1])
            ple_input_ids = input_ids
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != self.cache_mixin.batch_size:
                raise ValueError("inputs_embeds must be [batch, sequence, hidden]")
            supplied_length = int(inputs_embeds.shape[1])
            ple_input_ids = data.get("ple_input_ids")
            if ple_input_ids is None:
                raise ValueError("ple_input_ids is required with inputs_embeds")
        if supplied_length > self.input_sequence_length:
            raise ValueError("input exceeds the stage sequence length")
        current_length = int(data.get("current_input_length", supplied_length))
        if current_length <= 0 or current_length > self.input_sequence_length:
            raise ValueError("current_input_length is outside the stage width")
        past_length = int(data.get("past_seq_length", 0))
        if past_length < 0 or past_length + self.input_sequence_length > self.cache_mixin.context_length:
            raise ValueError("past sequence exceeds Qwen4 context capacity")
        if supplied_length < self.input_sequence_length:
            pad = self.input_sequence_length - supplied_length
            pad_ids = torch.full(
                (self.cache_mixin.batch_size, pad), self.pad_token_id, dtype=torch.long, device=inputs_embeds.device
            )
            inputs_embeds = torch.cat((inputs_embeds, self.embed_tokens(pad_ids)), dim=1)
            ple_input_ids = torch.cat((ple_input_ids, pad_ids), dim=1)
        positions = _mrope_inputs(
            batch_size=self.cache_mixin.batch_size,
            input_length=self.input_sequence_length,
            past_length=past_length,
            device=inputs_embeds.device,
        )
        controls = ()
        if self.qsa_topk is not None:
            from .qsa_host_controls import build_qsa_host_controls

            controls = build_qsa_host_controls(
                past_length, current_length, self.input_sequence_length,
                self.cache_mixin.context_length, self.qsa_topk,
                device=inputs_embeds.device, dtype=inputs_embeds.dtype,
            )
        return (
            inputs_embeds.to(self._device),
            torch.tensor([past_length], dtype=torch.int32, device=self._device),
            torch.tensor([current_length], dtype=torch.int32, device=self._device),
            ple_input_ids.to(self._device, dtype=torch.int32),
            *(value.to(self._device) for value in positions),
            tuple(value.to(self._device) for value in (*self.cache_mixin.inputs_in_contract_order(), *controls)),
        )


__all__ = ["Qwen4ExpCacheMixin", "Qwen4ExpDataPreprocess"]
