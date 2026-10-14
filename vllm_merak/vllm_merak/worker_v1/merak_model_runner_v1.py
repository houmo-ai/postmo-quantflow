# SPDX-License-Identifier: Apache-2.0

import itertools
import math
from types import MethodType
from typing import Any, cast

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.tasks import SupportedTask
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, KVCacheTensor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.block_table import PAD_SLOT_ID
from vllm.v1.worker.gpu_model_runner import (
    AsyncModelRunnerOutput,
    GPUModelRunner,
    IntermediateTensors,
)

from ..attention.backends.merak import (
    MerakAttentionMetadata,
    MerakPageAttentionGroupContext,
)
from ..constants import (
    HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST,
    MERAK_DECODE_QUERY_LENGTH,
    MERAK_MAX_NARROW_QUERY_LENGTH,
    MERAK_MIN_VERIFY_QUERY_LENGTH,
)
from ..determinism_trace import emit_event as emit_determinism_event
from ..determinism_trace import fingerprint as determinism_fingerprint
from ..determinism_trace import trace_enabled as determinism_trace_enabled
from ..gemma4_trace import KV_TRACE_ENV, emit_trace, trace_enabled
from ..hmfp_kv_cache import (
    HMFPFullAttentionSpec,
    HMFPPagedKVCache,
    HMFPSlidingWindowSpec,
    get_hmfp_cache_block_size_bytes,
    hmfp_paged_kv_cache_from_raw_tensor,
    is_hmfp_cache_dtype,
)
from ..kv_capacity import resolve_finalized_group_layer_specs
from ..model_executor import get_model
from ..model_executor.model_loader.merak_linear_checkpoint_store import (
    LinearStateLayerSpec,
    MerakLinearCheckpointStore,
)

logger = init_logger(__name__)


def _validate_merak_narrow_query_length(
    execution_phase: str,
    query_length: int,
) -> None:
    if execution_phase == "decode":
        if query_length != MERAK_DECODE_QUERY_LENGTH:
            raise RuntimeError(
                "Merak normal decode must execute exactly one token row: "
                f"got M={query_length}"
            )
        return
    if execution_phase == "verify" and not (
        MERAK_MIN_VERIFY_QUERY_LENGTH <= query_length <= MERAK_MAX_NARROW_QUERY_LENGTH
    ):
        raise RuntimeError(
            "Merak speculative verify must stay within the packed-W4 "
            "narrow-token contract "
            f"M={MERAK_MIN_VERIFY_QUERY_LENGTH}.."
            f"{MERAK_MAX_NARROW_QUERY_LENGTH}, got M={query_length}"
        )


def _runtime_linear_cache_fingerprint(model: Any) -> dict[str, Any]:
    get_mixin = getattr(model, "_get_kvcache_mixin", None)
    mixin = get_mixin() if callable(get_mixin) else None
    if mixin is None:
        return {}
    return {
        "past_conv_caches": determinism_fingerprint(
            getattr(mixin, "past_conv_caches", None)
        ),
        "past_recurrent_states": determinism_fingerprint(
            getattr(mixin, "past_recurrent_states", None)
        ),
    }


def _page_cache_block_fingerprint(
    model: Any,
    block_ids: tuple[int, ...],
) -> dict[str, Any]:
    if not block_ids:
        return {}
    result: dict[str, Any] = {}
    for layer in getattr(model, "placeholder_attn_layers", ()):
        layer_name = str(getattr(layer, "layer_name", ""))
        cache = getattr(layer, "kv_cache", None)
        if not isinstance(cache, HMFPPagedKVCache):
            continue
        valid_ids = [
            block_id for block_id in block_ids if 0 <= block_id < cache.num_blocks
        ]
        indices = torch.as_tensor(
            valid_ids,
            dtype=torch.long,
            device=cache.device,
        )
        layer_values = {
            name: tensor.index_select(0, indices)
            for name, tensor in cache.named_tensors()
        }
        result[layer_name] = {
            "block_ids": valid_ids,
            "tensors": determinism_fingerprint(layer_values),
        }
    return result


def _spec_is_mamba(kv_cache_spec: object) -> bool:
    return type(kv_cache_spec).__name__ == "MambaSpec"


def _spec_is_hmfp_attention(kv_cache_spec: object) -> bool:
    return isinstance(kv_cache_spec, (HMFPFullAttentionSpec, HMFPSlidingWindowSpec))


class MerakModelRunnerV1(GPUModelRunner):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: str = "auto",
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super().__init__(vllm_config, device)
        self.vllm_config = vllm_config
        self.device = device
        self.cache_dtype = kv_cache_dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.use_async_scheduling = False
        self._merak_runtime_linear_owner_req_id: str | None = None
        # req_id -> logical mamba state-block index (mirrors official vLLM's
        # self.mamba_state_idx). Used to restore the correct linear-attention
        # state for each request before running the HMONNX runtime.
        self.merak_linear_state_idx: dict[str, int] = {}
        self.merak_linear_checkpoint_store: MerakLinearCheckpointStore | None = None
        self._merak_verify_req_ids: set[str] = set()
        self._merak_pending_verify_req_ids: set[str] = set()
        self._merak_pending_linear_verifies: dict[str, dict[str, Any]] = {}
        self._merak_pending_page_verifies: dict[str, int] = {}
        self._merak_spec_proposer: Any | None = None

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return super().get_supported_tasks()

    def load_model(self, load_dummy_weights: bool = False) -> None:
        del load_dummy_weights
        self.model = get_model(
            vllm_config=self.vllm_config, kv_cache_dtype=self.kv_cache_dtype
        )
        contract = getattr(self.model, "gemma4_cache_contract", None)
        placeholders = getattr(self.model, "placeholder_attn_layers", ())
        if contract is not None and not placeholders:
            from vllm.config import set_current_vllm_config

            # vLLM attention modules read the active config during __init__.
            # Keep late Gemma placeholder construction inside the same scoped
            # lifecycle context used by model construction; never install a
            # process-global fallback config.
            with set_current_vllm_config(self.vllm_config):
                self.model._register_placeholder_attention_layers(  # type: ignore[attr-defined]
                    self.model.config.get_text_config()  # type: ignore[attr-defined]
                )
        set_runner_device = getattr(self.model, "set_runner_device", None)
        if callable(set_runner_device):
            set_runner_device(self.device)
        drafter = getattr(self, "drafter", None)
        attach_runner = getattr(drafter, "attach_merak_runner", None)
        if callable(attach_runner):
            attach_runner(self)
            self._merak_spec_proposer = drafter

    def get_model(self) -> nn.Module:
        return self.model

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return super().get_kv_cache_spec()

    @staticmethod
    def _normalize_layer_device(device: object, fallback: torch.device) -> torch.device:
        if device is None:
            return fallback
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        if isinstance(device, (str, torch.device)):
            return torch.device(device)
        return fallback

    @staticmethod
    def _hmonnx_session(runtime_model: Any) -> Any:
        return getattr(runtime_model, "hmonnx_session", None) or runtime_model

    @staticmethod
    def _page_attention_node_devices(
        runtime_model: Any, fallback: torch.device
    ) -> list[torch.device]:
        return [
            device
            for _layer_identity, device in MerakModelRunnerV1._page_attention_node_map(
                runtime_model, fallback
            )
        ]

    @staticmethod
    def _page_or_flash_attention_node_map(
        runtime_model: Any,
        fallback: torch.device,
    ) -> list[tuple[str, torch.device]]:
        """Resolve cache-layer devices before HMONNX page binding.

        The graph is initially exposed with ``FlashAttention`` modules and is
        rewritten to ``PageAttention`` when the runtime receives the vLLM KV
        cache.  KV-cache initialization happens between those two phases, so
        use the original attention nodes only as an early placement map.
        """

        page_map = MerakModelRunnerV1._page_attention_node_map(
            runtime_model, fallback
        )
        if page_map:
            resolved = page_map
        else:
            resolved = MerakModelRunnerV1._hmonnx_attention_node_map(
                runtime_model,
                fallback,
                module_class_names=frozenset(
                    {"FlashAttention", "FlashAttentionReadOnly"}
                ),
            )
        if (
            resolved
            and fallback.type == "cuda"
            and all(device.type == "cpu" for _identity, device in resolved)
        ):
            # Legacy HMONNX exposes parser/interpreter metadata on CPU even
            # after ``HMONNXModel.to(cuda)``.  The runtime's inference device
            # is authoritative in this case; allocating the physical page
            # backing on CPU would later pass CPU tensors to Triton PageAttention.
            return [(identity, fallback) for identity, _device in resolved]
        return resolved

    @staticmethod
    def _page_attention_node_map(
        runtime_model: Any,
        fallback: torch.device,
    ) -> list[tuple[str, torch.device]]:
        return MerakModelRunnerV1._hmonnx_attention_node_map(
            runtime_model,
            fallback,
            module_class_names=frozenset(
                {
                    "PageAttention",
                    "PageAttentionReadOnly",
                    "PageKVCache",
                    "PageKVCacheReadOnly",
                }
            ),
        )

    @staticmethod
    def _hmonnx_attention_module_identity(node: Any, module: Any) -> str:
        """Return a stable prefill/decode identity for an attention module."""

        raw_layer_idx = getattr(module, "layer_idx", None)
        try:
            layer_idx = int(raw_layer_idx)
        except (TypeError, ValueError):
            layer_idx = -1
        if layer_idx < 0:
            # Legacy graphs did not publish semantic ownership. Preserve their
            # existing name-based validation instead of inventing an identity.
            return str(node.target)

        raw_cache_index = getattr(module, "cache_index", -1)
        try:
            cache_index = int(raw_cache_index)
        except (TypeError, ValueError):
            cache_index = -1
        attention_kind = str(getattr(module, "attention_kind", "") or "")
        return (
            f"{module.__class__.__name__}:layer={layer_idx}:"
            f"cache={cache_index}:kind={attention_kind}"
        )

    @staticmethod
    def _hmonnx_attention_node_map(
        runtime_model: Any,
        fallback: torch.device,
        *,
        module_class_names: frozenset[str],
    ) -> list[tuple[str, torch.device]]:
        """Resolve graph-order attention owners before or after page binding.

        Freshly loaded HMONNX graphs still expose ``FlashAttention`` modules;
        PageAttention replacement happens later when vLLM binds its cache.
        Callers that only operate on bound page caches keep using
        :meth:`_page_attention_node_map`, while early device placement (Gemma4
        MTP draft loading) can explicitly accept both physical forms.
        """

        session = MerakModelRunnerV1._hmonnx_session(runtime_model)
        # Legacy HMONNX creates the actual graph session lazily.  Resolve that
        # wrapper here as well as in the context-binding path; otherwise the
        # worker observes neither PageAttention nor the pre-bind
        # FlashAttention modules and incorrectly reports zero cache layers.
        if getattr(session, "graph_module", None) is None:
            initialize = getattr(session, "initialize", None)
            if callable(initialize):
                initialize()
            session = getattr(session, "_session", session)
        graph_module = getattr(session, "graph_module", None)
        if graph_module is None:
            node_modules = getattr(session, "node_modules", None)
            if node_modules is None:
                return []
            result: list[tuple[str, torch.device]] = []
            attention_index = 0
            for module in node_modules:
                if module.__class__.__name__ not in module_class_names:
                    continue
                # Legacy node_modules do not carry stable graph node names;
                # attention order is the reliable identity shared by the
                # prefill and decode graphs.
                result.append(
                    (
                        f"{module.__class__.__name__}:index={attention_index}",
                        MerakModelRunnerV1._normalize_layer_device(
                            getattr(module, "device", None), fallback
                        ),
                    )
                )
                attention_index += 1
            return result

        interpreter = getattr(session, "interpreter", None)
        node_device_map = getattr(interpreter, "node_device_map", None)
        if node_device_map is None:
            node_device_map = getattr(graph_module, "node_device_map", {})
        interpreter_device = getattr(interpreter, "device", None)
        if isinstance(interpreter_device, (list, tuple)):
            interpreter_device = interpreter_device[0] if interpreter_device else None
        page_attention_map: list[tuple[str, torch.device]] = []
        attention_index = 0
        for node in graph_module.graph.nodes:
            if node.op != "call_module":
                continue
            try:
                module = graph_module.get_submodule(str(node.target))
            except AttributeError:
                continue
            if module.__class__.__name__ not in module_class_names:
                continue

            node_info = (
                node_device_map.get(node.name)
                if isinstance(node_device_map, dict)
                else None
            )
            node_device = getattr(node_info, "device", None)
            if node_device is None:
                node_device = interpreter_device
            if node_device is None:
                module_device = getattr(module, "device", None)
                node_device = (
                    module_device() if callable(module_device) else module_device
                )
            identity = MerakModelRunnerV1._hmonnx_attention_module_identity(
                node, module
            )
            if identity == str(node.target) and str(node.target).startswith(
                "node_modules."
            ):
                identity = f"{module.__class__.__name__}:index={attention_index}"
            page_attention_map.append(
                (
                    identity,
                    MerakModelRunnerV1._normalize_layer_device(node_device, fallback),
                )
            )
            attention_index += 1
        return page_attention_map

    @staticmethod
    def _iter_hmonnx_models(runtime_model: Any) -> list[Any]:
        models: list[Any] = []
        if runtime_model is None:
            return models
        for attr_name in ("prefill_model", "decode_model"):
            model = getattr(runtime_model, attr_name, None)
            if model is not None:
                models.append(model)
        if any(
            getattr(runtime_model, attr_name, None) is not None
            for attr_name in ("hmonnx_session", "graph_module", "interpreter")
        ):
            models.append(runtime_model)
        return models

    def _get_kv_cache_layer_devices(self) -> dict[str, torch.device]:
        from ..model_executor.model_loader.gemma4_cache_contract import (
            Gemma4CacheContract,
        )

        merak_model = cast(Any, getattr(self, "model", None))
        placeholder_attn_layers = getattr(merak_model, "placeholder_attn_layers", [])
        fallback = torch.device(getattr(merak_model, "inference_device", self.device))

        runtime_model = getattr(merak_model, "inference_model", None)
        prefill_model = getattr(runtime_model, "prefill_model", None)
        decode_model = getattr(runtime_model, "decode_model", None)
        prefill_map = (
            self._page_or_flash_attention_node_map(prefill_model, fallback)
            if prefill_model is not None
            else []
        )
        decode_map = (
            self._page_or_flash_attention_node_map(decode_model, fallback)
            if decode_model is not None
            else []
        )

        if prefill_model is not None and decode_model is not None:
            if len(prefill_map) != len(decode_map):
                raise RuntimeError(
                    "Merak prefill/decode PageAttention count mismatch: "
                    f"prefill has {len(prefill_map)} nodes, decode has {len(decode_map)} nodes."
                )
            for layer_idx, (prefill_entry, decode_entry) in enumerate(
                zip(prefill_map, decode_map)
            ):
                prefill_identity, prefill_device = prefill_entry
                decode_identity, decode_device = decode_entry
                if prefill_identity != decode_identity:
                    raise RuntimeError(
                        "Merak prefill/decode PageAttention layer identity/order mismatch at "
                        f"index {layer_idx}: prefill={prefill_identity!r}, decode={decode_identity!r}."
                    )
                if prefill_device != decode_device:
                    raise RuntimeError(
                        "Merak prefill/decode PageAttention device mismatch at "
                        f"index {layer_idx} ({prefill_identity}): "
                        f"prefill={prefill_device}, decode={decode_device}."
                    )
            page_attention_map = prefill_map
        else:
            page_attention_map = prefill_map or decode_map
            if not page_attention_map:
                page_attention_map = self._page_or_flash_attention_node_map(
                    runtime_model, fallback
                )

        page_attention_required = bool(
            getattr(merak_model, "enable_page_attention", False)
        )
        gemma4_contract = getattr(merak_model, "gemma4_cache_contract", None)
        expected_page_bindings = (
            tuple(gemma4_contract.context_bindings)
            if isinstance(gemma4_contract, Gemma4CacheContract)
            else ()
        )
        expected_page_count = (
            len(expected_page_bindings)
            if expected_page_bindings
            else len(placeholder_attn_layers)
        )
        if (page_attention_map or page_attention_required) and len(
            page_attention_map
        ) != expected_page_count:
            raise RuntimeError(
                "Merak PageAttention count "
                f"({len(page_attention_map)}) does not match placeholder count "
                f"({expected_page_count})."
            )

        page_attention_devices = [device for _identity, device in page_attention_map]

        layer_devices: dict[str, torch.device] = {}
        if expected_page_bindings:
            for binding, device in zip(
                expected_page_bindings, page_attention_devices, strict=True
            ):
                layer_devices[str(binding.layer_name)] = device
            layer_name_by_idx = {
                int(binding.layer_idx): str(binding.layer_name)
                for binding in gemma4_contract.bindings
            }
            for binding in gemma4_contract.bindings:
                layer_name = str(binding.layer_name)
                if layer_name in layer_devices:
                    continue
                owner_idx = binding.cache_owner_layer_idx
                owner_name = (
                    layer_name_by_idx.get(int(owner_idx))
                    if owner_idx is not None
                    else None
                )
                if owner_name is None or owner_name not in layer_devices:
                    raise RuntimeError(
                        f"Gemma4 cache layer {layer_name} has no physical page-module device"
                    )
                layer_devices[layer_name] = layer_devices[owner_name]
            return layer_devices

        for layer_idx, attn_layer in enumerate(placeholder_attn_layers):
            layer_name = getattr(attn_layer, "layer_name", None)
            if layer_name is None:
                continue
            if layer_idx < len(page_attention_devices):
                layer_devices[str(layer_name)] = page_attention_devices[layer_idx]
            else:
                layer_devices[str(layer_name)] = fallback
        return layer_devices

    def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        layer_devices = self._get_kv_cache_layer_devices()
        if self._uses_contiguous_single_request_cache():
            return self._allocate_contiguous_single_request_cache_tensors(
                kv_cache_config,
                layer_devices,
            )
        if not is_hmfp_cache_dtype(getattr(self, "kv_cache_dtype", None)):
            if not layer_devices:
                return super()._allocate_kv_cache_tensors(kv_cache_config)
            return self._allocate_device_mapped_kv_cache_tensors(
                kv_cache_config, layer_devices
            )

        num_blocks = int(kv_cache_config.num_blocks)
        if num_blocks <= 0:
            raise ValueError(
                f"Merak exact HMFP allocation requires positive num_blocks, got {num_blocks}."
            )

        layer_specs: dict[str, object] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                if layer_name in layer_specs:
                    raise RuntimeError(
                        f"Merak KV layer {layer_name} appears in more than one cache-spec group."
                    )
                layer_specs[layer_name] = group.kv_cache_spec

        expected_layer_names = {
            layer_name
            for group in kv_cache_config.kv_cache_groups
            for layer_name in group.layer_names
            if layer_name not in self.runner_only_attn_layers
        }
        if set(layer_specs) != expected_layer_names:
            missing = sorted(expected_layer_names - set(layer_specs))
            extras = sorted(set(layer_specs) - expected_layer_names)
            raise RuntimeError(
                f"Merak exact HMFP allocation layer/spec mismatch: missing specs={missing}, unexpected specs={extras}."
            )

        linear_slots = int(
            getattr(getattr(self, "scheduler_config", None), "max_num_seqs", 1)
        )
        if linear_slots <= 0:
            raise ValueError(
                f"Merak compact Mamba allocation requires positive max_num_seqs, got {linear_slots}."
            )

        from ..model_executor.model_loader.gemma4_cache_contract import (
            Gemma4CacheContract,
        )

        gemma4_contract = getattr(
            getattr(self, "model", None), "gemma4_cache_contract", None
        )
        uses_authoritative_gemma4_ownership = (
            isinstance(gemma4_contract, Gemma4CacheContract)
            and gemma4_contract.version == 2
        )
        gemma4_bindings_by_layer: dict[str, Any] = {}
        if uses_authoritative_gemma4_ownership:
            gemma4_bindings_by_layer = {
                str(binding.layer_name): binding
                for binding in getattr(
                    getattr(self, "model", None), "page_attention_bindings", ()
                )
            }
            binding_layer_names = set(gemma4_bindings_by_layer)
            if not expected_layer_names.issubset(binding_layer_names):
                missing = sorted(expected_layer_names - set(gemma4_bindings_by_layer))
                raise RuntimeError(
                    "Gemma4 finalized cache bindings do not match allocation layers: "
                    f"missing={missing}, unexpected=[]"
                )
            layer_name_by_index = {
                int(binding.layer_idx): str(binding.layer_name)
                for binding in gemma4_bindings_by_layer.values()
                if getattr(binding, "layer_idx", None) is not None
            }
            invalid_aliases = []
            for layer_name in sorted(binding_layer_names - expected_layer_names):
                binding = gemma4_bindings_by_layer[layer_name]
                owner_layer_idx = getattr(binding, "cache_owner_layer_idx", None)
                owner_name = (
                    layer_name_by_index.get(int(owner_layer_idx))
                    if owner_layer_idx is not None
                    else None
                )
                if owner_name not in expected_layer_names:
                    invalid_aliases.append((layer_name, owner_name))
            if invalid_aliases:
                raise RuntimeError(
                    "Gemma4 logical cache aliases do not target finalized "
                    f"allocation owners: {invalid_aliases}"
                )

        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        if not uses_authoritative_gemma4_ownership:
            # Preserve the legacy Qwen/HMFP allocation contract: vLLM may pair
            # unrelated hybrid layers in KVCacheTensor.shared_by, but HMONNX
            # owns one physical raw cache per exported layer.
            for layer_name, kv_cache_spec in layer_specs.items():
                if _spec_is_mamba(kv_cache_spec):
                    allocation_size = linear_slots * self._mamba_native_page_size_bytes(
                        kv_cache_spec
                    )
                    device = self.device
                else:
                    sefp_cache_dtype = getattr(
                        kv_cache_spec,
                        "sefp_cache_dtype",
                        getattr(kv_cache_spec, "dtype", self.kv_cache_dtype),
                    )
                    if not is_hmfp_cache_dtype(sefp_cache_dtype):
                        raise NotImplementedError(
                            "Mixed SEFP and non-SEFP KV cache groups are not supported by Merak runner."
                        )
                    if layer_name not in layer_devices:
                        raise RuntimeError(
                            f"Merak FullAttention layer {layer_name} has no validated PageAttention device."
                        )
                    allocation_size = num_blocks * int(
                        getattr(kv_cache_spec, "real_page_size_bytes")
                    )
                    device = layer_devices[layer_name]
                kv_cache_raw_tensors[layer_name] = torch.zeros(
                    allocation_size,
                    dtype=torch.int8,
                    device=device,
                )
            return kv_cache_raw_tensors

        assigned_layer_names: set[str] = set()
        for tensor_idx, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            owner_names = [
                str(layer_name)
                for layer_name in kv_cache_tensor.shared_by
                if layer_name not in self.runner_only_attn_layers
            ]
            if not owner_names:
                continue
            if len(set(owner_names)) != len(owner_names):
                raise RuntimeError(
                    f"Merak KVCacheTensor[{tensor_idx}].shared_by contains duplicates"
                )
            duplicate_owners = sorted(set(owner_names) & assigned_layer_names)
            if duplicate_owners:
                raise RuntimeError(
                    f"Merak KV layers appear in multiple KVCacheTensor.shared_by entries: {duplicate_owners}"
                )
            missing_specs = sorted(set(owner_names) - set(layer_specs))
            if missing_specs:
                raise RuntimeError(
                    f"Merak KVCacheTensor[{tensor_idx}].shared_by has no exact specs: {missing_specs}"
                )

            owner_specs = [layer_specs[layer_name] for layer_name in owner_names]
            signatures = {
                self._shared_cache_spec_signature(spec) for spec in owner_specs
            }
            if len(signatures) != 1:
                raise RuntimeError(
                    f"Merak KVCacheTensor[{tensor_idx}].shared_by has incompatible specs"
                )
            kv_cache_spec = owner_specs[0]

            if _spec_is_mamba(kv_cache_spec):
                allocation_size = linear_slots * self._mamba_native_page_size_bytes(
                    kv_cache_spec
                )
                device = self.device
            else:
                sefp_cache_dtype = getattr(
                    kv_cache_spec,
                    "sefp_cache_dtype",
                    getattr(kv_cache_spec, "dtype", self.kv_cache_dtype),
                )
                if not is_hmfp_cache_dtype(sefp_cache_dtype):
                    missing_devices = sorted(set(owner_names) - set(layer_devices))
                    if missing_devices:
                        raise RuntimeError(
                            "Gemma4 FP sliding-attention layers have no validated "
                            f"cache device: {missing_devices}"
                        )
                    device = layer_devices[owner_names[0]]
                    if any(layer_devices[name] != device for name in owner_names[1:]):
                        raise RuntimeError(
                            "Gemma4 shared FP sliding cache spans multiple devices"
                        )
                    allocation_size = int(kv_cache_tensor.size)
                    raw_tensor = torch.zeros(
                        allocation_size,
                        dtype=torch.int8,
                        device=device,
                    )
                    for layer_name in owner_names:
                        kv_cache_raw_tensors[layer_name] = raw_tensor
                    assigned_layer_names.update(owner_names)
                    continue
                missing_devices = sorted(set(owner_names) - set(layer_devices))
                if missing_devices:
                    raise RuntimeError(
                        f"Merak attention layers have no validated PageAttention device: {missing_devices}"
                    )
                device = layer_devices[owner_names[0]]
                mismatched_devices = {
                    layer_name: layer_devices[layer_name]
                    for layer_name in owner_names[1:]
                    if layer_devices[layer_name] != device
                }
                if mismatched_devices:
                    raise RuntimeError(
                        "Merak KVCacheTensor"
                        f"[{tensor_idx}].shared_by has ambiguous physical owner devices: "
                        f"owner={owner_names[0]}:{device}, mismatches={mismatched_devices}"
                    )
                real_page_size_bytes = int(
                    getattr(kv_cache_spec, "real_page_size_bytes")
                )
                page_ratios = {
                    gemma4_bindings_by_layer[
                        layer_name
                    ].physical_pages_per_scheduler_block
                    for layer_name in owner_names
                }
                if len(page_ratios) != 1:
                    raise RuntimeError(
                        "Gemma4 shared physical cache owners have inconsistent "
                        f"scheduler/physical page ratios: tensor={tensor_idx}, "
                        f"ratios={sorted(page_ratios)}"
                    )
                page_ratio = next(iter(page_ratios))
                spec_block_size = int(getattr(kv_cache_spec, "block_size"))
                physical_block_sizes = {
                    int(gemma4_bindings_by_layer[layer_name].physical_block_size)
                    for layer_name in owner_names
                }
                scheduler_block_sizes = {
                    int(gemma4_bindings_by_layer[layer_name].scheduler_block_size)
                    for layer_name in owner_names
                }
                if len(physical_block_sizes) != 1 or len(scheduler_block_sizes) != 1:
                    raise RuntimeError(
                        f"Gemma4 shared physical cache owners have inconsistent block sizes: tensor={tensor_idx}"
                    )
                physical_block_size = next(iter(physical_block_sizes))
                scheduler_block_size = next(iter(scheduler_block_sizes))
                if spec_block_size == physical_block_size:
                    allocation_page_multiplier = page_ratio
                    physical_page_size_bytes = real_page_size_bytes
                elif spec_block_size == scheduler_block_size:
                    allocation_page_multiplier = 1
                    if real_page_size_bytes % page_ratio != 0:
                        raise RuntimeError(
                            "Gemma4 finalized page bytes are not divisible by the "
                            f"physical page ratio: tensor={tensor_idx}, "
                            f"page_bytes={real_page_size_bytes}, ratio={page_ratio}"
                        )
                    physical_page_size_bytes = real_page_size_bytes // page_ratio
                else:
                    raise RuntimeError(
                        "Gemma4 allocator spec is neither physical nor finalized: "
                        f"tensor={tensor_idx}, spec_block={spec_block_size}, "
                        f"physical={physical_block_size}, "
                        f"scheduler={scheduler_block_size}"
                    )
                fixed_page_counts = {
                    gemma4_bindings_by_layer[layer_name].fixed_physical_pages
                    for layer_name in owner_names
                }
                if len(fixed_page_counts) != 1:
                    raise RuntimeError(
                        "Gemma4 shared physical cache owners disagree on fixed page "
                        f"count: tensor={tensor_idx}, counts={fixed_page_counts}"
                    )
                fixed_page_count = next(iter(fixed_page_counts))
                if fixed_page_count is None:
                    allocation_size = (
                        num_blocks * real_page_size_bytes * allocation_page_multiplier
                    )
                else:
                    allocation_size = fixed_page_count * physical_page_size_bytes
                declared_size = int(kv_cache_tensor.size)
                if declared_size not in {
                    num_blocks * real_page_size_bytes,
                    allocation_size,
                }:
                    raise RuntimeError(
                        "Gemma4 authoritative KVCacheTensor size is incompatible "
                        "with its physical page projection: "
                        f"tensor={tensor_idx}, declared={declared_size}, "
                        f"physical_page={real_page_size_bytes}, ratio={page_ratio}, "
                        f"scheduler_blocks={num_blocks}"
                    )
            raw_tensor = torch.zeros(
                allocation_size,
                dtype=torch.int8,
                device=device,
            )
            for layer_name in owner_names:
                kv_cache_raw_tensors[layer_name] = raw_tensor
            assigned_layer_names.update(owner_names)

        if assigned_layer_names != expected_layer_names:
            missing = sorted(expected_layer_names - assigned_layer_names)
            extras = sorted(assigned_layer_names - expected_layer_names)
            raise RuntimeError(
                f"Merak KVCacheTensor.shared_by ownership mismatch: missing={missing}, unexpected={extras}"
            )

        return kv_cache_raw_tensors

    def _uses_contiguous_single_request_cache(self) -> bool:
        return (
            getattr(getattr(self, "model", None), "hybrid_cache_policy", None)
            == HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST
        )

    def _allocate_contiguous_single_request_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        layer_devices: dict[str, torch.device],
    ) -> dict[str, torch.Tensor]:
        """Allocate contiguous-policy caches independently of virtual packing.

        vLLM unifies hybrid cache page sizes for scheduler accounting and may
        pack FullAttention and Mamba layers into one block-strided backing.
        A contiguous-policy HMONNX runtime instead owns request-local K/V
        caches and one linear state slot per admitted request. Keep those
        physical tensors exact while leaving unified page geometry to the
        scheduler.
        """

        num_blocks = int(kv_cache_config.num_blocks)
        max_num_seqs = int(
            getattr(getattr(self, "scheduler_config", None), "max_num_seqs", 1)
        )
        if num_blocks <= 0 or max_num_seqs <= 0:
            raise ValueError(
                "Merak contiguous hybrid allocation requires positive block and "
                f"request counts, got blocks={num_blocks}, max_num_seqs={max_num_seqs}."
            )

        layer_specs: dict[str, object] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                if layer_name in layer_specs:
                    raise RuntimeError(
                        f"Merak KV layer {layer_name} appears in more than one cache-spec group."
                    )
                layer_specs[layer_name] = group.kv_cache_spec

        expected_layer_names = {
            layer_name
            for group in kv_cache_config.kv_cache_groups
            for layer_name in group.layer_names
            if layer_name not in self.runner_only_attn_layers
        }
        if set(layer_specs) != expected_layer_names:
            missing = sorted(expected_layer_names - set(layer_specs))
            extras = sorted(set(layer_specs) - expected_layer_names)
            raise RuntimeError(
                "Merak contiguous hybrid allocation layer/spec mismatch: "
                f"missing={missing}, unexpected={extras}."
            )

        raw_tensors: dict[str, torch.Tensor] = {}
        for layer_name, spec in layer_specs.items():
            if _spec_is_mamba(spec):
                allocation_size = max_num_seqs * self._mamba_native_page_size_bytes(
                    spec
                )
                device = self.device
            else:
                real_page_size = int(getattr(spec, "real_page_size_bytes"))
                if real_page_size <= 0:
                    raise RuntimeError(
                        f"Merak attention layer {layer_name} has invalid real page size {real_page_size}."
                    )
                allocation_size = num_blocks * real_page_size
                device = layer_devices.get(layer_name, self.device)
            raw_tensors[layer_name] = torch.zeros(
                allocation_size,
                dtype=torch.int8,
                device=device,
            )
        return raw_tensors

    @staticmethod
    def _shared_cache_spec_signature(kv_cache_spec: object) -> tuple[object, ...]:
        if _spec_is_mamba(kv_cache_spec):
            return (
                type(kv_cache_spec),
                tuple(
                    tuple(int(dim) for dim in shape) for shape in kv_cache_spec.shapes
                ),
                tuple(kv_cache_spec.dtypes),
            )
        head_size = int(getattr(kv_cache_spec, "head_size"))
        raw_head_size_v = getattr(kv_cache_spec, "head_size_v", None)
        head_size_v = head_size if raw_head_size_v is None else int(raw_head_size_v)
        return (
            type(kv_cache_spec),
            int(getattr(kv_cache_spec, "real_page_size_bytes")),
            int(getattr(kv_cache_spec, "block_size")),
            int(getattr(kv_cache_spec, "num_kv_heads")),
            head_size,
            head_size_v,
            str(
                getattr(
                    kv_cache_spec,
                    "sefp_cache_dtype",
                    getattr(kv_cache_spec, "dtype", None),
                )
            ),
            getattr(kv_cache_spec, "sliding_window", None),
        )

    @staticmethod
    def _gemma4_required_physical_blocks(binding: Any, scheduler_blocks: int) -> int:
        fixed_physical_pages = getattr(binding, "fixed_physical_pages", None)
        if fixed_physical_pages is not None:
            return int(fixed_physical_pages)
        return scheduler_blocks * int(binding.physical_pages_per_scheduler_block)

    def _apply_gemma4_cache_tensor_ownership(
        self,
        kv_cache_config: KVCacheConfig,
        bindings: tuple[Any, ...],
    ) -> None:
        """Make exported cache indices authoritative for physical tensor owners."""

        groups = kv_cache_config.kv_cache_groups
        layer_name_by_index = {
            int(binding.layer_idx): str(binding.layer_name)
            for binding in bindings
            if getattr(binding, "layer_idx", None) is not None
        }
        shared_layer_targets: dict[str, str] = {}
        for binding in bindings:
            owner_layer_idx = getattr(binding, "cache_owner_layer_idx", None)
            layer_idx = getattr(binding, "layer_idx", None)
            if owner_layer_idx is None or owner_layer_idx == layer_idx:
                continue
            owner_layer_name = layer_name_by_index.get(int(owner_layer_idx))
            if owner_layer_name is None:
                raise RuntimeError(
                    "Gemma4 cache owner binding is missing: "
                    f"layer={binding.layer_name}, owner_layer_idx={owner_layer_idx}"
                )
            shared_layer_targets[str(binding.layer_name)] = owner_layer_name

        bindings_by_cache_index: dict[int, list[Any]] = {}
        for binding in bindings:
            if binding.cache_group_idx is None:
                raise RuntimeError(
                    f"Gemma4 binding {binding.layer_name} has no finalized cache group"
                )
            bindings_by_cache_index.setdefault(binding.cache_index, []).append(binding)

        exact_tensors: list[KVCacheTensor] = []
        for cache_index, cache_bindings in sorted(bindings_by_cache_index.items()):
            specs = []
            for binding in cache_bindings:
                group = groups[binding.cache_group_idx]
                specs_by_layer = resolve_finalized_group_layer_specs(
                    group,
                    shared_layer_targets=shared_layer_targets,
                )
                layer_name = str(binding.layer_name)
                spec = specs_by_layer.get(layer_name)
                if spec is None:
                    # vLLM removes modules with kv_sharing_target_layer_name
                    # from finalized groups. The export contract still keeps
                    # those logical aliases so every HMONNX layer can bind to
                    # its physical cache index.
                    owner_name = shared_layer_targets.get(layer_name)
                    if owner_name is not None:
                        spec = specs_by_layer.get(owner_name)
                if spec is None:
                    raise RuntimeError(
                        "Gemma4 finalized cache group has no spec for binding: "
                        f"layer={layer_name}, group_idx={binding.cache_group_idx}"
                    )
                specs.append(spec)
            signatures = {self._shared_cache_spec_signature(spec) for spec in specs}
            if len(signatures) != 1:
                raise RuntimeError(
                    f"Gemma4 shared cache_index has incompatible finalized specs: cache_index={cache_index}"
                )
            spec = specs[0]
            if _spec_is_mamba(spec):
                raise RuntimeError(
                    "Gemma4 contract-v2 does not support Mamba cache specs"
                )
            fixed_page_counts = {
                getattr(binding, "fixed_physical_pages", None)
                for binding in cache_bindings
            }
            if len(fixed_page_counts) != 1:
                raise RuntimeError(
                    "Gemma4 shared cache_index has inconsistent fixed page counts: "
                    f"cache_index={cache_index}, counts={fixed_page_counts}"
                )
            fixed_page_count = next(iter(fixed_page_counts))
            tensor_size = int(kv_cache_config.num_blocks) * int(
                getattr(spec, "real_page_size_bytes")
            )
            if fixed_page_count is not None:
                binding = cache_bindings[0]
                page_ratio = binding.physical_pages_per_scheduler_block
                spec_page_bytes = int(getattr(spec, "real_page_size_bytes"))
                spec_block_size = int(getattr(spec, "block_size"))
                if spec_block_size == int(binding.physical_block_size):
                    physical_page_bytes = spec_page_bytes
                elif spec_block_size == int(binding.scheduler_block_size):
                    if spec_page_bytes % page_ratio != 0:
                        raise RuntimeError(
                            "Gemma4 finalized page bytes are not divisible by the "
                            f"physical page ratio: cache_index={cache_index}"
                        )
                    physical_page_bytes = spec_page_bytes // page_ratio
                else:
                    raise RuntimeError(
                        "Gemma4 finalized cache spec has an unknown block size: "
                        f"cache_index={cache_index}, spec={spec_block_size}"
                    )
                tensor_size = fixed_page_count * physical_page_bytes
            exact_tensors.append(
                KVCacheTensor(
                    size=tensor_size,
                    # Current vLLM removes kv_sharing_target_layer_name aliases
                    # from finalized groups and wires their module tensors to
                    # the owner after allocation. Keep only finalized physical
                    # members here; page_attention_bindings retains every
                    # logical alias for HMONNX cache-index routing.
                    shared_by=[
                        binding.layer_name
                        for binding in cache_bindings
                        if binding.layer_name
                        in groups[binding.cache_group_idx].layer_names
                    ],
                )
            )

        kv_cache_config.kv_cache_tensors = exact_tensors

    def _allocate_device_mapped_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        layer_devices: dict[str, torch.device],
    ) -> dict[str, torch.Tensor]:
        """Preserve the legacy virtual-backing allocator for non-HMFP caches."""
        if not layer_devices:
            return super()._allocate_kv_cache_tensors(kv_cache_config)

        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        packed_backing_by_device: dict[torch.device, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensors_by_device: dict[torch.device, torch.Tensor] = {}
            for layer_name in kv_cache_tensor.shared_by:
                device = layer_devices.get(layer_name, self.device)
                if getattr(kv_cache_tensor, "block_stride", 0) > 0:
                    tensor = packed_backing_by_device.get(device)
                    if tensor is None:
                        tensor = torch.zeros(
                            kv_cache_tensor.size, dtype=torch.int8, device=device
                        )
                        packed_backing_by_device[device] = tensor
                else:
                    tensor = tensors_by_device.get(device)
                    if tensor is None:
                        tensor = torch.zeros(
                            kv_cache_tensor.size, dtype=torch.int8, device=device
                        )
                        tensors_by_device[device] = tensor
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors

    @staticmethod
    def _mamba_native_page_size_bytes(kv_cache_spec: object) -> int:
        shapes = tuple(getattr(kv_cache_spec, "shapes"))
        dtypes = tuple(getattr(kv_cache_spec, "dtypes"))
        if len(shapes) != len(dtypes):
            raise RuntimeError(
                f"Merak Mamba state spec has mismatched shapes/dtypes: {len(shapes)} shapes vs {len(dtypes)} dtypes."
            )
        page_size_bytes = sum(
            math.prod(int(dim) for dim in shape)
            * torch.empty((), dtype=dtype).element_size()
            for shape, dtype in zip(shapes, dtypes, strict=True)
        )
        if page_size_bytes <= 0:
            raise RuntimeError(
                f"Merak Mamba native page size must be positive, got {page_size_bytes}."
            )
        return page_size_bytes

    @staticmethod
    def _exact_hmfp_page_view(
        raw_tensor: torch.Tensor,
        *,
        page_size_bytes: int,
        layer_name: str,
    ) -> torch.Tensor:
        """View one exact per-layer HMFP backing without virtual padding."""
        if page_size_bytes <= 0:
            raise ValueError(f"SEFP page size must be positive, got {page_size_bytes}.")
        raw_bytes = raw_tensor.view(torch.uint8)

        if raw_bytes.dim() == 2:
            if tuple(raw_bytes.shape[1:]) != (
                page_size_bytes,
            ) or raw_bytes.stride() != (page_size_bytes, 1):
                raise ValueError(
                    f"Exact HMFP cache for {layer_name} must have page width/stride {page_size_bytes}, "
                    f"got shape={tuple(raw_bytes.shape)}, stride={tuple(raw_bytes.stride())}."
                )
            return raw_bytes

        if raw_bytes.numel() % page_size_bytes != 0:
            raise ValueError(
                f"Exact HMFP cache for {layer_name} has {raw_bytes.numel()} bytes, "
                f"which is not divisible by real page size {page_size_bytes}."
            )
        return raw_bytes.view(-1, page_size_bytes)

    @staticmethod
    def _reshape_mamba_state_cache(
        raw_tensor: torch.Tensor,
        kv_cache_spec: object,
        *,
        layer_name: str,
    ) -> list[torch.Tensor]:
        raw_bytes = raw_tensor.view(torch.uint8)
        page_size_bytes = MerakModelRunnerV1._mamba_native_page_size_bytes(
            kv_cache_spec
        )
        block_stride = page_size_bytes
        if raw_bytes.numel() % block_stride != 0:
            raise ValueError(
                f"Mamba raw state cache for {layer_name} has {raw_bytes.numel()} bytes, "
                f"which is not divisible by block stride {block_stride}."
            )
        num_blocks = raw_bytes.numel() // block_stride

        state_tensors: list[torch.Tensor] = []
        storage_offset_bytes = 0
        for shape, dtype in zip(
            getattr(kv_cache_spec, "shapes"),
            getattr(kv_cache_spec, "dtypes"),
            strict=True,
        ):
            dtype_size = torch.empty((), dtype=dtype).element_size()
            if storage_offset_bytes % dtype_size != 0 or block_stride % dtype_size != 0:
                raise ValueError(
                    f"Mamba state cache for {layer_name} is not aligned for dtype {dtype}: "
                    f"offset={storage_offset_bytes}, block_stride={block_stride}."
                )
            target_shape = (num_blocks, *shape)
            contiguous_stride = torch.empty(target_shape, dtype=dtype).stride()
            target_stride = (block_stride // dtype_size, *contiguous_stride[1:])
            tensor = torch.as_strided(
                raw_bytes.view(dtype),
                size=target_shape,
                stride=target_stride,
                storage_offset=(raw_bytes.storage_offset() + storage_offset_bytes)
                // dtype_size,
            )
            state_tensors.append(tensor)
            storage_offset_bytes += contiguous_stride[0] * dtype_size

        if storage_offset_bytes > block_stride:
            raise ValueError(
                f"Mamba states for {layer_name} require {storage_offset_bytes} bytes, "
                f"but block stride is only {block_stride}."
            )
        return state_tensors

    @staticmethod
    def _reshape_padded_kv_first_attention_cache(
        raw_tensor: torch.Tensor,
        kv_cache_spec: object,
        attn_backend: type,
        *,
        kernel_block_size: int,
        packing: tuple[int, int] | None,
        layer_name: str,
    ) -> torch.Tensor:
        """Create a KV-first view over block-padded FP cache pages.

        Upstream vLLM only supports padded attention pages when the backend's
        physical layout starts with ``num_blocks``. Merak's FP backend keeps
        the established ``[2, N, H, B, D]`` ABI, while hybrid cache unification
        pads every logical block to the larger linear-state page. A strided
        view can represent that layout without copying: K/V remain adjacent
        inside each block and the block axis skips the inter-page padding.
        """

        block_size = int(getattr(kv_cache_spec, "block_size"))
        storage_block_size = int(
            getattr(kv_cache_spec, "storage_block_size", block_size)
        )
        if block_size != kernel_block_size or storage_block_size != block_size:
            raise RuntimeError(
                "Merak padded KV-first cache currently requires one kernel block "
                f"per cache page: layer={layer_name}, cache_block={block_size}, "
                f"storage_block={storage_block_size}, kernel_block={kernel_block_size}."
            )

        page_stride_bytes = int(getattr(kv_cache_spec, "page_size_bytes"))
        if packing is None:
            offset_bytes = 0
            block_stride_bytes = page_stride_bytes
        else:
            offset_bytes, block_stride_bytes = map(int, packing)
        if block_stride_bytes <= 0 or raw_tensor.numel() % block_stride_bytes != 0:
            raise RuntimeError(
                "Merak padded KV-first backing is not block-stride aligned: "
                f"layer={layer_name}, bytes={raw_tensor.numel()}, "
                f"block_stride={block_stride_bytes}."
            )
        num_blocks = raw_tensor.numel() // block_stride_bytes

        cache_dtype = getattr(kv_cache_spec, "dtype")
        num_kv_heads = int(getattr(kv_cache_spec, "num_kv_heads"))
        head_size = int(getattr(kv_cache_spec, "head_size"))
        cache_shape = tuple(
            int(dimension)
            for dimension in attn_backend.get_kv_cache_shape(
                num_blocks,
                kernel_block_size,
                num_kv_heads,
                head_size,
                cache_dtype_str=str(cache_dtype),
            )
        )
        expected_shape = (
            2,
            num_blocks,
            num_kv_heads,
            kernel_block_size,
            head_size,
        )
        if cache_shape != expected_shape:
            raise RuntimeError(
                "Merak padded FP cache requires the KV-first backend shape "
                f"{expected_shape}, got {cache_shape} for {layer_name}."
            )
        try:
            stride_order = tuple(attn_backend.get_kv_cache_stride_order())
        except (AttributeError, NotImplementedError):
            stride_order = tuple(range(len(cache_shape)))
        if stride_order != tuple(range(len(cache_shape))):
            raise RuntimeError(
                "Merak padded KV-first cache requires identity backend stride "
                f"order, got {stride_order} for {layer_name}."
            )

        dtype_size = torch.empty((), dtype=cache_dtype).element_size()
        if offset_bytes % dtype_size or block_stride_bytes % dtype_size:
            raise RuntimeError(
                "Merak padded KV-first backing is not dtype aligned: "
                f"layer={layer_name}, offset={offset_bytes}, "
                f"block_stride={block_stride_bytes}, dtype={cache_dtype}."
            )
        values_per_kv = num_kv_heads * kernel_block_size * head_size
        real_page_bytes = 2 * values_per_kv * dtype_size
        if real_page_bytes > page_stride_bytes:
            raise RuntimeError(
                "Merak padded KV-first page is smaller than its K/V payload: "
                f"layer={layer_name}, payload={real_page_bytes}, "
                f"page_stride={page_stride_bytes}."
            )
        required_bytes = (
            offset_bytes + max(0, num_blocks - 1) * block_stride_bytes + real_page_bytes
        )
        if required_bytes > raw_tensor.numel():
            raise RuntimeError(
                "Merak padded KV-first view exceeds its backing: "
                f"layer={layer_name}, required={required_bytes}, "
                f"available={raw_tensor.numel()}."
            )

        typed = raw_tensor.view(torch.uint8).view(cache_dtype)
        return torch.as_strided(
            typed,
            size=cache_shape,
            stride=(
                values_per_kv,
                block_stride_bytes // dtype_size,
                kernel_block_size * head_size,
                head_size,
                1,
            ),
            storage_offset=typed.storage_offset() + offset_bytes // dtype_size,
        )

    @staticmethod
    def _reshape_exact_kv_first_attention_cache(
        raw_tensor: torch.Tensor,
        kv_cache_spec: object,
        attn_backend: type,
        *,
        kernel_block_size: int,
        layer_name: str,
    ) -> torch.Tensor:
        """Restore the backend ABI over an exact, unpadded per-layer backing."""

        block_size = int(getattr(kv_cache_spec, "block_size"))
        storage_block_size = int(
            getattr(kv_cache_spec, "storage_block_size", block_size)
        )
        if block_size != kernel_block_size or storage_block_size != block_size:
            raise RuntimeError(
                "Merak exact KV-first cache requires one kernel block per page: "
                f"layer={layer_name}, cache_block={block_size}, "
                f"storage_block={storage_block_size}, kernel_block={kernel_block_size}."
            )

        real_page_bytes = int(getattr(kv_cache_spec, "real_page_size_bytes"))
        if real_page_bytes <= 0 or raw_tensor.numel() % real_page_bytes != 0:
            raise RuntimeError(
                "Merak exact KV-first backing is not real-page aligned: "
                f"layer={layer_name}, bytes={raw_tensor.numel()}, "
                f"real_page={real_page_bytes}."
            )
        num_blocks = raw_tensor.numel() // real_page_bytes
        cache_dtype = getattr(kv_cache_spec, "dtype")
        cache_shape = tuple(
            int(dimension)
            for dimension in attn_backend.get_kv_cache_shape(
                num_blocks,
                kernel_block_size,
                int(getattr(kv_cache_spec, "num_kv_heads")),
                int(getattr(kv_cache_spec, "head_size")),
                cache_dtype_str=str(cache_dtype),
            )
        )
        expected_shape = (
            2,
            num_blocks,
            int(getattr(kv_cache_spec, "num_kv_heads")),
            kernel_block_size,
            int(getattr(kv_cache_spec, "head_size")),
        )
        if cache_shape != expected_shape:
            raise RuntimeError(
                "Merak exact FP cache requires the KV-first backend shape "
                f"{expected_shape}, got {cache_shape} for {layer_name}."
            )
        try:
            stride_order = tuple(attn_backend.get_kv_cache_stride_order())
        except (AttributeError, NotImplementedError):
            stride_order = tuple(range(len(cache_shape)))
        if stride_order != tuple(range(len(cache_shape))):
            raise RuntimeError(
                "Merak exact KV-first cache requires identity backend stride "
                f"order, got {stride_order} for {layer_name}."
            )
        dtype_size = torch.empty((), dtype=cache_dtype).element_size()
        if raw_tensor.numel() != math.prod(cache_shape) * dtype_size:
            raise RuntimeError(
                "Merak exact KV-first shape does not cover its backing: "
                f"layer={layer_name}, shape={cache_shape}, bytes={raw_tensor.numel()}."
            )
        return raw_tensor.view(cache_dtype).view(cache_shape)

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        needs_padded_kv_first_reshape = any(
            not _spec_is_mamba(group.kv_cache_spec)
            and not _spec_is_hmfp_attention(group.kv_cache_spec)
            and getattr(group.kv_cache_spec, "page_size_padded", None) is not None
            for group in self._kv_cache_spec_attn_group_iterator()
        )
        if (
            not is_hmfp_cache_dtype(self.kv_cache_dtype)
            and not needs_padded_kv_first_reshape
            and not self._uses_contiguous_single_request_cache()
        ):
            return super()._reshape_kv_cache_tensors(
                kv_cache_raw_tensors, kernel_block_sizes
            )

        layer_packing: dict[str, tuple[int, int]] = {}
        kv_cache_tensors = getattr(
            getattr(self, "kv_cache_config", None), "kv_cache_tensors", ()
        )
        for kv_tensor in kv_cache_tensors:
            if getattr(kv_tensor, "block_stride", 0) > 0:
                for layer_name in kv_tensor.shared_by:
                    layer_packing[layer_name] = (
                        int(getattr(kv_tensor, "offset", 0)),
                        int(kv_tensor.block_stride),
                    )

        gemma4_bindings_by_layer = {
            str(binding.layer_name): binding
            for binding in getattr(
                getattr(self, "model", None), "page_attention_bindings", ()
            )
        }
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            if group.kv_cache_group_id == len(kernel_block_sizes):
                continue

            if _spec_is_mamba(kv_cache_spec):
                for layer_name in group.layer_names:
                    if layer_name in self.runner_only_attn_layers:
                        continue
                    kv_caches[layer_name] = self._reshape_mamba_state_cache(
                        kv_cache_raw_tensors[layer_name],
                        kv_cache_spec,
                        layer_name=layer_name,
                    )
                continue

            sefp_cache_dtype = getattr(
                kv_cache_spec,
                "sefp_cache_dtype",
                getattr(kv_cache_spec, "dtype", self.kv_cache_dtype),
            )
            if not is_hmfp_cache_dtype(sefp_cache_dtype):
                cache_dtype = getattr(kv_cache_spec, "dtype", torch.float16)
                for layer_name in group.layer_names:
                    if layer_name in self.runner_only_attn_layers:
                        continue
                    binding = gemma4_bindings_by_layer.get(str(layer_name))
                    if binding is None:
                        if self._uses_contiguous_single_request_cache():
                            kv_caches[layer_name] = (
                                self._reshape_exact_kv_first_attention_cache(
                                    kv_cache_raw_tensors[layer_name],
                                    kv_cache_spec,
                                    group.backend,
                                    kernel_block_size=kernel_block_sizes[
                                        group.kv_cache_group_id
                                    ],
                                    layer_name=layer_name,
                                )
                            )
                            continue
                        if getattr(kv_cache_spec, "page_size_padded", None) is None:
                            raise RuntimeError(
                                "Merak custom FP cache reshape requires either a "
                                f"Gemma4 binding or padded pages for {layer_name}."
                            )
                        kv_caches[layer_name] = (
                            self._reshape_padded_kv_first_attention_cache(
                                kv_cache_raw_tensors[layer_name],
                                kv_cache_spec,
                                group.backend,
                                kernel_block_size=kernel_block_sizes[
                                    group.kv_cache_group_id
                                ],
                                packing=layer_packing.get(layer_name),
                                layer_name=layer_name,
                            )
                        )
                        continue
                    raw_cache = kv_cache_raw_tensors[layer_name]
                    block_size = int(binding.physical_block_size)
                    num_kv_heads = int(binding.kv_shape[1])
                    head_size = int(binding.kv_shape[-1])
                    values_per_page = 2 * num_kv_heads * block_size * head_size
                    dtype_size = torch.empty((), dtype=cache_dtype).element_size()
                    page_bytes = values_per_page * dtype_size
                    if raw_cache.numel() % page_bytes != 0:
                        raise RuntimeError(
                            "Gemma4 FP sliding cache backing is not page aligned: "
                            f"layer={layer_name}, bytes={raw_cache.numel()}, page_bytes={page_bytes}"
                        )
                    num_pages = raw_cache.numel() // page_bytes
                    kv_caches[layer_name] = raw_cache.view(cache_dtype).view(
                        2,
                        num_pages,
                        num_kv_heads,
                        block_size,
                        head_size,
                    )
                continue

            logical_page_size = int(
                getattr(
                    kv_cache_spec, "real_page_size_bytes", kv_cache_spec.page_size_bytes
                )
            )
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_page_size = logical_page_size
                binding = gemma4_bindings_by_layer.get(str(layer_name))
                if binding is not None:
                    kv_shape = tuple(int(dim) for dim in binding.kv_shape)
                    if len(kv_shape) != 4:
                        raise RuntimeError(
                            "Gemma4 binding KV shape must be [B, H, S, D] when "
                            f"reshaping physical cache: layer={layer_name}, "
                            f"shape={kv_shape}"
                        )
                    layer_page_size = get_hmfp_cache_block_size_bytes(
                        block_size=int(binding.physical_block_size),
                        num_kv_heads=kv_shape[1],
                        head_size=kv_shape[-1],
                        cache_dtype=sefp_cache_dtype,
                    )
                kv_caches[layer_name] = self._exact_hmfp_page_view(
                    kv_cache_raw_tensors[layer_name],
                    page_size_bytes=layer_page_size,
                    layer_name=layer_name,
                )

        return cast(dict[str, torch.Tensor], kv_caches)

    def _get_attn_metadata(self) -> MerakAttentionMetadata | None:
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, list):
            if not attn_metadata:
                return None
            attn_metadata = attn_metadata[0]

        merak_model = cast(Any, self.model)
        if isinstance(attn_metadata, dict):
            layer_name = merak_model.placeholder_attn_layers[0].layer_name
            layer_metadata = attn_metadata.get(layer_name)
            if isinstance(layer_metadata, MerakAttentionMetadata):
                return layer_metadata
            return None

        if isinstance(attn_metadata, MerakAttentionMetadata):
            return attn_metadata
        return None

    def _get_attn_metadata_by_group(self) -> dict[int, MerakAttentionMetadata]:
        merak_model = cast(Any, self.model)
        resolver = getattr(merak_model, "_get_attn_metadata_by_group", None)
        if not callable(resolver):
            raise RuntimeError(
                "Gemma4 contract-v2 model has no group metadata resolver"
            )
        return resolver()

    @staticmethod
    def _common_gemma4_request_metadata(
        metadata_by_group: dict[int, MerakAttentionMetadata],
    ) -> MerakAttentionMetadata:
        if not metadata_by_group:
            raise RuntimeError("Gemma4 contract-v2 has no attention metadata groups")
        canonical_group = min(metadata_by_group)
        canonical = metadata_by_group[canonical_group]
        for group_idx in sorted(metadata_by_group):
            metadata = metadata_by_group[group_idx]
            for field_name in ("past_lens", "subsequence_begins"):
                if not torch.equal(
                    getattr(canonical, field_name), getattr(metadata, field_name)
                ):
                    raise RuntimeError(
                        "Gemma4 request layout differs across cache groups: "
                        f"canonical_group={canonical_group}, group={group_idx}, "
                        f"field={field_name}"
                    )
        return canonical

    @staticmethod
    def _slice_gemma4_group_metadata(
        metadata: MerakAttentionMetadata,
        req_idx: int,
    ) -> tuple[MerakAttentionMetadata, int]:
        query_bounds = metadata.subsequence_begins.to(device="cpu").tolist()
        block_bounds = metadata.block_indices_begins.to(device="cpu").tolist()
        query_start = int(query_bounds[req_idx])
        query_end = int(query_bounds[req_idx + 1])
        block_start = int(block_bounds[req_idx])
        block_end = int(block_bounds[req_idx + 1])
        query_length = query_end - query_start
        sliced = MerakAttentionMetadata(
            past_lens=metadata.past_lens[req_idx : req_idx + 1],
            subsequence_begins=torch.tensor(
                [0, query_length],
                dtype=metadata.subsequence_begins.dtype,
                device=metadata.subsequence_begins.device,
            ),
            block_indices=metadata.block_indices[block_start:block_end],
            block_indices_begins=torch.tensor(
                [0, block_end - block_start],
                dtype=metadata.block_indices_begins.dtype,
                device=metadata.block_indices_begins.device,
            ),
            max_context_len=metadata.max_context_len,
            slot_mapping=metadata.slot_mapping[query_start:query_end],
            sampled_token_indices=metadata.sampled_token_indices,
            seq_lens=(
                metadata.seq_lens[req_idx : req_idx + 1]
                if metadata.seq_lens is not None
                else None
            ),
        )
        return sliced, query_length

    def _resolve_mamba_group_id(self) -> int | None:
        cached = getattr(self, "_merak_mamba_group_id", "unset")
        if cached != "unset":
            return cached

        kv_cache_config = getattr(self, "kv_cache_config", None)
        merak_model = cast(Any, self.model)
        placeholders = list(
            getattr(merak_model, "placeholder_linear_state_layers", None) or ()
        )
        if not placeholders:
            self._merak_mamba_group_id = None
            self._merak_layer_group_ids = []
            return None
        if kv_cache_config is None:
            raise RuntimeError(
                "Merak linear-state placeholders require Mamba KV-cache groups, but no KV cache config exists."
            )

        placeholder_names = [str(ph.layer_name) for ph in placeholders]
        duplicate_placeholders = sorted(
            {name for name in placeholder_names if placeholder_names.count(name) > 1}
        )
        if duplicate_placeholders:
            raise RuntimeError(
                "Merak duplicate linear-state placeholder layer(s): "
                + ", ".join(duplicate_placeholders)
            )

        name_to_group: dict[str, int] = {}
        mamba_group_signatures: dict[int, tuple[int, int]] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            if not _spec_is_mamba(spec):
                continue
            mamba_group_signatures[gid] = (
                int(spec.block_size),
                int(getattr(spec, "num_speculative_blocks", 0)),
            )
            for raw_layer_name in group.layer_names:
                layer_name = str(raw_layer_name)
                previous_gid = name_to_group.get(layer_name)
                if previous_gid is not None:
                    raise RuntimeError(
                        "Merak duplicate Mamba group ownership for layer "
                        f"{layer_name}: groups {previous_gid} and {gid}."
                    )
                name_to_group[layer_name] = gid

        missing = [name for name in placeholder_names if name not in name_to_group]
        if missing:
            raise RuntimeError(
                "Merak Mamba group mapping is missing layer(s): " + ", ".join(missing)
            )
        extras = sorted(set(name_to_group) - set(placeholder_names))
        if extras:
            raise RuntimeError(
                "Merak Mamba groups contain unregistered linear-state layer(s): "
                + ", ".join(extras)
            )

        layer_group_ids = [name_to_group[name] for name in placeholder_names]
        used_signatures = {mamba_group_signatures[gid] for gid in layer_group_ids}
        if len(used_signatures) != 1:
            details = ", ".join(
                f"group {gid}={mamba_group_signatures[gid]}"
                for gid in sorted(set(layer_group_ids))
            )
            raise RuntimeError(
                "Merak Mamba groups must share (block_size, num_speculative_blocks); got "
                + details
            )

        group_id = layer_group_ids[0]
        self._merak_mamba_group_id = group_id
        self._merak_layer_group_ids = layer_group_ids
        return group_id

    def _normalize_mamba_logical_index(
        self, logical_idx: int, *, width: int, group_id: int
    ) -> int:
        if width <= 0:
            raise RuntimeError(
                f"Merak Mamba group {group_id} has an empty block table."
            )
        if 0 <= logical_idx < width:
            return logical_idx

        prefix_caching = bool(
            getattr(self.cache_config, "enable_prefix_caching", False)
        )
        mamba_cache_mode = str(getattr(self.cache_config, "mamba_cache_mode", "none"))
        if not prefix_caching and mamba_cache_mode == "none" and width == 1:
            return 0

        mode = (
            "prefix caching enabled"
            if prefix_caching
            else f"prefix caching disabled, mode={mamba_cache_mode}"
        )
        raise RuntimeError(
            f"Merak Mamba group {group_id} {mode}: logical index {logical_idx} "
            f"is out of range for block-table width {width}; refusing to clamp."
        )

    def _validate_preallocation_runtime_contracts(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        # Expose the candidate configuration to the Mamba group validator
        # before the parent runner performs any multi-GiB cache allocation.
        self.kv_cache_config = kv_cache_config
        self._merak_mamba_group_id = "unset"
        merak_model = cast(Any, self.model)
        runtime_model = getattr(merak_model, "inference_model", None)
        if (
            bool(getattr(merak_model, "enable_page_attention", False))
            and getattr(merak_model, "placeholder_attn_layers", None)
            and not callable(getattr(runtime_model, "set_page_attention_context", None))
        ):
            raise RuntimeError(
                "Merak PageAttention initialization requires HMONNX runtime API "
                "set_page_attention_context(...), but it is unavailable."
            )
        self._get_kv_cache_layer_devices()
        self._resolve_mamba_group_id()

    def _validate_postprepare_runtime_contracts(self) -> None:
        merak_model = cast(Any, self.model)
        validate_linear_state = getattr(
            merak_model, "_validate_runtime_linear_state_contract", None
        )
        if callable(validate_linear_state):
            validate_linear_state()

    def _validate_runtime_contracts(self) -> None:
        """Validate all contracts for direct callers and focused tests."""
        self._validate_preallocation_runtime_contracts(self.kv_cache_config)
        self._validate_postprepare_runtime_contracts()

    def _mamba_block_size(self, group_id: int) -> int:
        spec = self.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
        return int(spec.block_size)

    def _mamba_num_speculative_blocks(self, group_id: int) -> int:
        spec = self.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
        return int(getattr(spec, "num_speculative_blocks", 0))

    def _initialize_linear_checkpoint_store(self) -> None:
        """Adopt vLLM's compact Mamba placeholder tensors as checkpoint slots.

        The placeholder tensors are the sole persistent Linear Attention state
        allocation.  The store only adds scheduler ownership metadata and views;
        it must never allocate a second copy of the state pool.
        """
        merak_model = cast(Any, self.model)
        placeholders = list(
            getattr(merak_model, "placeholder_linear_state_layers", None) or ()
        )
        if not placeholders:
            self.merak_linear_checkpoint_store = None
            return

        layer_group_ids = list(getattr(self, "_merak_layer_group_ids", ()))
        if len(layer_group_ids) != len(placeholders):
            raise RuntimeError(
                "Merak Linear checkpoint initialization requires one validated Mamba group ID per placeholder layer."
            )

        max_num_seqs = int(self.scheduler_config.max_num_seqs)
        layer_specs: list[LinearStateLayerSpec] = []
        storage_by_layer: dict[str, tuple[torch.Tensor, ...]] = {}
        for placeholder, group_id in zip(placeholders, layer_group_ids, strict=True):
            layer_name = str(placeholder.layer_name)
            states = tuple(placeholder.kv_cache)
            if not states or any(
                not isinstance(state, torch.Tensor) for state in states
            ):
                raise RuntimeError(
                    f"Merak Linear checkpoint layer {layer_name} has no bound Tensor state."
                )
            devices = {state.device for state in states}
            if len(devices) != 1:
                raise RuntimeError(
                    f"Merak Linear checkpoint layer {layer_name} spans devices {sorted(map(str, devices))}."
                )
            layer_specs.append(
                LinearStateLayerSpec(
                    layer_name=layer_name,
                    group_id=int(group_id),
                    shapes=tuple(
                        tuple(int(dim) for dim in state.shape[1:]) for state in states
                    ),
                    dtypes=tuple(state.dtype for state in states),
                    device=states[0].device,
                )
            )
            storage_by_layer[layer_name] = states

        self.merak_linear_checkpoint_store = MerakLinearCheckpointStore(
            layer_specs,
            max_num_seqs=max_num_seqs,
            storage_by_layer=storage_by_layer,
        )

    def _update_states(self, scheduler_output: "SchedulerOutput"):
        preempted = getattr(scheduler_output, "preempted_req_ids", None) or ()
        checkpoint_store = getattr(self, "merak_linear_checkpoint_store", None)
        if checkpoint_store is not None and preempted:
            raise RuntimeError(
                "Merak Linear checkpoint Phase 1 does not support request preemption; "
                "refusing to discard or restore incomplete runtime state."
            )

        uses_gemma4_contract = self._uses_gemma4_page_attention_contract()
        upstream_zero_ids = getattr(scheduler_output, "new_block_ids_to_zero", None)
        new_block_records = (
            self._gemma4_new_block_records(scheduler_output)
            if uses_gemma4_contract or not upstream_zero_ids
            else []
        )
        pending_zero_records = new_block_records if uses_gemma4_contract else []
        if pending_zero_records:
            self._gemma4_pending_zero_records = pending_zero_records
            if not upstream_zero_ids:
                # Upstream reports only full-attention allocations. Sliding
                # HMFP pages still need exact-group clearing before use.
                self._zero_gemma4_records_with_evidence(pending_zero_records)
                self._gemma4_pending_zero_records = []
        elif new_block_records:
            # vLLM records blocks for zeroing only when the cache spec has an
            # exact built-in FullAttentionSpec type. Merak's HMFPFullAttentionSpec
            # is a subclass, so hybrid Qwen requests otherwise reuse dirty SEFP
            # pages even though the model also has Mamba layers and therefore
            # advertises needs_kv_cache_zeroing=True.
            hmfp_group_blocks: dict[int, set[int]] = {}
            for record in new_block_records:
                group_idx = int(record["group_idx"])
                group = self.kv_cache_config.kv_cache_groups[group_idx]
                if _spec_is_hmfp_attention(group.kv_cache_spec):
                    hmfp_group_blocks.setdefault(group_idx, set()).update(
                        int(block_id) for block_id in record["block_ids"]
                    )
            self._zero_hmfp_group_blocks(hmfp_group_blocks)

        trace_gemma4 = self._gemma4_kv_trace_enabled()
        released_by_request = (
            {
                req_id: self._request_group_block_ids(self.requests[req_id])
                for req_id in scheduler_output.finished_req_ids or ()
                if req_id in self.requests
            }
            if trace_gemma4
            else {}
        )
        if trace_gemma4:
            prior_ownership = getattr(self, "_gemma4_trace_ownership", {})
            for req_id in set(prior_ownership).difference(self.requests):
                released_by_request.setdefault(req_id, prior_ownership[req_id])
            released_ids = getattr(self, "_gemma4_trace_released_block_ids", set())
            backing_ptrs = self._gemma4_group_backing_ptrs()
            for req_id, group_blocks in released_by_request.items():
                flattened = sorted(
                    {
                        block_id
                        for blocks in group_blocks.values()
                        for block_id in blocks
                    }
                )
                released_ids.update(
                    (group_idx, block_id)
                    for group_idx, blocks in group_blocks.items()
                    for block_id in blocks
                )
                release_records = [
                    {
                        "request_id": req_id,
                        "group_idx": group_idx,
                        "backing_ptr": backing_ptr,
                        "block_ids": sorted(set(block_ids)),
                    }
                    for group_idx, block_ids in group_blocks.items()
                    for backing_ptr in backing_ptrs.get(group_idx, ())
                ]
                emit_trace(
                    KV_TRACE_ENV,
                    component="kv_cache",
                    event="release",
                    payload={
                        "request_id": req_id,
                        "released_block_ids_by_group": {
                            str(group_idx): block_ids
                            for group_idx, block_ids in group_blocks.items()
                        },
                        "released_block_ids": flattened,
                        "release_records": release_records,
                        "unique_backings": self._gemma4_hmfp_backings(),
                    },
                )
            self._gemma4_trace_released_block_ids = released_ids
        result = super()._update_states(scheduler_output)
        self.__dict__.pop("_gemma4_pending_zero_records", None)
        finished = scheduler_output.finished_req_ids or ()
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        resumed = (
            getattr(cached_reqs, "resumed_req_ids", None) or () if cached_reqs else ()
        )

        merak_model = cast(Any, self.model)
        if bool(getattr(merak_model, "uses_internal_hmonnx_cache", False)):
            owner = getattr(merak_model, "_cache_owner_request_id", None)
            active_req_ids = set(self.input_batch.req_ids)
            if owner is not None and owner not in active_req_ids:
                merak_model.release_internal_cache_owner(owner)

        if checkpoint_store is not None:
            new_block_ids = tuple(
                int(block_id)
                for block_id in (
                    getattr(scheduler_output, "new_block_ids_to_zero", None) or ()
                )
            )
            if new_block_ids:
                checkpoint_store.invalidate_physical_ids(
                    {
                        group_id: new_block_ids
                        for group_id in checkpoint_store.group_layer_names
                    }
                )
            for req_id in finished:
                checkpoint_store.release_request(req_id)

            # vLLM can consume a finish-only scheduler step without invoking
            # the worker. In that case the next model-execution SchedulerOutput
            # has an empty finished_req_ids set, while the parent runner has
            # already replaced the completed request in its persistent batch.
            # Reconcile against that post-super batch so compact slots do not
            # remain owned forever across sequential requests.
            active_req_ids = set(self.input_batch.req_ids)
            stale_checkpoint_owners = [
                req_id
                for req_id in checkpoint_store.active_request_ids
                if req_id not in active_req_ids
            ]
            for req_id in stale_checkpoint_owners:
                checkpoint_store.release_request(req_id)

            runtime_owner = getattr(self, "_merak_runtime_linear_owner_req_id", None)
            if runtime_owner is not None and runtime_owner not in active_req_ids:
                self._merak_runtime_linear_owner_req_id = None

        for req_id in itertools.chain(finished, resumed):
            self.merak_linear_state_idx.pop(req_id, None)

        if self.merak_linear_state_idx:
            active_req_ids = set(self.input_batch.req_ids)
            stale = [
                rid for rid in self.merak_linear_state_idx if rid not in active_req_ids
            ]
            for req_id in stale:
                self.merak_linear_state_idx.pop(req_id, None)
        proposer = getattr(self, "_merak_spec_proposer", None)
        if proposer is not None:
            release_requests = getattr(proposer, "release_requests", None)
            if callable(release_requests) and finished:
                release_requests(tuple(finished))
            release_stale = getattr(
                proposer,
                "release_stale_requests",
                None,
            )
            if callable(release_stale):
                release_stale(tuple(self.input_batch.req_ids))
        if trace_gemma4:
            current_ownership = {
                req_id: self._request_group_block_ids(request)
                for req_id, request in self.requests.items()
            }
            self._gemma4_trace_ownership = current_ownership
            for req_id, group_blocks in current_ownership.items():
                emit_trace(
                    KV_TRACE_ENV,
                    component="kv_cache",
                    event="ownership",
                    payload={
                        "request_id": req_id,
                        "owned_block_ids_by_group": {
                            str(group_idx): block_ids
                            for group_idx, block_ids in group_blocks.items()
                        },
                        "owned_count_by_group": {
                            str(group_idx): len(set(block_ids))
                            for group_idx, block_ids in group_blocks.items()
                        },
                        "unique_backings": self._gemma4_hmfp_backings(),
                    },
                )
        return result

    def _gemma4_kv_trace_enabled(self) -> bool:
        if not trace_enabled(KV_TRACE_ENV):
            return False
        uses_contract = getattr(
            getattr(self, "model", None),
            "_uses_gemma4_page_attention_contract",
            None,
        )
        return bool(callable(uses_contract) and uses_contract())

    def _uses_gemma4_page_attention_contract(self) -> bool:
        uses_contract = getattr(
            getattr(self, "model", None),
            "_uses_gemma4_page_attention_contract",
            None,
        )
        return bool(callable(uses_contract) and uses_contract())

    def _gemma4_new_block_records(self, scheduler_output: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for request in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            for group_idx, block_ids in enumerate(request.block_ids):
                block_size = int(
                    self.kv_cache_config.kv_cache_groups[
                        group_idx
                    ].kv_cache_spec.block_size
                )
                num_existing_blocks = (
                    int(request.num_computed_tokens) + block_size - 1
                ) // block_size
                new_block_ids = block_ids[num_existing_blocks:]
                if new_block_ids:
                    records.append(
                        {
                            "request_id": request.req_id,
                            "group_idx": group_idx,
                            "block_ids": [int(value) for value in new_block_ids],
                            "logical_block_start": num_existing_blocks,
                        }
                    )
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None:
            for request_id, grouped_ids in zip(
                cached.req_ids, cached.new_block_ids, strict=True
            ):
                if grouped_ids is None:
                    continue
                for group_idx, block_ids in enumerate(grouped_ids):
                    if block_ids:
                        existing = getattr(
                            self.requests.get(request_id), "block_ids", ()
                        )
                        logical_block_start = (
                            len(existing[group_idx]) if group_idx < len(existing) else 0
                        )
                        records.append(
                            {
                                "request_id": request_id,
                                "group_idx": group_idx,
                                "block_ids": [int(value) for value in block_ids],
                                "logical_block_start": logical_block_start,
                            }
                        )
        return records

    @staticmethod
    def _request_group_block_ids(request: Any) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        for group_idx, raw_ids in enumerate(getattr(request, "block_ids", ()) or ()):
            if isinstance(raw_ids, torch.Tensor):
                if raw_ids.device.type != "cpu":
                    raise RuntimeError(
                        f"Gemma4 KV trace ownership must remain host-resident; got block IDs on {raw_ids.device}"
                    )
                values = raw_ids.detach().reshape(-1).tolist()
            else:
                values = list(raw_ids or ())
            result[group_idx] = [int(value) for value in values if int(value) >= 0]
        return result

    def _gemma4_hmfp_backings(self) -> list[dict[str, Any]]:
        layer_groups = {
            str(layer_name): group_idx
            for group_idx, group in enumerate(self.kv_cache_config.kv_cache_groups)
            for layer_name in group.layer_names
        }
        by_ptr: dict[int, dict[str, Any]] = {}
        for layer in getattr(
            getattr(self, "model", None), "placeholder_attn_layers", ()
        ):
            group_idx = layer_groups.get(str(getattr(layer, "layer_name", "")))
            cache = getattr(layer, "kv_cache", None)
            if group_idx is None or not isinstance(cache, HMFPPagedKVCache):
                continue
            tensors = cache.tensors()
            ptr = int(tensors[0].data_ptr())
            entry = by_ptr.setdefault(
                ptr,
                {
                    "ptr": ptr,
                    "bytes": sum(
                        tensor.numel() * tensor.element_size() for tensor in tensors
                    ),
                    "num_blocks": cache.num_blocks,
                    "block_size": cache.block_size,
                    "cache_group_indices": [],
                },
            )
            if group_idx not in entry["cache_group_indices"]:
                entry["cache_group_indices"].append(group_idx)
        return sorted(by_ptr.values(), key=lambda entry: entry["ptr"])

    def _gemma4_group_backing_ptrs(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        for backing in self._gemma4_hmfp_backings():
            for group_idx in backing["cache_group_indices"]:
                result.setdefault(int(group_idx), []).append(int(backing["ptr"]))
        return result

    def _gemma4_physical_block_ids(
        self,
        group_idx: int,
        scheduler_block_ids: set[int] | list[int],
    ) -> set[int]:
        """Project scheduler-output ids; attention metadata is already physical.

        Scheduler allocation/release records retain the KV manager's virtual
        block ids, unlike MultiGroupBlockTable metadata, which vLLM has already
        split into kernel-block ids before model execution.
        """

        bindings = [
            binding
            for binding in getattr(
                getattr(self, "model", None), "page_attention_bindings", ()
            )
            if binding.cache_group_idx == group_idx
        ]
        if not bindings:
            scheduler_block_size = int(
                self.kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec.block_size
            )
            if scheduler_block_size % 64 != 0:
                raise RuntimeError(
                    "Gemma4 scheduler block size is not divisible by physical "
                    f"64-token pages: cache_group_idx={group_idx}, "
                    f"scheduler={scheduler_block_size}"
                )
            ratios = {scheduler_block_size // 64}
        else:
            ratios = {
                binding.physical_pages_per_scheduler_block for binding in bindings
            }
        if len(ratios) != 1:
            raise RuntimeError(
                "Gemma4 cache group has inconsistent scheduler/physical page "
                f"ratios: cache_group_idx={group_idx}, ratios={sorted(ratios)}"
            )
        ratio = next(iter(ratios))
        fixed_page_counts = {
            binding.fixed_physical_pages
            for binding in bindings
            if binding.fixed_physical_pages is not None
        }
        allocated_fixed_pages = self._gemma4_allocated_fixed_pages(
            group_idx, fixed_page_counts
        )
        if allocated_fixed_pages is not None:
            if len(fixed_page_counts) != 1:
                raise RuntimeError(
                    "Gemma4 cache group has inconsistent fixed page capacities: "
                    f"cache_group_idx={group_idx}, pages={sorted(fixed_page_counts)}"
                )
            return {
                (int(block_id) * ratio + physical_offset) % allocated_fixed_pages
                for block_id in scheduler_block_ids
                for physical_offset in range(ratio)
            }
        return {
            int(block_id) * ratio + physical_offset
            for block_id in scheduler_block_ids
            for physical_offset in range(ratio)
        }

    def _gemma4_allocated_fixed_pages(
        self, group_idx: int, declared_pages: set[int | None]
    ) -> int | None:
        declared = {int(value) for value in declared_pages if value is not None}
        if not declared:
            return None
        if len(declared) != 1:
            raise RuntimeError(
                "Gemma4 cache group has inconsistent fixed page capacities: "
                f"cache_group_idx={group_idx}, pages={sorted(declared)}"
            )
        expected = next(iter(declared))
        layer_names = set(self.kv_cache_config.kv_cache_groups[group_idx].layer_names)
        allocated = {
            cache.num_blocks
            for layer in getattr(self.model, "placeholder_attn_layers", ())
            if str(getattr(layer, "layer_name", "")) in layer_names
            and isinstance(
                (cache := getattr(layer, "kv_cache", None)), HMFPPagedKVCache
            )
        }
        return expected if allocated == {expected} else None

    def _gemma4_record_physical_block_ids(self, record: dict[str, Any]) -> set[int]:
        group_idx = int(record["group_idx"])
        bindings = [
            binding
            for binding in getattr(
                getattr(self, "model", None), "page_attention_bindings", ()
            )
            if binding.cache_group_idx == group_idx
        ]
        fixed_page_counts = {
            binding.fixed_physical_pages
            for binding in bindings
            if binding.fixed_physical_pages is not None
        }
        fixed_pages = self._gemma4_allocated_fixed_pages(group_idx, fixed_page_counts)
        if fixed_pages is None:
            return self._gemma4_physical_block_ids(
                group_idx, set(int(value) for value in record["block_ids"])
            )
        if len(fixed_page_counts) != 1:
            raise RuntimeError(
                "Gemma4 cache group has inconsistent fixed page capacities: "
                f"cache_group_idx={group_idx}, pages={sorted(fixed_page_counts)}"
            )
        ratios = {binding.physical_pages_per_scheduler_block for binding in bindings}
        if len(ratios) != 1:
            raise RuntimeError(
                "Gemma4 cache group has inconsistent scheduler/physical page "
                f"ratios: cache_group_idx={group_idx}, ratios={sorted(ratios)}"
            )
        ratio = next(iter(ratios))
        logical_start = int(record.get("logical_block_start", 0))
        return {
            (logical_block * ratio + physical_offset) % fixed_pages
            for logical_block in range(
                logical_start, logical_start + len(record["block_ids"])
            )
            for physical_offset in range(ratio)
        }

    def _zero_gemma4_records(self, records: list[dict[str, Any]]) -> None:
        physical_grouped: dict[int, set[int]] = {}
        for record in records:
            physical_grouped.setdefault(int(record["group_idx"]), set()).update(
                self._gemma4_record_physical_block_ids(record)
            )
        self._zero_hmfp_group_blocks(physical_grouped)

    def _emit_gemma4_zero_complete(
        self,
        block_ids: list[int],
        pending_records: list[dict[str, Any]] | None,
    ) -> None:
        if not self._gemma4_kv_trace_enabled():
            return

        released_ids = getattr(self, "_gemma4_trace_released_block_ids", set())
        recycled_records = [
            {
                "request_id": record["request_id"],
                "group_idx": int(record["group_idx"]),
                "block_ids": sorted(
                    block_id
                    for block_id in set(record["block_ids"])
                    if (int(record["group_idx"]), block_id) in released_ids
                ),
            }
            for record in pending_records or ()
        ]
        recycled_records = [
            record for record in recycled_records if record["block_ids"]
        ]
        recycled = sorted(
            {
                block_id
                for record in recycled_records
                for block_id in record["block_ids"]
            }
        )
        if pending_records is None:
            recycled = sorted(
                set(block_ids).intersection(block_id for _, block_id in released_ids)
            )
        backing_ptrs = self._gemma4_group_backing_ptrs()
        zero_records = []
        for record in pending_records or ():
            for backing_ptr in backing_ptrs.get(int(record["group_idx"]), ()):
                zero_record = {
                    "request_id": record["request_id"],
                    "group_idx": int(record["group_idx"]),
                    "backing_ptr": backing_ptr,
                    "block_ids": sorted(set(record["block_ids"])),
                }
                physical_ids = sorted(self._gemma4_record_physical_block_ids(record))
                if physical_ids != zero_record["block_ids"]:
                    zero_record["physical_block_ids"] = physical_ids
                zero_records.append(zero_record)
        normalized_ids = sorted(set(block_ids))
        emit_trace(
            KV_TRACE_ENV,
            component="kv_cache",
            event="zero_complete",
            payload={
                "new_block_ids": normalized_ids,
                "recycled_block_ids": recycled,
                "recycled_records": recycled_records,
                "zeroed_block_ids": normalized_ids,
                "ordering": "before_model_execute",
                "zero_records": zero_records,
                "unique_backings": self._gemma4_hmfp_backings(),
            },
        )

    def _zero_gemma4_records_with_evidence(
        self,
        records: list[dict[str, Any]],
        *,
        standard_block_ids: list[int] | None = None,
    ) -> None:
        """Zero exact Gemma groups once and emit the matching proof once."""

        self._zero_gemma4_records(records)
        if standard_block_ids:
            # Retain vLLM's normal zeroer for standard Tensor-backed caches.
            super()._zero_block_ids(standard_block_ids)
        exact_block_ids = sorted(
            {int(block_id) for record in records for block_id in record["block_ids"]}
        )
        self._emit_gemma4_zero_complete(exact_block_ids, records)

    def _zero_hmfp_group_blocks(
        self,
        group_block_ids: dict[int, set[int]],
    ) -> None:
        if not any(group_block_ids.values()):
            return

        layer_group_ids = {
            str(layer_name): group_id
            for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups)
            if group_id in group_block_ids
            for layer_name in group.layer_names
        }
        backing_blocks: dict[
            tuple[torch.device, int], tuple[HMFPPagedKVCache, set[int]]
        ] = {}
        for attn_layer in getattr(self.model, "placeholder_attn_layers", ()):
            group_id = layer_group_ids.get(str(getattr(attn_layer, "layer_name", "")))
            if group_id is None or not group_block_ids[group_id]:
                continue
            cache = getattr(attn_layer, "kv_cache", None)
            if not isinstance(cache, HMFPPagedKVCache):
                continue
            cache_key = (cache.k_qdata.device, cache.k_qdata.data_ptr())
            if cache_key not in backing_blocks:
                backing_blocks[cache_key] = (cache, set())
            backing_blocks[cache_key][1].update(group_block_ids[group_id])

        for cache, block_ids in backing_blocks.values():
            for _, tensor in cache.named_tensors():
                ids = torch.as_tensor(
                    sorted(block_ids),
                    dtype=torch.long,
                    device=tensor.device,
                )
                tensor.index_fill_(0, ids, 0)

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        """Clear recycled SEFP pages before their first read-modify-write.

        Merak wraps each FullAttention cache tensor in ``HMFPPagedKVCache``.
        vLLM's generic block zeroer skips non-Tensor cache handles, but SEFP V
        writes requantize all 64 lanes sharing an exponent.  Leaving unwritten
        lanes from the previous owner would therefore change the new request's
        valid values.
        """
        if not block_ids:
            return

        pending_records = getattr(self, "_gemma4_pending_zero_records", None)
        if pending_records is None:
            hmfp_group_ids = {
                group_id: (
                    self._gemma4_physical_block_ids(group_id, set(block_ids))
                    if self._uses_gemma4_page_attention_contract()
                    else set(block_ids)
                )
                for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups)
                if _spec_is_hmfp_attention(group.kv_cache_spec)
            }
            self._zero_hmfp_group_blocks(hmfp_group_ids)
            # Preserve vLLM's normal zeroer for any standard Tensor-backed
            # caches that coexist with Merak's typed HMFP handles.
            super()._zero_block_ids(block_ids)
            self._emit_gemma4_zero_complete(block_ids, None)
        else:
            self._zero_gemma4_records_with_evidence(
                pending_records,
                standard_block_ids=block_ids,
            )

    def _install_cpu_slot_mapping_fallback(self) -> None:
        """Avoid vLLM's CUDA Triton slot-mapping kernel for CPU tensors.

        Merak presents itself as a CPU vLLM platform while the HMONNX runtime
        manages the actual accelerator placement.  With CUDA vLLM wheels, the
        inherited CPUModelRunner can still dispatch BlockTable slot mapping to
        a NVIDIA Triton kernel.  That kernel cannot consume the CPU buffers used
        by this runner, so compute the same mapping with torch on CPU.
        """

        def compute_slot_mapping_cpu(
            table: Any,
            num_reqs: int,
            query_start_loc: torch.Tensor,
            positions: torch.Tensor,
        ) -> None:
            num_tokens = int(positions.shape[0])
            slot_mapping = table.slot_mapping.gpu
            if num_tokens < table.max_num_batched_tokens:
                slot_mapping[num_tokens : table.max_num_batched_tokens].fill_(
                    PAD_SLOT_ID
                )

            query_start_cpu = query_start_loc.to(device="cpu", dtype=torch.long)
            positions_cpu = positions.to(device="cpu", dtype=torch.long)
            block_table_cpu = table.block_table.gpu.to(device="cpu")

            total_cp_world_size = table.pcp_world_size * table.dcp_world_size
            total_cp_rank = table.pcp_rank * table.dcp_world_size + table.dcp_rank
            virtual_block_size = table.block_size * total_cp_world_size
            interleave = table.cp_kv_cache_interleave_size

            for req_idx in range(num_reqs):
                start_idx = int(query_start_cpu[req_idx])
                end_idx = int(query_start_cpu[req_idx + 1])
                if end_idx <= start_idx:
                    continue

                pos = positions_cpu[start_idx:end_idx]
                block_indices = torch.div(
                    pos, virtual_block_size, rounding_mode="floor"
                )
                block_numbers = block_table_cpu[req_idx, block_indices].to(torch.long)

                virtual_block_offsets = pos - block_indices * virtual_block_size
                is_local = (
                    (virtual_block_offsets // interleave) % total_cp_world_size
                ) == total_cp_rank
                local_block_offsets = (
                    virtual_block_offsets // (total_cp_world_size * interleave)
                ) * interleave + (virtual_block_offsets % interleave)
                slot_ids = block_numbers * table.block_size + local_block_offsets
                slot_ids = torch.where(
                    is_local,
                    slot_ids,
                    torch.full_like(slot_ids, PAD_SLOT_ID),
                )
                slot_mapping[start_idx:end_idx].copy_(slot_ids.to(slot_mapping.device))

            if hasattr(table.slot_mapping, "cpu"):
                table.slot_mapping.cpu[: table.max_num_batched_tokens].copy_(
                    slot_mapping[: table.max_num_batched_tokens].to(device="cpu"),
                    non_blocking=True,
                )

        multi_group_block_table = self.input_batch.block_table
        block_tables = getattr(
            multi_group_block_table,
            "block_tables",
            (multi_group_block_table,),
        )
        for block_table in block_tables:
            if hasattr(block_table, "_merak_original_compute_slot_mapping"):
                # Already patched (e.g. reused table across reinit); skip to
                # avoid wrapping our own wrapper.
                continue
            block_table._merak_original_compute_slot_mapping = (
                block_table.compute_slot_mapping
            )
            block_table.compute_slot_mapping = MethodType(
                compute_slot_mapping_cpu, block_table
            )

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        contract = getattr(self.model, "gemma4_cache_contract", None)
        if contract is not None:
            from ..model_executor.model_loader.merak_for_causal_lm import (
                bind_gemma4_cache_groups,
            )

            self.model.page_attention_bindings = bind_gemma4_cache_groups(
                contract, kv_cache_config.kv_cache_groups
            )
            self._apply_gemma4_cache_tensor_ownership(
                kv_cache_config, self.model.page_attention_bindings
            )
        self._validate_preallocation_runtime_contracts(kv_cache_config)
        super().initialize_kv_cache(kv_cache_config, is_profiling)

        self._publish_finalized_attention_block_size()
        self._wrap_sefp_kv_cache_handles()
        self._install_cpu_slot_mapping_fallback()

        self.model.inference_model.get_kvcache_mixin().prepare_kv_cache()  # type: ignore
        self.model.inference_model.get_kvcache_mixin().prepare_other_cache()  # type: ignore
        self._validate_postprepare_runtime_contracts()
        self._initialize_linear_checkpoint_store()

    def _publish_finalized_attention_block_size(self) -> None:
        """Publish the logical attention page width selected by EngineCore.

        vLLM may replace the global cache block size with the minimum width of
        all hybrid groups after planning.  HMONNX request metadata must keep
        using the finalized attention spec's logical width.
        """

        merak_model = cast(Any, self.model)
        if self._uses_gemma4_page_attention_contract():
            return
        attention_layer_names = {
            str(getattr(layer, "layer_name", ""))
            for layer in getattr(merak_model, "placeholder_attn_layers", ())
        }
        block_sizes: set[int] = set()
        for group in getattr(
            getattr(self, "kv_cache_config", None), "kv_cache_groups", ()
        ):
            for layer_name, spec in resolve_finalized_group_layer_specs(group).items():
                if layer_name in attention_layer_names:
                    block_sizes.add(int(spec.block_size))
        if not block_sizes:
            return
        if len(block_sizes) != 1:
            raise RuntimeError(
                "Merak HMONNX attention layers require one logical block size; "
                f"finalized sizes={sorted(block_sizes)}"
            )
        merak_model._merak_attention_block_size = block_sizes.pop()

    def _wrap_sefp_kv_cache_handles(self) -> None:
        merak_model = cast(Any, self.model)
        cache_dtype = getattr(merak_model, "cache_dtype", self.cache_dtype)
        if not is_hmfp_cache_dtype(cache_dtype):
            return

        placeholder_attn_layers = getattr(merak_model, "placeholder_attn_layers", [])
        layers_by_name = {
            str(getattr(layer, "layer_name", "")): layer
            for layer in placeholder_attn_layers
        }
        # EngineCore replaces ``vllm_config.cache_config.block_size`` with the
        # minimum block size across finalized hybrid groups.  For Qwen3.5 this
        # can be the Mamba block size (for example 16), while the finalized
        # HMFP attention spec still owns 64-token physical pages.  Recomputing
        # an attention spec from the mutated global config therefore produces
        # the wrong layout.  Bind handles from the finalized per-layer specs
        # that were used to allocate the backing tensors.
        uses_gemma4_contract = self._uses_gemma4_page_attention_contract()
        finalized_specs: dict[str, KVCacheSpec] = {}
        if not uses_gemma4_contract:
            for group in getattr(
                getattr(self, "kv_cache_config", None), "kv_cache_groups", ()
            ):
                finalized_specs.update(resolve_finalized_group_layer_specs(group))
        wrapped_by_owner: dict[tuple[object, ...], HMFPPagedKVCache] = {}
        signature_by_raw_owner: dict[tuple[torch.device, int], tuple[object, ...]] = {}
        for attn_layer in placeholder_attn_layers:
            raw_cache = getattr(attn_layer, "kv_cache", None)
            if not isinstance(raw_cache, torch.Tensor) or raw_cache.numel() == 0:
                continue
            layer_name = str(getattr(attn_layer, "layer_name", ""))
            spec = finalized_specs.get(layer_name)
            if spec is None:
                spec = attn_layer.get_kv_cache_spec(self.vllm_config)
            if spec is None:
                owner_name = getattr(attn_layer, "kv_sharing_target_layer_name", None)
                owner_layer = layers_by_name.get(str(owner_name))
                if owner_layer is None:
                    raise RuntimeError(
                        "Merak shared KV layer has no registered owner: "
                        f"layer={getattr(attn_layer, 'layer_name', None)!r}, "
                        f"owner={owner_name!r}"
                    )
                spec = finalized_specs.get(str(owner_name))
                if spec is None:
                    spec = owner_layer.get_kv_cache_spec(self.vllm_config)
                if spec is None:
                    raise RuntimeError(
                        f"Merak KV owner {owner_name!r} did not publish a cache spec"
                    )
            spec_cache_dtype = getattr(
                spec, "sefp_cache_dtype", getattr(spec, "dtype", None)
            )
            if not is_hmfp_cache_dtype(spec_cache_dtype):
                # The mixed Gemma4 sliding cache is already reshaped by
                # _reshape_kv_cache_tensors into [2,N,H,B,D] FP storage.
                continue
            spec_signature = self._shared_cache_spec_signature(spec)
            raw_owner = (raw_cache.device, raw_cache.data_ptr())
            prior_signature = signature_by_raw_owner.setdefault(
                raw_owner, spec_signature
            )
            if prior_signature != spec_signature:
                raise RuntimeError(
                    "Merak shared raw KV cache has incompatible attention specs"
                )
            owner_key = (*raw_owner, spec_signature)
            wrapped_cache = wrapped_by_owner.get(owner_key)
            if wrapped_cache is None:
                raw_cache.zero_()
                head_size = int(attn_layer.head_size)
                raw_head_size_v = getattr(spec, "head_size_v", None)
                if raw_head_size_v is None:
                    raw_head_size_v = getattr(attn_layer, "head_size_v", None)
                head_size_v = (
                    head_size if raw_head_size_v is None else int(raw_head_size_v)
                )
                wrapped_cache = hmfp_paged_kv_cache_from_raw_tensor(
                    raw_cache,
                    block_size=int(spec.block_size),
                    num_kv_heads=int(attn_layer.num_kv_heads),
                    head_size=head_size,
                    head_size_v=head_size_v,
                    cache_dtype=str(getattr(spec, "sefp_cache_dtype", cache_dtype)),
                )
                wrapped_by_owner[owner_key] = wrapped_cache
            attn_layer.kv_cache = wrapped_cache

        if uses_gemma4_contract:
            bindings_by_layer = {
                binding.layer_name: binding
                for binding in getattr(merak_model, "page_attention_bindings", ())
            }
            scheduler_blocks = int(self.kv_cache_config.num_blocks)
            for attn_layer in placeholder_attn_layers:
                layer_name = str(getattr(attn_layer, "layer_name", ""))
                binding = bindings_by_layer.get(layer_name)
                cache = getattr(attn_layer, "kv_cache", None)
                expects_fp_sliding = (
                    getattr(
                        getattr(merak_model, "gemma4_cache_contract", None),
                        "attention_lowering",
                        "flash_attention",
                    )
                    == "full_flash_attention"
                    and binding is not None
                    and binding.attention_kind == "sliding_attention"
                )
                if (
                    binding is None
                    or (
                        not expects_fp_sliding
                        and not isinstance(cache, HMFPPagedKVCache)
                    )
                    or (expects_fp_sliding and not isinstance(cache, torch.Tensor))
                ):
                    raise RuntimeError(
                        f"Gemma4 physical cache capacity validation is missing layer={layer_name!r}"
                    )
                required_physical_blocks = self._gemma4_required_physical_blocks(
                    binding, scheduler_blocks
                )
                cache_block_size = (
                    int(cache.shape[3])
                    if isinstance(cache, torch.Tensor)
                    else int(cache.block_size)
                )
                cache_num_blocks = (
                    int(cache.shape[1])
                    if isinstance(cache, torch.Tensor)
                    else int(cache.num_blocks)
                )
                if cache_block_size != binding.physical_block_size:
                    raise RuntimeError(
                        "Gemma4 HMONNX physical block size mismatch: "
                        f"layer={layer_name}, cache={cache_block_size}, "
                        f"binding={binding.physical_block_size}"
                    )
                if cache_num_blocks < required_physical_blocks:
                    raise RuntimeError(
                        "Gemma4 physical cache is smaller than the finalized "
                        "scheduler projection: "
                        f"layer={layer_name}, physical_blocks={cache_num_blocks}, "
                        f"required={required_physical_blocks}"
                    )

    @staticmethod
    def _begin_merak_hybrid_verify_transaction(runtime: Any) -> None:
        from xhmodel_merak.xh_llm.models.qwen3_5.hybrid_cache_runtime import (
            begin_hybrid_cache_transaction,
        )

        begin_hybrid_cache_transaction(runtime)

    @staticmethod
    def _detach_merak_hybrid_verify_transaction(runtime: Any) -> Any:
        from xhmodel_merak.xh_llm.models.qwen3_5.hybrid_cache_runtime import (
            detach_hybrid_cache_transaction,
        )

        return detach_hybrid_cache_transaction(runtime)

    @staticmethod
    def _abort_merak_hybrid_verify_transaction(runtime: Any) -> None:
        from xhmodel_merak.xh_llm.models.qwen3_5.hybrid_cache_runtime import (
            abort_hybrid_cache_transaction,
        )

        abort_hybrid_cache_transaction(runtime)

    @staticmethod
    def _begin_merak_hybrid_output_passthrough(runtime: Any) -> None:
        from xhmodel_merak.xh_llm.models.qwen3_5.hybrid_cache_runtime import (
            begin_hybrid_cache_output_passthrough,
        )

        # Non-verify decode has one real token in a graph exported at the
        # K+1 verify width. Commit step zero and preserve raw hidden outputs.
        begin_hybrid_cache_output_passthrough(
            runtime,
            selected_step=0,
        )

    @staticmethod
    def _end_merak_hybrid_output_passthrough(runtime: Any) -> None:
        from xhmodel_merak.xh_llm.models.qwen3_5.hybrid_cache_runtime import (
            end_hybrid_cache_output_passthrough,
        )

        end_hybrid_cache_output_passthrough(runtime)

    @staticmethod
    def _uses_qwen_hybrid_state_contract(runtime: Any) -> bool:
        """Detect a fixed-width Qwen target graph from its runtime contract.

        The target graph still exports one Linear-state snapshot per verify
        position when speculative decoding is disabled at deployment time.
        Detect that contract from the target runtime rather than from the
        optional proposer: ordinary Q=1 decode must select snapshot zero even
        when no MTP/DFlash proposer is attached.
        """

        get_verify_steps = getattr(runtime, "_get_spec_decode_verify_steps", None)
        return callable(get_verify_steps) and int(get_verify_steps()) > 1

    @staticmethod
    def _commit_merak_hybrid_verify_transaction(
        runtime: Any,
        *,
        accepted_steps: int,
        transaction: Any,
    ) -> None:
        from xhmodel_merak.xh_llm.models.qwen3_5.hybrid_cache_runtime import (
            commit_hybrid_cache_transaction,
        )

        commit_hybrid_cache_transaction(
            runtime,
            accepted_steps,
            model_label="Qwen3.5 Merak",
            transaction=transaction,
        )

    def _abort_all_merak_hybrid_verify_transactions(self) -> None:
        merak_model = getattr(self, "model", None)
        abort_page_verify = getattr(
            merak_model,
            "_abort_page_attention_verify_transaction",
            None,
        )
        if callable(abort_page_verify):
            abort_page_verify()
        runtime = getattr(
            merak_model,
            "inference_model",
            None,
        )
        if runtime is not None and (
            getattr(runtime, "_hybrid_cache_transaction", None) is not None
            or getattr(runtime, "_defer_hybrid_cache_commit", False)
        ):
            self._abort_merak_hybrid_verify_transaction(runtime)
        self._merak_pending_linear_verifies.clear()
        getattr(self, "_merak_pending_page_verifies", {}).clear()
        self._merak_pending_verify_req_ids.clear()
        proposer = getattr(self, "_merak_spec_proposer", None)
        abort_pending = getattr(proposer, "abort_pending_verifies", None)
        if callable(abort_pending):
            abort_pending()

    def _commit_pending_merak_page_verify(
        self,
        req_id: str,
        *,
        accepted_steps: int,
    ) -> None:
        pending_page_verifies = getattr(
            self,
            "_merak_pending_page_verifies",
            {},
        )
        query_len = pending_page_verifies.get(req_id)
        if query_len is None:
            return
        if not 1 <= accepted_steps <= query_len:
            raise RuntimeError(
                "Merak accepted target steps are outside the PageAttention "
                f"verify block: request={req_id}, accepted={accepted_steps}, "
                f"verify_length={query_len}"
            )
        merak_model = cast(Any, self.model)
        merak_model._commit_page_attention_verify_transaction(
            request_id=req_id,
            accepted_steps=accepted_steps,
        )
        del pending_page_verifies[req_id]

    def _commit_pending_merak_linear_verify(
        self,
        req_id: str,
        *,
        accepted_steps: int,
    ) -> None:
        pending = self._merak_pending_linear_verifies.get(req_id)
        if pending is None:
            return
        query_len = int(pending["query_len"])
        if not 1 <= accepted_steps <= query_len:
            raise RuntimeError(
                "Merak accepted target steps are outside the verify block: "
                f"request={req_id}, accepted={accepted_steps}, "
                f"verify_length={query_len}"
            )

        merak_model = cast(Any, self.model)
        runtime = merak_model.inference_model
        checkpoint_store = self.merak_linear_checkpoint_store
        if checkpoint_store is None:
            raise RuntimeError("Merak Qwen verify has no Linear checkpoint store")
        # The verify transaction deliberately leaves the accepted input state
        # untouched.  With the serial HMONNX speculative contract the runtime
        # therefore still owns this request when rejection sampling commits.
        # Re-loading the checkpoint here copied every conv/recurrent tensor for
        # no semantic effect.  Keep a fail-safe restore for a future caller
        # that switches runtime ownership between verify and commit.
        if getattr(self, "_merak_runtime_linear_owner_req_id", None) != req_id:
            previous_slot = checkpoint_store.load_slot(
                req_id,
                pending["previous_physical_ids"],
                past_len=int(pending["past_len"]),
            )
            merak_model._load_runtime_linear_state_from_store(previous_slot)
            self._merak_runtime_linear_owner_req_id = req_id
        self._commit_merak_hybrid_verify_transaction(
            runtime,
            accepted_steps=accepted_steps,
            transaction=pending["transaction"],
        )

        total_len = int(pending["past_len"]) + accepted_steps
        block_size = int(pending["block_size"])
        num_speculative_blocks = int(pending["num_speculative_blocks"])
        num_blocks = (total_len + block_size - 1) // block_size + num_speculative_blocks
        logical_idx = max(0, num_blocks - 1 - num_speculative_blocks)
        destination_physical_ids: dict[int, int] = {}
        for group_id, row in pending["group_rows"].items():
            width = int(len(row))
            normalized_idx = self._normalize_mamba_logical_index(
                logical_idx,
                width=width,
                group_id=int(group_id),
            )
            destination_physical_ids[int(group_id)] = int(row[normalized_idx])

        merak_model._scatter_runtime_linear_state_to_store(
            checkpoint_store,
            req_id,
        )
        checkpoint_store.transition(req_id, destination_physical_ids)
        self.merak_linear_state_idx[req_id] = logical_idx
        del self._merak_pending_linear_verifies[req_id]

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        del intermediate_tensors

        if input_ids is None and inputs_embeds is None:
            raise ValueError(
                "MerakModelRunnerV1._model_forward needs either input_ids or inputs_embeds."
            )

        merak_model = cast(Any, self.model)
        if bool(getattr(merak_model, "uses_internal_hmonnx_cache", False)):
            # DeepSeek-V4 Flash owns a heterogeneous SWA/CSA/HCA cache inside
            # HMONNX.  It is not representable as vLLM's per-layer K/V pages,
            # so let the architecture adapter consume the scheduler metadata
            # and keep its request-local cache across steps.
            req_ids = list(self.input_batch.req_ids)
            if len(req_ids) != 1:
                raise RuntimeError(
                    "DeepSeek-V4 internal HMONNX cache requires exactly one active "
                    f"request, got {req_ids!r}."
                )
            prompt_logprobs = getattr(self, "num_prompt_logprobs", {})
            if any(int(value or 0) > 0 for value in prompt_logprobs.values()):
                raise RuntimeError(
                    "DeepSeek-V4 HMONNX exports only final-token logits and does not "
                    "support prompt logprobs."
                )
            spec_proposer = getattr(self, "_merak_spec_proposer", None)
            if spec_proposer is not None:
                return spec_proposer.forward_target(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    request_id=req_ids[0],
                    **model_kwargs,
                )
            return merak_model.forward(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                request_id=req_ids[0],
                **model_kwargs,
            )
        uses_gemma4_group_contexts = bool(
            callable(getattr(merak_model, "_uses_gemma4_page_attention_contract", None))
            and merak_model._uses_gemma4_page_attention_contract()
        )
        metadata_by_group: dict[int, MerakAttentionMetadata] = {}
        if uses_gemma4_group_contexts:
            metadata_by_group = self._get_attn_metadata_by_group()
            attn_metadata = self._common_gemma4_request_metadata(metadata_by_group)
        else:
            attn_metadata = self._get_attn_metadata()

        flat_input_ids = (
            merak_model._flatten_input_ids(input_ids) if input_ids is not None else None
        )
        flat_inputs_embeds = (
            merak_model._flatten_inputs_embeds(inputs_embeds)
            if inputs_embeds is not None
            else None
        )

        flat_inputs = (
            flat_inputs_embeds if flat_inputs_embeds is not None else flat_input_ids
        )
        if flat_inputs is None:
            raise ValueError(
                "MerakModelRunnerV1._model_forward failed to flatten model inputs."
            )

        device = self.device
        flat_logits = torch.zeros(
            flat_inputs.shape[0],
            merak_model.logits_processor.vocab_size,
            device=device,
            dtype=merak_model.input_embeddings.weight.dtype,
        )

        placeholder_attn_layers = merak_model.placeholder_attn_layers
        if (
            attn_metadata is None
            or not placeholder_attn_layers
            or placeholder_attn_layers[0].kv_cache.numel() == 0
        ):
            return flat_logits

        num_reqs = int(attn_metadata.past_lens.shape[0])
        active_req_ids = list(self.input_batch.req_ids)
        if len(active_req_ids) == num_reqs:
            req_ids = active_req_ids
        else:
            req_ids = [f"__merak_batch_req_{idx}" for idx in range(num_reqs)]

        enable_page_attention = (
            merak_model.enable_page_attention
            if hasattr(merak_model, "enable_page_attention")
            else False
        )
        spec_proposer = getattr(self, "_merak_spec_proposer", None)
        spec_artifact = getattr(spec_proposer, "artifact", None)
        runtime_uses_qwen_spec_hybrid = self._uses_qwen_hybrid_state_contract(
            getattr(merak_model, "inference_model", None)
        )
        uses_qwen_spec_hybrid = bool(
            runtime_uses_qwen_spec_hybrid
            or (
                spec_artifact is not None
                and getattr(spec_artifact, "is_qwen_hybrid_spec_decode", False)
            )
        )
        hidden_output_name = (
            getattr(spec_artifact, "target_hidden_output_name", None)
            if spec_artifact is not None
            else None
        )
        verify_req_ids = getattr(self, "_merak_verify_req_ids", set())

        # Linear-attention (MambaSpec) state bridging. Only enabled for hybrid
        # Merak models where vLLM allocated a mamba cache group and we can map
        # requests to their persistent batch rows (real req_ids).
        mamba_group_id = self._resolve_mamba_group_id()
        manage_linear_state = mamba_group_id is not None and req_ids is active_req_ids
        mamba_layer_group_ids: list[int] = []
        mamba_group_tables: dict[int, Any] = {}
        mamba_block_size = 0
        mamba_num_spec_blocks = 0
        if manage_linear_state:
            mamba_layer_group_ids = list(self._merak_layer_group_ids)
            # One block table per distinct mamba group; the block_size and
            # num_speculative_blocks are shared across the split groups (all are
            # merged from the same MambaSpec), so read them from the first group.
            for gid in set(mamba_layer_group_ids):
                mamba_group_tables[gid] = self.input_batch.block_table[
                    gid
                ].get_numpy_array()
            mamba_block_size = self._mamba_block_size(mamba_group_id)
            mamba_num_spec_blocks = self._mamba_num_speculative_blocks(mamba_group_id)
            if getattr(self, "merak_linear_checkpoint_store", None) is None:
                raise RuntimeError(
                    "Merak Mamba groups are active but the Linear checkpoint store was not initialized."
                )
            if not getattr(self, "_merak_linear_logged", False):
                logger.info(
                    "Merak linear-state bridging active: %d layers across mamba groups %s "
                    "(block_size=%d, num_speculative_blocks=%d).",
                    len(mamba_layer_group_ids),
                    sorted(mamba_group_tables),
                    mamba_block_size,
                    mamba_num_spec_blocks,
                )
                self._merak_linear_logged = True
        if uses_qwen_spec_hybrid and not manage_linear_state:
            raise RuntimeError(
                "Qwen HMONNX speculative decoding requires the Merak "
                "Linear/Mamba checkpoint bridge, but no managed Mamba cache "
                "group is active"
            )

        image_grid_offset = 0
        video_grid_offset = 0
        # Hoist all D2H syncs out of the per-request loop.
        past_lens_list = attn_metadata.past_lens.tolist()
        subsequence_begins_list = attn_metadata.subsequence_begins.tolist()
        block_indices_begins_list = attn_metadata.block_indices_begins.tolist()
        block_indices_list = attn_metadata.block_indices.tolist()
        slot_mapping = attn_metadata.slot_mapping

        for req_idx, req_id in enumerate(req_ids):
            past_len = int(past_lens_list[req_idx])
            query_start = int(subsequence_begins_list[req_idx])
            query_end = int(subsequence_begins_list[req_idx + 1])
            query_len = query_end - query_start
            if query_len <= 0:
                continue
            is_verify = req_id in verify_req_ids
            execution_phase = (
                "verify"
                if is_verify
                else ("prefill" if past_len == 0 or query_len > 1 else "decode")
            )
            _validate_merak_narrow_query_length(execution_phase, query_len)

            block_begin = int(block_indices_begins_list[req_idx])
            block_end = int(block_indices_begins_list[req_idx + 1])
            block_ids = tuple(
                int(block_id) for block_id in block_indices_list[block_begin:block_end]
            )
            trace_prefill_request = determinism_trace_enabled() and past_len == 0

            # --- restore / clear linear-attention state (spec §6) ---
            curr_phys_by_group: dict[int, int] = {}
            curr_logical_idx = -1
            if manage_linear_state:
                total_len = past_len + query_len
                num_blocks = (
                    total_len + mamba_block_size - 1
                ) // mamba_block_size + mamba_num_spec_blocks
                curr_logical_idx = max(0, num_blocks - 1 - mamba_num_spec_blocks)
                prev_logical_idx = self.merak_linear_state_idx.get(req_id)
                if prev_logical_idx is None and past_len > 0:
                    # Match vLLM's Mamba preprocessing for new/resumed requests:
                    # when vLLM already has computed tokens but the Merak runner
                    # has no local running-state record (for example after resume
                    # or prefix-cache reuse), the previous state lives at the
                    # logical block containing the last computed token.  Clearing
                    # here drops conversation history for hybrid linear layers.
                    prev_logical_idx = (past_len - 1) // mamba_block_size
                do_clear = past_len == 0

                # Translate each distinct Mamba group exactly once. Multiple
                # Linear layers may share a group and therefore one scheduler
                # physical identity.
                prev_phys_by_group: dict[int, int] = {}
                for gid in sorted(mamba_group_tables):
                    table = mamba_group_tables[gid]
                    width = int(table.shape[1])
                    c_log = self._normalize_mamba_logical_index(
                        curr_logical_idx,
                        width=width,
                        group_id=gid,
                    )
                    curr_phys_by_group[gid] = int(table[req_idx, c_log])
                    if not do_clear:
                        if prev_logical_idx is None:
                            raise RuntimeError(
                                f"past_len={past_len} request {req_id} has no previous "
                                "Merak Linear checkpoint index; fail closed."
                            )
                        p_log = self._normalize_mamba_logical_index(
                            prev_logical_idx,
                            width=width,
                            group_id=gid,
                        )
                        prev_phys_by_group[gid] = int(table[req_idx, p_log])

                checkpoint_store = self.merak_linear_checkpoint_store
                assert checkpoint_store is not None
                if do_clear:
                    checkpoint_store.acquire(
                        req_id,
                        curr_phys_by_group,
                        past_len=0,
                    )
                    if trace_prefill_request:
                        emit_determinism_event(
                            "linear_cache_before_clear",
                            request_id=req_id,
                            payload={
                                "runtime_owner": getattr(
                                    self,
                                    "_merak_runtime_linear_owner_req_id",
                                    None,
                                ),
                                "cache": _runtime_linear_cache_fingerprint(merak_model),
                            },
                        )
                    merak_model._clear_runtime_linear_caches()
                    self._merak_runtime_linear_owner_req_id = req_id
                    if trace_prefill_request:
                        emit_determinism_event(
                            "linear_cache_after_clear",
                            request_id=req_id,
                            payload={
                                "runtime_owner": (
                                    self._merak_runtime_linear_owner_req_id
                                ),
                                "cache": _runtime_linear_cache_fingerprint(merak_model),
                            },
                        )
                else:
                    # Runtime linear state is a one-entry execution cache in
                    # front of the persistent checkpoint store.  Consecutive
                    # decode/verify calls for the same request already have
                    # the exact committed state resident, so avoid a complete
                    # store -> runtime copy.  A request switch still restores
                    # from the authoritative store and changes ownership.
                    if (
                        getattr(self, "_merak_runtime_linear_owner_req_id", None)
                        != req_id
                    ):
                        loaded_slot = checkpoint_store.load_slot(
                            req_id,
                            prev_phys_by_group,
                            past_len=past_len,
                        )
                        merak_model._load_runtime_linear_state_from_store(loaded_slot)
                        self._merak_runtime_linear_owner_req_id = req_id
                    if determinism_trace_enabled() and is_verify:
                        emit_determinism_event(
                            "spec_target_state_before_verify",
                            request_id=req_id,
                            payload={
                                "past_len": past_len,
                                "query_len": query_len,
                                "block_ids": list(block_ids),
                                "slot_mapping": determinism_fingerprint(
                                    slot_mapping[query_start:query_end]
                                ),
                                "slot_mapping_values": [
                                    int(value)
                                    for value in slot_mapping[query_start:query_end]
                                    .detach()
                                    .to(device="cpu")
                                    .reshape(-1)
                                    .tolist()
                                ],
                                "linear_cache": (
                                    _runtime_linear_cache_fingerprint(merak_model)
                                ),
                                "page_cache": _page_cache_block_fingerprint(
                                    merak_model,
                                    block_ids,
                                ),
                            },
                        )

            request_input_ids = (
                flat_input_ids[query_start:query_end]
                if flat_input_ids is not None
                else None
            )
            request_inputs_embeds = (
                flat_inputs_embeds[query_start:query_end]
                if flat_inputs_embeds is not None
                else None
            )
            request_positions = (
                positions[..., query_start:query_end] if positions is not None else None
            )

            request_mm_kwargs, image_grid_offset, video_grid_offset = (
                merak_model._extract_request_multimodal_kwargs(
                    input_ids=request_input_ids,
                    inputs_embeds=request_inputs_embeds,
                    image_grid_thw=cast(
                        torch.Tensor | None, model_kwargs.get("image_grid_thw")
                    ),
                    image_grid_offset=image_grid_offset,
                    video_grid_thw=cast(
                        torch.Tensor | None, model_kwargs.get("video_grid_thw")
                    ),
                    video_grid_offset=video_grid_offset,
                )
            )

            if trace_prefill_request:
                emit_determinism_event(
                    "scheduler_prefill_request",
                    request_id=req_id,
                    payload={
                        "req_idx": req_idx,
                        "past_len": past_len,
                        "query_start": query_start,
                        "query_end": query_end,
                        "query_len": query_len,
                        "block_ids": list(block_ids),
                        "input_ids": determinism_fingerprint(request_input_ids),
                        "inputs_embeds": determinism_fingerprint(request_inputs_embeds),
                        "positions": determinism_fingerprint(request_positions),
                        "slot_mapping": determinism_fingerprint(
                            slot_mapping[query_start:query_end]
                        ),
                        "multimodal_kwargs": determinism_fingerprint(request_mm_kwargs),
                    },
                )
                emit_determinism_event(
                    "page_cache_before_prefill",
                    request_id=req_id,
                    payload={
                        "blocks": _page_cache_block_fingerprint(merak_model, block_ids)
                    },
                )

            page_attention_group_contexts: (
                dict[int, MerakPageAttentionGroupContext] | None
            ) = None
            if uses_gemma4_group_contexts and enable_page_attention:
                request_metadata_by_group: dict[int, MerakAttentionMetadata] = {}
                for group_idx, group_metadata in metadata_by_group.items():
                    sliced_metadata, group_query_length = (
                        self._slice_gemma4_group_metadata(group_metadata, req_idx)
                    )
                    if group_query_length != query_len:
                        raise RuntimeError(
                            "Gemma4 query length differs across cache groups: "
                            f"cache_group_idx={group_idx}, "
                            f"expected={query_len}, got={group_query_length}"
                        )
                    request_metadata_by_group[group_idx] = sliced_metadata
                visual_ranges = merak_model._page_attention_visual_ranges(
                    request_input_ids,
                    past_len=past_len,
                )
                page_attention_group_contexts = (
                    merak_model._build_page_attention_group_contexts(
                        metadata_by_group=request_metadata_by_group,
                        query_lengths=torch.tensor(
                            [query_len], dtype=torch.int64, device=device
                        ),
                        visual_ranges=visual_ranges,
                    )
                )

            hybrid_transaction_started = False
            hybrid_passthrough_started = False
            if is_verify and uses_qwen_spec_hybrid:
                self._begin_merak_hybrid_verify_transaction(merak_model.inference_model)
                hybrid_transaction_started = True
            elif uses_qwen_spec_hybrid:
                self._begin_merak_hybrid_output_passthrough(merak_model.inference_model)
                hybrid_passthrough_started = True
            try:
                request_result = merak_model._run_request(
                    past_len,
                    input_ids=request_input_ids,
                    inputs_embeds=request_inputs_embeds,
                    positions=request_positions,
                    page_attention_block_ids=(
                        block_ids
                        if enable_page_attention and not uses_gemma4_group_contexts
                        else None
                    ),
                    page_attention_slot_mapping=(
                        slot_mapping[query_start:query_end]
                        if enable_page_attention and not uses_gemma4_group_contexts
                        else None
                    ),
                    page_attention_group_contexts=page_attention_group_contexts,
                    request_id=req_id,
                    execution_phase=execution_phase,
                    return_result=spec_proposer is not None,
                    target_hidden_output_name=hidden_output_name,
                    **request_mm_kwargs,
                )
                if determinism_trace_enabled() and is_verify:
                    emit_determinism_event(
                        "spec_target_state_after_verify",
                        request_id=req_id,
                        payload={
                            "past_len": past_len,
                            "query_len": query_len,
                            "block_ids": list(block_ids),
                            "page_cache": _page_cache_block_fingerprint(
                                merak_model,
                                block_ids,
                            ),
                        },
                    )
                detached_hybrid_transaction = (
                    self._detach_merak_hybrid_verify_transaction(
                        merak_model.inference_model
                    )
                    if hybrid_transaction_started
                    else None
                )
            except Exception:
                if hybrid_transaction_started:
                    self._abort_merak_hybrid_verify_transaction(
                        merak_model.inference_model
                    )
                raise
            finally:
                if hybrid_passthrough_started:
                    self._end_merak_hybrid_output_passthrough(
                        merak_model.inference_model
                    )
            if spec_proposer is None:
                request_logits = request_result
            else:
                request_logits = request_result.logits
                if (
                    is_verify
                    and getattr(
                        merak_model,
                        "_page_attention_verify_transaction_request_id",
                        None,
                    )
                    == req_id
                ):
                    if req_id in self._merak_pending_page_verifies:
                        raise RuntimeError(
                            "Merak request already has an uncommitted "
                            f"PageAttention verify: {req_id}"
                        )
                    self._merak_pending_page_verifies[req_id] = query_len
                spec_proposer.on_target_forward(
                    req_id=req_id,
                    phase=execution_phase,
                    past_len=past_len,
                    input_ids=request_input_ids,
                    input_embeddings=request_inputs_embeds,
                    result=request_result,
                    page_attention_group_contexts=(page_attention_group_contexts),
                )
                if is_verify:
                    self._merak_pending_verify_req_ids.add(req_id)
            if trace_prefill_request:
                emit_determinism_event(
                    "page_cache_after_prefill",
                    request_id=req_id,
                    payload={
                        "blocks": _page_cache_block_fingerprint(merak_model, block_ids)
                    },
                )
                emit_determinism_event(
                    "linear_cache_after_prefill",
                    request_id=req_id,
                    payload={
                        "runtime_owner": self._merak_runtime_linear_owner_req_id,
                        "cache": _runtime_linear_cache_fingerprint(merak_model),
                    },
                )
            # --- persist updated linear-attention state + record state idx ---
            if manage_linear_state:
                if is_verify:
                    group_rows = {
                        gid: table[req_idx].copy()
                        for gid, table in mamba_group_tables.items()
                    }
                    self._merak_pending_linear_verifies[req_id] = {
                        "past_len": past_len,
                        "query_len": query_len,
                        "previous_physical_ids": prev_phys_by_group,
                        "group_rows": group_rows,
                        "block_size": mamba_block_size,
                        "num_speculative_blocks": mamba_num_spec_blocks,
                        "transaction": detached_hybrid_transaction,
                    }
                else:
                    checkpoint_store = self.merak_linear_checkpoint_store
                    assert checkpoint_store is not None
                    merak_model._scatter_runtime_linear_state_to_store(
                        checkpoint_store,
                        req_id,
                    )
                    checkpoint_store.transition(req_id, curr_phys_by_group)
                    self.merak_linear_state_idx[req_id] = curr_logical_idx

            if is_verify:
                if request_logits.shape[0] < query_len:
                    raise RuntimeError(
                        "Merak speculative target returned too few verify "
                        f"logits for request {req_id}: "
                        f"{request_logits.shape[0]} < {query_len}"
                    )
                flat_logits[query_start:query_end].copy_(
                    request_logits[:query_len].to(flat_logits.dtype)
                )
            else:
                last_request_logits = (
                    request_logits if spec_proposer is None else request_logits[-1]
                )
                flat_logits[query_end - 1].copy_(
                    last_request_logits.to(flat_logits.dtype)
                )

        return flat_logits

    def _update_states_after_model_execute(
        self,
        output_token_ids: torch.Tensor,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        super()._update_states_after_model_execute(
            output_token_ids,
            scheduler_output,
        )
        pending_verify_ids = getattr(
            self,
            "_merak_pending_verify_req_ids",
            set(),
        )
        if not pending_verify_ids:
            return

        req_ids = list(self.input_batch.req_ids)
        proposer = getattr(self, "_merak_spec_proposer", None)
        for req_idx, req_id in enumerate(req_ids[: output_token_ids.shape[0]]):
            if req_id not in pending_verify_ids:
                continue
            sampled_row = output_token_ids[req_idx]
            accepted_steps = int((sampled_row != -1).sum().item())
            if accepted_steps <= 0:
                raise RuntimeError(
                    "Merak speculative verification produced no valid target "
                    f"token for request {req_id}"
                )
            self._commit_pending_merak_page_verify(
                req_id,
                accepted_steps=accepted_steps,
            )
            self._commit_pending_merak_linear_verify(
                req_id,
                accepted_steps=accepted_steps,
            )
            if proposer is not None:
                proposer.on_target_accept(
                    req_id=req_id,
                    accepted_steps=accepted_steps,
                    sampled_token_ids=sampled_row,
                )
            pending_verify_ids.remove(req_id)

    def _copy_draft_token_ids_to_cpu(
        self,
        scheduler_output: "SchedulerOutput",
        zeros_only: bool = False,
    ) -> None:
        """Publish an empty custom proposal when drafting no longer fits.

        Upstream uses a zero-filled tensor as the ``zeros_only`` fast path for
        padded GPU proposers.  A ``custom_class`` proposer, however, publishes
        Python token lists and the scheduler interprets every zero in that
        tensor as a real token id.  Near ``max_model_len`` this made vLLM
        schedule a bogus verify block even though the HMONNX controller had
        intentionally skipped drafting.

        Preserve the upstream path for built-in proposers, but represent "no
        proposal" with the custom-class ABI's native empty lists.
        """

        if zeros_only and getattr(self, "_merak_spec_proposer", None) is not None:
            req_ids = list(self.input_batch.req_ids)
            self._draft_token_ids = [[] for _ in req_ids]
            self._draft_token_req_ids = req_ids
            return
        super()._copy_draft_token_ids_to_cpu(
            scheduler_output,
            zeros_only=zeros_only,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        self._merak_verify_req_ids = set(scheduler_output.scheduled_spec_decode_tokens)
        try:
            return super().execute_model(scheduler_output, intermediate_tensors)
        except Exception:
            self._abort_all_merak_hybrid_verify_transactions()
            raise
        finally:
            self._merak_verify_req_ids.clear()
