"""Concrete HMONNX draft execution with shared weights and in-place caches."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from vllm.logger import init_logger

from ..constants import (
    MERAK_SPEC_DRAFT_CONTEXT_CUDA_GRAPH,
    MERAK_SPEC_DRAFT_CUDA_GRAPH,
)
from ..determinism_trace import (
    emit_event as emit_determinism_event,
)
from ..determinism_trace import fingerprint as determinism_fingerprint
from ..determinism_trace import trace_enabled as determinism_trace_enabled
from ..utils import env_flag_enabled
from .artifact import HMONNXSpecDecodeArtifact


logger = init_logger(__name__)

_CACHE_INPUT_PREFIXES = (
    "past_key_cache",
    "past_value_cache",
    "past_conv_cache_",
    "past_recurrent_state_",
)


def _is_cache_input(name: str) -> bool:
    return (
        name.startswith(_CACHE_INPUT_PREFIXES)
        or "kcache_input" in name
        or "vcache_input" in name
    )


def _as_3d(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[0] != 1:
        raise RuntimeError(
            f"{name} must have shape [Q, D] or [1, Q, D], got {tuple(value.shape)}"
        )
    return value


def _gemma_readonly_sliding_cache_range(
    target_visible_length: int,
    sliding_window: int,
) -> tuple[int, int]:
    """Return compact target-KV ``(absolute_start, valid_width)`` metadata."""

    target_visible_length = int(target_visible_length)
    sliding_window = int(sliding_window)
    if target_visible_length <= 0:
        raise ValueError("target_visible_length must be positive")
    if sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    start = max(0, target_visible_length - sliding_window)
    return start, target_visible_length - start


def _dflash2_attention_mask(
    shape: Sequence[int],
    *,
    past_length: int,
    current_length: int,
    sliding_window: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build DFlash2's symmetric, non-causal sliding-window mask."""

    mask_shape = tuple(int(dim) for dim in shape)
    if len(mask_shape) != 3:
        raise RuntimeError(
            f"DFlash2 attention mask must have shape [B, Q, K], got {mask_shape}"
        )
    if sliding_window <= 0:
        raise RuntimeError("DFlash2 sliding_window must be positive")
    valid_length = min(mask_shape[-1], int(past_length) + int(current_length))
    key_positions = torch.arange(mask_shape[-1], device=device).reshape(1, 1, -1)
    query_positions = int(past_length) + torch.arange(
        mask_shape[1], device=device
    ).reshape(1, -1, 1)
    visible = (
        (key_positions < valid_length)
        & (query_positions - key_positions < sliding_window)
        & (key_positions - query_positions < sliding_window)
    )
    mask = torch.full(mask_shape, -10000.0, dtype=dtype, device=device)
    return mask.masked_fill(visible, 0)


def _dflash2_query_kv_ranges(
    shape: Sequence[int],
    *,
    past_length: int,
    current_length: int,
    sliding_window: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build absolute [start, end) ranges for DFlash2 FlashAttention."""

    range_shape = tuple(int(dim) for dim in shape)
    if len(range_shape) != 3 or range_shape[-1] != 2:
        raise RuntimeError(
            f"DFlash2 query_kv_range_abs must have shape [B, Q, 2], got {range_shape}"
        )
    if sliding_window <= 0:
        raise RuntimeError("DFlash2 sliding_window must be positive")
    valid_length = int(past_length) + int(current_length)
    query_positions = int(past_length) + torch.arange(
        range_shape[1], dtype=dtype, device=device
    )
    starts = torch.clamp(query_positions - sliding_window + 1, min=0)
    ends = torch.clamp(query_positions + sliding_window, max=valid_length)
    ranges = torch.stack((starts, ends), dim=-1).unsqueeze(0)
    return ranges.expand(range_shape[0], -1, -1).contiguous()


def _dflash2_greedy_selector_walk(
    candidate_ids: torch.Tensor,
    first_scores: torch.Tensor,
    transition_scores: torch.Tensor,
    *,
    num_draft_tokens: int,
) -> torch.Tensor:
    """Walk the precomputed Scheme-B selector without re-running the head."""

    if candidate_ids.ndim != 3:
        raise RuntimeError(
            "DFlash2 candidate_ids must have shape [B, L, K], "
            f"got {tuple(candidate_ids.shape)}"
        )
    if first_scores.ndim != 2:
        raise RuntimeError(
            "DFlash2 selector_first_scores must have shape [B, K], "
            f"got {tuple(first_scores.shape)}"
        )
    if transition_scores.ndim != 4:
        raise RuntimeError(
            "DFlash2 selector_transition_scores must have shape [B, L-1, K, K], "
            f"got {tuple(transition_scores.shape)}"
        )
    batch_size, candidate_steps, top_k = candidate_ids.shape
    if not 1 <= num_draft_tokens <= candidate_steps:
        raise RuntimeError(
            "DFlash2 candidates are shorter than the requested draft block: "
            f"candidates={candidate_steps}, requested={num_draft_tokens}"
        )
    if tuple(first_scores.shape) != (batch_size, top_k):
        raise RuntimeError("DFlash2 first-score shape does not match candidates")
    expected_transition_prefix = (batch_size, candidate_steps - 1, top_k, top_k)
    if tuple(transition_scores.shape) != expected_transition_prefix:
        raise RuntimeError(
            "DFlash2 transition-score shape does not match candidates: "
            f"expected={expected_transition_prefix}, "
            f"got={tuple(transition_scores.shape)}"
        )

    selected_index = first_scores.float().argmax(dim=-1)
    selected_tokens = [candidate_ids[:, 0].gather(-1, selected_index.unsqueeze(-1))]
    for step in range(1, num_draft_tokens):
        score_row = transition_scores[:, step - 1].gather(
            1,
            selected_index.reshape(batch_size, 1, 1).expand(-1, 1, top_k),
        )[:, 0]
        selected_index = score_row.float().argmax(dim=-1)
        selected_tokens.append(
            candidate_ids[:, step].gather(-1, selected_index.unsqueeze(-1))
        )
    return torch.cat(selected_tokens, dim=-1)


def _dflash2_sample_selector_walk(
    candidate_ids: torch.Tensor,
    first_scores: torch.Tensor,
    transition_scores: torch.Tensor,
    *,
    num_draft_tokens: int,
    temperature: float,
    vocab_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample Scheme B and materialize its sparse proposal probabilities."""

    if temperature <= 0:
        raise ValueError("DFlash2 sampling temperature must be positive")
    if vocab_size <= 0:
        raise ValueError("DFlash2 target vocabulary size must be positive")
    # Reuse the greedy walk's strict shape validation before sampling. Its
    # result is intentionally discarded.
    _dflash2_greedy_selector_walk(
        candidate_ids,
        first_scores,
        transition_scores,
        num_draft_tokens=num_draft_tokens,
    )
    batch_size, _, top_k = candidate_ids.shape
    if torch.any(candidate_ids < 0) or torch.any(candidate_ids >= vocab_size):
        raise RuntimeError("DFlash2 candidate id is outside the target vocabulary")

    selected_indices: list[torch.Tensor] = []
    selected_tokens: list[torch.Tensor] = []
    dense_probs = torch.zeros(
        (batch_size, num_draft_tokens, vocab_size),
        dtype=torch.float32,
        device=candidate_ids.device,
    )
    previous_index: torch.Tensor | None = None
    for step in range(num_draft_tokens):
        if step == 0:
            score_row = first_scores
        else:
            assert previous_index is not None
            score_row = transition_scores[:, step - 1].gather(
                1,
                previous_index.reshape(batch_size, 1, 1).expand(-1, 1, top_k),
            )[:, 0]
        probabilities = torch.softmax(score_row.float() / float(temperature), dim=-1)
        exponential_noise = torch.empty_like(probabilities)
        exponential_noise.exponential_(generator=generator)
        selected_index = (probabilities / exponential_noise).argmax(dim=-1)
        step_candidates = candidate_ids[:, step]
        selected_indices.append(selected_index)
        selected_tokens.append(step_candidates.gather(-1, selected_index.unsqueeze(-1)))
        dense_probs[:, step].scatter_(
            -1,
            step_candidates.to(dtype=torch.long),
            probabilities,
        )
        previous_index = selected_index
    del selected_indices
    return torch.cat(selected_tokens, dim=-1), dense_probs


def _pad_sequence(
    value: torch.Tensor,
    *,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    value = _as_3d(value, name="HMONNX sequence input").to(
        device=device,
        dtype=dtype,
    )
    if value.shape[1] > target_length:
        raise RuntimeError(
            f"HMONNX sequence input length {value.shape[1]} exceeds "
            f"graph capacity {target_length}"
        )
    if value.shape[1] == target_length:
        return value.contiguous()
    padded = torch.zeros(
        (1, target_length, value.shape[2]),
        dtype=dtype,
        device=device,
    )
    padded[:, : value.shape[1]].copy_(value)
    return padded


def _output_map(session: Any, raw_outputs: Any) -> dict[str, torch.Tensor]:
    outputs = (
        tuple(raw_outputs) if isinstance(raw_outputs, (tuple, list)) else (raw_outputs,)
    )
    names = tuple(session.get_output_names())
    if len(names) != len(outputs):
        raise RuntimeError(
            f"HMONNX output arity mismatch: names={len(names)}, tensors={len(outputs)}"
        )
    result = dict(zip(names, outputs, strict=True))
    if any(not isinstance(value, torch.Tensor) for value in result.values()):
        raise TypeError("HMONNX draft returned a non-tensor output")
    return result


@dataclass
class _PrivateCacheState:
    values: dict[str, torch.Tensor]
    storage_ptrs: dict[str, int]
    owner_req_id: str | None = None


@dataclass
class _DFlashPageCacheState:
    caches: tuple[Any, ...]
    modules: tuple[Any, ...]
    writer_modules: dict[str, tuple[Any, ...]]
    contexts: tuple[Any, ...]
    block_ids: torch.Tensor
    committed_block_ids: torch.Tensor
    slot_mapping: torch.Tensor
    slot_offsets: torch.Tensor
    scratch_block_ids: tuple[int, int]
    block_size: int
    max_sequence_length: int
    storage_ptrs: tuple[int, ...]


@dataclass
class _MTPPageCacheState:
    cache: Any
    snapshot_cache: Any
    snapshot_blocks_per_step: int
    modules: dict[str, tuple[Any, ...]]
    contexts: dict[str, tuple[Any, ...]]
    block_ids: torch.Tensor
    slot_mappings: dict[str, torch.Tensor]
    block_size: int
    max_sequence_length: int
    storage_ptrs: tuple[int, ...]
    transaction: "_MTPPageCacheTransaction | None" = None


@dataclass
class _MTPPageCacheTransaction:
    req_id: str
    past_length: int
    num_draft_tokens: int
    physical_block_ids: tuple[int, ...]


@dataclass
class _GemmaReadOnlyPageBinding:
    """Stable module-side inputs for one target attention-type owner.

    Gemma4's assistant never appends K/V.  The only mutable state needed by
    PageAttentionReadOnly is therefore the target block table and the
    committed visible length.  Keeping those tensors stable lets the M=1
    assistant graph be captured once without cloning or rebinding target KV.
    """

    context: Any
    block_ids: torch.Tensor
    seq_lens: torch.Tensor
    cache: Any
    group_idx: int
    sliding_window: int | None = None


class HMONNXDraftRuntimeBackend:
    """Run Qwen MTP/DFlash or Gemma4 MTP draft graphs.

    All graph weights are loaded through ``MultiHMONNXLoader``.  Identical
    initializers therefore remain one ``SharedTensor`` across every draft
    session, and W4 packing happens once on CPU before CUDA placement.
    """

    def __init__(
        self,
        *,
        runner: Any,
        artifact: HMONNXSpecDecodeArtifact,
    ) -> None:
        self.runner = runner
        self.artifact = artifact
        self.target_model = runner.model
        self.device = self._select_draft_device()
        self._models: dict[str, Any] = {}
        self._sessions: dict[str, Any] = {}
        self._private_cache: _PrivateCacheState | None = None
        self._dflash_page_cache: _DFlashPageCacheState | None = None
        self._dflash_page_cache_spec: tuple[int, int, int] | None = None
        self._mtp_page_cache: _MTPPageCacheState | None = None
        self._mtp_page_cache_spec: tuple[int, int, int] | None = None
        self._gemma_owner_bindings: dict[str, Any] | None = None
        self._gemma_page_bindings: dict[str, _GemmaReadOnlyPageBinding] | None = None
        self._last_dflash_draft_probs: torch.Tensor | None = None
        # Qwen owns persistent private cache. Gemma4 instead owns stable
        # read-only PageAttention metadata buffers which are populated before
        # each proposal and point at the target's persistent cache owner.
        # Both therefore have stable addresses suitable for CUDA Graph replay.
        self._enable_private_decode_cuda_graph = env_flag_enabled(
            MERAK_SPEC_DRAFT_CUDA_GRAPH
        )
        self._enable_private_context_decode_cuda_graph = (
            not artifact.is_gemma4
            and artifact.mode == "dflash"
            and env_flag_enabled(MERAK_SPEC_DRAFT_CONTEXT_CUDA_GRAPH)
        )
        self.target_borrowed_output_count = (
            self._enable_target_transactional_output_borrowing()
        )
        logger.warning(
            "Target transactional CUDA Graph output lease: mode=%s, "
            "borrowed_outputs=%d",
            artifact.mode,
            self.target_borrowed_output_count,
        )
        self._load_graphs()
        self._initialize_private_cache()
        # Current vLLM explicitly warms the speculator prefill path before
        # serving.  Page-cache write kernels specialize on the prompt tile
        # width, so decode-only capture otherwise leaves a multi-second Triton
        # compile on the first real request.
        if self._uses_private_dflash_page_cache():
            self._warm_dflash_page_context()
        elif self._uses_private_mtp_page_cache():
            self._warm_mtp_page_prefill()
        if (
            self._enable_private_context_decode_cuda_graph
            and "context_decode" in self._sessions
        ):
            self._prime_private_context_decode_cuda_graph()
        # Qwen capture warm-up mutates private cache and must be primed/reset
        # before a request claims it. Gemma4 is read-only and has no block
        # table until the first target forward, so its capture is deliberately
        # lazy after the stable read-only binding has been staged.
        if self._enable_private_decode_cuda_graph and not artifact.is_gemma4:
            self._prime_private_decode_cuda_graph()

    def _enable_target_transactional_output_borrowing(self) -> int:
        """Borrow target Linear-state snapshots until accept/reject commits.

        Qwen verify graphs expose ``4 * layers * verify_width`` state
        snapshots.  Cloning all of them after every CUDA Graph replay is a
        full-state bandwidth tax even though rejection sampling selects only
        one step.  The Merak runner already serializes the target transaction
        and commits it before the next replay, so those snapshot outputs can
        be leased directly from the graph's stable output arena.
        """

        runtime = getattr(self.target_model, "inference_model", None)
        if runtime is None:
            return 0
        enabled = 0
        for graph_name in ("prefill_model", "decode_model"):
            graph_model = getattr(runtime, graph_name, None)
            session = getattr(graph_model, "hmonnx_session", None)
            if session is None:
                continue
            interpreter = getattr(session, "interpreter", None)
            if not callable(getattr(interpreter, "set_borrowed_output_indices", None)):
                # Eager sessions return ordinary ephemeral tensors and do not
                # perform the CUDA Graph replay clone this optimization avoids.
                continue
            output_names = tuple(session.get_output_names())
            transactional = [
                name
                for name in output_names
                if name.startswith("conv_cache_out_")
                or name.startswith("recurrent_state_out_")
            ]
            if not transactional:
                continue
            count = session.enable_borrowed_cuda_graph_outputs(transactional)
            if count != len(transactional):
                raise RuntimeError(
                    "Target CUDA Graph borrowed-output contract is incomplete: "
                    f"graph={graph_name}, requested={len(transactional)}, "
                    f"enabled={count}"
                )
            enabled += count
            logger.info(
                "Borrowing %d transactional Linear-state outputs from target %s "
                "CUDA Graph until acceptance",
                count,
                graph_name,
            )
        return enabled

    def _select_draft_device(self) -> torch.device:
        if not self.artifact.is_gemma4:
            return torch.device(self.runner.device)

        # The proposer attaches immediately after target model load, before
        # vLLM allocates/binds target KV cache.  At that point a valid target
        # graph still contains FlashAttention modules rather than the later
        # PageAttention replacements.  Device placement is a graph property,
        # so discover either physical form without forcing cache allocation or
        # falling back to an unrelated default device.
        attention_classes = frozenset(
            {
                "FlashAttention",
                "PageAttention",
                "PageAttentionReadOnly",
                "PageKVCache",
                "PageKVCacheReadOnly",
            }
        )
        candidate_maps = [
            self.runner._hmonnx_attention_node_map(
                runtime_model,
                torch.device(self.runner.device),
                module_class_names=attention_classes,
            )
            for runtime_model in self.runner._iter_hmonnx_models(
                self.target_model.inference_model
            )
        ]
        candidate_maps = [mapping for mapping in candidate_maps if mapping]
        node_map = candidate_maps[-1] if candidate_maps else []
        if not node_map:
            raise RuntimeError(
                "Gemma4 MTP could not discover target PageAttention devices"
            )
        # Official Gemma4 maps every assistant layer to the last non-shared
        # target layer of the same attention type. Those final local/global
        # layers are expected to be colocated for the serial HMONNX runtime.
        draft_types = self.artifact.draft_layer_attention_types
        target_bindings = tuple(
            getattr(self.target_model, "page_attention_bindings", ())
        )
        # A mixed Gemma4 graph has one physical PageKVCache node for each
        # sliding-cache owner, while its cache contract still contains every
        # logical alias layer.  Full-attention layers remain one-to-one.  Use
        # the contract's physical context bindings so graph-order device
        # indices cannot be inflated by logical sliding aliases.
        cache_contract = getattr(
            self.target_model,
            "gemma4_cache_contract",
            None,
        )
        physical_bindings = tuple(getattr(cache_contract, "context_bindings", ()))
        if not physical_bindings:
            physical_bindings = target_bindings
        if len(physical_bindings) != len(node_map):
            raise RuntimeError(
                "Gemma4 target physical attention map does not match its "
                "cache contract: "
                f"bindings={len(physical_bindings)}, nodes={len(node_map)}"
            )
        target_types = [
            str(getattr(binding, "attention_kind", "")) for binding in physical_bindings
        ]
        selected_devices: list[torch.device] = []
        for attention_kind in sorted(set(draft_types)):
            indices = [
                index
                for index, kind in enumerate(target_types)
                if kind == attention_kind
            ]
            if not indices:
                raise RuntimeError(
                    f"Gemma4 MTP has no target cache owner for {attention_kind!r}"
                )
            target_index = indices[-1]
            if target_index >= len(node_map):
                raise RuntimeError(
                    "Gemma4 target PageAttention device map is shorter than "
                    f"the cache contract: index={target_index}, "
                    f"nodes={len(node_map)}"
                )
            selected_devices.append(node_map[target_index][1])
        if len(set(selected_devices)) != 1:
            raise RuntimeError(
                "Gemma4 MTP target attention-type owners span devices "
                f"{sorted(map(str, set(selected_devices)))}; the current "
                "serial assistant graph requires them to be colocated"
            )
        return selected_devices[0]

    def _uses_private_dflash_page_cache(self) -> bool:
        artifact = getattr(self, "artifact", None)
        return bool(
            getattr(
                artifact,
                "uses_private_dflash_page_cache",
                False,
            )
        )

    def _uses_graph_dflash_page_cache_writes(self) -> bool:
        artifact = getattr(self, "artifact", None)
        return bool(
            getattr(
                artifact,
                "uses_graph_dflash_page_cache_writes",
                False,
            )
        )

    def _uses_private_mtp_page_cache(self) -> bool:
        artifact = getattr(self, "artifact", None)
        return bool(
            getattr(
                artifact,
                "uses_private_mtp_page_cache",
                False,
            )
        )

    def _graph_files(self) -> dict[str, str]:
        graphs = self.artifact.draft_graphs
        values = {
            "prefill": graphs.prefill,
            "context": graphs.context,
            "context_decode": graphs.context_decode,
            "decode": graphs.decode,
        }
        graph_files = {
            name: str(path) for name, path in values.items() if path is not None
        }
        if self._uses_private_dflash_page_cache():
            decode_path = values["decode"]
            if decode_path is None:
                raise RuntimeError("Paged DFlash artifact has no decode graph")
            self._dflash_page_cache_spec = self._read_draft_cache_spec(decode_path)
            graph_files["decode"] = str(
                self._convert_draft_to_page_attention(
                    decode_path,
                    insert_layer_tags=False,
                )
            )
            if self._uses_graph_dflash_page_cache_writes():
                for graph_name in ("context", "context_decode"):
                    graph_path = values[graph_name]
                    if graph_path is None:
                        raise RuntimeError(
                            "Shared paged DFlash requires both context graphs"
                        )
                    graph_files[graph_name] = str(
                        self._convert_draft_to_page_attention(
                            graph_path,
                            standalone_cache_write=True,
                            insert_layer_tags=False,
                        )
                    )
        elif self._uses_private_mtp_page_cache():
            decode_path = values["decode"]
            prefill_path = values["prefill"]
            if decode_path is None or prefill_path is None:
                raise RuntimeError(
                    "Paged MTP artifact requires prefill and decode graphs"
                )
            self._mtp_page_cache_spec = self._read_draft_cache_spec(decode_path)
            graph_files["prefill"] = str(
                self._convert_draft_to_page_attention(prefill_path)
            )
            graph_files["decode"] = str(
                self._convert_draft_to_page_attention(decode_path)
            )
        return graph_files

    @staticmethod
    def _read_draft_cache_spec(path: Path) -> tuple[int, int, int]:
        """Read [capacity, kv_heads, head_dim] without loading external data."""

        import onnx

        model = onnx.load(str(path), load_external_data=False)
        cache_inputs = [
            value
            for value in model.graph.input
            if value.name.startswith("past_key_cache")
        ]
        if not cache_inputs:
            raise RuntimeError(
                "Paged draft source has no key-cache input to infer "
                "its private cache capacity"
            )
        tensor_type = cache_inputs[0].type.tensor_type
        shape = [
            int(dim.dim_value) if dim.HasField("dim_value") else 0
            for dim in tensor_type.shape.dim
        ]
        if len(shape) != 4 or any(value <= 0 for value in shape):
            raise RuntimeError(
                "Paged draft key-cache input must have static "
                f"[B,H,S,D] shape, got {shape}"
            )
        for value in cache_inputs[1:]:
            other = [
                int(dim.dim_value) if dim.HasField("dim_value") else 0
                for dim in value.type.tensor_type.shape.dim
            ]
            if other != shape:
                raise RuntimeError(
                    "Paged draft key-cache inputs have inconsistent shapes: "
                    f"{shape} vs {other}"
                )
        return shape[2], shape[1], shape[3]

    @staticmethod
    def _convert_draft_to_page_attention(
        path: Path,
        *,
        standalone_cache_write: bool = False,
        insert_layer_tags: bool = True,
    ) -> Path:
        """Create a small weight-sharing derived graph beside the artifact.

        The ONNX initializers remain external-data references to the original
        file in the same directory. Conversion therefore neither duplicates
        the multi-GB draft weights nor raises peak host/GPU packing memory.
        """

        from xhquant.xhonnxruntime.convert_to_page_attention import (
            convert_to_page_attention,
        )

        suffix = (
            "merak_v3_page_cache_write"
            if standalone_cache_write
            else "merak_v3_page_attention"
        )
        output = path.with_name(f"{path.stem}.{suffix}.onnx")
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            converted = convert_to_page_attention(
                path,
                temporary,
                standalone_cache_write=standalone_cache_write,
                insert_layer_tags=insert_layer_tags,
                # DFlash page-cache graphs are single-device deploy graphs.
                # Remove exporter boundary Tags rather than allowing an
                # identity-only custom op to leak into context/decode.
                strip_layer_tags=not insert_layer_tags,
            )
            expected_op = (
                "PageKVCacheWrite" if standalone_cache_write else "PageAttention"
            )
            converted_count = sum(
                node.op_type == expected_op for node in converted.graph.node
            )
            if converted_count <= 0:
                raise RuntimeError(f"Draft conversion produced no {expected_op} layers")
            forbidden_ops = {"KVcache", "FlashAttention"}
            if standalone_cache_write:
                forbidden_ops.add("PageAttention")
            residual_ops = sorted(
                {
                    node.op_type
                    for node in converted.graph.node
                    if node.op_type in forbidden_ops
                }
            )
            if residual_ops:
                raise RuntimeError(
                    f"Draft conversion left unfused cache/attention ops: {residual_ops}"
                )
            if not insert_layer_tags and any(
                node.op_type == "Tag" for node in converted.graph.node
            ):
                raise RuntimeError("DFlash derived graph unexpectedly contains Tag")
            if standalone_cache_write:
                dense_cache_io = [
                    value.name
                    for value in (
                        *converted.graph.input,
                        *converted.graph.output,
                    )
                    if value.name.startswith(("past_key_cache_", "past_value_cache_"))
                    or value.name.startswith(
                        ("present_key_cache_", "present_value_cache_")
                    )
                ]
                if dense_cache_io:
                    raise RuntimeError(
                        "DFlash page writer retained dense cache graph I/O: "
                        f"{dense_cache_io}"
                    )
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return output

    def _load_graphs(self) -> None:
        from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import (
            HMONNXInferenceV2,
        )
        from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader

        if not HMONNXModel._is_inference_v2_env_enabled():
            raise RuntimeError(
                "HMONNX speculative decoding requires "
                "ENABLE_HMINFERENCE_V2=1 before the target model is loaded"
            )
        graph_files = self._graph_files()
        loader = MultiHMONNXLoader(graph_files)
        self.initializer_summary = loader.initializer_pool.summary()
        for graph_name, graph_path in graph_files.items():
            model = HMONNXModel(
                graph_path,
                onnx_graph=loader.graphs[graph_name],
                enable_cuda_graph=(
                    (self._enable_private_decode_cuda_graph and graph_name == "decode")
                    or (
                        self._enable_private_context_decode_cuda_graph
                        and graph_name == "context_decode"
                    )
                ),
                enable_auto_offload=False,
                device_map=[self.device],
            )
            session = model.hmonnx_session
            if not isinstance(session, HMONNXInferenceV2):
                raise RuntimeError(
                    "HMONNX speculative decoding requires "
                    "ENABLE_HMINFERENCE_V2=1 so graph weights and in-place "
                    "cache storage are shared deterministically"
                )
            if (
                self.artifact.mode == "dflash"
                and not self._uses_private_dflash_page_cache()
                and graph_name
                in {
                    "context",
                    "context_decode",
                }
            ):
                session.enable_inplace_cache_output_reuse(session.get_output_names())
            elif (
                self.artifact.mode == "dflash"
                and not self._uses_private_dflash_page_cache()
                and graph_name == "decode"
            ):
                enabled_updates = session.enable_inplace_cache_updates()
                if enabled_updates <= 0:
                    raise RuntimeError(
                        "DFlash decode declares mutable private cache but its "
                        "HMONNX graph contains no in-place LLMCache updates"
                    )
            self._models[graph_name] = model
            self._sessions[graph_name] = session

        # Keep the loader alive: its mmap pool owns CPU pages until all
        # initializers have reached their final runtime placement.
        self._multi_graph_loader = loader

    def _cache_schema(self, session: Any) -> dict[str, tuple[Any, ...]]:
        return {
            name: (
                tuple(int(dim) for dim in session.get_input(name).shape),
                session.get_input(name).dtype,
            )
            for name in session.get_input_names()
            if _is_cache_input(name)
        }

    def _initialize_private_cache(self) -> None:
        if self.artifact.is_gemma4:
            return
        if self._uses_private_dflash_page_cache():
            self._initialize_dflash_page_cache()
            return
        if self._uses_private_mtp_page_cache():
            self._initialize_mtp_page_cache()
            return
        if self.artifact.mode == "dflash":
            source_name = "context"
        else:
            source_name = "prefill"
        source = self._sessions.get(source_name)
        if source is None:
            source = self._sessions["decode"]
        schema = self._cache_schema(source)
        if not schema:
            raise RuntimeError(
                f"{self.artifact.draft_abi} graph has no private cache inputs"
            )
        for graph_name, session in self._sessions.items():
            graph_schema = self._cache_schema(session)
            if graph_schema and graph_schema != schema:
                raise RuntimeError(
                    "Speculative draft cache schemas differ across shared "
                    f"graphs: source={source_name}, graph={graph_name}"
                )

        from xhquant.core import CacheTensor

        values = {
            name: CacheTensor(torch.zeros(shape, dtype=dtype, device=self.device))
            for name, (shape, dtype) in schema.items()
        }
        self._private_cache = _PrivateCacheState(
            values=values,
            storage_ptrs={
                name: int(value.data_ptr()) for name, value in values.items()
            },
        )

    def _initialize_dflash_page_cache(self) -> None:
        from xhquant.core.hmfp_kv_cache import allocate_hmfp_paged_kv_cache
        from xhquant.xhonnxruntime.parsers import PageAttentionContext

        spec = self._dflash_page_cache_spec
        if spec is None:
            raise RuntimeError("Paged DFlash cache shape was not inferred")
        export_capacity, expected_kv_heads, expected_head_dim = spec
        decode = self._sessions["decode"]
        noise_capacity = int(decode.get_input("noise_embedding").shape[1])
        # The private runtime cache follows the HMONNX cache contract exactly.
        # Do not silently shrink an exported 262144-token cache based on the
        # serving process' current max_model_len.
        max_sequence_length = export_capacity
        page_modules = self._page_attention_modules_in_graph_order(decode.graph_module)
        if not page_modules:
            raise RuntimeError("Paged DFlash decode graph has no PageAttention modules")
        for module in page_modules:
            if int(module.num_kv_heads) != expected_kv_heads:
                raise RuntimeError(
                    "Paged DFlash KV-head mismatch: "
                    f"graph={module.num_kv_heads}, cache={expected_kv_heads}"
                )
            if (int(module.k_bits), int(module.v_bits)) != (8, 8):
                raise RuntimeError(
                    "Private DFlash PageAttention currently requires "
                    f"k_bits=v_bits=8, got {module.k_bits}/{module.v_bits}"
                )
        writer_modules: dict[str, tuple[Any, ...]] = {}
        if self._uses_graph_dflash_page_cache_writes():
            for graph_name in ("context", "context_decode"):
                modules = self._page_cache_write_modules_in_graph_order(
                    self._sessions[graph_name].graph_module
                )
                if len(modules) != len(page_modules):
                    raise RuntimeError(
                        "DFlash page cache writer/attention layer mismatch: "
                        f"graph={graph_name}, writers={len(modules)}, "
                        f"attention={len(page_modules)}"
                    )
                for layer_idx, module in enumerate(modules):
                    if int(module.layer_idx) != layer_idx:
                        raise RuntimeError(
                            "DFlash page cache writer layer order mismatch: "
                            f"graph={graph_name}, expected={layer_idx}, "
                            f"actual={module.layer_idx}"
                        )
                    if (int(module.k_bits), int(module.v_bits)) != (8, 8):
                        raise RuntimeError(
                            "DFlash page cache writer requires k_bits=v_bits=8"
                        )
                writer_modules[graph_name] = modules

        # SEFP V-cache stores token groups of 64, so this physical block size
        # is independent of the scheduler's logical request block size.
        block_size = int(
            getattr(
                self.artifact,
                "draft_page_cache_block_size",
                64,
            )
            or 64
        )
        committed_num_blocks = (max_sequence_length + block_size - 1) // block_size
        # DFlash decode concatenates a transient anchor/noise K/V tail to the
        # committed target prefix.  Two private COW pages cover the worst case
        # where that short tail crosses a 64-token page boundary.  This avoids
        # letting rejected draft values perturb the shared SEFP V scale of a
        # committed page.
        scratch_block_ids = (
            committed_num_blocks,
            committed_num_blocks + 1,
        )
        num_blocks = committed_num_blocks + len(scratch_block_ids)
        caches = tuple(
            allocate_hmfp_paged_kv_cache(
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=expected_kv_heads,
                head_size=expected_head_dim,
                device=self.device,
            )
            for _ in page_modules
        )
        committed_block_ids = torch.arange(
            committed_num_blocks,
            dtype=torch.int64,
            device=self.device,
        )
        block_ids = committed_block_ids.clone()
        slot_capacity = noise_capacity
        for graph_name in writer_modules:
            slot_capacity = max(
                slot_capacity,
                int(self._sessions[graph_name].get_input("target_hidden").shape[1]),
            )
        slot_mapping = torch.full(
            (slot_capacity,),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        slot_offsets = torch.arange(
            slot_capacity,
            dtype=torch.int64,
            device=self.device,
        )
        contexts = tuple(
            PageAttentionContext(
                paged_kv_cache=cache,
                block_ids=block_ids,
                slot_mapping=slot_mapping,
                block_size=block_size,
            )
            for cache in caches
        )
        for module, context in zip(page_modules, contexts, strict=True):
            module.set_context(context)
        for modules in writer_modules.values():
            for module, context in zip(modules, contexts, strict=True):
                module.set_context(context)

        def pointers(cache: Any) -> tuple[int, ...]:
            return tuple(
                int(getattr(cache, name).data_ptr())
                for name in (
                    "k_qdata",
                    "k_scale_exp",
                    "v_qdata",
                    "v_scale_exp",
                )
            )

        self._dflash_page_cache = _DFlashPageCacheState(
            caches=caches,
            modules=page_modules,
            writer_modules=writer_modules,
            contexts=contexts,
            block_ids=block_ids,
            committed_block_ids=committed_block_ids,
            slot_mapping=slot_mapping,
            slot_offsets=slot_offsets,
            scratch_block_ids=scratch_block_ids,
            block_size=block_size,
            max_sequence_length=max_sequence_length,
            storage_ptrs=tuple(
                pointer for cache in caches for pointer in pointers(cache)
            ),
        )
        # Reuse the existing serial request-ownership guard. Paged graphs have
        # no tensor-valued cache inputs by construction.
        self._private_cache = _PrivateCacheState(
            values={},
            storage_ptrs={},
        )
        fp16_bytes = (
            len(caches)
            * 2
            * expected_kv_heads
            * max_sequence_length
            * expected_head_dim
            * 2
        )
        page_bytes = sum(
            tensor.numel() * tensor.element_size()
            for cache in caches
            for tensor in (
                cache.k_qdata,
                cache.k_scale_exp,
                cache.v_qdata,
                cache.v_scale_exp,
            )
        )
        logger.warning(
            "DFlash private cache uses %d PageAttention layers: %.2f GiB "
            "(legacy contiguous FP16 at export capacity would be %.2f GiB, "
            "export capacity=%d)",
            len(caches),
            page_bytes / 2**30,
            fp16_bytes / 2**30,
            export_capacity,
        )

    def _initialize_mtp_page_cache(self) -> None:
        from xhquant.core.hmfp_kv_cache import allocate_hmfp_paged_kv_cache
        from xhquant.xhonnxruntime.parsers import PageAttentionContext

        spec = self._mtp_page_cache_spec
        if spec is None:
            raise RuntimeError("Paged MTP cache shape was not inferred")
        export_capacity, expected_kv_heads, expected_head_dim = spec
        # Prefill and decode share the exact capacity declared by HMONNX.
        max_sequence_length = export_capacity
        block_size = int(
            getattr(
                self.artifact,
                "draft_page_cache_block_size",
                64,
            )
            or 64
        )
        num_blocks = (max_sequence_length + block_size - 1) // block_size
        owner = allocate_hmfp_paged_kv_cache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=expected_kv_heads,
            head_size=expected_head_dim,
            device=self.device,
        )
        # A K-token proposal can touch at most two 64-token pages for the
        # exported K<=15 contracts.  Keep one exact page snapshot for the
        # pre-draft state and for every autoregressive draft step.  Rejection
        # can then restore the accepted prefix without letting rejected rows
        # perturb the shared SEFP V exponent of a committed page.
        snapshot_blocks_per_step = 2
        snapshot_cache = allocate_hmfp_paged_kv_cache(
            num_blocks=(
                (int(self.artifact.num_draft_tokens) + 1) * snapshot_blocks_per_step
            ),
            block_size=block_size,
            num_kv_heads=expected_kv_heads,
            head_size=expected_head_dim,
            device=self.device,
        )
        block_ids = torch.arange(
            num_blocks,
            dtype=torch.int64,
            device=self.device,
        )
        modules: dict[str, tuple[Any, ...]] = {}
        contexts: dict[str, tuple[Any, ...]] = {}
        slot_mappings: dict[str, torch.Tensor] = {}
        for graph_name in ("prefill", "decode"):
            session = self._sessions[graph_name]
            graph_modules = self._page_attention_modules_in_graph_order(
                session.graph_module
            )
            if len(graph_modules) != 1:
                raise RuntimeError(
                    "Paged Qwen MTP expects one assistant attention layer per "
                    f"graph, got graph={graph_name}, layers={len(graph_modules)}"
                )
            module = graph_modules[0]
            if int(module.num_kv_heads) != expected_kv_heads:
                raise RuntimeError(
                    "Paged MTP KV-head mismatch: "
                    f"graph={module.num_kv_heads}, cache={expected_kv_heads}"
                )
            if (int(module.k_bits), int(module.v_bits)) != (8, 8):
                raise RuntimeError(
                    "Private MTP PageAttention currently requires "
                    f"k_bits=v_bits=8, got {module.k_bits}/{module.v_bits}"
                )
            capacity = int(session.get_input("post_norm_hidden").shape[1])
            slot_mapping = torch.arange(
                capacity,
                dtype=torch.int64,
                device=self.device,
            )
            context = PageAttentionContext(
                paged_kv_cache=owner,
                block_ids=block_ids,
                slot_mapping=slot_mapping,
                block_size=block_size,
            )
            module.set_context(context)
            modules[graph_name] = graph_modules
            contexts[graph_name] = (context,)
            slot_mappings[graph_name] = slot_mapping

        storage_ptrs = tuple(
            int(getattr(owner, name).data_ptr())
            for name in (
                "k_qdata",
                "k_scale_exp",
                "v_qdata",
                "v_scale_exp",
            )
        )
        self._mtp_page_cache = _MTPPageCacheState(
            cache=owner,
            snapshot_cache=snapshot_cache,
            snapshot_blocks_per_step=snapshot_blocks_per_step,
            modules=modules,
            contexts=contexts,
            block_ids=block_ids,
            slot_mappings=slot_mappings,
            block_size=block_size,
            max_sequence_length=max_sequence_length,
            storage_ptrs=storage_ptrs,
        )
        self._private_cache = _PrivateCacheState(
            values={},
            storage_ptrs={},
        )
        fp16_bytes = 2 * expected_kv_heads * max_sequence_length * expected_head_dim * 2
        page_bytes = sum(
            getattr(owner, name).numel() * getattr(owner, name).element_size()
            for name in (
                "k_qdata",
                "k_scale_exp",
                "v_qdata",
                "v_scale_exp",
            )
        )
        logger.warning(
            "MTP private cache uses shared prefill/decode PageAttention: "
            "%.2f GiB (legacy contiguous FP16 at export capacity would be "
            "%.2f GiB, export capacity=%d)",
            page_bytes / 2**30,
            fp16_bytes / 2**30,
            export_capacity,
        )

    @staticmethod
    def _page_attention_modules_in_graph_order(
        graph_module: Any,
    ) -> tuple[Any, ...]:
        modules: list[Any] = []
        for node in graph_module.graph.nodes:
            if node.op != "call_module":
                continue
            module = graph_module.get_submodule(str(node.target))
            if module.__class__.__name__ == "PageAttention":
                modules.append(module)
        return tuple(modules)

    @staticmethod
    def _page_cache_write_modules_in_graph_order(
        graph_module: Any,
    ) -> tuple[Any, ...]:
        modules: list[Any] = []
        for node in graph_module.graph.nodes:
            if node.op != "call_module":
                continue
            module = graph_module.get_submodule(str(node.target))
            if module.__class__.__name__ == "PageKVCacheWrite":
                modules.append(module)
        return tuple(modules)

    def _prime_private_cuda_graph(
        self,
        graph_name: str,
        *,
        current_length: int,
    ) -> None:
        """Capture a fixed-shape Qwen graph before a request owns the cache.

        HMONNX CUDA Graph capture performs warm-up executions. Running that
        lazily on the first real request would mutate private state multiple
        times. Prime against the final persistent CacheTensor addresses while
        they are still zero, then reset those same tensors in-place. No shadow
        cache or duplicate full-capacity KV allocation is created, and replay
        observes the cache populated by the eager context graph.
        """

        cache = self._private_cache
        if cache is None:
            raise RuntimeError(
                "Private draft CUDA Graph requires initialized cache inputs"
            )
        session = self._sessions[graph_name]
        feed = self._base_feed(
            session,
            cache.values,
            past_length=0,
            current_length=current_length,
        )
        for name in session.get_input_names():
            if name in feed:
                continue
            info = session.get_input(name)
            feed[name] = torch.zeros(
                tuple(int(dim) for dim in info.shape),
                dtype=info.dtype,
                device=self.device,
            )

        capture_error: Exception | None = None
        try:
            if self._uses_private_dflash_page_cache():
                self._stage_dflash_page_slots(
                    past_length=0,
                    current_length=current_length,
                    physical_length=current_length,
                    transient=graph_name == "decode",
                )
            self._run(graph_name, feed)
        except Exception as exc:  # pragma: no cover - hardware/runtime-specific.
            capture_error = exc
        finally:
            # Capture warm-ups are deliberately outside request semantics.
            # Restore the exact persistent objects rather than rebinding them.
            for value in cache.values.values():
                reset = getattr(value, "reset", None)
                if callable(reset):
                    reset()
                else:
                    value.zero_()
            self._reset_dflash_page_cache()
            self._reset_mtp_page_cache()
            cache.owner_req_id = None
            self._assert_private_cache_pointers()

        interpreter = getattr(session, "interpreter", None)
        if capture_error is not None:
            reason = (
                f"speculative draft {graph_name} CUDA Graph priming failed: "
                f"{type(capture_error).__name__}: {capture_error}"
            )
            if interpreter is not None:
                try:
                    interpreter.clear(clear_disabled_reason=False)
                    interpreter._capture_disabled_reason = reason
                except (AttributeError, TypeError):
                    pass
            logger.warning("%s; using eager %s", reason, graph_name)
            return

        if bool(getattr(interpreter, "has_captured_graph", False)):
            logger.info(
                "Captured fixed-shape %s draft %s CUDA Graph at M=%d",
                self.artifact.mode,
                graph_name,
                current_length,
            )
            return
        reason = getattr(interpreter, "capture_disabled_reason", None)
        logger.warning(
            "Draft %s CUDA Graph was requested but not captured%s; using eager %s",
            graph_name,
            f": {reason}" if reason else "",
            graph_name,
        )

    def _warm_dflash_page_context(self) -> None:
        """Warm projection plus M=256 page writes outside request semantics."""

        session = self._sessions["context"]
        hidden_info = session.get_input("target_hidden")
        capacity = int(hidden_info.shape[1])
        feed = self._base_feed(
            session,
            {},
            past_length=0,
            current_length=capacity,
        )
        feed["target_hidden"] = torch.zeros(
            tuple(int(dim) for dim in hidden_info.shape),
            dtype=hidden_info.dtype,
            device=self.device,
        )
        started = time.perf_counter()
        if self._uses_graph_dflash_page_cache_writes():
            self._stage_dflash_page_slots(
                past_length=0,
                current_length=capacity,
                physical_length=capacity,
            )
        outputs = self._run("context", feed)
        if not self._uses_graph_dflash_page_cache_writes():
            self._write_dflash_page_context(
                outputs,
                past_length=0,
                current_length=capacity,
            )
        self._reset_dflash_page_cache()
        cache = self._private_cache
        if cache is None:
            raise RuntimeError("DFlash page warmup lost private cache ownership")
        cache.owner_req_id = None
        self._assert_private_cache_pointers()
        logger.info(
            "Warmed DFlash projection and private page writes at M=%d in %.3fs",
            capacity,
            time.perf_counter() - started,
        )

    def _warm_mtp_page_prefill(self) -> None:
        """Warm MTP prompt kernels and page writes before the first request."""

        session = self._sessions["prefill"]
        hidden_info = session.get_input("post_norm_hidden")
        capacity = int(hidden_info.shape[1])
        feed = self._base_feed(
            session,
            {},
            past_length=0,
            current_length=capacity,
        )
        for name in session.get_input_names():
            if name in feed:
                continue
            info = session.get_input(name)
            feed[name] = torch.zeros(
                tuple(int(dim) for dim in info.shape),
                dtype=info.dtype,
                device=self.device,
            )
        self._stage_mtp_page_slots(
            "prefill",
            past_length=0,
            current_length=capacity,
        )
        started = time.perf_counter()
        self._run("prefill", feed)
        self._reset_mtp_page_cache()
        cache = self._private_cache
        if cache is None:
            raise RuntimeError("MTP page warmup lost private cache ownership")
        cache.owner_req_id = None
        self._assert_private_cache_pointers()
        logger.info(
            "Warmed MTP prefill and private page writes at M=%d in %.3fs",
            capacity,
            time.perf_counter() - started,
        )

    def _prime_private_context_decode_cuda_graph(self) -> None:
        session = self._sessions["context_decode"]
        current_length = int(session.get_input("target_hidden").shape[1])
        self._prime_private_cuda_graph(
            "context_decode",
            current_length=current_length,
        )

    def _prime_private_decode_cuda_graph(self) -> None:
        session = self._sessions["decode"]
        current_length = 1
        if self.artifact.mode == "dflash":
            current_length = int(session.get_input("noise_embedding").shape[1])
        self._prime_private_cuda_graph(
            "decode",
            current_length=current_length,
        )

    def _claim_private_cache(self, req_id: str) -> dict[str, torch.Tensor]:
        cache = self._private_cache
        if cache is None:
            raise RuntimeError(f"{self.artifact.draft_abi} has no private cache state")
        if cache.owner_req_id not in (None, req_id):
            raise RuntimeError(
                "Serial HMONNX draft cache is already owned by request "
                f"{cache.owner_req_id!r}; cannot serve {req_id!r}"
            )
        cache.owner_req_id = req_id
        self._assert_private_cache_pointers()
        return cache.values

    def _assert_private_cache_pointers(self) -> None:
        cache = self._private_cache
        if cache is None:
            return
        actual = {name: int(value.data_ptr()) for name, value in cache.values.items()}
        if actual != cache.storage_ptrs:
            raise RuntimeError(
                "HMONNX draft cache storage changed address; cache outputs "
                "must mutate the original inputs in place without copies"
            )
        page = getattr(self, "_dflash_page_cache", None)
        if page is not None:
            actual_page = tuple(
                int(getattr(owner, name).data_ptr())
                for owner in page.caches
                for name in (
                    "k_qdata",
                    "k_scale_exp",
                    "v_qdata",
                    "v_scale_exp",
                )
            )
            if actual_page != page.storage_ptrs:
                raise RuntimeError("DFlash PageAttention cache storage changed address")
        mtp_page = getattr(self, "_mtp_page_cache", None)
        if mtp_page is not None:
            actual_mtp = tuple(
                int(getattr(mtp_page.cache, name).data_ptr())
                for name in (
                    "k_qdata",
                    "k_scale_exp",
                    "v_qdata",
                    "v_scale_exp",
                )
            )
            if actual_mtp != mtp_page.storage_ptrs:
                raise RuntimeError("MTP PageAttention cache storage changed address")

    def _reset_dflash_page_cache(self) -> None:
        page = getattr(self, "_dflash_page_cache", None)
        if page is None:
            return
        for owner in page.caches:
            owner.k_qdata.zero_()
            owner.k_scale_exp.zero_()
            owner.v_qdata.zero_()
            owner.v_scale_exp.zero_()
        page.block_ids.copy_(page.committed_block_ids)
        page.slot_mapping.fill_(-1)

    def _reset_mtp_page_cache(self) -> None:
        page = getattr(self, "_mtp_page_cache", None)
        if page is None:
            return
        page.transaction = None
        page.cache.k_qdata.zero_()
        page.cache.k_scale_exp.zero_()
        page.cache.v_qdata.zero_()
        page.cache.v_scale_exp.zero_()
        for slot_mapping in page.slot_mappings.values():
            slot_mapping.copy_(
                torch.arange(
                    slot_mapping.numel(),
                    dtype=torch.int64,
                    device=self.device,
                )
            )

    @staticmethod
    def _copy_hmfp_block_between(
        source: Any,
        source_block: int,
        destination: Any,
        destination_block: int,
    ) -> None:
        for tensor_name in (
            "k_qdata",
            "k_scale_exp",
            "v_qdata",
            "v_scale_exp",
        ):
            getattr(destination, tensor_name)[destination_block].copy_(
                getattr(source, tensor_name)[source_block]
            )

    def _snapshot_mtp_page_transaction(self, step: int) -> None:
        page = getattr(self, "_mtp_page_cache", None)
        if page is None:
            return
        if page.transaction is None:
            raise RuntimeError("MTP PageAttention transaction is not pending")
        transaction = page.transaction
        if not 0 <= step <= transaction.num_draft_tokens:
            raise RuntimeError(f"MTP snapshot step is outside proposal: {step}")
        snapshot_base = step * page.snapshot_blocks_per_step
        for block_offset, physical_block in enumerate(transaction.physical_block_ids):
            self._copy_hmfp_block_between(
                page.cache,
                physical_block,
                page.snapshot_cache,
                snapshot_base + block_offset,
            )

    def _restore_mtp_page_transaction(self, step: int) -> None:
        page = getattr(self, "_mtp_page_cache", None)
        if page is None or page.transaction is None:
            raise RuntimeError("MTP PageAttention transaction is not pending")
        transaction = page.transaction
        if not 0 <= step <= transaction.num_draft_tokens:
            raise RuntimeError(f"MTP restore step is outside proposal: {step}")
        snapshot_base = step * page.snapshot_blocks_per_step
        for block_offset, physical_block in enumerate(transaction.physical_block_ids):
            self._copy_hmfp_block_between(
                page.snapshot_cache,
                snapshot_base + block_offset,
                page.cache,
                physical_block,
            )

    def _begin_mtp_page_transaction(
        self,
        *,
        req_id: str,
        past_length: int,
        num_draft_tokens: int,
    ) -> None:
        page = getattr(self, "_mtp_page_cache", None)
        if page is None:
            return
        if page.transaction is not None:
            raise RuntimeError(
                "MTP PageAttention proposal already has an uncommitted "
                f"transaction for request {page.transaction.req_id!r}"
            )
        if not 1 <= num_draft_tokens <= int(self.artifact.num_draft_tokens):
            raise RuntimeError(
                "MTP runtime draft width exceeds snapshot capacity: "
                f"runtime={num_draft_tokens}, "
                f"exported={self.artifact.num_draft_tokens}"
            )
        end_position = past_length + num_draft_tokens - 1
        first_block = past_length // page.block_size
        last_block = end_position // page.block_size
        physical_block_ids = tuple(range(first_block, last_block + 1))
        if len(physical_block_ids) > page.snapshot_blocks_per_step:
            raise RuntimeError(
                "MTP proposal crosses more cache pages than the transactional "
                f"snapshot supports: blocks={physical_block_ids}"
            )
        if last_block >= page.cache.num_blocks:
            raise RuntimeError("MTP proposal exceeds private cache capacity")
        page.transaction = _MTPPageCacheTransaction(
            req_id=req_id,
            past_length=past_length,
            num_draft_tokens=num_draft_tokens,
            physical_block_ids=physical_block_ids,
        )
        self._snapshot_mtp_page_transaction(0)

    def commit_mtp_draft(
        self,
        *,
        req_id: str,
        accepted_steps: int,
    ) -> None:
        """Commit the exact private-MTP cache state selected by verification."""

        page = getattr(self, "_mtp_page_cache", None)
        if page is None:
            # Legacy dense FP16 draft caches have per-token storage, so stale
            # rejected rows are safely hidden by the logical length.
            return
        transaction = page.transaction
        if transaction is None or transaction.req_id != req_id:
            raise RuntimeError(
                "MTP PageAttention transaction owner mismatch: "
                f"pending={getattr(transaction, 'req_id', None)!r}, "
                f"commit={req_id!r}"
            )
        if not 1 <= accepted_steps <= transaction.num_draft_tokens + 1:
            raise RuntimeError(
                "MTP accepted steps are outside the proposal: "
                f"accepted={accepted_steps}, "
                f"drafts={transaction.num_draft_tokens}"
            )
        # K draft calls materialize K shifted pairs.  Full acceptance includes
        # one target bonus step, whose missing pair is appended by the
        # controller after this restore.
        restore_step = min(accepted_steps, transaction.num_draft_tokens)
        try:
            self._restore_mtp_page_transaction(restore_step)
        finally:
            page.transaction = None

    def _stage_mtp_page_slots(
        self,
        graph_name: str,
        *,
        past_length: int,
        current_length: int,
    ) -> None:
        page = getattr(self, "_mtp_page_cache", None)
        if page is None:
            return
        slot_mapping = page.slot_mappings[graph_name]
        capacity = slot_mapping.numel()
        if current_length > capacity:
            raise RuntimeError(
                f"MTP {graph_name} current length exceeds graph capacity"
            )
        if past_length + current_length > page.max_sequence_length:
            raise RuntimeError("MTP private page cache capacity exceeded")
        staged = torch.arange(
            past_length,
            past_length + capacity,
            dtype=torch.int64,
            device=self.device,
        )
        if current_length < capacity:
            staged[current_length:].fill_(-1)
        slot_mapping.copy_(staged)

    def _run(
        self,
        graph_name: str,
        feed: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        session = self._sessions[graph_name]
        expected = tuple(session.get_input_names())
        if set(feed) != set(expected):
            raise RuntimeError(
                f"HMONNX {graph_name} input mismatch: "
                f"missing={sorted(set(expected) - set(feed))}, "
                f"unexpected={sorted(set(feed) - set(expected))}"
            )
        raw_outputs = session.run(dict(feed))
        self._assert_private_cache_pointers()
        return _output_map(session, raw_outputs)

    @staticmethod
    def _scalar(
        session: Any,
        name: str,
        value: int,
        device: torch.device,
    ) -> torch.Tensor:
        info = session.get_input(name)
        return torch.full(
            tuple(int(dim) for dim in info.shape),
            int(value),
            dtype=info.dtype,
            device=device,
        )

    def _target_embeddings(
        self,
        token_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        token_ids = token_ids.to(device=self.device, dtype=torch.long)
        embeddings = self.target_model.embed_input_ids(token_ids)
        return embeddings.to(device=self.device, dtype=dtype)

    def _target_vocab_size(self) -> int:
        model_config = getattr(self.runner, "model_config", None)
        get_vocab_size = getattr(model_config, "get_vocab_size", None)
        if callable(get_vocab_size):
            vocab_size = int(get_vocab_size())
            if vocab_size > 0:
                return vocab_size
        candidates = (
            getattr(model_config, "vocab_size", None),
            getattr(getattr(model_config, "hf_config", None), "vocab_size", None),
            getattr(getattr(self.target_model, "config", None), "vocab_size", None),
        )
        for value in candidates:
            if value is not None and int(value) > 0:
                return int(value)
        raise RuntimeError("Cannot resolve the target vocabulary size for DFlash2")

    def take_last_dflash_draft_probs(self) -> torch.Tensor | None:
        probabilities = self._last_dflash_draft_probs
        self._last_dflash_draft_probs = None
        return probabilities

    def _base_feed(
        self,
        session: Any,
        cache_values: Mapping[str, torch.Tensor],
        *,
        past_length: int,
        current_length: int,
    ) -> dict[str, torch.Tensor]:
        feed: dict[str, torch.Tensor] = {}
        for name in session.get_input_names():
            if name in cache_values:
                feed[name] = cache_values[name]
            elif name in {"past_seq_length", "valid_length"}:
                feed[name] = self._scalar(
                    session,
                    name,
                    past_length,
                    self.device,
                )
            elif name in {"current_input_length", "current_length"}:
                feed[name] = self._scalar(
                    session,
                    name,
                    current_length,
                    self.device,
                )
        return feed

    def append_mtp_context(
        self,
        *,
        req_id: str,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        past_length: int,
        next_token_embeddings: torch.Tensor | None = None,
    ) -> None:
        session = self._sessions["prefill"]
        cache_values = self._claim_private_cache(req_id)
        hidden_states = _as_3d(hidden_states, name="MTP target hidden")
        next_token_ids = next_token_ids.reshape(1, -1)
        if hidden_states.shape[1] != next_token_ids.shape[1]:
            raise RuntimeError("MTP hidden/token pair count mismatch")
        if next_token_embeddings is not None:
            next_token_embeddings = _as_3d(
                next_token_embeddings,
                name="MTP target input embeddings",
            )
            if hidden_states.shape[1] != next_token_embeddings.shape[1]:
                raise RuntimeError("MTP hidden/embedding pair count mismatch")
        if hidden_states.shape[1] == 1:
            # A full-accept round leaves exactly one shifted MTP pair to
            # commit. Running the M=256 prefill graph for that single row is
            # needlessly expensive; the autoregressive M=1 decode graph has
            # the same in-place cache mutation contract.
            decode = self._sessions["decode"]
            hidden_info = decode.get_input("post_norm_hidden")
            embedding_info = decode.get_input("next_token_embedding")
            feed = self._base_feed(
                decode,
                cache_values,
                past_length=past_length,
                current_length=1,
            )
            feed["post_norm_hidden"] = hidden_states.to(
                device=self.device,
                dtype=hidden_info.dtype,
            )
            feed["next_token_embedding"] = (
                self._target_embeddings(
                    next_token_ids,
                    dtype=embedding_info.dtype,
                )
                if next_token_embeddings is None
                else next_token_embeddings.to(
                    device=self.device,
                    dtype=embedding_info.dtype,
                )
            )
            self._stage_mtp_page_slots(
                "decode",
                past_length=past_length,
                current_length=1,
            )
            self._run("decode", feed)
            return
        capacity = int(session.get_input("post_norm_hidden").shape[1])
        for offset in range(0, hidden_states.shape[1], capacity):
            chunk_hidden = hidden_states[:, offset : offset + capacity]
            chunk_tokens = next_token_ids[:, offset : offset + capacity]
            chunk_embeddings = (
                next_token_embeddings[:, offset : offset + capacity]
                if next_token_embeddings is not None
                else None
            )
            chunk_length = int(chunk_tokens.shape[1])
            hidden_info = session.get_input("post_norm_hidden")
            embedding_info = session.get_input("next_token_embedding")
            feed = self._base_feed(
                session,
                cache_values,
                past_length=past_length + offset,
                current_length=chunk_length,
            )
            feed["post_norm_hidden"] = _pad_sequence(
                chunk_hidden,
                target_length=capacity,
                dtype=hidden_info.dtype,
                device=self.device,
            )
            embeddings = (
                self._target_embeddings(
                    chunk_tokens,
                    dtype=embedding_info.dtype,
                )
                if chunk_embeddings is None
                else chunk_embeddings.to(
                    device=self.device,
                    dtype=embedding_info.dtype,
                )
            )
            feed["next_token_embedding"] = _pad_sequence(
                embeddings,
                target_length=capacity,
                dtype=embedding_info.dtype,
                device=self.device,
            )
            self._stage_mtp_page_slots(
                "prefill",
                past_length=past_length + offset,
                current_length=chunk_length,
            )
            self._run("prefill", feed)
            if determinism_trace_enabled():
                emit_determinism_event(
                    "spec_mtp_context_append",
                    request_id=req_id,
                    payload={
                        "past_length": past_length + offset,
                        "current_length": chunk_length,
                        "next_token_ids": [
                            int(token_id)
                            for token_id in chunk_tokens.reshape(-1)
                            .detach()
                            .to(device="cpu")
                            .tolist()
                        ],
                        "hidden_states": determinism_fingerprint(chunk_hidden),
                        "next_token_embedding": determinism_fingerprint(embeddings),
                    },
                )

    def draft_mtp(
        self,
        *,
        req_id: str,
        current_token_id: int,
        target_hidden: torch.Tensor,
        past_length: int,
        num_draft_tokens: int,
    ) -> Sequence[int]:
        session = self._sessions["decode"]
        cache_values = self._claim_private_cache(req_id)
        self._begin_mtp_page_transaction(
            req_id=req_id,
            past_length=past_length,
            num_draft_tokens=num_draft_tokens,
        )
        current_hidden = _as_3d(
            target_hidden,
            name="MTP current target hidden",
        )
        current_token = torch.tensor(
            [[current_token_id]],
            dtype=torch.long,
            device=self.device,
        )
        draft_tokens = torch.empty(
            (num_draft_tokens,),
            dtype=torch.long,
            device=self.device,
        )
        try:
            for draft_index in range(num_draft_tokens):
                hidden_info = session.get_input("post_norm_hidden")
                embedding_info = session.get_input("next_token_embedding")
                feed = self._base_feed(
                    session,
                    cache_values,
                    past_length=past_length + draft_index,
                    current_length=1,
                )
                feed["post_norm_hidden"] = current_hidden.to(
                    device=self.device,
                    dtype=hidden_info.dtype,
                )
                feed["next_token_embedding"] = self._target_embeddings(
                    current_token,
                    dtype=embedding_info.dtype,
                )
                self._stage_mtp_page_slots(
                    "decode",
                    past_length=past_length + draft_index,
                    current_length=1,
                )
                outputs = self._run("decode", feed)
                self._snapshot_mtp_page_transaction(draft_index + 1)
                logits = outputs.get("logits")
                next_hidden = outputs.get("post_norm_out")
                if logits is None or next_hidden is None:
                    raise RuntimeError(
                        "Qwen MTP decode must return logits and post_norm_out"
                    )
                token_tensor = (
                    logits.reshape(-1, logits.shape[-1])[-1]
                    .argmax(dim=-1)
                    .reshape(1, 1)
                )
                draft_tokens[draft_index].copy_(token_tensor.reshape(-1)[0])
                if determinism_trace_enabled():
                    token = int(token_tensor.item())
                    emit_determinism_event(
                        "spec_mtp_draft_step",
                        request_id=req_id,
                        payload={
                            "draft_index": draft_index,
                            "past_length": past_length + draft_index,
                            "input_token_id": int(current_token.item()),
                            "input_hidden": determinism_fingerprint(current_hidden),
                            "input_embedding": determinism_fingerprint(
                                feed["next_token_embedding"]
                            ),
                            "logits_argmax": token,
                            "logits": determinism_fingerprint(logits),
                            "output_hidden": determinism_fingerprint(next_hidden),
                        },
                    )
                # Match vLLM's autoregressive speculator contract: token
                # feedback remains device-side between graph replays. There is
                # no K-times .item() stream synchronization on production.
                current_token = token_tensor
                current_hidden = _as_3d(
                    next_hidden,
                    name="MTP assistant hidden",
                )[:, -1:]
        except Exception:
            page = self._mtp_page_cache
            if page is not None and page.transaction is not None:
                try:
                    self._restore_mtp_page_transaction(0)
                finally:
                    page.transaction = None
            raise
        return [
            int(token)
            for token in draft_tokens.to(
                device="cpu",
                non_blocking=False,
            ).tolist()
        ]

    @staticmethod
    def _present_cache_input_name(output_name: str) -> str | None:
        if output_name.startswith("present_key_cache"):
            return output_name.replace("present_key_cache", "past_key_cache", 1)
        if output_name.startswith("present_value_cache"):
            return output_name.replace(
                "present_value_cache",
                "past_value_cache",
                1,
            )
        return None

    def append_dflash_context(
        self,
        *,
        req_id: str,
        hidden_states: torch.Tensor,
        past_length: int,
        decode_step: bool,
    ) -> None:
        graph_name = (
            "context_decode"
            if decode_step and "context_decode" in self._sessions
            else "context"
        )
        session = self._sessions[graph_name]
        cache_values = self._claim_private_cache(req_id)
        hidden_states = _as_3d(
            hidden_states,
            name="DFlash target hidden",
        )
        capacity = int(session.get_input("target_hidden").shape[1])
        for offset in range(0, hidden_states.shape[1], capacity):
            chunk = hidden_states[:, offset : offset + capacity]
            chunk_length = int(chunk.shape[1])
            hidden_info = session.get_input("target_hidden")
            feed = self._base_feed(
                session,
                cache_values,
                past_length=past_length + offset,
                current_length=chunk_length,
            )
            feed["target_hidden"] = _pad_sequence(
                chunk,
                target_length=capacity,
                dtype=hidden_info.dtype,
                device=self.device,
            )
            if self._uses_graph_dflash_page_cache_writes():
                self._stage_dflash_page_slots(
                    past_length=past_length + offset,
                    current_length=chunk_length,
                    physical_length=capacity,
                )
            outputs = self._run(graph_name, feed)
            if self._uses_private_dflash_page_cache():
                if not self._uses_graph_dflash_page_cache_writes():
                    self._write_dflash_page_context(
                        outputs,
                        past_length=past_length + offset,
                        current_length=chunk_length,
                    )
                continue
            for output_name, output in outputs.items():
                input_name = self._present_cache_input_name(output_name)
                if input_name is None or input_name not in cache_values:
                    raise RuntimeError(
                        "DFlash context returned an output that cannot be "
                        "matched to its private cache input: "
                        f"output={output_name}, input={input_name}"
                    )
                if output.data_ptr() != cache_values[input_name].data_ptr():
                    raise RuntimeError(
                        "DFlash context materialized/copied an unused cache "
                        "output instead of reusing the persistent input: "
                        f"output={output_name}, input={input_name}"
                    )
            # The session's output-reuse policy makes these otherwise-unused
            # graph outputs alias the original full-capacity CacheTensors.
            # Deliberately discard the output references and retain those
            # persistent inputs; never copy/rebind a growing valid prefix.

    def _write_dflash_page_context(
        self,
        outputs: Mapping[str, torch.Tensor],
        *,
        past_length: int,
        current_length: int,
    ) -> None:
        """Quantize only the new target K/V rows into private draft pages."""

        from xhquant.xhonnxruntime.parsers import PageAttentionContext

        page = getattr(self, "_dflash_page_cache", None)
        if page is None:
            raise RuntimeError("Paged DFlash context has no private page cache")
        if past_length + current_length > page.max_sequence_length:
            raise RuntimeError(
                "DFlash private page cache capacity exceeded: "
                f"end={past_length + current_length}, "
                f"capacity={page.max_sequence_length}"
            )
        key_outputs = [
            outputs.get(f"present_key_cache_{index}")
            for index in range(len(page.modules))
        ]
        value_outputs = [
            outputs.get(f"present_value_cache_{index}")
            for index in range(len(page.modules))
        ]
        if any(value is None for value in (*key_outputs, *value_outputs)):
            raise RuntimeError(
                "Paged DFlash context must return one projected K/V tensor "
                "per draft decoder layer"
            )
        capacity = int(key_outputs[0].shape[2])  # type: ignore[union-attr]
        slot_mapping = torch.arange(
            capacity,
            dtype=torch.int64,
            device=self.device,
        )
        slot_mapping.add_(past_length)
        if current_length < capacity:
            slot_mapping[current_length:].fill_(-1)
        past_tensor = torch.tensor(
            [past_length],
            dtype=torch.int64,
            device=self.device,
        )
        current_tensor = torch.tensor(
            [current_length],
            dtype=torch.int64,
            device=self.device,
        )
        for index, (module, owner, decode_context) in enumerate(
            zip(page.modules, page.caches, page.contexts, strict=True)
        ):
            write_context = PageAttentionContext(
                paged_kv_cache=owner,
                block_ids=page.block_ids,
                slot_mapping=slot_mapping,
                block_size=page.block_size,
            )
            module.set_context(write_context)
            try:
                module._write_k_cache(
                    key_outputs[index],
                    current_tensor,
                )
                module._write_v_cache(
                    value_outputs[index],
                    past_tensor,
                    current_tensor,
                )
            finally:
                # Decode CUDA Graph capture and replay own stable metadata
                # addresses. Never leave a short-lived context tensor bound.
                module.set_context(decode_context)
        self._assert_private_cache_pointers()

    def _stage_dflash_page_slots(
        self,
        *,
        past_length: int,
        current_length: int,
        physical_length: int,
        transient: bool = False,
    ) -> None:
        page = getattr(self, "_dflash_page_cache", None)
        if page is None:
            return
        if not 0 <= current_length <= physical_length:
            raise RuntimeError(
                "DFlash page write length is outside its physical graph shape"
            )
        if physical_length > page.slot_mapping.numel():
            raise RuntimeError("DFlash page slot buffer is too small")
        if past_length + current_length > page.max_sequence_length:
            raise RuntimeError("DFlash page write exceeds private cache capacity")
        page.block_ids.copy_(page.committed_block_ids)
        if transient:
            self._stage_dflash_transient_cow(
                past_length=past_length,
                current_length=current_length,
                physical_length=physical_length,
            )
            return
        torch.add(
            page.slot_offsets[:physical_length],
            past_length,
            out=page.slot_mapping[:physical_length],
        )
        if current_length < physical_length:
            page.slot_mapping[current_length:physical_length].fill_(-1)
        if physical_length < page.slot_mapping.numel():
            page.slot_mapping[physical_length:].fill_(-1)

    def _stage_dflash_transient_cow(
        self,
        *,
        past_length: int,
        current_length: int,
        physical_length: int,
    ) -> None:
        """Expose draft K/V through scratch pages without mutating the prefix."""

        page = self._dflash_page_cache
        block_size = page.block_size
        first_logical_block = past_length // block_size
        first_lane = past_length % block_size
        first_scratch, second_scratch = page.scratch_block_ids
        first_count = min(current_length, block_size - first_lane)
        second_count = current_length - first_count
        if second_count > block_size:
            raise RuntimeError("DFlash transient tail exceeds two COW pages")

        from xhquant.core.paged_kv_cache_write import fork_hmfp_cache_page

        for owner in page.caches:
            fork_hmfp_cache_page(
                owner.k_qdata,
                owner.k_scale_exp,
                owner.v_qdata,
                owner.v_scale_exp,
                source_block=first_logical_block,
                first_scratch_block=first_scratch,
                second_scratch_block=second_scratch,
                copy_source=bool(first_lane),
            )

        page.block_ids[first_logical_block] = first_scratch
        if second_count:
            page.block_ids[first_logical_block + 1] = second_scratch

        page.slot_mapping.fill_(-1)
        if first_count:
            torch.add(
                page.slot_offsets[:first_count],
                first_scratch * block_size + first_lane,
                out=page.slot_mapping[:first_count],
            )
        if second_count:
            torch.add(
                page.slot_offsets[:second_count],
                second_scratch * block_size,
                out=page.slot_mapping[first_count : first_count + second_count],
            )
        if physical_length < page.slot_mapping.numel():
            page.slot_mapping[physical_length:].fill_(-1)

    def draft_dflash(
        self,
        *,
        req_id: str,
        current_token_id: int,
        past_length: int,
        num_draft_tokens: int,
        temperature: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Sequence[int]:
        self._last_dflash_draft_probs = None
        session = self._sessions["decode"]
        cache_values = self._claim_private_cache(req_id)
        noise_info = session.get_input("noise_embedding")
        block_length = int(noise_info.shape[1])
        if block_length < num_draft_tokens + 1:
            raise RuntimeError(
                "DFlash decode graph is shorter than verify_length: "
                f"graph={block_length}, required={num_draft_tokens + 1}"
            )
        noise_token_id = self.artifact.dflash_noise_token_id
        if noise_token_id is None:
            raise RuntimeError("DFlash artifact has no noise token id")
        block_ids = torch.full(
            (1, block_length),
            int(noise_token_id),
            dtype=torch.long,
            device=self.device,
        )
        block_ids[0, 0] = int(current_token_id)
        noise_embedding = self._target_embeddings(
            block_ids,
            dtype=noise_info.dtype,
        )
        feed = self._base_feed(
            session,
            cache_values,
            past_length=past_length,
            current_length=block_length,
        )
        self._stage_dflash_page_slots(
            past_length=past_length,
            current_length=block_length,
            physical_length=block_length,
            transient=True,
        )
        feed["noise_embedding"] = noise_embedding
        input_names = set(session.get_input_names())
        if "anchor_token_id" in input_names:
            anchor_info = session.get_input("anchor_token_id")
            feed["anchor_token_id"] = torch.full(
                tuple(int(dim) for dim in anchor_info.shape),
                int(current_token_id),
                dtype=anchor_info.dtype,
                device=self.device,
            )
        if "attn_mask" in input_names:
            mask_info = session.get_input("attn_mask")
            mask_shape = tuple(int(dim) for dim in mask_info.shape)
            if getattr(self.artifact, "is_qwen_dflash2", False):
                sliding_window = self.artifact.dflash_sliding_window
                if sliding_window is None:
                    raise RuntimeError("DFlash2 artifact has no sliding_window")
                mask = _dflash2_attention_mask(
                    mask_shape,
                    past_length=past_length,
                    current_length=block_length,
                    sliding_window=sliding_window,
                    dtype=mask_info.dtype,
                    device=self.device,
                )
            else:
                mask = torch.full(
                    mask_shape,
                    -10000.0,
                    dtype=mask_info.dtype,
                    device=self.device,
                )
                visible_length = min(
                    int(mask.shape[-1]),
                    past_length + block_length,
                )
                mask[..., :visible_length] = 0
            feed["attn_mask"] = mask
        if "query_kv_range_abs" in input_names:
            range_info = session.get_input("query_kv_range_abs")
            sliding_window = self.artifact.dflash_sliding_window
            if sliding_window is None:
                raise RuntimeError("DFlash2 artifact has no sliding_window")
            feed["query_kv_range_abs"] = _dflash2_query_kv_ranges(
                range_info.shape,
                past_length=past_length,
                current_length=block_length,
                sliding_window=sliding_window,
                dtype=range_info.dtype,
                device=self.device,
            )
        outputs = self._run("decode", feed)
        candidate_ids = outputs.get("candidate_ids")
        if candidate_ids is not None:
            first_scores = outputs.get("selector_first_scores")
            transition_scores = outputs.get("selector_transition_scores")
            if first_scores is None or transition_scores is None:
                raise RuntimeError(
                    "DFlash2 decode graph must return candidate_ids, "
                    "selector_first_scores, and selector_transition_scores"
                )
            if temperature is not None and temperature > 0:
                draft_tokens, self._last_dflash_draft_probs = (
                    _dflash2_sample_selector_walk(
                        candidate_ids,
                        first_scores,
                        transition_scores,
                        num_draft_tokens=num_draft_tokens,
                        temperature=temperature,
                        vocab_size=self._target_vocab_size(),
                        generator=generator,
                    )
                )
            else:
                draft_tokens = _dflash2_greedy_selector_walk(
                    candidate_ids,
                    first_scores,
                    transition_scores,
                    num_draft_tokens=num_draft_tokens,
                )
            if draft_tokens.shape[0] != 1:
                raise RuntimeError(
                    "The serial HMONNX DFlash2 runtime supports batch size 1"
                )
            return [
                int(token)
                for token in draft_tokens[0]
                .to(device="cpu", non_blocking=False)
                .tolist()
            ]
        logits = outputs.get("logits")
        if logits is None:
            raise RuntimeError("DFlash decode graph did not return logits")
        logits = logits.reshape(-1, logits.shape[-1])
        if logits.shape[0] < num_draft_tokens + 1:
            raise RuntimeError("DFlash logits are shorter than the draft block")
        # DFlash predicts the whole block in one graph.  Keep rejection
        # candidates on device through one vectorized argmax and perform one
        # batched D2H transfer; calling ``.item()`` K times inserted K stream
        # synchronizations and prevented the scheduler from overlapping the
        # remaining host work with GPU execution.
        draft_tokens = logits[1 : num_draft_tokens + 1].argmax(dim=-1)
        return [
            int(token)
            for token in draft_tokens.to(device="cpu", non_blocking=False).tolist()
        ]

    def can_draft_dflash(
        self,
        *,
        past_length: int,
        num_draft_tokens: int,
    ) -> bool:
        """Return whether the complete fixed DFlash block still fits.

        The exported decode graph keeps the checkpoint's physical block width,
        which can exceed ``num_draft_tokens + 1``. All physical rows
        participate in the non-causal cross-attention and consume RoPE
        positions, even when vLLM verifies only a prefix of its predictions.
        """

        session = self._sessions["decode"]
        physical_length = int(session.get_input("noise_embedding").shape[1])
        if physical_length < num_draft_tokens + 1:
            raise RuntimeError(
                "DFlash decode graph is shorter than verify_length: "
                f"graph={physical_length}, required={num_draft_tokens + 1}"
            )
        limits: list[int] = []
        page = getattr(self, "_dflash_page_cache", None)
        if page is not None:
            limits.append(int(page.max_sequence_length))
        runtime_max = int(
            getattr(
                getattr(self.runner, "model_config", None),
                "max_model_len",
                0,
            )
            or 0
        )
        if runtime_max > 0:
            limits.append(runtime_max)
        if not limits:
            raise RuntimeError(
                "DFlash runtime has no authoritative sequence-length limit"
            )
        return past_length + physical_length <= min(limits)

    def _resolve_gemma_owner_bindings(self) -> dict[str, Any]:
        if self._gemma_owner_bindings is not None:
            return self._gemma_owner_bindings
        bindings = tuple(getattr(self.target_model, "page_attention_bindings", ()))
        placeholders = tuple(getattr(self.target_model, "placeholder_attn_layers", ()))
        resolved: dict[str, Any] = {}
        for attention_kind in set(self.artifact.draft_layer_attention_types):
            candidates = [
                binding
                for binding in bindings
                if binding.attention_kind == attention_kind
                and getattr(binding, "cache_group_idx", None) is not None
                and getattr(binding, "cache_owner_layer_idx", None)
                in (None, binding.layer_idx)
            ]
            if not candidates:
                raise RuntimeError(
                    "Gemma4 MTP has no finalized non-shared target owner for "
                    f"{attention_kind!r}"
                )
            binding = max(candidates, key=lambda value: value.layer_idx)
            if binding.layer_idx >= len(placeholders):
                raise RuntimeError(
                    "Gemma4 target placeholder list is shorter than its "
                    f"cache contract: layer={binding.layer_idx}"
                )
            cache = getattr(placeholders[binding.layer_idx], "kv_cache", None)
            is_fp_page_cache = (
                isinstance(cache, torch.Tensor)
                and cache.ndim == 5
                and cache.shape[0] == 2
            )
            if cache is None or (
                not hasattr(cache, "block_size") and not is_fp_page_cache
            ):
                raise RuntimeError(
                    "Gemma4 target cache owners are not initialized before "
                    "assistant drafting"
                )
            resolved[attention_kind] = (binding, cache)
        self._gemma_owner_bindings = resolved
        return resolved

    def _bind_gemma_page_contexts(
        self,
        *,
        target_visible_length: int,
        page_attention_group_contexts: dict[int, Any],
    ) -> None:
        from xhquant.xhonnxruntime.parsers import PageAttentionContext

        session = self._sessions["decode"]
        graph_module = session.graph_module
        page_modules = []
        for node in graph_module.graph.nodes:
            if node.op != "call_module":
                continue
            module = graph_module.get_submodule(str(node.target))
            if module.__class__.__name__ in {
                "PageAttentionReadOnly",
                "PageKVCacheReadOnly",
            }:
                page_modules.append(module)
        draft_types = self.artifact.draft_layer_attention_types
        causal_readonly = bool(
            getattr(
                self.artifact,
                "uses_causal_readonly_target_page_cache",
                False,
            )
        )
        mixed_readonly = bool(
            getattr(
                self.artifact,
                "uses_mixed_readonly_target_page_cache",
                False,
            )
        )
        if causal_readonly or mixed_readonly:
            layer_indices = [
                int(getattr(module, "layer_idx", -1)) for module in page_modules
            ]
            if sorted(layer_indices) != list(range(len(page_modules))):
                raise RuntimeError(
                    "Gemma4 read-only PageAttention v2 requires stable, "
                    f"contiguous layer_idx attrs, got {layer_indices}"
                )
            page_modules = [
                module
                for _, module in sorted(
                    zip(layer_indices, page_modules, strict=True),
                    key=lambda item: item[0],
                )
            ]
        if len(page_modules) != len(draft_types):
            raise RuntimeError(
                "Gemma4 assistant PageAttention layer count mismatch: "
                f"graph={len(page_modules)}, contract={len(draft_types)}"
            )

        owners = self._resolve_gemma_owner_bindings()
        if self._gemma_page_bindings is None:
            max_model_len = int(
                getattr(getattr(self.runner, "model_config", None), "max_model_len", 0)
                or 0
            )
            stable_bindings: dict[str, _GemmaReadOnlyPageBinding] = {}
            for attention_kind in sorted(set(draft_types)):
                binding, cache = owners[attention_kind]
                cache_device = cache.device
                block_size = (
                    int(cache.shape[3])
                    if isinstance(cache, torch.Tensor)
                    else int(cache.block_size)
                )
                num_blocks = (
                    int(cache.shape[1])
                    if isinstance(cache, torch.Tensor)
                    else int(cache.num_blocks)
                )
                # Sliding pages form a compact physical ring, but the attention
                # kernel still resolves absolute logical page indices through
                # block_ids.  Size the table for the runtime context, not just
                # for the smaller physical backing.
                logical_blocks = (
                    (max_model_len + block_size - 1) // block_size
                    if max_model_len > 0
                    else 0
                )
                block_capacity = max(num_blocks, logical_blocks)
                if block_capacity <= 0:
                    raise RuntimeError(
                        "Gemma4 target page owner has no usable block-table "
                        f"capacity for {attention_kind}"
                    )
                block_ids = torch.zeros(
                    block_capacity,
                    dtype=torch.int64,
                    device=cache_device,
                )
                seq_lens = torch.zeros(
                    1,
                    dtype=torch.int64,
                    device=cache_device,
                )
                # PageAttentionReadOnly never consumes slot_mapping. Keep one
                # stable sentinel solely for the shared context ABI.
                slot_mapping = torch.full(
                    (1,),
                    -1,
                    dtype=torch.int64,
                    device=cache_device,
                )
                context = PageAttentionContext(
                    paged_kv_cache=cache,
                    block_ids=block_ids,
                    slot_mapping=slot_mapping,
                    block_size=block_size,
                    # Causal v2 carries compact start/valid metadata as graph
                    # inputs.  Keeping the context value empty avoids applying
                    # the target forward's stale pre-accept offset twice.
                    kv_window_start_abs=None,
                    seq_lens=seq_lens,
                    mm_prefix_ranges=None,
                )
                stable_bindings[attention_kind] = _GemmaReadOnlyPageBinding(
                    context=context,
                    block_ids=block_ids,
                    seq_lens=seq_lens,
                    cache=cache,
                    group_idx=int(binding.cache_group_idx),
                )
            for module, attention_kind in zip(
                page_modules,
                draft_types,
                strict=True,
            ):
                is_causal = bool(getattr(module, "is_causal", False))
                has_exact_range = bool(getattr(module, "has_query_kv_range_abs", False))
                has_window_start = bool(
                    getattr(module, "has_kv_window_start_abs", False)
                )
                has_valid_length = bool(getattr(module, "has_kv_valid_length", False))
                sliding_window = getattr(module, "sliding_window", None)
                if causal_readonly or mixed_readonly:
                    module_attention_kind = str(getattr(module, "attention_kind", ""))
                    if module_attention_kind != attention_kind:
                        raise RuntimeError(
                            "Gemma4 read-only PageAttention v2 layer identity "
                            f"mismatch: manifest={attention_kind}, "
                            f"graph={module_attention_kind!r}"
                        )
                    if not is_causal or has_exact_range:
                        raise RuntimeError(
                            "Gemma4 read-only PageAttention v2 requires causal "
                            f"attention without exact ranges for {attention_kind}"
                        )
                    if attention_kind == "sliding_attention":
                        if type(sliding_window) is not int or sliding_window <= 0:
                            raise RuntimeError(
                                "Gemma4 causal sliding draft is missing its "
                                "sliding_window"
                            )
                        if causal_readonly and (
                            not has_window_start or not has_valid_length
                        ):
                            raise RuntimeError(
                                "Gemma4 causal sliding draft must consume "
                                "kv_window_start_abs and kv_valid_length"
                            )
                        if mixed_readonly and (
                            module.__class__.__name__ != "PageKVCacheReadOnly"
                            or has_window_start
                            or has_valid_length
                        ):
                            raise RuntimeError(
                                "Gemma4 mixed MTP sliding layers must use "
                                "PageKVCacheReadOnly without compact metadata"
                            )
                        stable = stable_bindings[attention_kind]
                        if (
                            stable.sliding_window is not None
                            and stable.sliding_window != sliding_window
                        ):
                            raise RuntimeError(
                                "Gemma4 causal sliding draft layers disagree "
                                "on sliding_window"
                            )
                        stable.sliding_window = int(sliding_window)
                    elif (
                        sliding_window is not None
                        or has_window_start
                        or has_valid_length
                    ):
                        raise RuntimeError(
                            "Gemma4 causal full draft unexpectedly carries "
                            "sliding-cache metadata"
                        )
                    elif (
                        mixed_readonly
                        and module.__class__.__name__ != "PageAttentionReadOnly"
                    ):
                        raise RuntimeError(
                            "Gemma4 mixed MTP full layers must use PageAttentionReadOnly"
                        )
                else:
                    if is_causal or not has_exact_range:
                        raise RuntimeError(
                            "Gemma4 read-only PageAttention v1 requires "
                            "non-causal Padding Full Cross-Attention with "
                            f"explicit KV ranges for {attention_kind}"
                        )
                module.set_context(stable_bindings[attention_kind].context)
            self._gemma_page_bindings = stable_bindings

        assert self._gemma_page_bindings is not None
        for attention_kind, stable in self._gemma_page_bindings.items():
            binding, cache = owners[attention_kind]
            if cache is not stable.cache:
                raise RuntimeError(
                    "Gemma4 target page cache owner was rebound after draft "
                    f"initialization for {attention_kind}"
                )
            group_idx = int(binding.cache_group_idx)
            if group_idx != stable.group_idx:
                raise RuntimeError(
                    "Gemma4 target cache group changed after draft "
                    f"initialization for {attention_kind}: "
                    f"{stable.group_idx} -> {group_idx}"
                )
            group_context = page_attention_group_contexts.get(group_idx)
            if group_context is None:
                raise RuntimeError(
                    "Gemma4 assistant is missing target page context for "
                    f"group {group_idx}"
                )
            source_block_ids = group_context.block_ids.reshape(-1)
            cache_block_size = (
                int(cache.shape[3])
                if isinstance(cache, torch.Tensor)
                else int(cache.block_size)
            )
            required_blocks = (
                target_visible_length + cache_block_size - 1
            ) // cache_block_size
            if source_block_ids.numel() < required_blocks:
                raise RuntimeError(
                    "Gemma4 target block table does not cover the committed "
                    f"prefix for {attention_kind}: required={required_blocks}, "
                    f"available={source_block_ids.numel()}"
                )
            if required_blocks > stable.block_ids.numel():
                raise RuntimeError(
                    "Gemma4 target block table exceeds its stable draft "
                    f"capacity for {attention_kind}: "
                    f"required={required_blocks}, "
                    f"capacity={stable.block_ids.numel()}"
                )
            # No allocation/rebinding and no KV movement: copy only the tiny
            # logical table prefix the committed length can address.  Entries
            # after ``required_blocks`` are deliberately left stale because
            # seq_lens makes them unreachable; zeroing/copying the exported
            # max-length table on every proposal would add pure overhead.
            stable.block_ids[:required_blocks].copy_(
                source_block_ids[:required_blocks],
                non_blocking=False,
            )
            stable.seq_lens.fill_(target_visible_length)

    def draft_gemma4(
        self,
        *,
        req_id: str,
        current_token_id: int,
        target_hidden: torch.Tensor,
        target_visible_length: int,
        num_draft_tokens: int,
        page_attention_group_contexts: dict[int, Any],
    ) -> Sequence[int]:
        del req_id
        if target_visible_length <= 0:
            raise RuntimeError("Gemma4 MTP requires a non-empty target cache")
        self._bind_gemma_page_contexts(
            target_visible_length=target_visible_length,
            page_attention_group_contexts=page_attention_group_contexts,
        )
        session = self._sessions["decode"]
        input_name = "input_1"
        input_info = session.get_input(input_name)
        input_names = tuple(session.get_input_names())
        compact_input_names = {
            "kv_window_start_abs",
            "kv_valid_length",
        }.intersection(input_names)
        causal_readonly = bool(
            getattr(
                self.artifact,
                "uses_causal_readonly_target_page_cache",
                False,
            )
        )
        mixed_readonly = bool(
            getattr(
                self.artifact,
                "uses_mixed_readonly_target_page_cache",
                False,
            )
        )
        if causal_readonly and compact_input_names != {
            "kv_window_start_abs",
            "kv_valid_length",
        }:
            raise RuntimeError(
                "Gemma4 read-only PageAttention v2 graph must expose both "
                "kv_window_start_abs and kv_valid_length"
            )
        if not causal_readonly and compact_input_names:
            raise RuntimeError(
                "Legacy Gemma4 read-only PageAttention graph unexpectedly "
                "exposes compact sliding-cache metadata"
            )
        metadata_feed: dict[str, torch.Tensor] = {}
        for name in input_names:
            if name in {"past_seq_length", "valid_length"}:
                # This is the causal attention alignment input, not the query
                # RoPE position.  The graph reconstructs RoPE=N from
                # (past=N-1, current=1), while attention exposes the N
                # target-owned KV entries without appending assistant KV.
                metadata_feed[name] = self._scalar(
                    session,
                    name,
                    target_visible_length - 1,
                    self.device,
                )
            elif name in {"current_input_length", "current_length"}:
                metadata_feed[name] = self._scalar(
                    session,
                    name,
                    1,
                    self.device,
                )
        if causal_readonly:
            assert self._gemma_page_bindings is not None
            sliding_binding = self._gemma_page_bindings.get("sliding_attention")
            if sliding_binding is None or sliding_binding.sliding_window is None:
                raise RuntimeError(
                    "Gemma4 read-only PageAttention v2 has no sliding-cache binding"
                )
            window_start, valid_width = _gemma_readonly_sliding_cache_range(
                target_visible_length,
                sliding_binding.sliding_window,
            )
            metadata_feed["kv_window_start_abs"] = self._scalar(
                session,
                "kv_window_start_abs",
                window_start,
                self.device,
            )
            metadata_feed["kv_valid_length"] = self._scalar(
                session,
                "kv_valid_length",
                valid_width,
                self.device,
            )
        elif mixed_readonly:
            mask_name = next(
                (name for name in input_names if name == "sliding_attention_mask"),
                None,
            )
            if mask_name is None:
                raise RuntimeError(
                    "Gemma4 mixed MTP graph is missing sliding_attention_mask"
                )
            assert self._gemma_page_bindings is not None
            sliding_binding = self._gemma_page_bindings.get("sliding_attention")
            if sliding_binding is None or sliding_binding.sliding_window is None:
                raise RuntimeError("Gemma4 mixed MTP has no sliding cache binding")
            mask_info = session.get_input(mask_name)
            mask_shape = tuple(int(dim) for dim in mask_info.shape)
            if len(mask_shape) != 4:
                raise RuntimeError(
                    f"Gemma4 mixed MTP sliding mask must be rank 4, got {mask_shape}"
                )
            sliding_mask = torch.full(
                mask_shape,
                torch.finfo(mask_info.dtype).min,
                dtype=mask_info.dtype,
                device=self.device,
            )
            valid_width = min(
                target_visible_length,
                int(sliding_binding.sliding_window),
                mask_shape[-1],
            )
            sliding_mask[..., :valid_width] = 0
            metadata_feed[mask_name] = sliding_mask
        current_hidden = _as_3d(
            target_hidden,
            name="Gemma4 target hidden",
        )[:, -1:]
        current_token = torch.tensor(
            [[current_token_id]],
            dtype=torch.long,
            device=self.device,
        )
        draft_tokens = torch.empty(
            (num_draft_tokens,),
            dtype=torch.long,
            device=self.device,
        )
        for draft_index in range(num_draft_tokens):
            token_embedding = self._target_embeddings(
                current_token,
                dtype=input_info.dtype,
            )
            assistant_input = torch.cat(
                [
                    token_embedding,
                    current_hidden.to(
                        device=self.device,
                        dtype=input_info.dtype,
                    ),
                ],
                dim=-1,
            )
            expected_shape = tuple(int(dim) for dim in input_info.shape)
            if tuple(assistant_input.shape) != expected_shape:
                raise RuntimeError(
                    "Gemma4 assistant input shape mismatch: "
                    f"expected={expected_shape}, "
                    f"got={tuple(assistant_input.shape)}"
                )
            # Length/range inputs are constant for all proposals because the
            # assistant never appends KV. Reuse one allocation per round.
            feed: dict[str, torch.Tensor] = {
                input_name: assistant_input,
                **metadata_feed,
            }
            outputs = self._run("decode", feed)
            logits = outputs.get("logits")
            next_hidden = outputs.get("assistant_hidden_state")
            if logits is None or next_hidden is None:
                raise RuntimeError(
                    "Gemma4 MTP assistant must return logits and assistant_hidden_state"
                )
            token_tensor = (
                logits.reshape(-1, logits.shape[-1])[-1].argmax(dim=-1).reshape(1, 1)
            )
            draft_tokens[draft_index].copy_(token_tensor.reshape(-1)[0])
            # Keep autoregressive token feedback on device. Converting every
            # draft token with .item() serializes the stream K times and
            # defeats most of the MTP speedup.
            current_token = token_tensor
            current_hidden = _as_3d(
                next_hidden,
                name="Gemma4 assistant hidden",
            )[:, -1:]
        return [
            int(token)
            for token in draft_tokens.to(
                device="cpu",
                non_blocking=False,
            ).tolist()
        ]

    def release_request(self, req_id: str) -> None:
        cache = self._private_cache
        if cache is None or cache.owner_req_id != req_id:
            return
        for value in cache.values.values():
            value.data.zero_()
        # Paged speculative ABIs deliberately keep ``values`` empty: their
        # graph modules share a stable external cache owner instead of dense
        # tensor inputs.  Reset that owner before releasing the serial request
        # lease so the next request cannot inherit stale SEFP groups.
        self._reset_dflash_page_cache()
        self._reset_mtp_page_cache()
        cache.owner_req_id = None
        self._assert_private_cache_pointers()


__all__ = ["HMONNXDraftRuntimeBackend"]
