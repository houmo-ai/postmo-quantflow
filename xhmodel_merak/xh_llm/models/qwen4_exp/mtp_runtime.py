"""State-safe greedy MTP verification for Qwen4's recurrent target.

A rejected target verify call cannot be cropped by merely shortening K/V:
GDR recurrent state, PLE convolution/history, QSA indexer keys and positions
must all return to the same prefix. The caller supplies a complete snapshot
and replay implementation so PageKV and runtime-owned caches stay in sync.
"""

from __future__ import annotations

from collections import ChainMap
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch


@dataclass(frozen=True)
class MTPVerification:
    accepted_draft_tokens: int
    committed_input_tokens: tuple[int, ...]
    next_token: int
    target_hidden: torch.Tensor


def verify_draft_transaction(
    *,
    pending_token: int,
    draft_tokens: Sequence[int],
    snapshot: Callable[[], Any],
    restore: Callable[[Any], None],
    verify: Callable[[Sequence[int]], tuple[torch.Tensor, torch.Tensor]],
    replay: Callable[[Sequence[int]], tuple[torch.Tensor, torch.Tensor]],
) -> MTPVerification:
    """Verify a draft and leave every target state at the accepted prefix.

    ``pending_token`` is the already chosen target token that is not yet in
    its cache. ``verify`` consumes pending+draft and returns [1,N,V] logits
    and [1,N,HC*H] *pre-mixer* states. ``replay`` must use the single-token
    target graph, retaining the same numerical/cache semantics. On rejection
    it replays only the consumed accepted prefix after a complete restore.
    The returned ``next_token`` remains pending for the next transaction.
    """
    inputs = (int(pending_token), *(int(t) for t in draft_tokens))
    before = snapshot()
    try:
        logits, hidden = verify(inputs)
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(inputs):
            raise ValueError("Qwen4 MTP verify must return logits for every input token")
        if hidden.ndim != 3 or hidden.shape[:2] != logits.shape[:2]:
            raise ValueError("Qwen4 MTP verify must return aligned target HC states")
        predictions = logits.argmax(-1)[0].tolist()
        accepted = 0
        for candidate, expected in zip(draft_tokens, predictions, strict=False):
            if int(candidate) != expected:
                break
            accepted += 1
        committed = inputs[: accepted + 1]
        next_token = int(predictions[accepted])
        if accepted != len(draft_tokens):
            restore(before)
            _, replay_hidden = replay(committed)
            if replay_hidden.ndim != 3 or replay_hidden.shape[0] != 1 or replay_hidden.shape[1] == 0:
                raise ValueError("Qwen4 MTP replay must return accepted-prefix HC states")
            final_hidden = replay_hidden[:, -1:]
        else:
            final_hidden = hidden[:, -1:]
        return MTPVerification(accepted, committed, next_token, final_hidden)
    except BaseException:
        restore(before)
        raise


class MTPGraphRuntime:
    """Draft prefill/decode sessions sharing independent PageKV/QSA state.

    Unified speculative loading should pass ``onnx_graph`` taken from the
    same MultiHMONNXLoader as target prefill/verify, preserving shared head
    initializers. Standalone graph loading is available for Golden/debug.
    """

    def __init__(self, onnx_path, *, cache_contract, context_length, device, onnx_graph=None, golden_dir=None,
                 prefill_path=None, prefill_graph=None, prefill_chunk_length=None, device_map=None, cache_abi=None):
        from xhmodel_merak.exec_backends import HMONNXInferenceConfig, MultiHMONNXLoader
        from xhquant.core.cache_tensor import CacheTensor
        from xhquant.xhonnxruntime.parsers.page_attention import PageKVCache

        from .cache_contract import (
            QWEN4_EXP_LLM_CACHE_ABI,
            QWEN4_EXP_PAGE_CACHE_ABI,
            Qwen4ExpCacheContract,
            qwen4_exp_cache_input_specs,
            resolve_qwen4_exp_cache_abi,
            resolve_qwen4_exp_cache_shape,
        )
        from .host_ple_runtime import Qwen4ExpHostPLEInferenceV2, enable_qwen4_exp_inplace_kv
        from .xhquant_extensions import qwen4_exp_xhquant_extension_scope

        self.device = torch.device(device)
        self.context_length = int(context_length)
        self.contract = Qwen4ExpCacheContract.from_dict(cache_contract)
        devices = [self.device] if device_map is None else list(device_map)
        config = HMONNXInferenceConfig(
            exec_devices=devices, enable_auto_offload=any(torch.device(value).type == "cuda" for value in devices),
            enable_golden=golden_dir is not None,
        )
        if onnx_graph is None:
            loader = MultiHMONNXLoader({"draft": str(onnx_path)})
            onnx_graph = loader.graphs.pop("draft")
        with qwen4_exp_xhquant_extension_scope():
            self.session = Qwen4ExpHostPLEInferenceV2.from_onnx_graph(str(onnx_path), onnx_graph, config)
            self.session.initialize()
        if golden_dir is not None:
            self.session.save_golden_dir = str(golden_dir)
        self.page_modules = [m for m in self.session.graph_module.modules() if isinstance(m, PageKVCache)]
        inferred_abi = QWEN4_EXP_PAGE_CACHE_ABI if self.page_modules else QWEN4_EXP_LLM_CACHE_ABI
        self.cache_abi = resolve_qwen4_exp_cache_abi(cache_abi or inferred_abi)
        if self.cache_abi != inferred_abi:
            raise ValueError("MTP metadata cache ABI differs from its graph")
        if self.page_modules and len(self.page_modules) != 1:
            raise ValueError("Qwen4 MTP requires exactly one full-attention PageKV cache")
        if not self.page_modules:
            expected_kv = {name for name, spec in qwen4_exp_cache_input_specs(self.contract).items()
                           if spec.state_kind in {"kv_key", "kv_value"}}
            if len(expected_kv) != 2 or not expected_kv.issubset(self.session.get_input_names()):
                raise ValueError("MTP LLMCache graph requires explicit paired K/V inputs")
        cache_specs = qwen4_exp_cache_input_specs(self.contract)
        self.states = {}
        self.group_start_position = torch.zeros((3, 1, 1), dtype=torch.int32, device=self.device)
        for name in self.session.get_input_names():
            if name not in cache_specs:
                continue
            spec = cache_specs[name]
            info = self.session.get_input(name)
            shape = resolve_qwen4_exp_cache_shape(spec, batch_size=1, context_length=self.context_length)
            value = torch.zeros(shape, dtype=info.dtype, device=self.device)
            self.states[name] = value if spec.dtype == "integer" else CacheTensor(value)
        floating = self.session.get_input("next_token_embedding").dtype
        blocks = (self.context_length + 63) // 64
        self.page_caches = [
            torch.zeros((2, blocks, m.num_kv_heads, 64, self.contract.head_dim), dtype=floating, device=self.device)
            for m in self.page_modules
        ]
        self.prefill_session = None
        self.prefill_page_modules = []
        self.prefill_chunk_length = 1
        if prefill_path is not None:
            if prefill_graph is None:
                loader = MultiHMONNXLoader({"draft_prefill": str(prefill_path)})
                prefill_graph = loader.graphs.pop("draft_prefill")
            with qwen4_exp_xhquant_extension_scope():
                self.prefill_session = Qwen4ExpHostPLEInferenceV2.from_onnx_graph(
                    str(prefill_path), prefill_graph, config,
                )
                self.prefill_session.initialize()
            self.prefill_page_modules = [
                m for m in self.prefill_session.graph_module.modules() if isinstance(m, PageKVCache)
            ]
            self.prefill_chunk_length = int(self.prefill_session.get_input("next_token_embedding").shape[1])
            if prefill_chunk_length is not None and int(prefill_chunk_length) != self.prefill_chunk_length:
                raise ValueError("MTP prefill width differs from metadata")
            if not self.page_modules and not expected_kv.issubset(self.prefill_session.get_input_names()):
                raise ValueError("MTP prefill/decode explicit cache layouts differ")
            if len(self.prefill_page_modules) != len(self.page_modules):
                raise ValueError("MTP prefill/decode PageKV layouts differ")
            for first, second in zip(self.prefill_page_modules, self.page_modules, strict=True):
                if first.num_kv_heads != second.num_kv_heads:
                    raise ValueError("MTP prefill/decode head layouts differ")
        if self.cache_abi == QWEN4_EXP_LLM_CACHE_ABI:
            for session in (self.session, self.prefill_session):
                if session is not None:
                    enable_qwen4_exp_inplace_kv(session, expected_kv)
        self.past_length = 0

    def close(self):
        """Release draft-owned state and session references after a request."""
        self.states.clear()
        self.page_caches.clear()
        self.page_modules.clear()
        self.prefill_page_modules.clear()
        for session in (self.session, self.prefill_session):
            close = getattr(session, "close", None)
            if callable(close):
                close()
        self.session = self.prefill_session = None

    def reset(self):
        for state in (*self.states.values(), *self.page_caches):
            state.zero_()
        self.group_start_position.zero_()
        self.past_length = 0

    def snapshot(self, *, write_length=None):
        """Save mutable graph state and the current partial C4 position."""
        from .cache_contract import qwen4_exp_cache_input_specs
        from .mtp_snapshot import snapshot_state

        width = self.context_length if write_length is None else write_length
        snapshot = snapshot_state(
            self.states, self.page_caches, specs={**{spec.name: spec for spec in self.contract.tensors},
                                                  **qwen4_exp_cache_input_specs(self.contract)},
            past_length=0 if write_length is None else self.past_length, write_length=width,
        )
        return self.past_length, snapshot, self.group_start_position.clone()

    def restore(self, saved):
        past, snapshot, group_start_position = saved
        snapshot.restore(self.states, self.page_caches)
        self.group_start_position.copy_(group_start_position)
        self.past_length = int(past)

    @torch.inference_mode()
    def prefill(self, next_token_embedding, target_hidden, *, position_ids, capture=None):
        """Consume shifted prompt pairs in real blocks, preserving shared draft state."""
        count = int(next_token_embedding.shape[1])
        if next_token_embedding.shape[0] != 1 or target_hidden.shape[:2] != (1, count):
            raise ValueError("MTP prefill embeddings and HC states must align")
        if tuple(position_ids.shape) != (3, 1, count):
            raise ValueError("MTP prefill positions must have shape [3, 1, tokens]")
        if self.past_length + count > self.context_length:
            raise ValueError("MTP context capacity exceeded")
        result = None
        offset = 0
        while offset < count:
            use_prefill = self.prefill_session is not None and (
                self.past_length + self.prefill_chunk_length <= self.context_length
            )
            width = min(self.prefill_chunk_length if use_prefill else 1, count - offset)
            result = self._run(
                next_token_embedding[:, offset:offset + width], target_hidden[:, offset:offset + width],
                position_ids=position_ids[:, :, offset:offset + width], capture=capture,
                prefill=use_prefill,
            )
            offset += width
        return result

    @torch.inference_mode()
    def step(self, next_token_embedding, target_hidden, *, position_ids, capture=None):
        return self._run(next_token_embedding, target_hidden, position_ids=position_ids, capture=capture)

    def _run(self, next_token_embedding, target_hidden, *, position_ids, capture=None, prefill=False):
        from xhquant.xhonnxruntime.parsers.page_attention import PageAttentionContext

        session = self.prefill_session if prefill else self.session
        modules = self.prefill_page_modules if prefill else self.page_modules
        width = self.prefill_chunk_length if prefill else 1
        valid = int(next_token_embedding.shape[1])
        if not 0 < valid <= width or next_token_embedding.shape[0] != 1 or target_hidden.shape[:2] != (1, valid):
            raise ValueError("Qwen4 MTP requires aligned batch-one embeddings and HC states")
        if tuple(position_ids.shape) != (3, 1, valid):
            raise ValueError("MTP positions must contain three axes for each batch-one token")
        if self.past_length + width > self.context_length:
            raise ValueError("MTP context capacity exceeded")
        total = self.past_length + valid
        from ...mrope import MROPE_CURRENT_POSITION_NAMES, MROPE_POOL_POSITION_NAMES, prepare_mrope_position_inputs

        current_positions, pool_positions, next_group_start = prepare_mrope_position_inputs(
            self.group_start_position, position_ids, self.past_length, width, valid,
        )
        self.group_start_position.copy_(next_group_start)
        slots = torch.full((width,), -1, dtype=torch.int64, device=self.device)
        slots[:valid] = torch.arange(self.past_length, total, device=self.device)
        blocks = torch.arange((total + 63) // 64, dtype=torch.int64, device=self.device)
        for module, storage in zip(modules, self.page_caches, strict=True):
            module.set_context(PageAttentionContext(
                paged_kv_cache=storage, block_ids=blocks, slot_mapping=slots,
                block_size=64, effective_kv_length=total,
            ))

        def padded(value, name):
            value = value.to(device=self.device, dtype=session.get_input(name).dtype)
            return torch.nn.functional.pad(value, (0, 0, 0, width - valid))

        feed = dict(self.states)
        feed.update(
            next_token_embedding=padded(next_token_embedding, "next_token_embedding"),
            target_hidden=padded(target_hidden, "target_hidden"),
            past_seq_length=torch.tensor([self.past_length], dtype=torch.int32, device=self.device),
            current_input_length=torch.tensor([valid], dtype=torch.int32, device=self.device),
        )
        feed.update(zip(MROPE_CURRENT_POSITION_NAMES, current_positions.unbind(0), strict=True))
        feed.update(zip(MROPE_POOL_POSITION_NAMES, pool_positions.unbind(0), strict=True))
        from .cache_contract import qwen4_exp_cache_input_specs
        from .qsa_host_controls import QSA_HOST_CONTROL_NAMES, build_qsa_host_controls, qsa_host_control_spec

        qsa_spec = qsa_host_control_spec(session)
        if qsa_spec is not None:
            candidate_topk, mask_dtype = qsa_spec
            controls = build_qsa_host_controls(
                self.past_length, valid, width, self.context_length, candidate_topk,
                device=self.device, dtype=mask_dtype,
            )
            feed.update(zip(QSA_HOST_CONTROL_NAMES, controls, strict=True))
        runtime_state = ChainMap(
            ({**{f"page_kv_{i}": cache for i, cache in enumerate(self.page_caches)},
              "block_ids": blocks, "slot_mapping": slots} if self.page_caches else {}), self.states
        )
        scope = capture.step(
            "mtp", session, graph_name="mtp_prefill" if prefill else "mtp_decode", runtime_state=runtime_state,
        ) if capture is not None else nullcontext()
        with scope:
            outputs = session.run(feed)
        values = dict(zip(session.get_output_names(), outputs, strict=True))
        for name, spec in qwen4_exp_cache_input_specs(self.contract).items():
            if spec.state_kind in {"qsa_indexer_tail", "qsa_key_tail", "qsa_value_tail"} and name in self.states:
                self.states[name].copy_(values[f"present_{name}"])
        self.past_length = total
        return values["logits"][:, :valid], values["mtp_hidden_states"][:, :valid]
