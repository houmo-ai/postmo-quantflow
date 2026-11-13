"""Qwen4-Exp HMONNX runtime for serving-oriented exports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from xhquant.core import CacheTensor

from ...hmonnx import TextLLMHMONNXModel
from ...llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor
from ...types import CacheList, LLMModelMeta
from .cache_contract import (
    QWEN4_EXP_LLM_CACHE_ABI,
    Qwen4ExpCacheContract,
    Qwen4ExpTensorSpec,
    resolve_qwen4_exp_cache_abi,
    resolve_qwen4_exp_cache_shape,
)
from .host_ple_runtime import (
    Qwen4ExpServingGraphModel,
    enable_qwen4_exp_inplace_kv,
    qwen4_exp_execution_backend,
)


def _dtype_for_spec(spec: Qwen4ExpTensorSpec, floating_dtype: torch.dtype) -> torch.dtype:
    return floating_dtype if spec.dtype == "floating" else torch.int32


def _is_external_page_kv(spec: Qwen4ExpTensorSpec) -> bool:
    return spec.state_kind in {"kv_key", "kv_value"}


class Qwen4ExpServingState:
    """Own explicit Qwen4 states, retaining legacy external PageKV support."""

    def __init__(
        self,
        contract: Qwen4ExpCacheContract,
        *,
        batch_size: int,
        context_length: int,
        dtype: torch.dtype,
        device: torch.device | str,
        ple_eos_token_id: int = 0,
        own_kv: bool = False,
    ) -> None:
        self.contract = contract
        self.batch_size = int(batch_size)
        self.context_length = int(context_length)
        self.dtype = dtype
        self.device = torch.device(device)
        self.ple_eos_token_id = int(ple_eos_token_id)
        self.own_kv = bool(own_kv)
        self.tensors: dict[str, torch.Tensor] = {}
        self.reset()

    def reset(self) -> None:
        tensors: dict[str, torch.Tensor] = {}
        for spec in self.contract.tensors:
            if _is_external_page_kv(spec) and not self.own_kv:
                continue
            shape = resolve_qwen4_exp_cache_shape(spec, batch_size=self.batch_size, context_length=self.context_length)
            dtype = _dtype_for_spec(spec, self.dtype)
            if spec.state_kind in {"qsa_indexer_key", "delta_recurrent", "kv_key", "kv_value"}:
                tensors[spec.name] = CacheTensor(torch.zeros(shape, dtype=dtype, device=self.device))
            elif spec.state_kind == "ple_ngram_history":
                tensors[spec.name] = torch.full(
                    shape,
                    self.ple_eos_token_id,
                    dtype=dtype,
                    device=self.device,
                )
            else:
                tensors[spec.name] = torch.zeros(shape, dtype=dtype, device=self.device)
        self.tensors = tensors
        self.group_start_position = torch.zeros((3, self.batch_size, 1), dtype=torch.int32, device=self.device)

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Qwen4ExpServingState":
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        for name, tensor in tuple(self.tensors.items()):
            target_dtype = dtype if tensor.is_floating_point() else None
            self.tensors[name] = tensor.to(device=self.device, dtype=target_dtype)
        self.group_start_position = self.group_start_position.to(device=self.device)
        return self

    def inputs_in_contract_order(
        self,
        *,
        past_key_caches: CacheList | None,
        past_value_caches: CacheList | None,
        enable_page_attention: bool,
    ) -> tuple[torch.Tensor, ...]:
        values: list[torch.Tensor] = []
        key_index = 0
        value_index = 0
        for spec in self.contract.tensors:
            if spec.state_kind in {"kv_key", "kv_value"} and self.own_kv:
                values.append(self.tensors[spec.name])
            elif spec.state_kind == "kv_key":
                if enable_page_attention:
                    continue
                if past_key_caches is None:
                    raise ValueError("past_key_caches are required when PageAttention is disabled")
                values.append(past_key_caches[key_index])
                key_index += 1
            elif spec.state_kind == "kv_value":
                if enable_page_attention:
                    continue
                if past_value_caches is None:
                    raise ValueError("past_value_caches are required when PageAttention is disabled")
                values.append(past_value_caches[value_index])
                value_index += 1
            else:
                values.append(self.tensors[spec.name])
        return tuple(values)

    def commit_outputs(self, outputs: tuple[Any, ...], output_names: tuple[str, ...]) -> torch.Tensor:
        if not outputs:
            raise RuntimeError("Qwen4-Exp HMONNX returned no outputs")
        for name, value in zip(output_names[1:], outputs[1:], strict=False):
            state_name = str(name).removeprefix("present_")
            if state_name in self.tensors and torch.is_tensor(value):
                spec = next(spec for spec in self.contract.tensors if spec.name == state_name)
                if spec.state_kind in {"qsa_indexer_key", "kv_key", "kv_value"}:
                    # LLMCache mutates the caller-owned full-capacity
                    # CacheTensor and returns only its effective prefix.
                    continue
                if spec.state_kind == "delta_recurrent":
                    # Fused GDR mutates this object; the unfused graph returns
                    # a new tensor. Keep the caller-owned CacheTensor in both.
                    self.tensors[state_name].copy_(value)
                    continue
                self.tensors[state_name] = value
        return outputs[0]


class Qwen4ExpServingDataPreprocess(BaseLLMInputProcessor):
    """Prepare Qwen4 text, position, PLE token and internal state tensors."""

    def __init__(
        self,
        *,
        token_embedding: nn.Embedding,
        input_sequence_length: int,
        state: Qwen4ExpServingState,
        past_key_caches: CacheList | None,
        past_value_caches: CacheList | None,
        enable_page_attention: bool,
        pad_token_id: int,
        external_ple=None,
        qsa_control_spec=None,
    ) -> None:
        super().__init__(
            BaseInputProcessorConfig(
                token_embedding,
                input_sequence_length,
                past_key_caches,
                past_value_caches,
                enable_page_attention=enable_page_attention,
                pad_token_id=pad_token_id,
            )
        )
        self.state = state
        self.external_ple = external_ple
        self.qsa_control_spec = qsa_control_spec

    def forward(self, data: dict | tuple | list):
        if not isinstance(data, Mapping):
            raise TypeError(f"Qwen4ExpServingDataPreprocess expects dict input, got {type(data)}")
        input_ids = data.get("input_ids")
        inputs_embeds = data.get("inputs_embeds")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            if input_ids.shape[0] != 1:
                raise ValueError("Qwen4-Exp serving runtime currently supports batch size 1")
            seq_length = int(input_ids.shape[1])
            if seq_length < self.input_sequence_length:
                padding = torch.full(
                    (1, self.input_sequence_length - seq_length),
                    self.pad_token_id,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                input_ids = torch.cat([input_ids, padding], dim=-1)
            ple_input_ids = input_ids
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            if inputs_embeds.shape[0] != 1:
                raise ValueError("Qwen4-Exp serving runtime currently supports batch size 1")
            seq_length = int(inputs_embeds.shape[1])
            ple_input_ids = data.get("ple_input_ids")
            if ple_input_ids is None:
                raise ValueError("ple_input_ids are required when inputs_embeds are supplied")
            if seq_length < self.input_sequence_length:
                padding_ids = torch.full(
                    (1, self.input_sequence_length - seq_length),
                    self.pad_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                inputs_embeds = torch.cat([inputs_embeds, self.embed_tokens(padding_ids)], dim=1)
                ple_input_ids = torch.cat([ple_input_ids, padding_ids], dim=1)
        if seq_length > self.input_sequence_length:
            raise ValueError(f"Input sequence length {seq_length} exceeds graph length {self.input_sequence_length}")

        current_input_length = int(data.get("current_input_length", seq_length))
        if current_input_length <= 0 or current_input_length > seq_length:
            raise ValueError(
                "current_input_length must be positive and no greater than the supplied sequence length, "
                f"got {current_input_length} for sequence length {seq_length}"
            )
        past_seq_length = int(data.get("past_seq_length", 0))
        if past_seq_length < 0:
            raise ValueError("past_seq_length must be non-negative")
        positions = data.get("position_ids")
        if positions is None:
            positions = torch.arange(past_seq_length, past_seq_length + seq_length).view(1, 1, -1).expand(3, 1, -1)
        if tuple(positions.shape) != (3, 1, seq_length):
            raise ValueError("Qwen4 position_ids must have shape [3, 1, supplied_sequence_length]")
        if past_seq_length + current_input_length > self.state.context_length:
            raise ValueError("Valid graph positions exceed context capacity")
        from ...mrope import prepare_mrope_position_inputs

        current_positions, pool_positions, next_group_start = prepare_mrope_position_inputs(
            self.state.group_start_position, positions, past_seq_length, self.input_sequence_length,
            current_input_length,
        )
        self.next_group_start_position = next_group_start
        qsa_controls = ()
        if self.qsa_control_spec is not None:
            from .qsa_host_controls import build_qsa_host_controls

            candidate_topk, mask_dtype = self.qsa_control_spec
            qsa_controls = build_qsa_host_controls(
                past_seq_length, current_input_length, self.input_sequence_length,
                self.state.context_length, candidate_topk, device=self._device, dtype=mask_dtype,
            )
        return (
            inputs_embeds.to(self._device),
            torch.tensor([past_seq_length], dtype=torch.int32, device=self._device),
            torch.tensor([current_input_length], dtype=torch.int32, device=self._device),
            ple_input_ids.to(self._device, dtype=torch.int32),
            *current_positions.to(self._device).unbind(0),
            *pool_positions.to(self._device).unbind(0),
            *self.state.inputs_in_contract_order(
                past_key_caches=self.past_key_caches,
                past_value_caches=self.past_value_caches,
                enable_page_attention=self.enable_page_attention,
            ),
            *qsa_controls,
            *(self.external_ple.prepare(
                ple_input_ids, self.state.tensors, current_input_length, device=self._device,
            ) if self.external_ple is not None else ()),
        )


class XHQwen4ExpHMONNXModel(TextLLMHMONNXModel):
    """Serving runtime for explicit LLMCache and legacy PageKV packages."""

    def __init__(self, meta_info: LLMModelMeta, **kwargs):
        self.qwen4_cache_contract = Qwen4ExpCacheContract.from_dict(meta_info.qwen4_exp_cache_contract)
        self.qwen4_cache_abi = resolve_qwen4_exp_cache_abi(getattr(meta_info, "qwen4_exp_cache_abi", None))
        if self.qwen4_cache_abi == QWEN4_EXP_LLM_CACHE_ABI:
            kwargs["enable_page_attention"] = False
        self._qwen4_spec = getattr(meta_info, "spec_decode", None)
        self._qwen4_mtp_graph = None
        self._qwen4_mtp_prefill_graph = None
        self._qwen4_mtp_runtime = None
        self.last_mtp_stats = None
        self._verify_active = False
        self.verify_model = None
        super().__init__(meta_info, **kwargs)
        self.embed_tokens.to(self.device)
        if getattr(self, "_qwen4_shared_verify", False):
            self.verify_model = self.decode_model
        if getattr(self, "_qwen4_verify_graph", None) is not None:
            self.verify_model = Qwen4ExpServingGraphModel(
                self._qwen4_verify_path, onnx_graph=self._qwen4_verify_graph,
                enable_cuda_graph=False, enable_auto_offload=kwargs.get("enable_auto_offload", False),
                device_map=self._valid_devices, backend=self._backend,
            )
            self._qwen4_verify_graph = None
        model_config = meta_info.model_config
        self._qwen4_state = Qwen4ExpServingState(
            self.qwen4_cache_contract,
            batch_size=int(model_config.batch_size),
            context_length=int(model_config.context_max_length),
            dtype=self.dtype,
            device=self.device,
            ple_eos_token_id=int(meta_info.pad_token_id),
            own_kv=self.qwen4_cache_abi == QWEN4_EXP_LLM_CACHE_ABI,
        )
        if self.qwen4_cache_abi == QWEN4_EXP_LLM_CACHE_ABI:
            from .cache_contract import qwen4_exp_cache_input_specs

            required = {name for name, spec in qwen4_exp_cache_input_specs(self.qwen4_cache_contract).items()
                        if spec.state_kind in {"kv_key", "kv_value"}}
            for graph in (self.prefill_model, self.decode_model, self.verify_model):
                if graph is None:
                    continue
                session = graph.hmonnx_session
                if self._get_page_attention_modules(graph):
                    raise ValueError("Qwen4 explicit cache ABI cannot contain PageKVCache writers")
                if not required.issubset(session.get_input_names()):
                    raise ValueError("Qwen4 explicit cache graph lacks paired K/V inputs")
                enable_qwen4_exp_inplace_kv(session, required)
        self._data_processor = None
        self._processor = None
        self.vision_encoder = None
        visual = getattr(meta_info, "qwen4_exp_vision", None) or getattr(meta_info, "visual_config", None)
        if visual is not None:
            from pathlib import Path

            from .vision import load_qwen4_exp_vision

            root = Path(getattr(meta_info, "_meta_path_", Path(meta_info.hf_config).parent / "meta.json")).parent
            self.vision_encoder = load_qwen4_exp_vision(visual, root, device=self.device)

    def get_processor(self):
        if self._processor is None:
            from .multimodal import create_qwen4_exp_processor

            capacity = self.vision_encoder.max_image_tokens if self.vision_encoder is not None else 1536
            self._processor = create_qwen4_exp_processor(self.hf_model_dir, capacity)
        return self._processor

    @torch.inference_mode()
    def generate(
        self, input_ids, max_new_tokens=20, do_sample=False, pad_token_id=None,
        eos_token_id=None, streamer=None, attention_mask=None, inputs_embeds=None,
        position_ids=None, pixel_values=None, image_grid_thw=None, pixel_values_videos=None,
        video_grid_thw=None, mm_token_type_ids=None, rope_deltas=None, _capture=None, **kwargs,
    ):
        """Generate one unpadded greedy request using the serving cache ABI."""
        from contextlib import nullcontext

        from .generation import create_mtp_runtime, generate_token_ids, resolve_eos_token_ids

        draft = None
        self.last_mtp_stats = None
        encoder = getattr(self, "vision_encoder", None)
        previous_capture = getattr(encoder, "capture", None)
        try:
            if kwargs:
                raise ValueError(f"Unsupported Qwen4 generation arguments: {sorted(kwargs)}")
            if do_sample:
                raise ValueError("Qwen4 serving generation supports greedy decoding only")
            if not isinstance(input_ids, torch.Tensor) or input_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("input_ids must be an int32 or int64 tensor")
            if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
                raise ValueError("Qwen4 generation requires a nonempty batch-one token sequence")
            if attention_mask is not None and (
                attention_mask.shape != input_ids.shape or not bool(torch.all(attention_mask == 1))
            ):
                raise ValueError("Qwen4 generation requires an unpadded attention_mask of all ones")
            if inputs_embeds is not None and (
                inputs_embeds.ndim != 3 or inputs_embeds.shape[:2] != input_ids.shape
            ):
                raise ValueError("inputs_embeds must match the input_ids batch and sequence")
            media = any(value is not None for value in (
                pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw,
            ))
            if media and inputs_embeds is not None:
                raise ValueError("raw visual inputs and inputs_embeds are mutually exclusive")
            if media:
                from transformers import AutoConfig

                from .multimodal import fuse_multimodal_embeddings, multimodal_rope_positions

                config = AutoConfig.from_pretrained(self.hf_model_dir)
                native_positions, _ = multimodal_rope_positions(
                    config, input_ids, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                    mm_token_type_ids=mm_token_type_ids, attention_mask=attention_mask,
                )
                if (position_ids is not None
                        and not torch.equal(position_ids.to(native_positions.device), native_positions)):
                    raise ValueError("position_ids disagree with native visual positions")
                position_ids = native_positions
                if self.vision_encoder is not None:
                    self.vision_encoder.capture = _capture
                inputs_embeds = fuse_multimodal_embeddings(
                    input_ids, self.embed_tokens, self.vision_encoder, config,
                    pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                    pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw,
                )
            # pad_token_id is accepted for caller compatibility; padding is never
            # inferred from token values because pad and EOS can share an ID.
            eos_ids = resolve_eos_token_ids(self, eos_token_id)
            context = torch.cuda.device(self.device) if torch.device(self.device).type == "cuda" else nullcontext()
            with context:
                draft = create_mtp_runtime(self, capture=_capture)
                if streamer is not None:
                    streamer.put(input_ids.detach().cpu())
                output = generate_token_ids(
                    self, input_ids.to(device=self.device, dtype=torch.long),
                    max_new_tokens=max_new_tokens, eos_token_ids=eos_ids,
                    inputs_embeds=None if inputs_embeds is None else inputs_embeds.to(self.device),
                    position_ids=None if position_ids is None else position_ids.to(self.device),
                    mtp_runtime=draft, capture=_capture,
                )
                generated = input_ids.new_tensor(output).reshape(1, -1)
                if streamer is not None and generated.numel():
                    streamer.put(generated.detach().cpu())
                return torch.cat((input_ids, generated), dim=1)
        finally:
            if encoder is not None:
                encoder.capture = previous_capture
            try:
                if draft is not None:
                    draft.reset()
            finally:
                if streamer is not None:
                    streamer.end()

    def close(self):
        """Release the model-owned draft sessions; repeated calls are harmless."""
        draft = getattr(self, "_qwen4_mtp_runtime", None)
        self._qwen4_mtp_runtime = None
        if draft is not None:
            draft.close()

    def _load_hmonnx_graphs(self, prefill_hmonnx, decode_hmonnx):
        self._backend = qwen4_exp_execution_backend(self._backend)
        from pathlib import Path

        from .host_ple_inputs import HostPLEInputs

        prefill_ple = HostPLEInputs.from_graph(prefill_hmonnx)
        shared_ple = (Path(prefill_hmonnx).parent.parent / "ple" / "manifest.json").is_file()
        self.external_ple = {
            "prefill": prefill_ple,
            "decode": prefill_ple if shared_ple else HostPLEInputs.from_graph(decode_hmonnx),
        }
        spec = self._qwen4_spec
        if spec is None or spec.get("mode") != "mtp":
            return super()._load_hmonnx_graphs(prefill_hmonnx, decode_hmonnx)
        from xhmodel_merak.exec_backends import MultiHMONNXLoader

        draft_path = Path(spec["draft"]["decode_hmonnx"])
        if not draft_path.is_absolute():
            draft_path = Path(prefill_hmonnx).parent.parent / draft_path
        paths = {"prefill": str(prefill_hmonnx), "decode": str(decode_hmonnx), "mtp": str(draft_path)}
        if int(spec.get("runtime_contract_version", 2)) >= 3:
            from .mtp_contract import mtp_verify_phase, resolve_mtp_verify_length

            root = Path(prefill_hmonnx).parent.parent
            paths["mtp_prefill"] = str(root / spec["draft"]["prefill_hmonnx"])
            self._qwen4_verify_path = str(root / spec["target"]["verify_hmonnx"])
            default_count = 1 if spec.get("runtime_contract_version", 2) <= 3 else 4
            self._qwen4_verify_length = resolve_mtp_verify_length(
                {"verify_length": spec["target"]["verify_chunk_length"],
                 "num_draft_tokens": spec.get("num_draft_tokens", default_count)}
            )
            self._qwen4_verify_phase = spec.get("target_verify_graph", mtp_verify_phase(
                self._qwen4_verify_length, runtime_version=spec.get("runtime_contract_version", 2),
            ))
            if self._qwen4_verify_phase != mtp_verify_phase(
                self._qwen4_verify_length, runtime_version=spec.get("runtime_contract_version", 2),
            ):
                raise ValueError("Qwen4 target verify role disagrees with its graph length")
            self._qwen4_shared_verify = self._qwen4_verify_path == paths["decode"]
            if not self._qwen4_shared_verify:
                paths["verify"] = self._qwen4_verify_path
                self.external_ple[self._qwen4_verify_phase] = HostPLEInputs.from_graph(self._qwen4_verify_path)
        loader = MultiHMONNXLoader(paths)
        self._qwen4_mtp_graph = loader.graphs.pop("mtp")
        self._qwen4_mtp_prefill_graph = loader.graphs.pop("mtp_prefill", None)
        self._qwen4_verify_graph = loader.graphs.pop("verify", None)
        return loader.graphs["prefill"], loader.graphs["decode"], loader.initializer_pool.summary()

    def active_graph(self):
        if getattr(self, "_verify_active", False):
            return self.verify_model
        return self.prefill_model if self.is_prefill() else self.decode_model

    def active_component(self):
        if getattr(self, "_verify_active", False):
            from .mtp_contract import mtp_verify_phase

            return getattr(self, "_qwen4_verify_phase", mtp_verify_phase(self._qwen4_verify_length))
        return "prefill" if self.is_prefill() else "decode"

    def set_prefill(self):
        self._verify_active = False
        super().set_prefill()

    def set_decode(self):
        self._verify_active = False
        super().set_decode()
        if getattr(self, "_qwen4_shared_verify", False):
            self.set_input_sequence_length(self._qwen4_verify_length)

    def set_verify(self):
        if self.verify_model is None:
            raise ValueError("Package has no target verify graph")
        self._llm_prefill = False
        self._verify_active = True
        self.set_input_sequence_length(self._qwen4_verify_length)

    def get_input_sequence_length(self):
        if getattr(self, "_verify_active", False) or (
            not self.is_prefill() and getattr(self, "_qwen4_shared_verify", False)
        ):
            return self._qwen4_verify_length
        return super().get_input_sequence_length()

    def reset_qwen4_exp_state(self) -> None:
        self._qwen4_state.reset()

    def reset_qwen4_state(self) -> None:
        self.reset_qwen4_exp_state()

    def reset_internal_state(self) -> None:
        self.reset_qwen4_exp_state()
        self._data_processor = None
        self._verify_active = False

    @staticmethod
    def _prepare_hmonnx_paths(
        prefill_hmonnx,
        decode_hmonnx,
        *,
        enable_parallel_linear_fusion: bool,
        enable_page_attention: bool,
        convert_unfused_sliding_kv_cache: bool = False,
    ) -> tuple[str, str]:
        del convert_unfused_sliding_kv_cache
        if enable_parallel_linear_fusion:
            from xhquant.xhonnxruntime.hmonnx_optimizer import materialize_parallel_linear_fusion

            prefill_hmonnx = materialize_parallel_linear_fusion(prefill_hmonnx)
            decode_hmonnx = materialize_parallel_linear_fusion(decode_hmonnx)
        # The serving exporter has already converted only the physical K/V
        # writers.  Running the generic fused PageAttention rewrite here would
        # discard Qwen4's query-dependent TopK gather.
        return str(prefill_hmonnx), str(decode_hmonnx)

    def set_page_attention_context(
        self,
        paged_kv_caches,
        block_ids,
        slot_mapping,
        block_size,
        effective_kv_length=None,
    ):
        # PageKVCache keeps FP tensors, unlike the generic quantized PageAttention
        # allocator/binder which expects HMFPPagedKVCache objects.
        from xhquant.xhonnxruntime.parsers.page_attention import PageAttentionContext, PageKVCache

        graph = self.active_graph() if hasattr(self, "active_graph") else (
            self.prefill_model if self.is_prefill() else self.decode_model
        )
        modules = self._get_page_attention_modules(graph)
        if len(modules) != len(paged_kv_caches):
            raise ValueError("Qwen4 PageKV module/cache count mismatch")
        for module, cache in zip(modules, paged_kv_caches, strict=True):
            if not isinstance(module, PageKVCache):
                raise TypeError("Qwen4 serving requires PageKVCache graph writers")
            if not torch.is_tensor(cache) or cache.ndim != 5 or cache.shape[0] != 2:
                raise ValueError("Qwen4 PageKV cache must have shape [2, blocks, heads, block, dim]")
            module.set_context(
                PageAttentionContext(
                    paged_kv_cache=cache,
                    block_ids=torch.as_tensor(block_ids, dtype=torch.int64, device=cache.device),
                    slot_mapping=torch.as_tensor(slot_mapping, dtype=torch.int64, device=cache.device),
                    block_size=int(block_size),
                    effective_kv_length=effective_kv_length,
                )
            )

    def get_data_preprocessor(self) -> Qwen4ExpServingDataPreprocess:
        processor = super().get_data_preprocessor()
        processor.external_ple = getattr(self, "external_ple", {}).get(self.active_component())
        from .qsa_host_controls import qsa_host_control_spec

        processor.qsa_control_spec = qsa_host_control_spec(self.active_graph().hmonnx_session)
        return processor

    def _get_data_preprocessor(self) -> Qwen4ExpServingDataPreprocess:
        return Qwen4ExpServingDataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.get_input_sequence_length(),
            state=self._qwen4_state,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            enable_page_attention=self.enable_page_attention,
            pad_token_id=self.pad_token_id,
            external_ple=getattr(self, "external_ple", {}).get(self.active_component()),
        )

    def _set_device(self, device):
        draft = getattr(self, "_qwen4_mtp_runtime", None)
        if draft is not None and torch.device(device) != draft.device:
            raise ValueError(f"Qwen4 MTP sessions are fixed on device {draft.device}; reload to change device")
        super()._set_device(device)
        if hasattr(self, "_qwen4_state"):
            self._qwen4_state.to(device=device)
        encoder = getattr(self, "vision_encoder", None)
        if encoder is not None:
            encoder.to(device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        if hasattr(self, "_qwen4_state"):
            self._qwen4_state.to(dtype=dtype)
        return self

    def forward(self, *args):
        outputs = self.verify_model.forward(*args) if getattr(self, "_verify_active", False) else super().forward(*args)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        session = self.active_graph().hmonnx_session
        output_names = tuple(session.get_output_names())
        if "target_hidden_states" in output_names:
            self.last_hidden_states = outputs[output_names.index("target_hidden_states")]
        logits = self._qwen4_state.commit_outputs(outputs, output_names)
        processor = getattr(self, "_data_processor", None)
        pending = getattr(processor, "next_group_start_position", None)
        if pending is not None:
            self._qwen4_state.group_start_position.copy_(pending)
            processor.next_group_start_position = None
        return logits


__all__ = [
    "Qwen4ExpServingDataPreprocess",
    "Qwen4ExpServingState",
    "XHQwen4ExpHMONNXModel",
]
