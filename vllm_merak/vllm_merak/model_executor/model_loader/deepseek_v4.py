# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 Flash loader for the heterogeneous HMONNX cache ABI."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Optional, cast

import torch
from torch import nn
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.v1.sample.sampler import Sampler
from xhmodel_merak.xh_llm import AutoLLMHONNXModel
from xhmodel_merak.xh_llm.models.deepseek_v4 import XHDeepSeekV4HMONNXModel

from ...attention.backends.merak import MerakAttentionBackend
from ...config import MerakConfig
from ...constants import HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST
from ...utils import (
    apply_hmonnx_runtime_options,
    get_hmonnx_runtime_kwargs,
    move_hmonnx_auxiliary_tensors,
    resolve_hmonnx_primary_device,
    resolve_torch_dtype,
    should_preserve_hmonnx_runtime_device_map,
)
from ..register import MERAK_MODEL_REGISTRY
from .hf_assets import get_model_trust_remote_code, load_hf_config
from .merak_for_causal_lm import MerakAttention, MerakForCausalLM

logger = init_logger(__name__)


@MERAK_MODEL_REGISTRY.register_module("DeepseekV4ForCausalLM")
class DeepseekV4MerakForCausalLM(MerakForCausalLM):
    """Run DeepSeek-V4 through its request-local SWA/CSA/HCA cache.

    DeepSeek-V4 Flash does not expose ordinary K/V tensors per attention
    layer.  Its HMONNX runtime owns sliding, compressed-sparse, and heavily
    compressed states together, so vLLM must schedule one request at a time
    while the runtime retains the cache between calls.
    """

    hybrid_cache_policy = HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST
    uses_internal_hmonnx_cache = True
    supports_multimodal = False
    supports_multimodal_raw_input_only = False
    requires_raw_input_tokens = True
    supports_mrope = False
    _prefer_input_ids_runtime = True

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        kv_cache_dtype: str = "auto",
        *,
        vllm_config: Optional[VllmConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if vllm_config is not None:
            model_config = vllm_config.model_config
        if model_config is None:
            raise ValueError(
                "DeepseekV4MerakForCausalLM requires either model_config or vllm_config."
            )

        self.vllm_config = vllm_config
        self.prefix = prefix
        config = model_config.hf_config
        if not isinstance(config, MerakConfig):
            raise TypeError(
                "DeepseekV4MerakForCausalLM expects a MerakConfig as hf_config, "
                "got "
                f"{type(config).__name__}. Did you forget to set "
                "config_format='merak_llm' on the LLM?"
            )
        self.config: MerakConfig = config
        if bool(getattr(config, "enable_page_attention", False)):
            raise ValueError(
                "DeepSeek-V4 Flash owns SWA/CSA/HCA state in HMONNX and does not "
                "support PageAttention."
            )

        max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1))
        if max_num_seqs != 1:
            raise ValueError(
                "DeepSeek-V4 Flash Merak runtime requires max_num_seqs=1 because "
                "its heterogeneous HMONNX cache is request-local."
            )
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "DeepSeek-V4 Flash Merak runtime does not support speculative "
                "decoding because SWA/CSA/HCA state has no rollback contract."
            )
        normalized_cache_dtype = str(kv_cache_dtype).lower()
        if normalized_cache_dtype not in {"auto", "float16", "fp16", "torch.float16"}:
            raise ValueError(
                "DeepSeek-V4 Flash HMONNX cache is exported as float16; "
                f"got kv_cache_dtype={kv_cache_dtype!r}."
            )

        model_path = getattr(model_config, "model", None)
        name_or_path = getattr(config, "_name_or_path", None)
        config_dir_candidates = [
            Path(path) for path in (model_path, name_or_path) if path
        ]
        config_dir = next(
            (candidate for candidate in config_dir_candidates if candidate.is_dir()),
            Path(name_or_path or model_path or "."),
        )
        hf_config_dir = config_dir / "hf_config"
        hf_config_file = hf_config_dir / "config.json"
        hf_config = json.loads(hf_config_file.read_text(encoding="utf-8"))
        architectures = hf_config.get("architectures") or []
        if not architectures or str(architectures[0]) != "DeepseekV4ForCausalLM":
            raise ValueError(
                "DeepSeek-V4 Merak loader requires hf_config/config.json with "
                f"architectures=['DeepseekV4ForCausalLM'], got {architectures!r}."
            )

        self.device = torch.device("cpu")
        self.inference_device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        xh_model = config.xh_model
        if (
            not isinstance(xh_model, dict)
            or str(xh_model.get("model_type", "")).lower() != "hmonnx"
        ):
            raise ValueError(
                "DeepSeek-V4 Merak loader requires xh_model.model_type='hmonnx'."
            )

        hmonnx_meta_file = str(config_dir / "golden_meta_info.json")
        hmonnx_model = AutoLLMHONNXModel.from_pretrained(
            hmonnx_meta_file,
            **get_hmonnx_runtime_kwargs(config),
        )
        if not isinstance(hmonnx_model, XHDeepSeekV4HMONNXModel):
            raise TypeError(
                "DeepSeek-V4 Merak loader expected XHDeepSeekV4HMONNXModel, got "
                f"{type(hmonnx_model).__name__}."
            )
        if int(hmonnx_model.cache_abi.batch_size) != 1:
            raise ValueError("DeepSeek-V4 Merak runtime requires an exported batch size of one.")
        self.prefill_chunk_length = int(
            hmonnx_model.cache_abi.spec.prefill_chunk_length
        )
        if self.prefill_chunk_length <= 0:
            raise ValueError("DeepSeek-V4 HMONNX prefill chunk length must be positive.")
        exported_context_length = int(
            hmonnx_model.cache_abi.spec.max_context_length
        )
        configured_context_length = int(model_config.max_model_len)
        if configured_context_length > exported_context_length:
            raise ValueError(
                "vLLM max_model_len exceeds the DeepSeek-V4 HMONNX export: "
                f"configured={configured_context_length}, exported={exported_context_length}."
            )
        apply_hmonnx_runtime_options(hmonnx_model, config)
        if not should_preserve_hmonnx_runtime_device_map(config):
            hmonnx_model.to(self.inference_device)
        self.inference_device = resolve_hmonnx_primary_device(
            hmonnx_model, self.inference_device
        )
        self.inference_model = hmonnx_model
        move_hmonnx_auxiliary_tensors(hmonnx_model, self.inference_device)

        source_config = load_hf_config(
            hf_config_dir,
            trust_remote_code=get_model_trust_remote_code(model_config),
        )
        self.logits_processor = LogitsProcessor(
            int(source_config.vocab_size), logits_as_input=True
        )
        self.input_embeddings = cast(nn.Embedding, hmonnx_model.get_input_embeddings())
        self.sampler = Sampler()
        self.enable_page_attention = False
        self.cache_dtype = "float16"
        self.cache_torch_dtype = resolve_torch_dtype(self.cache_dtype)
        self.num_layers = 0
        self.key_cache_config: list[list[int]] = []
        self.value_cache_config: list[list[int]] = []
        self._cache_owner_request_id: str | None = None
        self._cache_owner_next_past_len = 0

        # vLLM still needs one tiny scheduler-visible KV group to produce
        # request metadata and block tables.  The DeepSeek runtime does not
        # read this placeholder; its real heterogeneous caches live in
        # ``inference_model.get_kvcache_mixin()``.
        self.placeholder_attn_layers = nn.ModuleList(
            [
                MerakAttention(
                    num_heads=1,
                    num_kv_heads=1,
                    head_size=1,
                    scale=1.0,
                    cache_config=self.vllm_config.cache_config,
                    prefix="model.layers.0.attn",
                    attn_backend=MerakAttentionBackend,
                )
            ]
        )
        logger.info(
            "Loaded DeepSeek-V4 Flash with request-local heterogeneous HMONNX cache; "
            "vLLM scheduler KV placeholder is intentionally not used by the runtime."
        )

    def _runtime_forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_len: int,
        request_id: str,
    ) -> torch.Tensor:
        runtime_model = cast(Any, self.inference_model)
        if not request_id:
            raise ValueError("DeepSeek-V4 internal cache requires a non-empty request ID.")
        if past_len < 0:
            raise ValueError("past_len must be non-negative.")

        owner = self._cache_owner_request_id
        if past_len == 0:
            cache_mixin = runtime_model.get_kvcache_mixin()
            reset_cache = getattr(cache_mixin, "reset_kv_cache", None)
            if not callable(reset_cache):
                reset_cache = getattr(cache_mixin, "clear_kv_cache", None)
            if callable(reset_cache):
                reset_cache()
            self._cache_owner_request_id = request_id
            self._cache_owner_next_past_len = 0
        elif owner != request_id:
            raise RuntimeError(
                "DeepSeek-V4 HMONNX cache resume has no matching resident owner: "
                f"request={request_id!r}, owner={owner!r}, past_len={past_len}."
            )
        elif past_len != self._cache_owner_next_past_len:
            raise RuntimeError(
                "DeepSeek-V4 HMONNX cache length is not monotonic: "
                f"request={request_id!r}, expected={self._cache_owner_next_past_len}, "
                f"got={past_len}."
            )

        runtime_device = torch.device(
            getattr(runtime_model, "device", self.inference_device)
        )
        flat_input_ids = input_ids.to(
            device=runtime_device,
            dtype=torch.int32,
        ).reshape(-1)
        if flat_input_ids.numel() == 0:
            raise ValueError("DeepSeek-V4 HMONNX request must contain at least one token.")

        last_logits: torch.Tensor | None = None
        try:
            for chunk_start in range(0, int(flat_input_ids.numel()), self.prefill_chunk_length):
                chunk = flat_input_ids[
                    chunk_start : chunk_start + self.prefill_chunk_length
                ]
                chunk_past_len = past_len + chunk_start
                if int(chunk.numel()) == 1 and chunk_past_len > 0:
                    runtime_model.set_decode()
                else:
                    runtime_model.set_prefill()
                data_preprocessor = runtime_model.get_data_preprocessor()
                net_inputs = data_preprocessor(
                    {
                        "input_ids": chunk.reshape(1, -1),
                        "past_seq_length": chunk_past_len,
                    }
                )
                model_output = runtime_model.forward(*net_inputs)
                result = self._extract_request_result(
                    runtime_model,
                    model_output,
                    current_input_length=int(chunk.numel()),
                    hidden_output_name=None,
                    require_all_logits=False,
                )
                last_logits = result.logits[-1]
        except Exception:
            self._cache_owner_request_id = None
            self._cache_owner_next_past_len = 0
            raise

        self._cache_owner_request_id = request_id
        self._cache_owner_next_past_len = past_len + int(flat_input_ids.numel())
        assert last_logits is not None
        return last_logits

    def release_internal_cache_owner(self, request_id: str | None = None) -> None:
        if request_id is None or request_id == self._cache_owner_request_id:
            self._cache_owner_request_id = None
            self._cache_owner_next_past_len = 0

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors: Any = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del intermediate_tensors, kwargs
        if input_ids is None:
            raise ValueError(
                "DeepSeek-V4 Flash Merak runtime requires input_ids for hash-MoE routing."
            )

        flat_input_ids = self._flatten_input_ids(input_ids)
        attn_metadata = self._get_attn_metadata()
        if attn_metadata is not None:
            past_lens = [int(value) for value in attn_metadata.past_lens.tolist()]
            query_bounds = [
                int(value) for value in attn_metadata.subsequence_begins.tolist()
            ]
        else:
            if positions is None:
                past_lens = [0]
            else:
                flat_positions = positions.reshape(-1)
                past_lens = [
                    int(flat_positions[0].item()) if flat_positions.numel() else 0
                ]
            query_bounds = [0, int(flat_input_ids.numel())]

        if len(past_lens) != 1 or len(query_bounds) != 2:
            raise RuntimeError(
                "DeepSeek-V4 Flash Merak runtime supports exactly one request per step; "
                f"past_lens={past_lens!r}, query_bounds={query_bounds!r}."
            )
        query_start, query_end = query_bounds
        if not 0 <= query_start < query_end <= int(flat_input_ids.numel()):
            raise RuntimeError(
                "Invalid vLLM request bounds for DeepSeek-V4 Flash: "
                f"start={query_start}, end={query_end}, tokens={flat_input_ids.numel()}"
            )

        logits = self._runtime_forward(
            flat_input_ids[query_start:query_end],
            past_len=past_lens[0],
            request_id=request_id or "__direct__",
        )
        output_device = flat_input_ids.device
        flat_logits = torch.empty(
            flat_input_ids.numel(),
            self.logits_processor.vocab_size,
            device=output_device,
            dtype=self.input_embeddings.weight.dtype,
        )
        flat_logits[query_end - 1].copy_(logits.to(output_device))
        return flat_logits


def get_model(
    vllm_config: VllmConfig,
    kv_cache_dtype: str,
    **kwargs: Any,
) -> nn.Module:
    del kwargs
    with cast(AbstractContextManager[None], set_current_vllm_config(vllm_config)):
        return DeepseekV4MerakForCausalLM(
            vllm_config=vllm_config,
            kv_cache_dtype=kv_cache_dtype,
        )


__all__ = ["DeepseekV4MerakForCausalLM", "get_model"]
