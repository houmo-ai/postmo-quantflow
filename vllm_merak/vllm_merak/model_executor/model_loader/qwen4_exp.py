# SPDX-License-Identifier: Apache-2.0
"""Qwen4 uses the shared serial scheduler and its own HC/GDN/QSA/PLE runtime."""

from pathlib import Path

import torch
from torch import nn
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.v1.sample.sampler import Sampler

from xhmodel_merak.xh_llm import AutoLLMHONNXModel

from ...attention.backends.merak import MerakAttentionBackend
from ...constants import HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST
from ...utils import get_hmonnx_runtime_kwargs, move_hmonnx_auxiliary_tensors
from ..register import MERAK_MODEL_REGISTRY
from ._contiguous_hmonnx import ContiguousHMONNXForCausalLM
from .merak_for_causal_lm import MerakAttention


@MERAK_MODEL_REGISTRY.register_module("Qwen4ExpForConditionalGeneration")
class Qwen4ExpMerakForCausalLM(ContiguousHMONNXForCausalLM):
    hybrid_cache_policy = HYBRID_CACHE_POLICY_CONTIGUOUS_SINGLE_REQUEST
    uses_internal_hmonnx_cache = True
    requires_raw_input_tokens = True
    supports_mrope = True
    _prefer_input_ids_runtime = True

    def __init__(self, model_config=None, kv_cache_dtype="auto", *, vllm_config=None, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.device = torch.device("cpu")
        if kv_cache_dtype not in {"auto", "float16", "fp16", "torch.float16"}:
            raise ValueError("Qwen4 HMONNX cache is exported as float16")
        if vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError("Qwen4's heterogeneous cache requires max_num_seqs=1")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Qwen4's heterogeneous cache requires enable_prefix_caching=False")
        if vllm_config.speculative_config is not None:
            spec = vllm_config.speculative_config
            proposer = "vllm_merak.spec_decode.qwen4.Qwen4MTPProposer"
            if spec.method != "custom_class" or spec.model != proposer:
                raise ValueError(f"Qwen4 MTP requires custom_class {proposer}")
            if vllm_config.scheduler_config.async_scheduling:
                raise ValueError("Qwen4 MTP cache commits require async_scheduling=False")
        root = Path(vllm_config.model_config.model)
        options = get_hmonnx_runtime_kwargs(self.config)
        # Qwen4 target and draft graphs already share quantized initializers.
        # Parallel-linear fusion is a separate graph transformation.
        options["enable_parallel_linear_fusion"] = False
        options["enable_page_attention"] = False
        self.inference_model = AutoLLMHONNXModel.from_pretrained(
            str(root / self.config.xh_model["meta_info"]),
            **options,
        )
        self.inference_device = torch.device(self.inference_model.device)
        move_hmonnx_auxiliary_tensors(self.inference_model, self.inference_device)
        self.input_embeddings = self.inference_model.get_input_embeddings()
        self.prefill_chunk_length = int(self.inference_model.meta_info.model_config.prefill_chunk_length)
        self.logits_processor = LogitsProcessor(self.input_embeddings.num_embeddings, logits_as_input=True)
        self.sampler = Sampler()
        self.enable_page_attention = False
        self.cache_dtype = "float16"
        self.cache_torch_dtype = torch.float16
        self.num_layers = 0
        self.key_cache_config = []
        self.value_cache_config = []
        self._cache_owner_request_id = None
        self._cache_owner_next_past_len = 0
        # As with GLM/DeepSeek, this tiny KV group only supplies scheduling
        # metadata. All model state is owned by the HMONNX runtime.
        self.placeholder_attn_layers = nn.ModuleList(
            [
                MerakAttention(
                    num_heads=1,
                    num_kv_heads=1,
                    head_size=1,
                    scale=1.0,
                    cache_config=vllm_config.cache_config,
                    prefix="model.layers.0.attn",
                    attn_backend=MerakAttentionBackend,
                )
            ]
        )

    def get_mrope_input_positions(self, input_tokens, mm_features):
        return Qwen3VLForConditionalGeneration._get_mrope_input_positions(
            input_tokens=input_tokens,
            mm_features=mm_features,
            config=self.config,
        )

    def _runtime_forward(self, input_ids, *, past_len, request_id, inputs_embeds=None):
        if past_len == 0:
            self.inference_model.reset_internal_state()
        return super()._runtime_forward(
            input_ids,
            past_len=past_len,
            request_id=request_id,
            inputs_embeds=inputs_embeds,
        )


__all__ = ["Qwen4ExpMerakForCausalLM"]
