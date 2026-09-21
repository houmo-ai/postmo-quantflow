# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Request-local GLM caches and transactional explicit graph outputs."""

import torch

from ...kv_cache_mixin import KVCacheMixin
from ...llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor
from .contract import build_transition


class Glm53CacheMixin(KVCacheMixin):
    def __init__(self, spec, config):
        super().__init__(config)
        self.spec = spec
        self.caches = ()

    def prepare_kv_cache(self):
        if not self._cache_initialized:
            device = torch.get_default_device() if self._device is None else self._device
            # Explicit recurrent/boundary inputs must be real tensors for FX.
            if torch.device(device).type == "meta":
                device = "cpu"
            self.caches = self.spec.initial_caches(device=device, dtype=self._dtype or torch.float16)
            self._cache_initialized = True

    def clear_kv_cache(self):
        self.caches = ()
        self._cache_initialized = False

    def _set_device(self, device):
        super()._set_device(device)
        self.caches = tuple(x.to(device) for x in self.caches)

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        self.caches = tuple(x.to(dtype) for x in self.caches)

    def commit(self, outputs):
        values = tuple(outputs)
        if len(values) != len(self.caches) + 1:
            raise ValueError("GLM graph outputs do not match cache ABI")
        for old, new in zip(self.caches, values[1:], strict=True):
            if old.shape != new.shape:
                raise ValueError("GLM graph returned an incompatible cache shape")
        # No state is replaced until every output has passed validation.
        self.caches = values[1:]
        return values[0]


class Glm53DataPreprocess(BaseLLMInputProcessor):
    def __init__(self, *, token_embedding, input_sequence_length, cache_mixin, pad_token_id):
        super().__init__(
            BaseInputProcessorConfig(embed_tokens=token_embedding, input_sequence_length=input_sequence_length)
        )
        self.embed_tokens = token_embedding
        self.cache_mixin, self.pad_token_id = cache_mixin, pad_token_id

    def __call__(self, data):
        ids, embeds = data.get("input_ids"), data.get("inputs_embeds")
        if (ids is None) == (embeds is None):
            raise ValueError("pass exactly one of input_ids or inputs_embeds")
        value = ids if embeds is None else embeds
        if value.ndim != (2 if embeds is None else 3) or value.shape[0] != 1:
            raise ValueError("GLM static input must have batch size one")
        if embeds is not None and embeds.shape[-1] != self.embed_tokens.weight.shape[-1]:
            raise ValueError("inputs_embeds has an incompatible language hidden dimension")
        n = value.shape[1]
        transition = build_transition(
            self.cache_mixin.spec,
            int(data.get("past_seq_length", 0)),
            n,
            self.input_sequence_length,
            device=self._device or value.device,
            dtype=self.embed_tokens.weight.dtype,
        )
        if embeds is None:
            embeds = self.embed_tokens(ids.to(self.embed_tokens.weight.device))
        embeds = embeds.to(device=self._device or value.device, dtype=self.embed_tokens.weight.dtype)
        if n < self.input_sequence_length:
            embeds = torch.nn.functional.pad(embeds, (0, 0, 0, self.input_sequence_length - n))
        self.cache_mixin.prepare_kv_cache()
        return (embeds, *transition.values(), self.cache_mixin.spec.stage_caches(self.cache_mixin.caches))


class Glm53Runner:
    """Execute fixed prefill/decode callables with commit-after-success semantics."""

    def __init__(self, prefill, decode, embedding, spec, prefill_length, *, device="cpu", dtype=torch.float32):
        self.prefill, self.decode, self.embedding = prefill, decode, embedding
        self.spec, self.prefill_length, self.device, self.dtype = spec, prefill_length, device, dtype
        self.reset()

    def reset(self):
        self.caches = self.spec.initial_caches(device=self.device, dtype=self.dtype)
        self.past_length = 0

    @torch.inference_mode()
    def step(self, input_ids):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("GLM runner supports one unpadded token sequence")
        n = input_ids.shape[1]
        p = 1 if n == 1 else self.prefill_length
        transition = build_transition(self.spec, self.past_length, n, p, device=self.device, dtype=self.dtype)
        embeds = self.embedding(input_ids.to(self.device))
        embeds = torch.nn.functional.pad(embeds, (0, 0, 0, p - n))
        staged = self.spec.stage_caches(self.caches)
        outputs = (self.decode if p == 1 else self.prefill)(embeds, *transition.values(), staged)
        if len(outputs) != 1 + len(self.caches):
            raise ValueError("GLM runtime cache output count mismatch")
        for old, new in zip(self.caches, outputs[1:], strict=True):
            if old.shape != new.shape:
                raise ValueError("GLM runtime cache output shape mismatch")
        self.caches = tuple(outputs[1:])
        self.past_length += n
        return outputs[0]

    def prefill_tokens(self, input_ids):
        if input_ids.shape[1] == 0:
            raise ValueError("prompt must contain tokens")
        # Validate the whole request before committing any chunk.
        if self.past_length + input_ids.shape[1] > self.spec.max_context_length:
            raise ValueError("request exceeds the configured context capacity")
        for chunk in input_ids.split(self.prefill_length, dim=1):
            logits = self.step(chunk)
        return logits
