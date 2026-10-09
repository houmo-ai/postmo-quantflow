# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Request-local GLM caches and transactional explicit graph outputs."""

import torch

from ...kv_cache_mixin import KVCacheMixin
from ...llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor
from .contract import DSA, build_transition


class Glm53CacheMixin(KVCacheMixin):
    def __init__(self, spec, config, *, inplace_step_staging=False):
        super().__init__(config)
        self.spec = spec
        self.caches = ()
        self.inplace_step_staging = inplace_step_staging
        self._staged_rows = None

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
        self._staged_rows = None
        self._cache_initialized = False

    def _set_device(self, device):
        super()._set_device(device)
        self.caches = tuple(x.to(device) for x in self.caches)

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        self.caches = tuple(x.to(dtype) for x in self.caches)

    def commit(self, outputs):
        from xhquant.core import CacheTensor

        values = tuple(outputs)
        if len(values) != len(self.caches) + 1:
            raise ValueError("GLM graph outputs do not match cache ABI")
        for old, new in zip(self.caches, values[1:], strict=True):
            if old.shape != new.shape:
                raise ValueError("GLM graph returned an incompatible cache shape")
        # No state is replaced until every output has passed validation.
        committed = list(values[1:])
        cursor = 0
        for kind in self.spec.layer_types:
            indices = (cursor, cursor + 1) if kind == DSA else (cursor + 1,)
            for index in indices:
                if not isinstance(committed[index], CacheTensor):
                    committed[index] = CacheTensor(committed[index])
            cursor += len(self.spec.cache_suffixes(kind))
        self.caches = tuple(committed)
        self._staged_rows = None
        return values[0]

    def stage_step(self, past_length, current_length):
        if not self.inplace_step_staging:
            return self.spec.stage_caches(self.caches)
        if self._staged_rows is not None:
            raise RuntimeError("GLM cache step is already staged")
        if past_length < 0 or current_length <= 0 or past_length + current_length > self.spec.max_context_length:
            raise ValueError("GLM cache step exceeds the configured context capacity")
        pool_start = past_length // self.spec.pool_size
        pool_count = (past_length % self.spec.pool_size + current_length) // self.spec.pool_size
        staged_rows = []
        cursor = 0
        for kind in self.spec.layer_types:
            if kind == DSA:
                latent = self.caches[cursor].data
                pool = self.caches[cursor + 1].data
                staged_rows.append((latent, past_length, latent.narrow(2, past_length, current_length).clone()))
                if pool_count:
                    staged_rows.append((pool, pool_start, pool.narrow(2, pool_start, pool_count).clone()))
            else:
                state = self.caches[cursor + 1].data
                staged_rows.append((state, 0, state.clone()))
            cursor += len(self.spec.cache_suffixes(kind))
        self._staged_rows = staged_rows
        return self.caches

    def rollback_step(self):
        if self._staged_rows is not None:
            for cache, start, original in self._staged_rows:
                cache.narrow(2, start, original.shape[2]).copy_(original)
            self._staged_rows = None


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
        past_length = int(data.get("past_seq_length", 0))
        transition = build_transition(
            self.cache_mixin.spec,
            past_length,
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
        return (embeds, *transition.values(), self.cache_mixin.stage_step(past_length, n))


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
