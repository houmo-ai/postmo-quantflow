# SPDX-License-Identifier: Apache-2.0
"""Single-request lifecycle for runtimes owning heterogeneous cache state."""

from __future__ import annotations

from typing import Any, Optional, cast

import torch

from .merak_for_causal_lm import MerakForCausalLM


class ContiguousHMONNXForCausalLM(MerakForCausalLM):
    def _runtime_forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_len: int,
        request_id: str,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        runtime_model = cast(Any, self.inference_model)
        if not request_id:
            raise ValueError("HMONNX internal cache requires a non-empty request ID.")
        if past_len < 0:
            raise ValueError("past_len must be non-negative.")
        if inputs_embeds is not None and (
            inputs_embeds.ndim != 2 or inputs_embeds.shape[0] != input_ids.numel()
            or inputs_embeds.shape[1] != self.input_embeddings.embedding_dim
        ):
            raise ValueError("HMONNX inputs_embeds must match token count and embedding width.")

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
                "HMONNX cache resume has no matching resident owner: "
                f"request={request_id!r}, owner={owner!r}, past_len={past_len}."
            )
        elif past_len != self._cache_owner_next_past_len:
            raise RuntimeError(
                "HMONNX cache length is not monotonic: "
                f"request={request_id!r}, expected={self._cache_owner_next_past_len}, "
                f"got={past_len}."
            )

        runtime_device = torch.device(getattr(runtime_model, "device", self.inference_device))
        flat_input_ids = input_ids.to(
            device=runtime_device,
            dtype=torch.int32,
        ).reshape(-1)
        if flat_input_ids.numel() == 0:
            raise ValueError("HMONNX request must contain at least one token.")

        last_logits: torch.Tensor | None = None
        try:
            for chunk_start in range(0, int(flat_input_ids.numel()), self.prefill_chunk_length):
                chunk = flat_input_ids[chunk_start : chunk_start + self.prefill_chunk_length]
                chunk_past_len = past_len + chunk_start
                if int(chunk.numel()) == 1 and chunk_past_len > 0:
                    runtime_model.set_decode()
                else:
                    runtime_model.set_prefill()
                data_preprocessor = runtime_model.get_data_preprocessor()
                data = {"past_seq_length": chunk_past_len}
                if inputs_embeds is None:
                    data["input_ids"] = chunk.reshape(1, -1)
                else:
                    data["inputs_embeds"] = inputs_embeds[
                        chunk_start : chunk_start + chunk.numel()
                    ].unsqueeze(0)
                net_inputs = data_preprocessor(data)
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
            raise ValueError("HMONNX runtime requires input_ids.")

        flat_input_ids = self._flatten_input_ids(input_ids)
        flat_inputs_embeds = self._flatten_inputs_embeds(inputs_embeds) if inputs_embeds is not None else None
        if flat_inputs_embeds is not None and flat_inputs_embeds.shape[0] != flat_input_ids.numel():
            raise ValueError("HMONNX inputs_embeds must match input_ids token count.")
        attn_metadata = self._get_attn_metadata()
        if attn_metadata is not None:
            past_lens = [int(value) for value in attn_metadata.past_lens.tolist()]
            query_bounds = [int(value) for value in attn_metadata.subsequence_begins.tolist()]
        else:
            if positions is None:
                past_lens = [0]
            else:
                flat_positions = positions.reshape(-1)
                past_lens = [int(flat_positions[0].item()) if flat_positions.numel() else 0]
            query_bounds = [0, int(flat_input_ids.numel())]

        if len(past_lens) != 1 or len(query_bounds) != 2:
            raise RuntimeError(
                "HMONNX Merak runtime supports exactly one request per step; "
                f"past_lens={past_lens!r}, query_bounds={query_bounds!r}."
            )
        query_start, query_end = query_bounds
        if not 0 <= query_start < query_end <= int(flat_input_ids.numel()):
            raise RuntimeError(
                "Invalid vLLM request bounds for HMONNX: "
                f"start={query_start}, end={query_end}, tokens={flat_input_ids.numel()}"
            )

        logits = self._runtime_forward(
            flat_input_ids[query_start:query_end],
            past_len=past_lens[0],
            request_id=request_id or "__direct__",
            inputs_embeds=flat_inputs_embeds[query_start:query_end] if flat_inputs_embeds is not None else None,
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
