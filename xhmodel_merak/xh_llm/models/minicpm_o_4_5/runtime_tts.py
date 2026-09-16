from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...types import LLMModelMeta
from .runtime_text_decoder import MiniCPMO45FixedCacheTextLLMHMONNXModel, _load_decoder_child_meta


class DeferredProjectorSemantic(nn.Module):
    def __init__(self, runtime: MiniCPMO45TTSHMONNXRuntime) -> None:
        super().__init__()
        object.__setattr__(self, "_runtime", runtime)

    def forward(self, hidden_states: Tensor) -> Tensor:
        runtime = self._runtime
        runtime.defer_semantic_hidden_states(hidden_states)
        return hidden_states.new_zeros((*hidden_states.shape[:-1], runtime.tts_hidden_size))


def _last_tts_decoder_output(outputs: Sequence[tuple[Tensor | Sequence[Tensor], int]]) -> Tensor | Sequence[Tensor]:
    last_output: Tensor | Sequence[Tensor] | None = None
    for output, _ in outputs:
        last_output = output
    if last_output is None:
        raise RuntimeError("TTS Prefill produced no output")
    return last_output


def _resolve_tts_meta(root: Path, component: dict[str, Any]) -> LLMModelMeta:
    """Load the standard text-runtime metadata for a TTS decoder artifact.

    TTS receives ``inputs_embeds`` from upstream rather than token ids, so its
    shared text-runtime subclass supplies a dummy embedding and never reads
    ``quant_embedding``.  The graph/session/KV portion is otherwise the same
    ``LLMModelMeta`` contract as the LLM export.
    """
    return _load_decoder_child_meta(root, component)


def _load_host_embedding(path: Path) -> nn.Embedding:
    state = torch.load(path, map_location="cpu", weights_only=True)
    weight = state["weight"]
    embedding = nn.Embedding(weight.shape[0], weight.shape[1], dtype=weight.dtype)
    embedding.load_state_dict(state)
    return embedding


class MiniCPMO45TTSHMONNXRuntime(MiniCPMO45FixedCacheTextLLMHMONNXModel, nn.Module):
    """Shared text runtime in the official TTS module slot.

    The upstream MiniCPM TTS model registers its ``model`` child with
    :class:`torch.nn.Module`; unlike ordinary text models it therefore rejects
    a non-module HMONNX adapter.  This narrow ``nn.Module`` mix-in is required
    by that external ABI.  Session/KV ownership remains entirely in the shared
    text base above.
    """

    cache_component = "tts"

    def __init__(
        self,
        root: Path,
        meta: dict[str, Any],
        *,
        enable_cuda_graph: bool = False,
        enable_auto_offload: bool = False,
        enable_golden: bool = False,
        device_map: str | torch.device | list[str | torch.device] | None = None,
    ) -> None:
        nn.Module.__init__(self)
        super().__init__(
            _resolve_tts_meta(root, meta),
            enable_cuda_graph=enable_cuda_graph,
            enable_auto_offload=enable_auto_offload,
            enable_golden=enable_golden,
            device_map=device_map,
        )
        self.semantic_hidden_size = int(meta["semantic_hidden_size"])
        self.tts_hidden_size = int(meta["tts_hidden_size"])
        self.text_embedding_path = root / str(meta["text_embedding"])
        self.code_embedding_path = root / str(meta["code_embedding"])
        self._pending_semantic_hidden_states: Tensor | None = None

    def install_host_embeddings(self, tts_module: nn.Module) -> None:
        text_embedding = _load_host_embedding(self.text_embedding_path)
        code_embedding = _load_host_embedding(self.code_embedding_path)
        tts_module.emb_text = text_embedding
        tts_module.emb_code = nn.ModuleList([code_embedding])

    @staticmethod
    def _build_embed_tokens_from_meta(_meta: LLMModelMeta) -> nn.Embedding:
        """TTS uses caller-supplied embeddings; satisfy the shared base only."""
        return nn.Embedding(1, 1, dtype=torch.float16)

    def defer_semantic_hidden_states(self, hidden_states: Tensor) -> None:
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(0)
        self._pending_semantic_hidden_states = hidden_states

    def _take_semantic_inputs(self, inputs_embeds: Tensor) -> tuple[Tensor, Tensor]:
        current_length = int(inputs_embeds.shape[1])
        semantic = torch.zeros(
            (inputs_embeds.shape[0], current_length, self.semantic_hidden_size),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        mask = torch.zeros(
            (inputs_embeds.shape[0], current_length, 1),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        pending = self._pending_semantic_hidden_states
        self._pending_semantic_hidden_states = None
        if pending is None:
            return semantic, mask
        semantic_length = pending.shape[1]
        semantic[:, :semantic_length].copy_(pending.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
        mask[:, :semantic_length] = 1
        return semantic, mask

    def __call__(self, **kwargs: Any) -> BaseModelOutputWithPast:
        return self.forward_hf(**kwargs)

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: Sequence[Tensor],
        past_value_caches: Sequence[Tensor],
        attention_mask: Tensor,
        semantic_hidden_states: Tensor,
        semantic_mask: Tensor,
    ) -> Tensor | Sequence[Tensor]:
        outputs = self.run_decoder_graph(
            inputs_embeds,
            past_seq_length=int(past_seq_length.reshape(-1)[0]),
            current_input_length=int(current_input_length.reshape(-1)[0]),
            past_key_caches=past_key_caches,
            past_value_caches=past_value_caches,
            attention_mask=attention_mask,
            extra_sequence_inputs=(semantic_hidden_states, semantic_mask),
        )
        return _last_tts_decoder_output(outputs)

    def forward_hf(
        self,
        *,
        position_ids: Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        del position_ids, kwargs
        if inputs_embeds is None:
            raise ValueError("HMONNX TTS forward requires inputs_embeds")
        if not return_dict:
            raise ValueError("HMONNX TTS forward requires return_dict=True")
        if use_cache and past_key_values is not None and past_key_values is not self.hf_cache:
            self._import_cache(past_key_values)

        cache = self.hf_cache if use_cache else None
        if cache is not None and past_key_values is None and cache.get_seq_length() > 0:
            for tensor in [*self.past_key_caches, *self.past_value_caches]:
                tensor.zero_()
            cache.reset()
        past_length = cache.get_seq_length() if cache is not None else 0
        current_length = int(inputs_embeds.shape[1])
        if past_length + current_length > self.max_sequence_length:
            raise RuntimeError(
                "tts cache capacity exceeded: "
                f"current={past_length}, requested={current_length}, capacity={self.max_sequence_length}"
            )

        if attention_mask is None:
            dtype = inputs_embeds.dtype
            attention_mask = torch.full(
                (1, 1, current_length, self.max_sequence_length),
                torch.finfo(dtype).min,
                dtype=dtype,
                device=inputs_embeds.device,
            )
            for row in range(current_length):
                attention_mask[:, :, row, : past_length + row + 1] = 0
        elif attention_mask.shape[-1] < self.max_sequence_length:
            attention_mask = torch.nn.functional.pad(
                attention_mask,
                (0, self.max_sequence_length - attention_mask.shape[-1]),
                value=torch.finfo(attention_mask.dtype).min,
            )
        elif attention_mask.shape[-1] > self.max_sequence_length:
            attention_mask = attention_mask[..., : self.max_sequence_length]

        def commit(valid_length: int) -> None:
            if cache is not None:
                cache.commit_length(valid_length)

        key_caches = self.past_key_caches if use_cache else [torch.zeros_like(value) for value in self.past_key_caches]
        value_caches = (
            self.past_value_caches if use_cache else [torch.zeros_like(value) for value in self.past_value_caches]
        )
        semantic_hidden_states, semantic_mask = self._take_semantic_inputs(inputs_embeds)
        output = _last_tts_decoder_output(
            self.run_decoder_graph(
                inputs_embeds,
                past_seq_length=past_length,
                current_input_length=current_length,
                past_key_caches=key_caches,
                past_value_caches=value_caches,
                attention_mask=attention_mask,
                extra_sequence_inputs=(semantic_hidden_states, semantic_mask),
                on_graph_call=commit,
            )
        )
        if isinstance(output, Tensor):
            hidden = output[:, :current_length]
        else:
            hidden = output[0]
            hidden = hidden[:, :current_length]
        return BaseModelOutputWithPast(
            last_hidden_state=hidden,
            past_key_values=cache,
            hidden_states=(hidden,) if output_hidden_states else None,
        )

    def _import_cache(self, cache: Any) -> None:
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            keys = list(cache.key_cache)
            values = list(cache.value_cache)
        elif isinstance(cache, (tuple, list)):
            keys = [layer[0] for layer in cache]
            values = [layer[1] for layer in cache]
        elif hasattr(cache, "__len__") and hasattr(cache, "__getitem__"):
            layers = [cache[index] for index in range(len(cache))]
            keys = [layer[0] for layer in layers]
            values = [layer[1] for layer in layers]
        else:
            raise ValueError("past_key_values must be runtime-owned or provide cache layer tuples")
        num_layers = self._kvcache_mixin.kvcache_config.num_layers
        if len(keys) != num_layers or len(values) != num_layers:
            raise ValueError("TTS cache layer count does not match the HMONNX runtime")
        imported_length = int(keys[0].shape[-2])
        for index, (key, value, key_target, value_target) in enumerate(
            zip(keys, values, self.past_key_caches, self.past_value_caches, strict=True)
        ):
            expected = key_target.shape[:2] + key_target.shape[-1:]
            if key.shape[:2] + key.shape[-1:] != expected or value.shape[:2] + value.shape[-1:] != expected:
                raise ValueError(f"TTS cache layer {index} shape is incompatible with HMONNX backing buffers")
            if key.shape[-2] != imported_length or value.shape[-2] != imported_length:
                raise ValueError("TTS cache layers must have one shared sequence length")
            if imported_length > self.max_sequence_length:
                raise ValueError("TTS cache sequence length exceeds HMONNX capacity")
            if key.dtype != key_target.dtype or value.dtype != value_target.dtype:
                raise ValueError(f"TTS cache layer {index} dtype is incompatible with HMONNX backing buffers")
            if key.device != key_target.device or value.device != value_target.device:
                raise ValueError(f"TTS cache layer {index} device is incompatible with HMONNX backing buffers")
        backups = [tensor.clone() for tensor in [*self.past_key_caches, *self.past_value_caches]]
        try:
            for target, value in zip(
                [*self.past_key_caches, *self.past_value_caches],
                [*keys, *values],
                strict=True,
            ):
                target.zero_()
                target[:, :, :imported_length].copy_(value)
        except RuntimeError:
            for target, backup in zip(
                [*self.past_key_caches, *self.past_value_caches],
                backups,
                strict=True,
            ):
                target.copy_(backup)
            raise
        self.hf_cache.commit_length(imported_length)

    def reset_state(self) -> None:
        self._pending_semantic_hidden_states = None
        super().reset_state()

    def release_state(self) -> None:
        self._pending_semantic_hidden_states = None
        super().release_state()


__all__ = ["MiniCPMO45TTSHMONNXRuntime", "_resolve_tts_meta"]
