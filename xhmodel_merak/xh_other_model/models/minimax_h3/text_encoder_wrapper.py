from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextMLP,
    Qwen3VLTextModel,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
)

from xhquant.nn import MaskedAdd, RMSNorm, Rope
from xhquant.utils.registry.dynamic_module import DynamicModule


DEFAULT_TEXT_ENCODER_LAYER = 50
DEFAULT_INPUT_SEQUENCE_LENGTH = 512


def _as_cfg(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {}


def build_minimax_h3_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Build Qwen3-VL text-only mrope position ids from a 2D attention mask.

    Matches the pure-text branch of `Qwen3VLModel.get_rope_index`:
    cumsum(mask) - 1, with padded positions filled by 1, then expanded to 3 axes.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be 2D [B, L], got shape {tuple(attention_mask.shape)}")
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 1)
    return position_ids.unsqueeze(0).expand(3, -1, -1).contiguous()


def build_minimax_h3_text_inputs(
    tokenizer,
    prompt: Union[str, Sequence[str]],
    input_sequence_length: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad/truncate MiniMax-H3 text prompts to a fixed export length.

    Diffusers MiniMax-H3 tokenizes with `add_special_tokens=False` and no chat
    template. For static HMONNX graphs we keep that tokenization contract and
    only add right-side pad/truncate to `input_sequence_length`.
    """
    if input_sequence_length <= 0:
        raise ValueError(f"input_sequence_length must be positive, got {input_sequence_length}")

    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    if not prompts:
        raise ValueError("prompt must contain at least one string")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id for fixed-length export")

    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=input_sequence_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if tokenizer.pad_token_id is None and pad_token_id is not None:
        # Some MiniMax tokenizers only expose eos as pad fallback.
        pad_positions = attention_mask == 0
        input_ids = input_ids.masked_fill(pad_positions, pad_token_id)

    position_ids = build_minimax_h3_position_ids(attention_mask)
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)
    return input_ids, attention_mask, position_ids


def build_minimax_h3_attention_bias(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be 2D [B, L], got shape {tuple(attention_mask.shape)}")
    sequence_length = attention_mask.shape[-1]
    blocked = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=attention_mask.device),
        diagonal=1,
    ).view(1, 1, sequence_length, sequence_length)
    blocked = blocked | (attention_mask[:, None, None, :] == 0)
    return blocked.to(dtype) * (-65504)


def build_minimax_h3_text_graph_inputs(
    language_model: Qwen3VLTextModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    deepstack_visual_embeds: Sequence[torch.Tensor] | None = None,
    visual_pos_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    hidden_states = language_model.embed_tokens(input_ids)
    rotary_cos, rotary_sin = language_model.rotary_emb(hidden_states, position_ids)
    attention_bias = build_minimax_h3_attention_bias(attention_mask, hidden_states.dtype)
    dense_deepstack = [torch.zeros_like(hidden_states) for _ in range(3)]
    if deepstack_visual_embeds is not None:
        if visual_pos_masks is None:
            raise ValueError("visual_pos_masks is required when deepstack_visual_embeds is provided")
        if len(deepstack_visual_embeds) != len(dense_deepstack):
            raise ValueError(
                f"Qwen3-VL must provide {len(dense_deepstack)} DeepStack tensors, got {len(deepstack_visual_embeds)}"
            )
        for index, visual_embeds in enumerate(deepstack_visual_embeds):
            dense_deepstack[index][visual_pos_masks] = visual_embeds.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
    return hidden_states, rotary_cos, rotary_sin, attention_bias, *dense_deepstack


class _Qwen3VLTextRMSNorm(DynamicModule):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)

    def _setup(self, cfg=None, *args, **kwargs):
        del cfg, args, kwargs
        self.norm = RMSNorm(self.weight.shape[0], self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


class _Qwen3VLTextRotaryEmbedding(DynamicModule):
    def _setup(self, cfg=None, *args, **kwargs):
        # Keep runtime position_ids-driven RoPE. MiniMax pad positions are filled
        # with 1 (Qwen3-VL text-only get_rope_index), so an arange cos/sin cache
        # would diverge from Diffusers numerics.
        del cfg, args, kwargs
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        inv_freq = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        frequencies = torch.matmul(inv_freq, position_ids[:, :, None, :].float()).transpose(2, 3)
        frequencies = frequencies[0].clone()
        for dimension, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dimension] * 3
            dimension_frequencies = torch.matmul(
                inv_freq[dimension],
                position_ids[dimension, :, None, :].float(),
            ).transpose(1, 2)
            frequencies[..., offset:length:3] = dimension_frequencies[..., offset:length:3]
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        cos = embeddings.cos() * self.attention_scaling
        sin = embeddings.sin() * self.attention_scaling
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)


class _Qwen3VLTextMLP(DynamicModule):
    def _setup(self, cfg=None, *args, **kwargs):
        del cfg, args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class _Qwen3VLTextAttention(DynamicModule):
    def _setup(self, cfg=None, *args, **kwargs):
        del cfg, args, kwargs
        self.rope = Rope()
        self.masked_add = MaskedAdd()
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, sequence_length, -1, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch_size, sequence_length, -1, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch_size, sequence_length, -1, self.head_dim).transpose(1, 2)
        query, key = self.q_norm(query), self.k_norm(key)
        cos, sin = position_embeddings
        query = self.rope(query, cos.unsqueeze(1), sin.unsqueeze(1))
        key = self.rope(key, cos.unsqueeze(1), sin.unsqueeze(1))
        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        if attention_mask is not None:
            scores = self.masked_add(scores, attention_mask)
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attention_output = torch.matmul(probabilities, value)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return self.o_proj(attention_output)


class _Qwen3VLTextDecoderLayer(DynamicModule):
    def _setup(self, cfg=None, *args, **kwargs):
        del cfg, args, kwargs
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), position_embeddings, attention_mask)
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class _Qwen3VLTextModel(DynamicModule):
    def _setup(self, cfg=None, *args, **kwargs):
        del args, kwargs
        cfg = _as_cfg(cfg)
        self.text_encoder_layer = int(cfg.get("text_encoder_layer", DEFAULT_TEXT_ENCODER_LAYER))
        self.input_sequence_length = int(cfg.get("input_sequence_length", DEFAULT_INPUT_SEQUENCE_LENGTH) or 0)
        num_layers = len(self.layers)
        if self.text_encoder_layer <= 0 or self.text_encoder_layer > num_layers:
            raise ValueError(
                f"text_encoder_layer={self.text_encoder_layer} is out of range for "
                f"{num_layers} decoder layers; MiniMax-H3 expects hidden_states[50]"
            )
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        attention_mask: torch.Tensor,
        deepstack_visual_embed_0: torch.Tensor,
        deepstack_visual_embed_1: torch.Tensor,
        deepstack_visual_embed_2: torch.Tensor,
    ) -> torch.Tensor:
        position_embeddings = (rotary_cos, rotary_sin)
        deepstack_visual_embeds = (
            deepstack_visual_embed_0,
            deepstack_visual_embed_1,
            deepstack_visual_embed_2,
        )
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, position_embeddings, attention_mask)
            if layer_idx < len(deepstack_visual_embeds):
                hidden_states = hidden_states + deepstack_visual_embeds[layer_idx]
            # HF capture_outputs: index 0 = embeddings, index k = after layer k-1.
            # Diffusers MiniMax-H3 conditions on hidden_states[50] == after layer 49.
            if layer_idx + 1 == self.text_encoder_layer:
                return hidden_states
        return self.norm(hidden_states)


_TEXT_ENCODER_WRAPPERS = {
    Qwen3VLTextRMSNorm: _Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding: _Qwen3VLTextRotaryEmbedding,
    Qwen3VLTextMLP: _Qwen3VLTextMLP,
    Qwen3VLTextAttention: _Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer: _Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel: _Qwen3VLTextModel,
}


def _get_language_model(text_encoder: nn.Module) -> nn.Module:
    if isinstance(text_encoder, Qwen3VLTextModel):
        return text_encoder
    model = getattr(text_encoder, "model", None)
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        language_model = getattr(text_encoder, "language_model", None)
    if not isinstance(language_model, Qwen3VLTextModel):
        raise TypeError("MiniMax-H3 text encoder must contain a Qwen3VLTextModel language tower")
    return language_model


def wrap_text_encoder(
    text_encoder: nn.Module,
    *,
    input_sequence_length: int = DEFAULT_INPUT_SEQUENCE_LENGTH,
    text_encoder_layer: int = DEFAULT_TEXT_ENCODER_LAYER,
) -> nn.Module:
    language_model = _get_language_model(text_encoder)
    wrap_cfg = {
        "input_sequence_length": int(input_sequence_length),
        "text_encoder_layer": int(text_encoder_layer),
    }
    for _, module in list(language_model.named_modules()):
        dynamic_cls = _TEXT_ENCODER_WRAPPERS.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            dynamic_cls.convert(module, wrap_cfg)
    return text_encoder


class MiniMaxH3TextEncoderWrapper(nn.Module):
    """Exportable MiniMax-H3 Qwen3-VL text tower with fixed-length I/O contract."""

    def __init__(
        self,
        text_encoder: nn.Module,
        input_sequence_length: int = DEFAULT_INPUT_SEQUENCE_LENGTH,
        text_encoder_layer: int = DEFAULT_TEXT_ENCODER_LAYER,
    ):
        super().__init__()
        self.input_sequence_length = int(input_sequence_length)
        self.text_encoder_layer = int(text_encoder_layer)
        language_model = _get_language_model(text_encoder)
        self.text_encoder = wrap_text_encoder(
            text_encoder,
            input_sequence_length=self.input_sequence_length,
            text_encoder_layer=self.text_encoder_layer,
        )
        self.language_model = language_model

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        attention_mask: torch.Tensor,
        deepstack_visual_embed_0: torch.Tensor,
        deepstack_visual_embed_1: torch.Tensor,
        deepstack_visual_embed_2: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[-2] != self.input_sequence_length:
            raise ValueError(
                f"hidden_states length {hidden_states.shape[-2]} != fixed input_sequence_length "
                f"{self.input_sequence_length}"
            )
        if rotary_cos.shape[-2] != self.input_sequence_length or rotary_sin.shape[-2] != self.input_sequence_length:
            raise ValueError(
                "rotary embedding length must match fixed input_sequence_length "
                f"{self.input_sequence_length}, got cos={rotary_cos.shape[-2]}, sin={rotary_sin.shape[-2]}"
            )
        if attention_mask.shape[-2:] != (self.input_sequence_length, self.input_sequence_length):
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} must end with "
                f"({self.input_sequence_length}, {self.input_sequence_length})"
            )
        deepstack_visual_embeds = (
            deepstack_visual_embed_0,
            deepstack_visual_embed_1,
            deepstack_visual_embed_2,
        )
        for index, visual_embeds in enumerate(deepstack_visual_embeds):
            if visual_embeds.shape != hidden_states.shape:
                raise ValueError(
                    f"deepstack_visual_embed_{index} shape {tuple(visual_embeds.shape)} must match "
                    f"hidden_states shape {tuple(hidden_states.shape)}"
                )
        return self.language_model(
            hidden_states,
            rotary_cos,
            rotary_sin,
            attention_mask,
            *deepstack_visual_embeds,
        )


__all__ = [
    "DEFAULT_INPUT_SEQUENCE_LENGTH",
    "DEFAULT_TEXT_ENCODER_LAYER",
    "MiniMaxH3TextEncoderWrapper",
    "build_minimax_h3_attention_bias",
    "build_minimax_h3_position_ids",
    "build_minimax_h3_text_graph_inputs",
    "build_minimax_h3_text_inputs",
    "wrap_text_encoder",
]
