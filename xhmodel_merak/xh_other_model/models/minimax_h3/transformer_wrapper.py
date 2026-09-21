import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward

from xhquant import nn as xhnn
from xhquant.utils.registry.dynamic_module import DynamicModule

from .transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    _ATTN_FIXED_SCALE,
    _FFN_FIXED_SCALE,
    MiniMaxH3AdaLayerNormModulation,
    MiniMaxH3AdaLayerNormOut,
    MiniMaxH3Attention,
    MiniMaxH3TokenRefiner,
    MiniMaxH3TokenRefinerBlock,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerBlock,
    MiniMaxH3TransformerOutput,
)

# Denoise plans expose at most three distinct noise levels per step
# (video_t, audio_t, and a condition/reference level such as 0.999/1.0).
# Export/HMONNX freezes temb to this leading dim; unused pad rows are never indexed.
MINIMAX_H3_FIXED_TEMB_TIMESTEPS = 3


def pad_minimax_h3_temb(
    temb: torch.Tensor,
    num_timesteps: int = MINIMAX_H3_FIXED_TEMB_TIMESTEPS,
) -> torch.Tensor:
    """Pad ``temb`` on dim0 to a fixed export/HMONNX capacity.

    Only the leading ``unique_timesteps`` rows are meaningful. Pad rows are zeros and must
    stay unreferenced by ``timestep_indices`` / ``adaln_indices``.
    """
    if num_timesteps < 1:
        raise ValueError(f"`num_timesteps` must be >= 1, got {num_timesteps}.")
    if temb.ndim != 2:
        raise ValueError(f"`temb` must be rank-2 `(num_timesteps, time_embed_dim)`, got {tuple(temb.shape)}.")
    current = int(temb.shape[0])
    if current == num_timesteps:
        return temb
    if current > num_timesteps:
        raise ValueError(
            f"`temb` has {current} distinct timesteps, which exceeds the fixed export capacity {num_timesteps}."
        )
    pad = temb.new_zeros((num_timesteps - current, temb.shape[1]))
    return torch.cat([temb, pad], dim=0)


def build_minimax_h3_transformer_conditioning(
    transformer: nn.Module,
    *,
    timestep: torch.Tensor,
    timestep_indices: torch.Tensor,
    token_tags: torch.Tensor,
    position_ids: torch.Tensor,
    fixed_temb_timesteps: int | None = MINIMAX_H3_FIXED_TEMB_TIMESTEPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Host-side precompute for the HMONNX/export transformer ABI."""
    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError(f"`position_ids` must be `(seq_len, 3)`, got {list(position_ids.shape)}.")
    sequence_length = position_ids.shape[0]
    if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
        raise ValueError(
            "`token_tags` and `timestep_indices` must both be `(seq_len,)` matching `position_ids`, got "
            f"{list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}."
        )
    if timestep.ndim != 1 or timestep.numel() < 1:
        raise ValueError(f"`timestep` must be a non-empty 1-D unique-timestep vector, got {tuple(timestep.shape)}.")
    if int(timestep_indices.min()) < 0 or int(timestep_indices.max()) >= int(timestep.numel()):
        raise ValueError(
            "`timestep_indices` must index into `timestep`, got "
            f"min={int(timestep_indices.min())} max={int(timestep_indices.max())} "
            f"for num_timesteps={int(timestep.numel())}."
        )
    if (
        transformer.time_embedder.linear_1.weight.device != timestep.device
        or transformer.time_embedder.linear_1.weight.dtype != timestep.dtype
    ):
        transformer.time_proj = transformer.time_proj.float().to(timestep.device)
        transformer.time_embedder = transformer.time_embedder.float().to(timestep.device)
        transformer.rope = transformer.rope.float().to(timestep.device)

    temb = transformer.time_embedder(transformer.time_proj(timestep))
    if fixed_temb_timesteps is not None:
        temb = pad_minimax_h3_temb(temb, fixed_temb_timesteps)
    # token_tags are layout-fixed; timestep_indices change every denoise step, so
    # adaln_indices are NOT constant across the schedule for a packed sequence.
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
    rotary_cos, rotary_sin = transformer.rope(position_ids)
    return temb.half(), adaln_indices.to(torch.int32), rotary_cos.half(), rotary_sin.half()


def build_packed_attention_mask(
    encoder_attention_mask: torch.Tensor,
    text_indices: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    """Host-side expand of text key-padding bias onto the packed sequence.

    ``encoder_attention_mask`` stays text-only for ``token_refiner``. The packed mask
    writes those biases at ``text_indices`` and leaves video/audio rows as 0.
    """
    if encoder_attention_mask.ndim != 4:
        raise ValueError(
            "`encoder_attention_mask` must be rank-4 `(batch, 1, 1, num_text_tokens)`, "
            f"got {tuple(encoder_attention_mask.shape)}."
        )
    if sequence_length < 1:
        raise ValueError(f"`sequence_length` must be >= 1, got {sequence_length}.")
    text_length = int(encoder_attention_mask.shape[-1])
    if text_indices.ndim != 1 or int(text_indices.numel()) != text_length:
        raise ValueError(
            "`text_indices` must be 1-D with length equal to `encoder_attention_mask`'s "
            f"last dim ({text_length}), got {tuple(text_indices.shape)}."
        )
    if int(text_indices.min()) < 0 or int(text_indices.max()) >= sequence_length:
        raise ValueError(
            "`text_indices` must index into the packed sequence "
            f"[0, {sequence_length}), got min={int(text_indices.min())} "
            f"max={int(text_indices.max())}."
        )
    packed_attention_mask = encoder_attention_mask.new_zeros(
        (encoder_attention_mask.shape[0], 1, 1, int(sequence_length))
    )
    return packed_attention_mask.index_copy(
        -1,
        text_indices.to(dtype=torch.long),
        encoder_attention_mask,
    )


def build_pack_gather_indices(
    text_indices: torch.Tensor,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    sequence_length: int | None = None,
) -> torch.Tensor:
    """Host-side inverse permutation for Concat([text, video, audio]) packing.

    Export packing concatenates modality streams in fixed order
    ``[text | video | audio]``, then ``Gather``s with this index vector so the
    packed sequence matches arbitrary (possibly non-contiguous) layout indices
    without ``ScatterElements``.
    """
    for name, indices in (
        ("text_indices", text_indices),
        ("video_indices", video_indices),
        ("audio_indices", audio_indices),
    ):
        if indices.ndim != 1:
            raise ValueError(f"`{name}` must be 1-D, got {tuple(indices.shape)}.")
    text_len = int(text_indices.numel())
    video_len = int(video_indices.numel())
    audio_len = int(audio_indices.numel())
    total = text_len + video_len + audio_len
    if sequence_length is None:
        sequence_length = total
    if sequence_length != total:
        raise ValueError(
            "Packing indices must partition the packed sequence: "
            f"text({text_len}) + video({video_len}) + audio({audio_len}) != {sequence_length}."
        )
    if sequence_length < 1:
        raise ValueError(f"`sequence_length` must be >= 1, got {sequence_length}.")

    device = text_indices.device
    dtype = torch.int64
    all_indices = torch.cat(
        [
            text_indices.to(device=device, dtype=dtype),
            video_indices.to(device=device, dtype=dtype),
            audio_indices.to(device=device, dtype=dtype),
        ],
        dim=0,
    )
    if int(all_indices.min()) < 0 or int(all_indices.max()) >= sequence_length:
        raise ValueError(
            "Packing indices must stay inside "
            f"[0, {sequence_length}), got min={int(all_indices.min())} max={int(all_indices.max())}."
        )
    if int(torch.unique(all_indices).numel()) != sequence_length:
        raise ValueError("Packing indices must be a permutation of the packed sequence positions.")

    # packed[i] = concat[pack_gather_indices[i]], where concat = [text|video|audio].
    pack_gather_indices = torch.empty(sequence_length, device=device, dtype=dtype)
    pack_gather_indices[text_indices.to(dtype=dtype)] = torch.arange(text_len, device=device, dtype=dtype)
    pack_gather_indices[video_indices.to(dtype=dtype)] = text_len + torch.arange(
        video_len, device=device, dtype=dtype
    )
    pack_gather_indices[audio_indices.to(dtype=dtype)] = text_len + video_len + torch.arange(
        audio_len, device=device, dtype=dtype
    )
    return pack_gather_indices.to(torch.int32)


def _move_module(module: nn.Module, device: torch.device | str) -> None:
    module.to(device)


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def prepare_transformer_block_cpu_offload(transformer: nn.Module, compute_device: torch.device | str) -> None:
    """Keep shared modules on ``compute_device`` and park every transformer block on CPU."""
    compute_device = torch.device(compute_device)
    for child_name, child in transformer.named_children():
        if child_name == "transformer_blocks":
            continue
        _move_module(child, compute_device)
    for block in transformer.transformer_blocks:
        _move_module(block, "cpu")
    transformer.enable_block_cpu_offload = True
    _empty_cache(compute_device)


def disable_transformer_block_cpu_offload(transformer: nn.Module) -> None:
    transformer.enable_block_cpu_offload = False


def _as_xhnn_rmsnorm(norm: nn.Module) -> xhnn.RMSNorm:
    """Convert torch.nn.RMSNorm to export-friendly xhnn.RMSNorm, preserving weights."""
    hidden_size = int(norm.normalized_shape[0])
    eps = float(getattr(norm, "eps", 1e-5))
    replaced = xhnn.RMSNorm(hidden_size, eps=eps)
    if getattr(norm, "weight", None) is not None:
        replaced.weight = nn.Parameter(norm.weight.data.clone())
    else:
        replaced.weight = nn.Parameter(torch.ones(hidden_size))
    return replaced


# class _MiniMaxH3RotaryPosEmbed(DynamicModule):
#     def _setup(self, *args, **kwargs):
#         del args, kwargs
#         self.position_cast = xhnn.Cast(torch.float32, follow_main_dtype=False)
#         self.position_unsqueeze = xhnn.Unsqueeze(axes=(-1,))
#         self.frequency_mul = xhnn.Mul()
#         self.frequency_reshape = xhnn.Reshape()
#         self.frequency_concat = xhnn.Concat(dim=-1)
#         self.frequency_cos = xhnn.Cos()
#         self.frequency_sin = xhnn.Sin()
#         return self

#     def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         position_ids = self.position_cast(position_ids)
#         position_ids = self.position_unsqueeze(position_ids)
#         frequencies = self.frequency_mul(position_ids, self.inv_freq)
#         frequencies = self.frequency_reshape(frequencies, (-1, 3 * self.rope_freq_dim))
#         frequencies = self.frequency_concat(frequencies, frequencies)
#         return self.frequency_cos(frequencies), self.frequency_sin(frequencies)


class _MiniMaxH3ApplyRotaryEmb(nn.Module):
    def __init__(self, rotary_dim: int):
        super().__init__()
        self.hidden_rotary_slice = xhnn.Slice([0], [rotary_dim], [-1], [1])
        self.hidden_pass_slice = xhnn.Slice([rotary_dim], [sys.maxsize], [-1], [1])
        self.cos_unsqueeze = xhnn.Unsqueeze(axes=(0, 2))
        self.sin_unsqueeze = xhnn.Unsqueeze(axes=(0, 2))
        self.rope = xhnn.Rope()
        self.output_concat = xhnn.Concat(dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_rotary = self.hidden_rotary_slice(hidden_states)
        hidden_states_pass = self.hidden_pass_slice(hidden_states)
        cos = self.cos_unsqueeze(cos)
        sin = self.sin_unsqueeze(sin)
        hidden_states_rotary = self.rope(hidden_states_rotary, cos, sin)
        return self.output_concat(hidden_states_rotary, hidden_states_pass)


class _MiniMaxH3AdaLayerNormModulation(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.output_slices = nn.ModuleList(
            [
                xhnn.Slice(
                    [output_index * self.hidden_size],
                    [(output_index + 1) * self.hidden_size],
                    [-1],
                    [1],
                )
                for output_index in range(6)
            ]
        )
        return self

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        temb = self.linear(F.silu(temb))
        temb = temb.view(-1, 6 * self.hidden_size)
        return tuple(output_slice(temb) for output_slice in self.output_slices)


class _MiniMaxH3AdaLayerNormOut(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        hidden_size = int(self.norm.normalized_shape[0])
        self.shift_slice = xhnn.Slice([0], [hidden_size], [-1], [1])
        self.scale_slice = xhnn.Slice([hidden_size], [2 * hidden_size], [-1], [1])
        # Keep AdaLNOut modulation; only swap the inner nn.RMSNorm.
        self.norm_h = _as_xhnn_rmsnorm(self.norm)
        # xhquant/HMONNX: Gather(axis=0) replaces torch.index_select(0, ...).
        self.scale_gather = xhnn.Gather(axis=0)
        self.shift_gather = xhnn.Gather(axis=0)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        modulation = self.linear(F.silu(temb))
        shift = self.shift_slice(modulation)
        scale = self.scale_slice(modulation)
        hidden_states = self.norm_h(hidden_states)
        scale = 1.0 + self.scale_gather(scale, timestep_indices)
        shift = self.shift_gather(shift, timestep_indices)
        return hidden_states * scale + shift


class _MiniMaxH3AttentionBase(DynamicModule):
    """Shared export attention body; subclasses decide whether to apply /16."""

    _apply_attn_fixed_scale = True

    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.scale = self.head_dim**-0.5
        self.norm_q = _as_xhnn_rmsnorm(self.norm_q)
        self.norm_k = _as_xhnn_rmsnorm(self.norm_k)
        self.apply_rotary_emb = _MiniMaxH3ApplyRotaryEmb(self.rotary_dim)
        self.qkv_reshape = xhnn.Reshape()
        self.output_reshape = xhnn.Reshape()
        self.masked_add = xhnn.MaskedAdd()
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._apply_attn_fixed_scale:
            # Compensated later by FixedPointMulAdd(rhs_scale=16) in main transformer blocks.
            hidden_states = hidden_states * (1.0 / _ATTN_FIXED_SCALE)

        batch_size, sequence_length = hidden_states.shape[:2]
        qkv_shape = (batch_size, sequence_length, self.heads, self.head_dim)
        query = self.qkv_reshape(self.to_q(hidden_states), qkv_shape)
        key = self.qkv_reshape(self.to_k(hidden_states), qkv_shape)
        value = self.qkv_reshape(self.to_v(hidden_states), qkv_shape)
        query, key = self.norm_q(query), self.norm_k(key)
        if rotary_emb is not None:
            query = self.apply_rotary_emb(query, *rotary_emb)
            key = self.apply_rotary_emb(key, *rotary_emb)
        scores = torch.matmul(query.transpose(1, 2), key.transpose(1, 2).transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            # Export graph uses MaskedAdd (double-add + saturate), same as text/vision wrappers.
            # Host mask stays additive key-padding bias: valid=0, pad=finfo(dtype).min.
            scores = self.masked_add(scores, attention_mask)
        hidden_states = torch.matmul(torch.softmax(scores, dim=-1), value.transpose(1, 2))
        hidden_states = hidden_states.transpose(1, 2)
        output_shape = (batch_size, sequence_length, self.heads * self.head_dim)
        hidden_states = self.output_reshape(hidden_states, output_shape)
        return self.to_out[0](hidden_states)


class _MiniMaxH3Attention(_MiniMaxH3AttentionBase):
    """Main-block attention: /16 pre-scale + residual rhs_scale=16 compensation."""

    _apply_attn_fixed_scale = True


class _MiniMaxH3TokenRefinerAttention(_MiniMaxH3AttentionBase):
    """TokenRefiner attention: plain residual add, so no /16 pre-scale."""

    _apply_attn_fixed_scale = False


class _MiniMaxH3TokenRefinerBlock(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.norm1_h = _as_xhnn_rmsnorm(self.norm1)
        self.norm2_h = _as_xhnn_rmsnorm(self.norm2)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.attn(self.norm1_h(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states.clip(min=-65504.0, max=65504.0)
        feed_forward_output = self.ff(self.norm2_h(hidden_states))
        out = hidden_states + feed_forward_output
        return out.clip(min=-65504.0, max=65504.0)


class _MiniMaxH3TokenRefiner(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.final_norm_h = _as_xhnn_rmsnorm(self.final_norm)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask)
        return self.final_norm_h(hidden_states)


class _MiniMaxH3FeedForward(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        inner_dim = self.net[0].proj.out_features // 2
        self.hidden_slice = xhnn.Slice([0], [inner_dim], [-1], [1])
        self.gate_slice = xhnn.Slice([inner_dim], [2 * inner_dim], [-1], [1])
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        swiglu = self.net[0]
        projected = swiglu.proj(hidden_states)
        hidden = self.hidden_slice(projected)
        gate = self.gate_slice(projected)
        activation = swiglu.activation(gate) / 16
        hidden_states = hidden * activation
        hidden_states = hidden_states / 8
        return self.net[2](hidden_states)


class _MiniMaxH3FeedForward_ffn(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        inner_dim = self.net[0].proj.out_features // 2
        self.hidden_slice = xhnn.Slice([0], [inner_dim], [-1], [1])
        self.gate_slice = xhnn.Slice([inner_dim], [2 * inner_dim], [-1], [1])
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        swiglu = self.net[0]
        projected = swiglu.proj(hidden_states)
        hidden = self.hidden_slice(projected)
        gate = self.gate_slice(projected)
        activation = swiglu.activation(gate)
        hidden_states = hidden * activation
        hidden_states = hidden_states
        return self.net[2](hidden_states)


class _MiniMaxH3TransformerBlock(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.norm1_h = _as_xhnn_rmsnorm(self.norm1)
        self.norm2_h = _as_xhnn_rmsnorm(self.norm2)
        # xhquant/HMONNX: Gather(axis=0) replaces torch.index_select(0, ...).
        self.adaln_gather = xhnn.Gather(axis=0)
        self._refresh_fixed_point_ops()
        return self

    def _refresh_fixed_point_ops(self) -> None:
        point1_config = self.fixed_point_configs[1]
        point3_config = self.fixed_point_configs[3]
        self.fixed_point_mul_add_1 = xhnn.FixedPointMulAdd(
            point1_config,
            rhs_scale=int(_ATTN_FIXED_SCALE),
            input_scale=self.input_residual_scale,
        )
        self.fixed_point_mul_add_3 = xhnn.FixedPointMulAdd(
            point3_config,
            rhs_scale=int(_FFN_FIXED_SCALE),
            input_scale=point1_config[4] - point1_config[2],
        )

    def _fixed_mul_add(
        self,
        point_index: int,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        bias: torch.Tensor | None = None,
        input_scale: int = 0,
    ) -> torch.Tensor:
        if self.fixed_point_calibration_enabled:
            return MiniMaxH3TransformerBlock._fixed_mul_add(
                self,
                point_index,
                lhs,
                rhs,
                bias,
                input_scale,
            )
        if point_index == 1:
            return self.fixed_point_mul_add_1(lhs, rhs, bias)
        if point_index == 3:
            return self.fixed_point_mul_add_3(lhs, rhs, bias)
        raise ValueError(f"Unsupported MiniMax-H3 fixed-point residual point: {point_index}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
        residual = hidden_states
        norm_hidden_states = self.norm1_h(hidden_states)
        norm_hidden_states = norm_hidden_states * (1.0 + self.adaln_gather(scale_msa, adaln_indices))
        norm_hidden_states = norm_hidden_states.clip(min=-65504.0, max=65504.0)
        norm_hidden_states = norm_hidden_states + self.adaln_gather(shift_msa, adaln_indices)
        norm_hidden_states = norm_hidden_states.clip(min=-65504.0, max=65504.0)
        attention_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
        hidden_states = self._fixed_mul_add(
            1,
            self.adaln_gather(gate_msa, adaln_indices),
            attention_output,
            residual,
            input_scale=self.input_residual_scale,
        )

        residual = hidden_states
        norm_hidden_states = self.norm2_h(hidden_states)
        norm_hidden_states = norm_hidden_states * (1.0 + self.adaln_gather(scale_mlp, adaln_indices))
        norm_hidden_states = norm_hidden_states.clip(min=-65504.0, max=65504.0)
        norm_hidden_states = norm_hidden_states + self.adaln_gather(shift_mlp, adaln_indices)
        norm_hidden_states = norm_hidden_states.clip(min=-65504.0, max=65504.0)
        feed_forward_output = self.ff(norm_hidden_states)
        return self._fixed_mul_add(
            3,
            self.adaln_gather(gate_mlp, adaln_indices),
            feed_forward_output,
            residual,
            input_scale=self.fixed_point_configs[1][4] - self.fixed_point_configs[1][2],
        )


class _MiniMaxH3Transformer3DModel(DynamicModule):
    # FP warm-up / debug only. Must stay False during Dynamo capture so `.to()` is not traced.
    enable_block_cpu_offload = False

    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.enable_block_cpu_offload = False
        # xhquant/HMONNX: Gather(axis=1) replaces torch.index_select(1, ...).
        self.video_gather = xhnn.Gather(axis=1)
        self.audio_gather = xhnn.Gather(axis=1)
        # Avoid ScatterElements packing: concat modality streams in fixed order, then
        # Gather with host-built pack_gather_indices (inverse of text/video/audio indices).
        self.pack_concat = xhnn.Concat(dim=1)
        self.pack_gather = xhnn.Gather(axis=1)
        return self

    def set_fixed_point_configs(self, fixed_point_configs) -> None:
        MiniMaxH3Transformer3DModel.set_fixed_point_configs(self, fixed_point_configs)
        self._refresh_fixed_point_ops()

    def finish_fixed_point_calibration(self, safety_margin=2, amax_scale=1.0):
        configs = MiniMaxH3Transformer3DModel.finish_fixed_point_calibration(
            self,
            safety_margin=safety_margin,
            amax_scale=amax_scale,
        )
        self._refresh_fixed_point_ops()
        return configs

    def _refresh_fixed_point_ops(self) -> None:
        for layer_index, block in enumerate(self.transformer_blocks):
            if layer_index:
                previous_config = self.transformer_blocks[layer_index - 1].fixed_point_configs[3]
                block.input_residual_scale = previous_config[4] - previous_config[2]
            block._refresh_fixed_point_ops()

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        pack_gather_indices: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        packed_attention_mask: torch.Tensor,
        return_dict: bool = True,
    ):
        """Export/HMONNX ABI: host provides temb/adaln/rotary + pack gather + masks.

        ``text_indices`` stay host-only (used to build ``pack_gather_indices`` /
        ``packed_attention_mask``). Keeping them as unused forward args makes Dynamo
        drop the placeholder and break ``input_names`` vs ``example_inputs`` arity.
        """
        sequence_length = int(adaln_indices.shape[0])
        if timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "`timestep_indices` must match `adaln_indices` length, got "
                f"{tuple(timestep_indices.shape)} vs {(sequence_length,)}."
            )
        if rotary_cos.shape[0] != sequence_length or rotary_sin.shape[0] != sequence_length:
            raise ValueError(
                "`rotary_cos`/`rotary_sin` must both be `(seq_len, rotary_dim)` matching `adaln_indices`, got "
                f"{tuple(rotary_cos.shape)} and {tuple(rotary_sin.shape)} for seq_len={sequence_length}."
            )
        if pack_gather_indices.shape != (sequence_length,):
            raise ValueError(
                "`pack_gather_indices` must be `(seq_len,)` matching `adaln_indices`, got "
                f"{tuple(pack_gather_indices.shape)} for seq_len={sequence_length}."
            )
        expected_mask_shape = (encoder_hidden_states.shape[0], 1, 1, encoder_hidden_states.shape[1])
        if tuple(encoder_attention_mask.shape) != expected_mask_shape:
            raise ValueError(
                "`encoder_attention_mask` must be an additive bias with shape "
                f"{expected_mask_shape}, got {tuple(encoder_attention_mask.shape)}."
            )
        expected_packed_mask_shape = (encoder_hidden_states.shape[0], 1, 1, sequence_length)
        if tuple(packed_attention_mask.shape) != expected_packed_mask_shape:
            raise ValueError(
                "`packed_attention_mask` must be an additive bias with shape "
                f"{expected_packed_mask_shape}, got {tuple(packed_attention_mask.shape)}."
            )

        compute_device = hidden_states.device
        block_offload = bool(getattr(self, "enable_block_cpu_offload", False))

        rotary_emb = (rotary_cos, rotary_sin)
        video_embeds = self.proj_in(hidden_states)
        audio_embeds = self.audio_proj_in(audio_hidden_states)
        text_embeds = self.context_embedder(encoder_hidden_states / 2)
        text_embeds = self.token_refiner(text_embeds, encoder_attention_mask)

        # ScatterElements is hardware-unfriendly. Concat modality streams in fixed
        # [text|video|audio] order, then Gather with host inverse permutation so
        # non-contiguous fl2va/ref2va layouts stay correct.
        packed_modalities = self.pack_concat(text_embeds, video_embeds, audio_embeds)
        hidden_states = self.pack_gather(packed_modalities, pack_gather_indices)

        for layer_index, block in enumerate(self.transformer_blocks):
            if layer_index:
                previous_config = self.transformer_blocks[layer_index - 1].fixed_point_configs[3]
                block.input_residual_scale = previous_config[4] - previous_config[2]
            if block_offload:
                _move_module(block, compute_device)
            try:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        temb,
                        adaln_indices,
                        rotary_emb,
                        packed_attention_mask,
                    )
                else:
                    hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, packed_attention_mask)
            finally:
                if block_offload:
                    _move_module(block, "cpu")
                    _empty_cache(compute_device)

        hidden_states = self.norm_out(hidden_states, temb, timestep_indices)
        video_output = self.video_gather(self.proj_out(hidden_states), video_indices)
        audio_output = self.audio_gather(self.audio_proj_out(hidden_states), audio_indices)
        if not return_dict:
            return (video_output, audio_output)
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


_TRANSFORMER_WRAPPERS = {
    MiniMaxH3AdaLayerNormModulation: _MiniMaxH3AdaLayerNormModulation,
    MiniMaxH3AdaLayerNormOut: _MiniMaxH3AdaLayerNormOut,
    # Attention is converted explicitly below: main blocks keep /16, TokenRefiner does not.
    # MiniMaxH3RotaryPosEmbed: _MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TokenRefiner: _MiniMaxH3TokenRefiner,
    MiniMaxH3TokenRefinerBlock: _MiniMaxH3TokenRefinerBlock,
    MiniMaxH3TransformerBlock: _MiniMaxH3TransformerBlock,
    MiniMaxH3Transformer3DModel: _MiniMaxH3Transformer3DModel,
}


def wrap_transformer(transformer: nn.Module) -> nn.Module:
    rotary_dim = 6 * transformer.rope.rope_freq_dim
    for block in transformer.transformer_blocks:
        if type(block.ff) is FeedForward:
            _MiniMaxH3FeedForward.convert(block.ff)
        if type(block.attn) is MiniMaxH3Attention:
            block.attn.rotary_dim = rotary_dim
            _MiniMaxH3Attention.convert(block.attn)
    for block in transformer.token_refiner.refiner_blocks:
        if type(block.ff) is FeedForward:
            _MiniMaxH3FeedForward_ffn.convert(block.ff)
        if type(block.attn) is MiniMaxH3Attention:
            block.attn.rotary_dim = rotary_dim
            # Mirror FFN split: TokenRefiner residual has no rhs_scale=16 compensation.
            _MiniMaxH3TokenRefinerAttention.convert(block.attn)
    for _, module in list(transformer.named_modules()):
        dynamic_cls = _TRANSFORMER_WRAPPERS.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            dynamic_cls.convert(module)
    transformer._refresh_fixed_point_ops()
    return transformer


class MiniMaxH3TransformerWrapper(nn.Module):
    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = wrap_transformer(transformer)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        pack_gather_indices: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        packed_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep export/Dynamo placeholder names aligned with TENSOR_NAMES / host ABI.
        # Do not pass unused host-only tensors (e.g. text_indices): Dynamo drops them.
        return self.transformer(
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            temb,
            timestep_indices,
            adaln_indices,
            rotary_cos,
            rotary_sin,
            video_indices,
            audio_indices,
            pack_gather_indices,
            encoder_attention_mask,
            packed_attention_mask,
            return_dict=False,
        )


__all__ = [
    "MiniMaxH3TransformerWrapper",
    "build_minimax_h3_transformer_conditioning",
    "build_pack_gather_indices",
    "build_packed_attention_mask",
    "disable_transformer_block_cpu_offload",
    "prepare_transformer_block_cpu_offload",
    "wrap_transformer",
]