from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionMLP,
    Qwen3VLVisionModel,
    Qwen3VLVisionPatchMerger,
)
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from xhquant.nn import MaskedAdd, Rope
from xhquant.utils.registry.dynamic_module import DynamicModule


DEFAULT_VISION_MAX_PATCHES = 4352


class _Qwen3VLVisionMLP(DynamicModule):
    def _setup(self, cfg=None, *args, **kwargs):
        del cfg, args, kwargs
        return self

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class _Qwen3VLVisionAttention(DynamicModule):
    """Eager dense attention with additive key-padding mask for padded vision export."""

    def _setup(self, cfg=None, *args, **kwargs):
        del cfg, args, kwargs
        self.rope = Rope()
        self.masked_add = MaskedAdd()

        weight = self.qkv.weight.data.clone()
        bias = self.qkv.bias.data.clone()
        dim = weight.shape[0] // 3
        weight = weight.permute(1, 0).reshape(dim, 3, dim).permute(1, 2, 0)
        bias = bias.reshape(3, dim)

        device = self.qkv.weight.device
        dtype = self.qkv.weight.dtype
        self.q_proj = nn.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.k_proj = nn.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.v_proj = nn.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.q_proj.weight.data.copy_(weight[0])
        self.q_proj.bias.data.copy_(bias[0])
        self.k_proj.weight.data.copy_(weight[1])
        self.k_proj.bias.data.copy_(bias[1])
        self.v_proj.weight.data.copy_(weight[2])
        self.v_proj.bias.data.copy_(bias[2])
        del self.qkv
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_seqlen: int | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del cu_seqlens, max_seqlen, kwargs
        if position_embeddings is None:
            raise ValueError("Qwen3-VL vision attention requires position_embeddings")
        if attention_mask is None:
            raise ValueError("padded MiniMax-H3 vision attention requires attention_mask")

        seq_length = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        cos, sin = position_embeddings
        query_states = query_states.unsqueeze(0)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)
        if cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif cos.ndim == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        query_states = self.rope(query_states, cos, sin)
        key_states = self.rope(key_states, cos, sin)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling
        scores = self.masked_add(scores, attention_mask)
        probabilities = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(probabilities, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class _Qwen3VLVisionBlock(DynamicModule):
    def _setup(self, cfg=None, *args, **kwargs):
        del cfg, args, kwargs
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class _Qwen3VLVisionPatchMerger(DynamicModule):
    def _setup(self, cfg=None, *args, **kwargs):
        del cfg, args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.norm(hidden_states).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


_VISION_ENCODER_WRAPPERS = {
    Qwen3VLVisionAttention: _Qwen3VLVisionAttention,
    Qwen3VLVisionBlock: _Qwen3VLVisionBlock,
    Qwen3VLVisionMLP: _Qwen3VLVisionMLP,
    Qwen3VLVisionPatchMerger: _Qwen3VLVisionPatchMerger,
}


def wrap_vision_encoder(vision_encoder: Qwen3VLVisionModel) -> Qwen3VLVisionModel:
    if not isinstance(vision_encoder, Qwen3VLVisionModel):
        raise TypeError(f"MiniMax-H3 vision encoder must be Qwen3VLVisionModel, got {type(vision_encoder)!r}")
    for _, module in list(vision_encoder.named_modules()):
        dynamic_cls = _VISION_ENCODER_WRAPPERS.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            dynamic_cls.convert(module)
    return vision_encoder


def _normalize_image_grid_thw(image_grid_thw: torch.Tensor) -> torch.Tensor:
    grid_thw = image_grid_thw.detach().to(dtype=torch.long)
    if grid_thw.ndim == 1 and grid_thw.numel() == 3:
        grid_thw = grid_thw.view(1, 3)
    if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
        raise ValueError(f"image_grid_thw must have shape [N, 3], got {tuple(image_grid_thw.shape)}")
    if int(grid_thw.shape[0]) != 1:
        raise ValueError(f"MiniMax-H3 vision export currently supports one image, got {int(grid_thw.shape[0])}")
    return grid_thw


def validate_vision_max_patches(max_patches: int, spatial_merge_size: int) -> int:
    max_patches = int(max_patches)
    spatial_merge_size = int(spatial_merge_size)
    if max_patches <= 0:
        raise ValueError(f"vision_max_patches must be positive, got {max_patches}")
    if spatial_merge_size <= 0:
        raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}")
    merge_unit = spatial_merge_size * spatial_merge_size
    if max_patches % merge_unit:
        raise ValueError(
            f"vision_max_patches={max_patches} must be divisible by spatial merge unit {merge_unit}"
        )
    return max_patches


def pad_vision_pixel_values(pixel_values: torch.Tensor, max_patches: int) -> tuple[torch.Tensor, int]:
    """Right-pad flattened vision patches to ``[max_patches, patch_dim]``."""

    if pixel_values.ndim != 2:
        raise ValueError(f"pixel_values must have shape [N, D], got {tuple(pixel_values.shape)}")
    valid_patches = int(pixel_values.shape[0])
    max_patches = int(max_patches)
    if valid_patches > max_patches:
        raise ValueError(f"pixel_values have {valid_patches} patches, which exceeds capacity {max_patches}")
    if valid_patches == max_patches:
        return pixel_values, valid_patches
    padding = pixel_values.new_zeros((max_patches - valid_patches, pixel_values.shape[1]))
    return torch.cat((pixel_values, padding), dim=0), valid_patches


def build_vision_attention_mask(
    valid_patches: int,
    max_patches: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build additive key-padding bias with shape ``[1, 1, 1, max_patches]``."""

    valid_patches = int(valid_patches)
    max_patches = int(max_patches)
    if valid_patches < 0 or valid_patches > max_patches:
        raise ValueError(f"valid_patches must be in [0, {max_patches}], got {valid_patches}")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError(f"attention mask dtype must be floating point, got {dtype}")
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    attention_mask = torch.zeros((1, 1, 1, max_patches), dtype=dtype, device=target_device)
    if valid_patches < max_patches:
        attention_mask[..., valid_patches:] = -torch.finfo(dtype).max
    return attention_mask


def build_vision_geometry_inputs(
    vision_encoder,
    image_grid_thw: torch.Tensor,
    *,
    max_patches: int = DEFAULT_VISION_MAX_PATCHES,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor | int]:
    """Build padded position / RoPE / attention inputs for one image grid."""

    grid_thw = _normalize_image_grid_thw(image_grid_thw)
    spatial_merge_size = int(vision_encoder.spatial_merge_size)
    max_patches = validate_vision_max_patches(max_patches, spatial_merge_size)
    valid_patches = int(grid_thw.prod(dim=-1).sum().item())
    if valid_patches > max_patches:
        raise ValueError(
            f"image grid {grid_thw.tolist()} needs {valid_patches} patches, "
            f"which exceeds vision_max_patches={max_patches}"
        )
    merge_unit = spatial_merge_size * spatial_merge_size
    if valid_patches % merge_unit:
        raise ValueError(
            f"valid patch count {valid_patches} must be divisible by spatial merge unit {merge_unit}"
        )

    target_device = (
        torch.device(device)
        if device is not None
        else vision_encoder.pos_embed.weight.device
    )
    target_dtype = dtype or vision_encoder.pos_embed.weight.dtype
    grid_thw = grid_thw.to(device=target_device)

    with torch.no_grad():
        interpolation_indices, interpolation_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=vision_encoder.num_grid_per_side,
            mode=vision_encoder.interpolation_mode,
            align_corners=vision_encoder.interpolation_align_corners,
            spatial_merge_size=vision_encoder.config.spatial_merge_size,
        )
        position_ids = get_vision_position_ids(grid_thw, vision_encoder.spatial_merge_size)
        position_embeddings = (
            vision_encoder.pos_embed(interpolation_indices) * interpolation_weights[:, :, None]
        ).sum(1)
        rotary_pos_emb = vision_encoder.rotary_pos_emb(position_ids)
        rotary_pos_emb = rotary_pos_emb.reshape(position_embeddings.shape[0], -1)
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_cos = rotary_pos_emb.cos().to(dtype=target_dtype)
        rotary_sin = rotary_pos_emb.sin().to(dtype=target_dtype)
        position_embeddings = position_embeddings.to(dtype=target_dtype)

    if position_embeddings.shape[0] != valid_patches:
        raise RuntimeError(
            f"position embeddings produced {position_embeddings.shape[0]} rows, expected {valid_patches}"
        )

    if valid_patches < max_patches:
        pad_rows = max_patches - valid_patches
        position_embeddings = F.pad(position_embeddings, (0, 0, 0, pad_rows), value=0.0)
        rotary_cos = F.pad(rotary_cos, (0, 0, 0, pad_rows), value=0.0)
        rotary_sin = F.pad(rotary_sin, (0, 0, 0, pad_rows), value=0.0)

    attention_mask = build_vision_attention_mask(
        valid_patches,
        max_patches,
        dtype=target_dtype,
        device=target_device,
    )
    return {
        "position_embeddings": position_embeddings,
        "rotary_cos": rotary_cos,
        "rotary_sin": rotary_sin,
        "attention_mask": attention_mask,
        "valid_patches": valid_patches,
        "valid_image_tokens": valid_patches // merge_unit,
        "max_patches": max_patches,
        "max_image_tokens": max_patches // merge_unit,
    }


def build_vision_encoder_graph_inputs(
    vision_encoder,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    max_patches: int = DEFAULT_VISION_MAX_PATCHES,
    dtype: torch.dtype | None = None,
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor | int]]:
    """Pad patches and build the full MiniMax-H3 vision HMONNX input tuple."""

    target_dtype = dtype or pixel_values.dtype
    padded_pixel_values, valid_patches = pad_vision_pixel_values(
        pixel_values.to(dtype=target_dtype),
        max_patches,
    )
    geometry = build_vision_geometry_inputs(
        vision_encoder,
        image_grid_thw,
        max_patches=max_patches,
        dtype=target_dtype,
        device=padded_pixel_values.device,
    )
    if int(geometry["valid_patches"]) != valid_patches:
        raise ValueError(
            f"pixel_values have {valid_patches} patches, but image_grid_thw implies "
            f"{int(geometry['valid_patches'])}"
        )
    inputs = (
        padded_pixel_values,
        geometry["position_embeddings"],
        geometry["rotary_cos"],
        geometry["rotary_sin"],
        geometry["attention_mask"],
    )
    return inputs, geometry


class MiniMaxH3VisionEncoderWrapper(nn.Module):
    """Qwen3-VL vision tower with a fixed max-patch padded export contract."""

    def __init__(
        self,
        vision_encoder: Qwen3VLVisionModel,
        max_patches: int | torch.Tensor = DEFAULT_VISION_MAX_PATCHES,
        image_grid_thw: torch.Tensor | None = None,
    ):
        super().__init__()
        if len(vision_encoder.deepstack_visual_indexes) != 3:
            raise ValueError(
                "MiniMax-H3 expects exactly three Qwen3-VL DeepStack outputs, "
                f"got {len(vision_encoder.deepstack_visual_indexes)}"
            )

        # Older call sites used ``MiniMaxH3VisionEncoderWrapper(model, image_grid_thw)``.
        if isinstance(max_patches, torch.Tensor):
            image_grid_thw = max_patches
            max_patches = DEFAULT_VISION_MAX_PATCHES
        del image_grid_thw

        self.visual = wrap_vision_encoder(vision_encoder)
        self.spatial_merge_size = int(self.visual.spatial_merge_size)
        self.max_patches = validate_vision_max_patches(max_patches, self.spatial_merge_size)
        self.max_image_tokens = self.max_patches // (self.spatial_merge_size * self.spatial_merge_size)
        # Keep the old attribute name as an alias for capacity-based checks/tests.
        self.expected_patch_count = self.max_patches

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_embeddings: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if pixel_values.ndim != 2 or pixel_values.shape[0] != self.max_patches:
            raise ValueError(
                f"pixel_values must have shape [{self.max_patches}, D], got {tuple(pixel_values.shape)}"
            )
        if position_embeddings.shape[0] != self.max_patches:
            raise ValueError(
                f"position_embeddings must have {self.max_patches} rows, got {tuple(position_embeddings.shape)}"
            )
        if rotary_cos.shape[0] != self.max_patches or rotary_sin.shape[0] != self.max_patches:
            raise ValueError(
                "rotary_cos/rotary_sin must match max_patches="
                f"{self.max_patches}, got {tuple(rotary_cos.shape)} / {tuple(rotary_sin.shape)}"
            )
        if tuple(attention_mask.shape) != (1, 1, 1, self.max_patches):
            raise ValueError(
                f"attention_mask must have shape [1, 1, 1, {self.max_patches}], got {tuple(attention_mask.shape)}"
            )

        hidden_states = self.visual.patch_embed(pixel_values)
        if hidden_states.shape[0] != self.max_patches:
            raise ValueError(
                f"pixel_values produce {hidden_states.shape[0]} patches, expected {self.max_patches}"
            )
        hidden_states = hidden_states + position_embeddings.to(dtype=hidden_states.dtype)
        rope_position_embeddings = (
            rotary_cos.to(dtype=hidden_states.dtype),
            rotary_sin.to(dtype=hidden_states.dtype),
        )
        deepstack_features = []
        for layer_index, block in enumerate(self.visual.blocks):
            hidden_states = block(
                hidden_states,
                position_embeddings=rope_position_embeddings,
                attention_mask=attention_mask.to(dtype=hidden_states.dtype),
            )
            if layer_index in self.visual.deepstack_visual_indexes:
                merger_index = self.visual.deepstack_visual_indexes.index(layer_index)
                deepstack_features.append(self.visual.deepstack_merger_list[merger_index](hidden_states))

        image_embeds = self.visual.merger(hidden_states)
        if image_embeds.shape[0] != self.max_image_tokens:
            raise RuntimeError(
                f"vision merger produced {image_embeds.shape[0]} tokens, expected {self.max_image_tokens}"
            )
        return image_embeds, deepstack_features[0], deepstack_features[1], deepstack_features[2]


__all__ = [
    "DEFAULT_VISION_MAX_PATCHES",
    "MiniMaxH3VisionEncoderWrapper",
    "build_vision_attention_mask",
    "build_vision_encoder_graph_inputs",
    "build_vision_geometry_inputs",
    "pad_vision_pixel_values",
    "validate_vision_max_patches",
    "wrap_vision_encoder",
]
