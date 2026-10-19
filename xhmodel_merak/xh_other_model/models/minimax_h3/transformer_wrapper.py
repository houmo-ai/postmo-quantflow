import sys
from collections.abc import Sequence

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

# fl2va denoise plans expose up to four distinct noise levels per step
# (video_t, audio_t, condition_video≈0.999, condition_audio=1.0) once video≠audio.
# Export/HMONNX freezes AdaLN tables to this leading unique-timestep capacity;
# unused pad rows are never indexed.
MINIMAX_H3_FIXED_TEMB_TIMESTEPS = 4
MINIMAX_H3_DEFAULT_NUM_INFERENCE_STEPS = 4
MINIMAX_H3_DEFAULT_VIDEO_SHIFT = 12.0
MINIMAX_H3_DEFAULT_AUDIO_SHIFT = 3.0
MINIMAX_H3_DEFAULT_KEYFRAME_NOISE_AUG = 0.999
_MINIMAX_FIXED_ADALN_NAMES = (
    "minimax_fixed_shift_msa",
    "minimax_fixed_scale_msa",
    "minimax_fixed_gate_msa",
    "minimax_fixed_shift_mlp",
    "minimax_fixed_scale_mlp",
    "minimax_fixed_gate_mlp",
)


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


def _resolve_minimax_h3_scheduler_cls():
    try:
        from diffusers.schedulers import MiniMaxH3Scheduler
    except Exception as exc:  # pragma: no cover - environment-dependent import path
        raise ImportError(
            "MiniMaxH3Scheduler is required to bake AdaLN LUTs. Install/upgrade the local diffusers MiniMax-H3 package."
        ) from exc
    return MiniMaxH3Scheduler


def _resolve_minimax_h3_row_timestep_builder():
    try:
        from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep
    except Exception as exc:  # pragma: no cover - environment-dependent import path
        raise ImportError("MiniMaxH3SetTimestepsStep.build_row_timesteps is required to bake AdaLN LUTs.") from exc
    return MiniMaxH3SetTimestepsStep.build_row_timesteps


def build_minimax_h3_denoise_unique_timesteps(
    *,
    num_inference_steps: int = MINIMAX_H3_DEFAULT_NUM_INFERENCE_STEPS,
    video_shift: float = MINIMAX_H3_DEFAULT_VIDEO_SHIFT,
    audio_shift: float = MINIMAX_H3_DEFAULT_AUDIO_SHIFT,
    keyframe_noise_aug: float = MINIMAX_H3_DEFAULT_KEYFRAME_NOISE_AUG,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Build unique-timestep vectors for every denoise evaluation step.

    Matches the fl2va host schedule: video ``shift=12``, audio ``shift=3``,
    ``condition_video=max(t, keyframe_noise_aug)``, ``condition_audio=1.0``.
    """
    if int(num_inference_steps) < 2:
        raise ValueError(f"`num_inference_steps` must be >= 2, got {num_inference_steps}.")
    scheduler_cls = _resolve_minimax_h3_scheduler_cls()
    build_row_timesteps = _resolve_minimax_h3_row_timestep_builder()
    video_scheduler = scheduler_cls(shift=float(video_shift))
    audio_scheduler = scheduler_cls(shift=float(audio_shift))
    video_scheduler.set_timesteps(int(num_inference_steps))
    audio_scheduler.set_timesteps(int(num_inference_steps))
    video_ts = [float(t) for t in video_scheduler.timesteps]
    audio_ts = [float(t) for t in audio_scheduler.timesteps]
    if len(video_ts) != len(audio_ts):
        raise RuntimeError(
            f"Video/audio schedules disagree: video={len(video_ts)} audio={len(audio_ts)} "
            f"for num_inference_steps={num_inference_steps}."
        )

    # Minimal packed layout that still exercises all four noise levels:
    # video=[condition, generated], audio=[condition, generated], plus one text token.
    device = torch.device(device)
    video_indices = torch.tensor([0, 1], device=device, dtype=torch.long)
    audio_indices = torch.tensor([2, 3], device=device, dtype=torch.long)
    unique_list: list[torch.Tensor] = []
    for video_t, audio_t in zip(video_ts, audio_ts):
        unique, _ = build_row_timesteps(
            video_indices=video_indices,
            audio_indices=audio_indices,
            num_condition_video_rows=1,
            num_condition_audio_rows=1,
            num_text_tokens=1,
            video_timestep=float(video_t),
            audio_timestep=float(audio_t),
            condition_video_timestep=max(float(video_t), float(keyframe_noise_aug)),
            condition_audio_timestep=1.0,
        )
        unique_list.append(unique.to(device=device, dtype=dtype).contiguous())
    return unique_list


def build_minimax_h3_adaln_lut(
    transformer: nn.Module,
    *,
    num_inference_steps: int = MINIMAX_H3_DEFAULT_NUM_INFERENCE_STEPS,
    video_shift: float = MINIMAX_H3_DEFAULT_VIDEO_SHIFT,
    audio_shift: float = MINIMAX_H3_DEFAULT_AUDIO_SHIFT,
    keyframe_noise_aug: float = MINIMAX_H3_DEFAULT_KEYFRAME_NOISE_AUG,
    fixed_temb_timesteps: int = MINIMAX_H3_FIXED_TEMB_TIMESTEPS,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float16,
) -> dict[str, object]:
    """Precompute AdaLN / final-norm tables indexed by denoise step and block id."""
    if not hasattr(transformer, "transformer_blocks"):
        raise TypeError("Expected a MiniMax-H3 transformer with `transformer_blocks`.")
    if int(fixed_temb_timesteps) < 1:
        raise ValueError(f"`fixed_temb_timesteps` must be >= 1, got {fixed_temb_timesteps}.")

    compute_device = torch.device(device) if device is not None else next(transformer.parameters()).device
    unique_timesteps = build_minimax_h3_denoise_unique_timesteps(
        num_inference_steps=num_inference_steps,
        video_shift=video_shift,
        audio_shift=audio_shift,
        keyframe_noise_aug=keyframe_noise_aug,
        device=compute_device,
        dtype=torch.float32,
    )
    num_steps = len(unique_timesteps)
    num_blocks = len(transformer.transformer_blocks)
    first_adaln = transformer.transformer_blocks[0].adaln_proj
    if hasattr(first_adaln, "linear"):
        hidden_size = int(first_adaln.linear.out_features) // (6 * MINIMAX_H3_MODALITY_NUM)
        time_embed_dim = int(first_adaln.linear.in_features)
    else:
        hidden_size = int(first_adaln.out_features) // (6 * MINIMAX_H3_MODALITY_NUM)
        time_embed_dim = int(first_adaln.in_features)
    adaln_rows = int(fixed_temb_timesteps) * MINIMAX_H3_MODALITY_NUM

    # Ensure host-side time / final-norm modules live on the compute device for the bake pass.
    # Block adaln_proj weights may remain on CPU/offload; we move temb to each weight device.
    transformer.time_proj = transformer.time_proj.float().to(compute_device)
    transformer.time_embedder = transformer.time_embedder.float().to(compute_device)
    transformer.norm_out = transformer.norm_out.float().to(compute_device)

    shift_msa = torch.empty((num_steps, num_blocks, adaln_rows, hidden_size), dtype=dtype, device="cpu")
    scale_msa = torch.empty_like(shift_msa)
    gate_msa = torch.empty_like(shift_msa)
    shift_mlp = torch.empty_like(shift_msa)
    scale_mlp = torch.empty_like(shift_msa)
    gate_mlp = torch.empty_like(shift_msa)
    final_shift = torch.empty((num_steps, fixed_temb_timesteps, hidden_size), dtype=dtype, device="cpu")
    final_scale = torch.empty_like(final_shift)

    was_training = transformer.training
    transformer.eval()
    with torch.no_grad():
        for step_idx, unique_t in enumerate(unique_timesteps):
            temb = transformer.time_embedder(transformer.time_proj(unique_t.to(compute_device)))
            temb = pad_minimax_h3_temb(temb, fixed_temb_timesteps)

            for block_idx, block in enumerate(transformer.transformer_blocks):
                adaln_proj = block.adaln_proj
                # Native MiniMaxH3AdaLayerNormModulation already applies SiLU + Linear + chunk.
                weight = adaln_proj.linear.weight if hasattr(adaln_proj, "linear") else adaln_proj.weight
                mods = adaln_proj(temb.to(device=weight.device, dtype=torch.float32))
                packed = [m.to(device="cpu", dtype=dtype).contiguous() for m in mods]
                if packed[0].shape != (adaln_rows, hidden_size):
                    raise RuntimeError(
                        f"Unexpected AdaLN shape at step={step_idx} block={block_idx}: "
                        f"{tuple(packed[0].shape)} vs {(adaln_rows, hidden_size)}."
                    )
                shift_msa[step_idx, block_idx].copy_(packed[0])
                scale_msa[step_idx, block_idx].copy_(packed[1])
                gate_msa[step_idx, block_idx].copy_(packed[2])
                shift_mlp[step_idx, block_idx].copy_(packed[3])
                scale_mlp[step_idx, block_idx].copy_(packed[4])
                gate_mlp[step_idx, block_idx].copy_(packed[5])

            # Final AdaLN-zero uses unique-timestep rows (not modality expansion).
            norm_out = transformer.norm_out
            emb = norm_out.linear(
                F.silu(temb).to(device=norm_out.linear.weight.device, dtype=norm_out.linear.weight.dtype)
            )
            shift, scale = emb.chunk(2, dim=1)
            final_shift[step_idx].copy_(shift.to(device="cpu", dtype=dtype).contiguous())
            final_scale[step_idx].copy_(scale.to(device="cpu", dtype=dtype).contiguous())
    if was_training:
        transformer.train()

    return {
        "num_inference_steps": int(num_inference_steps),
        "num_denoise_evals": int(num_steps),
        "fixed_temb_timesteps": int(fixed_temb_timesteps),
        "hidden_size": int(hidden_size),
        "time_embed_dim": int(time_embed_dim),
        "unique_timesteps": [t.detach().cpu().tolist() for t in unique_timesteps],
        "shift_msa": shift_msa,
        "scale_msa": scale_msa,
        "gate_msa": gate_msa,
        "shift_mlp": shift_mlp,
        "scale_mlp": scale_mlp,
        "gate_mlp": gate_mlp,
        "final_shift": final_shift,
        "final_scale": final_scale,
    }


def install_fixed_adaln_lut(
    transformer: nn.Module,
    *,
    num_inference_steps: int = MINIMAX_H3_DEFAULT_NUM_INFERENCE_STEPS,
    video_shift: float = MINIMAX_H3_DEFAULT_VIDEO_SHIFT,
    audio_shift: float = MINIMAX_H3_DEFAULT_AUDIO_SHIFT,
    keyframe_noise_aug: float = MINIMAX_H3_DEFAULT_KEYFRAME_NOISE_AUG,
    fixed_temb_timesteps: int = MINIMAX_H3_FIXED_TEMB_TIMESTEPS,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float16,
    lut: dict[str, object] | None = None,
) -> dict[str, object]:
    """Bake AdaLN / final-norm constants and switch the transformer to step_index ABI.

    Must be called on the native transformer **before** :func:`wrap_transformer`.
    """
    tables = lut or build_minimax_h3_adaln_lut(
        transformer,
        num_inference_steps=num_inference_steps,
        video_shift=video_shift,
        audio_shift=audio_shift,
        keyframe_noise_aug=keyframe_noise_aug,
        fixed_temb_timesteps=fixed_temb_timesteps,
        device=device,
        dtype=dtype,
    )
    target_device = torch.device(device) if device is not None else next(transformer.parameters()).device
    num_steps = int(tables["num_denoise_evals"])
    num_blocks = len(transformer.transformer_blocks)
    if num_blocks != int(tables["shift_msa"].shape[1]):
        raise ValueError(f"LUT block count {tables['shift_msa'].shape[1]} != transformer blocks {num_blocks}.")

    for block_idx, block in enumerate(transformer.transformer_blocks):
        for name, key in zip(
            _MINIMAX_FIXED_ADALN_NAMES,
            ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"),
            strict=True,
        ):
            tensor = tables[key][:, block_idx].to(device=target_device, dtype=dtype).contiguous()
            block.register_buffer(name, tensor, persistent=False)
        # Drop the live Linear so export/HMONNX cannot reintroduce temb→adaln_proj.
        if hasattr(block, "adaln_proj"):
            delattr(block, "adaln_proj")
        block._minimax_use_fixed_adaln = True

    transformer.register_buffer(
        "minimax_fixed_final_shift",
        tables["final_shift"].to(device=target_device, dtype=dtype).contiguous(),
        persistent=False,
    )
    transformer.register_buffer(
        "minimax_fixed_final_scale",
        tables["final_scale"].to(device=target_device, dtype=dtype).contiguous(),
        persistent=False,
    )
    transformer._minimax_use_fixed_adaln = True
    transformer._minimax_fixed_adaln_meta = {
        "num_inference_steps": int(tables["num_inference_steps"]),
        "num_denoise_evals": num_steps,
        "fixed_temb_timesteps": int(tables["fixed_temb_timesteps"]),
        "unique_timesteps": tables["unique_timesteps"],
    }
    return tables


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
    pack_gather_indices[video_indices.to(dtype=dtype)] = text_len + torch.arange(video_len, device=device, dtype=dtype)
    pack_gather_indices[audio_indices.to(dtype=dtype)] = (
        text_len + video_len + torch.arange(audio_len, device=device, dtype=dtype)
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
    # Make the offload state obvious in logs and reclaim any transient CUDA caches
    # left behind by AdaLN bake / FixedPoint calibration before the warm-up forward.
    sample = next(transformer.transformer_blocks[0].parameters())
    if sample.device.type != "cpu":
        raise RuntimeError(
            "prepare_transformer_block_cpu_offload failed: first transformer block is still on "
            f"{sample.device}, expected CPU."
        )
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


def _make_saturating_add() -> xhnn.Add:
    """Binary Add leaf with saturate=True (replaces post-op clip[-65504, 65504])."""
    op = xhnn.Add()
    op.saturate = True
    return op


def _make_saturating_mul() -> xhnn.Mul:
    """Binary Mul leaf with saturate=True (replaces post-op clip[-65504, 65504])."""
    op = xhnn.Mul()
    op.saturate = True
    return op


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
        # Keep AdaLNOut modulation path only as a fallback; export uses baked tables.
        self.norm_h = _as_xhnn_rmsnorm(self.norm)
        # xhquant/HMONNX: Gather(axis=0) replaces torch.index_select(0, ...).
        self.scale_gather = xhnn.Gather(axis=0)
        self.shift_gather = xhnn.Gather(axis=0)
        # Quant path requires tensor operands; Python float 1.0 breaks MetaInfoPro.
        self.register_buffer(
            "adaln_scale_one",
            torch.tensor(1.0, dtype=torch.float16),
            persistent=False,
        )
        self.scale_bias_add = _make_saturating_add()
        self.scale_mul = _make_saturating_mul()
        self.shift_add = _make_saturating_add()
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        # ``shift`` / ``scale`` are already selected by denoise ``step_index`` and have
        # shape ``(num_unique_timesteps, hidden)``. Row gather still uses timestep_indices.
        hidden_states = self.norm_h(hidden_states)
        scale = self.scale_bias_add(self.adaln_scale_one, self.scale_gather(scale, timestep_indices))
        shift = self.shift_gather(shift, timestep_indices)
        return self.shift_add(self.scale_mul(hidden_states, scale), shift)


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
        # 480p packs S≈15k → one fp16 [B,H,S,S] is ~25.86 GiB. Eager warm-up must
        # keep a single scores buffer (inplace scale / MaskedAdd / softmax). Dynamo
        # capture later runs on CPU and prefers out-of-place ops for FX friendliness.
        scores = torch.matmul(query.transpose(1, 2), key.transpose(1, 2).transpose(-1, -2))
        compiling = torch.compiler.is_compiling()
        if compiling:
            scores = scores * self.scale
        else:
            scores.mul_(self.scale)
            del query, key
        if attention_mask is not None:
            # Export graph uses MaskedAdd (double-add + saturate), same as text/vision wrappers.
            # Host mask stays additive key-padding bias: valid=0, pad=finfo(dtype).min.
            scores = self.masked_add(scores, attention_mask)
        if compiling:
            probs = torch.softmax(scores, dim=-1)
            hidden_states = torch.matmul(probs, value.transpose(1, 2))
        else:
            torch.softmax(scores, dim=-1, out=scores)
            hidden_states = torch.matmul(scores, value.transpose(1, 2))
            del scores, value
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
        # Explicit saturating Add leaves replace previous clip[-65504, 65504].
        self.attn_residual_add = _make_saturating_add()
        self.ff_residual_add = _make_saturating_add()
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.attn(self.norm1_h(hidden_states), attention_mask=attention_mask)
        hidden_states = self.attn_residual_add(hidden_states, attention_output)
        feed_forward_output = self.ff(self.norm2_h(hidden_states))
        return self.ff_residual_add(hidden_states, feed_forward_output)


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
        required = _MINIMAX_FIXED_ADALN_NAMES
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                f"Call install_fixed_adaln_lut() before wrap_transformer(); missing AdaLN LUT buffers: {missing}."
            )
        self.norm1_h = _as_xhnn_rmsnorm(self.norm1)
        self.norm2_h = _as_xhnn_rmsnorm(self.norm2)
        # xhquant/HMONNX: Gather(axis=0) replaces torch.index_select / data-dependent
        # Python indexing so Dynamo can specialize scalar step_index LUT lookups.
        self.step_gather = xhnn.Gather(axis=0)
        self.adaln_gather = xhnn.Gather(axis=0)
        # Quant path requires tensor operands; Python float 1.0 breaks MetaInfoPro.
        self.register_buffer(
            "adaln_scale_one",
            torch.tensor(1.0, dtype=torch.float16),
            persistent=False,
        )
        # Explicit saturating Mul/Add leaves replace previous AdaLN clip[-65504, 65504].
        self.msa_scale_bias_add = _make_saturating_add()
        self.msa_scale_mul = _make_saturating_mul()
        self.msa_shift_add = _make_saturating_add()
        self.mlp_scale_bias_add = _make_saturating_add()
        self.mlp_scale_mul = _make_saturating_mul()
        self.mlp_shift_add = _make_saturating_add()
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
        step_index: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Baked AdaLN tables: [num_steps, capacity*MODALITY, H], indexed by denoise step.
        # Use xhnn.Gather instead of table[step_index] so Dynamo/export can specialize.
        shift_msa = self.step_gather(self.minimax_fixed_shift_msa, step_index)
        scale_msa = self.step_gather(self.minimax_fixed_scale_msa, step_index)
        gate_msa = self.step_gather(self.minimax_fixed_gate_msa, step_index)
        shift_mlp = self.step_gather(self.minimax_fixed_shift_mlp, step_index)
        scale_mlp = self.step_gather(self.minimax_fixed_scale_mlp, step_index)
        gate_mlp = self.step_gather(self.minimax_fixed_gate_mlp, step_index)
        residual = hidden_states
        norm_hidden_states = self.norm1_h(hidden_states)
        msa_scale = self.msa_scale_bias_add(
            self.adaln_scale_one,
            self.adaln_gather(scale_msa, adaln_indices),
        )
        norm_hidden_states = self.msa_scale_mul(norm_hidden_states, msa_scale)
        norm_hidden_states = self.msa_shift_add(
            norm_hidden_states,
            self.adaln_gather(shift_msa, adaln_indices),
        )
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
        mlp_scale = self.mlp_scale_bias_add(
            self.adaln_scale_one,
            self.adaln_gather(scale_mlp, adaln_indices),
        )
        norm_hidden_states = self.mlp_scale_mul(norm_hidden_states, mlp_scale)
        norm_hidden_states = self.mlp_shift_add(
            norm_hidden_states,
            self.adaln_gather(shift_mlp, adaln_indices),
        )
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
        required = (
            "minimax_fixed_final_shift",
            "minimax_fixed_final_scale",
        )
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                f"Call install_fixed_adaln_lut() before wrap_transformer(); missing final-norm LUT buffers: {missing}."
            )
        self.enable_block_cpu_offload = False
        # xhquant/HMONNX: Gather(axis=0) for denoise step LUT; Gather(axis=1) for token indices.
        self.step_gather = xhnn.Gather(axis=0)
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
        step_index: torch.Tensor,
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
        """Export/HMONNX ABI: host provides step_index/adaln/rotary + pack gather + masks.

        ``text_indices`` stay host-only (used to build ``pack_gather_indices`` /
        ``packed_attention_mask``). Keeping them as unused forward args makes Dynamo
        drop the placeholder and break ``input_names`` vs ``example_inputs`` arity.
        AdaLN / final-norm tables are baked by :func:`install_fixed_adaln_lut` and
        selected with scalar ``step_index``.
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
                        step_index,
                        adaln_indices,
                        rotary_emb,
                        packed_attention_mask,
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        step_index,
                        adaln_indices,
                        rotary_emb,
                        packed_attention_mask,
                    )
            finally:
                if block_offload:
                    _move_module(block, "cpu")
                    _empty_cache(compute_device)

        final_shift = self.step_gather(self.minimax_fixed_final_shift, step_index)
        final_scale = self.step_gather(self.minimax_fixed_final_scale, step_index)
        hidden_states = self.norm_out(hidden_states, final_shift, final_scale, timestep_indices)
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
        if not bool(getattr(transformer, "_minimax_use_fixed_adaln", False)):
            install_fixed_adaln_lut(transformer)
        self.transformer = wrap_transformer(transformer)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        step_index: torch.Tensor,
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
        legacy_step_index = step_index
        if isinstance(step_index, torch.Tensor) and step_index.ndim == 2:
            # Legacy host-side ABI: some callers still feed a fixed-capacity temb table.
            # The wrapped export model only needs a scalar denoise step index.
            legacy_step_index = torch.tensor(
                int(timestep_indices.min().item()) if timestep_indices.numel() else 0,
                dtype=torch.int32,
                device=hidden_states.device,
            )
        return self.transformer(
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            legacy_step_index,
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
    "MINIMAX_H3_DEFAULT_AUDIO_SHIFT",
    "MINIMAX_H3_DEFAULT_KEYFRAME_NOISE_AUG",
    "MINIMAX_H3_DEFAULT_NUM_INFERENCE_STEPS",
    "MINIMAX_H3_DEFAULT_VIDEO_SHIFT",
    "MINIMAX_H3_FIXED_TEMB_TIMESTEPS",
    "MiniMaxH3TransformerWrapper",
    "build_minimax_h3_adaln_lut",
    "build_minimax_h3_denoise_unique_timesteps",
    "build_minimax_h3_transformer_conditioning",
    "build_pack_gather_indices",
    "build_packed_attention_mask",
    "disable_transformer_block_cpu_offload",
    "install_fixed_adaln_lut",
    "pad_minimax_h3_temb",
    "prepare_transformer_block_cpu_offload",
    "wrap_transformer",
]
