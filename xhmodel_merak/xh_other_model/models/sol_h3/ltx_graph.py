# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Tensor-only LTX-2.5 joint AV graphs, assembled from external native weights.

The source builder is used only during export. No LTX class implementation is
vendored, and deployed graphs do not import the external source package.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .conditioning import validate_fp16_constants
from .precision import ATTENTION_VALUE_DIVISOR, StableRMSNorm, scaled_linear


class StepTable(nn.Module):
    def __init__(self, value, *, sdk_ops=False):
        super().__init__()
        validate_fp16_constants(value, name="LTX folded constants")
        self.register_buffer("value", value.detach().float().contiguous())
        self.sdk_ops = sdk_ops
        if sdk_ops:
            from xhquant.nn import Gather

            self.gather = Gather(axis=0)

    def forward(self, step):
        result = self.gather(self.value, step) if self.sdk_ops else self.value.index_select(0, step.long())
        return result.squeeze(0)


class AudioModeStepTable(nn.Module):
    """Select a step and then the joint/frozen audio conditioning for it."""

    def __init__(self, value, *, sdk_ops=False):
        super().__init__()
        if value.shape[1] != 2:
            raise ValueError("LTX audio policy tables require joint and frozen entries")
        validate_fp16_constants(value, name="LTX folded audio policy constants")
        self.register_buffer("value", value.detach().float().contiguous())
        self.sdk_ops = sdk_ops
        if sdk_ops:
            from xhquant.nn import Gather

            self.step_gather, self.mode_gather = Gather(axis=0), Gather(axis=1)

    def forward(self, step, audio_mode):
        if self.sdk_ops:
            value = self.mode_gather(self.step_gather(self.value, step), audio_mode)
        else:
            value = self.value.index_select(0, step.long()).index_select(1, audio_mode.long())
        return value.squeeze(0).squeeze(0)


def unit_rms(dim, eps, *, sdk_ops=False):
    return StableRMSNorm(nn.RMSNorm(dim, eps=eps), sdk_ops=sdk_ops)


class LTXAttention(nn.Module):
    def __init__(
        self,
        source,
        *,
        sdk_ops=False,
        cached_kv=None,
        rotary=False,
        query_tokens=None,
        key_tokens=None,
        audio_mode_tables=False,
    ):
        super().__init__()
        self.heads, self.head_dim = source.heads, source.dim_head
        self.query = source.to_q
        self.output = scaled_linear(source.to_out[0], input_multiplier=ATTENTION_VALUE_DIVISOR)
        self.q_norm = StableRMSNorm(source.q_norm, sdk_ops=sdk_ops)
        self.gate = source.to_gate_logits
        self.sdk_ops, self.rotary_enabled = sdk_ops, rotary
        self.cached = cached_kv is not None
        self.audio_mode_tables = audio_mode_tables
        if query_tokens is not None:
            self.register_buffer("query_positions", torch.arange(query_tokens, dtype=torch.int32).reshape(1, -1, 1))
            self.register_buffer("query_zero", torch.tensor(0.0))
            if sdk_ops:
                from xhquant.nn import Less, Where

                self.query_less, self.query_where = Less(), Where()
        if key_tokens is not None:
            self.register_buffer("key_positions", torch.arange(key_tokens, dtype=torch.int32).reshape(1, 1, 1, -1))
            self.register_buffer("full_key_length", torch.tensor([key_tokens], dtype=torch.int32))
            self.register_buffer("key_zero", torch.tensor(0.0, dtype=torch.float16))
            self.register_buffer("key_negative", torch.tensor(-65504.0, dtype=torch.float16))
            if sdk_ops:
                from xhquant.nn import Less, Where

                self.key_less, self.key_where = Less(), Where()
        if self.cached:
            table_type = AudioModeStepTable if audio_mode_tables else StepTable
            self.keys = table_type(cached_kv[0], sdk_ops=sdk_ops)
            self.values = table_type(cached_kv[1] / ATTENTION_VALUE_DIVISOR, sdk_ops=sdk_ops)
        else:
            self.key = source.to_k
            self.value = scaled_linear(source.to_v, output_divisor=ATTENTION_VALUE_DIVISOR)
            self.k_norm = StableRMSNorm(source.k_norm, sdk_ops=sdk_ops)
        if sdk_ops:
            from xhquant.nn import Rope

            from ...common.attention import CompilerAttention as FlashAttention

            self.attention = FlashAttention(num_heads=self.heads, scale=self.head_dim**-0.5, is_causal=False)
            self.rotary = Rope()

    def heads_view(self, value):
        return value.reshape(1, -1, self.heads, self.head_dim).transpose(1, 2)

    def rotate(self, x, cos, sin):
        if self.sdk_ops:
            return self.rotary(x, cos, sin)
        first, second = x.chunk(2, -1)
        return x * cos + torch.cat((-second, first), -1) * sin

    def forward(
        self,
        x,
        context,
        step,
        qcos,
        qsin,
        kcos,
        ksin,
        query_valid_length=None,
        key_valid_length=None,
        audio_mode=None,
    ):
        q = self.heads_view(self.q_norm(self.query(x)))
        if self.cached:
            if self.audio_mode_tables:
                k, v = self.keys(step, audio_mode), self.values(step, audio_mode)
            else:
                k, v = self.keys(step), self.values(step)
        else:
            k = self.heads_view(self.k_norm(self.key(context)))
            v = self.heads_view(self.value(context))
        if self.rotary_enabled:
            q, k = self.rotate(q, qcos, qsin), self.rotate(k, kcos, ksin)
        if self.sdk_ops:
            # Share the compiler's broadcast-key-mask entry with H3. The
            # static device entry does not accept simulator-only prefix lens.
            key_mask = None
            if key_valid_length is not None:
                key_mask = self.key_where(
                    self.key_less(self.key_positions, key_valid_length.reshape(1, 1, 1, 1)),
                    self.key_zero,
                    self.key_negative,
                )
            attended = self.attention(q, k, v, key_mask=key_mask)
        else:
            mask = None if key_valid_length is None else self.key_positions < key_valid_length.reshape(1, 1, 1, 1)
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        if self.gate is not None:
            attended = attended * (2 * torch.sigmoid(self.gate(x))).transpose(1, 2).unsqueeze(-1)
        result = self.output(attended.transpose(1, 2).flatten(2))
        if query_valid_length is not None:
            # Mask after the biased output projection as well.
            if self.sdk_ops:
                result = self.query_where(
                    self.query_less(self.query_positions, query_valid_length.reshape(1, 1, 1)), result, self.query_zero
                )
            else:
                result = torch.where(self.query_positions < query_valid_length.reshape(1, 1, 1), result, 0.0)
        return result


class FirstFrameTimeTable(nn.Module):
    """Select one small time vector for the target's first frame, then broadcast."""

    def __init__(self, *, sdk_ops=False):
        super().__init__()
        self.sdk_ops = sdk_ops
        if sdk_ops:
            from xhquant.nn import Gather

            self.gather = Gather(axis=1)

    def forward(self, params, first_role):
        first = self.gather(params, first_role) if self.sdk_ops else params.index_select(1, first_role.long())
        return torch.cat((first, params), dim=1)


class LTXFFN(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.input, self.output = source.net[0].proj, source.net[2]

    def forward(self, x):
        return self.output(F.gelu(self.input(x), approximate="tanh"))


@torch.no_grad()
def fold_ltx_block(source, video_args, audio_args, *, joint_video_args=None, joint_audio_args=None):
    """Fold step tables, optionally retaining joint and frozen audio policies.

    ``video_args`` and ``audio_args`` describe the frozen policy when the joint
    arguments are supplied. Video's own tables are shared; its cross gate is
    conditioned on the other modality and must retain both policy entries.
    """

    if (joint_video_args is None) != (joint_audio_args is None):
        raise ValueError("joint video and audio conditioning must be provided together")
    dual_audio_modes = joint_audio_args is not None

    def block_values(table, args):
        return torch.stack(source.get_ada_values(table, 1, args.timesteps, slice(None)), 0).squeeze(1)

    def av_values(table, args):
        affine = source.get_ada_values(table[:4], 1, args.cross_scale_shift_timestep, slice(None))
        gate = source.get_ada_values(table[4:], 1, args.cross_gate_timestep, slice(None))
        # Per-token video time has separate noisy/clean roles, whereas the
        # cross-modality gate still uses the other modality's global sigma.
        return torch.stack((*affine, *(value.expand_as(affine[0]) for value in gate)), 0).squeeze(1)

    def prompt_kv(attn, prompt_table, args):
        modulation = prompt_table[None, None].to(args.context.dtype)
        if args.prompt_timestep is not None:
            modulation = modulation + args.prompt_timestep.reshape(1, 1, 2, -1)
        shift, scale = modulation[:, :, 0], modulation[:, :, 1]
        context = args.context * (1 + scale) + shift

        def heads(x):
            return x.reshape(1, -1, attn.heads, attn.dim_head).transpose(1, 2)

        return heads(attn.k_norm(attn.to_k(context))), heads(attn.to_v(context))

    output = {
        "video": [],
        "audio": [],
        "av_video": [],
        "av_audio": [],
        "video_k": [],
        "video_v": [],
        "audio_k": [],
        "audio_v": [],
    }
    if dual_audio_modes:
        steps = zip(video_args, audio_args, joint_video_args, joint_audio_args, strict=True)
    else:
        steps = ((video, audio, None, None) for video, audio in zip(video_args, audio_args, strict=True))
    for video, audio, joint_video, joint_audio in steps:
        output["video"].append(block_values(source.scale_shift_table, video))
        key, value = prompt_kv(source.attn2, source.prompt_scale_shift_table, video)
        output["video_k"].append(key)
        output["video_v"].append(value)

        def audio_policy_values(policy_video, policy_audio):
            key, value = prompt_kv(source.audio_attn2, source.audio_prompt_scale_shift_table, policy_audio)
            return dict(
                audio=block_values(source.audio_scale_shift_table, policy_audio),
                av_video=av_values(source.scale_shift_table_a2v_ca_video, policy_video),
                av_audio=av_values(source.scale_shift_table_a2v_ca_audio, policy_audio),
                audio_k=key,
                audio_v=value,
            )

        frozen = audio_policy_values(video, audio)
        if dual_audio_modes:
            joint = audio_policy_values(joint_video, joint_audio)
            for name, value in frozen.items():
                output[name].append(torch.stack((joint[name], value)))
        else:
            for name, value in frozen.items():
                output[name].append(value)
    return {name: torch.stack(values).detach() for name, values in output.items()}


def video_modulation(x, params, ranges, *, shift=None, scale=None, gate=None):
    """Apply compact time-role constants without expanding every AdaLN table.

    ``ranges`` partition the video sequence into contiguous ``(start, end,
    role)`` spans. The same clean role covers both the frozen first frame and
    appended keyframe guides. Audio and prompt K/V use their own modality's
    global sigma, selected for joint or clean frozen audio.
    """
    if ranges is None:
        if gate is not None:
            return x * params[gate]
        return x * (1 + params[scale]) + params[shift]
    values = []
    for start, end, role in ranges:
        value = x[:, start:end]
        if gate is not None:
            value = value * params[gate, role : role + 1]
        else:
            value = value * (1 + params[scale, role : role + 1]) + params[shift, role : role + 1]
        values.append(value)
    return torch.cat(values, dim=1)


class LTXBlock(nn.Module):
    """Six attentions with simultaneous A2V/V2A snapshots and two GELU FFNs."""

    def __init__(
        self,
        source,
        folded,
        *,
        sdk_ops=False,
        video_timestep_ranges=None,
        video_tokens=None,
        audio_tokens=None,
        dual_audio_modes=False,
    ):
        super().__init__()
        self.video_timestep_ranges = video_timestep_ranges
        self.first_time = FirstFrameTimeTable(sdk_ops=sdk_ops) if video_tokens is not None else None
        self.dual_audio_modes = dual_audio_modes
        self.video_params = StepTable(folded["video"], sdk_ops=sdk_ops)
        audio_table_type = AudioModeStepTable if dual_audio_modes else StepTable
        self.audio_params = audio_table_type(folded["audio"], sdk_ops=sdk_ops)
        self.av_video, self.av_audio = (
            audio_table_type(folded[name], sdk_ops=sdk_ops) for name in ("av_video", "av_audio")
        )
        self.video_norm = unit_rms(source.attn1.to_q.in_features, source.norm_eps, sdk_ops=sdk_ops)
        self.audio_norm = unit_rms(source.audio_attn1.to_q.in_features, source.norm_eps, sdk_ops=sdk_ops)
        self.video_self = LTXAttention(
            source.attn1, sdk_ops=sdk_ops, rotary=True, query_tokens=video_tokens, key_tokens=video_tokens
        )
        self.audio_self = LTXAttention(source.audio_attn1, sdk_ops=sdk_ops, rotary=True)
        self.video_text = LTXAttention(
            source.attn2,
            sdk_ops=sdk_ops,
            cached_kv=(folded["video_k"], folded["video_v"]),
            query_tokens=video_tokens,
            key_tokens=folded["video_k"].shape[-2] if video_tokens is not None else None,
        )
        self.audio_text = LTXAttention(
            source.audio_attn2,
            sdk_ops=sdk_ops,
            cached_kv=(folded["audio_k"], folded["audio_v"]),
            audio_mode_tables=dual_audio_modes,
        )
        self.audio_to_video = LTXAttention(
            source.audio_to_video_attn, sdk_ops=sdk_ops, rotary=True, query_tokens=video_tokens, key_tokens=audio_tokens
        )
        self.video_to_audio = LTXAttention(
            source.video_to_audio_attn, sdk_ops=sdk_ops, rotary=True, key_tokens=video_tokens
        )
        self.video_ffn, self.audio_ffn = LTXFFN(source.ff), LTXFFN(source.audio_ff)

    @staticmethod
    def affine(x, params, shift, scale):
        return x * (1 + params[scale]) + params[shift]

    def forward(
        self,
        video,
        audio,
        step,
        vcos,
        vsin,
        acos,
        asin,
        vccos,
        vcsin,
        accos,
        acsin,
        first_role=None,
        video_valid_length=None,
        audio_mode=None,
    ):
        vp = self.video_params(step)
        if self.dual_audio_modes:
            ap = self.audio_params(step, audio_mode)
        else:
            ap = self.audio_params(step)
        if self.first_time is not None:
            vp = self.first_time(vp, first_role)
        ranges = self.video_timestep_ranges
        v = video_modulation(self.video_norm(video), vp, ranges, shift=0, scale=1)
        video = video + video_modulation(
            self.video_self(v, v, step, vcos, vsin, vcos, vsin, video_valid_length, video_valid_length),
            vp,
            ranges,
            gate=2,
        )
        v = video_modulation(self.video_norm(video), vp, ranges, shift=6, scale=7)
        video = video + video_modulation(
            self.video_text(v, v, step, vcos, vsin, vcos, vsin, video_valid_length), vp, ranges, gate=8
        )
        a = self.affine(self.audio_norm(audio), ap, 0, 1)
        audio = audio + self.audio_self(a, a, step, acos, asin, acos, asin) * ap[2]
        a = self.affine(self.audio_norm(audio), ap, 6, 7)
        audio = audio + self.audio_text(a, a, step, acos, asin, acos, asin, audio_mode=audio_mode) * ap[8]
        # Both directions use the state before either direction's residual update.
        v, a = self.video_norm(video), self.audio_norm(audio)
        if self.dual_audio_modes:
            cv, ca = self.av_video(step, audio_mode), self.av_audio(step, audio_mode)
        else:
            cv, ca = self.av_video(step), self.av_audio(step)
        if self.first_time is not None:
            cv = self.first_time(cv, first_role)
        va, aa = video_modulation(v, cv, ranges, shift=1, scale=0), self.affine(a, ca, 1, 0)
        av, vv = self.affine(a, ca, 3, 2), video_modulation(v, cv, ranges, shift=3, scale=2)
        video = video + video_modulation(
            self.audio_to_video(va, aa, step, vccos, vcsin, accos, acsin, video_valid_length), cv, ranges, gate=4
        )
        audio = audio + self.video_to_audio(av, vv, step, accos, acsin, vccos, vcsin, None, video_valid_length) * ca[4]
        video = video + video_modulation(
            self.video_ffn(video_modulation(self.video_norm(video), vp, ranges, shift=3, scale=4)),
            vp,
            ranges,
            gate=5,
        )
        audio = audio + self.audio_ffn(self.affine(self.audio_norm(audio), ap, 3, 4)) * ap[5]
        return video, audio


class LTXOutput(nn.Module):
    def __init__(
        self,
        source,
        video_args,
        audio_args,
        *,
        sdk_ops=False,
        video_timestep_ranges=None,
        dynamic_first_role=False,
        joint_audio_args=None,
    ):
        super().__init__()
        self.video_timestep_ranges = video_timestep_ranges
        self.first_time = FirstFrameTimeTable(sdk_ops=sdk_ops) if dynamic_first_role else None
        self.video_norm, self.audio_norm = source.norm_out, source.audio_norm_out
        self.video_head, self.audio_head = source.proj_out, source.audio_proj_out
        self.dual_audio_modes = joint_audio_args is not None

        def values(table, args):
            return torch.stack([table.to(x.x.dtype)[None, None] + x.embedded_timestep[:, :, None] for x in args])

        self.video_params = StepTable(values(source.scale_shift_table, video_args), sdk_ops=sdk_ops)
        audio_values = values(source.audio_scale_shift_table, audio_args)
        if self.dual_audio_modes:
            audio_values = torch.stack((values(source.audio_scale_shift_table, joint_audio_args), audio_values), dim=1)
        audio_table_type = AudioModeStepTable if self.dual_audio_modes else StepTable
        self.audio_params = audio_table_type(audio_values, sdk_ops=sdk_ops)

    def forward(self, video, audio, step, first_role=None, audio_mode=None):
        vp = self.video_params(step)
        ap = self.audio_params(step, audio_mode) if self.dual_audio_modes else self.audio_params(step)
        if self.video_timestep_ranges is None:
            video = self.video_head(self.video_norm(video) * (1 + vp[:, :, 1]) + vp[:, :, 0])
        else:
            params = vp.squeeze(0).transpose(0, 1)
            if self.first_time is not None:
                params = self.first_time(params, first_role)
            video = self.video_head(
                video_modulation(self.video_norm(video), params, self.video_timestep_ranges, shift=0, scale=1)
            )
        audio = self.audio_head(self.audio_norm(audio) * (1 + ap[:, :, 1]) + ap[:, :, 0])
        return video, audio


class LTXDenoiserGraph(nn.Module):
    def __init__(
        self,
        source,
        folded_blocks,
        video_args,
        audio_args,
        *,
        sdk_ops=False,
        video_timestep_ranges=None,
        keyframes_mask=None,
        first_frame_tokens=None,
        joint_audio_args=None,
    ):
        super().__init__()
        video_tokens = None
        if first_frame_tokens is not None:
            if video_timestep_ranges is not None or type(first_frame_tokens) is not int or first_frame_tokens <= 0:
                raise ValueError("static LTX guides require a positive first-frame count and implicit time ranges")
            video_tokens = video_args[0].x.shape[1]
            if video_tokens % first_frame_tokens or video_tokens < 3 * first_frame_tokens:
                raise ValueError("static LTX guides require target frames and one complete guide frame")
            target_tokens = video_tokens - first_frame_tokens
            video_timestep_ranges = (
                (0, first_frame_tokens, 0),
                (first_frame_tokens, target_tokens, 1),
                (target_tokens, video_tokens, 2),
            )
            self.register_buffer("video_positions", torch.arange(video_tokens, dtype=torch.int32).reshape(1, -1, 1))
            self.register_buffer("video_zero", torch.tensor(0.0))
            if sdk_ops:
                from xhquant.nn import Less, Where

                self.video_less, self.video_where = Less(), Where()
        if video_timestep_ranges is not None:
            ranges = tuple(tuple(span) for span in video_timestep_ranges)
            count = video_args[0].x.shape[1]
            end = 0
            for start, stop, role in ranges:
                if start != end or stop <= start or role not in ((0, 1, 2) if video_tokens is not None else (0, 1)):
                    raise ValueError("video timestep ranges must partition the sequence into noisy/clean roles")
                end = stop
            if end != count:
                raise ValueError("video timestep ranges must cover every target and guide token")
            video_timestep_ranges = ranges
        self.video_projection, self.audio_projection = source.patchify_proj, source.audio_patchify_proj
        embedding = source._keyframes_embedding() if keyframes_mask is not None else None
        if embedding is not None:
            self.register_buffer("keyframe_embedding", embedding.detach().clone())
            self.register_buffer("keyframe_mask", (keyframes_mask > 0).detach().clone())
        else:
            self.keyframe_embedding = None
        self.blocks = nn.ModuleList(
            [
                LTXBlock(
                    block,
                    folded,
                    sdk_ops=sdk_ops,
                    video_timestep_ranges=video_timestep_ranges,
                    video_tokens=video_tokens,
                    audio_tokens=audio_args[0].x.shape[1] if video_tokens is not None else None,
                    dual_audio_modes=joint_audio_args is not None,
                )
                for block, folded in zip(source.transformer_blocks, folded_blocks, strict=True)
            ]
        )
        self.output = LTXOutput(
            source,
            video_args,
            audio_args,
            sdk_ops=sdk_ops,
            video_timestep_ranges=video_timestep_ranges,
            dynamic_first_role=video_tokens is not None,
            joint_audio_args=joint_audio_args,
        )
        self.static_guide = video_tokens is not None
        self.sdk_ops = sdk_ops
        self.register_buffer("default_audio_mode", torch.tensor([1], dtype=torch.int32))

    def forward(
        self,
        video,
        audio,
        step,
        vcos,
        vsin,
        acos,
        asin,
        vccos,
        vcsin,
        accos,
        acsin,
        first_role=None,
        video_valid_length=None,
        audio_mode=None,
    ):
        if audio_mode is None:
            audio_mode = self.default_audio_mode
        if self.static_guide:
            if self.sdk_ops:
                video = self.video_where(
                    self.video_less(self.video_positions, video_valid_length.reshape(1, 1, 1)), video, self.video_zero
                )
            else:
                video = torch.where(self.video_positions < video_valid_length.reshape(1, 1, 1), video, 0.0)
        video, audio = self.video_projection(video), self.audio_projection(audio)
        if self.keyframe_embedding is not None:
            video = video + self.keyframe_mask.to(video.dtype) * self.keyframe_embedding.to(video.dtype)
        for block in self.blocks:
            video, audio = block(
                video,
                audio,
                step,
                vcos,
                vsin,
                acos,
                asin,
                vccos,
                vcsin,
                accos,
                acsin,
                first_role,
                video_valid_length,
                audio_mode,
            )
        video, audio = self.output(video, audio, step, first_role, audio_mode)
        if self.static_guide:
            if self.sdk_ops:
                video = self.video_where(
                    self.video_less(self.video_positions, video_valid_length.reshape(1, 1, 1)), video, self.video_zero
                )
            else:
                video = torch.where(self.video_positions < video_valid_length.reshape(1, 1, 1), video, 0.0)
        return video, audio


class LTXInputRoPEGraph(nn.Module):
    """Expose RoPE and audio policy without static guide controls."""

    def __init__(self, denoiser):
        super().__init__()
        if denoiser.static_guide:
            raise ValueError("target-only LTX graphs cannot accept static guide controls")
        self.denoiser = denoiser

    def forward(self, video, audio, step, vcos, vsin, acos, asin, vccos, vcsin, accos, acsin, audio_mode=None):
        return self.denoiser(
            video, audio, step, vcos, vsin, acos, asin, vccos, vcsin, accos, acsin, audio_mode=audio_mode
        )


class LTXConstantRoPEGraph(nn.Module):
    """Bind one deployment grid, shared by every layer and denoising step."""

    def __init__(self, denoiser, rope):
        super().__init__()
        if len(rope) != 8:
            raise ValueError("LTX requires four cosine/sine pairs")
        self.denoiser = denoiser
        for name, value in zip(("vcos", "vsin", "acos", "asin", "vccos", "vcsin", "accos", "acsin"), rope, strict=True):
            if not torch.isfinite(value).all():
                raise ValueError("LTX RoPE constants must be finite")
            self.register_buffer(name, value.detach().clone())

    def forward(self, video, audio, step, audio_mode=None):
        return self.denoiser(
            video,
            audio,
            step,
            self.vcos,
            self.vsin,
            self.acos,
            self.asin,
            self.vccos,
            self.vcsin,
            self.accos,
            self.acsin,
            audio_mode=audio_mode,
        )
