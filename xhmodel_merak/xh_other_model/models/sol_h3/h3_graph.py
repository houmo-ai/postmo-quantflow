# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Static T2VA graphs built from native Diffusers weights, without time MLPs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .conditioning import IndexedModulation, validate_fp16_constants
from .precision import ATTENTION_VALUE_DIVISOR, H3PrecisionPolicy, StableRMSNorm, scaled_linear, split_swiglu_projection
from .role_layout import role_affine, role_spans

DEFAULT_H3_POLICY = H3PrecisionPolicy()


class H3Attention(nn.Module):
    def __init__(self, source, *, output_divisor, sdk_ops=False, use_rotary=True, attention_block_size=None):
        super().__init__()
        self.heads, self.head_dim = source.heads, source.head_dim
        self.q, self.k = source.to_q, source.to_k
        self.v = scaled_linear(source.to_v, output_divisor=ATTENTION_VALUE_DIVISOR)
        self.out = scaled_linear(
            source.to_out[0], input_multiplier=ATTENTION_VALUE_DIVISOR, output_divisor=output_divisor
        )
        self.q_norm = StableRMSNorm(source.norm_q, sdk_ops=sdk_ops)
        self.k_norm = StableRMSNorm(source.norm_k, sdk_ops=sdk_ops)
        self.sdk_ops = sdk_ops
        self.use_rotary = use_rotary
        self.attention_block_size = attention_block_size
        if sdk_ops:
            from xhquant.nn import FlashAttention, Rope

            self.rotary = Rope()
            self.attention = FlashAttention(num_heads=self.heads, is_causal=False, scale=self.head_dim**-0.5)

    def rotate(self, x, cos, sin):
        if not self.use_rotary:
            return x
        width = cos.shape[-1]
        rotated, tail = x[..., :width], x[..., width:]
        if self.sdk_ops:
            rotated = self.rotary(rotated, cos, sin)
        else:
            a, b = rotated.chunk(2, -1)
            rotated = rotated * cos + torch.cat((-b, a), -1) * sin
        return torch.cat((rotated, tail), -1)

    def forward(self, x, cos, sin, valid_length, attention_metadata=None):
        def heads(v):
            return v.reshape(1, -1, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = self.q_norm(heads(self.q(x))), self.k_norm(heads(self.k(x))), heads(self.v(x))
        q, k = self.rotate(q, cos, sin), self.rotate(k, cos, sin)
        if attention_metadata is not None:
            if self.attention_block_size is None:
                raise ValueError("fixed-role H3 attention requires an explicit block size")
            if self.sdk_ops:
                attended = self.attention(q, k, v, key_mask=attention_metadata)
            else:
                # Only a [B,1,1,N] mask is supplied. Query padding is discarded
                # by the final origin map; it stays masked as KV in each block.
                attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_metadata.to(q.dtype))
        elif self.sdk_ops:
            attended = self.attention(q, k, v, current_input_length=valid_length)
        else:
            # The padded text rows are at the tail of every static sequence.
            mask = torch.arange(k.shape[2], device=x.device)[None, :] < valid_length.reshape(-1, 1)
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        return self.out(attended.transpose(1, 2).flatten(2))


class H3SwiGLU(nn.Module):
    def __init__(self, source, *, product_divisor, output_divisor):
        super().__init__()
        self.projection = split_swiglu_projection(source.net[0].proj, product_divisor)
        self.output = scaled_linear(source.net[2], input_multiplier=product_divisor, output_divisor=output_divisor)

    def forward(self, x):
        linear, gate = self.projection(x).chunk(2, -1)
        return self.output(linear * F.silu(gate))


class H3TextGraph(nn.Module):
    """Run the prompt-only token refiner once, with a compact valid text prefix."""

    def __init__(self, source, policy=DEFAULT_H3_POLICY, *, sdk_ops=False, input_divisor=1.0):
        super().__init__()
        # At /4, epsilon=1e-5 becomes 6.25e-7: reciprocal RMS <=1265,
        # safely within FP16. Keep bounded normalization for other policies.
        self.residual_scale = policy.text_residual_scale
        bounded = self.residual_scale != 4.0
        self.projection = scaled_linear(
            source.context_embedder, output_divisor=self.residual_scale, input_multiplier=input_divisor
        )
        self.blocks = nn.ModuleList()
        for block in source.token_refiner.refiner_blocks:
            item = nn.ModuleDict(
                {
                    "norm1": StableRMSNorm(
                        block.norm1, input_divisor=self.residual_scale, sdk_ops=sdk_ops, bounded_reciprocal=bounded
                    ),
                    "norm2": StableRMSNorm(
                        block.norm2, input_divisor=self.residual_scale, sdk_ops=sdk_ops, bounded_reciprocal=bounded
                    ),
                    "attention": H3Attention(
                        block.attn, output_divisor=self.residual_scale, sdk_ops=sdk_ops, use_rotary=False
                    ),
                    "ffn": H3SwiGLU(
                        block.ff, product_divisor=policy.ffn_product_scale, output_divisor=self.residual_scale
                    ),
                }
            )
            self.blocks.append(item)
        self.norm = StableRMSNorm(
            source.token_refiner.final_norm,
            input_divisor=self.residual_scale,
            sdk_ops=sdk_ops,
            bounded_reciprocal=bounded,
        )

    def forward(self, text_features, valid_text_length):
        x = self.projection(text_features)
        # Identity rotary factors retain the same static attention interface.
        cos, sin = x.new_ones((1, 1, x.shape[1], 0)), x.new_zeros((1, 1, x.shape[1], 0))
        for block in self.blocks:
            x = x + block["attention"](block["norm1"](x), cos, sin, valid_text_length)
            x = x + block["ffn"](block["norm2"](x))
        return self.norm(x)


class H3Block(nn.Module):
    """Three contiguous modal groups; six broadcast vectors, shared joint attention."""

    def __init__(
        self,
        source,
        table,
        video_tokens,
        audio_tokens,
        policy=DEFAULT_H3_POLICY,
        *,
        sdk_ops=False,
        role_capacities=None,
        attention_block_size=None,
    ):
        super().__init__()
        self.video_tokens, self.audio_tokens = video_tokens, audio_tokens
        table = table.detach().float().clone()
        table[:, 1].add_(1)
        table[:, 4].add_(1)
        table[:, 2].mul_(policy.attention_output_scale / policy.residual_scale)
        table[:, 5].mul_(policy.ffn_output_scale / policy.residual_scale)
        validate_fp16_constants(table, name="H3 folded modulation")
        self.modulation = IndexedModulation(table, sdk_ops=sdk_ops)
        self.role_spans = role_spans(role_capacities) if role_capacities is not None else None
        if sdk_ops and self.role_spans is None:
            from xhquant.nn import Gather

            self.role_gather = Gather(axis=0)
        self.norm1 = policy.residual_norm(source.norm1, sdk_ops=sdk_ops)
        self.norm2 = policy.residual_norm(source.norm2, sdk_ops=sdk_ops)
        self.attention = H3Attention(
            source.attn,
            output_divisor=policy.attention_output_scale,
            sdk_ops=sdk_ops,
            attention_block_size=attention_block_size,
        )
        self.ffn = H3SwiGLU(source.ff, product_divisor=policy.ffn_product_scale, output_divisor=policy.ffn_output_scale)

    def apply_modal(self, x, scale, shift=None, row_roles=None):
        if self.role_spans is not None:
            if row_roles is not None:
                raise ValueError("fixed-role H3 blocks do not accept token-wise modulation indices")
            return role_affine(x, scale, shift, self.role_spans)
        if row_roles is not None:
            values = (
                self.role_gather(scale, row_roles)
                if self.modulation.sdk_ops
                else scale.index_select(0, row_roles.long())
            )
            result = x * values
            if shift is not None:
                values = (
                    self.role_gather(shift, row_roles)
                    if self.modulation.sdk_ops
                    else shift.index_select(0, row_roles.long())
                )
                result = result + values
            return result
        split1, split2 = self.video_tokens, self.video_tokens + self.audio_tokens
        # Table order video=0,text=1,audio=2; sequence order video,audio,text.
        video, audio, text = x[:, :split1], x[:, split1:split2], x[:, split2:]
        groups = (video * scale[0], audio * scale[2], text * scale[1])
        if shift is not None:
            groups = (groups[0] + shift[0], groups[1] + shift[2], groups[2] + shift[1])
        return torch.cat(groups, 1)

    def forward(self, hidden, step, cos, sin, valid_length, row_roles=None, attention_metadata=None):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(step)
        h = self.apply_modal(self.norm1(hidden), scale1, shift1, row_roles)
        hidden = hidden + self.apply_modal(
            self.attention(h, cos, sin, valid_length, attention_metadata), gate1, row_roles=row_roles
        )
        h = self.apply_modal(self.norm2(hidden), scale2, shift2, row_roles)
        return hidden + self.apply_modal(self.ffn(h), gate2, row_roles=row_roles)


class H3OutputGraph(nn.Module):
    def __init__(self, source, table, video_tokens, audio_tokens, policy=DEFAULT_H3_POLICY, *, sdk_ops=False):
        super().__init__()
        self.video_tokens, self.audio_tokens = video_tokens, audio_tokens
        self.norm = policy.residual_norm(source.norm_out.norm, sdk_ops=sdk_ops)
        table = table.detach().float().clone()
        table[:, 1].add_(1)
        validate_fp16_constants(table, name="H3 folded output modulation")
        self.register_buffer("modulation", table)
        self.sdk_ops = sdk_ops
        if sdk_ops:
            from xhquant.nn import Gather

            self.gather = Gather(axis=0)
        self.video_head, self.audio_head = source.proj_out, source.audio_proj_out

    def forward(self, hidden, step, row_roles=None):
        rows = self.gather(self.modulation, step) if self.sdk_ops else self.modulation.index_select(0, step.long())
        rows = rows.squeeze(0)
        shift, scale = rows[0], rows[1]
        if row_roles is not None:
            roles = row_roles[: self.video_tokens + self.audio_tokens]
            selected_scale = self.gather(scale, roles) if self.sdk_ops else scale.index_select(0, roles.long())
            selected_shift = self.gather(shift, roles) if self.sdk_ops else shift.index_select(0, roles.long())
            values = self.norm(hidden[:, : roles.shape[0]]) * selected_scale + selected_shift
            return self.video_head(values[:, : self.video_tokens]), self.audio_head(values[:, self.video_tokens :])
        video = self.norm(hidden[:, : self.video_tokens]) * scale[0] + shift[0]
        audio = self.norm(hidden[:, self.video_tokens : self.video_tokens + self.audio_tokens]) * scale[2] + shift[2]
        return self.video_head(video), self.audio_head(audio)


class H3DenoiserGraph(nn.Module):
    def __init__(
        self, source, tables, output_table, video_tokens, audio_tokens, policy=DEFAULT_H3_POLICY, *, sdk_ops=False
    ):
        super().__init__()
        if len(tables) != len(source.transformer_blocks):
            raise ValueError("one folded table is required per H3 block")
        self.residual_scale = policy.residual_scale
        self.video_projection, self.audio_projection = source.proj_in, source.audio_proj_in
        self.blocks = nn.ModuleList(
            [
                H3Block(block, table, video_tokens, audio_tokens, policy, sdk_ops=sdk_ops)
                for block, table in zip(source.transformer_blocks, tables, strict=True)
            ]
        )
        self.output = H3OutputGraph(source, output_table, video_tokens, audio_tokens, policy, sdk_ops=sdk_ops)

    def forward(self, video_tokens, audio_tokens, text_features, step, cos, sin, valid_length):
        return self._forward(video_tokens, audio_tokens, text_features, step, cos, sin, valid_length, None)

    def _forward(self, video_tokens, audio_tokens, text_features, step, cos, sin, valid_length, row_roles):
        hidden = self.project_inputs(video_tokens, audio_tokens, text_features)
        for block in self.blocks:
            hidden = block(hidden, step, cos, sin, valid_length, row_roles)
        return self.output(hidden, step, row_roles)

    def project_inputs(self, video_tokens, audio_tokens, text_features):
        return torch.cat(
            (
                self.video_projection(video_tokens) / self.residual_scale,
                self.audio_projection(audio_tokens) / self.residual_scale,
                text_features / self.residual_scale,
            ),
            1,
        )


class ConditionedH3DenoiserGraph(H3DenoiserGraph):
    """Explicit eighth graph input for reference/keyframe row roles."""

    def forward(self, video_tokens, audio_tokens, text_features, step, cos, sin, valid_length, row_roles):
        return self._forward(video_tokens, audio_tokens, text_features, step, cos, sin, valid_length, row_roles)


class PackedConditionedH3DenoiserGraph(H3DenoiserGraph):
    """Compact variable reference lengths before attention; restore modal output slots."""

    def __init__(self, *args, sdk_ops=False, **kwargs):
        super().__init__(*args, sdk_ops=sdk_ops, **kwargs)
        self.sdk_ops = sdk_ops
        if sdk_ops:
            from xhquant.nn import Gather

            self.sequence_gather = Gather(axis=1)
            self.role_gather = Gather(axis=0)

    def forward(
        self,
        video_tokens,
        audio_tokens,
        text_features,
        step,
        cos,
        sin,
        valid_length,
        row_roles,
        pack_indices,
        unpack_indices,
    ):
        hidden = self.project_inputs(video_tokens, audio_tokens, text_features)
        if self.sdk_ops:
            hidden = self.sequence_gather(hidden, pack_indices)
            roles = self.role_gather(row_roles, pack_indices)
        else:
            hidden = hidden.index_select(1, pack_indices.long())
            roles = row_roles.index_select(0, pack_indices.long())
        for block in self.blocks:
            hidden = block(hidden, step, cos, sin, valid_length, roles)
        hidden = (
            self.sequence_gather(hidden, unpack_indices)
            if self.sdk_ops
            else hidden.index_select(1, unpack_indices.long())
        )
        return self.output(hidden, step, row_roles)


class RolePartitionedH3Output(nn.Module):
    """Time-only native output modulation, broadcast in the fixed role layout."""

    def __init__(self, source, table, video_tokens, audio_tokens, policy, *, role_capacities, sdk_ops=False):
        super().__init__()
        self.video_tokens, self.audio_tokens = video_tokens, audio_tokens
        self.spans = role_spans(role_capacities)
        self.norm = policy.residual_norm(source.norm_out.norm, sdk_ops=sdk_ops)
        table = table.detach().float().clone()
        table[:, 1].add_(1)
        validate_fp16_constants(table, name="H3 fixed-role output modulation")
        self.register_buffer("modulation", table)
        self.video_head, self.audio_head = source.proj_out, source.audio_proj_out
        self.register_buffer("padding_row", torch.zeros(1, 1, source.proj_out.in_features))
        self.sdk_ops = sdk_ops
        if sdk_ops:
            from xhquant.nn import Gather

            self.step_gather = Gather(axis=0)
            self.head_gather = Gather(axis=1)

    def forward(self, hidden, step, head_unpack):
        rows = self.step_gather(self.modulation, step) if self.sdk_ops else self.modulation.index_select(0, step.long())
        rows = rows.squeeze(0)
        shift, scale = rows[0:1].squeeze(0), rows[1:2].squeeze(0)
        adapted = role_affine(self.norm(hidden), scale, shift, self.spans)
        # Invalid source output slots select this explicit zero row. Reference
        # velocities retain the old ABI; the host solver never consumes them.
        adapted = torch.cat((adapted, self.padding_row), dim=1)
        restored = (
            self.head_gather(adapted, head_unpack) if self.sdk_ops else adapted.index_select(1, head_unpack.long())
        )
        return self.video_head(restored[:, : self.video_tokens]), self.audio_head(restored[:, self.video_tokens :])


class RolePartitionedH3DenoiserGraph(nn.Module):
    """Fixed role spans and masked dense attention without layer repacking.

    The one-dimensional key mask describes each role's independent padding.
    FlashAttention consumes it without materializing a query-by-key mask.
    """

    def __init__(
        self,
        source,
        tables,
        output_table,
        video_tokens,
        audio_tokens,
        policy=DEFAULT_H3_POLICY,
        *,
        role_capacities,
        attention_block_size=256,
        sdk_ops=False,
    ):
        super().__init__()
        role_spans(role_capacities)
        if attention_block_size not in (64, 128, 256, 512) or any(n % attention_block_size for n in role_capacities):
            raise ValueError("H3 fixed role spans must align to the attention block size")
        if len(tables) != len(source.transformer_blocks) or any(table.shape[2] != 5 for table in tables):
            raise ValueError("fixed-role H3 requires one five-role table per transformer block")
        self.role_capacities = tuple(role_capacities)
        self.attention_block_size = attention_block_size
        self.residual_scale, self.sdk_ops = policy.residual_scale, sdk_ops
        self.video_projection, self.audio_projection = source.proj_in, source.audio_proj_in
        self.register_buffer("padding_row", torch.zeros(1, 1, source.proj_in.out_features))
        self.blocks = nn.ModuleList(
            H3Block(
                block,
                table,
                video_tokens,
                audio_tokens,
                policy,
                sdk_ops=sdk_ops,
                role_capacities=role_capacities,
                attention_block_size=attention_block_size,
            )
            for block, table in zip(source.transformer_blocks, tables, strict=True)
        )
        self.output = RolePartitionedH3Output(
            source, output_table, video_tokens, audio_tokens, policy, role_capacities=role_capacities, sdk_ops=sdk_ops
        )
        if sdk_ops:
            from xhquant.nn import Gather

            self.input_gather = Gather(axis=1)

    def forward(
        self,
        video_tokens,
        audio_tokens,
        text_features,
        step,
        cos,
        sin,
        role_pack,
        head_unpack,
        key_mask,
    ):
        hidden = torch.cat(
            (
                self.video_projection(video_tokens) / self.residual_scale,
                self.audio_projection(audio_tokens) / self.residual_scale,
                text_features / self.residual_scale,
            ),
            dim=1,
        )
        hidden = torch.cat((hidden, self.padding_row), dim=1)
        hidden = self.input_gather(hidden, role_pack) if self.sdk_ops else hidden.index_select(1, role_pack.long())
        for block in self.blocks:
            hidden = block(hidden, step, cos, sin, None, attention_metadata=key_mask)
        return self.output(hidden, step, head_unpack)
