# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Constructed component weights for offline Sol-H3 graph integration.

These small PyTorch networks exercise the deployment wrappers. They are not
copies of, or equivalence oracles for, the author's LTX-2.5 implementation.
Every learned branch contains real parameters; no forward is mocked.
"""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from xhmodel_merak.xh_other_model.models.sol_h3.vae_graph import (
    CausalPadConv,
    ChannelPixelNorm,
    DepthExpansion,
    DepthReduction,
)


class UpscaleResidual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.in_layers = nn.Sequential(nn.GroupNorm(1, width), nn.SiLU(), nn.Conv3d(width, width, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 2))
        self.out_norm = nn.GroupNorm(1, width)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Conv3d(width, width, 3, padding=1))
        self.skip = nn.Identity()

    def forward(self, value, embedding):
        scale, shift = self.emb_layers(embedding).chunk(2, -1)
        hidden = self.out_norm(self.in_layers(value))
        hidden = hidden * (1 + scale[..., None, None, None]) + shift[..., None, None, None]
        return value + self.out_layers(hidden)


class SmallUpscaler(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.conv_in = nn.Conv3d(24, width, 3, padding=1)
        self.in_blocks = nn.ModuleList([UpscaleResidual(width)])
        self.out_blocks = nn.ModuleList([UpscaleResidual(width)])
        self.norm_out = nn.GroupNorm(1, width)
        self.conv_out = nn.Conv3d(width, 24, 3, padding=1)
        self.register_buffer("comfy_latents_mean", torch.linspace(-0.1, 0.1, 24).reshape(1, 24, 1, 1, 1))
        self.register_buffer("comfy_latents_std", torch.linspace(0.8, 1.2, 24).reshape(1, 24, 1, 1, 1))

    def forward(self, value):
        embedding = self.embed(value.new_ones((1, 1)))
        value = self.conv_in(value)
        for block in self.in_blocks:
            value = block(value, embedding)
        value = F.interpolate(value, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            value = block(value, embedding)
        return self.conv_out(F.silu(self.norm_out(value)))


class AdapterBlock(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.norm = nn.GroupNorm(2, width)
        self.spatial = nn.Conv3d(width, width, (1, 3, 3), padding=(0, 1, 1), groups=2)
        self.temporal = nn.Conv3d(width, width, (3, 1, 1), padding=(1, 0, 0), groups=2)
        self.in_proj = nn.Conv3d(width, width * 4, 1)
        self.out_proj = nn.Conv3d(width * 2, width, 1)

    def forward(self, value):
        hidden = self.temporal(self.spatial(F.silu(self.norm(value))))
        signal, gate = self.in_proj(F.silu(hidden)).chunk(2, 1)
        return value + self.out_proj(signal * F.silu(gate))


class SmallAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.skip = nn.Conv3d(384, 128, 1)
        self.stem = nn.Conv3d(384, 8, 1)
        self.blocks = nn.Sequential(AdapterBlock(), AdapterBlock())
        self.final_norm = nn.GroupNorm(2, 8)
        self.head = nn.Conv3d(8, 128, 1)

    def forward(self, aligned):
        return self.skip(aligned) + self.head(F.silu(self.final_norm(self.blocks(self.stem(aligned)))))


def bridge_sources():
    return SmallUpscaler().eval(), SmallAdapter().eval()


def causal_conv(cin, cout, *, causal=False):
    return CausalPadConv(
        SimpleNamespace(conv=nn.Conv3d(cin, cout, 3, padding=(0, 1, 1)), time_kernel_size=3),
        static_causal=causal,
    )


def statistics(channels, *, audio=False):
    result = nn.Module()
    shape = (channels, 16) if audio else (channels,)
    result.register_buffer(
        "mean-of-means", torch.linspace(-0.1, 0.1, channels).reshape(-1, *([1] if audio else [])).expand(shape).clone()
    )
    result.register_buffer(
        "std-of-means", torch.linspace(0.8, 1.2, channels).reshape(-1, *([1] if audio else [])).expand(shape).clone()
    )
    return result


def video_encoder_source():
    source = nn.Module()
    source.patch_size, source.latent_channels = 4, 128
    source.video_scale_factors = SimpleNamespace(time=8)
    source.latent_log_var = "per_channel"
    source.conv_in = causal_conv(48, 8, causal=True)
    source.down_blocks = nn.ModuleList(
        [
            DepthReduction(SimpleNamespace(conv=causal_conv(8, 2, causal=True), stride=(1, 2, 2), group_size=4))
            for _ in range(3)
        ]
    )
    source.conv_norm_out = ChannelPixelNorm(1e-6)
    source.conv_act = nn.SiLU()
    source.conv_out = causal_conv(8, 256, causal=True)
    source.per_channel_statistics = statistics(128)
    return source.eval()


def video_decoder_source():
    source = nn.Module()
    source.timestep_conditioning, source.causal, source.patch_size = False, False, 4
    source.conv_in = causal_conv(128, 8)
    source.up_blocks = nn.ModuleList(
        [
            DepthExpansion(
                SimpleNamespace(
                    conv=causal_conv(8, 64), stride=(2, 2, 2), residual=False, out_channels_reduction_factor=1
                )
            )
            for _ in range(3)
        ]
    )
    source.conv_norm_out = ChannelPixelNorm(1e-6)
    source.conv_act = nn.SiLU()
    source.conv_out = causal_conv(8, 48)
    source.per_channel_statistics = statistics(128)
    return source.eval()


class AudioResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(ChannelPixelNorm(1e-6), nn.SiLU(), nn.Conv2d(8, 8, 3, padding=1))

    def forward(self, value, temb=None):
        return value + self.layers(value)


def audio_encoder_source():
    source = nn.Module()
    source.sample_rate, source.mel_bins, source.mel_hop_length, source.n_fft = 16000, 64, 160, 1024
    source.num_res_blocks = 1
    source.conv_in = nn.Conv2d(2, 8, 3, padding=1)
    source.down = nn.ModuleList()
    for index in range(3):
        stage = nn.Module()
        stage.block, stage.attn = nn.ModuleList([AudioResidual()]), nn.ModuleList()
        stage.downsample = nn.Conv2d(8, 8, 3, stride=2, padding=1) if index < 2 else nn.Identity()
        source.down.append(stage)
    source.mid = nn.Module()
    source.mid.block_1, source.mid.attn_1, source.mid.block_2 = AudioResidual(), nn.Identity(), AudioResidual()
    source.norm_out, source.non_linearity = ChannelPixelNorm(1e-6), nn.SiLU()
    source.conv_out = nn.Conv2d(8, 16, 3, padding=1)
    source.per_channel_statistics = statistics(8, audio=True)
    return source.eval()


class SmallAttention(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.heads, self.dim_head = 2, 64
        self.to_q, self.to_k, self.to_v = (nn.Linear(width, width) for _ in range(3))
        self.to_out = nn.ModuleList([nn.Linear(width, width)])
        self.q_norm, self.k_norm = nn.RMSNorm(width, eps=1e-6), nn.RMSNorm(width, eps=1e-6)
        self.to_gate_logits = nn.Linear(width, self.heads)


class SmallAVBlock(nn.Module):
    def __init__(self):
        super().__init__()
        from diffusers.models.attention import FeedForward

        self.norm_eps = 1e-6
        for name in ("attn1", "audio_attn1", "attn2", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn"):
            setattr(self, name, SmallAttention())
        self.ff, self.audio_ff = (FeedForward(128, inner_dim=256, activation_fn="gelu-approximate") for _ in range(2))
        for name, count in (
            ("scale_shift_table", 9),
            ("audio_scale_shift_table", 9),
            ("scale_shift_table_a2v_ca_video", 5),
            ("scale_shift_table_a2v_ca_audio", 5),
            ("prompt_scale_shift_table", 2),
            ("audio_prompt_scale_shift_table", 2),
        ):
            self.register_parameter(name, nn.Parameter(torch.randn(count, 128) * 0.02))

    @staticmethod
    def get_ada_values(table, batch, timesteps, indices):
        values = table[None, None] + timesteps.reshape(batch, -1, table.shape[0], table.shape[1])
        return values.unbind(2)


def refiner_source():
    source = nn.Module()
    source.patchify_proj, source.audio_patchify_proj = nn.Linear(128, 128), nn.Linear(128, 128)
    source.transformer_blocks = nn.ModuleList([SmallAVBlock()])
    source.norm_out, source.audio_norm_out = nn.RMSNorm(128, eps=1e-6), nn.RMSNorm(128, eps=1e-6)
    source.proj_out, source.audio_proj_out = nn.Linear(128, 128), nn.Linear(128, 128)
    source.scale_shift_table = nn.Parameter(torch.randn(2, 128) * 0.02)
    source.audio_scale_shift_table = nn.Parameter(torch.randn(2, 128) * 0.02)
    source.keyframe_embedding = nn.Parameter(torch.randn(1, 1, 128) * 0.02)
    source._keyframes_embedding = lambda: source.keyframe_embedding
    source.time_projection = nn.Linear(4, 9 * 128)
    source.cross_projection = nn.Linear(4, 4 * 128)
    source.gate_projection = nn.Linear(4, 128)
    source.output_time_projection = nn.Linear(4, 128)
    source.prompt_projection = nn.Linear(4, 2 * 128)
    return source.eval()


@torch.no_grad()
def refiner_conditioning(source, cache, *, size, keyframe=False):
    from xhmodel_merak.xh_other_model.models.sol_h3.conditioning import REFINER_SIGMAS

    source.half()
    count = (17 if keyframe else 16) * (size[0] // 32) * (size[1] // 32)
    args = [[], []]
    for sigma in REFINER_SIGMAS[:-1]:
        for index, tokens in enumerate((count, 126)):
            times = torch.tensor([sigma, 0.0] if index == 0 and keyframe else [sigma]).half()
            features = torch.stack((times, times.square(), times.sin(), times.cos()), -1)
            global_features = features[:1]
            args[index].append(
                SimpleNamespace(
                    x=torch.zeros(1, tokens, 128).half(),
                    timesteps=source.time_projection(features)[None],
                    cross_scale_shift_timestep=source.cross_projection(features)[None],
                    cross_gate_timestep=source.gate_projection(global_features)[None],
                    embedded_timestep=source.output_time_projection(features)[None],
                    prompt_timestep=source.prompt_projection(global_features)[None],
                    context=cache["contexts"]["video" if index == 0 else "audio"].half(),
                )
            )
    ropes = []
    for tokens in (count, 126, count, 126):
        angles = torch.arange(tokens)[:, None] / (10000 ** (torch.arange(32).float() / 32))
        for value in (angles.cos(), angles.sin()):
            ropes.append(torch.cat((value, value), -1)[None, None].half())
    mask = torch.zeros(1, count, 1).half()
    mask[:, : (size[0] // 32) * (size[1] // 32)] = 1
    return args[0], args[1], tuple(ropes), mask
