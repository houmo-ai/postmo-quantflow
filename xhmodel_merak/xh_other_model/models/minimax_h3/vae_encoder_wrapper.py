from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
    MiniMaxH3VideoCausalConv3d,
    MiniMaxH3VideoDownBlock3d,
    MiniMaxH3VideoDownsample3d,
    MiniMaxH3VideoEncoder3d,
    MiniMaxH3VideoGroupNorm,
    MiniMaxH3VideoResnetBlock3d,
)

from xhmodel_merak.xh_other_model.models.minimax_h3 import model as _model  # noqa: F401
from xhquant.utils.registry.dynamic_module import DynamicModule


class _MiniMaxH3VideoCausalConv3d(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.conv = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
        )
        self.conv.weight = nn.Parameter(deepcopy(self.weight.data))
        if self.bias is not None:
            self.conv.bias = nn.Parameter(deepcopy(self.bias.data))
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_padding > 0:
            padding = self.spatial_padding
            hidden_states = F.pad(
                hidden_states,
                (padding, padding, padding, padding, 0, 0),
                mode=self.spatial_padding_mode,
            )
        if self.temporal_padding > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0))
        return self.conv(hidden_states)


class _MiniMaxH3VideoGroupNorm(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.norm = nn.GroupNorm(
            self.num_groups,
            self.num_channels,
            eps=self.eps,
            affine=self.affine,
        )
        if self.affine:
            self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
            self.norm.bias = nn.Parameter(deepcopy(self.bias.data))
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        hidden_states = hidden_states.view(batch_size * num_frames, num_channels, 1, height, width)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(batch_size, num_frames, num_channels, height, width)
        return hidden_states.permute(0, 2, 1, 3, 4)


class _MiniMaxH3VideoResnetBlock3d(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.silu(self.norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)
        hidden_states = F.silu(self.norm2(hidden_states))
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class _MiniMaxH3VideoDownsample3d(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_stride == 2:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1, 0, 0), mode=self.spatial_padding_mode)
        return self.conv(hidden_states)


class _MiniMaxH3VideoDownBlock3d(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class _MiniMaxH3VideoEncoder3d(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        hidden_states = F.silu(self.norm_out(hidden_states))
        return self.conv_out(hidden_states)


_ENCODER_WRAPPERS = {
    MiniMaxH3VideoCausalConv3d: _MiniMaxH3VideoCausalConv3d,
    MiniMaxH3VideoGroupNorm: _MiniMaxH3VideoGroupNorm,
    MiniMaxH3VideoResnetBlock3d: _MiniMaxH3VideoResnetBlock3d,
    MiniMaxH3VideoDownsample3d: _MiniMaxH3VideoDownsample3d,
    MiniMaxH3VideoDownBlock3d: _MiniMaxH3VideoDownBlock3d,
    MiniMaxH3VideoEncoder3d: _MiniMaxH3VideoEncoder3d,
}


def wrap_vae_encoder(vae: nn.Module) -> nn.Module:
    for _, module in list(vae.encoder.named_modules()):
        dynamic_cls = _ENCODER_WRAPPERS.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            dynamic_cls.convert(module)
    return vae


class MiniMaxH3VAEEncoderTileWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, quant_conv: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv

    def forward(self, tile: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(tile))


class MiniMaxH3VAEEncoderWrapper(nn.Module):
    def __init__(
        self,
        vae: nn.Module,
        tile_encoder: nn.Module | None = None,
        *,
        wrap_modules: bool = True,
    ):
        super().__init__()
        self.vae = wrap_vae_encoder(vae) if wrap_modules else vae
        self.tile_encoder = tile_encoder or MiniMaxH3VAEEncoderTileWrapper(self.vae.encoder, self.vae.quant_conv)

    def _encode_clip(self, video: torch.Tensor) -> torch.Tensor:
        if not self.vae.use_tiling:
            return self.tile_encoder(video)

        height, width = video.shape[-2:]
        y_indices, y_lengths, y_overlaps = self.vae._split_tiles(
            height,
            self.vae.tile_sample_min_height,
            self.vae.tile_sample_min_overlap_height,
        )
        x_indices, x_lengths, x_overlaps = self.vae._split_tiles(
            width,
            self.vae.tile_sample_min_width,
            self.vae.tile_sample_min_overlap_width,
        )
        rows = []
        for y_index, y_length in zip(y_indices, y_lengths, strict=True):
            row = []
            for x_index, x_length in zip(x_indices, x_lengths, strict=True):
                tile = video[..., y_index : y_index + y_length, x_index : x_index + x_length]
                row.append(self.tile_encoder(tile))
            rows.append(row)
        ratio = self.vae.spatial_compression_ratio
        latent_y_overlaps = [overlap // ratio for overlap in y_overlaps]
        latent_x_overlaps = [overlap // ratio for overlap in x_overlaps]
        return self.vae._stitch_tiles(rows, latent_y_overlaps, latent_x_overlaps)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        clip_length = self.vae.config.clip_length
        num_frames = video.shape[2]
        if num_frames == 1:
            return self._encode_clip(video)
        if num_frames % clip_length != 0:
            pad_frames = (-num_frames) % clip_length
            video = torch.cat([video, video[:, :, -1:].repeat(1, 1, pad_frames, 1, 1)], dim=2)

        moments = torch.cat(
            [
                self._encode_clip(video[:, :, index * clip_length : (index + 1) * clip_length])
                for index in range(video.shape[2] // clip_length)
            ],
            dim=2,
        )
        if self.vae.config.token_drop > 0:
            moments = moments[:, :, : -self.vae.config.token_drop]
        return moments


__all__ = ["MiniMaxH3VAEEncoderTileWrapper", "MiniMaxH3VAEEncoderWrapper", "wrap_vae_encoder"]
