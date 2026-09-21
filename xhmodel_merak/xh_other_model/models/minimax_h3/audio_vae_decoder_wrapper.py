import torch
import torch.nn as nn
from torch.nn.utils import parametrize, remove_weight_norm
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
    MiniMaxH3AudioActivation1d,
    MiniMaxH3AudioAMPBlock,
    MiniMaxH3AudioBigVGANDecoder,
    MiniMaxH3AudioDownSample1d,
    MiniMaxH3AudioLowPassFilter1d,
    MiniMaxH3AudioSnakeBeta,
    MiniMaxH3AudioUpSample1d,
)

from xhquant import nn as xhnn
from xhquant.utils.registry.dynamic_module import DynamicModule


class _MiniMaxH3AudioDepthwiseConv1d(nn.Module):
    def __init__(self, filter_: torch.Tensor, stride: int, transpose: bool = False):
        super().__init__()
        convolution_cls = nn.ConvTranspose1d if transpose else nn.Conv1d
        self.convolution = convolution_cls(
            in_channels=1,
            out_channels=1,
            kernel_size=filter_.shape[-1],
            stride=stride,
            bias=False,
            device=filter_.device,
            dtype=filter_.dtype,
        )
        with torch.no_grad():
            self.convolution.weight.copy_(filter_)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, sequence_length = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size * num_channels, 1, sequence_length)
        hidden_states = self.convolution(hidden_states)
        return hidden_states.reshape(batch_size, num_channels, -1)


class _MiniMaxH3AudioClamp(nn.Module):
    def __init__(self, minimum: float, maximum: float):
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.clamp(hidden_states, min=self.minimum, max=self.maximum)


class _MiniMaxH3AudioSnakeBeta(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.register_buffer("alpha_exp", torch.exp(self.alpha.detach()).reshape(1, -1, 1))
        self.register_buffer("beta_exp", torch.exp(self.beta.detach()).reshape(1, -1, 1))
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + (self.beta_exp + 1e-9).reciprocal() * torch.sin(
            self.alpha_exp * hidden_states
        ) * torch.sin(self.alpha_exp * hidden_states)


class _MiniMaxH3AudioLowPassFilter1d(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.padding = xhnn.Pad(mode="replicate")
        self.convolution = _MiniMaxH3AudioDepthwiseConv1d(self.filter, self.stride)
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.padding(hidden_states, [self.pad_left, self.pad_right], 0.0)
        return self.convolution(hidden_states)


class _MiniMaxH3AudioUpSample1d(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.padding = xhnn.Pad(mode="replicate")
        self.convolution = _MiniMaxH3AudioDepthwiseConv1d(self.filter, self.stride, transpose=True)
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.padding(hidden_states, [self.pad, self.pad], 0.0)
        hidden_states = self.ratio * self.convolution(hidden_states)
        return hidden_states[..., self.pad_left : -self.pad_right]


class _MiniMaxH3AudioDownSample1d(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lowpass(hidden_states)


class _MiniMaxH3AudioActivation1d(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(hidden_states)))


class _MiniMaxH3AudioAMPBlock(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2, strict=True):
            residual = conv2(act2(conv1(act1(hidden_states))))
            hidden_states = residual + hidden_states
        return hidden_states


class _MiniMaxH3AudioBigVGANDecoder(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.output_clamp = _MiniMaxH3AudioClamp(-1.0, 1.0)
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_pre(hidden_states)
        for index in range(self.num_upsamples):
            hidden_states = self.ups[index][0](hidden_states)
            residual = self.resblocks[index * self.num_kernels](hidden_states)
            for kernel_index in range(1, self.num_kernels):
                residual = residual + self.resblocks[index * self.num_kernels + kernel_index](hidden_states)
            hidden_states = residual / self.num_kernels
        hidden_states = self.conv_post(self.activation_post(hidden_states))
        return self.output_clamp(hidden_states)


_AUDIO_DECODER_WRAPPERS = {
    MiniMaxH3AudioSnakeBeta: _MiniMaxH3AudioSnakeBeta,
    MiniMaxH3AudioLowPassFilter1d: _MiniMaxH3AudioLowPassFilter1d,
    MiniMaxH3AudioUpSample1d: _MiniMaxH3AudioUpSample1d,
    MiniMaxH3AudioDownSample1d: _MiniMaxH3AudioDownSample1d,
    MiniMaxH3AudioActivation1d: _MiniMaxH3AudioActivation1d,
    MiniMaxH3AudioAMPBlock: _MiniMaxH3AudioAMPBlock,
    MiniMaxH3AudioBigVGANDecoder: _MiniMaxH3AudioBigVGANDecoder,
}


def _materialize_weight_norm(module: nn.Module) -> None:
    for child in module.modules():
        try:
            remove_weight_norm(child)
        except ValueError:
            pass
        if parametrize.is_parametrized(child, "weight"):
            parametrize.remove_parametrizations(child, "weight", leave_parametrized=True)


def wrap_audio_vae_decoder(audio_vae: nn.Module) -> nn.Module:
    _materialize_weight_norm(audio_vae)
    for _, module in list(audio_vae.decoder.named_modules()):
        dynamic_cls = _AUDIO_DECODER_WRAPPERS.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            dynamic_cls.convert(module)
    return audio_vae


class MiniMaxH3AudioVAEDecoderWrapper(nn.Module):
    def __init__(self, audio_vae: nn.Module):
        super().__init__()
        self.audio_vae = wrap_audio_vae_decoder(audio_vae)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = self.audio_vae.decoder(self.audio_vae.dec_in_proj(latents))
        return decoded


__all__ = ["MiniMaxH3AudioVAEDecoderWrapper", "wrap_audio_vae_decoder"]
