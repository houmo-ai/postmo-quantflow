from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
import torch
from onnx import TensorProto
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.nn.utils import remove_weight_norm as remove_legacy_weight_norm

from xhquant import nn as xhnn

from .assets import KokoroAssets


FORBIDDEN_NPU_OPS = frozenset(
    {
        "Atan",
        "If",
        "Loop",
        "NonZero",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
        "SequenceAt",
        "SequenceEmpty",
        "SequenceInsert",
        "Sign",
        "STFT",
    }
)


@dataclass(frozen=True)
class GraphArtifact:
    role: str
    path: Path
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_contracts: tuple[dict[str, Any], ...]
    output_contracts: tuple[dict[str, Any], ...]
    node_count: int
    op_counts: dict[str, int]
    onnx_sha256: str
    structural_rewrites: dict[str, int]


def load_official_model(assets: KokoroAssets) -> nn.Module:
    root = str(assets.source_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import kokoro
        from kokoro.model import KModel
    except ImportError as error:
        raise ImportError(
            "Kokoro export requires the pinned upstream package and its dependencies. "
            "Install it with `pip install <model-dir>/source/kokoro`."
        ) from error
    version = getattr(kokoro, "__version__", None)
    if version != "0.9.4":
        raise RuntimeError(f"Kokoro 0.9.4 is required, found {version!r}")
    imported_from = Path(kokoro.__file__).resolve()
    if not imported_from.is_relative_to(assets.source_root.resolve()):
        raise RuntimeError(
            f"Kokoro was imported from {imported_from}, not the pinned source "
            f"checkout {assets.source_root}; restart Python after installing the pinned checkout"
        )
    model = KModel(
        repo_id="hexgrad/Kokoro-82M-v1.1-zh",
        config=str(assets.config),
        model=str(assets.checkpoint),
    )
    return model.eval()


def materialize_weight_norm(root: nn.Module) -> int:
    """Freeze both modern parametrization and legacy weight_norm wrappers."""

    count = 0
    for module in list(root.modules()):
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is not None and "weight" in parametrizations:
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            count += 1
            continue
        if hasattr(module, "weight_g") and hasattr(module, "weight_v"):
            remove_legacy_weight_norm(module, name="weight")
            count += 1
    leftovers = [
        name
        for name, module in root.named_modules()
        if (getattr(module, "parametrizations", None) is not None and "weight" in module.parametrizations)
        or hasattr(module, "weight_g")
        or hasattr(module, "weight_v")
    ]
    if leftovers:
        raise RuntimeError(f"weight_norm materialization incomplete: {leftovers}")
    return count


class DepthwiseDeconvK3S2AsElementwise(nn.Module):
    def __init__(self, source: nn.ConvTranspose1d) -> None:
        super().__init__()
        if not (
            source.in_channels == source.out_channels == source.groups
            and source.kernel_size == (3,)
            and source.stride == (2,)
            and source.padding == (1,)
            and source.output_padding == (1,)
            and source.dilation == (1,)
        ):
            raise ValueError("unsupported depthwise ConvTranspose1d contract")
        channels = source.in_channels
        weight = source.weight.detach()
        self.channels = channels
        self.register_buffer("left", weight[:, 0, 0].reshape(1, channels, 1).clone())
        self.register_buffer("center", weight[:, 0, 1].reshape(1, channels, 1).clone())
        self.register_buffer("right", weight[:, 0, 2].reshape(1, channels, 1).clone())
        self.bias = source.bias.detach().reshape(1, channels, 1).clone() if source.bias is not None else None

    def forward(self, values: Tensor) -> Tensor:
        even = values * self.center
        shifted = torch.cat([values[:, :, 1:], values[:, :, -1:] * 0.0], dim=-1)
        odd = values * self.right + shifted * self.left
        if self.bias is not None:
            even = even + self.bias
            odd = odd + self.bias
        return torch.stack([even, odd], dim=-1).reshape(values.shape[0], self.channels, values.shape[-1] * 2)


class SingleChannelConvK3S2AsElementwise(nn.Module):
    def __init__(self, source: nn.Conv1d) -> None:
        super().__init__()
        if not (
            source.in_channels == source.out_channels == source.groups == 1
            and source.kernel_size == (3,)
            and source.stride == (2,)
            and source.padding == (1,)
            and source.dilation == (1,)
        ):
            raise ValueError("unsupported single-channel Conv1d contract")
        weight = source.weight.detach().reshape(3)
        self.register_buffer("left", weight[0].reshape(1, 1, 1).clone())
        self.register_buffer("center", weight[1].reshape(1, 1, 1).clone())
        self.register_buffer("right", weight[2].reshape(1, 1, 1).clone())
        self.bias = source.bias.detach().reshape(1, 1, 1).clone() if source.bias is not None else None

    def forward(self, values: Tensor) -> Tensor:
        padded = F.pad(values, (1, 1), mode="constant", value=0.0)
        result = padded[:, :, 0:-2:2] * self.left + padded[:, :, 1:-1:2] * self.center + padded[:, :, 2::2] * self.right
        return result + self.bias if self.bias is not None else result


def rewrite_depthwise_deconvolution(root: nn.Module) -> int:
    replacements: list[tuple[nn.Module, str, nn.ConvTranspose1d]] = []
    for parent in root.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.ConvTranspose1d) and (
                child.in_channels == child.out_channels == child.groups
                and child.kernel_size == (3,)
                and child.stride == (2,)
                and child.padding == (1,)
                and child.output_padding == (1,)
                and child.dilation == (1,)
            ):
                replacements.append((parent, name, child))
    for parent, name, source in replacements:
        replacement = DepthwiseDeconvK3S2AsElementwise(source).to(
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        with torch.no_grad():
            sample = torch.randn(1, source.in_channels, 8, dtype=source.weight.dtype)
            error = float((source(sample) - replacement(sample)).abs().max().item())
        if error > 1e-5:
            raise RuntimeError(f"depthwise ConvTranspose rewrite mismatch: {error}")
        setattr(parent, name, replacement)
    return len(replacements)


def rewrite_single_channel_stride2_convolution(root: nn.Module) -> int:
    replacements: list[tuple[nn.Module, str, nn.Conv1d]] = []
    for parent in root.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Conv1d) and (
                child.in_channels == child.out_channels == child.groups == 1
                and child.kernel_size == (3,)
                and child.stride == (2,)
                and child.padding == (1,)
                and child.dilation == (1,)
            ):
                replacements.append((parent, name, child))
    for parent, name, source in replacements:
        replacement = SingleChannelConvK3S2AsElementwise(source).to(
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        with torch.no_grad():
            for length in (7, 8):
                sample = torch.randn(1, 1, length, dtype=source.weight.dtype)
                error = float((source(sample) - replacement(sample)).abs().max().item())
                if error > 1e-6:
                    raise RuntimeError(f"single-channel Conv rewrite mismatch: {error}")
        setattr(parent, name, replacement)
    return len(replacements)


def _copy_lstm_weights(target: nn.LSTM, source: nn.LSTM, direction: str) -> None:
    suffix = "" if direction == "forward" else "_reverse"
    with torch.no_grad():
        target.weight_ih_l0.copy_(getattr(source, f"weight_ih_l0{suffix}"))
        target.weight_hh_l0.copy_(getattr(source, f"weight_hh_l0{suffix}"))
        if source.bias:
            target.bias_ih_l0.copy_(getattr(source, f"bias_ih_l0{suffix}"))
            target.bias_hh_l0.copy_(getattr(source, f"bias_hh_l0{suffix}"))


def _dual_forward_lstm(source: nn.LSTM) -> tuple[nn.LSTM, nn.LSTM]:
    common = {
        "input_size": source.input_size,
        "hidden_size": source.hidden_size,
        "num_layers": 1,
        "batch_first": True,
        "bidirectional": False,
        "bias": source.bias,
    }
    forward = nn.LSTM(**common)
    backward = nn.LSTM(**common)
    _copy_lstm_weights(forward, source, "forward")
    _copy_lstm_weights(backward, source, "backward")
    return forward, backward


def _double_mask_add_albert_attention_forward(
    self: nn.Module,
    hidden_states: Tensor,
    attention_mask: Tensor | None = None,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Keep ALBERT eager attention with two explicit mask Add operations."""

    del kwargs
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.attention_head_size)
    query = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
    key = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
    value = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

    attention_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        attention_weights = attention_weights + attention_mask
        attention_weights = attention_weights + attention_mask
    attention_weights = F.softmax(attention_weights, dim=-1)
    attention_weights = F.dropout(
        attention_weights,
        p=self.attention_dropout.p,
        training=self.training,
    )

    attention_output = torch.matmul(attention_weights, value)
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(*input_shape, -1).contiguous()
    attention_output = self.dense(attention_output)
    attention_output = self.output_dropout(attention_output)
    attention_output = self.LayerNorm(hidden_states + attention_output)
    return attention_output, attention_weights


def rewrite_albert_attention_double_mask_add(root: nn.Module) -> int:
    """Wrap ALBERT attention so ONNX contains two direct mask Adds."""

    from transformers.models.albert.modeling_albert import AlbertAttention

    rewritten = 0
    for module in root.modules():
        if not isinstance(module, AlbertAttention):
            continue
        if getattr(module, "_kokoro_double_mask_add_rewritten", False):
            continue
        module.forward = types.MethodType(
            _double_mask_add_albert_attention_forward,
            module,
        )
        module._kokoro_double_mask_add_rewritten = True
        rewritten += 1
    if rewritten == 0 and not any(
        isinstance(module, AlbertAttention) and getattr(module, "_kokoro_double_mask_add_rewritten", False)
        for module in root.modules()
    ):
        raise RuntimeError("Kokoro BERT contains no Transformers AlbertAttention module")
    return rewritten


class KokoroFrontBaseStatic(nn.Module):
    def __init__(self, model: nn.Module, text_max_length: int) -> None:
        super().__init__()
        rewrite_albert_attention_double_mask_add(model.bert)
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        self.embedding = model.text_encoder.embedding
        self.cnn = model.text_encoder.cnn
        self.register_buffer(
            "position",
            torch.arange(text_max_length, dtype=torch.int32).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        valid_len: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ids = input_ids.long()
        padding = self.position >= valid_len.reshape(1, 1)
        # A four-dimensional additive mask bypasses Transformers' internal
        # dtype-min mask construction.  Host code supplies only 0 and -65504,
        # so conversion to FP16 never turns a float32 minimum into -inf.
        bert_hidden = self.bert(ids, attention_mask=attention_mask)
        duration_base = self.bert_encoder(bert_hidden).transpose(-1, -2)
        values = self.embedding(ids)
        values = values.transpose(1, 2).masked_fill(padding.unsqueeze(1), 0.0)
        for layer in self.cnn:
            values = layer(values).masked_fill(padding.unsqueeze(1), 0.0)
        return duration_base, values


class PrefixReversedBidirectionalLSTMStatic(nn.Module):
    """Lower one padded BiLSTM to two standard all-valid forward LSTMs.

    ONNX ``LSTM`` has no private valid-prefix permutation input.  Kokoro's
    padded batch-1 inputs therefore make the reverse dataflow explicit:
    Gather the valid prefix into reverse order, run the copied reverse weights
    through a normal forward LSTM, then Gather once more to restore time order.
    Both recurrent nodes exported by this wrapper are standards-compliant
    ``direction=forward`` ONNX LSTMs; padding is removed by the caller's mask.
    """

    def __init__(self, source_lstm: nn.LSTM, total_length: int) -> None:
        super().__init__()
        if not source_lstm.bidirectional:
            raise ValueError("Kokoro deployment requires a bidirectional LSTM")
        if source_lstm.num_layers != 1:
            raise ValueError("Kokoro deployment supports one-layer LSTM modules")
        if not source_lstm.batch_first:
            raise ValueError("Kokoro deployment expects batch_first LSTM modules")
        self.forward_lstm, self.backward_lstm = _dual_forward_lstm(source_lstm)
        self.total_length = int(total_length)

    def forward(self, values: Tensor, reverse_indices: Tensor) -> Tensor:
        forward, _ = self.forward_lstm(values)
        reversed_values = torch.index_select(values, 1, reverse_indices)
        backward_reversed, _ = self.backward_lstm(reversed_values)
        backward = torch.index_select(backward_reversed, 1, reverse_indices)
        return torch.cat((forward, backward), dim=-1)


class DurationBidirectionalLSTMBlockStatic(nn.Module):
    def __init__(self, source_lstm: nn.LSTM, adaln: nn.Module, text_max_length: int) -> None:
        super().__init__()
        self.lstm = PrefixReversedBidirectionalLSTMStatic(source_lstm, text_max_length)
        self.adaln = adaln
        self.text_max_length = int(text_max_length)

    def forward(
        self,
        values: Tensor,
        style: Tensor,
        reverse_indices: Tensor,
        mask: Tensor,
    ) -> Tensor:
        values = self.lstm(values, reverse_indices) * mask
        values = self.adaln(values, style) * mask
        expanded_style = style.unsqueeze(1).expand(-1, self.text_max_length, -1)
        return torch.cat([values, expanded_style], dim=-1) * mask


class DurationPredictorBidirectionalStatic(nn.Module):
    def __init__(self, model: nn.Module, text_max_length: int) -> None:
        super().__init__()
        self.lstm = PrefixReversedBidirectionalLSTMStatic(model.predictor.lstm, text_max_length)
        self.projection = model.predictor.duration_proj

    def forward(self, values: Tensor, reverse_indices: Tensor, mask: Tensor) -> Tensor:
        return self.projection(self.lstm(values, reverse_indices) * mask)


def _eps(module: nn.Module) -> float:
    return float(getattr(getattr(module, "norm", module), "eps", 1e-5))


def _masked_adain(module: nn.Module, values: Tensor, style: Tensor, mask: Tensor) -> Tensor:
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (values * mask).sum(dim=-1, keepdim=True) / count
    variance = ((values - mean).square() * mask).sum(dim=-1, keepdim=True) / count
    normalized = (values - mean) / torch.sqrt(variance + _eps(module))
    gamma, beta = module.fc(style).unsqueeze(-1).chunk(2, dim=1)
    return ((1.0 + gamma) * normalized + beta) * mask


class _MaskedTemporalRMSNorm(nn.Module):
    """Use native XH2 RMSNorm with Host-provided valid-region scales."""

    def __init__(self, temporal_length: int, eps: float) -> None:
        super().__init__()
        if temporal_length <= 0:
            raise ValueError("temporal_length must be positive")
        self.norm = xhnn.RMSNorm(int(temporal_length), eps=eps, reduce_dim=-1)
        self.norm.weight.requires_grad_(False)

    def forward(self, centered: Tensor, scale: Tensor, inverse_scale: Tensor) -> Tensor:
        # centered is zero in the padded region. RMSNorm divides its squared
        # sum by static T, while masked AdaIN divides by valid L. Host supplies
        # sqrt(T/L) and sqrt(L/T) independently so each gets the nearest input
        # representation instead of deriving one from an already-rounded one.
        return self.norm(centered * scale) * inverse_scale


def _masked_adain_rmsnorm(
    module: nn.Module,
    rmsnorm: _MaskedTemporalRMSNorm,
    values: Tensor,
    style: Tensor,
    mask: Tensor,
    scale: Tensor,
    inverse_scale: Tensor,
) -> Tensor:
    mean = _masked_temporal_mean(values, mask, scale)
    centered = (values - mean) * mask
    normalized = rmsnorm(centered, scale, inverse_scale)
    gamma, beta = module.fc(style).unsqueeze(-1).chunk(2, dim=1)
    return ((1.0 + gamma) * normalized + beta) * mask


def _masked_temporal_mean(values: Tensor, mask: Tensor, scale: Tensor) -> Tensor:
    """Compute a valid-prefix mean without an overflow-prone FP16 sum."""

    # ReduceSum over Generator's 120F+1 axis overflows FP16 once the valid
    # extent exceeds 65504.  ReduceMean keeps the reduction bounded; scale^2
    # restores T/L exactly because Host supplies scale=sqrt(T/L).
    return (values * mask).mean(dim=-1, keepdim=True) * scale.square()


def _masked_adain_block(
    block: nn.Module,
    values: Tensor,
    style: Tensor,
    input_mask: Tensor,
    output_mask: Tensor,
) -> Tensor:
    values = values * input_mask
    shortcut = block.upsample(values)
    if block.learned_sc:
        shortcut = block.conv1x1(shortcut)
    shortcut = shortcut * output_mask
    residual = _masked_adain(block.norm1, values, style, input_mask)
    residual = block.actv(residual)
    residual = block.pool(residual) * output_mask
    residual = block.conv1(block.dropout(residual)) * output_mask
    residual = _masked_adain(block.norm2, residual, style, output_mask)
    residual = block.actv(residual)
    residual = block.conv2(block.dropout(residual)) * output_mask
    return (residual + shortcut) * output_mask / math.sqrt(2.0)


def _masked_adain_rmsnorm_block(
    block: nn.Module,
    norm1: _MaskedTemporalRMSNorm,
    norm2: _MaskedTemporalRMSNorm,
    values: Tensor,
    style: Tensor,
    input_mask: Tensor,
    output_mask: Tensor,
    scale: Tensor,
    inverse_scale: Tensor,
) -> Tensor:
    values = values * input_mask
    shortcut = block.upsample(values)
    if block.learned_sc:
        shortcut = block.conv1x1(shortcut)
    shortcut = shortcut * output_mask
    residual = _masked_adain_rmsnorm(
        block.norm1,
        norm1,
        values,
        style,
        input_mask,
        scale,
        inverse_scale,
    )
    residual = block.actv(residual)
    residual = block.pool(residual) * output_mask
    residual = block.conv1(block.dropout(residual)) * output_mask
    residual = _masked_adain_rmsnorm(
        block.norm2,
        norm2,
        residual,
        style,
        output_mask,
        scale,
        inverse_scale,
    )
    residual = block.actv(residual)
    residual = block.conv2(block.dropout(residual)) * output_mask
    return (residual + shortcut) * output_mask / math.sqrt(2.0)


def _masked_generator_block(
    block: nn.Module,
    values: Tensor,
    style: Tensor,
    mask: Tensor,
) -> Tensor:
    values = values * mask
    for conv1, conv2, norm1, norm2, alpha1, alpha2 in zip(
        block.convs1,
        block.convs2,
        block.adain1,
        block.adain2,
        block.alpha1,
        block.alpha2,
        strict=True,
    ):
        residual = _masked_adain(norm1, values, style, mask)
        residual = (residual + torch.sin(alpha1 * residual).square() / alpha1) * mask
        residual = conv1(residual) * mask
        residual = _masked_adain(norm2, residual, style, mask)
        residual = (residual + torch.sin(alpha2 * residual).square() / alpha2) * mask
        residual = conv2(residual) * mask
        values = (values + residual) * mask
    return values


def _masked_generator_block_rmsnorm(
    block: nn.Module,
    rmsnorm: _MaskedTemporalRMSNorm,
    values: Tensor,
    style: Tensor,
    mask: Tensor,
    scale: Tensor,
    inverse_scale: Tensor,
) -> Tensor:
    values = values * mask
    for conv1, conv2, norm1, norm2, alpha1, alpha2 in zip(
        block.convs1,
        block.convs2,
        block.adain1,
        block.adain2,
        block.alpha1,
        block.alpha2,
        strict=True,
    ):
        residual = _masked_adain_rmsnorm(
            norm1,
            rmsnorm,
            values,
            style,
            mask,
            scale,
            inverse_scale,
        )
        residual = (residual + torch.sin(alpha1 * residual).square() / alpha1) * mask
        residual = conv1(residual) * mask
        residual = _masked_adain_rmsnorm(
            norm2,
            rmsnorm,
            residual,
            style,
            mask,
            scale,
            inverse_scale,
        )
        residual = (residual + torch.sin(alpha2 * residual).square() / alpha2) * mask
        residual = conv2(residual) * mask
        values = (values + residual) * mask
    return values


class F0BranchStatic(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        frame_max_length: int,
        *,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = model.predictor.F0
        self.projection = model.predictor.F0_proj
        self.frame_max_length = int(frame_max_length)
        self.use_rmsnorm = bool(use_rmsnorm)
        self.register_buffer(
            "frame_bucket",
            torch.tensor(float(frame_max_length), dtype=torch.float32),
            persistent=False,
        )
        self.rmsnorms = nn.ModuleList()
        if not self.use_rmsnorm:
            return

        current_length = int(frame_max_length)
        if current_length <= 0:
            raise ValueError("frame_max_length must be positive")
        upsampled = False
        for block in self.blocks:
            output_length = current_length
            if block.upsample_type != "none":
                if upsampled:
                    raise ValueError("F0 branch supports exactly one temporal upsample")
                output_length *= 2
                upsampled = True
            self.rmsnorms.append(
                nn.ModuleList(
                    [
                        _MaskedTemporalRMSNorm(current_length, _eps(block.norm1)),
                        _MaskedTemporalRMSNorm(output_length, _eps(block.norm2)),
                    ]
                )
            )
            current_length = output_length
        if current_length != 2 * frame_max_length:
            raise ValueError(f"F0 branch must upsample F={frame_max_length} to 2F, got {current_length}")

    def forward(
        self,
        shared: Tensor,
        style: Tensor,
        mask_f: Tensor,
        mask_2f: Tensor,
        norm_scales: Tensor | None = None,
    ) -> Tensor:
        values = shared * mask_f
        current_mask = mask_f
        scale = inverse_scale = None
        if self.use_rmsnorm:
            if norm_scales is None:
                # Single-graph compatibility path. Standalone/modular F0 graphs
                # supply these values from Host and do not export this path.
                valid_count = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
                scale = torch.sqrt(self.frame_bucket.to(dtype=shared.dtype) / valid_count)
                inverse_scale = torch.reciprocal(scale)
            else:
                if not torch.jit.is_tracing() and tuple(norm_scales.shape) != (1, 2, 1):
                    raise ValueError(f"norm_scales must be [1,2,1], got {tuple(norm_scales.shape)}")
                scale = norm_scales[:, 0:1, :]
                inverse_scale = norm_scales[:, 1:2, :]
        for index, block in enumerate(self.blocks):
            output_mask = mask_2f if block.upsample_type != "none" else current_mask
            if self.use_rmsnorm:
                values = _masked_adain_rmsnorm_block(
                    block,
                    self.rmsnorms[index][0],
                    self.rmsnorms[index][1],
                    values,
                    style,
                    current_mask,
                    output_mask,
                    scale,
                    inverse_scale,
                )
            else:
                values = _masked_adain_block(block, values, style, current_mask, output_mask)
            current_mask = output_mask
        return (self.projection(values) * mask_2f).reshape(1, 2 * self.frame_max_length)


class NoiseBranchStatic(nn.Module):
    def __init__(self, model: nn.Module, frame_max_length: int) -> None:
        super().__init__()
        self.blocks = model.predictor.N
        self.projection = model.predictor.N_proj
        self.frame_max_length = int(frame_max_length)

    def forward(self, shared: Tensor, style: Tensor, mask_f: Tensor, mask_2f: Tensor) -> Tensor:
        values = shared * mask_f
        current_mask = mask_f
        for block in self.blocks:
            output_mask = mask_2f if block.upsample_type != "none" else current_mask
            values = _masked_adain_block(block, values, style, current_mask, output_mask)
            current_mask = output_mask
        return (self.projection(values) * mask_2f).reshape(1, 2 * self.frame_max_length)


class DecoderStatic(nn.Module):
    def __init__(self, model: nn.Module, frame_max_length: int) -> None:
        super().__init__()
        decoder = model.decoder
        self.f0_conv = decoder.F0_conv
        self.noise_conv = decoder.N_conv
        self.encode = decoder.encode
        self.asr_residual = decoder.asr_res
        self.decode = decoder.decode
        self.frame_max_length = frame_max_length

    def forward(
        self,
        asr: Tensor,
        f0: Tensor,
        noise: Tensor,
        style: Tensor,
        mask_f: Tensor,
        mask_2f: Tensor,
    ) -> Tensor:
        mask_2f_flat = mask_2f.reshape(1, 2 * self.frame_max_length)
        f0 = f0 * mask_2f_flat
        noise = noise * mask_2f_flat
        asr = asr * mask_f
        f0_down = self.f0_conv(f0.unsqueeze(1)).reshape(1, 1, self.frame_max_length)
        noise_down = self.noise_conv(noise.unsqueeze(1)).reshape(1, 1, self.frame_max_length)
        f0_down = f0_down * mask_f
        noise_down = noise_down * mask_f
        values = torch.cat([asr, f0_down, noise_down], dim=1)
        values = _masked_adain_block(self.encode, values, style, mask_f, mask_f)
        asr_residual = self.asr_residual(asr) * mask_f
        current_mask = mask_f
        append_condition = True
        for block in self.decode:
            if append_condition:
                values = torch.cat([values, asr_residual, f0_down, noise_down], dim=1)
                values = values * current_mask
            output_mask = mask_2f if block.upsample_type != "none" else current_mask
            values = _masked_adain_block(block, values, style, current_mask, output_mask)
            current_mask = output_mask
            if block.upsample_type != "none":
                append_condition = False
        return values * mask_2f


class SourceMergeStatic(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        source = model.decoder.generator.m_source
        self.linear = source.l_linear
        self.tanh = source.l_tanh

    def forward(self, sine_wavs: Tensor) -> Tensor:
        return self.tanh(self.linear(sine_wavs))


class GeneratorStatic(nn.Module):
    def __init__(self, model: nn.Module, frame_max_length: int) -> None:
        super().__init__()
        self.generator = model.decoder.generator
        self.frame_max_length = int(frame_max_length)
        if self.frame_max_length <= 0:
            raise ValueError("frame_max_length must be positive")
        if self.generator.num_upsamples != 2:
            raise ValueError(f"Kokoro Generator must have two upsample stages, got {self.generator.num_upsamples}")

        stage_lengths = (20 * self.frame_max_length, 120 * self.frame_max_length + 1)
        self.rmsnorms = nn.ModuleList()
        for stage, temporal_length in enumerate(stage_lengths):
            blocks = [
                self.generator.noise_res[stage],
                *[
                    self.generator.resblocks[stage * self.generator.num_kernels + branch]
                    for branch in range(self.generator.num_kernels)
                ],
            ]
            eps_values = {_eps(norm) for block in blocks for norms in (block.adain1, block.adain2) for norm in norms}
            if len(eps_values) != 1:
                raise ValueError(f"Generator stage {stage} AdaIN eps values must match, got {sorted(eps_values)}")
            self.rmsnorms.append(_MaskedTemporalRMSNorm(temporal_length, eps_values.pop()))

    def forward(
        self,
        generator_feature: Tensor,
        style: Tensor,
        harmonic: Tensor,
        mask_2f: Tensor,
        mask_20f: Tensor,
        mask_wave: Tensor,
        norm_scales: Tensor,
    ) -> Tensor:
        if not torch.jit.is_tracing() and tuple(norm_scales.shape) != (1, 2, 2, 1):
            raise ValueError(f"norm_scales must be [1,2,2,1], got {tuple(norm_scales.shape)}")
        generator = self.generator
        values = generator_feature * mask_2f
        harmonic = harmonic * mask_wave
        stage_masks = (mask_20f, mask_wave)
        for stage in range(generator.num_upsamples):
            stage_mask = stage_masks[stage]
            scale = norm_scales[:, stage, 0:1, :]
            inverse_scale = norm_scales[:, stage, 1:2, :]
            values = F.leaky_relu(values, 0.1)
            source = generator.noise_convs[stage](harmonic) * stage_mask
            source = _masked_generator_block_rmsnorm(
                generator.noise_res[stage],
                self.rmsnorms[stage],
                source,
                style,
                stage_mask,
                scale,
                inverse_scale,
            )
            values = generator.ups[stage](values)
            if stage == generator.num_upsamples - 1:
                values = torch.cat([values[:, :, 1:2], values], dim=-1)
            values = (values + source) * stage_mask
            branches = [
                _masked_generator_block_rmsnorm(
                    generator.resblocks[stage * generator.num_kernels + branch],
                    self.rmsnorms[stage],
                    values,
                    style,
                    stage_mask,
                    scale,
                    inverse_scale,
                )
                for branch in range(generator.num_kernels)
            ]
            values = torch.stack(branches).sum(dim=0) / generator.num_kernels
            values = values * stage_mask
        values = F.leaky_relu(values) * mask_wave
        values = generator.conv_post(values)
        channels = generator.post_n_fft // 2 + 1
        magnitude = torch.exp(values[:, :channels]) * mask_wave
        phase = torch.sin(values[:, channels:]) * mask_wave
        return torch.cat([magnitude, phase], dim=1)


def _value_contract(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor = value.type.tensor_type
    dimensions: list[int | str | None] = []
    for dimension in tensor.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return {
        "name": value.name,
        "dtype": TensorProto.DataType.Name(tensor.elem_type),
        "shape": dimensions,
    }


def artifact_metadata(artifact: GraphArtifact, root: Path) -> dict[str, Any]:
    metadata = {
        "role": artifact.role,
        "onnx_file": str(artifact.path.relative_to(root)),
        "onnx_sha256": artifact.onnx_sha256,
        "inputs": list(artifact.input_contracts),
        "outputs": list(artifact.output_contracts),
        "node_count": artifact.node_count,
        "op_counts": artifact.op_counts,
        "structural_rewrites": artifact.structural_rewrites,
    }
    return metadata
