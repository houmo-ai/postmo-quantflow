"""Encoder component: official-shaped Whisper + time merge + VQ adaptor."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .processing import AUDIO_MERGE_SIZE

ENCODER_MEL_BINS = 80
ENCODER_MEL_LENGTH = 3000
ENCODER_WHISPER_FRAMES = 1500
ENCODER_AUDIO_TOKENS = 375
LLM_HIDDEN_SIZE = 1024


def time_merge(features: torch.Tensor, merge_size: int = AUDIO_MERGE_SIZE) -> torch.Tensor:
    batch, time, channels = features.shape
    remainder = time % merge_size
    if remainder:
        features = features[:, : time - remainder]
        time = features.shape[1]
    return features.reshape(batch, time // merge_size, merge_size * channels)


def slice_encoder_window(audio_embeds: torch.Tensor, token_len: int) -> torch.Tensor:
    """Host-side valid-length slice; the encoder graph always emits a full window."""
    return audio_embeds[:, : int(token_len)]


class MossWhisperAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        if n_state % n_head != 0:
            raise ValueError(f"n_state {n_state} must be divisible by n_head {n_head}")
        self.n_head = n_head
        self.head_dim = n_state // n_head
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(n_state, n_state)
        self.k_proj = nn.Linear(n_state, n_state, bias=False)
        self.v_proj = nn.Linear(n_state, n_state)
        self.out_proj = nn.Linear(n_state, n_state)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, time, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        context = torch.matmul(torch.softmax(scores, dim=-1), value)
        context = context.transpose(1, 2).contiguous().view(batch, time, -1)
        return self.out_proj(context)


class MossWhisperEncoderLayer(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.self_attn = MossWhisperAttention(n_state, n_head)
        self.self_attn_layer_norm = nn.LayerNorm(n_state)
        self.fc1 = nn.Linear(n_state, n_state * 4)
        self.activation_fn = nn.GELU()
        self.fc2 = nn.Linear(n_state * 4, n_state)
        self.final_layer_norm = nn.LayerNorm(n_state)

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        residual = hidden_states
        hidden_states = residual + self.self_attn(self.self_attn_layer_norm(hidden_states))
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(self.final_layer_norm(hidden_states)))
        return residual + self.fc2(hidden_states)


class MossWhisperEncoder(nn.Module):
    """Official-shaped Whisper encoder: log-mel window in, 1500 frames out."""

    def __init__(self, n_mels: int, n_state: int, n_head: int, n_layer: int, n_ctx: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(n_ctx, n_state)
        self.layers = nn.ModuleList(MossWhisperEncoderLayer(n_state, n_head) for _ in range(n_layer))
        self.layer_norm = nn.LayerNorm(n_state)

    def forward(self, input_features: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        hidden = torch.nn.functional.gelu(self.conv1(input_features))
        hidden = torch.nn.functional.gelu(self.conv2(hidden))
        hidden = hidden.permute(0, 2, 1)
        hidden = hidden + self.embed_positions.weight[: hidden.shape[1]]
        for layer in self.layers:
            hidden = layer(hidden)
        return self.layer_norm(hidden)


class MossVQAdaptor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size, eps=1e-6),
        )

    def forward(self, merged: torch.Tensor) -> torch.Tensor:
        return self.layers(merged)


def encoder_hidden_states(hidden: Any) -> torch.Tensor:
    """HF WhisperEncoder returns BaseModelOutput; reconstructed modules return a tensor."""
    last = getattr(hidden, "last_hidden_state", None)
    if last is not None:
        return last
    if isinstance(hidden, (tuple, list)):
        return hidden[0]
    return hidden


class EncoderComponent(nn.Module):
    """Published encoder graph wrapping official-named Whisper and adaptor modules."""

    def __init__(self, whisper_encoder: nn.Module, vq_adaptor: nn.Module, merge_size: int = AUDIO_MERGE_SIZE):
        super().__init__()
        self.whisper_encoder = whisper_encoder
        self.vq_adaptor = vq_adaptor
        self.merge_size = int(merge_size)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden = encoder_hidden_states(self.whisper_encoder(input_features))
        return self.vq_adaptor(time_merge(hidden, self.merge_size))


def build_encoder_component(
    *,
    n_state: int,
    n_head: int,
    n_layer: int,
    n_mels: int = ENCODER_MEL_BINS,
    n_ctx: int = ENCODER_WHISPER_FRAMES,
    hidden_size: int = LLM_HIDDEN_SIZE,
    merge_size: int = AUDIO_MERGE_SIZE,
) -> EncoderComponent:
    whisper = MossWhisperEncoder(n_mels, n_state, n_head, n_layer, n_ctx)
    adaptor = MossVQAdaptor(n_state * merge_size, hidden_size)
    return EncoderComponent(whisper, adaptor, merge_size).eval()


def build_tiny_encoder_component() -> EncoderComponent:
    """CI-only one-layer encoder for window I/O. Not an export fallback."""
    return build_encoder_component(n_state=32, n_head=2, n_layer=1)


def encoder_parts_from_hf(hf_model: Any) -> tuple[nn.Module, nn.Module]:
    root = hf_model.model if hasattr(hf_model, "model") else hf_model
    if not hasattr(root, "whisper_encoder") or not hasattr(root, "vq_adaptor"):
        raise TypeError("MOSS upstream model must expose model.whisper_encoder and model.vq_adaptor")
    return root.whisper_encoder, root.vq_adaptor


def encoder_component_from_hf(hf_model: Any) -> EncoderComponent:
    whisper, adaptor = encoder_parts_from_hf(hf_model)
    return EncoderComponent(whisper, adaptor).eval()


def load_encoder_component(model_dir: str | Path) -> EncoderComponent:
    """Load official remote-code modules from the user checkpoint. Missing checkpoint fails."""
    from transformers import AutoModel

    root = Path(model_dir).resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"MOSS encoder export requires a checkpoint with config.json: {root}")
    hf_model = AutoModel.from_pretrained(str(root), trust_remote_code=True, torch_dtype=torch.float32)
    return encoder_component_from_hf(hf_model)


def _encode_graph_stem(model_name: str) -> str:
    name = str(model_name).strip().lower()
    if not name:
        raise ValueError("encode export requires model_name")
    if not name.startswith("hmquant_"):
        name = f"hmquant_{name}"
    return name


def _publish_llm_style_hmonnx(hmonnx_file: Path) -> Path:
    """Match decode layout: {stem}_with_act.onnx next to {stem}_external_data."""
    if hmonnx_file.stem.endswith("_with_act"):
        return hmonnx_file
    published = hmonnx_file.with_name(f"{hmonnx_file.stem}_with_act.onnx")
    if published.exists():
        published.unlink()
    hmonnx_file.replace(published)
    return published


def export_encoder_component(
    encoder: nn.Module,
    output_dir: str | Path,
    *,
    target_device: str = "XH2a",
    quant_type: str = "w8a8_sefp",
    convert_hmonnx: bool = False,
    model_name: str = "moss_transcribe_diarize_encode",
) -> dict[str, Any]:
    """Export one Audio window encoder via ONNX PTQ.

    Float ONNX is a conversion input only. Published HMONNX uses the LLM phase
    layout: ``hmquant_{model_name}_with_act.onnx`` next to ``*_external_data``.
    CI can keep ``convert_hmonnx=False`` to inspect the ONNX ABI.
    """
    if target_device != "XH2a":
        raise ValueError(f"First encoder recipe targets XH2a only, got {target_device}")
    if quant_type != "w8a8_sefp":
        raise ValueError(f"First encoder recipe uses w8a8_sefp Export PTQ, got {quant_type}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_stem = _encode_graph_stem(model_name)
    dummy = torch.zeros(1, ENCODER_MEL_BINS, ENCODER_MEL_LENGTH, dtype=torch.float32)
    encoder = encoder.to(dtype=torch.float32).eval()
    meta: dict[str, Any] = {
        "quant_type": quant_type,
        "target_device": target_device,
        "input_names": ("input_features",),
        "output_names": ("audio_embeds",),
    }
    with tempfile.TemporaryDirectory(prefix="moss_encode_onnx_") as tmp:
        onnx_file = Path(tmp) / "encoder_fp32.onnx"
        torch.onnx.export(
            encoder,
            dummy,
            str(onnx_file),
            input_names=["input_features"],
            output_names=["audio_embeds"],
            opset_version=18,
        )
        if convert_hmonnx:
            hmonnx_file = output_dir / f"{graph_stem}.onnx"
            _convert_encoder_hmonnx(onnx_file, dummy, hmonnx_file, target_device, quant_type)
            published = _publish_llm_style_hmonnx(hmonnx_file)
            meta["hmonnx_file"] = published.name
        else:
            onnx_name = f"{graph_stem}.onnx"
            shutil.copy2(onnx_file, output_dir / onnx_name)
            meta["onnx_file"] = onnx_name
    return meta


def _convert_encoder_hmonnx(
    onnx_file: Path,
    dummy: torch.Tensor,
    hmonnx_file: Path,
    target_device: str,
    quant_type: str,
) -> None:
    from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config

    try:
        device_type = getattr(DeviceType, target_device)
    except AttributeError as error:
        raise ValueError(f"unsupported xhquant target device: {target_device}") from error
    hmonnx_file.parent.mkdir(parents=True, exist_ok=True)
    convert_onnx_to_hmonnx(
        str(onnx_file),
        [dummy],
        device_type,
        str(hmonnx_file),
        quant_config=create_quant_config(QuantScheme(target_device=device_type, quant_type=quant_type)),
        input_names=["input_features"],
        output_names=["audio_embeds"],
    )


# Compatibility names used by remaining tests during the swap.
__all__ = [
    "ENCODER_AUDIO_TOKENS",
    "ENCODER_MEL_BINS",
    "ENCODER_MEL_LENGTH",
    "ENCODER_WHISPER_FRAMES",
    "LLM_HIDDEN_SIZE",
    "EncoderComponent",
    "MossVQAdaptor",
    "MossWhisperEncoder",
    "build_encoder_component",
    "build_tiny_encoder_component",
    "encoder_component_from_hf",
    "encoder_hidden_states",
    "export_encoder_component",
    "load_encoder_component",
    "slice_encoder_window",
    "time_merge",
]
