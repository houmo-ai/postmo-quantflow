"""Host-side MiniMax-H3 text/vision leftovers for HMONNX installs.

The exported text_encoder HMONNX graph starts from precomputed ``hidden_states``.
Prompt encoding still needs:

- ``language_model.embed_tokens`` for ``input_ids -> inputs_embeds``
- ``language_model.rotary_emb`` rebuilt from ``text_config`` (no persistent weights)
- ``visual.pos_embed`` / ``visual.rotary_pos_emb`` for FL2VA vision geometry

These helpers export those tiny host artifacts and rebuild a stripped Qwen3-VL
shell so ``hmonnx_demo`` does not have to ``load_components(text_encoder)``.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionRotaryEmbedding,
)


TEXT_EMBEDDING_FILENAME = "text_embeddings.safetensors"
VISION_GEOMETRY_FILENAME = "vision_geometry.safetensors"

_EMBED_TOKEN_KEY = "embed_tokens.weight"
_VISION_POS_EMBED_KEY = "pos_embed.weight"


def text_embedding_path(export_dir: str | Path) -> Path:
    return Path(export_dir).resolve() / "text_encoder" / TEXT_EMBEDDING_FILENAME


def vision_geometry_path(export_dir: str | Path) -> Path:
    return Path(export_dir).resolve() / "vision_encoder" / VISION_GEOMETRY_FILENAME


def _tensor_meta(weight: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(weight.shape),
        "dtype": str(weight.dtype).removeprefix("torch."),
    }


def export_text_embedding_weights(language_model: nn.Module, export_dir: str | Path) -> dict[str, Any]:
    """Save ``embed_tokens.weight`` next to the text_encoder HMONNX graph."""
    embed_tokens = getattr(language_model, "embed_tokens", None)
    weight = getattr(embed_tokens, "weight", None)
    if not torch.is_tensor(weight):
        raise AttributeError("MiniMax-H3 language_model is missing embed_tokens.weight")

    path = text_embedding_path(export_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_weight = weight.detach().to(device="cpu").contiguous()
    save_file({_EMBED_TOKEN_KEY: cpu_weight}, str(path))
    return {
        "text_embeddings_file": str(path.relative_to(Path(export_dir).resolve())),
        "text_embeddings": _tensor_meta(cpu_weight),
    }


def export_vision_geometry_weights(visual: nn.Module, export_dir: str | Path) -> dict[str, Any]:
    """Save vision ``pos_embed`` used to rebuild padded geometry on the host."""
    pos_embed = getattr(visual, "pos_embed", None)
    weight = getattr(pos_embed, "weight", None)
    if not torch.is_tensor(weight):
        raise AttributeError("MiniMax-H3 visual is missing pos_embed.weight")

    path = vision_geometry_path(export_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_weight = weight.detach().to(device="cpu").contiguous()
    save_file({_VISION_POS_EMBED_KEY: cpu_weight}, str(path))
    return {
        "vision_geometry_file": str(path.relative_to(Path(export_dir).resolve())),
        "vision_geometry": {
            "pos_embed": _tensor_meta(cpu_weight),
        },
    }


def export_host_embedding_artifacts_from_text_encoder_dir(
    text_encoder_dir: str | Path,
    export_dir: str | Path,
    *,
    export_vision_geometry: bool = True,
) -> dict[str, Any]:
    """Backfill host leftovers from a Qwen3-VL ``text_encoder`` checkpoint directory."""
    from safetensors import safe_open

    text_encoder_dir = Path(text_encoder_dir).expanduser().resolve()
    export_dir = Path(export_dir).expanduser().resolve()
    index_path = text_encoder_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing text encoder index: {index_path}")

    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
    embed_key = "model.language_model.embed_tokens.weight"
    pos_key = "model.visual.pos_embed.weight"
    embed_shard = weight_map.get(embed_key)
    if not embed_shard:
        raise KeyError(f"Missing {embed_key} in {index_path}")

    result: dict[str, Any] = {}
    with safe_open(text_encoder_dir / str(embed_shard), framework="pt") as handle:
        embed_weight = handle.get_tensor(embed_key)
    path = text_embedding_path(export_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_weight = embed_weight.detach().to(device="cpu").contiguous()
    save_file({_EMBED_TOKEN_KEY: cpu_weight}, str(path))
    result.update(
        {
            "text_embeddings_file": str(path.relative_to(export_dir)),
            "text_embeddings": _tensor_meta(cpu_weight),
        }
    )

    if export_vision_geometry:
        pos_shard = weight_map.get(pos_key)
        if not pos_shard:
            raise KeyError(f"Missing {pos_key} in {index_path}")
        with safe_open(text_encoder_dir / str(pos_shard), framework="pt") as handle:
            pos_weight = handle.get_tensor(pos_key)
        vision_path = vision_geometry_path(export_dir)
        vision_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_pos = pos_weight.detach().to(device="cpu").contiguous()
        save_file({_VISION_POS_EMBED_KEY: cpu_pos}, str(vision_path))
        result.update(
            {
                "vision_geometry_file": str(vision_path.relative_to(export_dir)),
                "vision_geometry": {"pos_embed": _tensor_meta(cpu_pos)},
            }
        )
    return result


def _materialize_embedding(module: nn.Embedding, weight: torch.Tensor, *, dtype: torch.dtype) -> None:
    # Empty meta tensors may keep the config default dtype (bf16/fp32). Replace the
    # Parameter so host leftovers match the HMONNX demo dtype instead of copying.
    materialized = weight.detach().to(device="cpu", dtype=dtype).contiguous()
    module.weight = nn.Parameter(materialized)


def _strip_language_model_body(language_model: nn.Module, text_config: Any) -> None:
    language_model.layers = nn.ModuleList()
    if hasattr(language_model, "norm"):
        language_model.norm = nn.Identity()
    language_model.rotary_emb = Qwen3VLTextRotaryEmbedding(text_config)


def _strip_vision_model_body(visual: nn.Module, vision_config: Any) -> None:
    visual.blocks = nn.ModuleList()
    visual.deepstack_merger_list = nn.ModuleList()
    visual.merger = nn.Identity()
    visual.patch_embed = nn.Identity()
    head_dim = int(vision_config.hidden_size) // int(vision_config.num_heads)
    visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)


def build_hmonnx_text_encoder_host(
    model_dir: str | Path,
    export_dir: str | Path,
    *,
    dtype: torch.dtype = torch.float16,
    require_vision_geometry: bool = True,
) -> Qwen3VLForConditionalGeneration:
    """Build a stripped Qwen3-VL host that only keeps TE/VE leftovers for HMONNX."""
    model_dir = Path(model_dir).expanduser().resolve()
    export_dir = Path(export_dir).expanduser().resolve()
    embed_path = text_embedding_path(export_dir)
    if not embed_path.is_file():
        raise FileNotFoundError(
            f"Missing exported MiniMax-H3 text embeddings: {embed_path}. "
            "Re-export text_encoder or backfill with export_text_embedding_weights()."
        )

    vision_path = vision_geometry_path(export_dir)
    if require_vision_geometry and not vision_path.is_file():
        raise FileNotFoundError(
            f"Missing exported MiniMax-H3 vision geometry: {vision_path}. "
            "Re-export vision_encoder or backfill with export_vision_geometry_weights()."
        )

    config = Qwen3VLConfig.from_pretrained(model_dir / "text_encoder")
    with init_empty_weights():
        text_encoder = Qwen3VLForConditionalGeneration(config)

    language_model = text_encoder.model.language_model
    visual = text_encoder.model.visual
    _strip_language_model_body(language_model, config.text_config)
    _strip_vision_model_body(visual, config.vision_config)

    embed_weight = load_file(str(embed_path), device="cpu")[_EMBED_TOKEN_KEY]
    expected_shape = (int(config.text_config.vocab_size), int(config.text_config.hidden_size))
    if tuple(embed_weight.shape) != expected_shape:
        raise ValueError(
            f"Exported text embeddings have shape {tuple(embed_weight.shape)}, expected {expected_shape}"
        )
    _materialize_embedding(language_model.embed_tokens, embed_weight, dtype=dtype)

    if vision_path.is_file():
        pos_weight = load_file(str(vision_path), device="cpu")[_VISION_POS_EMBED_KEY]
        expected_pos = (int(config.vision_config.num_position_embeddings), int(config.vision_config.hidden_size))
        if tuple(pos_weight.shape) != expected_pos:
            raise ValueError(
                f"Exported vision pos_embed has shape {tuple(pos_weight.shape)}, expected {expected_pos}"
            )
        _materialize_embedding(visual.pos_embed, pos_weight, dtype=dtype)
    else:
        # Text-only installs can drop vision geometry entirely.
        visual.pos_embed = nn.Embedding(1, int(config.vision_config.hidden_size), dtype=dtype)

    # Drop LM head / unused top-level weights; MiniMax-H3 only reads hidden_states[50].
    if hasattr(text_encoder, "lm_head"):
        text_encoder.lm_head = nn.Identity()

    text_encoder.eval()
    gc.collect()
    return text_encoder


def install_hmonnx_text_encoder_host(
    pipe: Any,
    model_dir: str | Path,
    export_dir: str | Path,
    *,
    dtype: torch.dtype = torch.float16,
    require_vision_geometry: bool = True,
) -> Qwen3VLForConditionalGeneration:
    """Attach the lightweight TE host onto a modular MiniMax-H3 pipeline."""
    text_encoder = build_hmonnx_text_encoder_host(
        model_dir,
        export_dir,
        dtype=dtype,
        require_vision_geometry=require_vision_geometry,
    )
    pipe.text_encoder = text_encoder
    if "text_encoder" in getattr(pipe, "_modules", {}):
        pipe._modules["text_encoder"] = text_encoder
    return text_encoder


__all__ = [
    "TEXT_EMBEDDING_FILENAME",
    "VISION_GEOMETRY_FILENAME",
    "build_hmonnx_text_encoder_host",
    "export_host_embedding_artifacts_from_text_encoder_dir",
    "export_text_embedding_weights",
    "export_vision_geometry_weights",
    "install_hmonnx_text_encoder_host",
    "text_embedding_path",
    "vision_geometry_path",
]
