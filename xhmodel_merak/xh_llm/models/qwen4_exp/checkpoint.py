"""Qwen4 checkpoint loading and ownership-preserving packed-weight adaptation."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .modeling import extract_qwen4_exp_text_model


def _build_qwen4_exp_gptq_export_device_map(config: Any) -> str | dict[str, str]:
    """Keep the oversized PLE table on CPU while sharding decoder layers.

    Qwen4-Exp's ngram embedding is roughly 95 GiB in FP16 and cannot fit on a
    single RTX 6000D.  GPTQModel's generic ``device_map="auto"`` assigns whole
    decoder layers round-robin, so the layer containing PLE otherwise OOMs.
    The native PLE implementation supports a CPU-resident table and transfers
    only the selected rows back to the decoder device.
    """

    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        return "auto"
    text_config = getattr(config, "text_config", config)
    num_hidden_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)
    if num_hidden_layers <= 0:
        return "auto"

    device_count = torch.cuda.device_count()
    last_device = f"cuda:{device_count - 1}"
    device_map: dict[str, str] = {"": "cpu"}
    device_map["model.language_model.embed_tokens"] = "cuda:0"
    ple_layer_indices = {
        int(ple_layer_id) - 1
        for ple_layer_id in (getattr(text_config, "ple_layer_ids", ()) or ())
        if 0 <= int(ple_layer_id) - 1 < num_hidden_layers
    }
    for layer_idx in range(num_hidden_layers):
        layer_device = "cpu" if layer_idx in ple_layer_indices else f"cuda:{layer_idx % device_count}"
        device_map[f"model.language_model.layers.{layer_idx}"] = layer_device
    device_map["model.language_model.hyper_connection_mixer"] = last_device
    device_map["model.language_model.rotary_emb"] = last_device
    device_map["model.visual"] = last_device
    device_map["lm_head"] = last_device
    return device_map

def _restore_qwen4_exp_ple_layer_devices(model: torch.nn.Module, config: Any) -> None:
    """Move PLE layers back to their GPU while leaving the huge table on CPU."""

    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        return
    text_config = getattr(config, "text_config", config)
    text_model = extract_qwen4_exp_text_model(model)
    for ple_layer_id in getattr(text_config, "ple_layer_ids", ()) or ():
        layer_idx = int(ple_layer_id) - 1
        if not 0 <= layer_idx < len(text_model.layers):
            continue
        layer = text_model.layers[layer_idx]
        ple = getattr(layer, "ple", None)
        ngram_owner = getattr(ple, "ple_embedding", None)
        ngram_embedding = getattr(ngram_owner, "ngram_embedding", None)
        if ngram_embedding is None:
            continue
        target_device = torch.device(f"cuda:{layer_idx % torch.cuda.device_count()}")
        ngram_owner.ngram_embedding = None
        try:
            layer.to(target_device)
        finally:
            ngram_owner.ngram_embedding = ngram_embedding

def load_qwen4_exp_model_for_export(
    model_dir: str | Path,
    *,
    device: str = "cpu",
    device_map: str | Mapping[str, Any] | None = None,
    dtype: torch.dtype = torch.float16,
) -> torch.nn.Module:
    """Load either a GPTQModel HF artifact or the original Qwen4-Exp model."""

    from transformers import AutoConfig

    model_dir = str(Path(model_dir).expanduser().resolve())
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    quantization_config = getattr(config, "quantization_config", None)
    if isinstance(quantization_config, Mapping):
        quant_method = quantization_config.get("quant_method")
    else:
        quant_method = getattr(quantization_config, "quant_method", None)
    quant_method = str(getattr(quant_method, "value", quant_method) or "").lower()
    if quant_method == "gptq":
        from ...base_model import XHBaseModel

        resolved_device_map = device_map or device
        if device_map == "auto" and str(getattr(config, "model_type", "")) == "qwen4_exp":
            resolved_device_map = _build_qwen4_exp_gptq_export_device_map(config)
        model = XHBaseModel._load_gptqmodel(
            model_dir,
            device_map=resolved_device_map,
            dtype=dtype,
            trust_remote_code=False,
            offload_to_disk=False,
        ).eval()
        if isinstance(resolved_device_map, Mapping):
            _restore_qwen4_exp_ple_layer_devices(model, config)
        return model

    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    auto_model = (AutoModelForCausalLM if config.model_type == "qwen4_exp_text"
                  else AutoModelForImageTextToText)
    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": False,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    model = auto_model.from_pretrained(model_dir, **load_kwargs).eval()
    if device_map is None:
        model.to(device)
    return model

def _adapt_gptq_linears(model: torch.nn.Module) -> tuple[str, ...]:
    if not any(type(module).__module__.startswith("gptqmodel.") for module in model.modules()):
        return ()
    try:
        from gptqmodel.nn_modules.qlinear import PackableQuantLinear
    except ImportError:
        return ()
    if not any(isinstance(module, PackableQuantLinear) for module in model.modules()):
        return ()
    from ..._dequant_converter import replace_gptqmodel_quant_linears_with_packed

    return replace_gptqmodel_quant_linears_with_packed(model)
