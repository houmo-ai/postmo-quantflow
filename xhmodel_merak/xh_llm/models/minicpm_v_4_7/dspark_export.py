"""MiniCPM DSpark export adapter.

The Qwen DFlash adapter only emits logits and only accepts an anchor for
DFlash2. DSpark keeps the full query block, consumes the anchor before the
mask, and exports token ids plus optional confidence logits.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def _build_dflash_export_adapter(
    core_model: nn.Module,
    *,
    mode: str,
    num_hidden_layers: int,
    use_flash_attention: bool = False,
    is_dflash2: bool = False,
    is_dspark: bool = False,
) -> nn.Module:
    if is_dspark and is_dflash2:
        raise ValueError("DSpark export cannot also be DFlash2")
    if is_dflash2:
        use_flash_attention = False
        prepare_for_export = getattr(core_model, "prepare_for_export", None)
        if not callable(prepare_for_export):
            raise TypeError("DFlash2 export core must implement prepare_for_export()")
        prepare_for_export()

    cache_names = [
        *[f"past_key_cache_{idx}" for idx in range(num_hidden_layers)],
        *[f"past_value_cache_{idx}" for idx in range(num_hidden_layers)],
    ]
    if mode == "context":
        arg_names = ["target_hidden", "past_seq_length", "current_input_length", *cache_names]
        call_args = list(arg_names)
        core_call = "self.core.forward_context"
    elif mode == "decode" and is_dspark:
        # Alignment traces this exact order: anchor precedes the attention mask.
        arg_names = [
            "noise_embedding",
            "past_seq_length",
            "current_input_length",
            "anchor_token_id",
            "attn_mask",
            *cache_names,
        ]
        call_args = [
            "noise_embedding",
            "past_seq_length",
            "current_input_length",
            "attn_mask",
            "anchor_token_id",
            *cache_names,
        ]
        core_call = "self.core.forward_decode"
    elif mode == "decode":
        arg_names = [
            "noise_embedding",
            "past_seq_length",
            "current_input_length",
            *(["cache_position"] if is_dflash2 else []),
            *(["anchor_token_id"] if is_dflash2 else []),
            *([] if use_flash_attention else ["attn_mask"]),
            *cache_names,
        ]
        call_args = [
            "noise_embedding",
            "past_seq_length",
            "current_input_length",
            ("None" if use_flash_attention else "attn_mask"),
            ("anchor_token_id" if is_dflash2 else "None"),
            *cache_names,
            *(["cache_position=cache_position"] if is_dflash2 else []),
        ]
        core_call = "self.core.forward_decode"
    else:
        raise ValueError(f"Unsupported DFlash mode: {mode}")

    forward_src = f"def forward(self, {', '.join(arg_names)}):\n    return {core_call}({', '.join(call_args)})\n"
    namespace: dict[str, Any] = {}
    exec(forward_src, {}, namespace)

    class _DSparkExportAdapter(nn.Module):
        def __init__(self, core: nn.Module):
            super().__init__()
            self.core = core

    _DSparkExportAdapter.forward = namespace["forward"]
    return _DSparkExportAdapter(core_model)
