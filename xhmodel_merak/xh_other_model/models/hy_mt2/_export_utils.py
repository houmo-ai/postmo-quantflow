# Copyright 2025 HOUMO AI
#
# SPDX-License-Identifier: Apache-2.0

import copy
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors_file

from xhquant.api import get_root_logger


def xh1_hmonnx_compatible(input_names: list[str]) -> list[str]:
    """Map legacy LLM input names to the runtime HMONNX contract."""
    compatible_names = copy.deepcopy(input_names)
    direct_mapping = {
        "inputs_embeds": "input_1",
        "past_seq_length": "valid_length",
        "current_input_length": "current_length",
    }
    cache_patterns = (
        (re.compile(r"^past_key_cache_(\d+)$"), "model_layers_{}_self_attn_kcache_input"),
        (re.compile(r"^past_value_cache_(\d+)$"), "model_layers_{}_self_attn_vcache_input"),
    )
    for index, name in enumerate(compatible_names):
        if name in direct_mapping:
            compatible_names[index] = direct_mapping[name]
            continue
        for pattern, template in cache_patterns:
            match = pattern.match(name)
            if match is not None:
                compatible_names[index] = template.format(match.group(1))
                break
    return compatible_names


def load_quant_weight(quant_weight_path: str, native_hf_model: nn.Module) -> bool:
    """Load an xhquant checkpoint without depending on legacy model-zoo code."""
    checkpoint = Path(quant_weight_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"quant weight checkpoint does not exist: {checkpoint}")

    logger = get_root_logger()
    logger.info("Load previously saved checkpoint from: %s", checkpoint)
    if checkpoint.suffix == ".safetensors":
        state_dict: dict[str, Any] = load_safetensors_file(str(checkpoint), device="cpu")
    else:
        state_dict = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(f"quant weight checkpoint must contain a state dict, got {type(state_dict)}")

    model_state_keys = set(native_hf_model.state_dict())
    for name in list(state_dict):
        if name in model_state_keys:
            continue
        if name.endswith(".quant_weight"):
            module_name, _, _ = name.rpartition(".")
            module = native_hf_model.get_submodule(module_name)
            value = state_dict.pop(name)
            minimum = value.min().item()
            maximum = value.max().item()
            if minimum >= -(2**7) and maximum <= 2**7 - 1:
                value = value.to(torch.int8)
            elif minimum >= -(2**15) and maximum <= 2**15 - 1:
                value = value.to(torch.int16)
            else:
                value = value.to(torch.float32)
            module.register_buffer("quant_weight", value, persistent=False)
            logger.debug("add quant_weight to %s", module_name)
            continue
        logger.warning("ignore unexpected quant checkpoint entry: %s", name)
        state_dict.pop(name)

    native_hf_model.load_state_dict(state_dict)
    return True
