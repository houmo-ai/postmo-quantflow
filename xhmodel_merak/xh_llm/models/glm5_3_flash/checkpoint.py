# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Strict, bounded tensor loading for the original BF16/FP8 GLM checkpoint.

The release stores separate experts, convolution branches and flat mHC names;
Transformers 5.16 presents fused experts/conv and nested mHC/forget-gate modules.
This is checkpoint consumption, not a new quantization producer.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig


class Glm53Checkpoint:
    def __init__(self, model_dir):
        self.path = Path(model_dir)
        self.config = AutoConfig.from_pretrained(self.path)
        self.text_config = self.config.get_text_config()
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(self.path / "model.safetensors", framework="pt") as f:
                self.weight_map = dict.fromkeys(f.keys(), "model.safetensors")
        quant = getattr(self.config, "quantization_config", {}) or {}
        if quant.get("quant_method") not in (None, "fp8", "gptq"):
            raise ValueError("GLM source loader expects BF16/FP16, block FP8 or GPTQ weights")
        self.quantization_config = quant
        self.is_gptq = quant.get("quant_method") == "gptq"
        if self.is_gptq:
            from gptqmodel.recipes.glm5_3_flash import EXPERT_PATTERN

            if (quant.get("bits"), quant.get("group_size"), quant.get("sym")) != (8, 64, True) or quant.get(
                "desc_act", False
            ) is not False:
                raise ValueError("GLM GPTQ deployment requires symmetric W8 / routed W4 G64 with desc_act=False")
            if quant.get("dynamic", {}).get(EXPERT_PATTERN) != {"bits": 4, "group_size": 64, "sym": True}:
                raise ValueError("GLM GPTQ checkpoint is missing the routed W4 G64 override")
        self.block_size = tuple(quant.get("weight_block_size", (128, 128)))

    def raw(self, name):
        if name not in self.weight_map:
            raise KeyError(f"checkpoint is missing required tensor {name}")
        with safe_open(self.path / self.weight_map[name], framework="pt") as f:
            return f.get_tensor(name)

    def tensor(self, name, *, device="cpu", dtype=torch.float16):
        value = self.raw(name)
        if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            if value.ndim != 2:
                raise ValueError(f"FP8 tensor {name} must be a matrix")
            scale = self.raw(name + "_scale_inv").float()
            rows, cols = value.shape
            br, bc = self.block_size
            expected = ((rows + br - 1) // br, (cols + bc - 1) // bc)
            if tuple(scale.shape) != expected or not torch.isfinite(scale).all():
                raise ValueError(f"invalid FP8 scale layout for {name}: {tuple(scale.shape)} != {expected}")
            output = torch.empty(value.shape, device=device, dtype=dtype)
            # Bound the dequantization temporary to a block row.
            for row in range(0, rows, br):
                expanded = scale[row // br].repeat_interleave(bc)[:cols].to(device)
                output[row : row + br] = value[row : row + br].to(device=device, dtype=torch.float32) * expanded
            return output
        return value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)

    def _mapped_tensor(self, target, shape, device, dtype):
        # Prefer an already HF-compatible checkpoint when available.
        if target in self.weight_map:
            return self.tensor(target, device=device, dtype=dtype)
        name = re.sub(r"\.(attn|ffn)_hc\.(fn|base|scale)$", r".hc_\1_\2", target)
        name = name.replace(".self_attn.forget_gate.", ".self_attn.")
        if name in self.weight_map:
            return self.tensor(name, device=device, dtype=dtype)
        if target.endswith(".self_attn.conv1d.weight"):
            prefix = target.removesuffix("conv1d.weight")
            return torch.cat(
                [
                    self.tensor(prefix + branch + "_conv1d.weight", device=device, dtype=dtype)
                    for branch in ("q", "k", "v")
                ],
                dim=0,
            )
        if target.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            prefix, projection = target.rsplit(".", 1)
            output = torch.empty(shape, device=device, dtype=dtype)
            for expert in range(shape[0]):
                if projection == "gate_up_proj":
                    width = shape[1] // 2
                    for offset, branch in ((0, "gate_proj"), (width, "up_proj")):
                        output[expert, offset : offset + width] = self.tensor(
                            f"{prefix}.{expert}.{branch}.weight", device=device, dtype=dtype
                        )
                else:
                    output[expert] = self.tensor(f"{prefix}.{expert}.down_proj.weight", device=device, dtype=dtype)
            return output
        raise KeyError(f"no released checkpoint mapping for {target}")

    def _packed_linear(self, linear, prefix, *, device):
        from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
        from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module
        from ..._dequant_converter import gptqmodel_torch_qlinear_packed_converter

        bits = 4 if re.search(r"\.mlp\.experts\.\d+\.", prefix) else 8
        with torch.device("meta"):
            quantized = TorchQuantLinear(
                bits=bits,
                group_size=64,
                sym=True,
                desc_act=False,
                in_features=linear.in_features,
                out_features=linear.out_features,
                bias=linear.bias is not None,
            )
        for name in ("qweight", "qzeros", "scales", "g_idx"):
            value = self.raw(prefix + "." + name).to(device)
            expected = getattr(quantized, name)
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"Invalid packed GPTQ tensor {prefix}.{name}: {value.shape}, {value.dtype}")
            # Preserve signed/zero checkpoint scales; the SDK folds signs when
            # converting packed weights into the target quantization format.
            if name == "scales" and not torch.isfinite(value).all():
                raise ValueError(f"Invalid GPTQ scales: {prefix}.{name}")
            quantized._buffers[name] = value
        if linear.bias is not None:
            quantized._buffers["bias"] = self.tensor(prefix + ".bias", device=device)
        checkpoint_format = self.quantization_config.get("checkpoint_format", "gptq")
        if checkpoint_format == "gptq":
            convert_gptq_v1_to_v2_format_module(quantized, bits=bits, pack_dtype=torch.int32)
        elif checkpoint_format == "gptq_v2":
            quantized.qzero_format(format=2)
        else:
            raise ValueError(f"Unsupported GLM GPTQ checkpoint format: {checkpoint_format}")
        return gptqmodel_torch_qlinear_packed_converter(quantized)

    def load_module(self, module, prefix, *, device="cpu", dtype=torch.float16, skip_prefixes=()):
        packed_names = []
        if self.is_gptq:
            from defuser.modeling.moe_experts_interface import prepare_model_for_moe_quantization

            # Only metadata is split here. Routed weights stay packed throughout
            # loading, placeholder export and SDK ownership transfer.
            prepare_model_for_moe_quantization(module)
            packed_names = [
                name
                for name, child in module.named_modules()
                if isinstance(child, nn.Linear)
                and prefix + name + ".qweight" in self.weight_map
                and not any(name == skip or name.startswith(skip + ".") for skip in skip_prefixes)
            ]
        packed_set = set(packed_names)
        for name, expected in list(module.state_dict().items()):
            if any(name == skip or name.startswith(skip + ".") for skip in skip_prefixes):
                continue
            if name.rpartition(".")[0] in packed_set:
                continue
            value = self._mapped_tensor(prefix + name, tuple(expected.shape), device, dtype)
            if value.shape != expected.shape:
                raise ValueError(f"{prefix + name}: checkpoint {tuple(value.shape)} != model {tuple(expected.shape)}")
            parent_name, _, leaf = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            if leaf in parent._parameters:
                setattr(parent, leaf, nn.Parameter(value, requires_grad=False))
            else:
                parent._buffers[leaf] = value
        for name in packed_names:
            child = module.get_submodule(name)
            module.set_submodule(name, self._packed_linear(child, prefix + name, device=device))
        return module.eval()

    def load_layer(self, index, *, device="cpu", dtype=torch.float16):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer

        if not 0 <= index < self.text_config.num_hidden_layers:
            raise ValueError("layer index outside checkpoint")
        with torch.device("meta"):
            layer = Glm5NextTextDecoderLayer(self.text_config, index)
        return self.load_module(layer, f"model.language_model.layers.{index}.", device=device, dtype=dtype)

    def load_text_model(self, *, max_layers=None, device="cpu", dtype=torch.float16):
        import copy

        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

        config = copy.deepcopy(self.text_config)
        if max_layers is not None:
            config.num_hidden_layers = int(max_layers)
            config.layer_types = config.layer_types[:max_layers]
            config.mlp_layer_types = config.mlp_layer_types[:max_layers]
            config.indexer_types = config.indexer_types[:max_layers]
        with torch.device("meta"):
            model = Glm5NextTextModel(config)
            head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.load_module(model, "model.language_model.", device=device, dtype=dtype)
        self.load_module(head, "lm_head.", device=device, dtype=dtype)
        return model, head


def empty_glm53_source(model_dir, max_layers=None):
    """Meta-only text shell used by streamed export workers."""
    import copy

    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    config = copy.deepcopy(AutoConfig.from_pretrained(model_dir))
    text = config.get_text_config()
    if max_layers is not None:
        text.num_hidden_layers = int(max_layers)
        text.layer_types = text.layer_types[:max_layers]
        text.mlp_layer_types = text.mlp_layer_types[:max_layers]
        text.indexer_types = text.indexer_types[:max_layers]
    with torch.device("meta"):
        source = nn.Module()
        source.config = config
        source.language_model = Glm5NextTextModel(text).half()
        source.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False).half()
    return source
