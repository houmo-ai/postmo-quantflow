"""Consume GPTQModel MTP payloads and reuse the packed main head."""
from __future__ import annotations

import copy

import torch


def load_mtp_packed_experts(source):
    from gptqmodel.quantization.qwen35_mtp import (
        CheckpointTensors,
        MTPQuantizationConfig,
        inspect_mtp_checkpoint,
        load_packed_linear,
    )

    from ..._dequant_converter import gptqmodel_torch_qlinear_converter

    inspect_mtp_checkpoint(source, MTPQuantizationConfig())
    reader = CheckpointTensors(source)
    text = reader.config.get("text_config", reader.config)
    weights, codes = {}, {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        dense, signed = [], []
        ins, outs = text["hidden_size"], text["moe_intermediate_size"]
        if proj == "down_proj":
            ins, outs = outs, ins
        for expert in range(text["num_experts"]):
            module = load_packed_linear(reader, f"mtp.layers.0.mlp.experts.{expert}.{proj}",
                                        bits=4, group_size=64, in_features=ins, out_features=outs)
            gptqmodel_torch_qlinear_converter(module)
            dense.append(module.weight.detach())
            signed.append(module.quant_weight)
        weights[proj] = torch.stack(dense)
        codes[proj] = torch.stack(signed)
    return weights, codes


def load_mtp_shared_head(source):
    from ...shared_head import load_packed_head
    return load_packed_head(source)


def configure_packed_mtp_scheme(scheme, model, *, explicit):
    """Bind calibrated integer weights to matching SEFP nodes before quantization."""
    from xhquant.api import QuantScheme

    requirements = {}
    if getattr(model, "_mtp_block_bits", None):
        requirements.update({name: model._mtp_block_bits
                             for name, module in model.named_modules()
                             if name.startswith("layer.") and hasattr(module, "quant_weight")})
    if getattr(model, "_mtp_expert_bits", None):
        requirements["layer_mlp_moeblock"] = model._mtp_expert_bits
    if getattr(model, "_mtp_head_bits", None):
        requirements["lm_head"] = model._mtp_head_bits
    if not requirements:
        return scheme
    if isinstance(scheme, QuantScheme):
        raise ValueError(
            "Packed MTP requires a dictionary quant_scheme so calibrated node requirements can be validated"
        )
    result = copy.deepcopy(scheme)
    # Default draft head used an SSFP RTN schema; calibrated head uses the
    # same integer SEFP scheme as the main model.
    for name, bits in requirements.items():
        aliases = tuple(dict.fromkeys((name, name.replace(".", "_"), name.replace("_", ".")))) if name != "lm_head" else (name,)
        for section in ("nodes", "nodes_cfg"):
            entries = result.setdefault(section, {})
            for alias in aliases:
                previous = entries.get(alias)
                if explicit and previous:
                    expected = {"quant_type": f"w{bits}a8h1_sefp"}
                    # Parent export serializes/clones the default draft config.
                    # Its legacy SSFP head schema is still a W4 integer schema.
                    legacy_head = {"w_schema": {"bits": bits, "fp_mode": "ssfp", "hidden_bit": False}}
                    if previous != expected and not (name == "lm_head" and previous == legacy_head):
                        raise ValueError(f"MTP calibrated {name} requires {expected}")
                entries.pop(alias, None)
        for alias in aliases:
            result["nodes"][alias] = {"quant_type": f"w{bits}a8h1_sefp"}
    return result


def load_mtp_packed_dense(source):
    from gptqmodel.quantization.qwen35_mtp import CheckpointTensors, MTPQuantizationConfig, inspect_mtp_checkpoint, load_packed_linear, Qwen35MTP
    from gptqmodel.quantization.qwen35_mtp_dense import dense_linears
    from ..._dequant_converter import gptqmodel_torch_qlinear_converter
    reader = CheckpointTensors(source)
    bits = reader.config["mtp_quantization"]["block_bits"]
    inspect_mtp_checkpoint(source, MTPQuantizationConfig(block_bits=bits))
    with torch.device("meta"):
        template = Qwen35MTP(reader.config)
    result = {}
    for name, layer in dense_linears(template).items():
        module = load_packed_linear(reader, "mtp." + name, bits=bits, group_size=64,
                                   in_features=layer.in_features, out_features=layer.out_features)
        gptqmodel_torch_qlinear_converter(module)
        result[name.replace("layers.0.", "layer.")] = module
    return result, bits
