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
    from gptqmodel.quantization.config import FORMAT, QuantizeConfig
    from gptqmodel.quantization.qwen35_mtp import CheckpointTensors, load_packed_linear

    from ..._dequant_converter import gptqmodel_torch_qlinear_converter

    reader = CheckpointTensors(source)
    config = QuantizeConfig.from_quant_config(reader.config["quantization_config"])
    if config.format not in (FORMAT.GPTQ, FORMAT.GPTQ_V2) or config.rotation:
        raise ValueError("MTP shared head requires unrotated GPTQ/GPTQ V2 weights")
    if not config.lm_head or config.dynamic_get("lm_head") is False:
        raise ValueError("Packed MTP shared head conflicts with main head metadata")
    values = {key: config.dynamic_get("lm_head", key, getattr(config, key))
              for key in ("bits", "group_size", "sym", "desc_act")}
    if not values.pop("sym") or values.pop("desc_act"):
        raise ValueError("MTP shared head requires symmetric quantization without act-order")
    text = reader.config.get("text_config", reader.config)
    module = load_packed_linear(reader, "lm_head", **values, in_features=text["hidden_size"],
                                out_features=text["vocab_size"], qzero_format=1 if config.format == FORMAT.GPTQ else 2)
    bits = values["bits"]
    gptqmodel_torch_qlinear_converter(module)
    return module, bits


def configure_packed_mtp_scheme(scheme, model, *, explicit):
    """Bind calibrated integer weights to matching SEFP nodes before quantization."""
    from xhquant.api import QuantScheme

    requirements = {}
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
        aliases = (name, name.replace("_", ".")) if name != "lm_head" else (name,)
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
