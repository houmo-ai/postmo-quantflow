"""Qwen target-owned speculative output-head contract."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

import torch


def load_checkpoint_head(source, *, dtype=torch.float16):
    """Load a target head; embedding fallback is legal only for tied models."""
    from safetensors import safe_open

    root = Path(source)
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", config)
    index = root / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
    else:
        mapping = {}
        for file in sorted(root.glob("*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as tensors:
                for key in tensors.keys():
                    if key in mapping:
                        raise ValueError(f"Duplicate checkpoint tensor: {key}")
                    mapping[key] = file.name
    if "lm_head.qweight" in mapping:
        if "lm_head.weight" in mapping:
            raise ValueError("Target has both floating and packed head weights")
        module, bits = load_packed_head(source)
        return module.to(dtype=dtype), bits
    key = "lm_head.weight"
    if key not in mapping:
        if not text.get("tie_word_embeddings", config.get("tie_word_embeddings", False)):
            raise ValueError("Untied target checkpoint is missing lm_head.weight")
        keys = [k for k in mapping if k in ("model.embed_tokens.weight", "model.language_model.embed_tokens.weight")]
        if len(keys) != 1:
            raise ValueError("Tied target requires exactly one token embedding")
        key = keys[0]
    with safe_open(root / mapping[key], framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor(key).to(dtype)
    if tuple(weight.shape) != (text["vocab_size"], text["hidden_size"]):
        raise ValueError("Target head shape does not match text configuration")
    module = torch.nn.Linear(text["hidden_size"], text["vocab_size"], bias=False, device="meta")
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module, None


def target_head_bits(scheme, node_name="lm_head"):
    """Read target precision from the same node-override precedence as export."""
    if scheme is None:
        scheme = {"quant_type": "w8a8h1_sefp"}
    if hasattr(scheme, "to_dict"):
        scheme = scheme.to_dict()
    if not isinstance(scheme, Mapping):
        raise ValueError("Shared Qwen head requires an explicit target quant_scheme mapping")
    # Use the export SDK's resolver, including op/node and legacy precedence.
    from xhquant.quantization.xh2a._xh2a_quant_schema import XH2aQuantSchema

    scheme = copy.deepcopy(dict(scheme))
    # Mirror XHBaseModel.get_quant_cfg's public/legacy alias normalization.
    for public, internal in (("nodes", "nodes_cfg"), ("ops", "ops_cfg")):
        overrides = scheme.pop(public, None) or {}
        destination = scheme.setdefault(internal, {})
        for name, value in overrides.items():
            destination.setdefault(name, {"quant_type": value} if isinstance(value, str) else value)
    resolved = XH2aQuantSchema.from_schema(scheme).get_quant_config(node_name, torch.nn.Linear)
    bits = resolved.w_schema.quantizer.qspec.man_bit
    if bits not in (4, 5, 8):
        raise ValueError(f"Unsupported shared target head W{bits}")
    return bits


def resolve_shared_head_bits(scheme, legacy_bits=None, *draft_configs):
    """Resolve the target head and reject explicitly conflicting legacy knobs."""
    bits = target_head_bits(scheme)
    declarations = [("spec_draft_head_weight_bits", legacy_bits)]
    for cfg in draft_configs:
        if cfg is not None:
            value = (
                cfg.get("draft_head_weight_bits")
                if isinstance(cfg, Mapping)
                else (cfg.draft_head_weight_bits if getattr(cfg, "_head_bits_explicit", True) else None)
            )
            declarations.append(("draft_head_weight_bits", value))
            draft_scheme = (
                cfg.get("quant_scheme")
                if isinstance(cfg, Mapping)
                else (cfg.quant_scheme if getattr(cfg, "_head_scheme_explicit", True) else None)
            )
            if draft_scheme is not None:
                if hasattr(draft_scheme, "to_dict"):
                    draft_scheme = draft_scheme.to_dict()
                for node_name in ("lm_head", "core_lm_head"):
                    if node_name in (draft_scheme.get("nodes") or {}) or node_name in (
                        draft_scheme.get("nodes_cfg") or {}
                    ):
                        declarations.append(
                            (f"draft quant_scheme {node_name}", target_head_bits(draft_scheme, node_name))
                        )
    for name, value in declarations:
        if value is not None and value != bits:
            raise ValueError(f"{name}={value} conflicts with shared target lm_head W{bits}; remove the draft override")
    return bits


def inherit_head_config(config, bits, node_name="lm_head"):
    """Resolve materialized draft configs as well as YAML dictionaries."""
    if config is None:
        return None
    if isinstance(config, Mapping):
        result = copy.deepcopy(dict(config))
        result["draft_head_weight_bits"] = bits
        scheme = result.get("quant_scheme")
        if scheme is not None:
            if hasattr(scheme, "to_dict"):
                scheme = scheme.to_dict()
            scheme.setdefault("nodes_cfg", {})[node_name] = {"quant_type": f"w{bits}a8h1_sefp"}
            result["quant_scheme"] = scheme
        return result
    result = copy.deepcopy(config)
    result.draft_head_weight_bits = bits
    scheme = result.quant_scheme
    if hasattr(scheme, "to_dict"):
        scheme = scheme.to_dict()
    scheme = copy.deepcopy(scheme or {"quant_type": "w8a8h1_sefp"})
    scheme.setdefault("nodes_cfg", {})[node_name] = {"quant_type": f"w{bits}a8h1_sefp"}
    result.quant_scheme = scheme
    return result


def quantized_head(graph):
    """Find the unique target head without depending on FX name flattening."""
    heads = [
        graph.get_submodule(n.target)
        for n in graph.graph.nodes
        if n.op == "call_module" and str(n.target).replace(".", "_").endswith("lm_head")
    ]
    if len(heads) != 1:
        raise ValueError(f"Expected one target lm_head, found {len(heads)}")
    head = heads[0]
    if not getattr(head, "is_quant_fixed", False) or getattr(head, "qweight", None) is None:
        raise ValueError("Target head must be quantized before exporting speculative graphs")
    return head


def bind_quantized_head(graph, source):
    """Clone operator metadata but alias all tensors of the fixed target head.

    PTQ recognizes fixed operators and skips weight quantization. Separate
    operator instances retain independent shape metadata for prefill/decode.
    """
    nodes = [
        n for n in graph.graph.nodes if n.op == "call_module" and str(n.target).replace(".", "_").endswith("lm_head")
    ]
    if len(nodes) != 1:
        raise ValueError(f"Expected one speculative lm_head, found {len(nodes)}")
    destination = graph.get_submodule(nodes[0].target)
    for name in ("in_features", "out_features"):
        if getattr(destination, name, None) != getattr(source, name, None):
            raise ValueError(f"Shared head {name} mismatch; trimmed draft heads cannot share the full target head")
    memo = {}
    visited = set()

    def collect(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, torch.Tensor):
            memo[id(value)] = value
        elif isinstance(value, torch.nn.Module):
            for item in value.__dict__.values():
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(source)
    graph.set_submodule(nodes[0].target, copy.deepcopy(source, memo))


class SharedHeadMixin:
    """Export consumers of the target's fixed quantized operator, including W8."""

    def bind_target_head(self, head):
        if head is None or not getattr(head, "is_quant_fixed", False):
            raise ValueError("A fixed target head is required")
        expected_bits = getattr(self.config, "draft_head_weight_bits", None)
        if expected_bits is None:
            expected_bits = getattr(self.config, "spec_draft_head_weight_bits", None)
        if expected_bits is not None and head.w_cfg.qspec.man_bit != expected_bits:
            raise ValueError("Fixed target head precision conflicts with inherited configuration")
        self._shared_target_head = head

    def _prepare_quantized_graph(self, graph):
        source = getattr(self, "_shared_target_head", None)
        if source is not None:
            bind_quantized_head(graph, source)
        return graph


def load_packed_head(source):
    from gptqmodel.quantization.config import FORMAT, QuantizeConfig
    from gptqmodel.quantization.qwen35_mtp import CheckpointTensors, load_packed_linear

    from ._dequant_converter import gptqmodel_torch_qlinear_converter

    reader = CheckpointTensors(source)
    config = QuantizeConfig.from_quant_config(reader.config["quantization_config"])
    if config.format not in (FORMAT.GPTQ, FORMAT.GPTQ_V2) or config.rotation:
        raise ValueError("Shared head requires unrotated GPTQ/GPTQ V2 weights")
    if not config.lm_head or config.dynamic_get("lm_head") is False:
        raise ValueError("Packed shared head conflicts with main head metadata")
    values = {
        key: config.dynamic_get("lm_head", key, getattr(config, key))
        for key in ("bits", "group_size", "sym", "desc_act")
    }
    if values["bits"] not in (4, 5, 8):
        raise ValueError(f"Unsupported packed shared head W{values['bits']}")
    if not values.pop("sym") or values.pop("desc_act"):
        raise ValueError("Shared head requires symmetric quantization without act-order")
    text = reader.config.get("text_config", reader.config)
    module = load_packed_linear(
        reader,
        "lm_head",
        **values,
        in_features=text["hidden_size"],
        out_features=text["vocab_size"],
        qzero_format=1 if config.format == FORMAT.GPTQ else 2,
    )
    bits = values["bits"]
    gptqmodel_torch_qlinear_converter(module)
    return module, bits
