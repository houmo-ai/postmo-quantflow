# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Relocatable graph inventory and deployment-only loading."""

import json
from pathlib import Path

from .capacity import validate_capacity
from .source import MODEL_ID


ROLES = ("text_encode", "vae_encode", "denoise", "vae_decode")


def bundle_path(root, relative):
    root, path = Path(root).resolve(), Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in str(path) or not path.parts:
        raise ValueError("bundle resources must use confined relative paths")
    target = root / path
    for item in (target, *target.parents):
        if item == root:
            break
        if item.is_symlink():
            raise ValueError("bundle resource cannot be a symlink")
    if not target.is_file():
        raise ValueError(f"bundle resource missing: {relative}")
    return target


def read_metadata(root):
    meta = json.loads(bundle_path(root, "runtime_meta_info.json").read_text())
    if meta.get("schema_version") != 1 or meta.get("model_id") != MODEL_ID or meta.get("kind") != "flux3_klein":
        raise ValueError("unsupported FLUX runtime identity/schema")
    if meta.get("quant_type") not in ("w16a16_sefp", "w8a16_sefp", "w8a8_sefp"):
        raise ValueError("unsupported deployment precision")
    if set(meta["graphs"]) != set(ROLES) or len(set(meta["graphs"].values())) != 4:
        raise ValueError("FLUX deployment requires all four distinct graphs")
    if validate_capacity(meta["capacity"]) != meta["capacity"]:
        raise ValueError("inconsistent derived capacity metadata")
    for relative in [*meta["graphs"].values(), *meta["resources"].values(), *meta["processor_files"]]:
        bundle_path(root, relative)
    for key in ("source_config", "text_config"):
        json.loads(bundle_path(root, meta["resources"][key]).read_text())
    source = json.loads(bundle_path(root, meta["resources"]["source_config"]).read_text())
    if (
        source["model"] != MODEL_ID
        or source["dit"]["flow_map"] is not True
        or source["sampler"]["guidance"] is not None
    ):
        raise ValueError("packaged source/sampling identity disagrees with runtime")
    if not meta["processor_files"] or not any(Path(p).name == "tokenizer.json" for p in meta["processor_files"]):
        raise ValueError("processor resources are incomplete")
    # A metadata edit cannot silently select a different physical graph gear.
    import onnx

    capacity = meta["capacity"]
    weight_bits, activation_bits = {
        "w16a16_sefp": (16, 16),
        "w8a16_sefp": (8, 16),
        "w8a8_sefp": (8, 8),
    }[meta["quant_type"]]
    ch, cw = capacity["canvas_size"]
    context = source["dit"]["in_channels"]["text"]
    expected = {
        "text_encode": {"embeddings": [1, capacity["text_capacity"], context // 8]},
        "vae_encode": {"vae_input": [1, 3, ch, cw]},
        "vae_decode": {"vae_input": [1, 128, ch // 16, cw // 16]},
        "denoise": {
            "target": [1, capacity["target_capacity"], 128],
            "references": [1, capacity["reference_total_capacity"], 128],
            "text_features": [1, capacity["text_capacity"], context],
        },
    }
    for role, path in meta["graphs"].items():
        graph = onnx.load(str(bundle_path(root, path)), load_external_data=False).graph
        ports = {value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in graph.input}
        for name, shape in expected[role].items():
            if ports.get(name) != shape:
                raise ValueError(f"{role}.{name} graph shape disagrees with capacity metadata")
        if len(graph.output) != 1 or not any(node.op_type == "FlashAttention" for node in graph.node):
            raise ValueError(f"{role} must be a single-output deployed FlashAttention graph")
        for node in graph.node:
            attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if node.op_type in ("Linear", "Conv2d"):
                expected_attributes = {
                    "hmfp_weight_man_bit": weight_bits,
                    "hmfp_act_man_bit": activation_bits,
                    "hmfp_weight_nshare": 64,
                    "hmfp_act_nshare": 64,
                    "mode": b"sefp",
                }
            elif node.op_type == "FlashAttention":
                expected_attributes = {f"{part}_bits": activation_bits for part in ("q", "k", "v", "p")}
            else:
                continue
            if any(attributes.get(key) != value for key, value in expected_attributes.items()):
                raise ValueError(f"{role}.{node.name} physical precision disagrees with runtime metadata")
    if "prompt_upsampler" in meta:
        from .upsampler_build import read_component

        if read_component(root) != meta["prompt_upsampler"]:
            raise ValueError("prompt upsampler component disagrees with FLUX metadata")
    return meta


def export_entries(meta):
    entries = {name: dict(path=path, type="hmonnx") for name, path in meta["graphs"].items()}
    entries.update(
        {
            name: dict(path=path, type="embedding" if name == "embedding" else "json")
            for name, path in meta["resources"].items()
        }
    )
    entries["runtime_meta_info"] = dict(path="runtime_meta_info.json", type="json")
    entries["tokenizer"] = dict(path=str(Path(meta["processor_files"][0]).parent), type="tokenizer")
    for index, path in enumerate(meta["processor_files"]):
        if path.endswith(".json"):
            entries[f"processor_{index}"] = dict(path=path, type="json")
    if "prompt_upsampler" in meta:
        component = meta["prompt_upsampler"]
        for role, path in component["graphs"].items():
            entries[f"upsampler_{role}"] = dict(path=path, type=role if role in ("prefill", "decode") else "hmonnx")
        entries["upsampler_component"] = dict(path="upsampler_component.json", type="json")
        entries["upsampler_tokenizer"] = dict(path=str(Path(component["processor_config"]).parent), type="tokenizer")
        entries["upsampler_embedding"] = dict(path="quant_embedding.pt", type="embedding")
        for index, path in enumerate(component["files"]):
            if path.endswith(".json"):
                entries[f"upsampler_json_{index}"] = dict(path=path, type="json")
    return entries


def load_runtime(root, *, device="cuda:0", session_factory=None):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoProcessor

    from .pipeline import ImageRuntime

    meta = read_metadata(root)

    def resource(name):
        return bundle_path(root, meta["resources"][name])

    source = json.loads(resource("source_config").read_text())
    text_config = json.loads(resource("text_config").read_text())
    weights = load_file(str(resource("embedding")))
    if set(weights) != {"embedding.weight"}:
        raise ValueError("invalid token embedding resource")
    weight = weights["embedding.weight"]
    tc = text_config["text_config"]
    if tuple(weight.shape) != (tc["vocab_size"], tc["hidden_size"]):
        raise ValueError("token embedding shape disagrees with packaged text config")
    if source["dit"]["in_channels"]["text"] != 8 * tc["hidden_size"]:
        raise ValueError("packaged feature width disagrees with source config")
    embedding = torch.nn.Embedding.from_pretrained(weight, freeze=True)
    processor = AutoProcessor.from_pretrained(
        bundle_path(root, meta["processor_files"][0]).parent, local_files_only=True, trust_remote_code=False
    )
    processor.tokenizer.padding_side = "right"
    if session_factory is None:
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

        def session_factory(role, path):
            return HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=[device]))

    sessions = {role: session_factory(role, bundle_path(root, path)) for role, path in meta["graphs"].items()}
    upsampler = None
    if "prompt_upsampler" in meta:
        from .upsampler import PromptUpsampler

        upsampler = PromptUpsampler(root, meta["prompt_upsampler"], device=device)
    return ImageRuntime(sessions, processor, embedding, meta["capacity"], source, device=device, upsampler=upsampler)
