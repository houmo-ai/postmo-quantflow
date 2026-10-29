# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Export all three learned graphs and package only portable inference resources."""

import gc
import json
from pathlib import Path

import torch


def validate_config(config):
    from .config import normalize_prefill_config

    config = normalize_prefill_config(config, default=256)
    scheme = config["model"]["quant_scheme"]
    if config.get("target_device") != "XH2a" or scheme.get("target_device") != "XH2a":
        raise ValueError("Wall export currently requires XH2a")
    if scheme.get("quant_type") not in ("w16a16_sefp", "w8a16_sefp", "w8a8_sefp"):
        raise ValueError("unsupported Wall deployment precision")
    for name, default in (("context_length", 1024), ("visual_capacity", 1024)):
        config.setdefault(name, default)
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"invalid {name}")
    if config["prefill_chunk_length"] > config["context_length"] or config["visual_capacity"] % 4:
        raise ValueError("invalid Wall capacities")
    return config


def package_resources(source, root):
    import yaml
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoProcessor

    source, root = Path(source), Path(root)
    root.mkdir(parents=True, exist_ok=True)
    from .config import validate_source_config

    cfg = validate_source_config(json.loads((source / "config.json").read_text()))
    train = yaml.safe_load((source / "config.yml").read_text())
    if sum(train["dof_config"].values()) != 26 or sum(train["agent_pos_config"].values()) != 26:
        raise ValueError("requires checkpoint 26-dimensional action/state layouts")
    if (
        train["data"].get("action_horizon_flow") != 32
        or not train["data"].get("use_state_string_representation")
        or train["data"].get("state_bins") != 512
    ):
        raise ValueError("requires Wall-OSS-0.5 horizon32 and 512-bin state strings")
    resolutions = train["data"]["resolution"]
    if resolutions != {"face_view": 448, "right_wrist_view": 448}:
        raise ValueError("requires the checkpoint's ordered face/right-wrist camera configuration")
    if train["data"].get("use_embodied_system_prompt_ratio", 0) != 0:
        raise ValueError("embodiment-specific system prompts are unsupported")
    proc = AutoProcessor.from_pretrained(source, local_files_only=True, use_fast=True)
    proc.tokenizer.add_tokens(["<|propri|>", "<|action|>"])
    proc.save_pretrained(root / "processor")
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as f:
        emb = f.get_tensor("model.embed_tokens.weight")
        if max(proc.tokenizer.get_vocab().values()) >= emb.shape[0]:
            raise ValueError("tokenizer ids exceed checkpoint embedding")
        save_file({"embedding.weight": emb}, str(root / "quant_embedding.safetensors"))
        stats = {}
        for kind in ("action", "propri"):
            stats[kind] = {}
            stem = f"action_preprocessor.normalizer_{kind}."
            for key in f.keys():
                if key.startswith(stem + "min."):
                    name = key[len(stem + "min.") :]
                    stats[kind][name] = {
                        "min": f.get_tensor(key).tolist(),
                        "delta": f.get_tensor(stem + "delta." + name).tolist(),
                    }
    (root / "normalizers.json").write_text(json.dumps(stats))
    (root / "wall_config.json").write_text(
        json.dumps(
            dict(
                horizon=32,
                action_dim=26,
                long_edge=448,
                vision_config=cfg["vision_config"],
                model_config=cfg,
                train_config=train,
                embedding_rows=emb.shape[0],
            ),
            indent=2,
        )
    )


def export_graph(module, inputs, path, names, device, quant_type):
    from copy import deepcopy

    from xhquant.api import (
        ConfigDict,
        DeviceType,
        QuantScheme,
        convert_fx_model_to_quanted_model,
        convert_quanted_model_to_hmonnx,
        create_quant_config,
    )

    torch.cuda.set_device(torch.device(device))
    quant = ConfigDict(create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)))
    quant.auto_release_unused_parameters = True
    path.parent.mkdir(parents=True, exist_ok=True)
    q = convert_fx_model_to_quanted_model(deepcopy(module).cpu().eval(), inputs, DeviceType.XH2a, quant)
    q.cpu()
    convert_quanted_model_to_hmonnx(q, inputs, str(path), names, ["output"])
    del q
    gc.collect()
    torch.cuda.empty_cache()


def build_bundle(source, root, config, device):
    from xhquant.core import CacheTensor

    from .graph import ActionDenoiser, ExpertStack
    from .processor import vision_geometry
    from .source import load_source, load_visual
    from .vision import PackedVision

    config = validate_config(config)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    package_resources(source, root)
    qtype = config["model"]["quant_scheme"]["quant_type"]
    visual = load_visual(source)
    capacity = config["visual_capacity"]
    side = min(32, int(capacity**0.5) // 2 * 2)
    geometry = vision_geometry(
        side, side, capacity=capacity, head_dim=visual.config.hidden_size // visual.config.num_heads
    )
    vinputs = (torch.randn(1, capacity, 1176), *geometry)
    graph = PackedVision(visual, capacity)
    paths = {
        "visual": "visual_p%d/wall_oss_visual.onnx" % capacity,
        "prefill": "prefill/wall_oss_prefill.onnx",
        "denoise": "denoise/wall_oss_denoise.onnx",
    }
    export_graph(
        graph,
        vinputs,
        root / paths["visual"],
        ["patches", "order", "reverse", "cos", "sin", "mask", "full_range", "window_range"],
        device,
        qtype,
    )
    del graph, visual
    gc.collect()
    src = load_source(source)
    c, p = config["context_length"], config["prefill_chunk_length"]
    arch = src.config
    layers, kvheads, dim, hidden, action_hidden = (
        len(src.layers),
        src.kvheads,
        src.dim,
        arch["hidden_size"],
        arch["dim_inputs"][1],
    )
    caches = tuple(CacheTensor(torch.zeros(1, kvheads, c, dim)) for _ in range(2 * layers))

    def args(n, past):
        return (
            torch.ones(1, 1, n, dim),
            torch.zeros(1, 1, n, dim),
            torch.tensor([past], dtype=torch.int32),
            torch.tensor([n], dtype=torch.int32),
            caches,
        )

    graph = ExpertStack(src, expert=0, query_capacity=p, cache_capacity=c)
    export_graph(
        graph,
        (torch.randn(1, p, hidden), *args(p, 0)),
        root / paths["prefill"],
        ["embeddings", "cos", "sin", "past_length", "input_length"] + [f"cache_{i}" for i in range(2 * layers)],
        device,
        qtype,
    )
    del graph
    gc.collect()
    graph = ActionDenoiser(src, query_capacity=32, cache_capacity=c)
    export_graph(
        graph,
        (torch.randn(1, 32, 26), torch.ones(1, 32, 26), torch.zeros(1, action_hidden), *args(32, p)),
        root / paths["denoise"],
        ["action", "dof", "time", "cos", "sin", "past_length", "input_length"]
        + [f"cache_{i}" for i in range(2 * layers)],
        device,
        qtype,
    )
    del graph, src
    gc.collect()
    meta = dict(
        schema_version=1,
        kind="wall_oss",
        model_id="wall-oss-0.5",
        target_device="XH2a",
        quant_type=qtype,
        context_length=c,
        prefill_chunk_length=p,
        visual_capacity=capacity,
        horizon=32,
        action_dim=26,
        layers=layers,
        kv_heads=kvheads,
        head_dim=dim,
        cache_policy="inplace_replace_action",
        instruction_truncation="right",
        graphs=paths,
        deployment_config=config,
    )
    (root / "runtime_meta_info.json").write_text(json.dumps(meta, indent=2))
    (root / "export_meta_info.json").write_text(json.dumps(export_entries(root, meta), indent=2))
    return meta


def export_entries(root, meta):
    entries = {name: {"path": path, "type": "hmonnx"} for name, path in meta["graphs"].items()}
    entries["embedding"] = {"path": "quant_embedding.safetensors", "type": "embedding"}
    entries["tokenizer"] = {"path": "processor", "type": "tokenizer"}
    for path in [
        Path(root) / "wall_config.json",
        Path(root) / "normalizers.json",
        Path(root) / "runtime_meta_info.json",
        *sorted((Path(root) / "processor").glob("*.json")),
    ]:
        entries[str(path.relative_to(root)).replace("/", "_")] = {"path": str(path.relative_to(root)), "type": "json"}
    return entries
