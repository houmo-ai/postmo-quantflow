# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Complete FastH3 Dense hybrid adapter merge, before folding and PTQ."""

from __future__ import annotations

import torch
from safetensors import safe_open


@torch.no_grad()
def merge_reference_adapter(source, checkpoint_path):
    """Consume the reference Turbo checkpoint through the existing H3 PEFT loader."""
    from ..minimax_h3.lora import load_minimax_h3_lora

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        alpha = int(metadata.get("alpha", "8"))
        keys = list(checkpoint.keys())
        akeys = [key for key in keys if key.endswith(".lora_A.default.weight")]
        if alpha <= 0 or not akeys:
            raise ValueError("reference adapter requires a positive alpha and PEFT LoRA pairs")
        rank = checkpoint.get_slice(akeys[0]).get_shape()[0]
    # The shared loader validates every pair, target and shape before fusion,
    # and removes the PEFT modules afterwards. Do not reinterpret alpha as scale.
    load_minimax_h3_lora(source, checkpoint_path, lora_alpha=alpha, lora_scale=1.0)
    return dict(format="peft", pairs=len(akeys), rank=rank, alpha=alpha, effective_scale=alpha / rank)


@torch.no_grad()
def merge_dense_adapter(source, checkpoint_path, *, compute_device="cuda:0"):
    """Validate every tensor, then apply native addmm and FP32 diff accumulation.

    A CUDA staging weight bounds memory to one target instead of moving the
    entire original 33B transformer to the quantization GPU.
    """
    suffixes = {".lora_A.weight": "A", ".lora_B.weight": "B"}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        meta = checkpoint.metadata() or {}
        if meta.get("format") != "fastvideo-lora-v2":
            raise ValueError("Sol-H3 Dense requires a FastVideo v2 hybrid checkpoint")
        pairs, diffs = {}, []
        for key in checkpoint.keys():
            matched = False
            for suffix, side in suffixes.items():
                if key.endswith(suffix):
                    pairs.setdefault(key[: -len(suffix)], {})[side] = key
                    matched = True
                    break
            if matched:
                continue
            if key.endswith(".diff_b"):
                diffs.append((key[:-7] + ".bias", key))
            elif key.endswith(".diff"):
                diffs.append((key[:-5] + ".weight", key))
            else:
                raise ValueError(f"Dense adapter contains an unsupported or replacement tensor: {key}")
        if int(meta["low_rank_tensors"]) != len(pairs) * 2 or int(meta["diff_tensors"]) != len(diffs):
            raise ValueError("hybrid adapter tensor counts disagree with metadata")
        if int(meta.get("set_weight_tensors", 0)) or not pairs:
            raise ValueError("Dense export cannot consume VSA replacement/gating weights")
        rank = int(meta["rank"])
        targets = []
        for name, pair in sorted(pairs.items()):
            if set(pair) != {"A", "B"}:
                raise ValueError(f"incomplete low-rank pair: {name}")
            module = source.get_submodule(name)
            ashape, bshape = (checkpoint.get_slice(pair[k]).get_shape() for k in ("A", "B"))
            if (
                not isinstance(module, torch.nn.Linear)
                or ashape != [rank, module.in_features]
                or bshape != [module.out_features, rank]
            ):
                raise ValueError(f"hybrid projection shape mismatch: {name}")
            targets.append((module.weight, pair["A"], pair["B"]))
        corrections = []
        for name, key in diffs:
            parameter = source.get_parameter(name)
            if list(parameter.shape) != checkpoint.get_slice(key).get_shape():
                raise ValueError(f"hybrid correction shape mismatch: {name}")
            corrections.append((parameter, key))
        for parameter, akey, bkey in targets:
            weight = parameter.detach().to(compute_device, copy=True)
            a, b = (checkpoint.get_tensor(key).to(device=compute_device, dtype=weight.dtype) for key in (akey, bkey))
            weight.addmm_(b, a)
            parameter.copy_(weight.to(parameter.device))
            del weight, a, b
        for parameter, key in corrections:
            delta = checkpoint.get_tensor(key)
            parameter.copy_((parameter.float() + delta.float()).to(parameter.dtype))
    source.eval().requires_grad_(False)
    return {
        "format": meta["format"],
        "pairs": len(targets),
        "diffs": len(corrections),
        "replacements": 0,
        "rank": rank,
        "effective_scale": 1.0,
    }
