# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Compare real GLM attention weights against Transformers, without loading MoE."""

import argparse
import json
from pathlib import Path

import torch
from transformers import DynamicCache
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention, Glm5NextTextLinearAttention

from xhmodel_merak.xh_llm.models.glm5_3_flash.checkpoint import Glm53Checkpoint
from xhmodel_merak.xh_llm.models.glm5_3_flash.contract import LINEAR, Glm53CacheSpec, build_transition
from xhmodel_merak.xh_llm.models.glm5_3_flash.modeling import StaticDSA, StaticKDA


def run(model_dir, layer, device, dtype, output, lengths=(78, 1, 1, 1, 256, 55)):
    torch.manual_seed(53)
    checkpoint = Glm53Checkpoint(model_dir)
    config = checkpoint.text_config
    spec = Glm53CacheSpec.from_config(config, 4096)
    linear = config.layer_types[layer] == LINEAR
    with torch.device("meta"):
        source = (Glm5NextTextLinearAttention if linear else Glm5NextTextAttention)(config, layer)
    checkpoint.load_module(source, f"model.language_model.layers.{layer}.self_attn.", device=device, dtype=dtype)
    prefill = (
        (StaticKDA(source, 256, chunk_size=64) if linear else StaticDSA(source, spec, 256))
        .to(device=device, dtype=dtype)
        .eval()
    )
    decode = (
        (StaticKDA(source, 1, chunk_size=64) if linear else StaticDSA(source, spec, 1))
        .to(device=device, dtype=dtype)
        .eval()
    )
    cache = DynamicCache(config=config)
    states = [torch.zeros(s, device=device, dtype=dtype) for s in spec.cache_shapes(config.layer_types[layer])]
    if not linear:
        states[3].fill_(-32768)
    past = 0
    rows = []
    with torch.inference_mode():
        for n in lengths:
            p = 1 if n == 1 else 256
            x = torch.randn(1, n, config.hidden_size, device=device, dtype=dtype) * 0.2
            mask = torch.ones(1, n, device=device, dtype=torch.bool)
            expected = (
                source(x, cache_params=cache, attention_mask=mask)
                if linear
                else source(x, attention_mask=mask, past_key_values=cache)[0]
            )
            t = build_transition(spec, past, n, p, device=device, dtype=dtype)
            padded = torch.nn.functional.pad(x, (0, 0, 0, p - n), value=13)
            model = decode if p == 1 else prefill
            if linear:
                actual, *states = model(padded, t["current_input_length"], t["query_validity"], *states)
            else:
                actual, *states = model(
                    padded,
                    t["past_seq_length"],
                    t["current_input_length"],
                    t["pool_offset"],
                    t["pool_new_count"],
                    t["pool_write_start"],
                    t["query_validity"],
                    t["dsa_attention_mask"],
                    *states,
                )
            a, b = actual[:, :n].float(), expected.float()
            error = (a - b).abs()
            cosine = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
            row = {
                "past": past,
                "length": n,
                "max_abs": error.max().item(),
                "mean_abs": error.mean().item(),
                "cosine": cosine,
                "finite": bool(torch.isfinite(actual).all()),
                "padding_zero": bool(torch.count_nonzero(actual[:, n:]) == 0),
            }
            if linear:
                row["state_max_abs"] = (states[1] - cache.layers[layer].recurrent_states[0]).abs().max().item()
            rows.append(row)
            print(json.dumps(row), flush=True)
            past += n
    result = {
        "layer": layer,
        "kind": config.layer_types[layer],
        "dtype": str(dtype),
        "device": device,
        "steps": rows,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else 0,
    }
    result["passed"] = all(r["finite"] and r["padding_zero"] and r["cosine"] > 0.999 for r in rows)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise AssertionError(f"real-weight attention comparison failed: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="weights/GLM-5.3-Flash")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[78, 1, 1, 1, 256, 55])
    args = parser.parse_args()
    run(args.model, args.layer, args.device, getattr(torch, args.dtype), args.output, args.lengths)
