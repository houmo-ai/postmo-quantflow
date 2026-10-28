# Derived from X-Square-Robot/wall-x, Apache-2.0, revision ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f.
# Copyright 2026 HOUMO AI (deployment adapters)
# See licenses/wall-x/LICENSE and THIRD_PARTY_NOTICES.
"""Strict checkpoint interpretation; reference joint forward uses native SDPA."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Expert(nn.Module):
    def __init__(self, hidden, intermediate, heads, kvheads, dim, eps):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(hidden, eps=eps), nn.RMSNorm(hidden, eps=eps)
        self.qkv = nn.Linear(hidden, (heads + 2 * kvheads) * dim, bias=True)
        self.out = nn.Linear(heads * dim, hidden, bias=False)
        self.gate_up = nn.Linear(hidden, 2 * intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def mlp(self, x):
        gate, up = self.gate_up(x).chunk(2, -1)
        return self.down(F.silu(gate) * up)


class Source(nn.Module):
    def __init__(self, config, action_dim=26):
        super().__init__()
        self.config = config
        self.heads, self.kvheads = config["num_attention_heads"], config["num_key_value_heads"]
        self.dim = config["hidden_size"] // self.heads
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Expert(
                            e["hidden_size"],
                            e["intermediate_size"],
                            self.heads,
                            self.kvheads,
                            self.dim,
                            config["rms_norm_eps"],
                        )
                        for e in config["experts"]
                    ]
                )
                for _ in range(config["num_hidden_layers"])
            ]
        )
        ah = config["experts"][1]["hidden_size"]
        self.action_norm = nn.RMSNorm(ah, eps=config["rms_norm_eps"])
        self.w1, self.w2, self.w3 = (
            nn.Linear(action_dim * 2, ah, bias=False),
            nn.Linear(ah * 2, ah, bias=False),
            nn.Linear(ah, ah, bias=False),
        )
        self.action_out = nn.Linear(ah, action_dim, bias=False)

    def action_embed(self, action, dof, time):
        x = self.w1(torch.cat((action, dof), -1))
        return self.w3(F.silu(self.w2(torch.cat((x, time[:, None].expand(-1, action.shape[1], -1)), -1))))

    def joint(self, prefix, action, dof, time, cos, sin):
        from .graph import rotate

        hidden = [prefix, self.action_embed(action, dof, time)]
        lengths = [x.shape[1] for x in hidden]
        for experts in self.layers:
            qkv = [e.qkv(e.norm1(x)) for e, x in zip(experts, hidden, strict=True)]
            q, k, v = torch.cat(qkv, 1).split(
                (self.heads * self.dim, self.kvheads * self.dim, self.kvheads * self.dim), -1
            )
            q = rotate(q.reshape(1, -1, self.heads, self.dim).transpose(1, 2), cos, sin)
            k = rotate(k.reshape(1, -1, self.kvheads, self.dim).transpose(1, 2), cos, sin)
            v = v.reshape(1, -1, self.kvheads, self.dim).transpose(1, 2)
            a = (
                F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                .transpose(1, 2)
                .reshape(1, -1, self.heads * self.dim)
            )
            hidden = [x + e.out(y) for e, x, y in zip(experts, hidden, a.split(lengths, 1), strict=True)]
            hidden = [x + e.mlp(e.norm2(x)) for e, x in zip(experts, hidden, strict=True)]
        return self.action_out(self.action_norm(hidden[1]))


def load_source(root):
    from safetensors import safe_open

    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    from .config import validate_source_config

    validate_source_config(config)
    with torch.device("meta"):
        source = Source(config)
    mapping = {}
    for i in range(config["num_hidden_layers"]):
        for j in range(2):
            p, q = f"layers.{i}.{j}.", f"model.layers.{i}."
            for a, b in {
                "norm1.weight": f"input_layernorms.{j}.weight",
                "norm2.weight": f"post_attention_layernorms.{j}.weight",
                "qkv.weight": f"self_attn.qkv_proj_experts.{j}.weight",
                "qkv.bias": f"self_attn.qkv_proj_experts.{j}.bias",
                "out.weight": f"self_attn.o_proj_experts.{j}.weight",
                "gate_up.weight": f"moe.experts.{j}.gate_up_proj.weight",
                "down.weight": f"moe.experts.{j}.down_proj.weight",
            }.items():
                mapping[p + a] = q + b
    mapping.update(
        {
            "action_norm.weight": "model.norms.1.weight",
            **{f"{n}.weight": f"action_preprocessor.{n}.weight" for n in ("w1", "w2", "w3")},
            "action_out.weight": "action_preprocessor.action_proj_back.weight",
        }
    )
    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as f:
        state = {dst: f.get_tensor(src).float() for dst, src in mapping.items()}
    source.load_state_dict(state, strict=True, assign=True)
    return source.eval()


def load_visual(root):
    from safetensors import safe_open
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

    config = json.loads((Path(root) / "config.json").read_text())["vision_config"]
    config["in_channels"] = config.get("in_chans", 3)
    cfg = Qwen2_5_VLVisionConfig(**config)
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen2_5_VisionTransformerPretrainedModel(cfg)
    state = {}
    with safe_open(Path(root) / "model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            if not key.startswith("visual."):
                continue
            name = key[7:]
            value = f.get_tensor(key).float()
            if ".mlp.gate_up_proj." in name:
                gate, up = value.chunk(2, 0)
                state[name.replace("gate_up_proj", "gate_proj")] = gate
                state[name.replace("gate_up_proj", "up_proj")] = up
            else:
                state[name] = value
    model.load_state_dict(state, strict=True, assign=True)
    model.rotary_pos_emb.inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, cfg.hidden_size // cfg.num_heads // 2, 2).float() / (cfg.hidden_size // cfg.num_heads // 2))
    )
    return model.eval()
