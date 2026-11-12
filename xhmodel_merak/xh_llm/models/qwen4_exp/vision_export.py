"""Bounded Qwen4 visual component loading and direct QuantGraph publication."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

from xhquant import nn as xhnn
from xhquant.api import (
    ConfigDict,
    FrontendType,
    PrecisionMode,
    ptq_quantize,
    to_export_hmonnx_from_quanted_graph,
    to_frontend_graph,
    to_quant_graph,
)

from ....utils import calculate_file_md5
from ...wrap_model import wrap_llm_model
from ..qwen3_5._vision_model_impl import _build_visual_rotary_cache
from ..qwen3_5.visual_token_gears import (
    VISUAL_ATTENTION_MASK_FORMAT,
    VISUAL_ATTENTION_MASK_OPERATOR,
    VISUAL_ATTENTION_MASK_SHAPE,
    VISUAL_ROTARY_POSITION_FORMAT,
    build_visual_token_gear_inputs,
    factor_image_token_grid,
    normalize_image_token_gears,
)


class Qwen4VisionCheckpoint:
    """Read only the selected native visual component, without text weights."""

    def __init__(self, model_dir, *, dtype=torch.float16):
        self.root = Path(model_dir)
        self.config = AutoConfig.from_pretrained(self.root).vision_config
        if getattr(self.config, "deepstack_visual_indexes", ()):
            raise ValueError("Qwen4 visual export does not support deepstack features")
        if self.config.hidden_size % self.config.num_heads or self.config.hidden_size // self.config.num_heads % 4:
            raise ValueError("Qwen4 visual RoPE requires head_dim divisible by four")
        self.dtype = dtype
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            self.weight_map = {}
            for path in sorted(self.root.glob("*.safetensors")):
                with safe_open(path, framework="pt", device="cpu") as source:
                    for name in source.keys():
                        if name.startswith("model.visual."):
                            if name in self.weight_map:
                                raise ValueError(f"duplicate visual checkpoint tensor {name}")
                            self.weight_map[name] = path.name

    def load_module(self, module, prefix):
        expected = module.state_dict()
        keys = {name for name in self.weight_map if name.startswith(prefix)}
        required = {prefix + name for name in expected}
        if keys != required:
            raise ValueError(
                f"visual component tensor keys differ at {prefix}: missing={required - keys}, extra={keys - required}"
            )
        for name, reference in expected.items():
            full_name = prefix + name
            with safe_open(self.root / self.weight_map[full_name], framework="pt", device="cpu") as source:
                value = source.get_tensor(full_name)
            if value.shape != reference.shape or value.dtype not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ):
                raise ValueError(f"invalid native visual tensor {full_name}: {value.shape}/{value.dtype}")
            value = value.to(dtype=self.dtype)
            parent_name, _, leaf = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            if leaf in parent._parameters:
                setattr(parent, leaf, nn.Parameter(value, requires_grad=False))
            else:
                parent._buffers[leaf] = value
        return module.eval()


class Qwen4VisionStem(nn.Module):
    def __init__(self, source, config):
        super().__init__()
        self.patch_embed = wrap_llm_model(source.patch_embed, config)
        self.pos_embed = source.pos_embed
        self.capacity = int(config.image_token_capacity) * source.spatial_merge_size**2
        cos, sin = _build_visual_rotary_cache(
            source.rotary_pos_emb,
            int(config.visual_rope_cache_length),
            device="cpu",
            dtype=self.pos_embed.weight.dtype,
        )
        self.register_buffer("rotary_cos_cached", cos, persistent=False)
        self.register_buffer("rotary_sin_cached", sin, persistent=False)
        self.rotary_gather = xhnn.Gather(axis=0)
        self.axis_dim = int(cos.shape[-1])
        head_dim = source.config.hidden_size // source.config.num_heads
        self.padding = ((head_dim + 63) // 64) * 32 - self.axis_dim * 2

    def forward(self, pixel_values, position_ids, position_weights, rotary_position_ids):
        hidden = self.patch_embed(pixel_values)
        hidden = hidden + (self.pos_embed(position_ids) * position_weights.unsqueeze(-1)).sum(dim=0).unsqueeze(0)
        cos = self.rotary_gather(self.rotary_cos_cached, rotary_position_ids)
        sin = self.rotary_gather(self.rotary_sin_cached, rotary_position_ids)
        cos = cos.permute(1, 0, 2).reshape(self.capacity, self.axis_dim * 2)
        sin = sin.permute(1, 0, 2).reshape(self.capacity, self.axis_dim * 2)
        if self.padding:
            cos = torch.nn.functional.pad(cos, (0, self.padding), value=1.0)
            sin = torch.nn.functional.pad(sin, (0, self.padding), value=0.0)
        return hidden, torch.cat((cos, cos), dim=-1).unsqueeze(-2), torch.cat((sin, sin), dim=-1).unsqueeze(-2)


class Qwen4VisionBlock(nn.Module):
    def __init__(self, source, config):
        super().__init__()
        self.block = wrap_llm_model(source, config)
        # The shared rewrite's executable projections own their padded weights.
        del self.block.attn.qkv

    def forward(self, hidden_states, rotary_cos, rotary_sin, attention_mask):
        return self.block(hidden_states, (rotary_cos, rotary_sin), attention_mask)


class Qwen4VisionMerger(nn.Module):
    def __init__(self, source, config):
        super().__init__()
        self.merger = wrap_llm_model(source, config)

    def forward(self, hidden_states):
        return self.merger(hidden_states)


def _export_component(wrapped, inputs, path, *, input_names, output_names, quant_scheme, chip_arch, device):
    frontend = quanted = None
    try:
        wrapped = wrapped.float().cpu().eval()
        frontend = to_frontend_graph(wrapped, FrontendType.TorchFX, inputs)
        quanted = to_quant_graph(frontend, chip_arch, ConfigDict(quant_scheme))
        ptq_quantize(
            quanted,
            [inputs],
            PrecisionMode.ALIGNED,
            [device],
            auto_release_unused_parameters=True,
            infer_shape=False,
        )
        quanted.fixed()
        return to_export_hmonnx_from_quanted_graph(
            quanted,
            inputs,
            str(path),
            input_names=input_names,
            output_names=output_names,
            save_as_external_data=True,
        )
    finally:
        del frontend, quanted, wrapped
        gc.collect()
        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


def export_qwen4_exp_vision_components(
    model_dir, output_dir, *, visual_config, quant_scheme, device="cpu", dtype=torch.float16
):
    # Importing the model-owned registrations is required for direct callers too.
    from . import vision  # noqa: F401

    reader = Qwen4VisionCheckpoint(model_dir, dtype=dtype)
    config = reader.config
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    gears = normalize_image_token_gears(visual_config.image_token_gears)
    records = []
    for gear in gears:
        cfg = ConfigDict(visual_config.to_dict())
        cfg.image_token_capacity = gear
        capacity = gear * config.spatial_merge_size**2
        with torch.device("meta"):
            shell = Qwen4ExpVisionModel(config).to(dtype=dtype)
        grid = factor_image_token_grid(gear, config.spatial_merge_size)
        geometry = build_visual_token_gear_inputs(
            grid,
            patch_capacity=capacity,
            num_position_embeddings=config.num_position_embeddings,
            spatial_merge_size=config.spatial_merge_size,
            rotary_cache_length=cfg.visual_rope_cache_length,
            dtype=torch.float32,
        )
        head_dim = ((config.hidden_size // config.num_heads + 63) // 64) * 64
        hidden = torch.zeros(1, capacity, config.hidden_size)
        cos, sin = torch.ones(capacity, 1, head_dim), torch.zeros(capacity, 1, head_dim)
        pixels = torch.zeros(1, capacity, config.in_channels * config.temporal_patch_size * config.patch_size**2)
        components = []
        try:
            for name in ("stem", *(f"block_{i:02d}" for i in range(config.depth)), "merger"):
                parts = (
                    ("patch_embed", "pos_embed")
                    if name == "stem"
                    else (("merger",) if name == "merger" else (f"blocks.{int(name.split('_')[1])}",))
                )
                wrapped = None
                try:
                    if any(not value.is_meta for value in shell.parameters()):
                        raise RuntimeError("a previous visual component retained resident parameters")
                    for part in parts:
                        reader.load_module(shell.get_submodule(part), f"model.visual.{part}.")
                    if name == "stem":
                        wrapped = Qwen4VisionStem(shell, cfg)
                        inputs = [
                            pixels,
                            geometry["position_ids"].int(),
                            geometry["position_weights"],
                            geometry["rotary_position_ids"].int(),
                        ]
                        input_names = ["pixel_values", "position_ids", "position_weights", "rotary_position_ids"]
                        output_names = ["hidden_states", "rotary_cos", "rotary_sin"]
                    elif name == "merger":
                        wrapped = Qwen4VisionMerger(shell.merger, cfg)
                        inputs, input_names, output_names = [hidden], ["hidden_states"], ["image_embeds"]
                    else:
                        wrapped = Qwen4VisionBlock(shell.blocks[int(name.split("_")[1])], cfg)
                        inputs = [hidden, cos, sin, geometry["attention_mask"]]
                        input_names = ["hidden_states", "rotary_cos", "rotary_sin", "attention_mask"]
                        output_names = ["hidden_states_out"]
                    path = Path(
                        _export_component(
                            wrapped,
                            inputs,
                            root / f"visual_m{gear}" / f"{name}.onnx",
                            input_names=input_names,
                            output_names=output_names,
                            quant_scheme=quant_scheme,
                            chip_arch=visual_config.chip_arch,
                            device=device,
                        )
                    )
                    components.append(
                        {
                            "name": name,
                            "hmonnx": path.relative_to(root).as_posix(),
                            "md5": calculate_file_md5(str(path)),
                            "input_names": input_names,
                            "output_names": output_names,
                        }
                    )
                finally:
                    for part in parts:
                        shell.get_submodule(part).to_empty(device="meta")
                    del wrapped
                    gc.collect()
            records.append({"image_token_capacity": gear, "patch_token_capacity": capacity, "components": components})
        finally:
            del shell
            gc.collect()
    metadata = {
        "abi_version": 2,
        "layout": "components",
        "hmonnx": None,
        "dtype": "float16",
        "vision_config": config.to_dict(),
        "quant_scheme": dict(quant_scheme),
        "patch_size": config.patch_size,
        "temporal_patch_size": config.temporal_patch_size,
        "spatial_merge_size": config.spatial_merge_size,
        "in_channels": config.in_channels,
        "hidden_size": config.hidden_size,
        "num_heads": config.num_heads,
        "num_position_embeddings": config.num_position_embeddings,
        "max_size_t": visual_config.max_size_t,
        "image_token_gears": list(gears),
        "image_token_capacity": max(gears),
        "patch_token_capacity": max(gears) * config.spatial_merge_size**2,
        "visual_rope_cache_length": visual_config.visual_rope_cache_length,
        "visual_input_mode": "patches",
        "token_unit": "post_spatial_merge",
        "routing_policy": "smallest_fit",
        "overflow_policy": "reject",
        "attention_mask_format": VISUAL_ATTENTION_MASK_FORMAT,
        "attention_mask_shape": VISUAL_ATTENTION_MASK_SHAPE,
        "attention_mask_operator": VISUAL_ATTENTION_MASK_OPERATOR,
        "rotary_position_format": VISUAL_ROTARY_POSITION_FORMAT,
        "input_names": ["pixel_values", "position_ids", "position_weights", "rotary_position_ids", "attention_mask"],
        "gears": records,
        "gear_manifest": "visual_gears.json",
    }
    manifest = {**metadata, "schema_version": 2}
    (root / "visual_gears.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return metadata
