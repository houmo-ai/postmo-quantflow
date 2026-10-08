# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Export vision one component at a time, independently of the text checkpoint."""

import gc
import json
import os
import shutil
from pathlib import Path

import torch
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

from xhquant.api import (
    PrecisionMode,
    QuantScheme,
    ptq_quantize,
    to_export_hmonnx_from_quanted_graph,
    to_frontend_graph,
    to_quant_graph,
)

from ....utils import calculate_file_md5
from .checkpoint import Glm53Checkpoint
from .vision import StaticVisionBlock, StaticVisionMerger, StaticVisionStem


def export_glm53_vision(model_dir, output_dir, capacity, *, device="cuda:0", chip_arch="XH2a"):
    """Bound live parameters by one vision block or the final language projector."""
    reader = Glm53Checkpoint(model_dir)
    config = reader.config.vision_config
    if capacity <= 0 or capacity % config.spatial_merge_size**2:
        raise ValueError("vision_max_patches must be a positive multiple of spatial_merge_size squared")
    if config.hidden_size % config.num_heads or (config.hidden_size // config.num_heads) % 4:
        raise ValueError("vision RoPE requires head_dim divisible by four")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    with torch.device("meta"):
        shell = Glm5NextVisionModel(config).half()
    width, dim = config.hidden_size, config.hidden_size // config.num_heads
    patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
    hidden = torch.zeros(1, capacity, width, dtype=torch.float16)
    pixels = torch.zeros(1, capacity, patch_dim, dtype=torch.float16)
    cos = torch.ones(1, 1, capacity, dim, dtype=torch.float16)
    sin = torch.zeros_like(cos)
    valid_length = torch.tensor([capacity], dtype=torch.int32)
    components = []
    names = ["stem", *[f"block_{i:02d}" for i in range(config.depth)], "merger"]
    for name in names:
        if name == "stem":
            reader.load_module(shell.patch_embed, "model.visual.patch_embed.")
            wrapped = StaticVisionStem(shell.patch_embed)
            inputs, input_names = [pixels], ["pixel_values"]
        elif name == "merger":
            for part in ("post_layernorm", "downsample", "merger"):
                reader.load_module(getattr(shell, part), f"model.visual.{part}.")
            wrapped = StaticVisionMerger(shell, capacity)
            inputs, input_names = [hidden], ["hidden_states"]
        else:
            index = int(name.split("_")[1])
            reader.load_module(shell.blocks[index], f"model.visual.blocks.{index}.")
            wrapped = StaticVisionBlock(shell.blocks[index], capacity)
            inputs, input_names = [hidden, cos, sin, valid_length], ["hidden_states", "rotary_cos", "rotary_sin", "valid_length"]
        wrapped.eval()
        frontend = to_frontend_graph(wrapped, "TorchFX", inputs)
        scheme = QuantScheme(quant_type="w8a16h0_sefp", target_device=chip_arch)
        quanted = to_quant_graph(frontend, chip_arch, scheme.to_dict())
        ptq_quantize(
            quanted, [inputs], PrecisionMode.ALIGNED, [device], auto_release_unused_parameters=True, infer_shape=False
        )
        output = directory / (name + ".onnx")
        to_export_hmonnx_from_quanted_graph(
            quanted,
            inputs,
            str(output),
            input_names=input_names,
            output_names=["image_embeds" if name == "merger" else "hidden_states_out"],
            save_as_external_data=True,
        )
        components.append({"name": name, "hmonnx": output.name, "md5": calculate_file_md5(str(output))})
        del wrapped, frontend, quanted
        # The shell retains references to loaded native parameters; release those too.
        if name == "stem":
            shell.patch_embed.to_empty(device="meta")
        elif name == "merger":
            for part in ("post_layernorm", "downsample", "merger"):
                getattr(shell, part).to_empty(device="meta")
        else:
            shell.blocks[index].to_empty(device="meta")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "abi_version": 2,
        "directory": "vision",
        "max_patches_per_frame": capacity,
        "quant_type": "w8a16h0_sefp",
        "dtype": "float16",
        "components": components,
        "vision_config": config.to_dict(),
    }
    (directory / "vision_meta.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def attach_glm53_vision(text_meta, model_dir, output_dir, capacity, *, device="cuda:0"):
    """Create a portable bundle, reusing existing text files through hard links."""
    from transformers import AutoConfig

    source_meta = Path(text_meta).resolve()
    original = json.loads(source_meta.read_text())
    if original.get("glm53_state_abi_version") != 1 or original.get("glm53_vision"):
        raise ValueError("attach requires an existing GLM-5.3 text artifact with state ABI 1")
    source_config = AutoConfig.from_pretrained(source_meta.parent / original["hf_config"])
    checkpoint_config = AutoConfig.from_pretrained(model_dir)
    for key in (
        "image_token_id",
        "video_token_id",
        "image_start_token_id",
        "image_end_token_id",
        "video_start_token_id",
        "video_end_token_id",
    ):
        if getattr(source_config, key) != getattr(checkpoint_config, key):
            raise ValueError(f"text artifact and vision checkpoint disagree on {key}")
    if source_config.to_dict() != checkpoint_config.to_dict():
        # Paths/version hints may differ after save_pretrained; compare model semantics.
        for name in ("text_config", "vision_config"):
            a, b = getattr(source_config, name).to_dict(), getattr(checkpoint_config, name).to_dict()
            for d in (a, b):
                for key in ("_name_or_path", "transformers_version", "dtype"):
                    d.pop(key, None)
            if a != b:
                raise ValueError("text artifact and vision checkpoint configurations differ")
    output = Path(output_dir).resolve()
    if output == source_meta.parent or source_meta.parent in output.parents:
        raise ValueError("the new bundle must be outside the source artifact directory")
    # Same filesystem is intentional: never silently copy hundreds of GB.
    shutil.copytree(source_meta.parent, output, copy_function=os.link)
    vision = export_glm53_vision(
        model_dir, output / "vision", capacity, device=device, chip_arch=original["model_config"]["chip_arch"]
    )
    original["glm53_vision"] = vision
    original["supported_modalities"] = ["text", "image", "video"]
    original["model_config"].update(export_vision=True, vision_max_patches=capacity)
    # Replace the linked metadata atomically; never modify the source hard link.
    destination = output / "golden_meta_info.json"
    temporary = output / ".multimodal_meta.json"
    temporary.write_text(json.dumps(original, indent=2) + "\n")
    temporary.replace(destination)
    return destination
