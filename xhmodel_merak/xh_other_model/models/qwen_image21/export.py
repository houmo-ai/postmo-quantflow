# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Sequential DiT component export using an explicit real calibration request."""

import gc
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .artifacts import DenoiseGear
from .graph import BlockCore, CachedBlock, ExtractBlock, ImageProjection, OutputHead, TextProjection, TimeConditioning
from .runtime import TokenProjection, prepare_conditioning, timestep_features


def export_cpu_graph(module, inputs, path, input_names, output_names, *, target_device, quant_config, dynamo=False):
    """Keep source/trace/export on CPU; SDK weight PTQ uses the current CUDA device."""
    from xhquant.api import (
        ConfigDict,
        convert_dynamo_model_to_quanted_model,
        convert_fx_model_to_quanted_model,
        convert_quanted_model_to_hmonnx,
    )

    def cpu_inputs(values):
        for value in values:
            if isinstance(value, (tuple, list)):
                cpu_inputs(value)
            elif isinstance(value, torch.Tensor) and value.device.type != "cpu":
                raise ValueError("graph export requires CPU calibration inputs")

    if any(value.device.type != "cpu" for value in (*module.parameters(), *module.buffers())):
        raise ValueError("graph export keeps source weights and buffers on CPU")
    cpu_inputs(inputs)
    config = ConfigDict(deepcopy(quant_config))
    config.auto_release_unused_parameters = True
    converter = convert_dynamo_model_to_quanted_model if dynamo else convert_fx_model_to_quanted_model
    quanted = None
    try:
        # SDK conversion mutates modules; only the current graph owns this copy.
        quanted = converter(deepcopy(module).eval(), inputs, target_device, config)
        quanted.cpu()
        convert_quanted_model_to_hmonnx(quanted, inputs, str(path), input_names, output_names)
    finally:
        del quanted
        gc.collect()
        torch.cuda.empty_cache()


def make_text_projection(source):
    """Preserve RMSNorm as an SDK operator and fold its zero-centered scale."""
    from xhquant.nn import RMSNorm

    norm = RMSNorm(source.txt_in.text_norm.weight.numel(), source.txt_in.text_norm.eps)
    norm.weight = torch.nn.Parameter(1 + source.txt_in.text_norm.weight.detach().float())
    return TextProjection(source, norm=norm)


@torch.no_grad()
def export_full_dit_bundle(
    source,
    output_dir,
    *,
    gear,
    layout,
    compact_rope,
    text_features,
    image_latents,
    timestep,
    target_device,
    quant_config,
    use_noncausal_flash_attention=False,
):
    """Export complete text/image DiT stacks with caller-owned in-place KVCache."""
    from .full_graph import SegmentDenoiser
    from .segment_runtime import SegmentDenoiseRequest

    if not source.config.causal_condition or source.config.patch_size != 1 or not source.transformer_blocks:
        raise ValueError("complete DiT export requires causal_condition, patch_size=1 and a nonempty stack")
    if any(p.dtype != torch.float32 for p in source.parameters()):
        raise ValueError("complete DiT export requires fresh FP32 weights")
    if text_features.dtype != torch.float32 or image_latents.dtype != torch.float32:
        raise ValueError("complete DiT export requires FP32 calibration inputs")
    if layout.prefix_capacity != gear.prefix_tokens or layout.target_capacity != gear.target_tokens:
        raise ValueError("calibration layout disagrees with DiT gear")
    if any(p.device.type != "cpu" for p in source.parameters()):
        raise ValueError("complete DiT export keeps source weights on CPU")
    if text_features.device.type != "cpu" or image_latents.device.type != "cpu" or compact_rope.device.type != "cpu":
        raise ValueError("complete DiT export requires CPU calibration inputs")
    if gear.text_tokens is None:
        gear = replace(gear, text_tokens=gear.encoder_tokens)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    core = source.transformer_blocks[0].attn
    layers = len(source.transformer_blocks)
    models = {
        "text": SegmentDenoiser(source, gear.text_tokens, text_projection=make_text_projection(source)),
        "image": SegmentDenoiser(
            source, gear.image_tokens, cache_capacity=gear.cache_tokens,
            use_noncausal_flash_attention=use_noncausal_flash_attention,
        ),
    }
    request = SegmentDenoiseRequest(
        layout,
        compact_rope,
        text_features,
        image_latents[:, : -layout.target_tokens],
        text=models["text"],
        image=models["image"],
        gear=gear,
        layers=layers,
        heads=core.heads,
        head_dim=core.to_q.out_features // core.heads,
    )
    request.initialize_prefix()
    first_text = next((start, end) for start, end, role in request.segments if role == "text")
    graphs = {}
    for role in ("text", "image"):
        if role == "text":
            start, end = first_text
            inputs = request.arguments(request.text[:, start:end], start, 0.0, text=True)
        else:
            inputs = request.arguments(
                image_latents[:, -layout.target_tokens :], layout.prefix_tokens, timestep, text=False
            )
        directory = root / f"denoise_{role}_{gear.name}"
        directory.mkdir()
        path = directory / f"qwen_image21_denoise_{role}.onnx"
        names = ["tokens", "time_features", "cos", "sin", "past_seq_length", "current_input_length"]
        names += [f"cache_{i}" for i in range(2 * layers)]
        export_cpu_graph(
            models[role],
            inputs,
            path,
            names,
            ["hidden" if role == "text" else "noise"],
            target_device=target_device,
            quant_config=quant_config,
        )
        graphs[role] = path.relative_to(root).as_posix()
        del models[role]
        gc.collect()
        torch.cuda.empty_cache()
    meta = dict(
        schema_version=4,
        kind="qwen_image21_dit",
        input_dtype="float16",
        cache_dtype="float16",
        gear=asdict(gear),
        layers=layers,
        heads=core.heads,
        head_dim=core.to_q.out_features // core.heads,
        context_channels=text_features.shape[-1],
        latent_channels=image_latents.shape[-1],
        cache_tokens=gear.cache_tokens,
        cache_policy="inplace_replace_target",
        attention="flash_attention" if use_noncausal_flash_attention else "causal_flash_noncausal_masked",
        attention_metadata="lengths",
        sequence_layout="compact_prefix_target",
        graphs=graphs,
    )
    (root / "runtime_meta_info.json").write_text(json.dumps(meta, indent=2) + "\n")
    entries = {f"denoise_{name}": {"path": path, "type": "hmonnx"} for name, path in graphs.items()}
    entries["runtime_meta_info"] = {"path": "runtime_meta_info.json", "type": "json"}
    (root / "export_meta_info.json").write_text(json.dumps(entries, indent=2) + "\n")
    return meta


@torch.no_grad()
def export_dit_bundle(
    source,
    output_dir,
    *,
    gear: DenoiseGear,
    layout,
    compact_rope,
    text_features,
    image_latents,
    timestep,
    target_device,
    quant_config,
):
    """Export all DiT layers and frontends, keeping each SDK conversion isolated.

    Source must be a fresh floating transformer. image_latents contains references
    followed by the target. Caller supplies SDK device/config and calibration data;
    this function neither downloads weights nor fabricates a calibration prompt.
    """
    if not source.config.causal_condition:
        raise ValueError("prefix cache export requires causal_condition=True")
    if source.config.patch_size != 1 or not source.transformer_blocks:
        raise ValueError("export requires patch_size=1 and a nonempty DiT stack")
    if layout.prefix_capacity != gear.prefix_tokens or layout.target_tokens != gear.target_tokens:
        raise ValueError("calibration layout must match the requested gear")
    if text_features.dtype != torch.float32 or image_latents.dtype != torch.float32:
        raise ValueError("use FP32 calibration inputs and fresh FP32 weights")
    if any(parameter.dtype != torch.float32 for parameter in source.parameters()):
        raise ValueError("use fresh FP32 source weights; SDK graphs use independent copies")
    if any(parameter.device.type != "cpu" for parameter in source.parameters()):
        raise ValueError("DiT export keeps source weights on CPU")
    if text_features.device.type != "cpu" or image_latents.device.type != "cpu" or compact_rope.device.type != "cpu":
        raise ValueError("DiT export requires CPU calibration inputs")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    graphs = {}

    def export(role, module, inputs, input_names, output_names):
        component = f"denoise_{role}_{gear.name}"
        directory = root / component
        directory.mkdir()
        path = directory / f"qwen_image21_{component}.onnx"
        # Conversion mutates modules. Only the current graph is copied, not the full DiT.
        export_cpu_graph(
            module, inputs, path, input_names, output_names, target_device=target_device, quant_config=quant_config
        )
        graphs[role] = path.relative_to(root).as_posix()

    # Explicit square/reduce/rsqrt lowering loses accuracy on large VLM outliers.
    # Keep normalization as a dedicated SDK operator and fold zero-centered scale
    # in FP32 before any weight quantization.
    text_module, image_module = make_text_projection(source), ImageProjection(source)
    time_module, head_module = TimeConditioning(source), OutputHead(source)
    text = TokenProjection(text_module, capacity=gear.encoder_tokens, input_channels=text_features.shape[-1])(
        text_features
    )
    # Combined image latents may exceed a projection gear that only covers one call.
    references, target = image_latents[:, : -gear.target_tokens], image_latents[:, -gear.target_tokens :]
    project = TokenProjection(image_module, capacity=gear.image_tokens, input_channels=image_latents.shape[-1])
    images = torch.cat((project(references), project(target)), dim=1) if references.shape[1] else project(target)
    hidden = layout.assemble(text, images)
    cos, sin = [x.to(device=hidden.device, dtype=hidden.dtype) for x in layout.pad_rope(compact_rope)]
    mask = layout.attention_bias(device=hidden.device, dtype=hidden.dtype)
    time_features = timestep_features(timestep, device=hidden.device)
    parameters = prepare_conditioning(*time_module(time_features))
    expanded = [
        torch.cat(
            (p[1:2, None].expand(1, gear.prefix_tokens, -1), p[:1, None].expand(1, gear.target_tokens, -1)), dim=1
        )
        for p in parameters[:4]
    ]
    current = [p[:1, None] for p in parameters[:4]]
    names = ["hidden", "scale1", "gate1", "scale2", "gate2", "cos", "sin", "mask"]
    for index, block in enumerate(source.transformer_blocks):
        core = BlockCore(block)
        extract, cached = ExtractBlock(core, gear.prefix_tokens), CachedBlock(core)
        inputs = (hidden, *expanded, cos, sin, mask)
        next_hidden, key, value = extract(*inputs)
        export(f"extract_{index}", extract, inputs, names, ["hidden_out", "prefix_k", "prefix_v"])
        p = gear.prefix_tokens
        export(
            f"cached_{index}",
            cached,
            (hidden[:, p:], *current, cos[:, :, p:], sin[:, :, p:], mask[:, :, p:], key, value),
            names + ["prefix_k", "prefix_v"],
            ["hidden_out"],
        )
        hidden = next_hidden
    padded_text = text_features.new_zeros(1, gear.encoder_tokens, text_features.shape[-1])
    padded_text[:, : text_features.shape[1]] = text_features
    padded_images = image_latents.new_zeros(1, gear.image_tokens, image_latents.shape[-1])
    padded_images[:, : target.shape[1]] = target
    for role, module, inputs, ins, outs in (
        ("text", text_module, (padded_text,), ["features"], ["hidden"]),
        ("image", image_module, (padded_images,), ["latents"], ["hidden"]),
        ("time", time_module, (time_features,), ["features"], ["block_parameters", "output_parameters"]),
        ("head", head_module, (hidden[:, -gear.target_tokens :], parameters[4]), ["hidden", "scale"], ["noise"]),
    ):
        export(role, module, inputs, ins, outs)
    meta = {
        "schema_version": 1,
        "kind": "qwen_image21_dit",
        "input_dtype": "float16",
        "gear": asdict(gear),
        "layers": len(source.transformer_blocks),
        "heads": core.heads,
        "head_dim": core.head_dim,
        "context_channels": text_features.shape[-1],
        "latent_channels": image_latents.shape[-1],
        "graphs": graphs,
    }
    # Only completed bundles receive metadata; a failed export is never loadable.
    (root / "runtime_meta_info.json").write_text(json.dumps(meta, indent=2) + "\n")
    entries = {f"denoise_{name}": {"path": path, "type": "hmonnx"} for name, path in graphs.items()}
    entries["runtime_meta_info"] = {"path": "runtime_meta_info.json", "type": "json"}
    (root / "export_meta_info.json").write_text(json.dumps(entries, indent=2) + "\n")
    return meta
