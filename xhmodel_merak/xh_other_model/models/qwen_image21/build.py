# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Export all learned components and package the resources used by ImageRuntime."""

import gc
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .sampling import BASE_MODEL_ID, MODEL_IDS, read_source_sampling


def deployment_quant_type(config):
    """Resolve the public model QuantScheme and the older workflow export field."""
    model = config.get("model", {})
    scheme = model.get("quant_scheme") if isinstance(model, Mapping) else None
    if scheme is not None:
        if not isinstance(scheme, Mapping) or scheme.get("target_device") != config.get("target_device"):
            raise ValueError("model.quant_scheme target_device disagrees with export.target_device")
        quant_type = scheme.get("quant_type")
        if config.get("quant_type", quant_type) != quant_type:
            raise ValueError("model.quant_scheme quant_type disagrees with export.quant_type")
    else:
        quant_type = config.get("quant_type")
    return quant_type


def validate_build_config(config):
    from .capacity import validate_capacity_config

    if config.get("target_device") != "XH2a" or deployment_quant_type(config) not in (
        "w16a16_sefp",
        "w8a16_sefp",
        "w8a8_sefp",
    ):
        raise ValueError("Qwen-Image-2.1 requires XH2a W16A16, W8A16 or W8A8 SEFP")
    if "feature_capacity" in config or "prefix_capacity" in config:
        raise ValueError("text capacities are derived from the tokenizer; configure prompt_budget/max_references")
    if config.get("model_id", BASE_MODEL_ID) not in MODEL_IDS:
        raise ValueError("unsupported Qwen-Image model_id")
    return validate_capacity_config(config)


def export_entries(root, meta):
    """Path/type entries consumed by the shared publisher, including every runtime JSON."""
    from .artifacts import read_dit_metadata

    dit, _ = read_dit_metadata(root, meta["dit_metadata"])
    entries = {f"denoise_{name}": {"path": path, "type": "hmonnx"} for name, path in dit["graphs"].items()}
    entries["text_encode"] = {"path": meta["feature"]["path"], "type": "hmonnx"}
    for role in ("visual", "vae_encode", "vae_decode"):
        entry = meta[role]
        size = f"p{entry['patch_capacity']}" if role == "visual" else "c" + "x".join(map(str, entry["canvas_size"]))
        entries[f"{role}_{size}"] = {"path": entry["path"], "type": "hmonnx"}
    for name, path in meta["resources"].items():
        entries[name] = {"path": path, "type": "embedding" if name == "embedding" else "json"}
    entries["tokenizer"] = {"path": str(Path(meta["processor_files"][0]).parent), "type": "tokenizer"}
    for index, path in enumerate(meta["processor_files"]):
        if path.endswith(".json"):
            entries[f"processor_{index}"] = {"path": path, "type": "json"}
    for name in ("runtime_meta_info.json", meta["dit_metadata"]):
        entries[Path(name).stem] = {"path": name, "type": "json"}
    return entries


def build_bundle(model_dir, output_dir, config, *, device="cuda:0"):
    config = validate_build_config(config)
    import torch

    # Conv3d -> single-frame Conv2d equivalence must be calibrated in FP32.
    # Preserve the caller's process settings, including exceptional exits.
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return _build_bundle(model_dir, output_dir, config, device=device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _build_bundle(model_dir, output_dir, config, *, device):
    import numpy as np
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from PIL import Image
    from safetensors.torch import save_file
    from transformers import AutoProcessor

    from xhquant.api import (
        ConfigDict,
        DeviceType,
        QuantScheme,
        create_quant_config,
    )

    from .artifacts import DenoiseGear
    from .bundle import read_bundle_metadata
    from .canvas_vae import VAE_INPUT_NAMES, CanvasVaeDecode, CanvasVaeEncode, spatial_inputs
    from .capacity import DEFAULT_CAPACITY, derive_text_capacities
    from .export import export_cpu_graph, export_full_dit_bundle
    from .feature_graph import (
        CAPACITY_FEATURE_INPUT_NAMES,
        CapacityFeatureSession,
        CapacityTextFeatures,
        load_feature_encoder,
    )
    from .layout import prepare_layout
    from .processor import PromptEncoder
    from .vision_graph import VISION_INPUT_NAMES, PackedVision, PackedVisionSession, geometry_inputs

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Qwen-Image weight PTQ requires a CUDA device; source/trace/export stay on CPU")
    torch.cuda.set_device(device)
    source_dir, root = Path(model_dir), Path(output_dir)
    source_config = json.loads((source_dir / "transformer/config.json").read_text())
    if source_config.get("_class_name") != "QwenImage21Transformer2DModel":
        raise ValueError("source must be a Qwen-Image-2.1 checkpoint")
    model_id = config.get("model_id", BASE_MODEL_ID)
    read_source_sampling(source_dir, model_id=model_id)
    root.mkdir(parents=True, exist_ok=False)
    height, width = config["output_size"]
    ref_height, ref_width = config["reference_size"]
    max_position = config["max_position"]
    quant_type = deployment_quant_type(config)
    quant = ConfigDict(create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)))
    quant.auto_release_unused_parameters = True
    meta = dict(
        schema_version=3,
        kind="qwen_image21",
        model_id=model_id,
        target_device="XH2a",
        quant_type=quant_type,
        attention="flash_attention",
        alignment=32,
        dit_metadata="dit_meta_info.json",
        resources={},
        processor_files=[],
    )
    meta["deployment_config"] = {key: config[key] for key in DEFAULT_CAPACITY}
    meta["deployment_config"].update(target_device="XH2a", quant_type=quant_type)

    def resource(name, relative):
        destination = root / "hf_config" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_dir / relative, destination)
        meta["resources"][name] = destination.relative_to(root).as_posix()

    for name, relative in (
        ("text_config", "text_encoder/config.json"),
        ("vae_config", "vae/config.json"),
        ("transformer_config", "transformer/config.json"),
        ("scheduler_config", "scheduler/scheduler_config.json"),
    ):
        resource(name, relative)
    if (source_dir / "model_index.json").is_file():
        resource("pipeline_config", "model_index.json")
    processor_dir = root / "hf_config/processor"
    processor_dir.mkdir()
    for source in sorted((source_dir / "processor").iterdir()):
        if source.is_file() and source.suffix in (".json", ".txt", ".jinja", ".model", ".tiktoken"):
            target = processor_dir / source.name
            shutil.copyfile(source, target)
            meta["processor_files"].append(target.relative_to(root).as_posix())

    def graph_path(role, size):
        folder = root / f"{role}_{size}"
        folder.mkdir()
        return folder / f"qwen_image21_{role}_{size}.onnx"

    def export_graph(module, inputs, path, names, *, dynamo=False, outputs=None):
        export_cpu_graph(
            module,
            inputs,
            path,
            names,
            outputs or ["output_features"],
            target_device=DeviceType.XH2a,
            quant_config=quant,
            dynamo=dynamo,
        )

    def release():
        gc.collect()
        torch.cuda.empty_cache()

    with torch.no_grad(), tempfile.TemporaryDirectory(prefix="qwen_image21_", dir=root) as temporary:
        temporary = Path(temporary)
        source = load_feature_encoder(source_dir / "text_encoder")
        embedding = source.model.language_model.embed_tokens
        save_file(
            {"embedding.weight": embedding.weight.detach().cpu().contiguous()}, root / "token_embedding.safetensors"
        )
        meta["resources"]["embedding"] = "token_embedding.safetensors"
        hidden = source.config.text_config.hidden_size
        processor = AutoProcessor.from_pretrained(processor_dir, local_files_only=True)
        text_capacities = derive_text_capacities(processor, config)
        capacity, prefix_capacity = text_capacities["feature_capacity"], text_capacities["prefix_capacity"]
        if capacity > max_position:
            raise ValueError("max_position must cover the automatically derived text capacity")
        meta["text_budget"] = text_capacities
        visual = PackedVision(source.model.visual).eval()
        patch_capacity = config["visual_patch_capacity"]
        grid_side = source.model.visual.num_grid_per_side
        head_dim = source.config.vision_config.hidden_size // source.config.vision_config.num_heads
        geometry = geometry_inputs(
            (ref_height // 16, ref_width // 16), patch_capacity, grid_side, head_dim, device="cpu", dtype=torch.float32
        )
        patches = torch.zeros(1, patch_capacity, 1536)
        visual_names = ["image_embeds", "deepstack_0", "deepstack_1", "deepstack_2"]
        path = graph_path("visual", f"p{patch_capacity}")
        export_graph(visual, (patches, *geometry), path, VISION_INPUT_NAMES, outputs=visual_names)
        meta["visual"] = dict(
            patch_capacity=patch_capacity,
            grid_side=grid_side,
            head_dim=head_dim,
            path=path.relative_to(root).as_posix(),
        )
        feature_boundary = CapacityFeatureSession(None, capacity, hidden, max_position, "cpu")
        vision_session = PackedVisionSession(
            visual, patch_capacity, grid_side, head_dim, device="cpu", dtype=torch.float32
        )
        encoder = PromptEncoder(
            processor,
            embedding,
            feature_boundary,
            vision_session,
            prompt_budget=config["prompt_budget"],
            max_references=config["max_references"],
        )
        reference_image = Image.new("RGBA", (ref_width, ref_height), (255, 255, 255, 255))
        prepared = encoder.prepare("A colorful object on a table.", [reference_image])
        arguments = tuple(
            x.float() if x.is_floating_point() else x
            for x in feature_boundary.prepare_inputs(prepared.embeddings, prepared.positions, prepared.deepstack)
        )
        features = CapacityTextFeatures(source.model.language_model, capacity, max_position).eval()
        text_features = (
            features(*arguments)[:, prepared.drop_tokens : prepared.embeddings.shape[1]].float().cpu().clone()
        )
        path = graph_path("text_encode", f"t{capacity}")
        export_graph(features, arguments, path, CAPACITY_FEATURE_INPUT_NAMES)
        meta["feature"] = dict(
            capacity=capacity, max_position=max_position, hidden_size=hidden, path=path.relative_to(root).as_posix()
        )
        image_mask = prepared.image_mask[:, prepared.drop_tokens :]
        del source, embedding, visual, encoder, features, arguments, prepared, vision_session, patches, geometry
        release()

        vae = AutoencoderKLQwenImage21.from_pretrained(
            source_dir / "vae", torch_dtype=torch.float32, local_files_only=True
        ).eval()
        mean = torch.tensor(vae.config.latents_mean).reshape(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).reshape(1, -1, 1, 1, 1)
        rgba = torch.from_numpy(np.asarray(reference_image).copy()).permute(2, 0, 1)[None, :, None].float()
        rgba = rgba / 127.5 - 1
        references = (vae.encode(rgba, return_dict=False)[0].mode() - mean) / std
        references = references.flatten(2).transpose(1, 2).clone()
        target = torch.randn(
            (1, mean.shape[1], 1, height // 16, width // 16), generator=torch.Generator(device="cpu").manual_seed(42)
        )
        canvas_h, canvas_w = config["canvas_size"]
        for role, module, value, size, token_capacity in (
            ("vae_encode", CanvasVaeEncode(vae), rgba, (ref_height, ref_width), config["reference_capacity"]),
            ("vae_decode", CanvasVaeDecode(vae), target * std + mean, (height, width), config["target_capacity"]),
        ):
            factor = 1 if role == "vae_encode" else 16
            padded = torch.nn.functional.pad(
                value.squeeze(2), (0, (canvas_w - size[1]) // factor, 0, (canvas_h - size[0]) // factor)
            )
            inputs = (padded, *spatial_inputs(size, config["canvas_size"], dtype=torch.float32))
            path = graph_path(role, f"c{canvas_h}x{canvas_w}")
            export_graph(module, inputs, path, VAE_INPUT_NAMES, dynamo=True)
            meta[role] = dict(
                canvas_size=list(config["canvas_size"]),
                token_capacity=token_capacity,
                layout="NCHW",
                normalization="rmsnorm",
                normalization_eps=1e-8,
                path=path.relative_to(root).as_posix(),
            )
        target = target.flatten(2).transpose(1, 2).clone()
        del vae, module, inputs, rgba, mean, std, value, padded
        release()

        transformer = QwenImage21Transformer2DModel.from_pretrained(
            source_dir / "transformer",
            torch_dtype=torch.float32,
            local_files_only=True,
        ).eval()
        shapes = [(1, ref_height // 16, ref_width // 16), (1, height // 16, width // 16)]
        layout = prepare_layout(
            image_mask, shapes, prefix_capacity=prefix_capacity, target_capacity=config["target_capacity"]
        )
        rope = transformer.pos_embed(shapes, layout.compact_image_mask, device="cpu")
        dit_root = temporary / "dit"
        export_full_dit_bundle(
            transformer,
            dit_root,
            gear=DenoiseGear(
                capacity,
                max(config["reference_capacity"], config["target_capacity"]),
                prefix_capacity,
                config["target_capacity"],
                text_capacities["dit_text_capacity"],
            ),
            layout=layout,
            compact_rope=rope,
            text_features=text_features,
            image_latents=torch.cat((references, target), dim=1),
            timestep=1.0,
            target_device=DeviceType.XH2a,
            quant_config=quant,
        )
        for entry in dit_root.iterdir():
            if entry.is_dir():
                entry.rename(root / entry.name)
        shutil.copyfile(dit_root / "runtime_meta_info.json", root / meta["dit_metadata"])
    # Incomplete exports never acquire a full runtime metadata file.
    (root / "runtime_meta_info.json").write_text(json.dumps(meta, indent=2) + "\n")
    read_bundle_metadata(root)
    (root / "export_meta_info.json").write_text(json.dumps(export_entries(root, meta), indent=2) + "\n")
    return meta
