# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Sequential four-component export and complete runtime resource packaging."""

import gc
import json
from pathlib import Path

from .capacity import validate_capacity


def validate_build_config(config):
    scheme = config.get("model", {}).get("quant_scheme", {})
    if config.get("target_device") != "XH2a" or scheme.get("target_device") != "XH2a":
        raise ValueError("FLUX 3 requires an explicit XH2a deployment scheme")
    if scheme.get("quant_type") not in ("w16a16_sefp", "w8a16_sefp", "w8a8_sefp"):
        raise ValueError("FLUX 3 supports W16A16, W8A16 or W8A8 SEFP")
    return validate_capacity(config)


def build_bundle(model_dir, output_dir, config, *, device="cuda:0"):
    import tempfile

    from .upsampler_build import attach_component, build_component, read_component, reuse_image_bundle, validate_source

    if config.get("prompt_upsampler") and config.get("upsampler_bundle"):
        raise ValueError("choose raw prompt_upsampler or upsampler_bundle")
    if config.get("prompt_upsampler"):
        validate_source(config["prompt_upsampler"])
    elif config.get("upsampler_bundle"):
        read_component(config["upsampler_bundle"])
    if config.get("image_bundle"):
        meta = reuse_image_bundle(config["image_bundle"], output_dir, config, model_dir=model_dir, device=device)
    else:
        meta = build_image_bundle(model_dir, output_dir, config, device=device)
    root = Path(output_dir)
    if config.get("prompt_upsampler"):
        with tempfile.TemporaryDirectory(prefix=".pup-component-", dir=root.parent) as temporary:
            component_root = Path(temporary) / "component"
            build_component(config["prompt_upsampler"], component_root, device=device)
            meta["prompt_upsampler"] = attach_component(root, component_root)
    elif config.get("upsampler_bundle"):
        meta["prompt_upsampler"] = attach_component(root, config["upsampler_bundle"])
    from .bundle import export_entries, read_metadata

    (root / "runtime_meta_info.json").write_text(json.dumps(meta, indent=2) + "\n")
    read_metadata(root)
    (root / "export_meta_info.json").write_text(json.dumps(export_entries(meta), indent=2) + "\n")
    return meta


def rebuild_denoise(model_dir, output_dir, source_bundle, capacity, quant_type, *, device):
    """Reuse the deployed text/VAE conditioning and export only the new DiT gear."""
    import torch
    from PIL import Image

    from xhquant.api import QuantScheme, create_quant_config

    from .bundle import load_runtime
    from .dit_graph import DIT_INPUT_NAMES, Denoiser, denoise_inputs
    from .export import export_graph
    from .source import load_native

    matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.no_grad():
            runtime = load_runtime(source_bundle, device=device)
            text = runtime.encode_text("A red teapot beside a blue ceramic cup on a wooden table.").float().cpu()
            rh, rw = capacity["reference_size"]
            count = min(capacity["max_references"], capacity["reference_total_capacity"] // (rh * rw // 256))
            colors = ((160, 90, 40), (40, 90, 160))
            references = torch.cat(
                [
                    runtime.encode_reference(Image.new("RGB", (rw, rh), colors[i % 2])).float().cpu()
                    for i in range(count)
                ],
                1,
            )
            del runtime
            gc.collect()
            torch.cuda.empty_cache()
            th, tw = capacity["output_size"]
            target = torch.randn(1, th * tw // 256, 128, generator=torch.Generator().manual_seed(42))
            capacities = (capacity["target_capacity"], capacity["reference_total_capacity"], capacity["text_capacity"])
            native = load_native(model_dir, "dit")
            graph = Denoiser(native, capacities).eval()
            del native
            gc.collect()
            values = denoise_inputs(target, references, text, (th, tw), [(rh, rw)] * count, capacities, 1.0, 0.978)
            path = Path(output_dir) / "denoise/flux3_klein_denoise.onnx"
            path.parent.mkdir()
            print(f"Re-exporting denoise: {path}", flush=True)
            quant = create_quant_config(QuantScheme(target_device="XH2a", quant_type=quant_type))
            export_graph(graph, values, path, DIT_INPUT_NAMES, quant, device=device)
            return path.relative_to(output_dir).as_posix()
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul_tf32, cudnn_tf32
        gc.collect()
        torch.cuda.empty_cache()


def build_image_bundle(model_dir, output_dir, config, *, device="cuda:0"):
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    from xhquant.api import QuantScheme, create_quant_config

    from .bundle import export_entries, read_metadata
    from .dit_graph import DIT_INPUT_NAMES, Denoiser, denoise_inputs
    from .export import export_graph
    from .processor import text_inputs
    from .source import MODEL_ID, load_native, read_source
    from .text_graph import TEXT_INPUT_NAMES, TextFeatures, feature_inputs
    from .vae_graph import VAE_INPUT_NAMES, CanvasVAE, spatial_inputs

    capacity = validate_build_config(config)
    source_config, text_config = read_source(model_dir)
    root, source = Path(output_dir), Path(model_dir)
    root.mkdir(parents=True, exist_ok=False)
    resources = root / "resources"
    resources.mkdir()
    quant_type = config["model"]["quant_scheme"]["quant_type"]
    quant = create_quant_config(QuantScheme(target_device="XH2a", quant_type=quant_type))
    meta = dict(
        schema_version=1,
        kind="flux3_klein",
        model_id=MODEL_ID,
        quant_type=quant_type,
        capacity=capacity,
        graphs={},
        resources={},
        processor_files=[],
        input_dtype="float16",
        reference_revision="e53f4ea697fe45db8f38fca91a90f5623f7f8f37",
    )
    for name, value in (("source_config", source_config), ("text_config", text_config)):
        path = resources / f"{name}.json"
        path.write_text(json.dumps(value, indent=2) + "\n")
        meta["resources"][name] = path.relative_to(root).as_posix()
    processor = AutoProcessor.from_pretrained(source / "text_encoder", local_files_only=True, trust_remote_code=False)
    processor.tokenizer.padding_side = "right"
    processor.save_pretrained(resources / "processor")
    meta["processor_files"] = sorted(
        path.relative_to(root).as_posix() for path in (resources / "processor").iterdir() if path.is_file()
    )

    def release():
        gc.collect()
        torch.cuda.empty_cache()

    def export(role, module, values, names):
        directory = root / (
            role + (f"_c{capacity['canvas_size'][0]}x{capacity['canvas_size'][1]}" if role.startswith("vae") else "")
        )
        directory.mkdir()
        path = directory / f"flux3_klein_{role}.onnx"
        print(f"Exporting {role}: {path}", flush=True)
        export_graph(module, values, path, names, quant, device=device, dynamo=role.startswith("vae"))
        meta["graphs"][role] = path.relative_to(root).as_posix()

    matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.no_grad():
            native = Qwen3VLForConditionalGeneration.from_pretrained(
                source / "text_encoder", dtype=torch.float32, device_map="cpu", local_files_only=True
            ).eval()
            embedding = native.model.language_model.embed_tokens
            save_file(
                {"embedding.weight": embedding.weight.detach().cpu().contiguous()},
                str(resources / "token_embedding.safetensors"),
            )
            meta["resources"]["embedding"] = "resources/token_embedding.safetensors"
            ids, positions, real, length = text_inputs(
                processor,
                "A red ceramic teapot rests on a wooden table beside a window, "
                "soft natural light, realistic photograph.",
                capacity["text_capacity"],
            )
            arguments = feature_inputs(embedding(ids).float(), positions, real, length)
            graph = TextFeatures(native.model.language_model).eval()
            del native, embedding
            release()
            text = graph(*arguments)[:, :length].float().clone()
            export("text_encode", graph, arguments, TEXT_INPUT_NAMES)
            del graph, arguments, ids, positions
            release()

            vae = load_native(source, "vae")
            rh, rw = capacity["reference_size"]
            th, tw = capacity["output_size"]
            ch, cw = capacity["canvas_size"]
            image = Image.new("RGB", (rw, rh), (160, 90, 40))
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
            graph = CanvasVAE(vae, encode=True).eval()
            values = (torch.nn.functional.pad(pixels, (0, cw - rw, 0, ch - rh)), *spatial_inputs((rh, rw), (ch, cw)))
            latent = graph(*values)[:, :, : rh // 16, : rw // 16].flatten(2).transpose(1, 2).contiguous()
            export("vae_encode", graph, values, VAE_INPUT_NAMES)
            del graph, values
            release()
            target = torch.randn(1, 128, th // 16, tw // 16, generator=torch.Generator().manual_seed(42))
            graph = CanvasVAE(vae, encode=False).eval()
            values = (
                torch.nn.functional.pad(target, (0, (cw - tw) // 16, 0, (ch - th) // 16)),
                *spatial_inputs((th, tw), (ch, cw)),
            )
            export("vae_decode", graph, values, VAE_INPUT_NAMES)
            target = target.flatten(2).transpose(1, 2).contiguous()
            del graph, values, vae
            release()

            native = load_native(source, "dit")
            capacities = (capacity["target_capacity"], capacity["reference_total_capacity"], capacity["text_capacity"])
            graph = Denoiser(native, capacities).eval()
            del native
            release()
            count = min(capacity["max_references"], capacity["reference_total_capacity"] // latent.shape[1])
            values = denoise_inputs(
                target, latent.repeat(1, count, 1), text, (th, tw), [(rh, rw)] * count, capacities, 1.0, 0.978
            )
            export("denoise", graph, values, DIT_INPUT_NAMES)
            del graph, values
            release()
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul_tf32, cudnn_tf32
    (root / "runtime_meta_info.json").write_text(json.dumps(meta, indent=2) + "\n")
    read_metadata(root)
    (root / "export_meta_info.json").write_text(json.dumps(export_entries(meta), indent=2) + "\n")
    return meta
