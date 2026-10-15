# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Complete, relocatable Qwen-Image deployment bundle loading."""

import json

from .artifacts import bundle_path, read_dit_metadata
from .sampling import BASE_MODEL_ID, TURBO_MODEL_ID, validate_sampling_config


def read_bundle_metadata(root):
    meta = json.loads(bundle_path(root, "runtime_meta_info.json").read_text())
    if meta.get("schema_version") != 3 or meta.get("kind") != "qwen_image21":
        raise ValueError("unsupported Qwen-Image deployment metadata")
    if meta.get("target_device") != "XH2a" or meta.get("quant_type") not in ("w16a16_sefp", "w8a16_sefp", "w8a8_sefp"):
        raise ValueError("this deployment contract requires XH2a W16A16, W8A16 or W8A8 SEFP")
    dit, gear = read_dit_metadata(root, meta["dit_metadata"])
    if dit["schema_version"] != 4 or meta.get("attention") != "flash_attention" or meta.get("alignment") != 32:
        raise ValueError("capacity bundles require FlashAttention and 32-pixel alignment")
    from .build import validate_build_config
    from .capacity import round_capacity

    config = validate_build_config(meta["deployment_config"])
    feature = meta["feature"]
    for key in ("capacity", "max_position", "hidden_size"):
        if type(feature.get(key)) is not int or feature[key] <= 0:
            raise ValueError(f"invalid feature {key}")
    if feature["capacity"] > feature["max_position"] or feature["hidden_size"] != dit["context_channels"]:
        raise ValueError("feature dimensions disagree with the DiT or RoPE capacity")
    budget = meta["text_budget"]
    if any(
        type(budget.get(name)) is not int or budget[name] <= 0
        for name in ("template_tokens", "feature_capacity", "prefix_capacity", "dit_text_capacity")
    ):
        raise ValueError("invalid tokenizer-derived text budget")
    text = budget["template_tokens"] + config["prompt_budget"]
    expected_feature = round_capacity(text + config["max_references"] * config["visual_patch_capacity"] // 4)
    expected_prefix = round_capacity(text + config["max_references"] * config["reference_capacity"])
    if (
        feature["capacity"] != expected_feature
        or gear.encoder_tokens != expected_feature
        or budget["feature_capacity"] != expected_feature
        or gear.prefix_tokens != expected_prefix
        or budget["prefix_capacity"] != expected_prefix
        or gear.text_tokens != round_capacity(text)
        or budget["dit_text_capacity"] != round_capacity(text)
        or gear.target_tokens != config["target_capacity"]
        or gear.image_tokens != max(config["target_capacity"], config["reference_capacity"])
        or feature["max_position"] != config["max_position"]
        or meta["target_device"] != config["target_device"]
        or meta["quant_type"] != config["quant_type"]
    ):
        raise ValueError("component capacities disagree with deployment_config or tokenizer budget")
    bundle_path(root, feature["path"])
    visual = meta["visual"]
    if visual.get("patch_capacity") != config["visual_patch_capacity"]:
        raise ValueError("visual patch capacity disagrees with deployment_config")
    if any(type(visual.get(name)) is not int or visual[name] <= 0 for name in ("grid_side", "head_dim")):
        raise ValueError("invalid visual position geometry")
    if visual["head_dim"] % 4:
        raise ValueError("visual RoPE head dimension must be divisible by four")
    bundle_path(root, visual["path"])
    for role, limit in (("vae_encode", "reference_capacity"), ("vae_decode", "target_capacity")):
        entry = meta[role]
        if entry.get("canvas_size") != config["canvas_size"] or entry.get("token_capacity") != config[limit]:
            raise ValueError(f"{role} canvas or token capacity disagrees with deployment_config")
        if (
            entry.get("layout") != "NCHW"
            or entry.get("normalization") != "rmsnorm"
            or entry.get("normalization_eps") != 1e-8
        ):
            raise ValueError("VAE requires four-dimensional boundaries and RMSNorm with epsilon=1e-8")
        bundle_path(root, entry["path"])
    for key in ("embedding", "text_config", "vae_config", "scheduler_config", "transformer_config"):
        bundle_path(root, meta["resources"][key])
    model_id = meta.get("model_id", BASE_MODEL_ID)
    pipeline_path = meta["resources"].get("pipeline_config")
    if model_id == TURBO_MODEL_ID and pipeline_path is None:
        raise ValueError("Turbo bundle requires the pipeline_config sampling resource")
    if pipeline_path is not None:
        pipeline_config = json.loads(bundle_path(root, pipeline_path).read_text())
        scheduler_config = json.loads(bundle_path(root, meta["resources"]["scheduler_config"]).read_text())
        validate_sampling_config(pipeline_config, scheduler_config, model_id=model_id)
    elif model_id != BASE_MODEL_ID:
        raise ValueError(f"unsupported Qwen-Image model_id: {model_id}")
    files = meta["processor_files"]
    if not files or len(files) != len(set(files)):
        raise ValueError("processor resource files must be nonempty and unique")
    paths = [bundle_path(root, path) for path in files]
    if len({path.parent for path in paths}) != 1:
        raise ValueError("processor resources must share one directory")
    names = {path.name for path in paths}
    required = {"tokenizer_config.json", "tokenizer.json"}
    if not required.issubset(names) or not names.intersection({"preprocessor_config.json", "processor_config.json"}):
        raise ValueError("processor resources are incomplete")
    return meta


def load_runtime(root, *, device="cuda:0"):
    """Load only package resources and HMONNX graphs, without source HF weights."""
    meta = read_bundle_metadata(root)
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope
    from safetensors.torch import load_file
    from transformers import AutoProcessor

    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    from .artifacts import DenoiseBundle
    from .canvas_vae import CanvasVaeSession
    from .feature_graph import CapacityFeatureSession
    from .pipeline import ImageRuntime
    from .processor import PromptEncoder
    from .vision_graph import PackedVisionSession

    def resource(name):
        return bundle_path(root, meta["resources"][name])

    def config(name):
        return json.loads(resource(name).read_text())

    def session(path):
        return HMONNXInferenceV2(
            str(bundle_path(root, path)), session_config=HMONNXInferenceConfig(exec_devices=[device])
        )

    text_config, transformer_config = config("text_config"), config("transformer_config")
    if not transformer_config.get("causal_condition") or transformer_config.get("patch_size") != 1:
        raise ValueError("the deployment requires causal conditioning and unpatched latents")
    weights = load_file(str(resource("embedding")), device="cpu")
    if set(weights) != {"embedding.weight"}:
        raise ValueError("invalid token embedding resource")
    weight = weights["embedding.weight"]
    if tuple(weight.shape) != (text_config["text_config"]["vocab_size"], meta["feature"]["hidden_size"]):
        raise ValueError("token embedding shape disagrees with the model config")
    embedding = torch.nn.Embedding.from_pretrained(weight, freeze=True)
    processor_dir = bundle_path(root, meta["processor_files"][0]).parent
    processor = AutoProcessor.from_pretrained(processor_dir, local_files_only=True, trust_remote_code=False)
    feature = meta["feature"]
    capacity = meta["deployment_config"]
    visual = meta["visual"]
    encoder = PromptEncoder(
        processor,
        embedding,
        CapacityFeatureSession(
            session(feature["path"]), feature["capacity"], feature["hidden_size"], feature["max_position"], device
        ),
        PackedVisionSession(
            session(visual["path"]), visual["patch_capacity"], visual["grid_side"], visual["head_dim"], device=device
        ),
        prompt_budget=capacity["prompt_budget"],
        max_references=capacity["max_references"],
    )
    denoiser = DenoiseBundle(root, device=device, metadata=meta["dit_metadata"])
    rope = QwenImage21Rope(theta=10000, axes_dim=transformer_config["axes_dims_rope"])
    vae = {}
    for role in ("vae_encode", "vae_decode"):
        entry = meta[role]
        vae[role] = CanvasVaeSession(
            session(entry["path"]),
            entry["canvas_size"],
            encode=role == "vae_encode",
            token_capacity=entry["token_capacity"],
        )
    return ImageRuntime(
        encoder,
        denoiser,
        vae["vae_encode"],
        vae["vae_decode"],
        config("vae_config"),
        config("scheduler_config"),
        rope,
        capacity,
        sample_sigmas=(
            config("pipeline_config").get("sample_sigmas") if "pipeline_config" in meta["resources"] else None
        ),
    )
