"""Capacity metadata fixtures for package contracts, without learned-graph claims."""


def bundle_metadata(quant_type):
    from xhmodel_merak.xh_other_model.models.qwen_image21.capacity import DEFAULT_CAPACITY

    config = {**DEFAULT_CAPACITY, "target_device": "XH2a", "quant_type": quant_type}
    graphs = {role: f"denoise_{role}/model.onnx" for role in ("text", "image")}
    dit = dict(
        schema_version=4,
        kind="qwen_image21_dit",
        input_dtype="float16",
        gear=dict(encoder_tokens=1664, image_tokens=4352, prefix_tokens=4928, target_tokens=4352, text_tokens=576),
        layers=1,
        heads=2,
        head_dim=64,
        context_channels=96,
        latent_channels=64,
        cache_policy="inplace_replace_target",
        cache_tokens=9280,
        attention="flash_attention",
        attention_metadata="lengths",
        sequence_layout="compact_prefix_target",
        graphs=graphs,
    )
    resources = {
        name: f"resources/{name}.json"
        for name in ("text_config", "vae_config", "scheduler_config", "transformer_config")
    }
    resources["embedding"] = "resources/embedding.safetensors"
    processor = [f"processor/{name}.json" for name in ("preprocessor_config", "tokenizer_config", "tokenizer")]
    meta = dict(
        schema_version=3,
        kind="qwen_image21",
        target_device="XH2a",
        quant_type=quant_type,
        attention="flash_attention",
        alignment=32,
        deployment_config=config,
        text_budget=dict(template_tokens=32, feature_capacity=1664, prefix_capacity=4928, dit_text_capacity=576),
        dit_metadata="dit_meta_info.json",
        resources=resources,
        processor_files=processor,
        feature=dict(capacity=1664, max_position=32768, hidden_size=96, path="text_encode/model.onnx"),
        visual=dict(patch_capacity=4352, grid_side=48, head_dim=72, path="visual/model.onnx"),
        vae_encode=dict(
            canvas_size=[1536, 1536],
            token_capacity=4352,
            layout="NCHW",
            normalization="rmsnorm",
            normalization_eps=1e-8,
            path="vae_encode/model.onnx",
        ),
        vae_decode=dict(
            canvas_size=[1536, 1536],
            token_capacity=4352,
            layout="NCHW",
            normalization="rmsnorm",
            normalization_eps=1e-8,
            path="vae_decode/model.onnx",
        ),
    )
    paths = [
        *graphs.values(),
        *resources.values(),
        *processor,
        meta["feature"]["path"],
        *(meta[role]["path"] for role in ("visual", "vae_encode", "vae_decode")),
    ]
    return meta, dit, paths
