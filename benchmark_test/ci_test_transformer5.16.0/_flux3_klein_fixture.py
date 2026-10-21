"""Small real FLUX/Qwen checkpoint retaining all feature layers and graph roles."""

import json
from dataclasses import replace


def prepare_source(root, monkeypatch):
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    from xhmodel_merak.model_catalog.registry import get_catalog
    from xhmodel_merak.xh_other_model.models.flux3_klein.native_autoencoder import AutoEncoder
    from xhmodel_merak.xh_other_model.models.flux3_klein.native_dit import DiT

    root.mkdir()
    config = dict(
        format_version=1,
        model="flux-3-image-klein",
        quantization=None,
        torch_dtype="bfloat16",
        dit=dict(
            hidden=128,
            heads=1,
            depth=1,
            depth_single=2,
            mlp_ratio=3.0,
            axes_dim=[32] * 4,
            theta=10000,
            image_rope_scale=0.5,
            in_channels=dict(image=128, image_cond=128, text=1024),
            out_modes=["image"],
            flow_map=True,
        ),
        text_encoder=dict(
            path="text_encoder",
            tokenizer="text_encoder",
            layers=[4, 8, 12, 16, 20, 24, 28, 32],
            pad_multiple=80,
            max_tokens=8192,
        ),
        vae=dict(
            path="ae.safetensors",
            architecture=dict(base_channels=64, channel_multipliers=[1, 1, 1, 1], residual_blocks=1),
        ),
        sampler=dict(steps=16, guidance=None, alpha=6.93, size=[1024, 1024]),
    )
    (root / "config.json").write_text(json.dumps(config))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        save_file(DiT(True, config["dit"]).eval().state_dict(), str(root / "model.safetensors"))
        save_file(AutoEncoder(config["vae"]["architecture"]).eval().state_dict(), str(root / "ae.safetensors"))
        text_config = Qwen3VLConfig(
            text_config=dict(
                vocab_size=64,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=36,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=64,
                rope_parameters={"rope_type": "default", "rope_theta": 5000000.0, "mrope_section": [12, 10, 10]},
            ),
            vision_config=dict(
                depth=3,
                hidden_size=32,
                intermediate_size=64,
                num_heads=2,
                out_hidden_size=128,
                deepstack_visual_indexes=[0, 1, 2],
            ),
        )
        Qwen3VLForConditionalGeneration(text_config).eval().save_pretrained(root / "text_encoder", max_shard_size="1MB")
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<|im_end|>": 2,
        "<|im_start|>": 3,
        "<|image_pad|>": 60,
        "<|video_pad|>": 61,
        "<|vision_start|>": 62,
        "<|vision_end|>": 63,
    }
    vocab.update({f"token_{i}": i for i in range(4, 60)})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        additional_special_tokens=[
            "<|im_start|>",
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        ],
    )
    template = (
        "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
        "{% if message['content'] is string %}{{ message['content'] }}{% else %}"
        "{% for item in message['content'] %}{{ item['text'] }}{% endfor %}"
        "{% endif %}<|im_end|>\n{% endfor %}"
    )
    Qwen3VLProcessor(
        tokenizer=tokenizer,
        chat_template=template,
        image_processor=Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2),
    ).save_pretrained(root / "text_encoder")
    catalog = get_catalog()
    recipe = catalog._recipes["flux-3-image-klein"]
    replacements = {
        "output_size": [32, 64],
        "reference_size": [32, 64],
        "canvas_size": [64, 64],
        "target_capacity": 16,
        "reference_capacity": 16,
        "reference_total_capacity": 32,
        "text_capacity": 80,
        "max_references": 2,
        "single_reference_max_pixels": 4096,
        "multi_reference_max_pixels": 4096,
    }
    monkeypatch.setitem(
        catalog._recipes,
        recipe.model_id,
        replace(recipe, config_replacements=tuple((f"export.{key}", value) for key, value in replacements.items())),
    )
    return config
