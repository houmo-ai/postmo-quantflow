"""Offline native Qwen-Image weights and processor, retaining all six graph roles."""

from dataclasses import replace


def prepare_qwen_image21(source, root, catalog, recipe, monkeypatch):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from PIL import Image
    from test_qwen_image21_feature_encoder import tiny_config
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        Qwen3VLForConditionalGeneration(tiny_config()).eval().save_pretrained(source / "text_encoder")
        QwenImage21Transformer2DModel(
            patch_size=1,
            in_channels=4,
            out_channels=4,
            context_in_dim=256,
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=64,
            axes_dims_rope=(16, 24, 24),
            causal_condition=True,
        ).eval().save_pretrained(source / "transformer")
        AutoencoderKLQwenImage21(
            base_dim=4,
            decoder_base_dim=8,
            z_dim=4,
            num_res_blocks=1,
            latents_mean=[0] * 4,
            latents_std=[1] * 4,
        ).eval().save_pretrained(source / "vae")
    FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256,
        base_shift=0.5,
        max_image_seq_len=8192,
        max_shift=0.9,
        shift_terminal=0.02,
        use_dynamic_shifting=True,
    ).save_pretrained(source / "scheduler")
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
    for index in range(4, 60):
        vocab[f"token_{index}"] = index
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
        "{% for item in message['content'] %}{{ item['text'] }}{% endfor %}"
        "<|im_end|>\n{% endfor %}"
    )
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        chat_template=template,
        image_processor=Qwen2VLImageProcessor(
            patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3
        ),
        video_processor=Qwen3VLVideoProcessor(
            patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3
        ),
    )
    processor.save_pretrained(source / "processor")
    monkeypatch.setitem(
        catalog._recipes,
        recipe.model_id,
        replace(
            recipe,
            default_profile="output64x32-reference32x64",
            profiles=("output64x32-reference32x64",),
            artifact_name_template="hmquant_xh2_qwen_image21_{deployment_precision}_output64x32_reference32x64",
            config_replacements=(
                *recipe.config_replacements,
                ("export.output_size", [64, 32]),
                ("export.reference_size", [32, 64]),
                ("export.canvas_size", [96, 96]),
                ("export.target_capacity", 24),
                ("export.reference_capacity", 24),
                ("export.visual_patch_capacity", 24),
                ("export.prompt_budget", 32),
                ("export.max_position", 256),
            ),
        ),
    )
    image = root / "reference.png"
    Image.new("RGBA", (64, 32), (160, 80, 40, 192)).save(image)
    return {"prompt": "A colorful object.", "steps": 3, "seed": 7}
