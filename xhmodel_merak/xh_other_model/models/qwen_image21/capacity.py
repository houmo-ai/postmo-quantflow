# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Host capacity contracts for the default 1MP deployment profile."""

from math import ceil, sqrt


VAE_SPATIAL_SCALE = 16
VAE_TOKEN_AREA = VAE_SPATIAL_SCALE**2


def vae_token_capacity(size):
    """Count spatial16 VAE tokens without silently truncating invalid geometry."""
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or any(type(n) is not int or n <= 0 or n % 32 for n in size)
    ):
        raise ValueError("height/width must be positive multiples of 32")
    return size[0] * size[1] // VAE_TOKEN_AREA


DEFAULT_CAPACITY = {
    "output_size": [1024, 1024],
    "reference_size": [1024, 1024],
    "canvas_size": [1536, 1536],
    "target_capacity": 4352,
    "reference_capacity": 4352,
    "visual_patch_capacity": 4352,
    "prompt_budget": 512,
    "max_references": 1,
    "max_position": 32768,
}


def round_capacity(value, block=64):
    return ceil(value / block) * block


def validate_capacity_config(config):
    result = {**DEFAULT_CAPACITY, **config}
    for name in ("output_size", "reference_size", "canvas_size"):
        size = result[name]
        try:
            vae_token_capacity(size)
        except ValueError as error:
            raise ValueError(f"{name} must contain positive height/width multiples of 32") from error
    for name in (
        "target_capacity",
        "reference_capacity",
        "visual_patch_capacity",
        "prompt_budget",
        "max_references",
        "max_position",
    ):
        if type(result[name]) is not int or result[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if result["visual_patch_capacity"] % 4:
        raise ValueError("visual patch capacity must support the 2x2 merge")
    if result["reference_capacity"] > result["visual_patch_capacity"]:
        raise ValueError("reference latent capacity exceeds the visual patch capacity")
    for name, capacity in (("output_size", "target_capacity"), ("reference_size", "reference_capacity")):
        validate_size(result[name], result["canvas_size"], result[capacity])
    return result


def validate_size(actual_hw, canvas_hw, token_capacity):
    actual_capacity = vae_token_capacity(actual_hw)
    vae_token_capacity(canvas_hw)
    if any(n > cap for n, cap in zip(actual_hw, canvas_hw, strict=True)):
        raise ValueError("height/width exceeds the published spatial canvas")
    if actual_capacity > token_capacity:
        raise ValueError("image exceeds the published token capacity")


def derive_text_capacities(processor, config):
    """Measure the actual tokenizer template and reserve each image's visual slots."""
    from .processor import PromptEncoder

    count = config["max_references"]
    images = " ".join(f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(count))
    text = (
        f"<|im_start|>system\n{PromptEncoder.SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{images} <|im_end|>\n<|im_start|>assistant\n"
    )
    ids = processor.tokenizer(text, add_special_tokens=False).input_ids
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    template_tokens = len(ids) - ids.count(image_id)
    visual_tokens = count * (config["visual_patch_capacity"] // 4)
    text_budget = template_tokens + config["prompt_budget"]
    # Keep the entire measured template in the upper bound; runtime removes
    # its system prefix only after the complete causal feature pass.
    # TCIM 1.5.0 uses a 256-token causal query tile. Align physical Q capacity;
    # runtime still passes real lengths, and prefix/cache budgets stay unchanged.
    return {
        "template_tokens": template_tokens,
        "feature_capacity": round_capacity(text_budget + visual_tokens, block=256),
        "dit_text_capacity": round_capacity(text_budget, block=256),
        "prefix_capacity": round_capacity(text_budget + count * config["reference_capacity"]),
    }


def fit_reference_size(height, width, canvas_hw, token_capacity):
    """Preserve aspect ratio approximately, fit capacity, and align both axes to 32."""
    if height <= 0 or width <= 0:
        raise ValueError("reference image dimensions must be positive")
    scale = min(
        1.0, sqrt(token_capacity * VAE_TOKEN_AREA / (height * width)), canvas_hw[0] / height, canvas_hw[1] / width
    )
    actual = [max(32, int(n * scale) // 32 * 32) for n in (height, width)]
    # Minimum-width alignment can increase area for extreme aspect ratios.
    if vae_token_capacity(actual) > token_capacity:
        axis = 0 if actual[0] >= actual[1] else 1
        actual[axis] = token_capacity * VAE_TOKEN_AREA // actual[1 - axis] // 32 * 32
    validate_size(actual, canvas_hw, token_capacity)
    return tuple(actual)
