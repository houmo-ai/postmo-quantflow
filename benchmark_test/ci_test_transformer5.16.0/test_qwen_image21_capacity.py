"""1MP size bounds and automatic template/visual/text capacity accounting."""

import pytest

from xhmodel_merak.xh_other_model.models.qwen_image21.capacity import (
    derive_text_capacities,
    fit_reference_size,
    validate_capacity_config,
    validate_size,
)


@pytest.mark.parametrize("size", [(1024, 1024), (1024, 1088), (768, 1376), (1376, 768), (672, 1536), (1536, 672)])
def test_one_mp_profile_covers_common_aspect_ratios(size):
    config = validate_capacity_config({})
    validate_size(size, config["canvas_size"], config["target_capacity"])
    assert fit_reference_size(*size, config["canvas_size"], config["reference_capacity"]) == size


@pytest.mark.parametrize("size", [(1056, 1056), (768, 2048), (0, 32), (48, 64), (1536, 1536)])
def test_capacity_and_axis_bounds_are_both_checked(size):
    config = validate_capacity_config({})
    with pytest.raises(ValueError):
        validate_size(size, config["canvas_size"], config["target_capacity"])


def test_reference_resize_aligns_both_axes_and_fits_capacity():
    config = validate_capacity_config({})
    for actual in ((4000, 6000), (17, 10000), (10000, 17), (361, 641)):
        output = fit_reference_size(*actual, config["canvas_size"], config["reference_capacity"])
        validate_size(output, config["canvas_size"], config["reference_capacity"])


def test_text_capacity_uses_real_tokenizer_template_and_visual_slots():
    from types import SimpleNamespace

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    special = ["[UNK]", "<|im_start|>", "<|im_end|>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]
    backend = Tokenizer(models.WordLevel(dict(zip(special, range(len(special)), strict=True)), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", additional_special_tokens=special[1:]
    )
    processor = SimpleNamespace(tokenizer=tokenizer)
    config = validate_capacity_config({})
    capacities = derive_text_capacities(processor, config)
    assert capacities["feature_capacity"] >= 1088 + 512 + capacities["template_tokens"]
    assert capacities["prefix_capacity"] >= 4352 + 512 + capacities["template_tokens"]
    config["max_references"] = 2
    larger = derive_text_capacities(processor, config)
    assert larger["feature_capacity"] >= 2176 + 512 + larger["template_tokens"]
    assert larger["prefix_capacity"] >= 8704 + 512 + larger["template_tokens"]
