"""CPU contracts for portable MiniMax-H3 fixed-length text conditioning."""

from types import SimpleNamespace

import torch

from xhmodel_merak.xh_other_model.models.minimax_h3.text_conditioning import (
    DEFAULT_TEXT_SEQUENCE_LENGTH,
    build_encoder_attention_mask,
    install_fixed_length_text_conditioning,
    normalize_fixed_length_text_conditioning,
)


class _FakePipelineState:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.kwargs_mapping = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value, kwargs_type=None):
        self.values[key] = value
        if kwargs_type is None:
            return
        names = self.kwargs_mapping.setdefault(kwargs_type, [])
        if key not in names:
            names.append(key)


class _FakeTextEncoderStep:
    def __call__(self, components, state):
        state.set("prompt_embeds", torch.randn(1, 3, 4))
        state.set("text_token_tags", torch.ones(3, dtype=torch.long))
        return components, state


class _FakeAutoTextEncoderStep:
    def __init__(self, child):
        self.sub_blocks = {"t2va": child}

    def __call__(self, components, state):
        return self.sub_blocks["t2va"](components, state)


class _FakeRootBlocks:
    def __init__(self, text_encoder):
        self.sub_blocks = {"text_encoder": text_encoder}


def test_build_encoder_attention_mask_marks_padding_rows():
    attention_mask = build_encoder_attention_mask(2, 5, dtype=torch.float16)

    assert tuple(attention_mask.shape) == (1, 1, 1, 5)
    torch.testing.assert_close(attention_mask[..., :2], torch.zeros(1, 1, 1, 2, dtype=torch.float16))
    assert torch.all(attention_mask[..., 2:] == torch.finfo(torch.float16).min)


def test_normalize_fixed_length_text_conditioning_pads_embeds_and_tags():
    prompt_embeds = torch.randn(1, 3, 4)
    text_token_tags = torch.tensor([7, 7, 7], dtype=torch.long)

    padded_embeds, padded_tags, attention_mask = normalize_fixed_length_text_conditioning(
        prompt_embeds,
        text_token_tags,
        sequence_length=8,
        pad_tag=7,
    )

    assert tuple(padded_embeds.shape) == (1, 8, 4)
    assert tuple(padded_tags.shape) == (8,)
    assert torch.equal(padded_tags[:3], text_token_tags)
    assert torch.all(padded_tags[3:] == 7)
    torch.testing.assert_close(padded_embeds[:, :3], prompt_embeds)
    torch.testing.assert_close(attention_mask[..., :3], torch.zeros(1, 1, 1, 3))
    assert torch.all(attention_mask[..., 3:] == torch.finfo(attention_mask.dtype).min)


def test_normalize_uses_shorter_of_embeds_and_tags_as_valid_length():
    prompt_embeds = torch.randn(1, 8, 4)
    text_token_tags = torch.ones(3, dtype=torch.long)

    _, padded_tags, attention_mask = normalize_fixed_length_text_conditioning(
        prompt_embeds,
        text_token_tags,
        sequence_length=8,
        pad_tag=1,
    )

    assert tuple(padded_tags.shape) == (8,)
    assert torch.all(padded_tags == 1)
    torch.testing.assert_close(attention_mask[..., :3], torch.zeros(1, 1, 1, 3))
    assert torch.all(attention_mask[..., 3:] == torch.finfo(attention_mask.dtype).min)


def test_install_fixed_length_text_conditioning_patches_leaf_text_encoder_steps():
    leaf = _FakeTextEncoderStep()
    auto = _FakeAutoTextEncoderStep(leaf)
    pipe = SimpleNamespace(_blocks=_FakeRootBlocks(auto), text_tag=9)

    patched = install_fixed_length_text_conditioning(pipe)

    assert patched == ("_FakeTextEncoderStep",)
    assert type(leaf).__name__.endswith("WithFixedTextConditioning")
    assert type(auto).__name__ == "_FakeAutoTextEncoderStep"

    state = _FakePipelineState()
    components = SimpleNamespace(text_tag=9)
    _, state = leaf(components, state)

    prompt_embeds = state.get("prompt_embeds")
    text_token_tags = state.get("text_token_tags")
    attention_mask = state.get("encoder_attention_mask")
    assert tuple(prompt_embeds.shape) == (1, DEFAULT_TEXT_SEQUENCE_LENGTH, 4)
    assert tuple(text_token_tags.shape) == (DEFAULT_TEXT_SEQUENCE_LENGTH,)
    assert tuple(attention_mask.shape) == (1, 1, 1, DEFAULT_TEXT_SEQUENCE_LENGTH)
    assert torch.all(text_token_tags[3:] == 9)
    assert torch.all(attention_mask[..., 3:] == torch.finfo(attention_mask.dtype).min)
    assert "encoder_attention_mask" in state.kwargs_mapping["denoiser_input_fields"]

    # Idempotent install.
    assert install_fixed_length_text_conditioning(pipe) == ()


def test_install_preserves_existing_fixed_mask_from_modified_diffusers():
    class MiniMaxH3TextEncoderStep:
        def __call__(self, components, state):
            prompt_embeds = torch.randn(1, DEFAULT_TEXT_SEQUENCE_LENGTH, 4)
            text_token_tags = torch.full((DEFAULT_TEXT_SEQUENCE_LENGTH,), 1, dtype=torch.long)
            attention_mask = build_encoder_attention_mask(
                4,
                DEFAULT_TEXT_SEQUENCE_LENGTH,
                dtype=prompt_embeds.dtype,
            )
            state.set("prompt_embeds", prompt_embeds)
            state.set("text_token_tags", text_token_tags)
            state.set("encoder_attention_mask", attention_mask)
            return components, state

    leaf = MiniMaxH3TextEncoderStep()
    pipe = SimpleNamespace(_blocks=_FakeRootBlocks(leaf), text_tag=1)
    install_fixed_length_text_conditioning(pipe)

    state = _FakePipelineState()
    _, state = leaf(SimpleNamespace(text_tag=1), state)
    attention_mask = state.get("encoder_attention_mask")

    torch.testing.assert_close(attention_mask[..., :4], torch.zeros(1, 1, 1, 4))
    assert torch.all(attention_mask[..., 4:] == torch.finfo(attention_mask.dtype).min)
    assert "encoder_attention_mask" in state.kwargs_mapping["denoiser_input_fields"]


def test_install_without_blocks_is_noop():
    assert install_fixed_length_text_conditioning(SimpleNamespace()) == ()
