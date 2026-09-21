"""Fixed-length MiniMax-H3 text conditioning helpers.

These helpers used to live only in an out-of-repo diffusers `encoders.py` patch.
Keep them here so HMONNX installs can normalize prompt embeds and emit
`encoder_attention_mask` without requiring a modified diffusers checkout.
"""

from __future__ import annotations

from typing import Any

import torch


DEFAULT_TEXT_SEQUENCE_LENGTH = 512

_TEXT_CONDITIONING_INSTALLED = "_xh_fixed_text_conditioning_installed"
_WRAPPED_SUFFIX = "WithFixedTextConditioning"


def build_encoder_attention_mask(
    valid_length: int,
    sequence_length: int,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build an additive `[1, 1, 1, sequence_length]` key-padding bias for text rows."""
    if not 0 <= valid_length <= sequence_length:
        raise ValueError(f"`valid_length` must be in [0, {sequence_length}], got {valid_length}.")
    attention_mask = torch.zeros(1, 1, 1, sequence_length, dtype=dtype, device=device)
    if valid_length < sequence_length:
        attention_mask[..., valid_length:] = torch.finfo(dtype).min
    return attention_mask


def normalize_fixed_length_text_conditioning(
    prompt_embeds: torch.Tensor,
    text_token_tags: torch.Tensor,
    *,
    sequence_length: int = DEFAULT_TEXT_SEQUENCE_LENGTH,
    pad_tag: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad/truncate text embeds and tags to a fixed length and build the transformer mask.

    HMONNX text may already return the fixed export length while `text_token_tags` still
    tracks the real presentation length. The shorter of the two is treated as valid.
    """
    if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
        raise ValueError(
            f"`prompt_embeds` must have shape `(1, num_text_tokens, hidden)`, got {tuple(prompt_embeds.shape)}."
        )
    if text_token_tags.ndim != 1:
        raise ValueError(f"`text_token_tags` must be a 1-D tensor, got shape {tuple(text_token_tags.shape)}.")
    if sequence_length <= 0:
        raise ValueError(f"`sequence_length` must be positive, got {sequence_length}.")

    embed_length = int(prompt_embeds.shape[1])
    tag_length = int(text_token_tags.shape[0])
    valid_length = min(embed_length, tag_length)
    if pad_tag is None:
        pad_tag = int(text_token_tags[0].item()) if tag_length else 1

    if tag_length < embed_length:
        text_token_tags = torch.nn.functional.pad(
            text_token_tags,
            (0, embed_length - tag_length),
            value=pad_tag,
        )
    elif tag_length > embed_length:
        text_token_tags = text_token_tags[:embed_length]

    current_length = int(prompt_embeds.shape[1])
    if current_length > sequence_length:
        prompt_embeds = prompt_embeds[:, :sequence_length].contiguous()
        text_token_tags = text_token_tags[:sequence_length].contiguous()
        valid_length = min(valid_length, sequence_length)
    elif current_length < sequence_length:
        pad = sequence_length - current_length
        prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, pad))
        text_token_tags = torch.nn.functional.pad(text_token_tags, (0, pad), value=pad_tag)

    attention_mask = build_encoder_attention_mask(
        valid_length,
        sequence_length,
        dtype=prompt_embeds.dtype,
        device=prompt_embeds.device,
    )
    return prompt_embeds, text_token_tags.to(dtype=torch.long), attention_mask


def _iter_pipeline_blocks(root: Any):
    stack = [root]
    seen: set[int] = set()
    while stack:
        block = stack.pop()
        block_id = id(block)
        if block_id in seen:
            continue
        seen.add(block_id)
        yield block
        sub_blocks = getattr(block, "sub_blocks", None)
        if not sub_blocks:
            continue
        values = sub_blocks.values() if hasattr(sub_blocks, "values") else sub_blocks
        stack.extend(values)


def _text_encoder_step_base_name(block: Any) -> str:
    return type(block).__name__.removesuffix(_WRAPPED_SUFFIX)


def _is_text_encoder_step(block: Any) -> bool:
    # Patch leaf TextEncoder steps only. Auto/Conditional parents also contain the
    # substring and would otherwise double-wrap the selected child.
    base_name = _text_encoder_step_base_name(block)
    return base_name.endswith("TextEncoderStep") and "Auto" not in base_name


def _persist_text_conditioning(state: Any, prompt_embeds, text_token_tags, attention_mask) -> None:
    # Named state fields are consumed by PrepareLayout; kwargs_type keeps stock denoise
    # forwarding working even when before_denoise does not re-declare the mask.
    state.set("prompt_embeds", prompt_embeds)
    state.set("text_token_tags", text_token_tags)
    state.set("encoder_attention_mask", attention_mask)
    state.set("encoder_attention_mask", attention_mask, "denoiser_input_fields")


def _wrap_text_encoder_step(step: Any) -> bool:
    if getattr(step, _TEXT_CONDITIONING_INSTALLED, False):
        return False

    original_call = step.__class__.__call__

    def call_with_fixed_text_conditioning(self, components, state, *args, **kwargs):
        components, state = original_call(self, components, state, *args, **kwargs)
        prompt_embeds = state.get("prompt_embeds")
        text_token_tags = state.get("text_token_tags")
        if prompt_embeds is None or text_token_tags is None:
            return components, state

        existing_mask = state.get("encoder_attention_mask")
        already_fixed = (
            int(prompt_embeds.shape[1]) == DEFAULT_TEXT_SEQUENCE_LENGTH
            and int(text_token_tags.shape[0]) == DEFAULT_TEXT_SEQUENCE_LENGTH
        )
        # Environments that already patched diffusers encoders emit the mask themselves.
        # Re-normalizing padded tags would treat every row as valid and wipe pad bias.
        if existing_mask is not None and already_fixed:
            state.set("encoder_attention_mask", existing_mask, "denoiser_input_fields")
            return components, state

        pad_tag = getattr(components, "text_tag", None)
        prompt_embeds, text_token_tags, attention_mask = normalize_fixed_length_text_conditioning(
            prompt_embeds,
            text_token_tags,
            pad_tag=None if pad_tag is None else int(pad_tag),
        )
        _persist_text_conditioning(state, prompt_embeds, text_token_tags, attention_mask)
        return components, state

    step.__class__ = type(
        f"{_text_encoder_step_base_name(step)}{_WRAPPED_SUFFIX}",
        (step.__class__,),
        {"__call__": call_with_fixed_text_conditioning},
    )
    setattr(step, _TEXT_CONDITIONING_INSTALLED, True)
    return True


def install_fixed_length_text_conditioning(pipe: Any) -> tuple[str, ...]:
    """Patch `pipe._blocks` TextEncoder leaves to emit fixed-512 text + mask.

    Prefer the live block tree on `pipe._blocks`. `pipe.blocks` is a deepcopy and
    patching it would not affect runtime. Missing `_blocks` is a no-op so unit-test
    fakes and partial pipes remain usable.
    """
    root = getattr(pipe, "_blocks", None)
    if root is None:
        return ()

    patched: list[str] = []
    for block in _iter_pipeline_blocks(root):
        if not _is_text_encoder_step(block):
            continue
        if _wrap_text_encoder_step(block):
            patched.append(_text_encoder_step_base_name(block))
    return tuple(patched)


__all__ = [
    "DEFAULT_TEXT_SEQUENCE_LENGTH",
    "build_encoder_attention_mask",
    "install_fixed_length_text_conditioning",
    "normalize_fixed_length_text_conditioning",
]
