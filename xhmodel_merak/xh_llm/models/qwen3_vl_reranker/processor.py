"""Pair templates and host scheduling; neural/cache updates remain in the graph."""

from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)


def reranker_token_indices(input_ids, tokenizer, *, max_length, protected_indices=()):
    """Keep special tokens and the official five-token scoring suffix.

    The official wrapper budgets the body separately from its final five
    tokens. Here max_length is the exported total capacity, so the suffix
    consumes five positions within that capacity.
    """
    suffix = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    if len(suffix) != 5 or list(input_ids[-5:]) != suffix:
        raise ValueError("Reranker requires the complete five-token template suffix; use the Reranker chat template")
    if max_length < 5:
        raise ValueError("Reranker context must accommodate the five-token template suffix")
    if len(input_ids) <= max_length:
        return list(range(len(input_ids)))
    body = input_ids[:-5]
    special = set(tokenizer.all_special_ids)
    # Some HF tokenizer versions (and vLLM's cached view) omit visual tokens
    # from all_special_ids. AddedToken metadata retains their special flags.
    special.update(token_id for token_id, token in tokenizer.added_tokens_decoder.items() if token.special)
    protected = set(protected_indices)
    if any(index < 0 or index >= len(body) for index in protected):
        raise ValueError("Protected visual positions must lie before the template suffix")
    ordinary_budget = max_length - 5 - sum(token in special or index in protected for index, token in enumerate(body))
    if ordinary_budget < 0:
        raise ValueError("Reranker special tokens and visual tokens exceed context capacity; reduce the media input")
    selected = []
    for index, token in enumerate(body):
        if token in special or index in protected:
            selected.append(index)
        elif ordinary_budget:
            selected.append(index)
            ordinary_budget -= 1
    return selected + list(range(len(body), len(input_ids)))


class RerankerProcessor:
    """Use the installed official processor, adding only the raw visual graph ABI."""

    def __init__(
        self,
        model_dir,
        *,
        width=None,
        height=None,
        patch_size,
        image_token_capacity=None,
        spatial_merge_size=2,
        context_length=32768,
    ):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.width, self.height, self.patch_size = width, height, patch_size
        self.image_token_capacity = image_token_capacity
        self.spatial_merge_size = spatial_merge_size
        self.context_length = context_length

    def apply_chat_template(self, messages):
        from qwen_vl_utils import process_vision_info

        messages = deepcopy(messages)
        for message in messages:
            for item in message["content"]:
                if item["type"] == "image":
                    if self.image_token_capacity is None:
                        item.update(resized_width=self.width, resized_height=self.height)
                    else:
                        unit_pixels = (self.patch_size * self.spatial_merge_size) ** 2
                        item.update(min_pixels=4 * unit_pixels, max_pixels=self.image_token_capacity * unit_pixels)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages, image_patch_size=self.patch_size)
        if videos is not None:
            raise ValueError("video is unsupported")
        inputs = self.processor(text=[text], images=images, do_resize=False, padding=False, return_tensors="pt")
        indices = reranker_token_indices(
            inputs["input_ids"][0].tolist(), self.processor.tokenizer, max_length=self.context_length
        )
        # Keep per-token masks/types aligned; visual patch/grid tensors are
        # unchanged because every visual special token is retained.
        for name in ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"):
            if name in inputs:
                inputs[name] = inputs[name][:, indices]
        if images and self.image_token_capacity is None:
            inputs["hm_pixel_values"] = [
                torch.from_numpy(np.array(image.convert("RGB"), copy=True))
                .permute(2, 0, 1)[None, :, None]
                .repeat(1, 1, 2, 1, 1)
                .float()
                for image in images
            ]
        return inputs


def normalize_item(item):
    if isinstance(item, str):
        item = {"text": item}
    if not isinstance(item, Mapping):
        raise ValueError("Each query/document must be a text or image mapping")
    unknown = set(item) - {"text", "image"}
    if unknown:
        raise ValueError(f"Unsupported query/document fields: {sorted(unknown)}; video is unsupported")
    text = item.get("text")
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise ValueError("text must be a non-empty string")
    images = item.get("image")
    images = images if isinstance(images, (list, tuple)) else ([] if images is None else [images])
    if any(not isinstance(image, (str, Image.Image)) or (isinstance(image, str) and not image) for image in images):
        raise ValueError("image must be an image path, PIL image, or a non-empty list of these")
    if text is None and not images:
        raise ValueError("Each query/document requires text or image")
    return {"text": text, "image": list(images)}


def build_messages(query, document, instruction=DEFAULT_INSTRUCTION):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    content = [{"type": "text", "text": "<Instruct>: " + instruction}]
    for prefix, item in [("<Query>:", query), ("\n<Document>:", document)]:
        item = normalize_item(item)
        content.append({"type": "text", "text": prefix})
        content.extend({"type": "image", "image": image} for image in item["image"])
        if item["text"] is not None:
            content.append({"type": "text", "text": item["text"]})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def merge_visual_features(groups):
    if not groups:
        return None, None
    images = torch.cat([image.reshape(-1, image.shape[-1]) for image, _ in groups], dim=0)
    deep = [
        torch.cat([features[index].reshape(-1, features[index].shape[-1]) for _, features in groups], dim=0)
        for index in range(3)
    ]
    return images, deep


def prefill_chunks(full_inputs, *, chunk_length, context_length):
    """Slice already merged embeddings and global M-RoPE coordinates together."""
    length = int(full_inputs[5].item())
    if chunk_length <= 0 or length <= 0 or length > context_length:
        raise ValueError(f"Input length {length} exceeds context capacity {context_length} or is empty")
    for offset in range(0, length, chunk_length):
        valid = min(chunk_length, length - offset)
        chunk = list(full_inputs)
        for slot in [0, 6, 7, 8]:
            chunk[slot] = F.pad(full_inputs[slot][:, offset : offset + valid], (0, 0, 0, chunk_length - valid))
        for slot in [1, 2, 3]:
            chunk[slot] = F.pad(full_inputs[slot][offset : offset + valid], (0, chunk_length - valid))
        chunk[4] = torch.tensor([offset], dtype=torch.int32, device=chunk[0].device)
        chunk[5] = torch.tensor([valid], dtype=torch.int32, device=chunk[0].device)
        yield tuple(chunk)
