# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image prompt semantics over the existing Qwen3-VL deployment graphs."""

from dataclasses import dataclass

import torch
from PIL import Image


@dataclass
class PreparedPrompt:
    embeddings: torch.Tensor
    positions: torch.Tensor
    deepstack: list[torch.Tensor]
    image_mask: torch.Tensor
    drop_tokens: int


class PromptEncoder:
    """Batch-one text/image features over one reusable packed vision capacity.

    The outer image pipeline chooses one resize for both this encoder and the
    VAE. Alpha is composited over white only for vision; callers retain RGBA for
    VAE encoding. A separate instance of the DiT request owns each CFG branch.
    """

    SYSTEM = "Comprehend and analyze the provided prompt."

    def __init__(self, processor, token_embedding, feature_session, vision_session, *, prompt_budget, max_references):
        from xhmodel_merak.xh_llm.models.qwen3_vl.data_preprocess import Qwen3VLDataPreprocess

        self.processor = processor
        self.token_embedding = token_embedding
        self.feature_session = feature_session
        self.vision_session = vision_session
        self.prompt_budget, self.max_references = prompt_budget, max_references
        image_processor = processor.image_processor
        if (image_processor.patch_size, image_processor.temporal_patch_size, image_processor.merge_size) != (16, 2, 2):
            raise ValueError("Qwen-Image vision requires patch16, temporal2, merge2")
        if list(image_processor.image_mean) != [0.5] * 3 or list(image_processor.image_std) != [0.5] * 3:
            raise ValueError("vision graph expects mean/std 0.5 normalization")
        tokenizer = processor.tokenizer
        self.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.positions = Qwen3VLDataPreprocess(
            token_embedding=token_embedding,
            image_token_id=self.image_token_id,
            video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
            vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        )
        system = [{"role": "system", "content": [{"type": "text", "text": self.SYSTEM}]}]
        self.drop_tokens = len(processor.apply_chat_template(system, tokenize=True, return_dict=False)[0])

    @staticmethod
    def vision_image(image):
        if not isinstance(image, Image.Image):
            raise TypeError("reference images must be PIL images")
        if image.mode == "RGBA":
            rgb = Image.new("RGB", image.size, (255, 255, 255))
            rgb.paste(image, mask=image.getchannel("A"))
            return rgb
        return image.convert("RGB")

    @torch.no_grad()
    def prepare(self, prompt, images=None):
        if not isinstance(prompt, str):
            raise TypeError("one prompt string is required per request")
        images = [] if images is None else list(images)
        if len(images) > self.max_references:
            raise ValueError("reference count exceeds the published capacity")
        if len(self.processor.tokenizer(prompt or " ", add_special_tokens=False).input_ids) > self.prompt_budget:
            raise ValueError("prompt exceeds the published token budget")
        rgb_images = [self.vision_image(image) for image in images]
        for image in rgb_images:
            if image.height % 32 or image.width % 32:
                raise ValueError("reference image dimensions must be multiples of 32")
        prefix = " ".join(f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(len(images)))
        text = (
            f"<|im_start|>system\n{self.SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{prefix}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
        )
        kwargs = dict(text=[text], padding=True, padding_side="left", return_tensors="pt")
        if images:
            kwargs.update(images=rgb_images, do_resize=False)
        batch = self.processor(**kwargs)
        ids = batch.input_ids
        if ids.shape[1] > self.feature_session.capacity:
            raise ValueError("tokenized prompt exceeds the exported feature capacity")
        embeddings = self.token_embedding(ids.to(self.token_embedding.weight.device)).float().cpu()
        image_mask = ids == self.image_token_id
        positions, _ = self.positions.get_rope_index(
            ids,
            image_grid_thw=batch.get("image_grid_thw"),
            attention_mask=batch.attention_mask,
        )
        deepstack = [torch.zeros_like(embeddings) for _ in range(3)]
        visual_outputs = []
        patch_offset = 0
        for index, image in enumerate(rgb_images):
            grid = batch.image_grid_thw[index]
            count = int(grid.prod())
            values = self.vision_session(batch.pixel_values[patch_offset : patch_offset + count], grid)
            patch_offset += count
            expected_shape = (1, image.height * image.width // 1024, embeddings.shape[-1])
            if len(values) != 4 or any(x.shape != expected_shape for x in values):
                raise ValueError("vision graph must return main features and three DeepStack features")
            visual_outputs.append([x.detach().float().cpu().clone() for x in values])
        if images and patch_offset != batch.pixel_values.shape[0]:
            raise ValueError("processor patch count disagrees with reference grids")
        if visual_outputs:
            main = torch.cat([values[0] for values in visual_outputs], dim=1)
            if int(image_mask.sum()) != main.shape[1]:
                raise ValueError("processor image slots do not match the vision gear")
            embeddings[image_mask] = main[0]
            for index in range(3):
                deepstack[index][image_mask] = torch.cat([values[index + 1] for values in visual_outputs], dim=1)[0]
        elif bool(image_mask.any()):
            raise ValueError("image placeholder requires a reference image")
        return PreparedPrompt(embeddings, positions[:, 0], deepstack, image_mask, self.drop_tokens)

    @torch.no_grad()
    def __call__(self, prompt, images=None):
        prepared = self.prepare(prompt, images)
        features = self.feature_session(prepared.embeddings, prepared.positions, prepared.deepstack)
        return features[:, prepared.drop_tokens :].clone(), prepared.image_mask[:, prepared.drop_tokens :].clone()
