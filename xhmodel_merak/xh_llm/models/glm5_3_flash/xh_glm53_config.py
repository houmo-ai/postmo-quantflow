# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from ...types import TextLLMModelConfig


class XHGlm53ModelConfig(TextLLMModelConfig):
    def __init__(
        self,
        *,
        model_name="glm5_3_flash",
        model_type="Glm5NextForConditionalGeneration",
        batch_size=1,
        context_max_length=4096,
        prefill_chunk_length=256,
        linear_chunk_size=64,
        num_logits_to_keep=1,
        max_layers=None,
        placeholder_export_workers=4,
        export_vision=True,
        vision_max_patches=1024,
        **kwargs,
    ):
        if batch_size != 1 or num_logits_to_keep != 1:
            raise ValueError("GLM-5.3 static export requires batch_size=1 and last-token logits")
        if linear_chunk_size <= 0 or linear_chunk_size % 8 or prefill_chunk_length % linear_chunk_size:
            raise ValueError("prefill length must be divisible by a KDA chunk size that is a multiple of eight")
        if not 0 < prefill_chunk_length <= context_max_length or not 4 <= context_max_length <= 1048576:
            raise ValueError("invalid fixed sequence/context length")
        if max_layers is not None and not 1 <= max_layers <= 45:
            raise ValueError("max_layers must be in [1,45]")
        if "low_memory" in kwargs:
            raise ValueError("GLM-5.3 always uses streamed export; low_memory is not a configurable option")
        if placeholder_export_workers < 1:
            raise ValueError("placeholder_export_workers must be positive")
        if vision_max_patches <= 0 or vision_max_patches % 4:
            raise ValueError("vision_max_patches must be a positive multiple of four")
        if kwargs.get("spec_decode_mode") is not None or kwargs.get("enable_page_attention", False):
            raise ValueError("GLM-5.3 requires its dedicated KDA/DSA state ABI")
        if kwargs.get("use_cache", True) is not True:
            raise ValueError("GLM static export requires cache state")
        kwargs.pop("enable_prefill_chunk", None)
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            batch_size=batch_size,
            context_max_length=context_max_length,
            prefill_chunk_length=prefill_chunk_length,
            linear_chunk_size=linear_chunk_size,
            num_logits_to_keep=num_logits_to_keep,
            max_layers=max_layers,
            enable_prefill_chunk=True,
            **kwargs,
        )
        self.placeholder_export_workers = placeholder_export_workers
        self.export_vision = export_vision
        self.vision_max_patches = vision_max_patches
