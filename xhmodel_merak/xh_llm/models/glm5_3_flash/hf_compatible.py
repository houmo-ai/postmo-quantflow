# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Hugging Face multimodal generation with Merak-owned GLM caches."""

from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from xhquant.utils.registry import _DMRegistryCls

from ...text_llm_hf_compatible import TextLLMHFCompatible
from .vision import fuse_multimodal_embeddings


class Glm53HFCompatible(TextLLMHFCompatible):
    def _setup(self, text_llm_model):
        model = super()._setup(text_llm_model)
        del model.model
        del model.lm_head
        return model

    @property
    def is_support_dynamic_input(self):
        return False

    def generate(self, *args, **kwargs):
        if kwargs.get("num_beams", 1) != 1 or kwargs.get("num_return_sequences", 1) != 1:
            raise ValueError("GLM static generation supports one sequence without beam search")
        if kwargs.get("past_key_values") is not None:
            raise ValueError("GLM static generation owns its caches and starts a new request")
        # HF's config-derived cache cannot query length for KDA-only prefixes.
        # Generation only needs a carrier; actual state and length live in Merak.
        kwargs["past_key_values"] = DynamicCache()
        self._llm_model.get_kvcache_mixin().clear_kv_cache()
        return super().generate(*args, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        # HF's cache is only a carrier. Merak owns the authoritative request length.
        if self._past_seq_length == 0:
            inputs.update(
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
        return inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        if input_ids is None or inputs_embeds is not None:
            raise ValueError(
                "GLM static generation requires input_ids; visual embedding fusion is performed internally"
            )
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("GLM static generation requires one nonempty token sequence")
        if use_cache is False:
            raise ValueError("GLM static generation requires use_cache=True")
        masks = attention_mask.values() if isinstance(attention_mask, dict) else (attention_mask,)
        if any(mask is not None and not bool(mask.all()) for mask in masks):
            raise ValueError("pass unpadded tokens; the GLM Host processor owns right padding")
        spec = self._llm_model.get_kvcache_mixin().spec
        if self._past_seq_length + input_ids.shape[1] > spec.max_context_length:
            raise ValueError("request exceeds the configured context capacity")
        values = dict(
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        multimodal = any(v is not None for v in values.values())
        # Only prompt placeholders denote image features; a generated special token
        # is still an ordinary decode input, as in the native model.
        if self._past_seq_length == 0 and bool((input_ids == self.config.image_token_id).any()):
            multimodal = True
        if multimodal:
            inputs_embeds = fuse_multimodal_embeddings(
                input_ids, self.embed_tokens, getattr(self._llm_model, "vision_encoder", None), self.config, **values
            )
        length = self._llm_model.get_input_sequence_length()
        offset = 0
        for chunk in input_ids.split(length, dim=1):
            data = (
                {"input_ids": chunk}
                if inputs_embeds is None
                else {"inputs_embeds": inputs_embeds[:, offset : offset + chunk.shape[1]]}
            )
            data["past_seq_length"] = self._past_seq_length + offset
            args = self._llm_model.get_data_preprocessor()(data)
            logits = self._llm_model.forward(*args)
            offset += chunk.shape[1]
        return CausalLMOutputWithPast(
            logits=logits, past_key_values=past_key_values if past_key_values is not None else DynamicCache()
        )


def build_glm53_hf_compatible(hf_model, xh_model):
    registry = _DMRegistryCls("XHCompatible")
    model_cls = type(hf_model)
    registry.register_module({model_cls: model_cls.__name__}, Glm53HFCompatible)
    return registry.convert(hf_model, text_llm_model=xh_model)
