# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native H3/Qwen presentation and M-RoPE over exported feature sessions."""

import torch
from torch import nn


class LanguageFeatureAdapter(nn.Module):
    def __init__(self, config, embedding, run_features, *, feature_layer, divisor):
        super().__init__()
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

        if not 0 < feature_layer < config.num_hidden_layers:
            raise ValueError("conditioning must select an intermediate, pre-norm hidden state")
        self.config, self.feature_layer, self.divisor = config, feature_layer, divisor
        self.embed_tokens = nn.Embedding.from_pretrained(embedding.detach().float(), freeze=True)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)
        self.run_features = run_features

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        *,
        inputs_embeds,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1 or past_key_values is not None:
            raise ValueError("H3 conditioning consumes a complete batch-one presentation")
        if attention_mask is not None and (
            attention_mask.shape != inputs_embeds.shape[:2] or not bool(attention_mask.bool().all())
        ):
            raise ValueError("H3 presentation must contain only valid tokens before chunk padding")
        hidden = inputs_embeds.float().cpu() / self.divisor
        if position_ids is None:
            position_ids = torch.arange(hidden.shape[1])[None, None].expand(3, 1, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        elif position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        cos, sin = self.rotary_emb(hidden, position_ids.cpu())
        additions = [torch.zeros_like(hidden) for _ in range(3)]
        if deepstack_visual_embeds is not None:
            if len(deepstack_visual_embeds) != 3 or visual_pos_masks is None:
                raise ValueError("H3 vision conditioning requires three DeepStack features and their token mask")
            mask = visual_pos_masks.cpu()
            for target, value in zip(additions, deepstack_visual_embeds, strict=True):
                target[mask] = value.float().cpu() / self.divisor
        # Transformers records each decoder output before that layer's DeepStack
        # addition. H3's layer 50 is beyond DeepStack; short calibration prefixes
        # still need the same selected-hidden-state convention.
        if self.feature_layer <= len(additions):
            additions[self.feature_layer - 1].zero_()
        features = self.run_features(hidden, cos, sin, additions).float() * self.divisor
        if features.shape != hidden.shape:
            raise ValueError("conditioner feature sessions must preserve the complete presentation length")
        # The native H3 block only consumes this selected prefix. Other hidden
        # states are explicitly absent; no final norm or LM head is evaluated.
        states = (None,) * self.feature_layer + (features,)
        return BaseModelOutputWithPast(last_hidden_state=features, hidden_states=states)


class H3FeatureEncoder(nn.Module):
    def __init__(self, config, embedding, vision, run_features, *, feature_layer=50, divisor=16):
        super().__init__()
        from transformers import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        self.config = Qwen3VLConfig.from_dict(config)
        # Keep upstream placeholder insertion, multimodal ordering and M-RoPE.
        # Meta construction never materializes the discarded transformer weights.
        with torch.device("meta"):
            self.model = Qwen3VLModel(self.config)
        self.model.visual = vision
        self.model.language_model = LanguageFeatureAdapter(
            self.config.text_config, embedding, run_features, feature_layer=feature_layer, divisor=divisor
        )
        self.dtype = torch.float32

    def forward(self, **kwargs):
        return self.model(**kwargs)
