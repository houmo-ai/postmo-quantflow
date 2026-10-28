"""Extend the existing Qwen3.5 token-gear encoder with Qwen3-VL DeepStack."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from xhmodel_merak.xh_llm.base_vision_model import BaseVisionModel
from xhmodel_merak.xh_llm.models.qwen3_5._vision_model_impl import (
    _Qwen3_5VisionAttention,
    _Qwen3_5VisionBlock,
    _Qwen3_5VisionModel,
    _Qwen3_5VisionPatchEmbed,
    _Qwen3_5VisionPatchMerger,
)
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import VisualTokenGearHMONNXModel
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_vision_model import XHQwen3_5VisionModel
from xhmodel_merak.xh_llm.models.qwen3_vl import modeling_qwen3_vl as hf
from xhmodel_merak.xh_llm.models.qwen3_vl.qwen3_vl_visual_model import XHQwen3VLVisualModel
from xhmodel_merak.xh_llm.register import XHLLM_TRACEABLE_MODULES


class _RerankerVisionModel(_Qwen3_5VisionModel):
    """Reuse patch-token setup and retain the three Qwen3-VL intermediate features."""

    def forward(
        self, hidden_states, position_ids=None, position_weights=None, rotary_position_ids=None, attention_mask=None
    ):
        hidden_states = self.patch_embed(hidden_states)
        if position_ids is None or position_weights is None:
            raise ValueError("patch-token visual input requires position_ids and position_weights")
        if rotary_position_ids is None or attention_mask is None:
            raise ValueError("patch-token visual input requires rotary_position_ids and attention_mask")
        position_values = self.pos_embed(position_ids)
        patch_pos_embeds = (position_values * position_weights.unsqueeze(-1)).sum(dim=0).unsqueeze(0)
        hidden_states = hidden_states + patch_pos_embeds
        rotary_cos = self.rotary_gather(self.rotary_cos_cached, rotary_position_ids)
        rotary_sin = self.rotary_gather(self.rotary_sin_cached, rotary_position_ids)
        cos_half = rotary_cos.permute(1, 0, 2).reshape(self.patch_token_capacity, self.rotary_axis_dim * 2)
        sin_half = rotary_sin.permute(1, 0, 2).reshape(self.patch_token_capacity, self.rotary_axis_dim * 2)
        if self.rotary_padding_size:
            cos_half = F.pad(cos_half, (0, self.rotary_padding_size), value=1.0)
            sin_half = F.pad(sin_half, (0, self.rotary_padding_size), value=0.0)
        cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(-2)
        sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(-2)
        position_embeddings = (cos, sin)
        deepstack = []
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask)
            if layer_index in self.deepstack_visual_indexes:
                index = self.deepstack_visual_indexes.index(layer_index)
                deepstack.append(self.deepstack_merger_list[index](hidden_states))
        return (self.merger(hidden_states), *deepstack)


@contextmanager
def token_vision_scope():
    registry = XHLLM_TRACEABLE_MODULES
    wrappers = {
        hf.Qwen3VLVisionAttention: _Qwen3_5VisionAttention,
        hf.Qwen3VLVisionBlock: _Qwen3_5VisionBlock,
        hf.Qwen3VLVisionModel: _RerankerVisionModel,
        hf.Qwen3VLVisionPatchEmbed: _Qwen3_5VisionPatchEmbed,
        hf.Qwen3VLVisionPatchMerger: _Qwen3_5VisionPatchMerger,
    }
    saved = {}
    for cls, wrapper in wrappers.items():
        saved[cls] = (registry._registry.get(cls), registry._key_registry.get(cls), registry._dynamic_classes.get(cls))
        registry._registry[cls] = wrapper
        registry._key_registry[cls] = cls.__name__
        registry._dynamic_classes.pop(cls, None)
    try:
        yield
    finally:
        for cls, values in saved.items():
            for mapping, previous in zip(
                (registry._registry, registry._key_registry, registry._dynamic_classes), values, strict=True
            ):
                if previous is None:
                    mapping.pop(cls, None)
                else:
                    mapping[cls] = previous


class RerankerTokenVisualModel(XHQwen3_5VisionModel):
    HF_MODEL_CLS = XHQwen3VLVisualModel.HF_MODEL_CLS
    HF_AUTO_MODEL_CLS = XHQwen3VLVisualModel.HF_AUTO_MODEL_CLS
    META_CLS = XHQwen3VLVisualModel.META_CLS
    VISUAL_HF_MODEL_CLS = hf.Qwen3VLVisionModel
    VISUAL_WEIGHT_PREFIX = "model.visual."
    OUTPUT_NAMES = ("image_embeds", "deepstack_feature_0", "deepstack_feature_1", "deepstack_feature_2")

    def __init__(self, config):
        BaseVisionModel.__init__(self, config)
        self._vision_config = AutoConfig.from_pretrained(self.hf_model_dir).vision_config
        for name in ("patch_size", "temporal_patch_size", "spatial_merge_size"):
            if getattr(self._vision_config, name) != getattr(config, name):
                raise ValueError(f"visual {name} must match checkpoint")

    def init_wrap_model(self, hf_model):
        with token_vision_scope():
            visual = hf_model if isinstance(hf_model, self.VISUAL_HF_MODEL_CLS) else hf_model.model.visual
            return BaseVisionModel.init_wrap_model(self, visual)

    @classmethod
    def get_hf_model(cls, hf_model_dir, quant_weight=None, **kwargs):
        return XHQwen3VLVisualModel.get_hf_model(hf_model_dir, quant_weight=quant_weight, **kwargs)

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs):
        return XHQwen3VLVisualModel.get_empty_hf_model(hf_model_dir, **kwargs)

    def get_export_cfg(self):
        cfg = super().get_export_cfg()
        cfg["output_names"] = list(self.OUTPUT_NAMES)
        return cfg

    def forward(self, *args, **kwargs):
        image, *deepstack = self._inference_model(*args, **kwargs)
        return image, deepstack


class RerankerTokenVisualRuntime(VisualTokenGearHMONNXModel):
    @staticmethod
    def _slice_valid_output(output, valid_image_tokens):
        image, *deepstack = output
        return (
            image[:, :valid_image_tokens],
            [feature[:, :valid_image_tokens] for feature in deepstack],
        )

    def encode(self, pixel_values, grid_thw):
        gear, inputs, valid = self._prepare_image(pixel_values, grid_thw)
        output = self.models[gear].forward(*inputs)
        if self.enable_golden:
            self.models[gear].update_step()
        return self._slice_valid_output(output, valid)
