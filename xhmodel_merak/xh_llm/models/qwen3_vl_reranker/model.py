"""Reuse VL graph rewriting and Prefill-only export with a binary score head."""

from pathlib import Path

import torch
from torch import nn

from xhmodel_merak.xh_llm.builder import register_llm_model
from xhmodel_merak.xh_llm.models.qwen3_vl.qwen3_vl_visual_model import XHQwen3VLVisualModel
from xhmodel_merak.xh_llm.models.qwen3_vl_embedding.model import (
    XHQwen3VLEmbeddingModel,
    XHQwen3VLEmbeddingModelConfig,
)

from .hmonnx_inference import RerankerScoringMixin, XHQwen3VLRerankerHMONNXModel
from .visual import RerankerTokenVisualModel


def binary_score_head(lm_head, yes_id, no_id):
    """Match the official difference projection, padded to the TE output tile."""
    weight = lm_head.weight.detach()
    if yes_id == no_id or not (0 <= yes_id < weight.shape[0] and 0 <= no_id < weight.shape[0]):
        raise ValueError("yes/no token IDs must identify distinct vocabulary rows")
    head = nn.Linear(weight.shape[1], 64, bias=False, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[0].copy_(weight[yes_id] - weight[no_id])
    return head


class _RerankerVisualModel(XHQwen3VLVisualModel):
    def _get_dummy_inputs(self):
        config = self.config
        return {
            "hm_pixel_values": [torch.zeros(1, 3, 2, config.max_size_h, config.max_size_w)],
            "image_grid_thw": torch.tensor(
                [[1, config.max_size_h // config.patch_size, config.max_size_w // config.patch_size]]
            ),
        }


class XHQwen3VLRerankerModelConfig(XHQwen3VLEmbeddingModelConfig):
    def __init__(
        self,
        *,
        num_logits_to_keep=1,
        output_hidden_states_for_export=False,
        output_pre_norm_hidden_states=False,
        visual_config=None,
        **kwargs,
    ):
        if num_logits_to_keep != 1 or output_hidden_states_for_export or output_pre_norm_hidden_states:
            raise ValueError("Reranker requires the final normalized token and binary score projection")
        if isinstance(visual_config, dict) and visual_config.get("visual_input_mode") == "patches":
            from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5_VisualConfig

            visual_config = XHQwen3_5_VisualConfig(
                **{
                    **{"model_name": kwargs["model_name"] + "_visual", "hf_model": kwargs.get("hf_model")},
                    **visual_config,
                }
            )
        super().__init__(
            num_logits_to_keep=1, output_hidden_states_for_export=False, visual_config=visual_config, **kwargs
        )
        if self.batch_size != 1 or not self.use_cache:
            raise ValueError("Reranker requires batch_size=1 and use_cache=True for chunked Prefill")


@register_llm_model("Qwen3VLForConditionalGeneration_reranker")
class XHQwen3VLRerankerModel(RerankerScoringMixin, XHQwen3VLEmbeddingModel):
    transformers_min_version = "4.57.6"
    CONFIG_CLS = XHQwen3VLRerankerModelConfig
    HMONNXINFERENCE_CLS = XHQwen3VLRerankerHMONNXModel
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.qwen3_vl_reranker.workflow:XHQwen3VLRerankerWorkflow"

    def __init__(self, config):
        from xhmodel_merak.xh_llm.vision_llm_model import VisionLLMModel

        VisionLLMModel.__init__(self, config)
        self.dynamic_visual = getattr(config.visual_config, "visual_input_mode", None) == "patches"
        self.visual = (RerankerTokenVisualModel if self.dynamic_visual else _RerankerVisualModel)(config.visual_config)

    def to_wrap(self):
        if self.dynamic_visual:
            from xhmodel_merak.xh_llm.models.qwen3_vl import XHQwen3VLModel

            return XHQwen3VLModel.to_wrap(self)
        return super().to_wrap()

    def export_hmonnx(self, output_dir):
        if not self.dynamic_visual:
            return super().export_hmonnx(output_dir)
        from xhmodel_merak.xh_llm.types import LLMModelState

        self.work_dir = str(output_dir)
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.fixed()
        self.visual.quanted_model.fixed()
        info = self.get_export_info(output_dir)
        self.config.model_name = info.model_name
        root = Path(info.exported_dir)
        visual = self.visual.export_hmonnx(str(root), write_standalone_metadata=False)
        visual.hmonnx = Path(visual.hmonnx).relative_to(root).as_posix()
        visual.gear_manifest = Path(visual.gear_manifest).relative_to(root).as_posix()
        info.meta.visual_config = visual
        self._export_hmonnx(info)
        return info.meta

    def get_export_info(self, output_dir):
        if self.dynamic_visual:
            from xhmodel_merak.xh_llm.base_llm_model import BaseLLMModel

            return BaseLLMModel.get_export_info(self, output_dir)
        return super().get_export_info(output_dir)

    def get_dummy_inputs(self):
        inputs = super().get_dummy_inputs()
        inputs["input_ids"].zero_()
        return inputs

    def _get_data_preprocessor(self):
        if not self.dynamic_visual:
            return super()._get_data_preprocessor()
        from xhmodel_merak.xh_llm.models.qwen3_vl.data_preprocess import Qwen3VLDataPreprocess

        return Qwen3VLDataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            patch_size=self.config.visual_config.patch_size,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            spatial_merge_size=self.config.spatial_merge_size,
        )

    @property
    def reranker_config(self):
        return self.config

    def _wraped_pre(self, hf_model):
        super()._wraped_pre(hf_model)
        vocabulary = self.get_tokenizer().get_vocab()
        hf_model.lm_head = binary_score_head(hf_model.lm_head, vocabulary["yes"], vocabulary["no"])

    def get_export_cfg(self):
        cfg = super().get_export_cfg()
        cfg["output_names"] = ["score_logits"]
        return cfg

    def _extra_export_metadata(self, output_dir, meta_info):
        meta = super()._extra_export_metadata(output_dir, meta_info)
        self._save_export_token_embedding(output_dir, meta)
        meta.embedding_output = "score_logits"
        meta.embedding_pooling = "last_token"
        meta.embedding_normalize = False
        meta.reranker = {
            "version": 1,
            "output": "score_logits",
            "score_index": 0,
            "head_width": 64,
            "activation": "sigmoid",
            "projection": "yes_minus_no",
        }
        return meta
