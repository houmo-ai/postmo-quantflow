"""WeMM-Embedding adapter built on the maintained Qwen3.5 implementation."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from transformers import AutoConfig

from xhmodel_merak.xh_llm.builder import register_llm_model
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model
from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5ModelConfig

from .hmonnx_inference import XHWeMMEmbeddingHMONNXModel


class XHWeMMEmbeddingModelConfig(XHQwen3_5ModelConfig):  # noqa: N801
    """Qwen3.5 export configuration with full-sequence hidden-state output."""

    def __init__(self, *, num_logits_to_keep: int = 0, **kwargs: Any):
        super().__init__(num_logits_to_keep=num_logits_to_keep, **kwargs)


@register_llm_model("WeMMEmbedding")
class XHWeMMEmbeddingModel(XHQwen3_5Model):  # noqa: N801
    """Merak model wrapper for Tencent WeMM-Embedding checkpoints."""

    CONFIG_CLS = XHWeMMEmbeddingModelConfig
    HMONNXINFERENCE_CLS = XHWeMMEmbeddingHMONNXModel
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.wemm_embedding.workflow:XHWeMMEmbeddingWorkflow"

    def init_wrap_model(self, hf_model: Any) -> Any:
        wrap_model = super().init_wrap_model(hf_model)
        wrap_model.lm_head = nn.Identity()
        return wrap_model

    def _extra_export_metadata(self, output_dir: str, meta_info: Any) -> Any:
        meta_info = super()._extra_export_metadata(output_dir, meta_info)
        meta_info.embedding_output = "hidden_states"
        meta_info.embedding_pooling = "last_token"
        meta_info.embedding_normalize = True
        config = AutoConfig.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        text_config = getattr(config, "text_config", config)
        meta_info.embedding_dimension = int(text_config.hidden_size)
        return meta_info

    def get_export_cfg(self) -> dict[str, list[str]]:
        export_cfg = super().get_export_cfg()
        output_names = list(export_cfg["output_names"])
        if not output_names:
            raise ValueError("Qwen3.5 export configuration did not declare an output")
        output_names[0] = "hidden_states"
        export_cfg["output_names"] = output_names
        return export_cfg


__all__ = ["XHWeMMEmbeddingModel", "XHWeMMEmbeddingModelConfig"]
