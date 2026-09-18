"""Compatibility model name for the text-only decode ACEP workflow."""

from __future__ import annotations

import json
from pathlib import Path

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

from xhquant.api import get_xhquant_logger

from ...builder import register_llm_model
from ...types import LLMModelState
from ..qwen3_5.qwen3_5_llm_model import XHQwen3_5Model
from ..qwen3_5.workflow import Qwen35Workflow
from ..qwen3_5_moe.qwen3_5_moe_hmonnx_inference import XHQwen3_5MoeHMONNXModel
from ..qwen3_5_moe.qwen3_5_moe_model import (
    Qwen3_5Moe_ModelMeta,
    XHQwen3_5MoeModel,
    build_qwen3_5_moe_hf_compatible_model,
)
from .xh_qwen3_5_moe_dynamic_prune_config import XHQwen3_5MoeDynamicPruneModelConfig


class Qwen35DynamicPruneWorkflow(Qwen35Workflow):
    """Keep the existing entrypoint; use the canonical MoE scalar/graph path."""

    expected_model_cls_names = {"XHQwen3_5MoeDynamicPruneModel"}
    expected_model_config_cls_names = {"XHQwen3_5MoeDynamicPruneModelConfig"}
    family_label = "Qwen3.5-MoE decode ACEP"


@register_llm_model("Qwen3_5MoeForConditionalGeneration_dynamic_prune")
class XHQwen3_5MoeDynamicPruneModel(XHQwen3_5MoeModel):  # noqa: N801
    HF_MODEL_CLS = Qwen3_5MoeForConditionalGeneration
    META_CLS = Qwen3_5Moe_ModelMeta
    HMONNXINFERENCE_CLS = XHQwen3_5MoeHMONNXModel
    CONFIG_CLS = XHQwen3_5MoeDynamicPruneModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_qwen3_5_moe_hf_compatible_model)
    WORKFLOW_CLS = (
        "xhmodel_merak.xh_llm.models.qwen3_5_moe_dynamic_prune.qwen3_5_moe_dynamic_prune_model:"
        "Qwen35DynamicPruneWorkflow"
    )

    def __init__(self, config: XHQwen3_5MoeDynamicPruneModelConfig):
        super().__init__(config)
        self.visual = None
        self._models.pop("visual", None)

    def get_export_info(self, output_dir):
        # Dynamic-prune delivery is LLM-only, matching the original prune
        # exporter. Bypass the Qwen3.5 VLM layout that requires a visual model.
        return super(XHQwen3_5Model, self).get_export_info(output_dir)

    def export_hmonnx(self, output_dir):
        # The Qwen3.5 text graph is a prefill/decode ModelSwitcher.  The
        # generic LLM exporter assumes one quant graph and calls fixed() on
        # the switcher itself, so retain the generic LLM-only directory/meta
        # layout while fixing each Qwen3.5 graph explicitly.
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.prefill.fixed()
        self._quanted_model.decode.fixed()

        exported_info = self.get_export_info(output_dir)
        self._export_hmonnx(exported_info)
        meta_info = exported_info.meta.to_dict()
        with open(Path(exported_info.exported_dir) / "golden_meta_info.json", "w", encoding="utf-8") as f:
            json.dump(meta_info, f, indent=4)
        get_xhquant_logger().info(f"Exporting completed! Exported model is saved at: {exported_info.exported_dir}")
        return meta_info
