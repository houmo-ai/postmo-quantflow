# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""OCR consistency value head using the shared Qwen3.5 deployment graphs."""

from xhmodel_merak.xh_llm.builder import register_llm_model
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model


def classification_shell(native):
    """Expose the trained two-class projection to the existing graph wrapper.

    Embedding vocabulary stays unchanged. Only the output projection is two
    dimensional; no vocabulary softmax or token generation is used at runtime.
    """
    from accelerate import init_empty_weights
    from transformers import GenerationConfig, Qwen3_5ForConditionalGeneration

    if native.score.weight.shape != (2, native.config.text_config.hidden_size):
        raise ValueError("OCR RM requires score.weight with shape [2, hidden_size]")
    with init_empty_weights():
        shell = Qwen3_5ForConditionalGeneration(native.config)
    shell.model = native.model
    shell.lm_head = native.score
    shell.generation_config = GenerationConfig.from_model_config(native.config)
    shell.name_or_path = native.name_or_path
    shell.config.tie_word_embeddings = False
    shell.config.text_config.tie_word_embeddings = False
    return shell.eval()


@register_llm_model("OcrRewardForSequenceClassification")
class XHOcrRewardModel(XHQwen3_5Model):
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.ocr_rm.workflow:OcrRewardWorkflow"

    @classmethod
    def get_hf_model(cls, hf_model_dir, quant_weight=None, **kwargs):
        from transformers import Qwen3_5ForSequenceClassification

        if quant_weight:
            raise ValueError("OCR RM does not accept independent quantized checkpoints")
        kwargs.pop("gptq_weight_mode", None)
        if "dtype" not in kwargs and "torch_dtype" not in kwargs:
            kwargs["dtype"] = cls.get_hf_model_dtype()
        # The delivered config advertises a generative architecture. Select
        # classification explicitly, and reject missing/mismatched score weights.
        native, info = Qwen3_5ForSequenceClassification.from_pretrained(
            hf_model_dir,
            num_labels=2,
            output_loading_info=True,
            **kwargs,
        )
        if "score.weight" in info.get("missing_keys", ()) or info.get("mismatched_keys"):
            raise ValueError("Checkpoint is missing its trained two-class score head")
        from xhmodel_merak.xh_llm.models.qwen3_5.modeling_qwen3_5_patch import qwen3_5_patch

        return qwen3_5_patch(classification_shell(native))
