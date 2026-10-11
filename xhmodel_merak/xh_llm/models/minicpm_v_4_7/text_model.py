"""MiniCPM-V-4.7 text-only adapters for the Merak Qwen3.5 Dense/MoE stacks."""

from __future__ import annotations

import torch.nn as nn
from accelerate import init_empty_weights

from xhmodel_merak.xh_llm.base_llm_model import BaseLLMModel
from xhmodel_merak.xh_llm.llm_data_processor import BaseLLMInputProcessor
from xhmodel_merak.xh_llm.minicpm_v_series.text_export import (
    export_minicpm_qwen35_text_hmonnx,
)
from xhmodel_merak.xh_llm.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from xhmodel_merak.xh_llm.models.qwen3_5.data_preprocess import Qwen3_5_DataPreprocess
from xhmodel_merak.xh_llm.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model
from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_model import XHQwen3_5MoeModel
from xhquant.utils import log_function_call


# MiniCPM keeps Qwen ABI placeholders for vision_start/end even though its
# multimodal chat template uses dedicated image/video pad tokens.
_DEFAULT_VISION_START_TOKEN_ID = 248053
_DEFAULT_VISION_END_TOKEN_ID = 248054


def _untie_lm_head_if_needed(text_model) -> None:
    input_weight = text_model.model.language_model.get_input_embeddings().weight
    if text_model.lm_head.weight is input_weight:
        text_model.lm_head.weight = nn.Parameter(
            input_weight.detach().clone(),
            requires_grad=input_weight.requires_grad,
        )
    text_model.config.text_config.tie_word_embeddings = False


def _apply_minicpm_text_abi(hf_model) -> None:
    hf_model.config.vision_start_token_id = (
        getattr(hf_model.config, "vision_start_token_id", None) or _DEFAULT_VISION_START_TOKEN_ID
    )
    hf_model.config.vision_end_token_id = (
        getattr(hf_model.config, "vision_end_token_id", None) or _DEFAULT_VISION_END_TOKEN_ID
    )
    if getattr(hf_model.config, "vision_config", None) is not None:
        hf_model.config.vision_config.spatial_merge_size = 2


class MiniCPMV47TextModel(XHQwen3_5Model):
    """Export the embedded Qwen3.5 Dense backend without Qwen's Vision graph."""

    transformers_min_version = "5.7.0"

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs):
        """Load MiniCPM, then expose its text branch through a Qwen3.5 shell.

        ``quant_weight`` optionally points at a GPTQ-dequantized CausalLM
        ``pytorch_model.bin`` (``model.*`` + ``lm_head.weight``). It is applied
        onto the MiniCPM host before the Qwen3.5 shell wrap, matching the
        MiniCPM-V-4.5 pattern.
        """

        from .quant_llm import apply_causal_dequant_to_minicpm_host

        kwargs.setdefault("trust_remote_code", True)
        full_model = super().get_hf_model(
            hf_model_dir,
            quant_weight=None,
            **kwargs,
        )
        if quant_weight:
            apply_causal_dequant_to_minicpm_host(full_model, quant_weight)
        qwen_config = Qwen3_5Config(
            text_config=full_model.config.text_config.to_dict(),
            image_token_id=full_model.config.image_token_id,
            video_token_id=full_model.config.video_token_id,
            vision_start_token_id=_DEFAULT_VISION_START_TOKEN_ID,
            vision_end_token_id=_DEFAULT_VISION_END_TOKEN_ID,
            tie_word_embeddings=False,
        )
        with init_empty_weights():
            text_model = Qwen3_5ForConditionalGeneration(qwen_config)

        text_model.model.language_model = full_model.model.language_model
        text_model.lm_head = full_model.lm_head
        text_model.generation_config = full_model.generation_config
        text_model.name_or_path = str(hf_model_dir)
        _untie_lm_head_if_needed(text_model)
        del full_model
        return text_model.eval()

    def _wraped_post(self, hf_model):
        # These fields only describe the text-graph preprocessing ABI. The
        # MiniCPM runtime intentionally uses sequential LLM positions and
        # passes image_grid_thw=None, matching the proven legacy adapter.
        _apply_minicpm_text_abi(hf_model)
        return super()._wraped_post(hf_model)

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        return Qwen3_5_DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            patch_size=14,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            spatial_merge_size=2,
        )

    def get_export_info(self, output_dir):
        return BaseLLMModel.get_export_info(self, output_dir)

    @log_function_call()
    def export_hmonnx(self, output_dir: str):
        """Export Qwen3.5 Prefill/Decode without its unrelated Vision graph."""

        return export_minicpm_qwen35_text_hmonnx(self, output_dir)


class MiniCPMV47MoeTextModel(XHQwen3_5MoeModel):
    """Export the embedded Qwen3.5 MoE backend without Qwen's Vision graph."""

    transformers_min_version = "5.7.0"

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs):
        """Load MiniCPM MoE, then expose its text branch through a MoE shell.

        ``XHQwen3_5MoeModel.get_hf_model`` is inherited from Dense Qwen3.5 and
        applies ``qwen3_5_patch``. MiniCPM MoE modules are not Dense Qwen
        classes, so skip that path and load through ``XHBaseModel`` instead.

        ``quant_weight`` optionally points at a GPTQ/AutoRound-dequantized
        CausalLM ``pytorch_model.bin`` and is applied onto the MiniCPM host
        before the MoE shell wrap.
        """

        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeForConditionalGeneration,
        )

        from .quant_llm import apply_causal_dequant_to_minicpm_host

        kwargs.setdefault("trust_remote_code", True)
        full_model = super(XHQwen3_5Model, cls).get_hf_model(
            hf_model_dir,
            quant_weight=None,
            **kwargs,
        )
        if quant_weight:
            apply_causal_dequant_to_minicpm_host(full_model, quant_weight)
        qwen_config = Qwen3_5MoeConfig(
            text_config=full_model.config.text_config.to_dict(),
            image_token_id=full_model.config.image_token_id,
            video_token_id=full_model.config.video_token_id,
            vision_start_token_id=_DEFAULT_VISION_START_TOKEN_ID,
            vision_end_token_id=_DEFAULT_VISION_END_TOKEN_ID,
            tie_word_embeddings=False,
        )
        with init_empty_weights():
            text_model = Qwen3_5MoeForConditionalGeneration(qwen_config)

        text_model.model.language_model = full_model.model.language_model
        text_model.lm_head = full_model.lm_head
        text_model.generation_config = full_model.generation_config
        text_model.name_or_path = str(hf_model_dir)
        _untie_lm_head_if_needed(text_model)
        del full_model
        return text_model.eval()

    def _wraped_post(self, hf_model):
        _apply_minicpm_text_abi(hf_model)
        return super()._wraped_post(hf_model)

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        return Qwen3_5_DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            patch_size=14,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            spatial_merge_size=2,
        )

    def get_export_info(self, output_dir):
        return BaseLLMModel.get_export_info(self, output_dir)

    @log_function_call()
    def export_hmonnx(self, output_dir: str):
        """Export Qwen3.5 MoE Prefill/Decode without its unrelated Vision graph."""

        return export_minicpm_qwen35_text_hmonnx(self, output_dir)


__all__ = ["MiniCPMV47TextModel", "MiniCPMV47MoeTextModel"]
