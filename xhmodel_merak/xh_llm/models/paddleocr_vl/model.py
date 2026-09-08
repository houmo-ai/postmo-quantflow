# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path

from transformers import AutoModelForImageTextToText, PaddleOCRVLForConditionalGeneration

from ...builder import register_llm_model
from ...llm_data_processor import BaseInputProcessorConfig
from ...types import LLMModelState
from ...vision_llm_model import VisionLLMModel
from .config import XHPaddleOCRVLModelConfig
from .data import PaddleOCRLLMInputProcessor
from .generation import PaddleOCRGenerationMixin
from .hf import load_paddleocr_config, load_paddleocr_model, load_paddleocr_processor
from .runtime import XHPaddleOCRVLHMONNXModel


@register_llm_model("PaddleOCRVLForConditionalGeneration")
class XHPaddleOCRVLModel(PaddleOCRGenerationMixin, VisionLLMModel):
    transformers_min_version = "5.16.0"
    HF_MODEL_CLS = PaddleOCRVLForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    CONFIG_CLS = XHPaddleOCRVLModelConfig
    HMONNXINFERENCE_CLS = XHPaddleOCRVLHMONNXModel
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.paddleocr_vl.workflow:XHPaddleOCRVLWorkflow"

    def __init__(self, config):
        super().__init__(config)
        from .visual import XHPaddleOCRVLVisualModel

        self.paddleocr_hf_config = load_paddleocr_config(self.hf_model_dir)
        self.visual = XHPaddleOCRVLVisualModel(config.visual_config)

    @VisionLLMModel.work_dir.setter
    def work_dir(self, value):
        self.config.work_dir = value
        self.visual.work_dir = str(Path(value) / "visual")

    @classmethod
    def _load_hf_model(cls, hf_model_dir, **kwargs):
        return load_paddleocr_model(hf_model_dir, **kwargs)

    @classmethod
    def get_hf_model(cls, hf_model_dir, quant_weight=None, **kwargs):
        config = load_paddleocr_config(hf_model_dir)
        quantization = getattr(config, "quantization_config", None)
        mode = kwargs.pop("gptq_weight_mode", "dequantize")
        if quantization is None:
            model = cls._load_hf_model(hf_model_dir, **kwargs)
            if quant_weight:
                cls._load_quant_weight(quant_weight, model)
            return model
        if quant_weight:
            raise ValueError("An already quantized checkpoint cannot accept additional quant_weight")
        if cls._get_quantization_method(quantization) != "gptq":
            raise ValueError("PaddleOCR Merak accepts native weights or a GPTQ-format recipe checkpoint")
        model = cls._load_gptqmodel(hf_model_dir, **kwargs)
        if mode == "dequantize":
            model = cls._dequantize_gptqmodel_hf_model(model)
        elif mode == "packed":
            model = cls._retain_gptqmodel_packed_hf_model(model)
        else:
            raise ValueError(f"Unsupported gptq_weight_mode: {mode}")
        return cls._postprocess_gptqmodel_structure(model, **kwargs)

    @classmethod
    def _load_gptqmodel(cls, hf_model_dir, **kwargs):
        from recipes.paddleocr_vl import (
            preserve_paddleocr_vl_tie_word_embeddings,
            register_paddleocr_vl_gptqmodel,
        )

        config = load_paddleocr_config(hf_model_dir)
        quantization = config.quantization_config
        contract = quantization.get("meta", {}).get("rotation_contract", {})
        if contract.get("runtime_wrapper_required"):
            raise ValueError("PaddleOCR Merak does not support online rotation wrappers")
        if (quantization.get("rotation") or contract.get("enabled")) and (
            contract.get("scope") != "offline_r1_hidden_basis"
        ):
            raise ValueError("Rotated PaddleOCR checkpoints require the offline R1 artifact contract")
        register_paddleocr_vl_gptqmodel()
        kwargs.setdefault("trust_remote_code", False)
        with preserve_paddleocr_vl_tie_word_embeddings(config.tie_word_embeddings):
            return super()._load_gptqmodel(hf_model_dir, **kwargs)

    def init_wrap_model(self, hf_model):
        from ._llm_model_impl import register_wrap_cls

        register_wrap_cls(hf_model)
        return super().init_wrap_model(hf_model)

    def _get_language_model(self, hf_model):
        return hf_model.model.language_model

    def get_tf_processor(self):
        return load_paddleocr_processor(self.hf_model_dir)

    def _get_data_preprocessor(self):
        return PaddleOCRLLMInputProcessor(BaseInputProcessorConfig(
            self.get_input_embeddings(), self.get_input_sequence_length(), self.past_key_caches,
            self.past_value_caches, pad_token_id=self.pad_token_id,
        ))

    def get_export_cfg(self):
        config = super().get_export_cfg()
        config["input_names"][1:1] = ["time_position_ids", "height_position_ids", "width_position_ids"]
        return config

    def _extra_export_metadata(self, output_dir, meta):
        hf_dir = Path(output_dir) / meta.hf_config
        config = self.paddleocr_hf_config.to_dict()
        def remove_remote_mapping(value):
            if isinstance(value, dict):
                value.pop("auto_map", None)
                for child in value.values():
                    remove_remote_mapping(child)
        remove_remote_mapping(config)
        (hf_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        for name in ("tokenizer.model",):
            source = Path(self.hf_model_dir) / name
            if source.is_file():
                shutil.copyfile(source, hf_dir / name)
        return meta

    def export_hmonnx(self, output_dir):
        self.work_dir = str(output_dir)
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.fixed()
        self.visual.quanted_model.fixed()
        info = self.get_export_info(output_dir)
        visual_meta = self.visual.export_hmonnx(str(Path(info.exported_dir) / "visual"))
        visual_meta.hmonnx = Path(visual_meta.hmonnx).relative_to(info.exported_dir).as_posix()
        info.meta.visual_config = visual_meta
        self._export_hmonnx(info)
        (Path(info.exported_dir) / "golden_meta_info.json").write_text(
            json.dumps(info.meta.to_dict(), indent=2) + "\n"
        )
        return info.meta
