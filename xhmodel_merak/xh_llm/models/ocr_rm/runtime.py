# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Single forward OCR consistency scoring; no token generation."""

import json
from pathlib import Path

import torch
from xhmodel_merak.xh_llm.infer_mixin import LLMInferenceContextManager
from xhmodel_merak.xh_llm.ocr_runtime import last_nonpad_index, prefill_logits
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import XHQwen3_5_HMONNXModel
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import Qwen3_5_ModelMeta
from .model import XHOcrRewardModel
from .processing import prepare_inputs


class OcrRewardHMONNXModel(XHQwen3_5_HMONNXModel):
    LLM_MODEL_CLS = XHOcrRewardModel


class OcrRewardRuntime:
    def __init__(self, meta_file, device="cuda:0"):
        from transformers import AutoProcessor

        path = Path(meta_file).resolve()
        meta = json.loads(path.read_text())
        meta["_meta_path_"] = str(path)
        self.device = torch.device(device)
        from xhmodel_merak._environment import scoped_environment

        with scoped_environment({"ENABLE_HMINFERENCE_V2": "1"}):
            self.model = OcrRewardHMONNXModel(Qwen3_5_ModelMeta.from_dict(meta), device_map=[self.device]).to(
                self.device
            )
        self.processor = AutoProcessor.from_pretrained(path.parent / meta["hf_config"])

    @torch.inference_mode()
    def score(self, orig_image, render_image, ocr_text):
        inputs = prepare_inputs(self.processor, orig_image, render_image, ocr_text)
        ids = inputs["input_ids"].to(self.device)
        pool_index = last_nonpad_index(ids, self.processor.tokenizer.pad_token_id)
        ids = ids[:, : pool_index + 1]
        grids = inputs["image_grid_thw"].to(self.device)
        features = self.model.visual.encode_many(inputs["pixel_values"].to(self.device), grids)
        features = torch.cat([feature.reshape(-1, feature.shape[-1]) for feature in features])
        with LLMInferenceContextManager(self.model):
            positions, _ = self.model.get_data_preprocessor().get_rope_index(ids, image_grid_thw=grids)
            logits = prefill_logits(self.model, ids, image_embeds=features, position_ids=positions)
            logits = logits[0, -1].float()
        if logits.numel() != 2 or not torch.isfinite(logits).all():
            raise RuntimeError("OCR RM graph must return two finite class logits")
        probs = logits.softmax(-1)
        return {
            "prob_0": float(probs[0]),
            "prob_1": float(probs[1]),
            "logit_0": float(logits[0]),
            "logit_1": float(logits[1]),
            "n_tokens": ids.shape[1],
        }

    def close(self):
        self.model = None
