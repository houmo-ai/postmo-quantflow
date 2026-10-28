# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-OCR host preprocessing and HMONNX execution."""

import json
from pathlib import Path

import torch

from xhmodel_merak.xh_llm.infer_mixin import LLMInferenceContextManager
from xhmodel_merak.xh_llm.ocr_runtime import match_position_ranks, prefill_logits
from xhmodel_merak.xh_llm.minicpm_v_series.token_capacity_vision import (
    crop_token_capacity_image_embeds,
    prepare_token_capacity_vision_inputs,
    split_packed_pixel_values,
)
from xhmodel_merak.xh_llm.models.minicpm_v_4_6.inference import MiniCPMV46TextHMONNXModel
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import Qwen3_5_ModelMeta
from .model import XHMiniCPMOcrModel


class MiniCPMOcrTextRuntime(MiniCPMV46TextHMONNXModel):
    LLM_MODEL_CLS = XHMiniCPMOcrModel


class MiniCPMOcrRuntime:
    def __init__(self, meta_file, device="cuda:0"):
        from xhquant.api import HMONNXInference
        from .native import load_processor, register_native

        register_native()

        self.root = Path(meta_file).resolve().parent
        self.meta = json.loads(Path(meta_file).read_text())
        if self.meta.get("model_type") != "minicpm_ocr" or self.meta.get("schema_version") != 1:
            raise ValueError("Unsupported MiniCPM-OCR runtime metadata")
        self.device = torch.device(device)
        child_file = self.root / self.meta["text_metadata"]
        child = json.loads(child_file.read_text())
        child["_meta_path_"] = str(child_file)
        from xhmodel_merak._environment import scoped_environment

        with scoped_environment({"ENABLE_HMINFERENCE_V2": "1"}):
            self.text = MiniCPMOcrTextRuntime(Qwen3_5_ModelMeta.from_dict(child), device_map=[self.device]).to(
                self.device
            )
        self.processor = load_processor(self.root / self.meta["hf_config"])
        self.visual = HMONNXInference(str(self.root / self.meta["visual"]["hmonnx"]))
        self.visual.exec_device = self.device
        self.visual.to(self.device)

    @torch.inference_mode()
    def prepare(self, image, prompt):
        from PIL import Image

        with Image.open(image) as source:
            pixels = source.convert("RGB")
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[text], images=[pixels], return_tensors="pt", downsample_mode="4x", max_slice_nums=9
        )
        features = []
        profile = self.meta["visual"]
        for pixel, target in zip(
            split_packed_pixel_values(inputs["pixel_values"], inputs["target_sizes"]),
            inputs["target_sizes"],
            strict=True,
        ):
            size = tuple(int(x) for x in target)
            args = prepare_token_capacity_vision_inputs(
                pixel.to(self.device, torch.float16),
                size,
                profile["token_capacity"],
                "4x",
                profile["positions_per_side"],
            )
            output = self.visual(*args[:3])
            if isinstance(output, (list, tuple)):
                output = output[0]
            features.append(crop_token_capacity_image_embeds(output, size, profile["token_capacity"], "4x").squeeze(0))
        return inputs, torch.cat(features)

    @torch.inference_mode()
    def generate(self, image, prompt, max_new_tokens=512):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        inputs, features = self.prepare(image, prompt)
        ids = inputs["input_ids"].to(self.device)
        length = ids.shape[1]
        if length + max_new_tokens > self.text.meta_info.model_config.context_max_length:
            raise ValueError("Prompt plus generated tokens exceeds exported context")
        generated = []
        eos = self.text.generation_config.eos_token_id if hasattr(self.text, "generation_config") else None
        eos_ids = {248044, 248046}
        if isinstance(eos, int):
            eos_ids.add(eos)
        with LLMInferenceContextManager(self.text):
            logits = prefill_logits(self.text, ids, image_embeds=features)
            token = int(logits[0, -1].argmax())
            generated.append(token)
            self.text.set_decode()
            while len(generated) < max_new_tokens and token not in eos_ids:
                args = self.text.get_data_preprocessor()(
                    {
                        "input_ids": torch.tensor([[token]], device=self.device),
                        "past_seq_length": length + len(generated) - 1,
                    }
                )
                args = match_position_ranks(args, self.text.decode_model.hmonnx_session)
                logits = self.text.forward(*args)[0]
                token = int(logits[0, -1].argmax())
                generated.append(token)
        return {
            "output_text": self.processor.tokenizer.decode(generated, skip_special_tokens=True),
            "output_tokens": len(generated),
            "token_ids": generated,
            "prompt_tokens": length,
        }

    def close(self):
        self.visual = None
        self.text = None
