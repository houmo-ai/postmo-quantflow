# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from ...workflows.base import BaseLLMWorkflow
from ...workflows.result import QuantResult


class XHPaddleOCRVLWorkflow(BaseLLMWorkflow):
    expected_model_config_cls_name = "XHPaddleOCRVLModelConfig"
    expected_model_cls_name = "XHPaddleOCRVLModel"

    def quant(self, output_dir, device, config_overrides=None):
        config = self.workflow_config.with_overrides(config_overrides)
        if config.quant is None:
            return QuantResult(raw_model_dir=self.model_dir, skipped=True)
        from recipes.paddleocr_vl import quantize_paddleocr_vl

        options = dict(config.quant)
        algorithm = options.pop("algorithm", "gptqmodel")
        if algorithm != "gptqmodel":
            raise ValueError("PaddleOCR workflow quant.algorithm must be gptqmodel")
        if "calibration_jsonl" in options:
            from .calibration import load_calibration_inputs

            if not options["calibration_jsonl"]:
                raise ValueError("Set quant.calibration_jsonl to image/prompt/answer calibration records")
            visual = config.export["model"].get("visual_config", {})
            calibration_data = load_calibration_inputs(
                self.model_dir, options.pop("calibration_jsonl"), nsamples=options.pop("nsamples", None),
                width=visual.get("max_size_w", 896), height=visual.get("max_size_h", 896),
            )
        else:
            calibration_data = options.pop("calibration_data")
        result = quantize_paddleocr_vl(
            model_dir=self.model_dir, output_dir=output_dir, calibration_data=calibration_data,
            device=device, seed=self.seed, **options,
        )
        return QuantResult(raw_model_dir=self.model_dir, quanted_model_dir=result.output_dir, skipped=False)

    def dump_golden(self, export_result, device, input_messages):
        from ... import AutoLLMHONNXModel
        from .processing import prepare_inputs

        if not isinstance(input_messages, dict) or "image" not in input_messages:
            raise ValueError("PaddleOCR golden input requires image and optional text/resize_mode")
        meta_file = self._find_golden_meta_file(export_result)
        model = AutoLLMHONNXModel.from_pretrained(meta_file, enable_golden=True, device_map=[device]).to(device)
        visual = model.meta_info.model_config.visual_config
        processor = model.get_tf_processor()
        inputs = prepare_inputs(
            processor, input_messages["image"], input_messages.get("text", "OCR:"),
            resize_mode=input_messages.get("resize_mode", "stretch"), width=visual.max_size_w,
            height=visual.max_size_h,
        ).to(device)
        generated = model.generate(**inputs, max_new_tokens=2, do_sample=False)
        text = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        (Path(meta_file).parent / "golden_output.txt").write_text("\n".join(text) + "\n")
        return meta_file
