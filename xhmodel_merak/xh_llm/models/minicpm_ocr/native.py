# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
# Copyright 2026 HOUMO AI (modifications)
# SPDX-License-Identifier: Apache-2.0
# Source: supplied Transformers 5.19.0.dev0 minicpm_ocr image processor.
# See licenses/transformers/LICENSE and THIRD_PARTY_NOTICES.
"""Load the delivered 4x checkpoint through the existing Transformers stack.

Reuse MiniCPM-V-4.6's 4x vision/text mathematics and preserve OCR-specific
image geometry below. OCR's newer 16x merger and optional adaptive resizing
are outside this integration's supported protocol.
"""

import json
import math
from pathlib import Path

from transformers.models.minicpmv4_6.configuration_minicpmv4_6 import MiniCPMV4_6Config
from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import MiniCPMV4_6ForConditionalGeneration
from transformers.models.minicpmv4_6.image_processing_minicpmv4_6 import MiniCPMV4_6ImageProcessor


class MiniCPMOcrConfig(MiniCPMV4_6Config):
    model_type = "minicpm_ocr"


class MiniCPMOcrNativeModel(MiniCPMV4_6ForConditionalGeneration):
    config_class = MiniCPMOcrConfig


class MiniCPMOcrImageProcessor(MiniCPMV4_6ImageProcessor):
    """Use OCR's 2x patch grid and its slice-grid tie breaking in 4x mode."""

    def find_best_resize(self, image_size, scale_resolution, patch_size, allow_upscale=False):
        height, width = image_size
        if height * width > scale_resolution * scale_resolution or allow_upscale:
            aspect_ratio = width / height
            height = int(scale_resolution / math.sqrt(aspect_ratio))
            width = int(height * aspect_ratio)
        factor = 2 if self.downsample_mode == "4x" else 4
        quantum = patch_size * factor
        return max(round(height / quantum) * quantum, quantum), max(round(width / quantum) * quantum, quantum)

    def get_sliced_grid(self, image_size, max_slice_nums, scale_resolution):
        height, width = image_size
        log_ratio = math.log(width / height)
        multiple = min(math.ceil(width * height / (scale_resolution * scale_resolution)), max_slice_nums)
        if multiple <= 1:
            return None
        best_grid = [1, 1]
        min_error = float("inf")
        for count in (multiple - 1, multiple, multiple + 1):
            if count == 1 or count > max_slice_nums:
                continue
            for rows in range(1, count + 1):
                if count % rows == 0:
                    columns = count // rows
                    error = abs(log_ratio - math.log(columns / rows))
                    if error < min_error or (error == min_error and rows > best_grid[0]):
                        best_grid, min_error = [rows, columns], error
        return best_grid


def register_native():
    from transformers import AutoConfig, AutoModelForImageTextToText

    AutoConfig.register("minicpm_ocr", MiniCPMOcrConfig, exist_ok=True)
    AutoModelForImageTextToText.register(MiniCPMOcrConfig, MiniCPMOcrNativeModel, exist_ok=True)


def load_processor(model_dir):
    from transformers import AutoTokenizer
    from transformers.models.minicpmv4_6.processing_minicpmv4_6 import MiniCPMV4_6Processor
    from transformers.models.minicpmv4_6.video_processing_minicpmv4_6 import MiniCPMV4_6VideoProcessor

    root = Path(model_dir)
    image_config = json.loads((root / "preprocessor_config.json").read_text())
    if image_config.get("disable_small_image_upsize") or image_config.get("adaptive_small_image_upsize"):
        raise ValueError("This OCR deployment supports the delivered default image resizing protocol")
    video_config = json.loads((root / "video_preprocessor_config.json").read_text())
    for config in (image_config, video_config):
        for key in ("processor_class", "image_processor_type", "video_processor_type"):
            config.pop(key, None)
        if config.get("downsample_mode", "4x") != "4x":
            raise ValueError("MiniCPM-OCR deployment currently supports only the 4x protocol")
    return MiniCPMV4_6Processor(
        image_processor=MiniCPMOcrImageProcessor(**image_config),
        video_processor=MiniCPMV4_6VideoProcessor(**video_config),
        tokenizer=AutoTokenizer.from_pretrained(root),
        chat_template=(root / "chat_template.jinja").read_text(),
    )
