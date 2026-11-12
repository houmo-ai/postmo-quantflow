"""Merak workflow configuration for Qwen4-Exp."""

import copy
import json
from collections.abc import Mapping
from pathlib import Path

from ...vision_llm_model import VisionLLMModelConfig
from ..qwen3_5.xh_qwen3_5_config import XHQwen3_5_VisualConfig
from .precision import DEFAULT_PLE_QUANT_TYPE, resolve_ple_quant_type


class XHQwen4ExpModelConfig(VisionLLMModelConfig):
    """Route Qwen4-Exp configs through the model-specific workflow."""

    def __init__(self, *, ple_quant_type: str = DEFAULT_PLE_QUANT_TYPE, visual_config=None, **kwargs):
        ple_quant_type = resolve_ple_quant_type(ple_quant_type)
        declared_quant = kwargs.get("quant_scheme")
        self._export_quant_config = copy.deepcopy(
            declared_quant.to_dict() if hasattr(declared_quant, "to_dict") else declared_quant
        )
        super().__init__(**kwargs)
        if type(self.batch_size) is not int or self.batch_size != 1:
            raise ValueError("Qwen4 static export requires batch_size=1")
        if (type(self.context_max_length) is not int or type(self.prefill_chunk_length) is not int
                or self.context_max_length < 4
                or not 1 <= self.prefill_chunk_length <= self.context_max_length):
            raise ValueError(
                "Qwen4 static export requires integer lengths with 4 <= context_max_length "
                "and 1 <= prefill_chunk_length <= context_max_length"
            )
        self.ple_quant_type = ple_quant_type
        if getattr(self, "spec_decode_mode", None) == "mtp":
            from .mtp_contract import resolve_mtp_num_draft_tokens, resolve_mtp_verify_length

            options = {key: kwargs[key] for key in ("num_draft_tokens", "verify_length") if key in kwargs}
            self.num_draft_tokens = resolve_mtp_num_draft_tokens(options)
            self.verify_length = resolve_mtp_verify_length(options)
            if self.verify_length > self.context_max_length:
                raise ValueError("Qwen4 MTP verify length exceeds context capacity")
        if isinstance(visual_config, Mapping) and visual_config.get("enable") is False:
            visual_config = None
        if isinstance(visual_config, Mapping):
            visual_config = dict(visual_config)
            visual_config.setdefault("model_name", f"{self.model_name}_visual")
            visual_config.setdefault("hf_model", self.hf_model)
            # The old Qwen4 decode-validation recipe described a fixed image
            # rectangle. Translate it at this model-local boundary; the shared
            # Qwen3.5 visual config only accepts post-merge token gears.
            source_geometry = {}
            config_path = Path(visual_config["hf_model"] or "") / "config.json"
            if visual_config["hf_model"] and config_path.is_file():
                source_geometry = json.loads(config_path.read_text(encoding="utf-8")).get("vision_config", {})
            for field, default in (("patch_size", 16), ("spatial_merge_size", 2), ("temporal_patch_size", 2)):
                visual_config.setdefault(field, source_geometry.get(field, default))
            width = visual_config.pop("max_size_w", None)
            height = visual_config.pop("max_size_h", None)
            if "image_token_gears" not in visual_config and (width is not None or height is not None):
                if width is None or height is None:
                    raise ValueError("legacy Qwen4 visual config requires both max_size_w and max_size_h")
                unit = int(visual_config["patch_size"]) * int(visual_config["spatial_merge_size"])
                temporal = int(visual_config["temporal_patch_size"])
                frames = int(visual_config.get("max_size_t", temporal))
                if unit <= 0 or temporal <= 0 or min(int(width), int(height), frames) <= 0:
                    raise ValueError("legacy Qwen4 visual dimensions and patch sizes must be positive")
                if int(width) % unit or int(height) % unit or frames % temporal:
                    raise ValueError("legacy Qwen4 visual dimensions must align with spatial and temporal patches")
                visual_config["image_token_gears"] = [int(width) // unit * (int(height) // unit) * (frames // temporal)]
            visual_config.setdefault("visual_input_mode", "patches")
            if "image_token_gears" in visual_config:
                visual_config.setdefault("image_token_capacity", max(visual_config["image_token_gears"]))
            visual_config = XHQwen3_5_VisualConfig(**visual_config)
        self.visual_config = visual_config


__all__ = ["XHQwen4ExpModelConfig"]
