"""Qwen4-Exp visual tower using the shared Qwen static patch-token ABI.

Qwen4-Exp and Qwen3.5 have identical visual patch embedding, learned-position
interpolation, two-dimensional RoPE and merger semantics. Register the native
Qwen4 classes against those existing graph rewrites instead of maintaining a
second implementation of their head padding and quantizable attention.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoProcessor
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpVisionAttention,
    Qwen4ExpVisionBlock,
    Qwen4ExpVisionModel,
    Qwen4ExpVisionPatchEmbed,
    Qwen4ExpVisionPatchMerger,
)

from ...builder import register_llm_model
from ...register import XHLLM_TRACEABLE_MODULES
from ..qwen3_5._vision_model_impl import (
    _Qwen3_5VisionAttention,
    _Qwen3_5VisionBlock,
    _Qwen3_5VisionModel,
    _Qwen3_5VisionPatchEmbed,
    _Qwen3_5VisionPatchMerger,
)
from ..qwen3_5.qwen3_5_hmonnx_inference import VisualTokenGearHMONNXModel
from ..qwen3_5.qwen3_5_vision_model import XHQwen3_5VisionModel
from ..qwen3_5.xh_qwen3_5_config import XHQwen3_5_VisualConfig


XHLLM_TRACEABLE_MODULES.register_module({Qwen4ExpVisionAttention: "Qwen4ExpVisionAttention"}, _Qwen3_5VisionAttention)
XHLLM_TRACEABLE_MODULES.register_module({Qwen4ExpVisionBlock: "Qwen4ExpVisionBlock"}, _Qwen3_5VisionBlock)
XHLLM_TRACEABLE_MODULES.register_module({Qwen4ExpVisionModel: "Qwen4ExpVisionModel"}, _Qwen3_5VisionModel)
XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen4ExpVisionPatchEmbed: "Qwen4ExpVisionPatchEmbed"}, _Qwen3_5VisionPatchEmbed
)
XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen4ExpVisionPatchMerger: "Qwen4ExpVisionPatchMerger"}, _Qwen3_5VisionPatchMerger
)


@register_llm_model("Qwen4ExpForConditionalGeneration_visual", master=False)
class XHQwen4ExpVisionModel(XHQwen3_5VisionModel):
    """Materialize only native ``model.visual.*`` weights, then export gears."""

    transformers_min_version = "5.16.0"
    HF_MODEL_CLS = Qwen4ExpForConditionalGeneration
    VISUAL_HF_MODEL_CLS = Qwen4ExpVisionModel

    def __init__(self, config):
        config.model_type = "Qwen4ExpForConditionalGeneration_visual"
        super().__init__(config)

    def get_tf_processor(self):
        # The published Flash Next checkpoint explicitly selects Qwen3VLProcessor.
        # Loading it from its own resources preserves its tokenizer/template and
        # image normalization rather than substituting Qwen3.5 defaults.
        return AutoProcessor.from_pretrained(self.hf_model_dir)


def export_qwen4_exp_visual(
    model_dir,
    output_dir,
    *,
    image_token_gears,
    quant_scheme,
    device="cpu",
    dtype="float16",
):
    """Export ``visual_m*`` components and relocatable standalone metadata."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_dir).vision_config
    if getattr(config, "deepstack_visual_indexes", ()):
        raise ValueError("Qwen4-Exp Flash Next visual export does not support deepstack features")
    model = XHQwen4ExpVisionModel(
        XHQwen3_5_VisualConfig(
            model_name="qwen3_8_flash_next_visual",
            model_type="Qwen4ExpForConditionalGeneration_visual",
            hf_model=str(model_dir),
            chip_arch="XH2a",
            quant_scheme=quant_scheme,
            dtype=dtype,
            visual_input_mode="patches",
            image_token_gears=list(image_token_gears),
            image_token_capacity=max(image_token_gears),
            spatial_merge_size=config.spatial_merge_size,
            temporal_patch_size=config.temporal_patch_size,
            patch_size=config.patch_size,
        )
    )
    model.to(device=device)
    # Shared PTQ resolves its ``cuda`` device from the current CUDA context.
    # Scope it to the requested device without changing the caller's selection.
    device = torch.device(device)
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        model.export_hmonnx(str(output_dir))
    import json

    return json.loads((Path(output_dir) / "visual_meta_info.json").read_text())


__all__ = ["XHQwen4ExpVisionModel", "VisualTokenGearHMONNXModel", "export_qwen4_exp_visual"]
