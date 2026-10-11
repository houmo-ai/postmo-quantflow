"""Shared Qwen3.5 text Prefill/Decode export finishing for MiniCPM-V adapters."""

from __future__ import annotations

import json
from pathlib import Path

from xhmodel_merak.xh_llm.types import LLMModelState
from xhquant.api import get_xhquant_logger


def export_minicpm_qwen35_text_hmonnx(model, output_dir: str):
    """Export Prefill/Decode graphs without Qwen's unrelated Vision graph."""

    logger = get_xhquant_logger()
    model.work_dir = str(output_dir)
    model.to_wrap()

    exported_info = model.get_export_info(output_dir)
    if model._state != LLMModelState.QUANTED_ALIGNED:
        model.to_quanted_aligned()

    # Qwen3.5 builds distinct Prefill and Decode graphs. BaseLLMModel's
    # single-graph ``fixed`` call is therefore not valid for this model.
    model._quanted_model.prefill.fixed()
    model._quanted_model.decode.fixed()
    model.config.model_name = exported_info.model_name
    model._export_hmonnx(exported_info)

    meta_info = exported_info.meta
    with open(Path(exported_info.exported_dir) / "golden_meta_info.json", "w", encoding="utf-8") as file:
        json.dump(meta_info.to_dict(), file, indent=4)
    logger.info("Exporting completed! Exported model is saved at: %s", exported_info.exported_dir)
    return meta_info


__all__ = ["export_minicpm_qwen35_text_hmonnx"]
